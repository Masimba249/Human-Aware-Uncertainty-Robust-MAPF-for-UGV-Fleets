# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Action Dependency Graph (ADG) execution of MAPF plans.

Following Hönig et al., "Persistent and Robust Execution of MAPF Schedules in
Warehouses" (RA-L 2019), a timed joint plan is compressed into, per robot, the ordered
list of its *move* actions (waits are dropped) plus type-2 dependencies between robots:

    action b of robot j (entering cell c at plan step t_b) depends on every action a of
    another robot i that leaves c at plan step t_a <= t_b.

A robot may start an action only after all of its dependencies are completed.
Executing an ADG built from a conflict-free plan is collision-free for *any* timing of
the robots, and if the ADG is acyclic every robot reaches its goal as long as every
enabled action completes in finite time (bounded delays). Plans produced with
k-robustness k >= 1 always give acyclic ADGs. Proofs: docs/THEORY.md.

In addition to execution bookkeeping this module computes the *earliest-start (ES)
schedule* of the remaining actions of a subset of robots, which the coordinator uses
as reservations when it locally re-plans the other robots.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]
ActionId = Tuple[str, int]


@dataclass(frozen=True)
class Action:
    robot: str
    index: int
    src: Cell
    dst: Cell
    plan_time: int      # plan step at which the move starts; the robot is in dst at plan_time + 1


class CyclicADGError(RuntimeError):
    pass


class ActionDependencyGraph:
    def __init__(self, paths: Dict[str, Sequence[Cell]]):
        """paths: timed joint plan (cell at every plan step, starting now) per robot."""
        self.start: Dict[str, Cell] = {r: tuple(p[0]) for r, p in paths.items()}
        self.actions: Dict[str, List[Action]] = {}
        for r, p in paths.items():
            acts = []
            for t in range(len(p) - 1):
                a, b = tuple(p[t]), tuple(p[t + 1])
                if a != b:
                    acts.append(Action(r, len(acts), a, b, t))
            self.actions[r] = acts
        self.deps: Dict[ActionId, List[ActionId]] = defaultdict(list)
        self.dependents: Dict[ActionId, List[ActionId]] = defaultdict(list)
        leaving: Dict[Cell, List[Action]] = defaultdict(list)
        for acts in self.actions.values():
            for a in acts:
                leaving[a.src].append(a)
        for acts in self.actions.values():
            for b in acts:
                for a in leaving.get(b.dst, ()):
                    if a.robot != b.robot and a.plan_time <= b.plan_time:
                        self.deps[(b.robot, b.index)].append((a.robot, a.index))
                        self.dependents[(a.robot, a.index)].append((b.robot, b.index))
        self.done: Dict[str, int] = {r: 0 for r in self.actions}
        self.topological_order()   # raises CyclicADGError if the plan is not executable

    # ------------------------------------------------------------------ structure
    def all_action_ids(self) -> Iterable[ActionId]:
        for r, acts in self.actions.items():
            for a in acts:
                yield (r, a.index)

    def predecessors(self, aid: ActionId) -> List[ActionId]:
        r, i = aid
        preds = list(self.deps.get(aid, ()))
        if i > 0:
            preds.append((r, i - 1))
        return preds

    def topological_order(self) -> List[ActionId]:
        indeg = {aid: len(self.predecessors(aid)) for aid in self.all_action_ids()}
        succ: Dict[ActionId, List[ActionId]] = defaultdict(list)
        for aid in indeg:
            for p in self.predecessors(aid):
                succ[p].append(aid)
        q = deque(a for a, d in indeg.items() if d == 0)
        order = []
        while q:
            a = q.popleft()
            order.append(a)
            for s in succ[a]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        if len(order) != len(indeg):
            raise CyclicADGError('the action dependency graph contains a cycle (plan would deadlock)')
        return order

    def num_edges(self) -> int:
        return sum(len(v) for v in self.deps.values())

    # ------------------------------------------------------------------ execution state
    def is_done(self, aid: ActionId) -> bool:
        return aid[1] < self.done[aid[0]]

    def mark_done(self, robot: str, index: int):
        """Marks actions 0..index of the robot as completed."""
        self.done[robot] = max(self.done[robot], min(index + 1, len(self.actions[robot])))

    def current_cell(self, robot: str) -> Cell:
        d = self.done[robot]
        return self.actions[robot][d - 1].dst if d > 0 else self.start[robot]

    def finished(self, robot: str) -> bool:
        return self.done[robot] >= len(self.actions[robot])

    def all_finished(self) -> bool:
        return all(self.finished(r) for r in self.actions)

    def pending(self, robot: str) -> List[Action]:
        return self.actions[robot][self.done[robot]:]

    def enabled(self, aid: ActionId) -> bool:
        return all(self.is_done(d) for d in self.deps.get(aid, ()))

    def dispatchable(self, robot: str) -> List[Action]:
        """Longest prefix of pending actions whose inter-robot dependencies are all done.
        (The robot executes them in order, so intra-robot order needs no check.)"""
        out = []
        for a in self.pending(robot):
            if not self.enabled((robot, a.index)):
                break
            out.append(a)
        return out

    def waiting_on(self, robot: str) -> Set[str]:
        """Robots whose unfinished actions block the next action of ``robot``."""
        p = self.pending(robot)
        if not p:
            return set()
        return {d[0] for d in self.deps.get((robot, p[0].index), ()) if not self.is_done(d)}

    def coupled_robots(self, robot: str) -> Set[str]:
        """Robots sharing a dependency (either direction) with remaining actions of ``robot``."""
        out = set()
        for a in self.pending(robot):
            aid = (robot, a.index)
            for d in self.deps.get(aid, ()):
                if not self.is_done(d):
                    out.add(d[0])
            for d in self.dependents.get(aid, ()):
                if not self.is_done(d):
                    out.add(d[0])
        out.discard(robot)
        return out

    def remaining_cells(self, robot: str) -> List[Cell]:
        return [self.current_cell(robot)] + [a.dst for a in self.pending(robot)]

    # ------------------------------------------------------------------ scheduling
    def earliest_start_schedule(self, robots: Iterable[str],
                                min_start: Optional[Dict[ActionId, int]] = None) -> Dict[str, List[Cell]]:
        """Timed paths (from t = 0 = now) of the remaining actions of ``robots`` when every
        action starts as early as its dependencies *among these robots* allow and takes
        one step. Dependencies on completed actions are satisfied; dependencies on actions
        of robots outside the set are dropped (those robots are being re-planned).

        The result is conflict free and 1-robust among the given robots (docs/THEORY.md).
        """
        robots = [r for r in robots]
        scope = set(robots)
        min_start = min_start or {}
        es: Dict[ActionId, int] = {}
        for aid in self.topological_order():
            r, i = aid
            if r not in scope or self.is_done(aid):
                continue
            t = min_start.get(aid, 0)
            if i > self.done[r]:
                t = max(t, es[(r, i - 1)] + 1)
            for d in self.deps.get(aid, ()):
                if d[0] in scope and not self.is_done(d):
                    t = max(t, es[d] + 1)
            es[aid] = t
        out = {}
        for r in robots:
            path = [self.current_cell(r)]
            for a in self.pending(r):
                s = es[(r, a.index)]
                while len(path) - 1 < s:
                    path.append(path[-1])
                path.append(a.dst)
            out[r] = path
        return out
