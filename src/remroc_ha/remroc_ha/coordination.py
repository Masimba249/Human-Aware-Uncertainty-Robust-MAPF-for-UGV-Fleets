# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Human-aware coordinator: risk-aware CBS planning + ADG execution + local repair.

The class is middleware-agnostic. Both the ROS 2 node
(``remroc/coordinator_ha_cbs.py``) and the grid-level simulator call ``step`` with the
current robot and human observations and forward the returned commands (a list of
grid cells per robot, to be driven through in order; an empty list means "hold").

Execution monitoring and repair
-------------------------------
* The joint plan is executed through an Action Dependency Graph: a robot receives
  only the prefix of its actions whose inter-robot dependencies are completed, so
  delays (acceleration, humans blocking, controller hiccups) can never cause
  robot-robot collisions or deadlocks.
* A robot that is *blocked* (an action is dispatched, it is at a vertex and has not
  progressed for ``blocked_timeout`` seconds -- typically a human stands in the way)
  or whose next cells are predicted to be occupied by humans (``risk_trigger``) is
  re-planned *locally*: the other robots keep their ADG-consistent earliest-start
  schedule as reservations and only the robot (then, if that fails, the robots
  coupled to it in the ADG, then all idle robots) is re-planned with the current risk
  map. The ADG is rebuilt from the new conflict-free joint plan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .adg import ActionDependencyGraph, CyclicADGError
from .grid import GridSpec
from .prediction import HumanRiskPredictor
from .risk_cbs import validate_joint_plan
from .solver import SolverBackend

Cell = Tuple[int, int]


@dataclass
class CoordinatorParams:
    risk_weight: float = 4.0              # lambda
    robustness: int = 1                   # k (1 keeps every ADG acyclic)
    use_prediction: bool = True
    solve_time_limit: float = 2.0
    suboptimality: float = 1.1            # w: plans cost at most w * optimal (focal search)
    step_duration: float = 3.0            # initial seconds per MAPF step (adapted online)
    adapt_step_duration: bool = True
    reached_tol_frac: float = 0.4         # action done when within this fraction of a cell of its target
    guard_radius_frac: float = 0.7        # don't dispatch into a cell another robot is physically in
    enable_repair: bool = True
    blocked_timeout: float = 4.0
    stuck_speed: float = 0.05
    risk_trigger: float = 0.5
    risk_lookahead: int = 3
    repair_cooldown: float = 6.0
    repair_slack: int = 1
    min_improvement: float = 1.0          # risk-triggered repairs must lower the objective by this many steps
    risk_update_period: float = 1.0       # seconds between risk map updates
    max_repair_levels: int = 3            # 1: robot only, 2: + coupled robots, 3: + all idle robots
    max_repairs: int = 200                # finite repair budget (keeps the completeness guarantee, THEORY.md)


@dataclass
class RobotObs:
    x: float
    y: float
    speed: float = 0.0


@dataclass
class CoordinatorStats:
    solver_calls: int = 0
    solver_time: float = 0.0
    initial_plans: int = 0
    local_repairs: int = 0
    coupled_repairs: int = 0
    global_replans: int = 0
    repair_failures: int = 0
    rejected_repairs: int = 0
    kept_plans: int = 0                   # risk-triggered repairs that found no clearly better plan
    blocked_triggers: int = 0
    risk_triggers: int = 0

    @property
    def replans(self) -> int:
        return self.local_repairs + self.coupled_repairs + self.global_replans


class HumanAwareCoordinator:
    def __init__(self, grid: GridSpec, goals: Dict[str, Cell], solver: SolverBackend,
                 params: Optional[CoordinatorParams] = None, predictor: Optional[HumanRiskPredictor] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.grid = grid
        self.goals = dict(goals)
        self.robots = list(goals)
        self.solver = solver
        self.p = params or CoordinatorParams()
        self.predictor = predictor if self.p.use_prediction else None
        self.log = log or (lambda s: None)
        self.adg: Optional[ActionDependencyGraph] = None
        self.stats = CoordinatorStats()
        self.events: List[dict] = []
        self.sent: Dict[str, List[Cell]] = {r: [] for r in self.robots}
        self.last_progress: Dict[str, float] = {}
        self.last_repair: Dict[str, float] = {r: -1e9 for r in self.robots}
        self.action_start: Dict[str, Optional[float]] = {r: None for r in self.robots}
        self.step_duration = self.p.step_duration
        self._risk: Optional[np.ndarray] = None
        self._risk_time = -1e9

    # ------------------------------------------------------------------ helpers
    @property
    def tol(self) -> float:
        return self.p.reached_tol_frac * self.grid.cell_size

    def _event(self, now: float, kind: str, **kw):
        e = {'time': now, 'type': kind}
        e.update(kw)
        self.events.append(e)
        self.log(f'[{now:7.1f}] {kind} {kw}')

    def _risk_map(self, now: float) -> Optional[np.ndarray]:
        if self.predictor is None:
            return None
        if now - self._risk_time >= self.p.risk_update_period - 1e-6:
            self.predictor.step_duration = self.step_duration
            self._risk = self.predictor.risk_map(now)
            self._risk_time = now
        return self._risk

    def _solve(self, agents, reservations=(), use_risk: bool = True):
        risk = self._risk_map(self._now) if use_risk else None
        lam = self.p.risk_weight if risk is not None else 0.0
        res = self.solver.solve(self.grid, agents, risk, lam, reservations, self.p.robustness,
                                self.p.solve_time_limit, self.p.suboptimality)
        self.stats.solver_calls += 1
        self.stats.solver_time += res.runtime_s
        return res

    def plan_cost(self, path: Sequence[Cell]) -> float:
        """Risk-weighted cost (in steps) of a timed path under the current risk map."""
        risk = self._risk_map(self._now)
        lam = self.p.risk_weight if risk is not None else 0.0
        cost = 0.0
        for t in range(1, len(path)):
            r = 0.0 if risk is None else float(risk[min(t, risk.shape[0] - 1), path[t][1], path[t][0]])
            cost += 1.0 + lam * r
        return cost

    def _near(self, obs: RobotObs, cell: Cell, radius: float) -> bool:
        cx, cy = self.grid.centroid(cell)
        return math.hypot(obs.x - cx, obs.y - cy) <= radius

    def is_idle(self, r: str, obs: Dict[str, RobotObs]) -> bool:
        """Robot stands at its current ADG vertex (not in the middle of a move)."""
        return self._near(obs[r], self.adg.current_cell(r), self.tol)

    def finished(self) -> bool:
        return self.adg is not None and self.adg.all_finished()

    # ------------------------------------------------------------------ main entry
    def step(self, now: float, robots: Dict[str, RobotObs], humans: Sequence[Tuple[object, float, float]] = ()):
        """Returns {robot: [cells]} for robots whose command changed."""
        self._now = now
        if self.predictor is not None:
            self.predictor.observe(now, humans)
        if self.adg is None:
            if not self._initial_plan(now, robots):
                return {r: [] for r in self.robots}
        self._update_progress(now, robots)
        if self.p.enable_repair:
            self._maybe_repair(now, robots)
        return self._dispatch(robots)

    # ------------------------------------------------------------------ planning
    def _initial_plan(self, now: float, robots: Dict[str, RobotObs]) -> bool:
        starts = self.grid.assign_unique_cells([(r, (robots[r].x, robots[r].y)) for r in self.robots])
        agents = [(starts[r], self.goals[r]) for r in self.robots]
        res = self._solve(agents)
        if not res.success and self.predictor is not None:
            # never keep the fleet idle: fall back to the risk-free problem, which is far easier
            # for CBS; risk is then handled by the execution-time repairs
            self._event(now, 'initial_plan_fallback', status=res.status)
            res = self._solve(agents, use_risk=False)
        if not res.success:
            self._event(now, 'initial_plan_failed', status=res.status)
            return False
        self.stats.initial_plans += 1
        self.adg = ActionDependencyGraph({r: p for r, p in zip(self.robots, res.paths)})
        for r in self.robots:
            self.last_progress[r] = now
        self._event(now, 'initial_plan', objective=res.objective, soc=res.sum_of_steps, makespan=res.makespan,
                    risk=res.total_risk, runtime=res.runtime_s, adg_edges=self.adg.num_edges())
        return True

    # ------------------------------------------------------------------ monitoring
    def _update_progress(self, now: float, robots: Dict[str, RobotObs]):
        for r in self.robots:
            pend = self.adg.pending(r)
            if not pend:
                continue
            if self.action_start[r] is None and self.sent[r]:
                self.action_start[r] = now
            # an action counts as done when the robot reached its target; check the
            # farthest dispatched target first (corners may be cut)
            n_sent = min(len(self.sent[r]), len(pend))
            for a in reversed(pend[:n_sent]):
                if self._near(robots[r], a.dst, self.tol):
                    steps = a.index + 1 - self.adg.done[r]
                    self.adg.mark_done(r, a.index)
                    self.sent[r] = self.sent[r][steps:]   # sent stays aligned with pending actions
                    if self.p.adapt_step_duration and self.action_start[r] is not None and steps > 0:
                        dur = (now - self.action_start[r]) / steps
                        if 0.5 < dur < 20.0:
                            self.step_duration = float(np.clip(0.8 * self.step_duration + 0.2 * dur, 1.5, 8.0))
                    self.action_start[r] = now if self.adg.pending(r) and self.sent[r] else None
                    self.last_progress[r] = now
                    break

    def _repair_candidate(self, now: float, robots: Dict[str, RobotObs]) -> Optional[Tuple[str, str]]:
        best = None
        for r in self.robots:
            if self.adg.finished(r) or now - self.last_repair[r] < self.p.repair_cooldown:
                continue
            if not self.is_idle(r, robots):
                continue
            disp = self.adg.dispatchable(r)
            # (1) blocked: allowed to move but not moving
            if disp and robots[r].speed < self.p.stuck_speed and now - self.last_progress[r] > self.p.blocked_timeout:
                waited = now - self.last_progress[r]
                if best is None or waited > best[2]:
                    best = (r, 'blocked', waited)
                continue
            # (2) predicted human occupancy on the next cells of the robot's path
            risk = self._risk_map(now)
            if risk is not None and self.p.risk_weight > 0 and best is None:
                pend = self.adg.pending(r)[:self.p.risk_lookahead]
                for k, a in enumerate(pend, start=1):
                    rv = risk[min(k, risk.shape[0] - 1), a.dst[1], a.dst[0]]
                    if rv > self.p.risk_trigger:
                        best = (r, 'risk', float(rv))
                        break
        return None if best is None else (best[0], best[1])

    def _maybe_repair(self, now: float, robots: Dict[str, RobotObs]):
        if self.stats.replans >= self.p.max_repairs:
            return
        cand = self._repair_candidate(now, robots)
        if cand is None:
            return
        r, why = cand
        if why == 'blocked':
            self.stats.blocked_triggers += 1
        else:
            self.stats.risk_triggers += 1
        self.last_repair[r] = now
        self.repair(now, r, robots, reason=why)

    def repair(self, now: float, robot: str, robots: Dict[str, RobotObs], reason: str = '') -> bool:
        """Local plan repair with escalation. Returns True if a new plan was installed."""
        idle = {r for r in self.robots if self.is_idle(r, robots)}
        levels: List[Set[str]] = [{robot}]
        coupled = ({robot} | self.adg.coupled_robots(robot)) & idle
        if self.p.max_repair_levels >= 2 and coupled != levels[-1]:
            levels.append(coupled)
        if self.p.max_repair_levels >= 3 and idle != levels[-1]:
            levels.append(set(idle))
        outcomes = []
        for level, S in enumerate(levels, start=1):
            outcome = self._try_replan(now, S, robots, reason, level)
            if outcome == 'installed':
                return True
            outcomes.append(outcome)
        if 'not_better' in outcomes:
            self.stats.kept_plans += 1
        else:
            self.stats.repair_failures += 1
            self._event(now, 'repair_failed', robot=robot, reason=reason)
        return False

    def _try_replan(self, now: float, S: Set[str], robots: Dict[str, RobotObs], reason: str, level: int) -> str:
        """Returns 'installed', 'failed' (no plan), 'not_better' or 'rejected' (safety check)."""
        adg = self.adg
        others = [r for r in self.robots if r not in S]
        s_cells = {adg.current_cell(r) for r in S}
        # others must not rush into cells the re-planned robots still occupy
        min_start = {}
        for o in others:
            for a in adg.pending(o):
                if a.dst in s_cells:
                    min_start[(o, a.index)] = 1 + self.p.repair_slack
        reserved = adg.earliest_start_schedule(others, min_start)
        order = [r for r in self.robots if r in S]
        agents = [(adg.current_cell(r), self.goals[r]) for r in order]
        res = self._solve(agents, [reserved[o] for o in others])
        if not res.success:
            self._event(now, 'replan_failed', robots=order, level=level, status=res.status)
            return 'failed'
        if reason == 'risk':
            # hysteresis: a predicted (not actual) obstruction only justifies a new plan if it is
            # clearly better than executing the current one
            current = adg.earliest_start_schedule(self.robots)
            old = sum(self.plan_cost(current[r]) for r in order)
            if res.objective > old - self.p.min_improvement:
                self._event(now, 'repair_not_better', robots=order, level=level, old=old, new=res.objective)
                return 'not_better'
        joint = dict(reserved)
        joint.update({r: p for r, p in zip(order, res.paths)})
        all_agents = [(joint[r][0], self.goals[r]) for r in self.robots]
        err = validate_joint_plan(self.grid.dimx, self.grid.dimy, self.grid.obstacles, all_agents,
                                  [joint[r] for r in self.robots], robustness=self.p.robustness)
        if err is not None:
            self.stats.rejected_repairs += 1
            self._event(now, 'repair_rejected', robots=order, why=err)
            return 'rejected'
        try:
            new = ActionDependencyGraph(joint)
        except CyclicADGError:
            self.stats.rejected_repairs += 1
            self._event(now, 'repair_rejected', robots=order, why='cyclic ADG')
            return 'rejected'
        # Robots that keep their plan must still be allowed to execute what they were sent.
        for o in others:
            n_sent = min(len(self.sent[o]), len(adg.pending(o)))
            if n_sent and len(new.dispatchable(o)) < n_sent:
                self.stats.rejected_repairs += 1
                self._event(now, 'repair_rejected', robots=order, why=f'would revoke dispatched actions of {o}')
                return 'rejected'
        self.adg = new
        for r in S:
            self.last_progress[r] = now
            self.action_start[r] = None
        if level == 1:
            self.stats.local_repairs += 1
        elif level == 2:
            self.stats.coupled_repairs += 1
        else:
            self.stats.global_replans += 1
        self._event(now, 'repair', robots=order, level=level, reason=reason, objective=res.objective,
                    risk=res.total_risk, runtime=res.runtime_s)
        return 'installed'

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, robots: Dict[str, RobotObs]) -> Dict[str, List[Cell]]:
        """Command = cells of the dispatchable actions still to be driven through. A command
        is only (re-)sent when it differs from the remaining part of the previous one."""
        commands = {}
        guard = self.p.guard_radius_frac * self.grid.cell_size
        for r in self.robots:
            cells = []
            for a in self.adg.dispatchable(r):
                if any(o != r and self._near(robots[o], a.dst, guard) for o in self.robots):
                    break
                cells.append(a.dst)
            if cells != self.sent[r]:
                if not cells and not self.sent[r]:
                    continue
                self.sent[r] = cells
                if self.action_start[r] is None and cells:
                    self.action_start[r] = self._now
                commands[r] = list(cells)
        return commands
