# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference implementation of the risk-aware, k-robust CBS.

It mirrors ``remroc_cbs_library/include/remroc_cbs_library/risk_cbs.hpp`` (same
integer fixed-point costs, same conflict model, same search bound) so that the two
implementations can be cross-validated: both are optimal, hence their objective values
must be identical on every instance (tie-breaking, and therefore the returned paths,
may differ). The C++ solver is the one used in ROS; this module is the fallback of the
simulator and the oracle of the tests.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]
MOVES5 = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class SolveResult:
    success: bool
    status: str
    paths: List[List[Cell]] = field(default_factory=list)
    objective: float = 0.0
    objective_scaled: int = 0
    sum_of_steps: int = 0
    makespan: int = 0
    total_risk: float = 0.0
    high_level_expanded: int = 0
    low_level_expanded: int = 0
    runtime_s: float = 0.0


class _Timeout(Exception):
    pass


def _risk_value(risk: Optional[np.ndarray], t: int, c: Cell) -> float:
    if risk is None:
        return 0.0
    t = min(max(t, 0), risk.shape[0] - 1)
    v = float(risk[t, c[1], c[0]])
    return min(1.0, max(0.0, v))


def path_risk(risk: Optional[np.ndarray], path: Sequence[Cell]) -> float:
    return float(sum(_risk_value(risk, t, path[t]) for t in range(1, len(path))))


class _Reservations:
    def __init__(self, reservations: Sequence[Sequence[Cell]], k: int):
        self.k = max(0, k)
        self.vertex = set()
        self.edge = set()
        self.parked_from: Dict[Cell, int] = {}
        self.last_dynamic: Dict[Cell, int] = {}
        self.horizon = 0
        for path in reservations:
            if not path:
                continue
            n = len(path)
            for t, c in enumerate(path):
                c = tuple(c)
                if t < n - 1:
                    self.vertex.add((t, c))
                    self.last_dynamic[c] = max(self.last_dynamic.get(c, -1), t)
                    nx = tuple(path[t + 1])
                    if nx != c:
                        self.edge.add((t, c, nx))
                else:
                    self.parked_from[c] = min(self.parked_from.get(c, 1 << 60), t)
            self.horizon = max(self.horizon, n - 1)

    def vertex_blocked(self, t: int, c: Cell) -> bool:
        if t + self.k >= self.parked_from.get(c, 1 << 60):
            return True
        for tt in range(max(0, t - self.k), t + self.k + 1):
            if (tt, c) in self.vertex:
                return True
        return False

    def edge_blocked(self, t: int, a: Cell, b: Cell) -> bool:
        return a != b and (t, b, a) in self.edge


class _Constraints:
    __slots__ = ('vertex', 'edge', 'last_vertex', 'max_time')

    def __init__(self):
        self.vertex = set()
        self.edge = set()
        self.last_vertex: Dict[Cell, int] = {}
        self.max_time = 0

    def copy(self) -> '_Constraints':
        c = _Constraints()
        c.vertex = set(self.vertex)
        c.edge = set(self.edge)
        c.last_vertex = dict(self.last_vertex)
        c.max_time = self.max_time
        return c

    def add_vertex(self, t: int, c: Cell):
        self.vertex.add((t, c))
        self.last_vertex[c] = max(self.last_vertex.get(c, -1), t)
        self.max_time = max(self.max_time, t)

    def add_edge(self, t: int, a: Cell, b: Cell):
        self.edge.add((t, a, b))
        self.max_time = max(self.max_time, t + 1)


class RiskCBS:
    def __init__(self, dimx: int, dimy: int, obstacles, agents: Sequence[Tuple[Cell, Cell]],
                 risk: Optional[np.ndarray] = None, risk_weight: float = 0.0,
                 reservations: Sequence[Sequence[Cell]] = (), robustness: int = 0,
                 time_limit: float = 0.0, cost_scale: int = 1000, max_high_level_nodes: int = 200000,
                 suboptimality: float = 1.0):
        self.dimx, self.dimy = dimx, dimy
        self.obstacles = set(tuple(o) for o in obstacles)
        self.agents = [(tuple(s), tuple(g)) for s, g in agents]
        self.risk = None if risk is None or risk.size == 0 else np.asarray(risk, dtype=np.float32)
        if self.risk is not None and self.risk.shape[1:] != (dimy, dimx):
            raise ValueError(f'risk map shape {self.risk.shape} does not match grid ({dimy}, {dimx})')
        self.risk_weight = float(risk_weight)
        self.k = max(0, int(robustness))
        self.res = _Reservations(reservations, self.k)
        self.time_limit = time_limit
        self.scale = int(cost_scale)
        self.max_hl = max_high_level_nodes
        self.w = max(1.0, float(suboptimality))
        self.deadline = time.monotonic() + time_limit if time_limit > 0 else None
        self.num_free = sum(1 for x in range(dimx) for y in range(dimy) if (x, y) not in self.obstacles)
        self.static_horizon = max(self.res.horizon, 0 if self.risk is None else self.risk.shape[0])
        self.heuristics = [self._bfs(g) for _, g in self.agents]
        self.hl_expanded = 0
        self.ll_expanded = 0
        # Precomputed integer step costs per (layer, cell) for speed.
        self._scaled_risk = None
        if self.risk is not None and self.risk_weight > 0:
            r = np.clip(self.risk.astype(np.float64), 0.0, 1.0)
            v = float(self.scale) * self.risk_weight * r
            self._scaled_risk = np.floor(v + 0.5).astype(np.int64)   # llround for v >= 0

    def _free(self, c: Cell) -> bool:
        return 0 <= c[0] < self.dimx and 0 <= c[1] < self.dimy and c not in self.obstacles

    def _bfs(self, goal: Cell) -> Dict[Cell, int]:
        if not self._free(goal):
            return {}
        dist = {goal: 0}
        q = [goal]
        for c in q:
            for dx, dy in MOVES5[1:]:
                n = (c[0] + dx, c[1] + dy)
                if self._free(n) and n not in dist:
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    def step_cost(self, t: int, c: Cell) -> int:
        if self._scaled_risk is None:
            return self.scale
        tt = min(max(t, 0), self._scaled_risk.shape[0] - 1)
        return self.scale + int(self._scaled_risk[tt, c[1], c[0]])

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _Timeout()

    # ------------------------------------------------------------------ low level
    def low_level(self, agent: int, cons: _Constraints) -> Optional[Tuple[List[Cell], int]]:
        start, goal = self.agents[agent]
        h = self.heuristics[agent]
        if start not in h:
            return None
        if goal in self.res.parked_from:
            return None
        if (0, start) in cons.vertex:
            return None
        last_res = self.res.last_dynamic.get(goal, -1)
        last_goal_block = max(cons.last_vertex.get(goal, -1), -1 if last_res < 0 else last_res + self.k)
        max_time = max(cons.max_time, self.static_horizon) + self.k + self.num_free + 1

        scale = self.scale
        counter = itertools.count()
        # entries: (f, -g, tie, t, cell, g)
        open_heap = [(scale * h[start], 0, next(counter), 0, start, 0)]
        parent: Dict[Tuple[int, Cell], Optional[Tuple[int, Cell]]] = {(0, start): None}
        best_g: Dict[Tuple[int, Cell], int] = {(0, start): 0}
        expansions = 0
        while open_heap:
            f, _, _, t, c, g = heapq.heappop(open_heap)
            if best_g.get((t, c), 1 << 62) < g:
                continue
            self.ll_expanded += 1
            expansions += 1
            if (expansions & 1023) == 0:
                self._check_deadline()
            if c == goal and t > last_goal_block:
                path = []
                node = (t, c)
                while node is not None:
                    path.append(node[1])
                    node = parent[node]
                return path[::-1], g
            if t >= max_time:
                continue
            nt = t + 1
            for dx, dy in MOVES5:
                n = (c[0] + dx, c[1] + dy)
                if n not in h:   # obstacle, out of bounds or disconnected from the goal
                    continue
                if (nt, n) in cons.vertex or (t, c, n) in cons.edge:
                    continue
                if self.res.vertex_blocked(nt, n) or self.res.edge_blocked(t, c, n):
                    continue
                ng = g + self.step_cost(nt, n)
                key = (nt, n)
                if best_g.get(key, 1 << 62) <= ng:
                    continue
                best_g[key] = ng
                parent[key] = (t, c)
                heapq.heappush(open_heap, (ng + scale * h[n], -ng, next(counter), nt, n, ng))
        return None

    # ------------------------------------------------------------------ high level
    @staticmethod
    def _pos(p: List[Cell], t: int) -> Cell:
        return p[t] if t < len(p) else p[-1]

    def first_conflict(self, paths: List[List[Cell]]):
        max_t = max(len(p) for p in paths) - 1
        n = len(paths)
        pos = self._pos
        for t in range(max_t + 1):
            for i in range(n):
                pi = pos(paths[i], t)
                for j in range(i + 1, n):
                    if pi == pos(paths[j], t):
                        return ('v', t, t, i, j, pi, None)
            for d in range(1, self.k + 1):
                for i in range(n):
                    pi = pos(paths[i], t)
                    for j in range(n):
                        if i != j and pi == pos(paths[j], t + d):
                            return ('v', t, t + d, i, j, pi, None)
            if t == max_t:
                break
            for i in range(n):
                ia, ib = pos(paths[i], t), pos(paths[i], t + 1)
                if ia == ib:
                    continue
                for j in range(i + 1, n):
                    if ia == pos(paths[j], t + 1) and ib == pos(paths[j], t):
                        return ('e', t, t, i, j, ia, ib)
        return None

    def count_conflicting_pairs(self, paths: List[List[Cell]]) -> int:
        pos = self._pos
        k = self.k
        count = 0
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                T = max(len(paths[i]), len(paths[j])) + k
                hit = False
                for t in range(T + 1):
                    pi = pos(paths[i], t)
                    for d in range(-k, k + 1):
                        if t + d >= 0 and pi == pos(paths[j], t + d):
                            hit = True
                            break
                    if not hit:
                        ni = pos(paths[i], t + 1)
                        hit = pi != ni and pi == pos(paths[j], t + 1) and ni == pos(paths[j], t)
                    if hit:
                        break
                count += hit
        return count

    def _bound(self, c: int) -> int:
        return int(math.floor(self.w * c + 1e-9))

    def solve(self) -> SolveResult:
        t0 = time.monotonic()
        try:
            res = self._solve()
        except _Timeout:
            res = SolveResult(False, 'timeout')
        res.high_level_expanded = self.hl_expanded
        res.low_level_expanded = self.ll_expanded
        res.runtime_s = time.monotonic() - t0
        return res

    def _solve(self) -> SolveResult:
        starts, goals = set(), set()
        for s, g in self.agents:
            if not self._free(s):
                return SolveResult(False, 'invalid_start')
            if not self._free(g):
                return SolveResult(False, 'invalid_goal')
            if s in starts:
                return SolveResult(False, 'duplicate_start')
            if g in goals:
                return SolveResult(False, 'duplicate_goal')
            starts.add(s)
            goals.add(g)
            if self.res.vertex_blocked(0, s):
                return SolveResult(False, 'start_reserved')

        n = len(self.agents)
        cons = [_Constraints() for _ in range(n)]
        paths, costs = [], []
        for i in range(n):
            r = self.low_level(i, cons[i])
            if r is None:
                return SolveResult(False, 'no_single_agent_path')
            paths.append(r[0])
            costs.append(r[1])
        # focal search (w = 1: optimal CBS with fewest-conflicts tie-breaking); mirrors risk_cbs.hpp
        nodes = []
        open_set = []           # heap of (cost, id); lazily deleted
        focal = []              # heap of (conflicts, cost, id); lazily deleted
        expanded = set()
        in_focal = set()
        bound = self._bound(sum(costs))

        def push(total, paths, costs, cons):
            nid = len(nodes)
            conf = self.count_conflicting_pairs(paths)
            nodes.append((total, conf, paths, costs, cons))
            heapq.heappush(open_set, (total, nid))
            if total <= bound:
                heapq.heappush(focal, (conf, total, nid))
                in_focal.add(nid)

        push(sum(costs), paths, costs, cons)
        while True:
            while open_set and open_set[0][1] in expanded:
                heapq.heappop(open_set)
            if not open_set:
                break
            self._check_deadline()
            if self.hl_expanded >= self.max_hl:
                return SolveResult(False, 'node_limit')
            new_bound = self._bound(open_set[0][0])
            if new_bound > bound:
                bound = new_bound
                for total, nid in open_set:
                    if total <= bound and nid not in in_focal and nid not in expanded:
                        heapq.heappush(focal, (nodes[nid][1], total, nid))
                        in_focal.add(nid)
            while focal[0][2] in expanded:
                heapq.heappop(focal)
            _, _, nid = heapq.heappop(focal)
            expanded.add(nid)
            total, _, paths, costs, cons = nodes[nid]
            self.hl_expanded += 1
            conflict = self.first_conflict(paths)
            if conflict is None:
                return self._result(paths, total)
            kind, t1, t2, a1, a2, c1, c2 = conflict
            for side in (0, 1):
                agent = a1 if side == 0 else a2
                new_cons = list(cons)
                nc = cons[agent].copy()
                if kind == 'v':
                    nc.add_vertex(t1 if side == 0 else t2, c1)
                elif side == 0:
                    nc.add_edge(t1, c1, c2)
                else:
                    nc.add_edge(t1, c2, c1)
                new_cons[agent] = nc
                r = self.low_level(agent, nc)
                if r is None:
                    continue
                new_paths = list(paths)
                new_costs = list(costs)
                new_paths[agent], new_costs[agent] = r
                push(sum(new_costs), new_paths, new_costs, new_cons)
        return SolveResult(False, 'no_solution')

    def _result(self, paths: List[List[Cell]], total: int) -> SolveResult:
        return SolveResult(
            True, 'success', [list(p) for p in paths], total / self.scale, total,
            sum(len(p) - 1 for p in paths), max(len(p) - 1 for p in paths),
            sum(path_risk(self.risk, p) for p in paths))


def solve(dimx, dimy, obstacles, agents, risk=None, risk_weight=0.0, reservations=(), robustness=0,
          time_limit=0.0, cost_scale=1000, max_high_level_nodes=200000, suboptimality=1.0) -> SolveResult:
    return RiskCBS(dimx, dimy, obstacles, agents, risk, risk_weight, reservations, robustness,
                   time_limit, cost_scale, max_high_level_nodes, suboptimality).solve()


def validate_joint_plan(dimx, dimy, obstacles, agents, paths, reservations=(), robustness=0) -> Optional[str]:
    """Independent checker. Returns None if valid, otherwise a description of the violation."""
    obstacles = set(tuple(o) for o in obstacles)

    def free(c):
        return 0 <= c[0] < dimx and 0 <= c[1] < dimy and c not in obstacles

    for i, ((s, g), p) in enumerate(zip(agents, paths)):
        if not p or tuple(p[0]) != tuple(s) or tuple(p[-1]) != tuple(g):
            return f'agent {i}: wrong start/goal'
        for t, c in enumerate(p):
            if not free(tuple(c)):
                return f'agent {i}: blocked cell {c} at t={t}'
            if t and abs(c[0] - p[t - 1][0]) + abs(c[1] - p[t - 1][1]) > 1:
                return f'agent {i}: jump at t={t}'
    allp = [list(map(tuple, p)) for p in paths] + [list(map(tuple, r)) for r in reservations]
    planned = len(paths)
    T = max(len(p) for p in allp) + robustness + 1

    def pos(p, t):
        return p[t] if t < len(p) else p[-1]

    for t in range(T):
        for i in range(len(allp)):
            for j in range(len(allp)):
                if i == j or (i >= planned and j >= planned):
                    continue
                for d in range(0, robustness + 1):
                    if pos(allp[i], t) == pos(allp[j], t + d):
                        return f'{i} and {j} too close in {pos(allp[i], t)} at t={t}, t+{d}'
                if (pos(allp[i], t) != pos(allp[i], t + 1) and pos(allp[i], t) == pos(allp[j], t + 1)
                        and pos(allp[i], t + 1) == pos(allp[j], t)):
                    return f'{i} and {j} swap at t={t}'
    return None
