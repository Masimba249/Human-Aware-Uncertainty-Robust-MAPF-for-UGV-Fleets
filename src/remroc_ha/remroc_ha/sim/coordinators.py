# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Coordinators for the grid-level simulator.

``MapfBaseline`` and ``PbcBaseline`` are ports of REMROC's ``coordinator_mapf.py`` and
``coordinator_pbc.py`` (same decision logic, same constants); ``HumanAwareSim`` wraps
the middleware-agnostic ``HumanAwareCoordinator`` that also runs in the ROS node.

Every coordinator returns, for robots whose command changed, a list of metric
waypoints to drive through (an empty list stops the robot).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree

from ..coordination import CoordinatorParams, HumanAwareCoordinator, RobotObs
from ..grid import GridSpec
from ..prediction import HumanRiskPredictor, MapOfDynamics
from ..solver import SolverBackend

Point = Tuple[float, float]


class SimCoordinator:
    period = 0.5

    def step(self, t: float, robots: Dict[str, RobotObs], humans) -> Dict[str, List[Point]]:
        raise NotImplementedError

    def stats(self) -> dict:
        return {}

    def events(self) -> list:
        return []


class MapfBaseline(SimCoordinator):
    """Iterative MAPF (coordinator_mapf.py): every 0.55 s map the robots to their nearest
    free cells, solve plain CBS (risk-free, k = 0), and send each robot its path through
    the solution poses up to the first wait. If the solver fails the robots are stopped."""

    period = 0.55

    def __init__(self, grid: GridSpec, names: Sequence[str], goals: Dict[str, Point], solver: SolverBackend,
                 time_limit: float = 5.0):
        self.grid = grid
        self.names = list(names)
        self.goal_cells = grid.assign_unique_cells([(r, goals[r]) for r in self.names])
        self.solver = solver
        self.time_limit = time_limit
        self.calls = 0
        self.failures = 0
        self.solve_time = 0.0

    def step(self, t, robots, humans):
        cells = self.grid.assign_unique_cells([(r, (robots[r].x, robots[r].y)) for r in self.names])
        res = self.solver.solve(self.grid, [(cells[r], self.goal_cells[r]) for r in self.names],
                                time_limit=self.time_limit)
        self.calls += 1
        self.solve_time += res.runtime_s
        if not res.success:
            self.failures += 1
            return {r: [] for r in self.names}
        cmds = {}
        for r, path in zip(self.names, res.paths):
            poses = [self.grid.centroid(c) for c in path]
            for i in range(len(poses) - 1):            # "wait" -> truncate after the first repeat
                if poses[i] == poses[i + 1]:
                    poses = poses[:i + 1]
                    break
            cmds[r] = poses
        return cmds

    def stats(self):
        # every solver call after the first one is a (global) re-plan
        return {'replans': max(0, self.calls - 1), 'solver_calls': self.calls, 'solver_failures': self.failures,
                'solver_time': self.solve_time}


class PbcBaseline(SimCoordinator):
    """Path-based coordination (coordinator_pbc.py): fixed global paths, critical sections
    between the remaining paths (points closer than 0.5 m) and precedence to the robot
    that is closer to the section; the other one stops 10 path points before it."""

    period = 0.5

    def __init__(self, grid: GridSpec, names: Sequence[str], goals: Dict[str, Point],
                 distance_threshold: float = 0.5, spacing: float = 0.05):
        self.grid = grid
        self.names = list(names)
        self.goals = goals
        self.distance_threshold = distance_threshold
        self.spacing = spacing
        self.paths: Dict[str, np.ndarray] = {}

    def _global_path(self, start: Point, goal: Point) -> np.ndarray:
        # Stand-in for the nav2 global planner: shortest grid path through cell centroids,
        # densified to the ~5 cm spacing of nav2 paths.
        sc = self.grid.nearest_free_cell(*start)
        gc = self.grid.nearest_free_cell(*goal)
        cells = self.grid.shortest_path(sc, gc) or [sc]
        pts = [start] + [self.grid.centroid(c) for c in cells[1:]] + [goal]
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / self.spacing)))
            for k in range(1, n + 1):
                dense.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
        return np.array(dense)

    def step(self, t, robots, humans):
        if not self.paths:
            for r in self.names:
                self.paths[r] = self._global_path((robots[r].x, robots[r].y), self.goals[r])
        # compute_paths_todo: truncate at the closest path point
        todo = {}
        for r in self.names:
            p = self.paths[r]
            idx = int(np.argmin(np.linalg.norm(p - np.array([robots[r].x, robots[r].y]), axis=1)))
            todo[r] = p[idx:]
        # compute_critical_sections
        trees = {}
        for r in self.names:
            pts = todo[r] if len(todo[r]) else np.array([[robots[r].x, robots[r].y]])
            trees[r] = KDTree(pts)
        critical = {}
        for r in self.names:
            pts = todo[r] if len(todo[r]) else np.array([[robots[r].x, robots[r].y]])
            critical[r] = {}
            for o in self.names:
                if o == r:
                    continue
                close = trees[o].query_ball_point(pts, self.distance_threshold)
                sections, cur = [], {'indices': [], 'close_points_range': []}
                for index, point in enumerate(close):
                    if point:
                        point = sorted(point)
                        cur['indices'].append(index)
                        cur['close_points_range'].append([point[0], point[-1]])
                        if index != len(close) - 1:
                            if not close[index + 1]:
                                sections.append(cur)
                                cur = {'indices': [], 'close_points_range': []}
                        else:
                            sections.append(cur)
                            cur = {'indices': [], 'close_points_range': []}
                critical[r][o] = sections
        # send_action_goal: truncate before the first critical section where the other robot has precedence
        cmds = {}
        for r in self.names:
            path = todo[r]
            index = len(path)
            for o, sections in critical[r].items():
                for section in sections:
                    first_conflict_index = section['indices'][0]
                    min_close = min(x[0] for x in section['close_points_range'])
                    # (identical for both section types in coordinator_pbc.py)
                    if first_conflict_index >= min_close and min_close < index:
                        index = max(0, first_conflict_index - 10)
            cmds[r] = [tuple(p) for p in path[:index]]
        return cmds

    def stats(self):
        return {'replans': 0, 'solver_calls': 0}


class HumanAwareSim(SimCoordinator):
    period = 0.5

    def __init__(self, grid: GridSpec, names: Sequence[str], goals: Dict[str, Point], solver: SolverBackend,
                 params: CoordinatorParams, prediction: str = 'cv', mod: Optional[MapOfDynamics] = None,
                 walkable=None, seed: int = 0, horizon: int = 20, log=None):
        self.grid = grid
        goal_cells = grid.assign_unique_cells([(r, goals[r]) for r in names])
        predictor = None
        if params.use_prediction:
            predictor = HumanRiskPredictor(grid, params.step_duration, horizon, method=prediction, mod=mod,
                                           walkable=walkable, seed=seed)
        self.coord = HumanAwareCoordinator(grid, goal_cells, solver, params, predictor, log=log)

    def step(self, t, robots, humans):
        cmds = self.coord.step(t, robots, humans)
        return {r: [self.grid.centroid(c) for c in cells] for r, cells in cmds.items()}

    def stats(self):
        s = self.coord.stats
        return {'replans': s.replans, 'solver_calls': s.solver_calls, 'solver_time': s.solver_time,
                'local_repairs': s.local_repairs, 'coupled_repairs': s.coupled_repairs,
                'global_replans': s.global_replans, 'repair_failures': s.repair_failures,
                'rejected_repairs': s.rejected_repairs, 'kept_plans': s.kept_plans,
                'blocked_triggers': s.blocked_triggers,
                'risk_triggers': s.risk_triggers, 'step_duration': self.coord.step_duration}

    def events(self):
        return self.coord.events
