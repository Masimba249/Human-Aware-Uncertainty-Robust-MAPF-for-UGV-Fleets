# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Grid-level simulation of a REMROC experiment (no ROS / Gazebo required).

The simulator uses the very same inputs as the Gazebo experiments -- MAPF grid
(``params/mapf_<map>.yaml``), experiment file (robot starts and goals), and the human
trajectories that are scripted into the SDF worlds -- and robot dynamics that
reproduce the plan-execution gap (see ``robot.py``). It is meant for fast, repeatable
comparisons and ablations; the Gazebo runs remain the reference (docs/EXPERIMENTS.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..coordination import CoordinatorParams, RobotObs
from ..grid import GridSpec
from ..humans import HumanScript, load_humans_json
from ..occupancy import OccupancyMap
from ..prediction import MapOfDynamics
from ..solver import SolverBackend, make_solver
from .coordinators import HumanAwareSim, MapfBaseline, PbcBaseline, SimCoordinator
from .robot import RobotParams, SimRobot

HUMAN_RADIUS = 0.3

EXPERIMENT_FILES = {'narrow_corridors': 'exp_ha_narrow_corridors.yaml', 'depot': 'exp_ha_depot.yaml'}


@dataclass
class Scenario:
    world: str
    n_humans: int
    sample: int
    grid: GridSpec
    robots: List[Tuple[str, Tuple[float, float, float], Tuple[float, float]]]
    humans: List[HumanScript]
    mod_path: Optional[Path] = None
    map_yaml: Optional[Path] = None

    @property
    def names(self) -> List[str]:
        return [r[0] for r in self.robots]

    @property
    def goals(self) -> Dict[str, Tuple[float, float]]:
        return {r[0]: r[2] for r in self.robots}


def load_scenario(remroc_dir, world: str, n_humans: int, sample: int, experiment_yaml: str = None) -> Scenario:
    remroc = Path(remroc_dir)
    grid = GridSpec.from_mapf_yaml(remroc / 'params' / f'mapf_{world}.yaml', world)
    exp = yaml.safe_load((remroc / 'params' / (experiment_yaml or EXPERIMENT_FILES[world])).read_text())
    robots = [(f"{r['type']}_{r['id']}", tuple(map(float, r['start'])), tuple(map(float, r['goal'][:2])))
              for r in exp['robots']]
    humans = load_humans_json(remroc / 'worlds' / 'humans' / f'{world}_{n_humans}_{sample}.json')
    mod = remroc / 'worlds' / 'humans' / f'{world}_mod.npz'
    map_yaml = remroc / 'worlds' / 'maps' / f'{world}.yaml'
    return Scenario(world, n_humans, sample, grid, robots, humans, mod if mod.exists() else None,
                    map_yaml if map_yaml.exists() else None)


# Coordinator variants evaluated in the simulator.
VARIANTS = {
    'mapf': {},                      # REMROC coordinator_mapf.py (baseline)
    'pbc': {},                       # REMROC coordinator_pbc.py (baseline)
    'ha_cbs': {'prediction': 'cv'},  # proposed: risk-aware CBS + ADG + local repair, CV prediction
    'ha_cbs_mod': {'prediction': 'mod'},                               # same, map-of-dynamics prediction
    'adg_cbs': {'params': {'use_prediction': False, 'risk_weight': 0.0}},  # ablation: no human awareness
    'ha_cbs_norepair': {'prediction': 'cv', 'params': {'enable_repair': False}},  # ablation: no repair
}


def make_coordinator(name: str, scn: Scenario, solver: SolverBackend, seed: int,
                     param_overrides: Optional[dict] = None, log=None) -> SimCoordinator:
    if name == 'mapf':
        return MapfBaseline(scn.grid, scn.names, scn.goals, solver)
    if name == 'pbc':
        return PbcBaseline(scn.grid, scn.names, scn.goals)
    if name not in VARIANTS:
        raise ValueError(f'unknown coordinator {name}')
    spec = VARIANTS[name]
    params = replace(CoordinatorParams(), **spec.get('params', {}), **(param_overrides or {}))
    mod, walkable = None, None
    # the map of dynamics (fitted on separate training worlds) provides the static prior for
    # both predictors and the flow field for the particle predictor
    if 'prediction' in spec and scn.mod_path is not None:
        mod = MapOfDynamics.load(scn.mod_path)
    if scn.map_yaml is not None and spec.get('prediction') == 'mod':
        occ = OccupancyMap.from_yaml(scn.map_yaml)
        walkable = occ.is_free
    return HumanAwareSim(scn.grid, scn.names, scn.goals, solver, params, spec.get('prediction', 'cv'), mod,
                         walkable, seed=seed, log=log)


def run_episode(scn: Scenario, coordinator: str, seed: int = 0, time_limit: float = 400.0, dt: float = 0.1,
                log_dt: float = 0.2, solver: Optional[SolverBackend] = None, robot_params: Optional[RobotParams] = None,
                param_overrides: Optional[dict] = None, obs_noise: float = 0.05, verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    solver = solver or make_solver()
    rp = robot_params or RobotParams()
    robots = {name: SimRobot(name, s[0], s[1], s[2], rp, np.random.default_rng(rng.integers(2 ** 31)))
              for name, s, _ in scn.robots}
    coord = make_coordinator(coordinator, scn, solver, seed, param_overrides, log=print if verbose else None)
    goals = scn.goals
    frames = []
    next_coord, next_log = 0.0, 0.0
    t = 0.0
    steps = int(round(time_limit / dt))
    for _ in range(steps + 1):
        hpos = [(h.name, *h.position(t)) for h in scn.humans]
        if t >= next_coord - 1e-9:
            obs = {n: RobotObs(r.x, r.y, r.speed) for n, r in robots.items()}
            noisy = [(hid, x + rng.normal(0, obs_noise), y + rng.normal(0, obs_noise)) for hid, x, y in hpos]
            for name, pts in coord.step(t, obs, noisy).items():
                robots[name].set_path(pts)
            next_coord += coord.period
        if t >= next_log - 1e-9:
            frames.append({'t': round(t, 3),
                           'robots': {n: [round(r.x, 3), round(r.y, 3), round(r.speed, 3)] for n, r in robots.items()},
                           'humans': [[round(x, 3), round(y, 3)] for _, x, y in hpos]})
            next_log += log_dt
        done = all(math.hypot(r.x - goals[n][0], r.y - goals[n][1]) < 0.3 and r.speed < 0.02 and not r.path
                   for n, r in robots.items())
        if done:
            break
        for n, r in robots.items():
            obstacles = [(o.x, o.y, rp.body_radius) for m, o in robots.items() if m != n] if rp.avoid_robots else []
            obstacles += [(x, y, HUMAN_RADIUS) for _, x, y in hpos]
            r.step(t, dt, obstacles)
        t += dt
    return {
        'meta': {'coordinator': coordinator, 'world': scn.world, 'number_of_humans': scn.n_humans,
                 'sample': scn.sample, 'seed': seed, 'robots': scn.names,
                 'goals': {n: list(g) for n, g in goals.items()}, 'time_limit': time_limit,
                 'simulator': 'remroc_ha grid-level simulator', 'solver': solver.name},
        'frames': frames,
        'events': coord.events(),
        'coordinator_stats': coord.stats(),
    }
