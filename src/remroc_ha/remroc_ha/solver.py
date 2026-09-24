# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Solver back ends with one interface.

* ``PythonSolver``: the pure-Python reference implementation (always available).
* ``CliSolver``: the C++ ``risk_cbs_cli`` executable (fast; used by the simulator).
* The ROS coordinator wraps the ``mapf_solver`` service in the same interface
  (see ``remroc/coordinator_ha_cbs.py``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .grid import GridSpec
from .risk_cbs import SolveResult, solve as py_solve

Cell = Tuple[int, int]


class SolverBackend:
    name = 'abstract'

    def solve(self, grid: GridSpec, agents: Sequence[Tuple[Cell, Cell]], risk: Optional[np.ndarray] = None,
              risk_weight: float = 0.0, reservations: Sequence[Sequence[Cell]] = (), robustness: int = 0,
              time_limit: float = 0.0, suboptimality: float = 1.0) -> SolveResult:
        raise NotImplementedError


class PythonSolver(SolverBackend):
    name = 'python'

    def solve(self, grid, agents, risk=None, risk_weight=0.0, reservations=(), robustness=0, time_limit=0.0,
              suboptimality=1.0):
        return py_solve(grid.dimx, grid.dimy, grid.obstacles, agents, risk, risk_weight, reservations,
                        robustness, time_limit, suboptimality=suboptimality)


def encode_instance(grid: GridSpec, agents, risk=None, risk_weight=0.0, reservations=(), robustness=0,
                    time_limit=0.0, suboptimality=1.0) -> str:
    out = [f'grid {grid.dimx} {grid.dimy}']
    obs = sorted(grid.obstacles)
    out.append(f'obstacles {len(obs)} ' + ' '.join(f'{x} {y}' for x, y in obs))
    out.append(f'agents {len(agents)} ' + ' '.join(f'{s[0]} {s[1]} {g[0]} {g[1]}' for s, g in agents))
    if risk is not None and risk.size:
        r = np.asarray(risk, dtype=np.float32)
        # 9 significant digits round-trip float32 exactly.
        out.append(f'risk {r.shape[0]} ' + ' '.join('%.9g' % v for v in r.ravel()))
    out.append(f'lambda {risk_weight!r}')
    if reservations:
        parts = [f'reservations {len(reservations)}']
        for p in reservations:
            parts.append(f'{len(p)} ' + ' '.join(f'{c[0]} {c[1]}' for c in p))
        out.append(' '.join(parts))
    out.append(f'robust {int(robustness)}')
    out.append(f'time_limit {float(time_limit)!r}')
    out.append(f'subopt {float(suboptimality)!r}')
    out.append('end')
    return '\n'.join(out) + '\n'


def decode_result(text: str) -> SolveResult:
    res = SolveResult(False, 'no_output')
    paths = {}
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == 'status':
            res.status = tok[1]
            res.success = tok[1] == 'success'
        elif tok[0] == 'stats':
            res.objective = float(tok[1])
            res.objective_scaled = int(tok[2])
            res.sum_of_steps = int(tok[3])
            res.makespan = int(tok[4])
            res.total_risk = float(tok[5])
            res.high_level_expanded = int(tok[6])
            res.low_level_expanded = int(tok[7])
            res.runtime_s = float(tok[8])
        elif tok[0] == 'path':
            n = int(tok[2])
            vals = list(map(int, tok[3:3 + 2 * n]))
            paths[int(tok[1])] = [(vals[2 * i], vals[2 * i + 1]) for i in range(n)]
    res.paths = [paths[i] for i in sorted(paths)]
    return res


class CliSolver(SolverBackend):
    name = 'cli'

    def __init__(self, executable: str):
        self.executable = executable

    def solve(self, grid, agents, risk=None, risk_weight=0.0, reservations=(), robustness=0, time_limit=0.0,
              suboptimality=1.0):
        text = encode_instance(grid, agents, risk, risk_weight, reservations, robustness, time_limit, suboptimality)
        timeout = None if time_limit <= 0 else time_limit + 5.0
        proc = subprocess.run([self.executable], input=text, capture_output=True, text=True, timeout=timeout)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f'risk_cbs_cli failed ({proc.returncode}): {proc.stdout} {proc.stderr}')
        return decode_result(proc.stdout)


def find_cli() -> Optional[str]:
    """Locate risk_cbs_cli: $REMROC_RISK_CBS_CLI, the colcon install/build trees, or PATH."""
    env = os.environ.get('REMROC_RISK_CBS_CLI')
    if env and Path(env).is_file():
        return env
    here = Path(__file__).resolve()
    for root in [here.parents[i] for i in range(min(6, len(here.parents)))]:
        for rel in ('install/remroc_cbs_library/lib/remroc_cbs_library/risk_cbs_cli',
                    'build/remroc_cbs_library/risk_cbs_cli', 'build/risk_cbs_cli',
                    'build/risk_cbs_cli.exe'):
            cand = root / rel
            if cand.is_file():
                return str(cand)
    return shutil.which('risk_cbs_cli')


def make_solver(prefer: str = 'cli') -> SolverBackend:
    if prefer == 'cli':
        exe = find_cli()
        if exe:
            return CliSolver(exe)
    return PythonSolver()
