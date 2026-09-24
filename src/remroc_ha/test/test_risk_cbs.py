# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Risk-aware CBS: optimality properties and C++/Python cross-validation."""
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from remroc_ha.grid import GridSpec
from remroc_ha.risk_cbs import solve, validate_joint_plan
from remroc_ha.solver import CliSolver, PythonSolver, find_cli

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def cli(tmp_path_factory):
    exe = find_cli()
    if exe is None:
        cxx = shutil.which('g++') or shutil.which('clang++')
        if cxx is None:
            pytest.skip('no C++ compiler and no risk_cbs_cli found')
        exe = str(tmp_path_factory.mktemp('cli') / 'risk_cbs_cli')
        subprocess.run([cxx, '-std=c++17', '-O2', '-I', str(ROOT / 'src/remroc_cbs_library/include'),
                        str(ROOT / 'src/remroc_cbs_library/src/risk_cbs_cli.cpp'), '-o', exe], check=True)
    return CliSolver(exe)


def random_case(seed):
    rng = random.Random(seed)
    W, H = rng.randint(5, 9), rng.randint(4, 7)
    obs = frozenset((rng.randrange(W), rng.randrange(H)) for _ in range(W * H // 6))
    g = GridSpec(W, H, obs)
    free = g.free_cells()
    rng.shuffle(free)
    n = rng.randint(2, 4)
    if len(free) < 2 * n + 2:
        return None
    agents = [(free[i], free[n + i]) for i in range(n)]
    T = rng.randint(1, 8)
    rs = np.random.RandomState(seed)
    risk = np.where(rs.rand(T, H, W) < 0.2, rs.rand(T, H, W), 0).astype(np.float32)
    lam = rng.choice([0.0, 0.5, 2.0, 7.3])
    k = rng.choice([0, 1])
    resv = []
    if rng.random() < 0.3:
        p = g.shortest_path(free[2 * n], free[2 * n + 1])
        if p:
            resv = [p]
    return g, agents, risk, lam, k, resv


@pytest.mark.parametrize('seed', range(60))
def test_cpp_and_python_agree_on_optimal_objective(cli, seed):
    case = random_case(seed)
    if case is None:
        pytest.skip('degenerate')
    g, agents, risk, lam, k, resv = case
    a = cli.solve(g, agents, risk, lam, resv, k, time_limit=3.0)
    b = PythonSolver().solve(g, agents, risk, lam, resv, k, time_limit=3.0)
    if 'timeout' in (a.status, b.status):
        pytest.skip('timeout')
    assert a.success == b.success
    if a.success:
        assert a.objective_scaled == b.objective_scaled
        for r in (a, b):
            assert validate_joint_plan(g.dimx, g.dimy, g.obstacles, agents, r.paths, resv, k) is None


def test_risk_optimality_tradeoff_is_monotone():
    """Scalarisation: as lambda grows, the plan's expected human encounters (total risk)
    never increase and its sum of costs never decreases (docs/THEORY.md, Prop. 3)."""
    rng = np.random.RandomState(3)
    g = GridSpec(8, 6, frozenset({(3, 1), (3, 2), (3, 4), (5, 3)}))
    agents = [((0, 0), (7, 5)), ((7, 0), (0, 5)), ((0, 3), (7, 2))]
    risk = (rng.rand(6, 6, 8) < 0.25) * rng.rand(6, 6, 8)
    risk = risk.astype(np.float32)
    prev = None
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        r = solve(g.dimx, g.dimy, g.obstacles, agents, risk, lam, robustness=1)
        assert r.success
        if prev is not None:
            assert r.total_risk <= prev.total_risk + 1e-6
            assert r.sum_of_steps >= prev.sum_of_steps
            # bound: SoC(lambda) - SoC* <= lambda * (R(pi*) - R(pi_lambda))
        prev = r
    base = solve(g.dimx, g.dimy, g.obstacles, agents, risk, 0.0, robustness=1)
    for lam in [0.5, 2.0, 8.0]:
        r = solve(g.dimx, g.dimy, g.obstacles, agents, risk, lam, robustness=1)
        assert r.sum_of_steps - base.sum_of_steps <= lam * (base.total_risk - r.total_risk) + 1e-2


def test_zero_lambda_equals_classical_cbs_cost():
    g = GridSpec(5, 2, frozenset({(0, 1), (1, 1), (3, 1), (4, 1)}))
    r = solve(g.dimx, g.dimy, g.obstacles, [((0, 0), (4, 0)), ((4, 0), (0, 0))])
    assert r.success and r.sum_of_steps == 11


@pytest.mark.parametrize('seed', range(30))
def test_focal_search_respects_suboptimality_bound(cli, seed):
    case = random_case(1000 + seed)
    if case is None:
        pytest.skip('degenerate')
    g, agents, risk, lam, k, resv = case
    opt = cli.solve(g, agents, risk, lam, resv, k, time_limit=3.0)
    if not opt.success:
        pytest.skip(opt.status)
    for w in (1.1, 1.5):
        for solver in (cli, PythonSolver()):
            r = solver.solve(g, agents, risk, lam, resv, k, time_limit=3.0, suboptimality=w)
            assert r.success
            assert opt.objective_scaled <= r.objective_scaled <= w * opt.objective_scaled + 1e-6
            assert validate_joint_plan(g.dimx, g.dimy, g.obstacles, agents, r.paths, resv, k) is None
