# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Properties of the Action Dependency Graph execution layer.

The central claims (docs/THEORY.md) are tested empirically on random instances:
executing an ADG built from a 1-robust conflict-free plan under *arbitrary* delays
(i) never puts two robots into the same cell or swaps them, and (ii) always completes.
"""
import random

import pytest

from remroc_ha.adg import ActionDependencyGraph, CyclicADGError
from remroc_ha.grid import GridSpec
from remroc_ha.risk_cbs import solve, validate_joint_plan


def random_instance(rng, n_agents=4, W=7, H=5, density=0.15):
    obs = frozenset((rng.randrange(W), rng.randrange(H)) for _ in range(int(W * H * density)))
    g = GridSpec(W, H, obs)
    free = g.free_cells()
    rng.shuffle(free)
    if len(free) < 2 * n_agents:
        return None
    return g, [(free[i], free[n_agents + i]) for i in range(n_agents)]


def simulate_adg(adg: ActionDependencyGraph, rng, p_delay=0.6, max_ticks=10000):
    """Asynchronous execution: in every tick each robot whose next action is enabled
    completes it with probability 1 - p_delay (arbitrary, adversarial-ish delays).
    Checks the physical occupancy invariant at every tick. Returns #ticks."""
    robots = list(adg.actions)
    for tick in range(max_ticks):
        if adg.all_finished():
            return tick
        occupied = {}
        for r in robots:
            c = adg.current_cell(r)
            assert c not in occupied, f'two robots in {c}'
            occupied[c] = r
        order = robots[:]
        rng.shuffle(order)
        for r in order:
            pend = adg.pending(r)
            if not pend or not adg.enabled((r, pend[0].index)):
                continue
            if rng.random() < p_delay:
                continue
            a = pend[0]
            # the target must be physically free when the robot moves in (vertex safety)
            others = {adg.current_cell(o) for o in robots if o != r}
            assert a.dst not in others, f'{r} enters occupied cell {a.dst}'
            adg.mark_done(r, a.index)
    raise AssertionError('execution did not finish: deadlock')


@pytest.mark.parametrize('seed', range(40))
def test_adg_execution_is_safe_and_live_under_random_delays(seed):
    rng = random.Random(seed)
    inst = random_instance(rng)
    if inst is None:
        pytest.skip('degenerate instance')
    g, agents = inst
    res = solve(g.dimx, g.dimy, g.obstacles, agents, robustness=1, time_limit=5.0)
    if not res.success:
        pytest.skip(res.status)
    assert validate_joint_plan(g.dimx, g.dimy, g.obstacles, agents, res.paths, robustness=1) is None
    adg = ActionDependencyGraph({f'r{i}': p for i, p in enumerate(res.paths)})
    for p_delay in (0.0, 0.5, 0.9):
        adg.done = {r: 0 for r in adg.done}
        simulate_adg(adg, rng, p_delay)


def test_rotation_plan_gives_cyclic_adg():
    # 4 agents rotating in a closed 2x2 block is a valid k=0 plan but cannot be executed
    # asynchronously: the ADG detects the cycle.
    paths = {'a': [(0, 0), (1, 0)], 'b': [(1, 0), (1, 1)], 'c': [(1, 1), (0, 1)], 'd': [(0, 1), (0, 0)]}
    with pytest.raises(CyclicADGError):
        ActionDependencyGraph(paths)


def test_following_dependency():
    # b follows a through a corridor: b's move into a's start must wait for a's move out
    adg = ActionDependencyGraph({'a': [(1, 0), (2, 0), (3, 0)], 'b': [(0, 0), (0, 0), (1, 0), (2, 0)]})
    assert adg.dispatchable('b') == []                 # b's first move enters (1,0) -> depends on a leaving it
    assert [a.dst for a in adg.dispatchable('a')] == [(2, 0), (3, 0)]
    adg.mark_done('a', 0)
    assert [a.dst for a in adg.dispatchable('b')] == [(1, 0)]   # (2,0) still needs a's second move
    adg.mark_done('a', 1)
    assert [a.dst for a in adg.dispatchable('b')] == [(1, 0), (2, 0)]


@pytest.mark.parametrize('seed', range(25))
def test_earliest_start_schedule_is_conflict_free(seed):
    rng = random.Random(100 + seed)
    inst = random_instance(rng, n_agents=5, W=8, H=6)
    if inst is None:
        pytest.skip('degenerate instance')
    g, agents = inst
    res = solve(g.dimx, g.dimy, g.obstacles, agents, robustness=1, time_limit=5.0)
    if not res.success:
        pytest.skip(res.status)
    names = [f'r{i}' for i in range(len(agents))]
    adg = ActionDependencyGraph(dict(zip(names, res.paths)))
    # random partial progress that respects the dependencies
    for _ in range(rng.randrange(0, 15)):
        r = rng.choice(names)
        pend = adg.pending(r)
        if pend and adg.enabled((r, pend[0].index)):
            adg.mark_done(r, pend[0].index)
    subset = [n for n in names if rng.random() < 0.7] or names[:1]
    es = adg.earliest_start_schedule(subset)
    sub_agents = [(es[r][0], agents[names.index(r)][1]) for r in subset]
    assert validate_joint_plan(g.dimx, g.dimy, g.obstacles, sub_agents, [es[r] for r in subset],
                               robustness=1) is None
