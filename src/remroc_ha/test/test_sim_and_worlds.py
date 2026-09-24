# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest

from remroc_ha.grid import GridSpec
from remroc_ha.humans import load_humans_from_sdf, load_humans_json
from remroc_ha.metrics import compute_metrics
from remroc_ha.occupancy import OccupancyMap
from remroc_ha.sim.engine import load_scenario, run_episode

REMROC = Path(__file__).resolve().parents[2] / 'remroc'


@pytest.mark.parametrize('world', ['narrow_corridors', 'depot'])
def test_generated_world_is_consistent(world):
    g = GridSpec.from_mapf_yaml(REMROC / 'params' / f'mapf_{world}.yaml', world)
    occ = OccupancyMap.from_yaml(REMROC / 'worlds' / 'maps' / f'{world}.yaml')
    # every free MAPF cell centre is free space in the occupancy map
    for c in g.free_cells():
        assert occ.is_free(*g.centroid(c)), c
    for n in (5, 20):
        js = load_humans_json(REMROC / 'worlds' / 'humans' / f'{world}_{n}_1.json')
        sdf = load_humans_from_sdf(REMROC / 'worlds' / 'sdfs' / f'{world}_{n}_1.sdf')
        assert len(js) == len(sdf) == n
        ts = np.linspace(0, 150, 61)
        for a, b in zip(js, sdf):
            assert np.abs(a.positions(ts) - b.positions(ts)).max() < 0.01
            # humans stay in free space (with a small tolerance for the walking footprint)
            pts = a.positions(ts)
            assert occ.is_free(pts[:, 0], pts[:, 1]).mean() > 0.97


def test_layout_walls_match_mapf_grid():
    from remroc_world_generation.layouts import get_layout
    lay = get_layout('narrow_corridors')
    g = lay.grid()
    img = lay.occupancy_image()
    cp = lay.cell_px
    for y in range(g.dimy):
        for x in range(g.dimx):
            block = img[y * cp:(y + 1) * cp, x * cp:(x + 1) * cp]
            if (x, y) in g.obstacles:
                assert (block == 0).mean() > 0.3
            else:
                assert (block == 0).mean() == 0.0


@pytest.mark.parametrize('coordinator', ['ha_cbs', 'adg_cbs'])
def test_human_aware_episode_succeeds_without_collisions(coordinator):
    scn = load_scenario(REMROC, 'narrow_corridors', 5, 0)
    log = run_episode(scn, coordinator, seed=3, time_limit=300)
    m = compute_metrics(log)
    assert m['success'] == 1.0
    assert m['robot_collision'] == 0.0
    assert m['min_robot_robot_dist'] > 0.6
