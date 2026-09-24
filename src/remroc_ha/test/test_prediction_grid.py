# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np

from remroc_ha.grid import KNOWN_GRID_METADATA, GridSpec
from remroc_ha.prediction import (HumanRiskPredictor, HumanTracker, MapOfDynamics,
                                  RiskMapBuilder)


def depot_like_grid():
    return GridSpec(25, 12, frozenset({(0, 0), (12, 7)}), **{k: v for k, v in KNOWN_GRID_METADATA['depot'].items()
                                                            if k != 'origin'},
                    origin=tuple(KNOWN_GRID_METADATA['depot']['origin']))


def test_centroid_convention_matches_coordinator_mapf():
    g = depot_like_grid()
    # coordinator_mapf.py: key f'{x}, {-(y - (dimy-1))}' -> [res*(x*cs+xo+cs/2)+ox, res*(y*cs+yo+cs/2)+oy]
    res, cs, xo, yo, (ox, oy) = 0.04, 29, 4, 14, (-15.1, -7.74)
    for x, yloop in [(0, 0), (3, 5), (24, 11)]:
        ycbs = -(yloop - (g.dimy - 1))
        expect = (res * (x * cs + xo + cs / 2) + ox, res * (yloop * cs + yo + cs / 2) + oy)
        assert np.allclose(g.centroid((x, ycbs)), expect)
        assert g.cell_at(*expect) == (x, ycbs)


def test_tracker_estimates_velocity():
    tr = HumanTracker()
    for k in range(10):
        t = k * 0.2
        tr.update(t, [('h', 1.0 + 0.8 * t, 2.0 - 0.3 * t)])
    trk = tr.tracks(1.8)['h']
    assert np.allclose(trk.vel, [0.8, -0.3], atol=1e-6)


def test_gaussian_rasterisation_mass_and_location():
    g = GridSpec(10, 10, frozenset(), resolution=0.1, cell_size_px=10, origin=(0.0, 0.0))
    b = RiskMapBuilder(g, step_duration=1.0, horizon=3, inflation=0.0)
    cx, cy = g.centroid((4, 6))
    means = np.array([[[cx, cy], [cx + 1.0, cy], [cx + 2.0, cy]]])
    sig = np.array([[0.05, 0.05, 0.05]])
    risk = b.from_gaussians(means, sig)
    assert risk.shape == (4, 10, 10)
    assert risk[0, 6, 4] > 0.99 and risk[1, 6, 5] > 0.99 and risk[2, 6, 6] > 0.99
    assert risk[0].sum() < 1.01
    assert np.all(risk[3] == 0)          # no prior -> empty tail layer


def test_particles_and_gaussians_agree_for_certain_motion():
    g = GridSpec(10, 4, frozenset(), resolution=0.1, cell_size_px=10, origin=(0.0, 0.0))
    b = RiskMapBuilder(g, 1.0, 2, inflation=0.0)
    cx, cy = g.centroid((2, 1))
    parts = np.tile(np.array([cx, cy]), (1, 2, 50, 1)).astype(float)
    r = b.from_particles(parts)
    assert math.isclose(r[0, 1, 2], 1.0, abs_tol=1e-6)


def test_map_of_dynamics_learns_flow_direction():
    # humans always walk in +x along y = 1
    trajs = [np.column_stack([np.arange(0, 10, 0.5), 0.5 + 0.8 * np.arange(0, 10, 0.5), np.full(20, 1.0)])
             for _ in range(5)]
    mod = MapOfDynamics((0, 0, 10, 2), resolution=0.5).fit(trajs, blur=0)
    probs, speeds = mod.direction_distribution(np.array([4.0]), np.array([1.1]))
    assert np.argmax(probs[0]) == 0 and probs[0, 0] > 0.99
    assert abs(speeds[0, 0] - 0.8) < 1e-6


def test_risk_predictor_blends_towards_prior():
    g = GridSpec(6, 3, frozenset(), resolution=0.1, cell_size_px=10, origin=(0.0, 0.0))
    trajs = [np.column_stack([np.arange(0, 6, 0.5), 0.3 + 0.9 * np.arange(0, 6, 0.5), np.full(12, 1.5)])]
    mod = MapOfDynamics((0, 0, 6, 3), resolution=0.5).fit(trajs)
    p = HumanRiskPredictor(g, step_duration=2.0, horizon=15, method='cv', mod=mod, prior_blend_time=4.0)
    for k in range(5):
        p.observe(k * 0.25, [('h', 1.0 + 0.9 * k * 0.25, 1.5)])
    risk = p.risk_map(1.0)
    prior = risk[-1]
    assert prior.max() > 0
    # far layers converge to the prior, early layers follow the tracked human
    assert np.abs(risk[14] - prior).max() < 0.01
    assert risk[0, 1, 1] > 0.5
