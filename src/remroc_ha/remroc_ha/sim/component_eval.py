# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Component evaluations (reproducible numbers quoted in docs/EXPERIMENTS.md).

    python -m remroc_ha.sim.component_eval prediction --out results/prediction
    python -m remroc_ha.sim.component_eval planner    --out results/planner

* prediction: Brier score of the occupancy forecast per MAPF step ahead, for the
  constant-velocity (CV) and map-of-dynamics (MoD) predictors, with and without
  blending into the static prior, against the all-zero forecast.
* planner: runtime / success / cost of risk-aware CBS on the experiment instances with
  realistic risk maps, optimal (w = 1) vs. focal search (w > 1).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ..prediction import HumanRiskPredictor, MapOfDynamics, _cell_coordinates
from ..occupancy import OccupancyMap
from ..solver import make_solver
from .engine import load_scenario


def truth_occupancy(grid, pts, inflation):
    out = np.zeros((grid.dimy, grid.dimx))
    if not len(pts):
        return out
    u, v = _cell_coordinates(grid, pts[:, 0], pts[:, 1])
    a = inflation / grid.cell_size
    for x in range(grid.dimx):
        for yy in range(grid.dimy):
            if ((np.abs(u - x) <= 0.5 + a) & (np.abs(v - yy) <= 0.5 + a)).any():
                out[grid.dimy - 1 - yy, x] = 1.0
    return out


def eval_prediction(remroc, out: Path, steps=10, step_duration=3.0, trials=20):
    rows = []
    for world in ('narrow_corridors', 'depot'):
        for n in (10, 20):
            for sample in range(5):
                scn = load_scenario(remroc, world, n, sample)
                g, free = scn.grid, scn.grid.free_mask()
                mod = MapOfDynamics.load(scn.mod_path)
                occ = OccupancyMap.from_yaml(scn.map_yaml)
                preds = {
                    'cv': HumanRiskPredictor(g, step_duration, steps, method='cv', use_prior=False),
                    'mod': HumanRiskPredictor(g, step_duration, steps, method='mod', mod=mod, walkable=occ.is_free,
                                              use_prior=False, seed=sample),
                    'cv+prior': HumanRiskPredictor(g, step_duration, steps, method='cv', mod=mod),
                    'mod+prior': HumanRiskPredictor(g, step_duration, steps, method='mod', mod=mod,
                                                    walkable=occ.is_free, seed=sample),
                }
                rng = np.random.default_rng(sample)
                acc = {k: np.zeros(steps) for k in list(preds) + ['zero']}
                for _ in range(trials):
                    t0 = rng.uniform(20, 300)
                    maps = {}
                    for name, p in preds.items():
                        p.tracker = type(p.tracker)()
                        for tt in np.arange(t0 - 2.0, t0 + 1e-9, 0.5):
                            p.observe(tt, [(h.name, *h.position(tt)) for h in scn.humans])
                        maps[name] = p.risk_map(t0)
                    for k in range(steps):
                        tr = truth_occupancy(g, np.array([h.position(t0 + k * step_duration) for h in scn.humans]),
                                             0.3)[free]
                        acc['zero'][k] += np.mean(tr ** 2)
                        for name in preds:
                            acc[name][k] += np.mean((maps[name][k][free] - tr) ** 2)
                for name, v in acc.items():
                    for k in range(steps):
                        rows.append({'world': world, 'humans': n, 'sample': sample, 'predictor': name,
                                     'step': k, 'seconds_ahead': k * step_duration, 'brier': v[k] / trials})
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'brier.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # summary table: mean Brier per predictor and step, per world
    lines = []
    for world in ('narrow_corridors', 'depot'):
        lines += [f'\n**{world}** (mean over 10/20 humans x 5 samples x {trials} start times; lower is better)\n',
                  '| predictor | ' + ' | '.join(f'+{k * step_duration:.0f}s' for k in range(steps)) + ' |',
                  '|---|' + '---|' * steps]
        for name in ('zero', 'cv', 'mod', 'cv+prior', 'mod+prior'):
            vals = [np.mean([r['brier'] for r in rows if r['world'] == world and r['predictor'] == name
                             and r['step'] == k]) for k in range(steps)]
            lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    (out / 'brier.md').write_text('\n'.join(lines) + '\n')
    print((out / 'brier.md').read_text())


def eval_planner(remroc, out: Path, weights=(1.0, 1.05, 1.1, 1.2, 1.5), time_limit=10.0):
    solver = make_solver()
    rows = []
    for world in ('narrow_corridors', 'depot'):
        for n in (0, 10, 20):
            for sample in range(5):
                scn = load_scenario(remroc, world, n, sample)
                g = scn.grid
                p = HumanRiskPredictor(g, 3.0, 20, method='cv', mod=MapOfDynamics.load(scn.mod_path))
                t0 = 10.0 + 30.0 * sample
                for tt in np.arange(t0 - 2.0, t0 + 1e-9, 0.5):
                    p.observe(tt, [(h.name, *h.position(tt)) for h in scn.humans])
                risk = p.risk_map(t0)
                starts = g.assign_unique_cells([(r, s[:2]) for r, s, _ in scn.robots])
                goals = g.assign_unique_cells([(r, gl) for r, _, gl in scn.robots])
                agents = [(starts[r], goals[r]) for r in scn.names]
                for w in weights:
                    res = solver.solve(g, agents, risk, 4.0, (), 1, time_limit, w)
                    rows.append({'world': world, 'humans': n, 'sample': sample, 'w': w, 'status': res.status,
                                 'runtime_s': res.runtime_s, 'objective': res.objective if res.success else '',
                                 'hl_expanded': res.high_level_expanded})
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'planner.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = ['| world | humans | w | solved | median runtime [s] | mean objective gap to best found |', '|---|---|---|---|---|---|']
    for world in ('narrow_corridors', 'depot'):
        for n in (0, 10, 20):
            best = {}
            for r in rows:
                if r['world'] == world and r['humans'] == n and r['objective'] != '':
                    best[r['sample']] = min(best.get(r['sample'], 1e18), r['objective'])
            for w in weights:
                rs = [r for r in rows if r['world'] == world and r['humans'] == n and r['w'] == w]
                ok = [r for r in rs if r['status'] == 'success']
                gap = np.mean([r['objective'] / best[r['sample']] - 1 for r in ok]) if ok else float('nan')
                lines.append(f'| {world} | {n} | {w:g} | {len(ok)}/{len(rs)} | '
                             f'{np.median([r["runtime_s"] for r in rs]):.3f} | {100 * gap:.2f}% |')
    (out / 'planner.md').write_text('\n'.join(lines) + '\n')
    print((out / 'planner.md').read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['prediction', 'planner'])
    ap.add_argument('--remroc-dir', default='src/remroc')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or f'results/{args.kind}')
    if args.kind == 'prediction':
        eval_prediction(args.remroc_dir, out)
    else:
        eval_planner(args.remroc_dir, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
