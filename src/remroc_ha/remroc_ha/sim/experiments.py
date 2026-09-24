# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Run the simulated experiment grid: worlds x number of humans x samples x coordinators.

Example (from the repository root, after ``scripts/build_solver.sh``)::

    python -m remroc_ha.sim.experiments --remroc-dir src/remroc --out results/sim \
        --worlds narrow_corridors depot --humans 0 5 10 20 --samples 5 \
        --coordinators mapf pbc ha_cbs ha_cbs_mod adg_cbs ha_cbs_norepair --jobs 10

Writes ``episodes.csv`` (one row per episode) and ``summary.csv`` / ``summary.md``
(aggregates), optionally the full trajectory logs (``--save-logs``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..metrics import compute_metrics
from .engine import VARIANTS, load_scenario, run_episode
from .robot import RobotParams

ALL_COORDINATORS = ['mapf', 'pbc'] + list(VARIANTS.keys() - {'mapf', 'pbc'})


def _run(job):
    remroc_dir, world, n, s, coord, time_limit, save_dir, robot_kw, param_kw, tag = job
    t0 = time.time()
    scn = load_scenario(remroc_dir, world, n, s)
    log = run_episode(scn, coord, seed=1000 + s, time_limit=time_limit,
                      robot_params=replace(RobotParams(), **robot_kw),
                      param_overrides=param_kw if coord not in ('mapf', 'pbc') else None)
    m = compute_metrics(log)
    if save_dir:
        p = Path(save_dir) / f'{world}_{n}_{s}_{coord}.json.gz'
        with gzip.open(p, 'wt') as f:
            json.dump(log, f)
    row = {'world': world, 'humans': n, 'sample': s, 'coordinator': coord + tag, 'wall_time': time.time() - t0}
    row.update(m)
    for k, v in log['coordinator_stats'].items():
        if k not in row and isinstance(v, (int, float)):
            row['stat_' + k] = v
    return row


def summarize(rows, keys=('world', 'humans', 'coordinator')):
    metrics = ['success', 'makespan', 'sum_of_costs', 'min_human_robot_dist', 'time_near_humans',
               'human_encounters', 'replans', 'stuck_time', 'deadlock_time', 'robot_collision']
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) if not isinstance(x, int) else f'{x:04d}' for x in kv[0])):
        row = dict(zip(keys, key))
        row['episodes'] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) not in (None, '') and not math.isnan(float(r[m]))]
            row[m] = sum(vals) / len(vals) if vals else math.nan
        out.append(row)
    return out, ['episodes'] + metrics


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path, summary, metric_cols, keys=('world', 'humans', 'coordinator')):
    fmt = {'success': '{:.2f}', 'robot_collision': '{:.2f}'}
    lines = ['| ' + ' | '.join(list(keys) + metric_cols) + ' |', '|' + '---|' * (len(keys) + len(metric_cols))]
    for r in summary:
        cells = [str(r[k]) for k in keys]
        for m in metric_cols:
            v = r[m]
            cells.append('-' if isinstance(v, float) and math.isnan(v) else
                         (fmt.get(m, '{:.1f}').format(v) if isinstance(v, float) else str(v)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--remroc-dir', default='src/remroc')
    ap.add_argument('--out', default='results/sim')
    ap.add_argument('--worlds', nargs='+', default=['narrow_corridors', 'depot'])
    ap.add_argument('--humans', nargs='+', type=int, default=[0, 5, 10, 20])
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--coordinators', nargs='+', default=ALL_COORDINATORS)
    ap.add_argument('--time-limit', type=float, default=400.0)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--save-logs', action='store_true')
    ap.add_argument('--stall-rate', type=float, default=None, help='robot stalls per second (default 1/60)')
    ap.add_argument('--stall-duration', type=float, nargs=2, default=None)
    ap.add_argument('--no-robot-avoidance', action='store_true',
                    help='local layer ignores other robots: safety must come from coordination')
    ap.add_argument('--param', action='append', default=[], metavar='KEY=VALUE',
                    help='override a CoordinatorParams field of the human-aware variants')
    ap.add_argument('--risk-weights', type=float, nargs='+', default=None,
                    help='sweep lambda (human-aware variants get the suffix _l<lambda>)')
    args = ap.parse_args(argv)
    robot_kw = {}
    if args.stall_rate is not None:
        robot_kw['stall_rate'] = args.stall_rate
    if args.stall_duration is not None:
        robot_kw['stall_duration'] = tuple(args.stall_duration)
    if args.no_robot_avoidance:
        robot_kw['avoid_robots'] = False
    param_kw = {}
    for kv in args.param:
        k, v = kv.split('=', 1)
        param_kw[k] = {'true': True, 'false': False}.get(v.lower(), None)
        if param_kw[k] is None:
            param_kw[k] = float(v) if '.' in v or 'e' in v else int(v)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_dir = None
    if args.save_logs:
        save_dir = out / 'logs'
        save_dir.mkdir(exist_ok=True)
    sweeps = [(dict(param_kw), '')]
    if args.risk_weights:
        sweeps = [(dict(param_kw, risk_weight=lam), f'_l{lam:g}') for lam in args.risk_weights]
    jobs = []
    for pk, tag in sweeps:
        for c in args.coordinators:
            if tag and c in ('mapf', 'pbc') and pk is not sweeps[0][0]:
                continue   # baselines do not depend on lambda: run them once
            jobs += [(args.remroc_dir, w, n, s, c, args.time_limit, str(save_dir) if save_dir else None,
                      robot_kw, pk, tag if c not in ('mapf', 'pbc') else '')
                     for w in args.worlds for n in args.humans for s in range(args.samples)]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"[{i:4d}/{len(jobs)}] {r['world']:16s} n={r['humans']:2d} s={r['sample']} {r['coordinator']:16s} "
                  f"success={r['success']:.0f} makespan={r['makespan']:.1f} ({r['wall_time']:.1f}s)", flush=True)
    rows.sort(key=lambda r: (r['world'], r['humans'], r['sample'], r['coordinator']))
    write_csv(out / 'episodes.csv', rows)
    summary, cols = summarize(rows)
    write_csv(out / 'summary.csv', summary)
    write_markdown(out / 'summary.md', summary, cols)
    print(f'{len(rows)} episodes in {time.time() - t0:.0f}s -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
