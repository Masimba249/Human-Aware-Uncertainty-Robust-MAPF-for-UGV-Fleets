# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Figures and tables from experiment results (simulator or REMROC/Gazebo).

    python -m remroc_ha.analysis sim    results/sim            # figures for the main grid
    python -m remroc_ha.analysis lambda results/sim_lambda     # risk / efficiency trade-off
    python -m remroc_ha.analysis ros    results                # aggregate Gazebo runs written by
                                                               # metrics_recorder into episodes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .sim.experiments import summarize, write_csv, write_markdown  # noqa: E402

# Categorical slots of the reference palette, in fixed order; a coordinator keeps its
# colour in every figure.
COLORS = {'mapf': '#2a78d6', 'pbc': '#eb6834', 'ha_cbs': '#1baf7a', 'ha_cbs_mod': '#eda100',
          'adg_cbs': '#e87ba4', 'ha_cbs_norepair': '#008300'}
LABELS = {'mapf': 'MAPF (baseline)', 'pbc': 'PBC (baseline)', 'ha_cbs': 'HA-CBS + ADG (CV)',
          'ha_cbs_mod': 'HA-CBS + ADG (MoD)', 'adg_cbs': 'CBS + ADG (no humans)',
          'ha_cbs_norepair': 'HA-CBS + ADG, no repair'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def read_csv(path):
    rows = list(csv.DictReader(Path(path).open()))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k != 'humans' else int(v)
            except (TypeError, ValueError):
                pass
    return rows


def plot_main(summary_rows, out: Path):
    metrics = [('success', 'success rate'), ('makespan', 'makespan [s]'), ('human_encounters', 'close encounters (<0.5 m)'),
               ('time_near_humans', 'time within 1 m of a human [s]'), ('stuck_time', 'stuck time [s]'),
               ('deadlock_time', 'fleet deadlock time [s]')]
    worlds = sorted({r['world'] for r in summary_rows})
    coords = [c for c in COLORS if any(r['coordinator'] == c for r in summary_rows)]
    fig, axes = plt.subplots(len(worlds), len(metrics), figsize=(3.1 * len(metrics), 2.8 * len(worlds)), squeeze=False)
    for i, w in enumerate(worlds):
        for j, (m, label) in enumerate(metrics):
            ax = axes[i][j]
            _style(ax)
            for c in coords:
                pts = sorted((r['humans'], r[m]) for r in summary_rows if r['world'] == w and r['coordinator'] == c)
                pts = [(x, y) for x, y in pts if not (isinstance(y, float) and math.isnan(y))]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[c], lw=2, marker='o', ms=4,
                            label=LABELS[c])
            ax.set_title(label, fontsize=9, color=INK)
            ax.set_xticks([0, 5, 10, 20])
            if i == len(worlds) - 1:
                ax.set_xlabel('number of humans', fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(w, fontsize=10, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(coords), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_lambda(summary_rows, out: Path):
    worlds = sorted({r['world'] for r in summary_rows})
    humans = sorted({r['humans'] for r in summary_rows})
    fig, axes = plt.subplots(len(worlds), len(humans), figsize=(4.2 * len(humans), 3.4 * len(worlds)), squeeze=False)
    for i, w in enumerate(worlds):
        for j, n in enumerate(humans):
            ax = axes[i][j]
            _style(ax)
            rows = [r for r in summary_rows if r['world'] == w and r['humans'] == n]
            sweep = sorted(((float(r['coordinator'].split('_l')[-1]), r) for r in rows if '_l' in r['coordinator']),
                           key=lambda x: x[0])
            ax.plot([r['makespan'] for _, r in sweep], [r['human_encounters'] for _, r in sweep], color=COLORS['ha_cbs'],
                    lw=2, marker='o', ms=5, label='HA-CBS + ADG, varying $\\lambda$')
            for lam, r in sweep:
                ax.annotate(f'$\\lambda$={lam:g}', (r['makespan'], r['human_encounters']), textcoords='offset points',
                            xytext=(5, 4), fontsize=7, color=MUTED)
            for r in rows:
                if r['coordinator'] == 'mapf':
                    ax.plot([r['makespan']], [r['human_encounters']], color=COLORS['mapf'], marker='s', ms=8, ls='',
                            label=LABELS['mapf'])
            ax.set_title(f'{w}, {n} humans', fontsize=9, color=INK)
            ax.set_xlabel('makespan [s]', fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel('close encounters per episode', fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def collect_ros(results_dir: Path):
    """Rows from <results>/<coordinator>/<world>/<N>/<sample>_metrics.json (metrics_recorder).
    For coordinator_mapf the number of re-plans is taken from its own result file."""
    rows = []
    for f in sorted(Path(results_dir).glob('*/*/*/*_metrics.json')):
        coord, world, n = f.parts[-4], f.parts[-3], int(f.parts[-2])
        sample = int(f.stem.split('_')[0])
        m = json.loads(f.read_text())
        if coord == 'mapf' and math.isnan(m.get('replans', math.nan)):
            own = Path(results_dir) / 'mapf' / world / str(n) / f'{sample}.json'
            if own.exists():
                m['replans'] = max(0, len(json.loads(own.read_text()).get('coordinator_loop_time', [])) - 1)
        if coord == 'pbc' and math.isnan(m.get('replans', math.nan)):
            m['replans'] = 0.0
        rows.append({'world': world, 'humans': n, 'sample': sample, 'coordinator': coord, **m})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['sim', 'lambda', 'ros'])
    ap.add_argument('results')
    args = ap.parse_args(argv)
    res = Path(args.results)
    if args.kind == 'ros':
        rows = collect_ros(res)
        write_csv(res / 'episodes.csv', rows)
        summary, cols = summarize(rows)
        write_csv(res / 'summary.csv', summary)
        write_markdown(res / 'summary.md', summary, cols)
        plot_main(summary, res / 'main.png')
        print(f'{len(rows)} Gazebo episodes -> {res / "summary.md"}')
        return 0
    summary = read_csv(res / 'summary.csv')
    if args.kind == 'sim':
        plot_main(summary, res / 'main.png')
    else:
        plot_lambda(summary, res / 'tradeoff.png')
    print('figures written to', res)
    return 0


if __name__ == '__main__':
    sys.exit(main())
