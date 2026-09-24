# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Batch generation of REMROC worlds populated with 0, 5, 10, 20, ... humans.

For every (map, number of humans, sample) this writes, into the ``remroc`` package:

* ``worlds/sdfs/<map>_<N>_<sample>.sdf``: the Gazebo world with scripted actors
  (naming convention of ``ignition.launch.py``),
* ``worlds/humans/<map>_<N>_<sample>.json``: the same trajectories as JSON (used by
  the grid-level simulator and, optionally, by the human pose publisher),

and, per map, ``worlds/humans/<map>_mod.npz``: a map of dynamics fitted on separate
*training* samples (never on evaluation worlds) for the MoD-based predictor.

Procedural layouts (``narrow_corridors``) also get their occupancy map, nav2 map yaml
and ``params/mapf_<map>.yaml``. For existing maps (``depot``) the base world
``worlds/sdfs/<map>_0_0.sdf`` and the map yaml are read from the package.

Example::

    ros2 run remroc_world_generation generate_worlds --map narrow_corridors depot \
        --humans 0 5 10 20 --samples 5 --remroc-dir src/remroc
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from remroc_ha.humans import HumanScript, all_trajectories, save_humans_json
from remroc_ha.occupancy import OccupancyMap
from remroc_ha.prediction import MapOfDynamics

from .human_trajectories import HumanTrajectoryGenerator
from .layouts import LAYOUTS, get_layout

WALK_ANIMATION = 'https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae'

# Per-map generation settings: minimum start-goal distance(s) drawn per human.
MAP_SETTINGS = {
    # half of the humans cross between the rooms (through a corridor), half stay in a room
    'narrow_corridors': {'min_distance': (9.0, 3.0), 'clearance': 0.35},
    'depot': {'min_distance': (8.0,), 'clearance': 0.35},
    'depot_mod': {'min_distance': (8.0,), 'clearance': 0.35},
    'simple': {'min_distance': (8.0,), 'clearance': 0.35},
}


def actor_xml(h: HumanScript) -> str:
    wps = ''.join(f'<waypoint><time>{t:.3f}</time><pose>{x:.3f} {y:.3f} 1 0 0 {yaw:.3f}</pose></waypoint>'
                  for t, x, y, yaw in h.wp[:, :4])
    return f'''    <actor name="{h.name}">
      <skin>
        <filename>{WALK_ANIMATION}</filename>
        <scale>1.0</scale>
      </skin>
      <animation name="walk">
        <filename>{WALK_ANIMATION}</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <delay_start>{h.delay_start:.3f}</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walk">
        {wps}
        </trajectory>
      </script>
    </actor>
'''


def insert_actors(base_sdf: str, humans) -> str:
    idx = base_sdf.rfind('</world>')
    if idx < 0:
        raise ValueError('base SDF has no </world>')
    # drop actors already present in the base world
    body = re.sub(r'\s*<actor .*?</actor>', '', base_sdf[:idx], flags=re.S)
    return body.rstrip() + '\n\n' + ''.join(actor_xml(h) for h in humans) + '\n  ' + base_sdf[idx:]


def generate_humans(occ: OccupancyMap, map_name: str, n: int, seed: int):
    settings = MAP_SETTINGS.get(map_name, {'min_distance': (6.0,), 'clearance': 0.35})
    gen = HumanTrajectoryGenerator(occ, clearance=settings['clearance'])
    rng = np.random.default_rng(seed)
    humans = []
    for i in range(n):
        dists = settings['min_distance']
        traj = gen.trajectory(rng, dists[i % len(dists)])
        humans.append(HumanScript(f'actor_walking_{i}', traj))
    return humans


def world_seed(map_name: str, n: int, sample: int, base_seed: int) -> int:
    return (base_seed * 1000003 + sum(map(ord, map_name)) * 7919 + n * 104729 + sample) % (2 ** 31)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', nargs='+', default=['narrow_corridors', 'depot'])
    ap.add_argument('--humans', nargs='+', type=int, default=[0, 5, 10, 20])
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--remroc-dir', default=None,
                    help='path to the remroc package (source or install share dir); default: ament index')
    ap.add_argument('--train-samples', type=int, default=4, help='training worlds per map for the map of dynamics')
    ap.add_argument('--train-humans', type=int, default=10)
    args = ap.parse_args(argv)

    if args.remroc_dir:
        remroc = Path(args.remroc_dir)
    else:
        from ament_index_python.packages import get_package_share_directory
        remroc = Path(get_package_share_directory('remroc'))
    maps_dir, params_dir = remroc / 'worlds' / 'maps', remroc / 'params'
    sdf_dir, humans_dir = remroc / 'worlds' / 'sdfs', remroc / 'worlds' / 'humans'
    humans_dir.mkdir(parents=True, exist_ok=True)

    for map_name in args.map:
        if map_name in LAYOUTS:
            out = get_layout(map_name).write(maps_dir, params_dir, sdf_dir)
            print(f'[{map_name}] layout written: {out["map"]}, {out["mapf"]}, {out["base_sdf"]}')
        base_path = sdf_dir / f'{map_name}_0_0.sdf'
        base_sdf = base_path.read_text()
        if '<actor' in base_sdf:
            raise SystemExit(f'{base_path} must be an empty base world')
        occ = OccupancyMap.from_yaml(maps_dir / f'{map_name}.yaml')

        for n in args.humans:
            for s in range(args.samples):
                if n == 0 and s == 0:
                    humans = []
                else:
                    humans = generate_humans(occ, map_name, n, world_seed(map_name, n, s, args.seed))
                meta = {'map': map_name, 'number_of_humans': n, 'sample': s, 'seed': args.seed}
                save_humans_json(humans_dir / f'{map_name}_{n}_{s}.json', humans, meta)
                if not (n == 0 and s == 0):
                    (sdf_dir / f'{map_name}_{n}_{s}.sdf').write_text(insert_actors(base_sdf, humans))
            print(f'[{map_name}] {args.samples} worlds with {n} humans')

        # map of dynamics from independent training worlds (seed offset keeps them disjoint)
        train = []
        for s in range(args.train_samples):
            hs = generate_humans(occ, map_name, args.train_humans,
                                 world_seed(map_name, args.train_humans, 10_000 + s, args.seed))
            train += all_trajectories(hs, horizon=None, dt=0.25)
        mod = MapOfDynamics(occ.bounds(), resolution=0.5).fit(train)
        mod.save(humans_dir / f'{map_name}_mod.npz')
        print(f'[{map_name}] map of dynamics fitted on {len(train)} training trajectories')
    return 0


if __name__ == '__main__':
    sys.exit(main())
