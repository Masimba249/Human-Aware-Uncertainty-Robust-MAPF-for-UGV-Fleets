# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Procedural grid layouts: one ASCII description -> occupancy map, nav2 map yaml,
Gazebo SDF walls and the MAPF grid (mapf_<name>.yaml), all geometrically consistent.

Every character is one MAPF cell: ``#`` wall, anything else free. Wall blocks are
drawn inset by ``inset`` metres on every side that faces a free cell, so a corridor
that is one MAPF cell wide is physically ``cell_size + 2 * inset`` wide.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from remroc_ha.grid import GridSpec
from remroc_ha.occupancy import write_pgm

# Two rooms joined by two long single-lane corridors: robots crossing in opposite
# directions must agree on corridor usage and direction, and humans walking between
# the rooms regularly occupy the corridors.
NARROW_CORRIDORS = [
    '###################',
    '#.....#######.....#',
    '#.................#',
    '#.....#######.....#',
    '#.....#######.....#',
    '#.....#######.....#',
    '#.................#',
    '#.....#######.....#',
    '###################',
]

LAYOUTS = {'narrow_corridors': NARROW_CORRIDORS}


@dataclass
class Layout:
    name: str
    rows: List[str]
    cell_size: float = 1.2
    resolution: float = 0.05
    inset: float = 0.25
    wall_height: float = 1.5

    @property
    def dimx(self) -> int:
        return len(self.rows[0])

    @property
    def dimy(self) -> int:
        return len(self.rows)

    @property
    def cell_px(self) -> int:
        return int(round(self.cell_size / self.resolution))

    @property
    def origin(self) -> Tuple[float, float]:
        return (-self.dimx * self.cell_size / 2.0, -self.dimy * self.cell_size / 2.0)

    def is_wall(self, x: int, y: int) -> bool:
        if not (0 <= x < self.dimx and 0 <= y < self.dimy):
            return True
        return self.rows[y][x] == '#'

    def grid(self) -> GridSpec:
        obstacles = frozenset((x, y) for y in range(self.dimy) for x in range(self.dimx) if self.is_wall(x, y))
        return GridSpec(self.dimx, self.dimy, obstacles, self.resolution, self.cell_px, 0.0, 0.0, self.origin)

    def wall_boxes(self) -> List[Tuple[float, float, float, float]]:
        """Metric boxes (xmin, ymin, xmax, ymax) of the wall geometry."""
        g = self.grid()
        h = self.cell_size / 2.0
        boxes = []
        for y in range(self.dimy):
            for x in range(self.dimx):
                if not self.is_wall(x, y):
                    continue
                cx, cy = g.centroid((x, y))
                xmin, xmax, ymin, ymax = cx - h, cx + h, cy - h, cy + h
                # MAPF y grows downwards: neighbour (x, y-1) is above (+y in metres)
                if x > 0 and not self.is_wall(x - 1, y):
                    xmin += self.inset
                if x < self.dimx - 1 and not self.is_wall(x + 1, y):
                    xmax -= self.inset
                if y > 0 and not self.is_wall(x, y - 1):
                    ymax -= self.inset
                if y < self.dimy - 1 and not self.is_wall(x, y + 1):
                    ymin += self.inset
                boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def occupancy_image(self) -> np.ndarray:
        """PGM image (row 0 = top): 0 occupied, 254 free."""
        w, h = self.dimx * self.cell_px, self.dimy * self.cell_px
        img = np.full((h, w), 254, dtype=np.uint8)
        ox, oy = self.origin
        for (xmin, ymin, xmax, ymax) in self.wall_boxes():
            c0 = int(round((xmin - ox) / self.resolution))
            c1 = int(round((xmax - ox) / self.resolution))
            r1 = h - int(round((ymin - oy) / self.resolution))
            r0 = h - int(round((ymax - oy) / self.resolution))
            img[max(r0, 0):min(r1, h), max(c0, 0):min(c1, w)] = 0
        return img

    def map_yaml(self) -> dict:
        return {'image': f'{self.name}.pgm', 'mode': 'trinary', 'resolution': self.resolution,
                'origin': [float(self.origin[0]), float(self.origin[1]), 0.0], 'negate': 0,
                'occupied_thresh': 0.65, 'free_thresh': 0.25}

    def sdf_world(self) -> str:
        """World without humans; actors are inserted before </world> by sdf_tools."""
        links = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(self.wall_boxes()):
            sx, sy = xmax - xmin, ymax - ymin
            px, py = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            geom = f'<geometry><box><size>{sx:.4f} {sy:.4f} {self.wall_height}</size></box></geometry>'
            links.append(
                f'      <link name="wall_{i}">\n'
                f'        <pose>{px:.4f} {py:.4f} {self.wall_height / 2} 0 0 0</pose>\n'
                f'        <collision name="collision">{geom}</collision>\n'
                f'        <visual name="visual">{geom}<material><ambient>0.6 0.6 0.6 1</ambient>'
                f'<diffuse>0.7 0.7 0.7 1</diffuse></material></visual>\n'
                f'      </link>')
        walls = '\n'.join(links)
        return f'''<?xml version="1.0"?>
<!-- Generated by remroc_world_generation/layouts.py ({self.name}). SPDX-License-Identifier: Apache-2.0 -->
<sdf version="1.6">
  <world name="{self.name}">
    <scene>
      <grid>false</grid>
    </scene>
    <physics name="1ms" type="ignored">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="libignition-gazebo-physics-system.so" name="ignition::gazebo::systems::Physics"></plugin>
    <plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre</render_engine>
      <namespace>test</namespace>
    </plugin>
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"></plugin>
    <plugin filename="libignition-gazebo-user-commands-system.so" name="ignition::gazebo::systems::UserCommands"></plugin>
    <plugin filename="libignition-gazebo-scene-broadcaster-system.so" name="ignition::gazebo::systems::SceneBroadcaster"></plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>

    <model name="{self.name}_walls">
      <static>true</static>
{walls}
    </model>

  </world>
</sdf>
'''

    def write(self, maps_dir: Path, params_dir: Path, sdf_dir: Path) -> dict:
        maps_dir, params_dir, sdf_dir = Path(maps_dir), Path(params_dir), Path(sdf_dir)
        for d in (maps_dir, params_dir, sdf_dir):
            d.mkdir(parents=True, exist_ok=True)
        write_pgm(maps_dir / f'{self.name}.pgm', self.occupancy_image())
        (maps_dir / f'{self.name}.yaml').write_text(yaml.safe_dump(self.map_yaml(), sort_keys=False))
        (params_dir / f'mapf_{self.name}.yaml').write_text(
            '# Generated by remroc_world_generation/layouts.py. "grid" holds the metric\n'
            '# embedding of the cells (see remroc_ha/grid.py).\n' +
            yaml.safe_dump(self.grid().to_yaml_dict(), sort_keys=False, default_flow_style=None))
        base = sdf_dir / f'{self.name}_0_0.sdf'
        base.write_text(self.sdf_world())
        return {'map': maps_dir / f'{self.name}.yaml', 'mapf': params_dir / f'mapf_{self.name}.yaml', 'base_sdf': base}


def get_layout(name: str, **kw) -> Layout:
    return Layout(name, LAYOUTS[name], **kw)
