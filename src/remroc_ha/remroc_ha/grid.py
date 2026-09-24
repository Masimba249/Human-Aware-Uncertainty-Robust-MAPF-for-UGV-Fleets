# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""MAPF grid abstraction shared by the coordinator, the predictor and the simulator.

Cells are indexed exactly like the ``mapf_<map>.yaml`` files consumed by
``remroc_mapf_solver`` and like ``coordinator_mapf.py``: ``x`` grows to the right and
``y`` is *inverted* w.r.t. the map frame (``y = 0`` is the top row of the image).
The metric centroid of cell ``(x, y)`` is::

    yy = dimy - 1 - y
    cx = resolution * (x  * cell_px + x_offset + cell_px / 2) + origin_x
    cy = resolution * (yy * cell_px + y_offset + cell_px / 2) + origin_y
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

Cell = Tuple[int, int]

# Grid metadata that coordinator_mapf.py hard-codes for the upstream maps.
KNOWN_GRID_METADATA = {
    'depot': dict(resolution=0.04, cell_size_px=29, x_offset=4, y_offset=14, origin=[-15.1, -7.74]),
    'depot_mod': dict(resolution=0.04, cell_size_px=29, x_offset=4, y_offset=14, origin=[-15.1, -7.74]),
    'simple': dict(resolution=0.05, cell_size_px=29, x_offset=15, y_offset=15, origin=[-5.08, -5.09]),
}

MOVES: Tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class GridSpec:
    dimx: int
    dimy: int
    obstacles: frozenset = field(default_factory=frozenset)
    resolution: float = 0.05
    cell_size_px: int = 20
    x_offset: float = 0.0
    y_offset: float = 0.0
    origin: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mapf_yaml(cls, path, map_name: Optional[str] = None) -> 'GridSpec':
        with Path(path).open('r') as f:
            data = yaml.safe_load(f)
        dimx, dimy = (int(v) for v in data['map']['dimensions'])
        obstacles = frozenset((int(o[0]), int(o[1])) for o in (data['map'].get('obstacles') or []))
        meta = data.get('grid')
        if meta is None:
            name = map_name or Path(path).stem.replace('mapf_', '', 1)
            if name not in KNOWN_GRID_METADATA:
                raise ValueError(f'{path} has no "grid" metadata block and "{name}" is not a known map')
            meta = KNOWN_GRID_METADATA[name]
        return cls(dimx, dimy, obstacles, float(meta['resolution']), int(meta['cell_size_px']),
                   float(meta['x_offset']), float(meta['y_offset']),
                   (float(meta['origin'][0]), float(meta['origin'][1])))

    def to_yaml_dict(self) -> dict:
        return {
            'map': {'dimensions': [self.dimx, self.dimy],
                    'obstacles': [list(o) for o in sorted(self.obstacles)]},
            'grid': {'resolution': self.resolution, 'cell_size_px': self.cell_size_px,
                     'x_offset': self.x_offset, 'y_offset': self.y_offset,
                     'origin': [float(self.origin[0]), float(self.origin[1])]},
        }

    # ------------------------------------------------------------------ topology
    @property
    def cell_size(self) -> float:
        return self.resolution * self.cell_size_px

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.dimx and 0 <= c[1] < self.dimy

    def is_free(self, c: Cell) -> bool:
        return self.in_bounds(c) and c not in self.obstacles

    def free_cells(self) -> List[Cell]:
        return [(x, y) for y in range(self.dimy) for x in range(self.dimx) if (x, y) not in self.obstacles]

    def neighbors(self, c: Cell) -> List[Cell]:
        return [(c[0] + dx, c[1] + dy) for dx, dy in MOVES if self.is_free((c[0] + dx, c[1] + dy))]

    def free_mask(self) -> np.ndarray:
        """Boolean array indexed [y, x] (MAPF convention)."""
        m = np.ones((self.dimy, self.dimx), dtype=bool)
        for (x, y) in self.obstacles:
            m[y, x] = False
        return m

    # ------------------------------------------------------------------ metric <-> cell
    def centroid(self, c: Cell) -> Tuple[float, float]:
        x, y = c
        yy = self.dimy - 1 - y
        r, cs = self.resolution, self.cell_size_px
        return (r * (x * cs + self.x_offset + cs / 2) + self.origin[0],
                r * (yy * cs + self.y_offset + cs / 2) + self.origin[1])

    def centroid_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Metric centroid coordinates as arrays indexed [y, x]."""
        xs = np.arange(self.dimx)
        ys = np.arange(self.dimy)
        yy = self.dimy - 1 - ys
        r, cs = self.resolution, self.cell_size_px
        cx = r * (xs * cs + self.x_offset + cs / 2) + self.origin[0]
        cy = r * (yy * cs + self.y_offset + cs / 2) + self.origin[1]
        return np.broadcast_to(cx[None, :], (self.dimy, self.dimx)), np.broadcast_to(cy[:, None], (self.dimy, self.dimx))

    def cell_at(self, x: float, y: float) -> Cell:
        """Cell whose square contains the metric point (may be an obstacle / out of bounds)."""
        r, cs = self.resolution, self.cell_size_px
        cx = math.floor(((x - self.origin[0]) / r - self.x_offset) / cs)
        yy = math.floor(((y - self.origin[1]) / r - self.y_offset) / cs)
        return (cx, self.dimy - 1 - yy)

    def nearest_free_cell(self, x: float, y: float, exclude: Iterable[Cell] = ()) -> Cell:
        excl = set(exclude)
        best, best_d = None, float('inf')
        for c in self.free_cells():
            if c in excl:
                continue
            cx, cy = self.centroid(c)
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best, best_d = c, d
        if best is None:
            raise ValueError('no free cell available')
        return best

    def assign_unique_cells(self, points: Sequence[Tuple[str, Tuple[float, float]]]) -> Dict[str, Cell]:
        """Greedy nearest-free-cell assignment in the given order, one robot per cell
        (the same procedure coordinator_mapf.py uses)."""
        taken: set = set()
        out = {}
        for name, (x, y) in points:
            c = self.nearest_free_cell(x, y, taken)
            out[name] = c
            taken.add(c)
        return out

    def bfs_distances(self, goal: Cell) -> Dict[Cell, int]:
        dist = {goal: 0}
        frontier = [goal]
        while frontier:
            nxt = []
            for c in frontier:
                for n in self.neighbors(c):
                    if n not in dist:
                        dist[n] = dist[c] + 1
                        nxt.append(n)
            frontier = nxt
        return dist

    def shortest_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Static BFS shortest path (used by the PBC baseline in the simulator)."""
        prev = {start: None}
        frontier = [start]
        while frontier and goal not in prev:
            nxt = []
            for c in frontier:
                for n in self.neighbors(c):
                    if n not in prev:
                        prev[n] = c
                        nxt.append(n)
            frontier = nxt
        if goal not in prev:
            return None
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        return path[::-1]
