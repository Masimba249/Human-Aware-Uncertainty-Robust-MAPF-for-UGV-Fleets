# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""ROS-free generation of human trajectories on an occupancy map.

This replaces the nav2 ``ComputePathToPose`` round trip of the original launch files
for batch generation: start/goal pairs are sampled in free space, connected with A*
on the (inflated) occupancy grid, shortcut-smoothed and time-parameterised with a
per-human walking speed. As in the original REMROC worlds every human walks to its
goal and back, looping forever; a random phase offset de-synchronises the humans.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from remroc_ha.occupancy import OccupancyMap


def _astar(free: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    h, w = free.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    def heur(a):
        dr, dc = abs(a[0] - goal[0]), abs(a[1] - goal[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

    g = {start: 0.0}
    parent = {start: None}
    heap = [(heur(start), 0.0, start)]
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        if gc > g.get(cur, math.inf):
            continue
        for dr, dc, cost in moves:
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < h and 0 <= n[1] < w) or not free[n]:
                continue
            if dr and dc and not (free[cur[0] + dr, cur[1]] and free[cur[0], cur[1] + dc]):
                continue
            ng = gc + cost
            if ng < g.get(n, math.inf):
                g[n] = ng
                parent[n] = cur
                heapq.heappush(heap, (ng + heur(n), ng, n))
    return None


def _line_free(free: np.ndarray, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rr = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return bool(free[rr, cc].all())


def _shortcut(free: np.ndarray, path):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(free, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


class HumanTrajectoryGenerator:
    def __init__(self, occ: OccupancyMap, clearance: float = 0.35, plan_resolution: float = 0.1,
                 min_distance: float = 6.0, speed_range=(0.6, 1.1), waypoint_spacing: float = 0.25):
        self.occ = occ
        self.clear = occ.inflated(clearance)
        # downsample for planning speed
        self.step = max(1, int(round(plan_resolution / occ.resolution)))
        self.free = self.clear.free[::self.step, ::self.step]
        self.min_distance = min_distance
        self.speed_range = speed_range
        self.spacing = waypoint_spacing

    def _px_to_world(self, rc):
        return self.occ.pixel_to_world(rc[0] * self.step + self.step // 2, rc[1] * self.step + self.step // 2)

    def _sample_free(self, rng) -> Tuple[int, int]:
        idx = np.argwhere(self.free)
        return tuple(idx[rng.integers(len(idx))])

    def plan(self, rng, min_distance: Optional[float] = None, max_tries: int = 500) -> List[Tuple[float, float]]:
        """Polyline (metres) between a random start and a goal at least min_distance away."""
        min_distance = self.min_distance if min_distance is None else min_distance
        for _ in range(max_tries):
            s, g = self._sample_free(rng), self._sample_free(rng)
            sx, sy = self._px_to_world(s)
            gx, gy = self._px_to_world(g)
            if math.hypot(gx - sx, gy - sy) < min_distance:
                continue
            path = _astar(self.free, s, g)
            if path is None:
                continue
            pts = [self._px_to_world(p) for p in _shortcut(self.free, path)]
            return [(float(x), float(y)) for x, y in pts]
        raise RuntimeError('could not sample a human path; is the map connected?')

    def trajectory(self, rng, min_distance: Optional[float] = None) -> np.ndarray:
        """Looping trajectory: array (N, 4) of [t, x, y, yaw], out and back, random phase."""
        pts = self.plan(rng, min_distance)
        speed = float(rng.uniform(*self.speed_range))
        # resample polyline at constant spacing
        out: List[Tuple[float, float, float]] = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            n = max(1, int(math.ceil(seg / self.spacing)))
            for k in range(n):
                out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, yaw))
        out.append((pts[-1][0], pts[-1][1], out[-1][2]))
        back = [(x, y, yaw + math.pi) for (x, y, yaw) in reversed(out)]
        loop = out + back[1:]
        # random phase: rotate the closed loop (first == last position), then
        # time-parameterise by arc length
        k = int(rng.integers(len(loop) - 1))
        rot = loop[k:-1] + loop[:k + 1]
        t_rot = [0.0]
        for a, b in zip(rot[:-1], rot[1:]):
            t_rot.append(t_rot[-1] + max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-3) / speed)
        return np.array([[t, x, y, (yaw + math.pi) % (2 * math.pi) - math.pi]
                         for t, (x, y, yaw) in zip(t_rot, rot)])

