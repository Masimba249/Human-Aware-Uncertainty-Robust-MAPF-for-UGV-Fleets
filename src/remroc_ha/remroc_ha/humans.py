# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Scripted human (Gazebo actor) trajectories: SDF/JSON I/O and playback.

A Gazebo actor with ``<loop>true</loop>`` and a ``<trajectory>`` of timed waypoints
moves along the waypoints by linear interpolation and restarts after the last one.
``HumanScript`` reproduces exactly that, so the simulator and the ROS human pose
publisher see the same human motion as the Gazebo rendering / lidar.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class HumanScript:
    def __init__(self, name: str, waypoints: np.ndarray, delay_start: float = 0.0, loop: bool = True):
        """waypoints: (N, 3+) array of [t, x, y, (yaw)] sorted by t."""
        wp = np.asarray(waypoints, dtype=np.float64)
        order = np.argsort(wp[:, 0], kind='stable')
        self.name = name
        self.wp = wp[order]
        self.delay_start = delay_start
        self.loop = loop

    @property
    def duration(self) -> float:
        return float(self.wp[-1, 0])

    def position(self, t: float) -> Tuple[float, float]:
        wp = self.wp
        tt = max(0.0, t - self.delay_start)
        T = self.duration
        if self.loop and T > 0:
            tt = tt % T
        tt = min(tt, T)
        i = int(np.searchsorted(wp[:, 0], tt, side='right')) - 1
        i = min(max(i, 0), len(wp) - 2) if len(wp) > 1 else 0
        if len(wp) == 1:
            return float(wp[0, 1]), float(wp[0, 2])
        t0, t1 = wp[i, 0], wp[i + 1, 0]
        a = 0.0 if t1 <= t0 else min(1.0, max(0.0, (tt - t0) / (t1 - t0)))
        return (float(wp[i, 1] + a * (wp[i + 1, 1] - wp[i, 1])),
                float(wp[i, 2] + a * (wp[i + 1, 2] - wp[i, 2])))

    def positions(self, times: Sequence[float]) -> np.ndarray:
        return np.array([self.position(t) for t in times])


def load_humans_json(path) -> List[HumanScript]:
    data = json.loads(Path(path).read_text())
    return [HumanScript(h['name'], np.asarray(h['waypoints']), h.get('delay_start', 0.0), h.get('loop', True))
            for h in data['humans']]


def save_humans_json(path, humans: Sequence[HumanScript], meta: dict = None):
    data = {'meta': meta or {},
            'humans': [{'name': h.name, 'delay_start': h.delay_start, 'loop': h.loop,
                        'waypoints': np.round(h.wp, 4).tolist()} for h in humans]}
    Path(path).write_text(json.dumps(data))


def load_humans_from_sdf(path) -> List[HumanScript]:
    """Parses <actor> elements with scripted trajectories from a Gazebo world file."""
    text = Path(path).read_text()
    root = ET.fromstring(re.sub(r'<\?xml[^>]*\?>', '', text, count=1).strip())
    out = []
    for actor in root.iter('actor'):
        script = actor.find('script')
        if script is None:
            continue
        loop = (script.findtext('loop') or 'true').strip().lower() == 'true'
        delay = float(script.findtext('delay_start') or 0.0)
        wps = []
        for wp in script.iter('waypoint'):
            t = float(wp.findtext('time'))
            pose = [float(v) for v in wp.findtext('pose').split()]
            wps.append([t, pose[0], pose[1], pose[5] if len(pose) > 5 else 0.0])
        if wps:
            out.append(HumanScript(actor.get('name'), np.array(wps), delay, loop))
    return out


def humans_at(humans: Sequence[HumanScript], t: float) -> List[Tuple[str, float, float]]:
    return [(h.name, *h.position(t)) for h in humans]


def all_trajectories(humans: Sequence[HumanScript], horizon: float = None, dt: float = 0.5) -> List[np.ndarray]:
    """Sampled (t, x, y) arrays, e.g. to fit a map of dynamics."""
    out = []
    for h in humans:
        T = horizon or h.duration
        ts = np.arange(0.0, T, dt)
        out.append(np.column_stack([ts, h.positions(ts)]))
    return out
