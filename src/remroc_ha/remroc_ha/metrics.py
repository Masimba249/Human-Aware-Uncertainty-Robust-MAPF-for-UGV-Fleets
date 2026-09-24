# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Episode metrics, computed identically for every coordinator from a trajectory log.

Log schema (produced by the simulator and by the ROS ``metrics_recorder`` node)::

    {
      "meta":   {"coordinator", "world", "number_of_humans", "sample", "robots": [names],
                 "goals": {name: [x, y]}, "time_limit", ...},
      "frames": [{"t": float, "robots": {name: [x, y, speed]}, "humans": [[x, y], ...]}, ...],
      "events": [...],                     # coordinator events (optional)
      "coordinator_stats": {...}           # e.g. replans, solver calls (optional)
    }

Definitions
-----------
* arrival time: the time a robot last *entered* the goal region (``goal_tol``) and
  stayed there until the end of the episode.
* success: every robot arrived before the time limit and no robot-robot collision
  (centre distance below ``2 * robot_body_radius``) occurred.
* makespan / sum of costs: max / sum of the arrival times (successful episodes).
* minimum human-robot distance: minimum centre distance over the episode. Scripted
  humans (Gazebo actors) do not avoid robots, so this extreme statistic is complemented
  by the time any robot spends within ``near_human_dist`` of a human and by the number
  of close encounters (a robot-human pair coming closer than ``encounter_dist``).
* stuck time: summed over robots, time spent below ``stuck_speed`` while not at the
  goal, counting only standstills lasting at least ``stuck_min_duration``.
* deadlock time: time during which *every* robot that has not arrived is standing
  still, counting only such fleet-wide standstills of at least ``deadlock_min_duration``.
* replans: number of plan changes after the initial plan (coordinator specific, from
  ``coordinator_stats['replans']``).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _runs(mask: np.ndarray, dt: np.ndarray, min_duration: float) -> float:
    """Total duration of runs of True lasting at least min_duration."""
    total, run = 0.0, 0.0
    for m, d in zip(mask, dt):
        if m:
            run += d
        else:
            if run >= min_duration:
                total += run
            run = 0.0
    if run >= min_duration:
        total += run
    return total


def compute_metrics(log: dict, goal_tol: float = 0.5, robot_body_radius: float = 0.3,
                    stuck_speed: float = 0.05, stuck_min_duration: float = 2.0,
                    deadlock_min_duration: float = 5.0, near_human_dist: float = 1.0,
                    encounter_dist: float = 0.5) -> Dict[str, float]:
    meta = log['meta']
    frames = log['frames']
    names: List[str] = list(meta['robots'])
    goals = {r: np.asarray(meta['goals'][r], dtype=float) for r in names}
    time_limit = float(meta.get('time_limit', math.inf))
    if not frames:
        return {'success': 0.0}
    t = np.array([f['t'] for f in frames])
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.0))
    pos = {r: np.array([f['robots'][r][:2] for f in frames]) for r in names}
    spd = {r: np.array([f['robots'][r][2] for f in frames]) for r in names}

    arrival = {}
    at_goal = {}
    for r in names:
        inside = np.linalg.norm(pos[r] - goals[r], axis=1) <= goal_tol
        at_goal[r] = inside
        if inside[-1]:
            k = len(inside) - 1
            while k > 0 and inside[k - 1]:
                k -= 1
            arrival[r] = float(t[k])
        else:
            arrival[r] = math.nan

    # robot-robot collisions / clearance
    min_rr = math.inf
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            min_rr = min(min_rr, float(np.min(np.linalg.norm(pos[a] - pos[b], axis=1))))
    collision = min_rr < 2 * robot_body_radius

    # human-robot distance
    min_hr = math.inf
    close_time = 0.0
    encounters = 0
    prev_close = None
    for k, f in enumerate(frames):
        hs = np.asarray(f.get('humans') or [], dtype=float).reshape(-1, 2)
        if not len(hs):
            prev_close = None
            continue
        rp = np.stack([pos[r][k] for r in names])
        d = np.linalg.norm(rp[:, None, :] - hs[None, :, :], axis=2)     # robots x humans
        min_hr = min(min_hr, float(d.min()))
        if d.min() < near_human_dist:
            close_time += dt[k]
        close = d < encounter_dist
        if prev_close is not None and prev_close.shape == close.shape:
            encounters += int((close & ~prev_close).sum())
        else:
            encounters += int(close.sum())
        prev_close = close

    stuck = {}
    for r in names:
        mask = (spd[r] < stuck_speed) & ~at_goal[r]
        stuck[r] = _runs(mask, dt, stuck_min_duration)
    active = np.array([[not at_goal[r][k] for r in names] for k in range(len(frames))])
    still = np.array([[spd[r][k] < stuck_speed for r in names] for k in range(len(frames))])
    fleet_frozen = active.any(axis=1) & np.all(~active | still, axis=1)
    deadlock = _runs(fleet_frozen, dt, deadlock_min_duration)

    arrived = [arrival[r] for r in names]
    all_arrived = all(not math.isnan(a) and a <= time_limit for a in arrived)
    success = all_arrived and not collision
    stats = log.get('coordinator_stats') or {}
    return {
        'success': float(success),
        'all_arrived': float(all_arrived),
        'robot_collision': float(collision),
        'robots_arrived': float(sum(not math.isnan(a) for a in arrived)),
        'makespan': max(arrived) if all_arrived else math.nan,
        'sum_of_costs': sum(arrived) if all_arrived else math.nan,
        'min_robot_robot_dist': min_rr,
        'min_human_robot_dist': min_hr if min_hr < math.inf else math.nan,
        'time_near_humans': close_time,
        'human_encounters': float(encounters),
        'stuck_time': float(sum(stuck.values())),
        'deadlock_time': deadlock,
        'replans': float(stats.get('replans', math.nan)),
        'solver_calls': float(stats.get('solver_calls', math.nan)),
        'episode_time': float(t[-1]),
    }
