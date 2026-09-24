# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Differential-drive robot with a nav2-like path follower and local safety behaviour.

This is the part of the simulator that creates the plan-execution gap:

* acceleration / deceleration limits and turning in place (SmallDeliveryRobot limits:
  0.5 m/s, MPPI ``vx_max``), so one grid step takes a variable amount of time,
* a random speed factor per commanded path and random short stalls (localisation or
  controller hiccups),
* a local safety layer that stops the robot when a human or another robot is in
  front of it (in narrow spaces nav2 cannot swerve either). Humans are scripted and
  do not react to robots, exactly like Gazebo actors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class RobotParams:
    v_max: float = 0.5
    acc: float = 0.5
    dec: float = 1.3
    w_max: float = 1.5
    k_heading: float = 2.5
    rotate_threshold: float = 0.6          # rad; larger heading errors are corrected in place
    body_radius: float = 0.3
    stop_margin: float = 0.25              # stop if an obstacle in front is closer than radii + margin
    slow_distance: float = 1.2             # start slowing down this far beyond the stop distance
    front_angle: float = 1.2               # rad; half-width of the "in front" cone
    reach_final: float = 0.12
    reach_intermediate: float = 0.3
    passed_goal_radius: float = 0.7        # nav2 RemovePassedGoals
    speed_factor_range: Tuple[float, float] = (0.75, 1.0)
    stall_rate: float = 1.0 / 60.0         # per second
    stall_duration: Tuple[float, float] = (1.0, 3.0)
    avoid_robots: bool = True              # False: the local layer ignores other robots (stress test)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimRobot:
    def __init__(self, name: str, x: float, y: float, theta: float, params: RobotParams, rng: np.random.Generator):
        self.name = name
        self.x, self.y, self.theta = x, y, theta
        self.v = 0.0
        self.w = 0.0
        self.p = params
        self.rng = rng
        self.path: List[Tuple[float, float]] = []
        self.speed_factor = 1.0
        self.stall_until = -1.0
        self.blocked = False

    @property
    def speed(self) -> float:
        return abs(self.v)

    def set_path(self, points: Sequence[Tuple[float, float]]):
        pts = [tuple(map(float, p)) for p in points]
        # nav2 RemovePassedGoals: drop leading goals the robot is already close to (keep >= 1)
        while len(pts) > 1 and math.hypot(pts[0][0] - self.x, pts[0][1] - self.y) < self.p.passed_goal_radius:
            pts.pop(0)
        if pts != self.path:
            self.path = pts
            self.speed_factor = float(self.rng.uniform(*self.p.speed_factor_range))

    def _clearance_limit(self, obstacles: Sequence[Tuple[float, float, float]]) -> float:
        """Speed limit from obstacles in front of the robot (1 = free, 0 = stop)."""
        limit = 1.0
        for ox, oy, r in obstacles:
            dx, dy = ox - self.x, oy - self.y
            d = math.hypot(dx, dy)
            if d < 1e-6:
                return 0.0
            if abs(wrap(math.atan2(dy, dx) - self.theta)) > self.p.front_angle:
                continue
            stop = self.p.body_radius + r + self.p.stop_margin
            if d <= stop:
                return 0.0
            if d < stop + self.p.slow_distance:
                limit = min(limit, (d - stop) / self.p.slow_distance)
        return limit

    def step(self, t: float, dt: float, obstacles: Sequence[Tuple[float, float, float]]):
        p = self.p
        # drop reached waypoints
        while self.path:
            gx, gy = self.path[0]
            tol = p.reach_final if len(self.path) == 1 else p.reach_intermediate
            if math.hypot(gx - self.x, gy - self.y) <= tol:
                self.path.pop(0)
            else:
                break
        if t >= self.stall_until and self.path and self.rng.random() < p.stall_rate * dt:
            self.stall_until = t + float(self.rng.uniform(*p.stall_duration))

        v_des, w_des = 0.0, 0.0
        self.blocked = False
        if self.path and t >= self.stall_until:
            gx, gy = self.path[0]
            err = wrap(math.atan2(gy - self.y, gx - self.x) - self.theta)
            if abs(err) > p.rotate_threshold:
                w_des = math.copysign(p.w_max, err)
                v_des = 0.0
            else:
                w_des = max(-p.w_max, min(p.w_max, p.k_heading * err))
                remaining = math.hypot(gx - self.x, gy - self.y)
                for a, b in zip(self.path[:-1], self.path[1:]):
                    remaining += math.hypot(b[0] - a[0], b[1] - a[1])
                v_des = min(p.v_max * self.speed_factor * math.cos(err),
                            math.sqrt(2.0 * p.dec * max(remaining - p.reach_final * 0.5, 0.0)))
                lim = self._clearance_limit(obstacles)
                if lim <= 0.0:
                    self.blocked = True
                v_des *= lim
        # acceleration limits
        if v_des > self.v:
            self.v = min(v_des, self.v + p.acc * dt)
        else:
            self.v = max(v_des, self.v - p.dec * dt)
        self.w = w_des
        self.theta = wrap(self.theta + self.w * dt)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
