# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Human motion prediction and rasterisation into time-indexed MAPF risk maps.

Two predictors are provided:

* ``ConstantVelocityPredictor``: Gaussian prediction with linearly growing
  uncertainty (the standard short-horizon baseline).
* ``ParticleMoDPredictor``: particles propagated through a *map of dynamics*
  (``MapOfDynamics``), a light-weight variant of CLiFF-maps (Kucner et al.): for each
  location it stores a distribution over motion directions (``n_dirs`` bins) with a
  mean speed per direction, learnt from observed human trajectories. Particles
  blend their current velocity with directions sampled from the flow field and are
  kept inside walkable space, so predictions follow corridors and typical flows.

``RiskMapBuilder`` turns either prediction into ``risk[t, y, x]``: the probability that
at least one human (footprint inflated by ``inflation``) occupies MAPF cell ``(x, y)``
at time ``t_now + t * step_duration``. An extra last layer holds the static prior
(time-averaged occupancy from the map of dynamics), which the solver uses for every
time step beyond the prediction horizon.

Tracking-based predictions lose their information quickly (with walking humans and
~3 s MAPF steps they beat an all-zero forecast only for a few steps, see
docs/EXPERIMENTS.md), whereas the static prior is a calibrated long-run base rate.
``HumanRiskPredictor`` therefore blends them convexly,
``risk_k = a_k * prediction_k + (1 - a_k) * prior`` with ``a_k = exp(-t_k / tau)``
(``tau = prior_blend_time``, selected on held-out worlds), which avoids counting the
tracked humans twice.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

from .grid import GridSpec


# ---------------------------------------------------------------------------- tracking
@dataclass
class Track:
    pos: np.ndarray
    vel: np.ndarray
    stamp: float


class HumanTracker:
    """Keeps a short history per human id and estimates velocity by least squares."""

    def __init__(self, window_s: float = 1.5, forget_s: float = 3.0, max_speed: float = 2.0):
        self.window_s = window_s
        self.forget_s = forget_s
        self.max_speed = max_speed
        self._hist: Dict[object, deque] = {}

    def update(self, t: float, observations: Iterable[Tuple[object, float, float]]):
        for hid, x, y in observations:
            h = self._hist.setdefault(hid, deque())
            if h and t <= h[-1][0]:
                continue
            h.append((t, float(x), float(y)))
            while h and t - h[0][0] > self.window_s:
                h.popleft()

    def tracks(self, t_now: float) -> Dict[object, Track]:
        out = {}
        for hid, h in list(self._hist.items()):
            if not h or t_now - h[-1][0] > self.forget_s:
                self._hist.pop(hid, None)
                continue
            arr = np.array(h)
            vel = np.zeros(2)
            if len(arr) >= 2 and arr[-1, 0] - arr[0, 0] >= 0.2:
                tt = arr[:, 0] - arr[:, 0].mean()
                denom = float((tt * tt).sum())
                vel = np.array([(tt * (arr[:, 1] - arr[:, 1].mean())).sum() / denom,
                                (tt * (arr[:, 2] - arr[:, 2].mean())).sum() / denom])
                s = float(np.hypot(*vel))
                if s > self.max_speed:
                    vel *= self.max_speed / s
            # extrapolate the last observation to t_now
            pos = arr[-1, 1:3] + vel * (t_now - arr[-1, 0])
            out[hid] = Track(pos, vel, float(arr[-1, 0]))
        return out


# ---------------------------------------------------------------------------- predictors
class ConstantVelocityPredictor:
    def __init__(self, sigma0: float = 0.15, sigma_growth: float = 0.35):
        self.sigma0 = sigma0
        self.sigma_growth = sigma_growth

    def predict(self, tracks: Dict[object, Track], dts: Sequence[float]):
        """Returns means (N, K, 2) and isotropic std devs (N, K)."""
        dts = np.asarray(dts, dtype=np.float64)
        ids = list(tracks)
        if not ids:
            return ids, np.zeros((0, len(dts), 2)), np.zeros((0, len(dts)))
        pos = np.stack([tracks[i].pos for i in ids])
        vel = np.stack([tracks[i].vel for i in ids])
        means = pos[:, None, :] + vel[:, None, :] * dts[None, :, None]
        sig = np.sqrt(self.sigma0 ** 2 + (self.sigma_growth * dts) ** 2)
        return ids, means, np.broadcast_to(sig, (len(ids), len(dts))).copy()


class MapOfDynamics:
    """Direction-histogram flow field (CLiFF-lite) plus a dwell-time occupancy prior."""

    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.5, n_dirs: int = 8):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.resolution = resolution
        self.n_dirs = n_dirs
        self.nx = max(1, int(math.ceil((self.xmax - self.xmin) / resolution)))
        self.ny = max(1, int(math.ceil((self.ymax - self.ymin) / resolution)))
        self.weight = np.zeros((self.ny, self.nx, n_dirs))
        self.speed_sum = np.zeros((self.ny, self.nx, n_dirs))
        self.dwell = np.zeros((self.ny, self.nx))
        self.total_time = 0.0
        self._probs = None
        self._speeds = None

    def _index(self, x, y):
        ix = np.clip(np.floor((np.asarray(x) - self.xmin) / self.resolution).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.floor((np.asarray(y) - self.ymin) / self.resolution).astype(np.int64), 0, self.ny - 1)
        return ix, iy

    def fit(self, trajectories: Iterable[np.ndarray], min_speed: float = 0.05, blur: int = 1):
        """trajectories: arrays of shape (N, 3) with columns t, x, y (one per human)."""
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if len(traj) < 2:
                continue
            dt = np.diff(traj[:, 0])
            ok = dt > 1e-6
            d = np.diff(traj[:, 1:3], axis=0)
            mid = 0.5 * (traj[1:, 1:3] + traj[:-1, 1:3])
            ix, iy = self._index(mid[:, 0], mid[:, 1])
            np.add.at(self.dwell, (iy[ok], ix[ok]), dt[ok])
            self.total_time += float(traj[-1, 0] - traj[0, 0])
            speed = np.zeros_like(dt)
            speed[ok] = np.hypot(d[ok, 0], d[ok, 1]) / dt[ok]
            moving = ok & (speed > min_speed)
            ang = np.arctan2(d[moving, 1], d[moving, 0])
            k = np.round(ang / (2 * np.pi / self.n_dirs)).astype(np.int64) % self.n_dirs
            np.add.at(self.weight, (iy[moving], ix[moving], k), dt[moving])
            np.add.at(self.speed_sum, (iy[moving], ix[moving], k), speed[moving] * dt[moving])
        self._finalize(blur)
        return self

    def _finalize(self, blur: int):
        w, s = self.weight, self.speed_sum
        if blur > 0:
            from scipy.ndimage import uniform_filter
            size = (2 * blur + 1, 2 * blur + 1, 1)
            w = uniform_filter(w, size=size, mode='constant')
            s = uniform_filter(s, size=size, mode='constant')
        tot = w.sum(axis=2, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            self._probs = np.where(tot > 1e-9, w / np.maximum(tot, 1e-12), np.nan)
            self._speeds = np.where(w > 1e-9, s / np.maximum(w, 1e-12), np.nan)

    def direction_distribution(self, x, y):
        """Arrays (M, n_dirs) of direction probabilities and mean speeds (NaN where unknown)."""
        ix, iy = self._index(x, y)
        return self._probs[iy, ix], self._speeds[iy, ix]

    def occupancy_prior_on_grid(self, grid: GridSpec, n_humans: int, inflation: float = 0.3) -> np.ndarray:
        """P(at least one of n_humans is in the (inflated) cell) at a random time, indexed [y, x]."""
        prior = np.zeros((grid.dimy, grid.dimx), dtype=np.float64)
        if self.total_time <= 0 or n_humans <= 0:
            return prior.astype(np.float32)
        # fraction of time one (average) human spends in each fine cell
        n_traj_time = self.total_time
        frac = self.dwell / n_traj_time
        ys, xs = np.nonzero(frac)
        px = self.xmin + (xs + 0.5) * self.resolution
        py = self.ymin + (ys + 0.5) * self.resolution
        u, v = _cell_coordinates(grid, px, py)
        a = inflation / grid.cell_size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cx = np.floor(u + 0.5).astype(np.int64) + dx
                cy = np.floor(v + 0.5).astype(np.int64) + dy
                inside = ((np.abs(u - cx) <= 0.5 + a) & (np.abs(v - cy) <= 0.5 + a) &
                          (cx >= 0) & (cx < grid.dimx) & (cy >= 0) & (cy < grid.dimy))
                np.add.at(prior, (grid.dimy - 1 - cy[inside], cx[inside]), frac[ys[inside], xs[inside]])
        p1 = np.clip(prior, 0.0, 1.0)
        return (1.0 - (1.0 - p1) ** n_humans).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, bounds=np.array([self.xmin, self.ymin, self.xmax, self.ymax]),
                            resolution=self.resolution, n_dirs=self.n_dirs, weight=self.weight,
                            speed_sum=self.speed_sum, dwell=self.dwell, total_time=self.total_time)

    @classmethod
    def load(cls, path, blur: int = 1) -> 'MapOfDynamics':
        d = np.load(path)
        m = cls(tuple(d['bounds']), float(d['resolution']), int(d['n_dirs']))
        m.weight, m.speed_sum, m.dwell = d['weight'], d['speed_sum'], d['dwell']
        m.total_time = float(d['total_time'])
        m._finalize(blur)
        return m


class ParticleMoDPredictor:
    """Particle prediction guided by a map of dynamics and constrained to walkable space."""

    def __init__(self, mod: Optional[MapOfDynamics], walkable: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 n_particles: int = 48, sub_dt: float = 0.5, mod_rate: float = 0.5, heading_kappa: float = 2.0,
                 heading_noise: float = 0.25, speed_noise: float = 0.1, pos_noise: float = 0.08, seed: int = 0):
        self.mod = mod
        self.walkable = walkable
        self.n = n_particles
        self.sub_dt = sub_dt
        self.mod_rate = mod_rate            # per second: how often a particle re-samples from the flow field
        self.kappa = heading_kappa          # preference for directions close to the current heading
        self.heading_noise = heading_noise
        self.speed_noise = speed_noise
        self.pos_noise = pos_noise
        self.rng = np.random.default_rng(seed)

    def predict(self, tracks: Dict[object, Track], dts: Sequence[float]):
        """Returns ids and particle positions of shape (N, K, P, 2) at the requested times."""
        dts = np.asarray(dts, dtype=np.float64)
        ids = list(tracks)
        K, P = len(dts), self.n
        out = np.zeros((len(ids), K, P, 2))
        if not ids:
            return ids, out
        rng = self.rng
        pos = np.concatenate([np.repeat(tracks[i].pos[None], P, axis=0) for i in ids])
        pos += rng.normal(0.0, self.pos_noise, pos.shape)
        vel = np.concatenate([np.repeat(tracks[i].vel[None], P, axis=0) for i in ids])
        speed = np.hypot(vel[:, 0], vel[:, 1]) * np.exp(rng.normal(0.0, self.speed_noise, len(pos)))
        heading = np.arctan2(vel[:, 1], vel[:, 0]) + rng.normal(0.0, self.heading_noise, len(pos))
        t, k = 0.0, 0
        while k < K and dts[k] <= 1e-9:
            out[:, k] = pos.reshape(len(ids), P, 2)
            k += 1
        while k < K:
            step = min(self.sub_dt, dts[k] - t)
            self._advance(pos, speed, heading, step)
            t += step
            if t >= dts[k] - 1e-9:
                out[:, k] = pos.reshape(len(ids), P, 2)
                k += 1
        return ids, out

    def _advance(self, pos, speed, heading, dt):
        rng = self.rng
        M = len(pos)
        if self.mod is not None:
            resample = rng.random(M) < 1.0 - math.exp(-self.mod_rate * dt)
            if resample.any():
                probs, speeds = self.mod.direction_distribution(pos[resample, 0], pos[resample, 1])
                known = ~np.isnan(probs).any(axis=1)
                idx = np.nonzero(resample)[0][known]
                if len(idx):
                    n_dirs = probs.shape[1]
                    dirs = np.arange(n_dirs) * 2 * np.pi / n_dirs
                    w = probs[known] * np.exp(self.kappa * np.cos(dirs[None, :] - heading[idx, None]))
                    w /= w.sum(axis=1, keepdims=True)
                    cum = np.cumsum(w, axis=1)
                    choice = (cum < rng.random(len(idx))[:, None]).sum(axis=1).clip(0, n_dirs - 1)
                    heading[idx] = dirs[choice] + rng.normal(0.0, self.heading_noise, len(idx))
                    s = speeds[known][np.arange(len(idx)), choice]
                    good = ~np.isnan(s)
                    speed[idx[good]] = 0.5 * speed[idx[good]] + 0.5 * s[good]
        heading += rng.normal(0.0, self.heading_noise * math.sqrt(dt), M)
        new = pos + np.stack([np.cos(heading), np.sin(heading)], axis=1) * (speed * dt)[:, None]
        ok = self.walkable(new[:, 0], new[:, 1])
        pos[ok] = new[ok]
        # blocked particles turn around (humans do not walk into walls)
        heading[~ok] += np.pi + rng.normal(0.0, 0.5, int((~ok).sum()))


# ---------------------------------------------------------------------------- rasterisation
def _cell_coordinates(grid: GridSpec, x, y):
    """Continuous cell coordinates: u == i at the centroid of column i, v == yy (un-inverted row)."""
    r, cs = grid.resolution, grid.cell_size_px
    u = ((np.asarray(x) - grid.origin[0]) / r - grid.x_offset) / cs - 0.5
    v = ((np.asarray(y) - grid.origin[1]) / r - grid.y_offset) / cs - 0.5
    return u, v


class RiskMapBuilder:
    def __init__(self, grid: GridSpec, step_duration: float, horizon: int, inflation: float = 0.3):
        self.grid = grid
        self.step_duration = step_duration
        self.horizon = horizon
        self.inflation = inflation
        self.free = grid.free_mask()

    def layer_times(self) -> np.ndarray:
        return np.arange(self.horizon) * self.step_duration

    def _combine(self, per_human: np.ndarray, prior: Optional[np.ndarray]) -> np.ndarray:
        """per_human: (N, K, dimy, dimx). Returns (K + 1, dimy, dimx) float32."""
        K = self.horizon
        g = self.grid
        if len(per_human):
            risk = 1.0 - np.prod(1.0 - np.clip(per_human, 0.0, 1.0), axis=0)
        else:
            risk = np.zeros((K, g.dimy, g.dimx))
        last = np.zeros((1, g.dimy, g.dimx)) if prior is None else np.asarray(prior, dtype=np.float64)[None]
        out = np.concatenate([risk, last], axis=0)
        out[:, ~self.free] = 0.0
        return out.astype(np.float32)

    def from_gaussians(self, means: np.ndarray, sigmas: np.ndarray, prior: Optional[np.ndarray] = None) -> np.ndarray:
        g = self.grid
        N, K = means.shape[:2]
        per = np.zeros((N, K, g.dimy, g.dimx))
        if N:
            u, v = _cell_coordinates(g, means[..., 0], means[..., 1])   # (N, K)
            su = np.maximum(sigmas / g.cell_size, 1e-6)
            a = self.inflation / g.cell_size
            ix = np.arange(g.dimx)[None, None, :]
            iv = np.arange(g.dimy)[None, None, :]
            px = ndtr((ix + 0.5 + a - u[..., None]) / su[..., None]) - ndtr((ix - 0.5 - a - u[..., None]) / su[..., None])
            pv = ndtr((iv + 0.5 + a - v[..., None]) / su[..., None]) - ndtr((iv - 0.5 - a - v[..., None]) / su[..., None])
            # pv is indexed by yy (un-inverted); MAPF y = dimy - 1 - yy
            per = pv[..., ::-1, None] * px[..., None, :]
        return self._combine(per, prior)

    def from_particles(self, particles: np.ndarray, prior: Optional[np.ndarray] = None) -> np.ndarray:
        """particles: (N, K, P, 2)."""
        g = self.grid
        N, K, P = particles.shape[:3]
        per = np.zeros((N, K, g.dimy, g.dimx))
        if N:
            u, v = _cell_coordinates(g, particles[..., 0], particles[..., 1])   # (N, K, P)
            a = self.inflation / g.cell_size
            nidx, kidx = np.meshgrid(np.arange(N), np.arange(K), indexing='ij')
            nidx = np.repeat(nidx[..., None], P, axis=2)
            kidx = np.repeat(kidx[..., None], P, axis=2)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cx = np.floor(u + 0.5).astype(np.int64) + dx
                    cy = np.floor(v + 0.5).astype(np.int64) + dy
                    inside = ((np.abs(u - cx) <= 0.5 + a) & (np.abs(v - cy) <= 0.5 + a) &
                              (cx >= 0) & (cx < g.dimx) & (cy >= 0) & (cy < g.dimy))
                    np.add.at(per, (nidx[inside], kidx[inside], g.dimy - 1 - cy[inside], cx[inside]), 1.0 / P)
        return self._combine(per, prior)


class HumanRiskPredictor:
    """Tracker + predictor + rasteriser: the component a coordinator talks to."""

    def __init__(self, grid: GridSpec, step_duration: float = 3.0, horizon: int = 20, inflation: float = 0.3,
                 method: str = 'cv', mod: Optional[MapOfDynamics] = None,
                 walkable: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                 n_particles: int = 96, use_prior: bool = True, prior_blend_time: float = 10.0, seed: int = 0):
        self.grid = grid
        self.tracker = HumanTracker()
        self.builder = RiskMapBuilder(grid, step_duration, horizon, inflation)
        self.method = method
        self.mod = mod
        self.use_prior = use_prior and mod is not None
        self.prior_blend_time = prior_blend_time
        self._prior_cache: Dict[int, np.ndarray] = {}
        if walkable is None:
            def walkable(x, y, _g=grid):
                u, v = _cell_coordinates(_g, x, y)
                cx = np.floor(u + 0.5).astype(np.int64)
                cy = _g.dimy - 1 - np.floor(v + 0.5).astype(np.int64)
                ok = (cx >= 0) & (cx < _g.dimx) & (cy >= 0) & (cy < _g.dimy)
                out = np.zeros(np.shape(cx), dtype=bool)
                out[ok] = self.builder.free[cy[ok], cx[ok]]
                return out
        self.cv = ConstantVelocityPredictor()
        self.particles = ParticleMoDPredictor(mod, walkable, n_particles=n_particles, seed=seed)
        self.last_prediction = None

    @property
    def step_duration(self) -> float:
        return self.builder.step_duration

    @step_duration.setter
    def step_duration(self, value: float):
        self.builder.step_duration = float(value)

    def observe(self, t: float, humans: Iterable[Tuple[object, float, float]]):
        self.tracker.update(t, humans)

    def risk_map(self, t_now: float) -> np.ndarray:
        tracks = self.tracker.tracks(t_now)
        dts = self.builder.layer_times()
        prior = None
        blend = self.use_prior and self.prior_blend_time > 0
        if self.use_prior:
            n = len(tracks)
            if n not in self._prior_cache:
                self._prior_cache[n] = self.mod.occupancy_prior_on_grid(self.grid, n, self.builder.inflation)
            prior = self._prior_cache[n]
        # layers whose blending weight is negligible are pure prior: skip predicting them
        K = len(dts)
        if blend:
            weights = np.exp(-dts / self.prior_blend_time)
            K = max(1, int(np.sum(weights > 0.02)))
        builder = self.builder
        if K < len(dts):
            builder = RiskMapBuilder(self.grid, self.builder.step_duration, K, self.builder.inflation)
        if self.method == 'mod':
            ids, parts = self.particles.predict(tracks, dts[:K])
            self.last_prediction = ('particles', ids, parts)
            risk = builder.from_particles(parts, prior)
        else:
            ids, means, sig = self.cv.predict(tracks, dts[:K])
            self.last_prediction = ('gaussian', ids, means, sig)
            risk = builder.from_gaussians(means, sig, prior)
        if blend:
            a = np.exp(-dts[:K] / self.prior_blend_time).astype(np.float32)[:, None, None]
            risk[:-1] = a * risk[:-1] + (1.0 - a) * risk[-1][None]
            if K < len(dts):   # pad with prior layers so the map always has horizon + 1 layers
                risk = np.concatenate([risk[:-1], np.repeat(risk[-1:], len(dts) - K + 1, axis=0)])
        return risk
