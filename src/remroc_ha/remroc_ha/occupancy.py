# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
"""Nav2-style occupancy maps (``.pgm`` + ``.yaml``) without ROS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


def read_pgm(path) -> np.ndarray:
    """Reads binary (P5) or ASCII (P2) PGM files, tolerating comment lines."""
    data = Path(path).read_bytes()
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic == b'P5':
        i += 1
        dtype = np.uint8 if maxval < 256 else np.dtype('>u2')
        img = np.frombuffer(data[i:i + w * h * np.dtype(dtype).itemsize], dtype=dtype)
    elif magic == b'P2':
        img = np.array(data[i:].split()[:w * h], dtype=np.int64)
    else:
        raise ValueError(f'unsupported PGM type {magic!r}')
    return img.reshape(h, w)


def write_pgm(path, img: np.ndarray):
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape
    with Path(path).open('wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())


@dataclass
class OccupancyMap:
    """free[row, col] with row 0 at the top of the image (as stored in the PGM)."""
    free: np.ndarray
    resolution: float
    origin: Tuple[float, float]

    @classmethod
    def from_yaml(cls, yaml_path) -> 'OccupancyMap':
        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text())
        img = read_pgm(yaml_path.parent / meta['image']).astype(np.float64)
        if int(meta.get('negate', 0)):
            img = 255.0 - img
        occ = (255.0 - img) / 255.0
        free = occ < float(meta.get('free_thresh', 0.196))
        return cls(free, float(meta['resolution']), (float(meta['origin'][0]), float(meta['origin'][1])))

    @property
    def height(self) -> int:
        return self.free.shape[0]

    @property
    def width(self) -> int:
        return self.free.shape[1]

    def world_to_pixel(self, x, y):
        col = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(np.int64)
        row = self.height - 1 - np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(np.int64)
        return row, col

    def pixel_to_world(self, row, col):
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin[1] + (self.height - 1 - np.asarray(row) + 0.5) * self.resolution
        return x, y

    def is_free(self, x, y) -> np.ndarray:
        row, col = self.world_to_pixel(x, y)
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        out = np.zeros(np.shape(row), dtype=bool)
        out[inside] = self.free[row[inside], col[inside]]
        return out

    def inflated(self, radius: float) -> 'OccupancyMap':
        """Free space eroded by ``radius`` metres (obstacles dilated)."""
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(self.free) * self.resolution
        return OccupancyMap(dist > radius, self.resolution, self.origin)

    def bounds(self):
        return (self.origin[0], self.origin[1],
                self.origin[0] + self.width * self.resolution, self.origin[1] + self.height * self.resolution)
