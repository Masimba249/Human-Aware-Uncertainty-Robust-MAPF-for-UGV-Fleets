# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
# Lets `pytest` run from the repository root without building the ROS workspace.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for pkg in ('remroc_ha', 'remroc_world_generation'):
    p = str(ROOT / 'src' / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)
