#!/usr/bin/env bash
# Builds the ROS-free risk-aware CBS command line tool and its unit tests into ./build.
# (Inside a colcon workspace the same targets are built by `colcon build`.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-g++}"
mkdir -p "$ROOT/build"
INC="$ROOT/src/remroc_cbs_library/include"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic -I"$INC" "$ROOT/src/remroc_cbs_library/src/risk_cbs_cli.cpp" -o "$ROOT/build/risk_cbs_cli"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic -I"$INC" "$ROOT/src/remroc_cbs_library/test/test_risk_cbs.cpp" -o "$ROOT/build/test_risk_cbs"
echo "built $ROOT/build/risk_cbs_cli and $ROOT/build/test_risk_cbs"
