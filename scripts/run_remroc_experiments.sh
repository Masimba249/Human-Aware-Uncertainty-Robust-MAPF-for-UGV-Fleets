#!/usr/bin/env bash
# Batch of REMROC (Gazebo Fortress + nav2) experiments: coordinators x worlds x humans x samples.
# Run from a sourced ROS 2 Humble workspace in which this repository was built:
#
#   source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash
#   scripts/run_remroc_experiments.sh                      # full grid (long!)
#   COORDINATORS="coordinator_ha_cbs" WORLDS="depot" HUMANS="10" SAMPLES="0" scripts/run_remroc_experiments.sh
#
# Each episode ends when all robots are settled at their goals or at TIME_LIMIT (the
# metrics_recorder exits and the launch file shuts the simulation down). A wall-clock
# guard kills runs that hang. Results: $RESULTS/<coordinator>/<world>/<humans>/<sample>_metrics.json
# Aggregate with:  python -m remroc_ha.analysis ros $RESULTS
set -uo pipefail
COORDINATORS="${COORDINATORS:-coordinator_mapf coordinator_pbc coordinator_ha_cbs}"
WORLDS="${WORLDS:-narrow_corridors depot}"
HUMANS="${HUMANS:-0 5 10 20}"
SAMPLES="${SAMPLES:-0 1 2 3 4}"
RESULTS="${RESULTS:-$PWD/results/remroc}"
TIME_LIMIT="${TIME_LIMIT:-600.0}"
WALL_GUARD="${WALL_GUARD:-1500}"   # seconds
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$RESULTS/logs"
for c in $COORDINATORS; do
  for w in $WORLDS; do
    for n in $HUMANS; do
      for s in $SAMPLES; do
        name="${c#coordinator_}"
        if [ -f "$RESULTS/$name/$w/$n/${s}_metrics.json" ]; then
          echo "skip $name $w $n $s (done)"; continue
        fi
        echo "=== $c world=$w humans=$n sample=$s"
        timeout --signal=INT "$WALL_GUARD" ros2 launch remroc ignition.launch.py \
          coordinator:="$c" world:="$w" number_of_humans:="$n" sample:="$s" \
          results_dir:="$RESULTS" time_limit:="$TIME_LIMIT" $EXTRA_ARGS \
          > "$RESULTS/logs/${name}_${w}_${n}_${s}.log" 2>&1
        # make sure nothing of the previous run survives
        pkill -f "ign gazebo" 2>/dev/null; pkill -f ros2 2>/dev/null; sleep 5
      done
    done
  done
done
python3 -m remroc_ha.analysis ros "$RESULTS"
