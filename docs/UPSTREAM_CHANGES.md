# Changes relative to upstream REMROC

`src/` contains the REMROC packages from
[boschresearch/remroc](https://github.com/boschresearch/remroc) at commit
`a2159cbc3c92a50dd505905ae8adc17ec75732d5` (Apache-2.0; see `third_party/remroc/`),
with the modifications listed here. The pre-generated human worlds of upstream
(`remroc/worlds/sdfs/{simple,depot}_{N}_{sample}.sdf`, ~30 MB) are not vendored. Only
the empty base worlds `depot_0_0.sdf` and `simple_0_0.sdf` are kept. Copy the others
from upstream to reproduce the original paper's scenarios.

| Package | File | Change |
|---|---|---|
| remroc_interfaces | `msg/RemrocRiskMap.msg` | **new**: time-indexed human-occupancy map on the MAPF grid |
| remroc_interfaces | `srv/RemrocMapfsolver.srv` | request gains `risk_map`, `risk_weight`, `reservations`, `time_limit`, `robustness`, `suboptimality`; response gains `success`, `status`, `objective`, `sum_of_costs`, `makespan`, `total_risk`, `solve_time`, `high_level_expanded`. Existing fields are unchanged, so `coordinator_mapf.py` works as before. |
| remroc_cbs_library | `include/remroc_cbs_library/risk_cbs.hpp` | **new**: risk-aware, k-robust, bounded-suboptimal CBS with reservations and a time limit (header-only, standard library only) |
| remroc_cbs_library | `src/risk_cbs_cli.cpp`, `test/test_risk_cbs.cpp` | **new**: command-line front end and unit tests |
| remroc_cbs_library | `CMakeLists.txt`, `package.xml` | build and install the CLI and tests; plain-CMake fallback without ament; declare yaml-cpp |
| remroc_mapf_solver | `src/mapf_solver_server.cpp` | answers extended requests with `risk_cbs`. Plain requests still use the original CBS (parameter `solver`: `auto`/`legacy`/`risk_cbs`). New parameters: `default_time_limit`, `cost_scale`. |
| remroc | `remroc/coordinator_ha_cbs.py` | **new** coordinator (based on `coordinator_template.py`) |
| remroc | `remroc/human_pose_publisher.py`, `remroc/metrics_recorder.py` | **new** nodes |
| remroc | `launch/ignition.launch.py` | coordinator, experiment yaml, `use_depot_mod`, results dir, time limit and human-aware settings are launch arguments; starts the solver for `coordinator_ha_cbs`, the human pose publisher and the metrics recorder; shuts down when the recorder exits |
| remroc | `remroc/coordinator_mapf.py` | reads the grid metadata from the `grid:` block of `mapf_<map>.yaml` for maps it does not know (configuration only; the algorithm is unchanged) |
| remroc | `setup.py`, `package.xml` | entry points for `coordinator_pbc` (missing upstream), `coordinator_ha_cbs`, `human_pose_publisher`, `metrics_recorder`; removed `coordinator_oru` (module does not exist) |
| remroc | `params/exp_ha_*.yaml`, `params/mapf_narrow_corridors.yaml`, `worlds/**` | **new** 5-robot experiments, corridor layout, generated worlds (0/5/10/20 humans × 5 samples), human trajectories, maps of dynamics |
| remroc_world_generation | `layouts.py`, `human_trajectories.py`, `generate_worlds.py` | **new**: procedural layouts and ROS-free batch world generation for any number of humans and samples (the nav2-based launch files are unchanged) |
| remroc_ha | whole package | **new**: prediction, ADG execution and repair, reference solver, metrics, simulator |
