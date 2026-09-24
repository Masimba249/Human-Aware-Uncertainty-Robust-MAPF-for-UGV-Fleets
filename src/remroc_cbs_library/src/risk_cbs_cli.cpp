// Copyright (c) 2026 Collins Masimba
// SPDX-License-Identifier: Apache-2.0
//
// Command line front end for the risk-aware CBS solver. Reads a whitespace
// separated instance from stdin and writes the solution to stdout. Used by the
// ROS-free simulator and by the cross-validation tests. Format:
//
//   grid <dimx> <dimy>
//   obstacles <n> <x> <y> ...
//   agents <n> <sx> <sy> <gx> <gy> ...
//   risk <horizon> <horizon*dimx*dimy floats, index (t*dimy + y)*dimx + x>   (optional)
//   lambda <risk weight>                                                     (optional)
//   reservations <n> { <len> <x> <y> ... }                                   (optional)
//   time_limit <seconds>                                                     (optional)
//   scale <fixed point units per step>                                       (optional)
//   robust <k>                                                               (optional)
//   subopt <w >= 1>                                                          (optional)
//   end
//
// Output:
//   status <status>
//   stats <objective> <objective_scaled> <sum_of_steps> <makespan> <total_risk> <hl> <ll> <runtime_s>
//   path <agent> <len> <x> <y> ...

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "remroc_cbs_library/risk_cbs.hpp"

namespace rc = remroc_cbs_library::risk_cbs;

int main()
{
  std::ios::sync_with_stdio(false);
  rc::GridMap grid;
  std::vector<rc::Agent> agents;
  std::vector<rc::Reservation> reservations;
  rc::RiskMap risk;
  rc::Options options;
  bool have_grid = false;

  std::string key;
  while (std::cin >> key) {
    if (key == "end") {
      break;
    } else if (key == "grid") {
      int dx = 0;
      int dy = 0;
      std::cin >> dx >> dy;
      grid = rc::GridMap(dx, dy);
      have_grid = true;
    } else if (key == "obstacles") {
      int n = 0;
      std::cin >> n;
      for (int i = 0; i < n; ++i) {
        int x = 0;
        int y = 0;
        std::cin >> x >> y;
        grid.setObstacle(x, y);
      }
    } else if (key == "agents") {
      int n = 0;
      std::cin >> n;
      agents.resize(n);
      for (auto & a : agents) {std::cin >> a.start.x >> a.start.y >> a.goal.x >> a.goal.y;}
    } else if (key == "risk") {
      int horizon = 0;
      std::cin >> horizon;
      std::vector<float> data(static_cast<size_t>(horizon) * grid.dimx() * grid.dimy());
      for (auto & v : data) {std::cin >> v;}
      risk = rc::RiskMap(grid.dimx(), grid.dimy(), horizon, std::move(data));
    } else if (key == "lambda") {
      std::cin >> options.risk_weight;
    } else if (key == "reservations") {
      int n = 0;
      std::cin >> n;
      reservations.resize(n);
      for (auto & r : reservations) {
        int len = 0;
        std::cin >> len;
        r.path.resize(len);
        for (auto & c : r.path) {std::cin >> c.x >> c.y;}
      }
    } else if (key == "time_limit") {
      std::cin >> options.time_limit_s;
    } else if (key == "scale") {
      std::cin >> options.cost_scale;
    } else if (key == "robust") {
      std::cin >> options.robustness;
    } else if (key == "subopt") {
      std::cin >> options.suboptimality;
    } else {
      std::cout << "status parse_error_" << key << "\n";
      return 2;
    }
    if (!std::cin) {
      std::cout << "status parse_error_" << key << "\n";
      return 2;
    }
  }
  if (!have_grid) {
    std::cout << "status parse_error_missing_grid\n";
    return 2;
  }

  const rc::Result res = rc::solve(grid, agents, risk, reservations, options);
  std::cout << "status " << res.status << "\n";
  std::cout << std::setprecision(10) << "stats " << res.objective << " " << res.objective_scaled << " "
            << res.sum_of_steps << " " << res.makespan << " " << res.total_risk << " "
            << res.high_level_expanded << " " << res.low_level_expanded << " " << res.runtime_s << "\n";
  for (size_t i = 0; i < res.paths.size(); ++i) {
    std::cout << "path " << i << " " << res.paths[i].size();
    for (const auto & c : res.paths[i]) {std::cout << " " << c.x << " " << c.y;}
    std::cout << "\n";
  }
  return res.success ? 0 : 1;
}
