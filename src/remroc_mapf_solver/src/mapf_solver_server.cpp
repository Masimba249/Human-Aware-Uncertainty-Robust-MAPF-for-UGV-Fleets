// Copyright (c) 2024 - for information on the respective copyright owner
// see the NOTICE file or the repository https://github.com/boschresearch/remroc/.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified 2026 by Collins Masimba: the service additionally accepts a
// time-indexed human-risk cost map, a risk weight, reservations (timed paths of
// robots that are not re-planned) and a time limit, and solves such requests with
// the risk-aware CBS in remroc_cbs_library/risk_cbs.hpp. Plain requests (as sent by
// coordinator_mapf.py) are still answered by the original CBS unless the
// "solver" parameter is set to "risk_cbs".


#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "rclcpp/rclcpp.hpp"
#include "remroc_interfaces/srv/remroc_mapfsolver.hpp"
#include "remroc_interfaces/msg/remroc_pose2darray.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "remroc_cbs_library/cbs.hpp"
#include "remroc_cbs_library/risk_cbs.hpp"

namespace rc = remroc_cbs_library::risk_cbs;
using MapfSrv = remroc_interfaces::srv::RemrocMapfsolver;

class RemrocMapfSolverServiceServer : public rclcpp::Node
{
  public:
    RemrocMapfSolverServiceServer() : Node("mapf_solver_server")
    {
      using namespace std::placeholders;
      this -> declare_parameter("mapf_file_path", ".");
      // "auto": original CBS for plain requests, risk-aware CBS when a risk map or
      // reservations are given. "legacy" / "risk_cbs" force one of the two.
      this -> declare_parameter("solver", "auto");
      this -> declare_parameter("default_time_limit", 5.0);
      this -> declare_parameter("cost_scale", 1000);
      srv_ = create_service<MapfSrv>("mapf_solver", std::bind(&RemrocMapfSolverServiceServer::solve, this, _1, _2));
    }

  private:
    rclcpp::Service<MapfSrv>::SharedPtr srv_;
    std::string loaded_grid_path_;
    rc::GridMap grid_;

    const rc::GridMap & grid()
    {
      const std::string path = this->get_parameter("mapf_file_path").as_string();
      if (path != loaded_grid_path_) {
        YAML::Node config = YAML::LoadFile(path);
        const auto & dim = config["map"]["dimensions"];
        grid_ = rc::GridMap(dim[0].as<int>(), dim[1].as<int>());
        for (const auto & node : config["map"]["obstacles"]) {
          grid_.setObstacle(node[0].as<int>(), node[1].as<int>());
        }
        loaded_grid_path_ = path;
        RCLCPP_INFO(get_logger(), "Loaded MAPF grid %dx%d from %s", grid_.dimx(), grid_.dimy(), path.c_str());
      }
      return grid_;
    }

    void solve(std::shared_ptr<MapfSrv::Request> request, std::shared_ptr<MapfSrv::Response> response)
    {
      const std::string solver = this->get_parameter("solver").as_string();
      const bool extended = !request->risk_map.data.empty() || !request->reservations.empty() ||
                            request->risk_weight != 0.0 || request->time_limit > 0.0 ||
                            request->robustness > 0 || request->suboptimality > 1.0;
      if (solver == "legacy" || (solver == "auto" && !extended)) {
        solveLegacy(request, response);
      } else {
        solveRiskAware(request, response);
      }
    }

    // The original REMROC behaviour, unchanged.
    void solveLegacy(std::shared_ptr<MapfSrv::Request> request, std::shared_ptr<MapfSrv::Response> response)
    {
      // Define the input vectors

      std::unordered_map<int, std::vector<std::vector<int>>> input_map;
      std::vector<std::vector<int>> robot_vector;
      std::string mapf_file_path = this->get_parameter("mapf_file_path").as_string();

      int my_x;
      int my_y;
      int id;

      for (const auto& pose_vec : request->poses) {
        id = pose_vec.id;

        for (const auto& pose : pose_vec.poses) {
          my_x = std::round(pose.x);
          my_y = std::round(pose.y);
          robot_vector.push_back({my_x, my_y});
        }

        input_map[id] = robot_vector;
        robot_vector.clear();
      }

      std::unordered_map<int, std::vector<std::vector<int>>> my_robots_waypoint_array = remroc_cbs_library::get_robots_waypoint_array(mapf_file_path, input_map);

      remroc_interfaces::msg::RemrocPose2darray robot_array;

      geometry_msgs::msg::Pose2D point;

      for (auto it = my_robots_waypoint_array.begin(); it != my_robots_waypoint_array.end(); ++it) {
        robot_array.id = it->first;

        for (const auto& pose : it->second) {
          point.x = pose[0];
          point.y = pose[1];
          robot_array.poses.emplace_back(point);
        }

        response->robots_waypoint_array.emplace_back(robot_array);
        robot_array.poses = {};
      }
      response->success = !my_robots_waypoint_array.empty();
      response->status = response->success ? "success" : "no_solution";
    }

    void solveRiskAware(std::shared_ptr<MapfSrv::Request> request, std::shared_ptr<MapfSrv::Response> response)
    {
      const rc::GridMap & g = grid();

      std::vector<rc::Agent> agents;
      std::vector<int> ids;
      for (const auto & pose_vec : request->poses) {
        if (pose_vec.poses.size() < 2) {
          response->success = false;
          response->status = "malformed_request";
          return;
        }
        rc::Agent a;
        a.start = rc::Cell{static_cast<int>(std::lround(pose_vec.poses[0].x)), static_cast<int>(std::lround(pose_vec.poses[0].y))};
        a.goal = rc::Cell{static_cast<int>(std::lround(pose_vec.poses[1].x)), static_cast<int>(std::lround(pose_vec.poses[1].y))};
        agents.push_back(a);
        ids.push_back(pose_vec.id);
      }

      std::vector<rc::Reservation> reservations;
      for (const auto & r : request->reservations) {
        rc::Reservation res;
        for (const auto & p : r.poses) {
          res.path.push_back(rc::Cell{static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y))});
        }
        if (!res.path.empty()) {reservations.push_back(std::move(res));}
      }

      rc::RiskMap risk;
      const auto & rm = request->risk_map;
      if (!rm.data.empty()) {
        if (static_cast<int>(rm.width) != g.dimx() || static_cast<int>(rm.height) != g.dimy()) {
          RCLCPP_WARN(get_logger(), "Risk map is %ux%u but the MAPF grid is %dx%d; ignoring the risk map.",
            rm.width, rm.height, g.dimx(), g.dimy());
        } else {
          risk = rc::RiskMap(g.dimx(), g.dimy(), static_cast<int>(rm.horizon),
            std::vector<float>(rm.data.begin(), rm.data.end()));
          if (risk.empty()) {
            RCLCPP_WARN(get_logger(), "Risk map data size does not match width*height*horizon; ignoring it.");
          }
        }
      }

      rc::Options options;
      options.risk_weight = request->risk_weight;
      options.robustness = request->robustness;
      options.suboptimality = std::max(1.0, request->suboptimality);
      options.cost_scale = this->get_parameter("cost_scale").as_int();
      options.time_limit_s = request->time_limit > 0.0 ? request->time_limit :
                             this->get_parameter("default_time_limit").as_double();

      const rc::Result res = rc::solve(g, agents, risk, reservations, options);

      response->success = res.success;
      response->status = res.status;
      response->objective = res.objective;
      response->sum_of_costs = static_cast<uint32_t>(res.sum_of_steps);
      response->makespan = static_cast<uint32_t>(res.makespan);
      response->total_risk = res.total_risk;
      response->solve_time = res.runtime_s;
      response->high_level_expanded = static_cast<uint32_t>(res.high_level_expanded);
      for (size_t i = 0; i < res.paths.size(); ++i) {
        remroc_interfaces::msg::RemrocPose2darray arr;
        arr.id = ids[i];
        for (const auto & c : res.paths[i]) {
          geometry_msgs::msg::Pose2D p;
          p.x = c.x;
          p.y = c.y;
          arr.poses.push_back(p);
        }
        response->robots_waypoint_array.push_back(arr);
      }
      RCLCPP_INFO(get_logger(), "risk-aware CBS: %s, %zu agents, %zu reservations, lambda=%.2f, J=%.2f, SoC=%d, risk=%.3f, %.3fs",
        res.status.c_str(), agents.size(), reservations.size(), options.risk_weight, res.objective,
        res.sum_of_steps, res.total_risk, res.runtime_s);
    }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = std::make_shared<RemrocMapfSolverServiceServer>();

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Launched MAPF solver server (legacy CBS + risk-aware CBS).");

  rclcpp::spin(node);
  rclcpp::shutdown();
}
