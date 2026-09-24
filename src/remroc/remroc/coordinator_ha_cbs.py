# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
#
# Based on coordinator_template.py of REMROC
# (Copyright (c) 2024 - for information on the respective copyright owner see the
# NOTICE file or the repository https://github.com/boschresearch/remroc/, Apache-2.0).

'''
Human-aware, uncertainty-robust MAPF coordinator ("ha_cbs").

* subscribes to the robot states (as the template) and to the human poses
  (/humans/poses, published by human_pose_publisher from the Gazebo world),
* predicts human occupancy and builds a time-indexed risk map on the MAPF grid,
* plans with the risk-aware, 1-robust, bounded-suboptimal CBS of the extended
  mapf_solver service (the risk map is sent with every request and also published
  on /ha_coordinator/risk_map),
* executes the plan through an Action Dependency Graph (robots only receive the
  actions whose inter-robot dependencies are completed) and repairs it locally when a
  robot is blocked or its next cells are predicted to be occupied by humans.

The decision logic is remroc_ha.coordination.HumanAwareCoordinator, which is shared
with the ROS-free simulator; this node only adapts it to ROS 2 / nav2.
'''

import json
import math
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, PoseArray, PoseStamped, Twist
from nav2_msgs.action import NavigateThroughPoses
from nav_msgs.msg import Odometry
from remroc_interfaces.msg import RemrocPose2darray, RemrocRiskMap
from remroc_interfaces.srv import RemrocMapfsolver
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from remroc_ha.coordination import CoordinatorParams, HumanAwareCoordinator, RobotObs
from remroc_ha.grid import GridSpec
from remroc_ha.occupancy import OccupancyMap
from remroc_ha.prediction import HumanRiskPredictor, MapOfDynamics
from remroc_ha.risk_cbs import SolveResult
from remroc_ha.solver import SolverBackend


class RosServiceSolver(SolverBackend):
    '''SolverBackend that calls the (extended) remroc mapf_solver service.'''

    name = 'ros_service'

    def __init__(self, node, client, risk_pub=None):
        self.node = node
        self.client = client
        self.risk_pub = risk_pub
        self.step_duration = 0.0

    def solve(self, grid, agents, risk=None, risk_weight=0.0, reservations=(), robustness=0, time_limit=0.0,
              suboptimality=1.0):
        req = RemrocMapfsolver.Request()
        for i, (s, g) in enumerate(agents):
            arr = RemrocPose2darray()
            arr.id = i
            arr.poses = [Pose2D(x=float(s[0]), y=float(s[1])), Pose2D(x=float(g[0]), y=float(g[1]))]
            req.poses.append(arr)
        if risk is not None and risk.size:
            rm = RemrocRiskMap()
            rm.header.stamp = self.node.get_clock().now().to_msg()
            rm.header.frame_id = 'mapf_grid'
            rm.step_duration = float(self.step_duration)
            rm.horizon, rm.height, rm.width = (int(v) for v in risk.shape)
            rm.data = np.asarray(risk, dtype=np.float32).ravel().tolist()
            req.risk_map = rm
            if self.risk_pub is not None:
                self.risk_pub.publish(rm)
        req.risk_weight = float(risk_weight)
        for i, path in enumerate(reservations):
            arr = RemrocPose2darray()
            arr.id = 1000 + i
            arr.poses = [Pose2D(x=float(c[0]), y=float(c[1])) for c in path]
            req.reservations.append(arr)
        req.time_limit = float(time_limit)
        req.robustness = int(robustness)
        req.suboptimality = float(suboptimality)
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=(time_limit or 5.0) + 5.0)
        resp = future.result()
        if resp is None:
            return SolveResult(False, 'service_timeout')
        paths = {a.id: [(int(round(p.x)), int(round(p.y))) for p in a.poses] for a in resp.robots_waypoint_array}
        return SolveResult(resp.success, resp.status, [paths[i] for i in sorted(paths)] if resp.success else [],
                           resp.objective, int(round(resp.objective * 1000)), resp.sum_of_costs, resp.makespan,
                           resp.total_risk, resp.high_level_expanded, 0, resp.solve_time)


class ExperimentAsync(Node):

    def __init__(self):
        super().__init__('experiment_async')

        self.declare_parameter('map_name', 'none')
        self.map_name = self.get_parameter('map_name').value
        self.declare_parameter('number_of_humans', 0)
        self.number_of_humans = self.get_parameter('number_of_humans').value
        self.declare_parameter('sample', 0)
        self.sample = self.get_parameter('sample').value
        self.declare_parameter('experiment_yaml', 'exp_ha_depot.yaml')
        self.experiment_yaml = self.get_parameter('experiment_yaml').value
        # human-aware coordination parameters
        self.declare_parameter('risk_weight', 4.0)
        self.declare_parameter('robustness', 1)
        self.declare_parameter('suboptimality', 1.1)
        self.declare_parameter('use_prediction', True)
        self.declare_parameter('prediction', 'cv')          # 'cv' or 'mod'
        self.declare_parameter('enable_repair', True)
        self.declare_parameter('mod_file', '')              # default: worlds/humans/<map>_mod.npz
        self.declare_parameter('loop_period', 0.5)
        self.declare_parameter('time_limit', 600.0)
        self.declare_parameter('results_dir', 'results')

        # make the node use Simtime
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.pkg_remroc = get_package_share_directory('remroc')
        self.node_start_time = self.get_clock().now()
        self.node_current_time = self.get_clock().now()

        with open(Path(self.pkg_remroc).joinpath('params', self.experiment_yaml), 'r') as file:
            experiment_dict = yaml.safe_load(file)
        self.robots_goal_dict = {robot['type'] + '_' + str(robot['id']): robot['goal'] for robot in experiment_dict['robots']}
        self.robot_names = [robot['type'] + '_' + str(robot['id']) for robot in experiment_dict['robots']]

        self.robot_states = {robot_name: None for robot_name in self.robot_names}
        self.humans = []
        self.result_dict = {
            'time_to_goal': {robot_name: None for robot_name in self.robot_names},
            'min_range': {robot_name: [] for robot_name in self.robot_names},
            'odometry': {robot_name: [] for robot_name in self.robot_names},
            'coordinator_loop_time': [],
        }

        for robot_name in self.robot_names:
            self.create_subscription(Odometry, f'{robot_name}/robot_state',
                                     lambda msg, n=robot_name: self.state_subscriber_callback(msg, n), 10)
            self.create_subscription(Odometry, f'{robot_name}/odometry/filtered',
                                     lambda msg, n=robot_name: self.odom_subscriber_callback(msg, n), 10)
            self.create_subscription(LaserScan, f'{robot_name}/laser_scan',
                                     lambda msg, n=robot_name: self.laser_subscriber_callback(msg, n), 10)
        self.create_subscription(PoseArray, '/humans/poses', self.human_callback, 10)

        self.action_client_dict = {n: ActionClient(self, NavigateThroughPoses, f'{n}/navigate_through_poses')
                                   for n in self.robot_names}
        self.goal_handles = {n: None for n in self.robot_names}
        self.cmd_vel_pub_dict = {n: self.create_publisher(Twist, f'{n}/cmd_vel', 10) for n in self.robot_names}

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.risk_pub = self.create_publisher(RemrocRiskMap, '/ha_coordinator/risk_map', 1)
        self.event_pub = self.create_publisher(String, '/coordinator/events', 50)
        self.stats_pub = self.create_publisher(String, '/coordinator/stats', latched)

        self.cli = self.create_client(RemrocMapfsolver, 'mapf_solver')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for mapf_solver_server service to become available.')
        self.solver = RosServiceSolver(self, self.cli, self.risk_pub)

        # MAPF grid (with its metric embedding) and goal cells
        self.grid = GridSpec.from_mapf_yaml(Path(self.pkg_remroc) / 'params' / f'mapf_{self.map_name}.yaml',
                                            self.map_name)
        goal_cells = self.grid.assign_unique_cells([(n, tuple(self.robots_goal_dict[n][:2])) for n in self.robot_names])
        self.goal_cells = goal_cells

        gp = self.get_parameter
        params = CoordinatorParams(risk_weight=gp('risk_weight').value, robustness=gp('robustness').value,
                                   suboptimality=gp('suboptimality').value,
                                   use_prediction=gp('use_prediction').value,
                                   enable_repair=gp('enable_repair').value)
        predictor = None
        if params.use_prediction:
            mod = None
            mod_file = gp('mod_file').value or str(Path(self.pkg_remroc) / 'worlds' / 'humans' / f'{self.map_name}_mod.npz')
            if Path(mod_file).exists():
                mod = MapOfDynamics.load(mod_file)
            else:
                self.get_logger().warn(f'No map of dynamics at {mod_file}: prediction without static prior.')
            walkable = None
            map_yaml = Path(self.pkg_remroc) / 'worlds' / 'maps' / f'{self.map_name}.yaml'
            if gp('prediction').value == 'mod' and map_yaml.exists():
                walkable = OccupancyMap.from_yaml(map_yaml).is_free
            predictor = HumanRiskPredictor(self.grid, params.step_duration, 20, method=gp('prediction').value,
                                           mod=mod, walkable=walkable)
        self.coordinator = HumanAwareCoordinator(self.grid, goal_cells, self.solver, params, predictor,
                                                 log=lambda s: self.get_logger().info(s))
        self._published_events = 0

    # ---------------------------------------------------------------- callbacks (as the template)
    def state_subscriber_callback(self, msg, robot_name):
        self.robot_states[robot_name] = msg

    def odom_subscriber_callback(self, msg, robot_name):
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lv, av = msg.twist.twist.linear, msg.twist.twist.angular
        self.result_dict['odometry'][robot_name].append(
            {'time_stamp': time_stamp, 'linear_vel': (lv.x, lv.y), 'angular_vel': (av.x, av.y, av.z)})

    def laser_subscriber_callback(self, msg, robot_name):
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.result_dict['min_range'][robot_name].append({'time_stamp': time_stamp, 'min_range': min(msg.ranges)})

    def human_callback(self, msg):
        self.humans = [(i, p.position.x, p.position.y) for i, p in enumerate(msg.poses)]

    # ---------------------------------------------------------------- coordination
    def now_sec(self) -> float:
        t = (self.get_clock().now() - self.node_start_time).to_msg()
        return t.sec + t.nanosec * 1e-9

    def observations(self):
        obs = {}
        for n, st in self.robot_states.items():
            tw = st.twist.twist.linear
            obs[n] = RobotObs(st.pose.pose.position.x, st.pose.pose.position.y, math.hypot(tw.x, tw.y))
        return obs

    def coordinate(self):
        self.solver.step_duration = self.coordinator.step_duration
        commands = self.coordinator.step(self.now_sec(), self.observations(), self.humans)
        for robot_name, cells in commands.items():
            self.send_path(robot_name, cells)
        for e in self.coordinator.events[self._published_events:]:
            self.event_pub.publish(String(data=json.dumps(e)))
        self._published_events = len(self.coordinator.events)

    def send_path(self, robot_name, cells):
        if not cells:
            # hold: cancel the current navigation goal and stop
            if self.goal_handles[robot_name] is not None:
                self.goal_handles[robot_name].cancel_goal_async()
                self.goal_handles[robot_name] = None
            self.cmd_vel_pub_dict[robot_name].publish(Twist())
            return
        st = self.robot_states[robot_name].pose.pose.position
        pts = [(st.x, st.y)] + [self.grid.centroid(c) for c in cells]
        goal_msg = NavigateThroughPoses.Goal()
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x, ps.pose.position.y = float(x1), float(y1)
            q = Rotation.from_euler('z', math.atan2(y1 - y0, x1 - x0)).as_quat()
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
            goal_msg.poses.append(ps)
        client = self.action_client_dict[robot_name]
        client.wait_for_server()
        future = client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f, n=robot_name: self._goal_response(f, n))

    def _goal_response(self, future, robot_name):
        handle = future.result()
        if handle is not None and handle.accepted:
            self.goal_handles[robot_name] = handle

    def check_robots_at_goal(self) -> bool:
        self.node_current_time = self.get_clock().now()
        all_at_goal = True
        for n, st in self.robot_states.items():
            gx, gy = self.grid.centroid(self.goal_cells[n])
            if math.hypot(st.pose.pose.position.x - gx, st.pose.pose.position.y - gy) < 0.5:
                if self.result_dict['time_to_goal'][n] is None:
                    self.result_dict['time_to_goal'][n] = self.now_sec()
            else:
                all_at_goal = False
                self.result_dict['time_to_goal'][n] = None
        return all_at_goal and self.coordinator.finished()

    def stats(self) -> dict:
        s = self.coordinator.stats
        return {'replans': s.replans, 'solver_calls': s.solver_calls, 'solver_time': s.solver_time,
                'local_repairs': s.local_repairs, 'coupled_repairs': s.coupled_repairs,
                'global_replans': s.global_replans, 'repair_failures': s.repair_failures,
                'rejected_repairs': s.rejected_repairs, 'kept_plans': s.kept_plans,
                'blocked_triggers': s.blocked_triggers, 'risk_triggers': s.risk_triggers,
                'step_duration': self.coordinator.step_duration}


def main():
    rclpy.init()
    experiment = ExperimentAsync()
    experiment.node_start_time = experiment.get_clock().now()
    period = rclpy.duration.Duration(seconds=experiment.get_parameter('loop_period').value)
    time_limit = experiment.get_parameter('time_limit').value
    loop_start_time = experiment.get_clock().now()

    robots_at_goal = False
    while not robots_at_goal:
        if experiment.get_clock().now() - loop_start_time < period:
            rclpy.spin_once(experiment, timeout_sec=0.05)
            continue
        loop_start_time = experiment.get_clock().now()
        # receive all robots' current state (as in the template)
        all_states_recieved = False
        while not all_states_recieved:
            rclpy.spin_once(experiment)
            all_states_recieved = all(s is not None for s in experiment.robot_states.values())

        experiment.coordinate()
        robots_at_goal = experiment.check_robots_at_goal()
        if experiment.now_sec() >= time_limit:
            robots_at_goal = True
        experiment.stats_pub.publish(String(data=json.dumps(experiment.stats())))

        t = (experiment.get_clock().now() - loop_start_time).to_msg()
        experiment.result_dict['coordinator_loop_time'].append(t.sec + t.nanosec * 1e-9)
        experiment.robot_states = {n: None for n in experiment.robot_names}

    experiment.result_dict['coordinator_stats'] = experiment.stats()
    experiment.result_dict['events'] = experiment.coordinator.events
    results_dir = experiment.get_parameter('results_dir').value
    target_path = Path(f'{results_dir}/ha_cbs/{experiment.map_name}/{experiment.number_of_humans}/{experiment.sample}.json')
    target_path.parent.mkdir(exist_ok=True, parents=True)
    with target_path.open('w') as result_file:
        json.dump(experiment.result_dict, result_file, indent=4, default=str)

    experiment.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
