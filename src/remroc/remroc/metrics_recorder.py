# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
'''
Coordinator-independent recorder of an experiment episode.

Records robot poses/speeds (<robot>/robot_state), human positions (/humans/poses) and
coordinator events/stats (/coordinator/events, /coordinator/stats; published by
coordinator_ha_cbs, absent for the baselines) at a fixed rate, and writes

    <results_dir>/<coordinator>/<world>/<humans>/<sample>_log.json.gz
    <results_dir>/<coordinator>/<world>/<humans>/<sample>_metrics.json

using remroc_ha.metrics.compute_metrics, so every coordinator is evaluated with the
same definitions as in the simulator. The node exits when all robots have been at
their goals and stopped for 'settle_time' seconds, or at 'time_limit'; the launch file
shuts the simulation down when it exits.
'''

import gzip
import json
import math
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from remroc_ha.metrics import compute_metrics


class MetricsRecorder(Node):

    def __init__(self):
        super().__init__('metrics_recorder')
        self.declare_parameter('coordinator', 'unknown')
        self.declare_parameter('map_name', 'depot')
        self.declare_parameter('number_of_humans', 0)
        self.declare_parameter('sample', 0)
        self.declare_parameter('experiment_yaml', 'exp_ha_depot.yaml')
        self.declare_parameter('results_dir', 'results')
        self.declare_parameter('rate', 5.0)
        self.declare_parameter('time_limit', 600.0)
        self.declare_parameter('settle_time', 3.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        with open(Path(get_package_share_directory('remroc')) / 'params' / gp('experiment_yaml')) as f:
            exp = yaml.safe_load(f)
        self.names = [f"{r['type']}_{r['id']}" for r in exp['robots']]
        self.goals = {f"{r['type']}_{r['id']}": [float(v) for v in r['goal'][:2]] for r in exp['robots']}
        self.state = {n: None for n in self.names}
        self.humans = []
        self.events = []
        self.stats = {}
        self.frames = []
        self.t0 = None
        self.settled_since = None
        self.done = False

        for n in self.names:
            self.create_subscription(Odometry, f'{n}/robot_state', lambda m, n=n: self.state.__setitem__(n, m), 10)
        self.create_subscription(PoseArray, '/humans/poses', self.on_humans, 10)
        self.create_subscription(String, '/coordinator/events', lambda m: self.events.append(json.loads(m.data)), 50)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/coordinator/stats', lambda m: self.stats.update(json.loads(m.data)), latched)
        self.create_timer(1.0 / gp('rate'), self.record)

    def on_humans(self, msg):
        self.humans = [[p.position.x, p.position.y] for p in msg.poses]

    def record(self):
        if any(s is None for s in self.state.values()) or self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        robots = {}
        at_goal = True
        for n, s in self.state.items():
            x, y = s.pose.pose.position.x, s.pose.pose.position.y
            v = math.hypot(s.twist.twist.linear.x, s.twist.twist.linear.y)
            robots[n] = [round(x, 3), round(y, 3), round(v, 3)]
            gx, gy = self.goals[n]
            at_goal = at_goal and math.hypot(x - gx, y - gy) < self.get_parameter('goal_tolerance').value and v < 0.05
        self.frames.append({'t': round(t, 3), 'robots': robots, 'humans': [list(map(float, h)) for h in self.humans]})
        if at_goal:
            self.settled_since = t if self.settled_since is None else self.settled_since
        else:
            self.settled_since = None
        if (self.settled_since is not None and t - self.settled_since >= self.get_parameter('settle_time').value) \
                or t >= self.get_parameter('time_limit').value:
            self.finish()

    def finish(self):
        self.done = True
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        log = {'meta': {'coordinator': gp('coordinator'), 'world': gp('map_name'),
                        'number_of_humans': gp('number_of_humans'), 'sample': gp('sample'),
                        'robots': self.names, 'goals': self.goals, 'time_limit': gp('time_limit'),
                        'simulator': 'REMROC (Gazebo Fortress + nav2)'},
               'frames': self.frames, 'events': self.events, 'coordinator_stats': self.stats}
        metrics = compute_metrics(log)
        out = Path(gp('results_dir')) / gp('coordinator') / gp('map_name') / str(gp('number_of_humans'))
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / f"{gp('sample')}_log.json.gz", 'wt') as f:
            json.dump(log, f)
        (out / f"{gp('sample')}_metrics.json").write_text(json.dumps(metrics, indent=2))
        self.get_logger().info(f'Episode finished: {metrics}')
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = MetricsRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
