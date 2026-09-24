# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
'''
Publishes the positions of the simulated humans as geometry_msgs/PoseArray on
/humans/poses (frame "map", one pose per actor, fixed order).

source := "script" (default)
    The actors of REMROC worlds follow scripted <trajectory> waypoints. Gazebo
    Fortress animates scripted actors on the rendering side, and their server-side
    pose is not reliably published, so this mode evaluates the very same script (from
    the world SDF) at the simulation time bridged from Gazebo (/clock). This is what
    the robots' lidars see.
source := "gazebo"
    Relays actor poses bridged from Gazebo as tf2_msgs/TFMessage on /humans/gz_poses
    (e.g. /world/<world>/dynamic_pose/info or /world/<world>/pose/info through
    ros_gz_bridge), keeping the child frames whose name starts with "actor".

Optional Gaussian position noise (noise_std) emulates a people tracker.
'''

from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
from tf2_msgs.msg import TFMessage

from remroc_ha.humans import load_humans_from_sdf


class HumanPosePublisher(Node):

    def __init__(self):
        super().__init__('human_pose_publisher')
        self.declare_parameter('world_name', 'depot')
        self.declare_parameter('number_of_humans', 0)
        self.declare_parameter('sample', 0)
        self.declare_parameter('world_sdf', '')
        self.declare_parameter('source', 'script')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('noise_std', 0.0)
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.pub = self.create_publisher(PoseArray, '/humans/poses', 10)
        self.noise = float(self.get_parameter('noise_std').value)
        self.rng = np.random.default_rng(0)
        self.source = self.get_parameter('source').value
        self.humans = []
        if self.source == 'script':
            sdf = self.get_parameter('world_sdf').value
            if not sdf:
                world = self.get_parameter('world_name').value
                n = self.get_parameter('number_of_humans').value
                s = self.get_parameter('sample').value
                sdf = str(Path(get_package_share_directory('remroc')) / 'worlds' / 'sdfs' / f'{world}_{n}_{s}.sdf')
            self.humans = load_humans_from_sdf(sdf) if Path(sdf).exists() else []
            self.get_logger().info(f'{len(self.humans)} scripted humans loaded from {sdf}')
            self.create_timer(1.0 / float(self.get_parameter('rate').value), self.publish_script)
        else:
            self.create_subscription(TFMessage, '/humans/gz_poses', self.relay, 10)

    def _pose(self, x, y):
        p = Pose()
        if self.noise > 0:
            x += self.rng.normal(0.0, self.noise)
            y += self.rng.normal(0.0, self.noise)
        p.position.x, p.position.y = float(x), float(y)
        p.orientation.w = 1.0
        return p

    def publish_script(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        msg = PoseArray()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'map'
        msg.poses = [self._pose(*h.position(t)) for h in self.humans]
        self.pub.publish(msg)

    def relay(self, tf_msg):
        actors = sorted((tr for tr in tf_msg.transforms if tr.child_frame_id.split('/')[-1].startswith('actor')),
                        key=lambda tr: tr.child_frame_id)
        if not actors:
            return
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.poses = [self._pose(tr.transform.translation.x, tr.transform.translation.y) for tr in actors]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
