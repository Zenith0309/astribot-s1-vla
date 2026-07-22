#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
go_start_pose.py — 将仿真机器人全身移动到官方 VLA 执行系统的初始姿态
（躯干前倾 + 双臂抬起 + 头部归零 + 双爪闭合，来自 astribot_eval config 的 start_joints_pose）

用法（容器内，先 source env.sh，仿真在跑）：
  python3 go_start_pose.py             # 5秒到位后退出
  python3 go_start_pose.py --time 8    # 自定义时长
"""

import argparse
import time
import numpy as np
import rclpy
from rclpy.node import Node
from astribot_msgs.msg import RobotJointController, RobotJointState

CTRL_HZ = 50.0

# 官方 astribot_eval 的 start_joints_pose
START_POSE = {
    "torso":         [0.29, -0.58, 0.29, 0.0],
    "arm_left":      [0.159, -0.022, -1.42, 1.66, -0.345, 0.115, 0.125],
    "gripper_left":  [0.0],
    "arm_right":     [-0.159, -0.022, 1.42, 1.66, 0.345, 0.115, -0.125],
    "gripper_right": [0.0],
    "head":          [0.0, 0.0],
}


class GoStart(Node):
    def __init__(self):
        super().__init__("go_start_pose")
        self.pubs, self.state = {}, {}
        for part in START_POSE:
            topic = f"/astribot_{part}/joint_space_command"
            self.pubs[part] = self.create_publisher(RobotJointController, topic, 10)
            self.create_subscription(
                RobotJointState, f"/astribot_{part}/joint_space_states",
                lambda msg, p=part: self.state.__setitem__(p, list(msg.position)), 10)

    def wait_ready(self, timeout=10.0):
        t0 = time.time()
        while rclpy.ok() and len(self.state) < len(START_POSE):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                missing = set(START_POSE) - set(self.state)
                raise RuntimeError(f"等待关节状态超时，缺: {missing}（确认仿真在跑且 ROS_DOMAIN_ID=25）")

    def send(self, part, q):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v) for v in q]
        self.pubs[part].publish(msg)

    def run(self, duration):
        q0 = {p: np.asarray(self.state[p][:len(START_POSE[p])]) for p in START_POSE}
        q1 = {p: np.asarray(START_POSE[p]) for p in START_POSE}
        steps = max(2, int(duration * CTRL_HZ))
        for i in range(steps + 1):
            s = i / steps
            k = 10 * s**3 - 15 * s**4 + 6 * s**5   # min-jerk
            for p in START_POSE:
                self.send(p, q0[p] + k * (q1[p] - q0[p]))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / CTRL_HZ)
        print("[done] 已到达官方 VLA 初始姿态")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=5.0, help="运动时长(秒)")
    args = ap.parse_args()
    rclpy.init()
    node = GoStart()
    try:
        node.wait_ready()
        node.run(args.time)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
