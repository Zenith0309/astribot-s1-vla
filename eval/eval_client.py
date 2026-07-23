#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_client.py — 仿真闭环评测客户端（ROS 容器内运行，先 source env.sh）

流程：每条 episode 回初始姿态 -> 随机摆球 -> 随机选色 -> 30Hz 循环
      [取观测 -> 发给 serve_policy.py -> 收 action -> 下发关节指令]
      -> 超时或球入箱结束 -> TF 判成功 -> 统计成功率

用法（serve_policy.py 已在宿主机/训练环境启动）：
  python3 eval/eval_client.py --episodes 20 --server 127.0.0.1:9000
  python3 eval/eval_client.py --episodes 20 --color red     # 固定目标颜色

默认观测 = 头 + 双腕三路相机（与真机/官方 pi05 对齐），须与训练数据一致；
仿真 yaml camera_names 需含 head_rgbd / left_wrist_rgbd / right_wrist_rgbd。

依赖 collect/ 下的采集脚本（同仓库，自动加入 sys.path）。
"""

import argparse
import base64
import json
import os
import random
import re
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
import rclpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "collect"))
from collect_demos import CollectorNode, random_scene, check_success, BALLS, PROMPT_TPL  # noqa: E402
from scripted_grasp import Kinematics, MJCF_PATH  # noqa: E402

CTRL_HZ = 30.0            # 与训练数据录制频率一致
MAX_STEPS = 1200          # 单条上限 40s
GRIP_MAX = 85.0           # 与采集时一致，防挤飞


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--server", default="127.0.0.1:9000")
    p.add_argument("--color", default=None, choices=[None, "red", "green", "blue"])
    p.add_argument("--cameras", nargs="+", default=[
        "/astribot_whole_body/camera/head_rgbd/image_raw",
        "/astribot_whole_body/camera/left_wrist_rgbd/image_raw",
        "/astribot_whole_body/camera/right_wrist_rgbd/image_raw"],
        help="观测相机 topic，短名须与训练特征一致")
    p.add_argument("--seed", type=int, default=None)
    # CollectorNode 需要的参数占位（评测不用夹取标定，但基类要求存在）
    p.add_argument("--site", default="arm_left_tool")
    p.add_argument("--open", type=float, default=0.0, dest="grip_open")
    p.add_argument("--preclose", type=float, default=35.0)
    p.add_argument("--close", type=float, default=70.0, dest="grip_close")
    p.add_argument("--seg-time", type=float, default=2.5)
    return p.parse_args()


def cam_short_name(topic):
    """/astribot_whole_body/camera/head_rgbd/image_raw -> head_rgbd"""
    m = re.search(r"camera/(.+?)/image", topic)
    return m.group(1) if m else topic.strip("/").replace("/", "_")


class PolicyLink:
    """与 serve_policy.py 的 TCP 连接（长度前缀 JSON）。"""

    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=10.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.sock.sendall(struct.pack(">I", len(body)) + body)

    def _recv(self):
        head = b""
        while len(head) < 4:
            chunk = self.sock.recv(4 - len(head))
            if not chunk:
                raise ConnectionError("服务端断开")
            head += chunk
        (length,) = struct.unpack(">I", head)
        body = b""
        while len(body) < length:
            chunk = self.sock.recv(min(65536, length - len(body)))
            if not chunk:
                raise ConnectionError("服务端断开")
            body += chunk
        return json.loads(body.decode())

    def reset(self):
        self._send({"type": "reset"})
        return self._recv()

    def act(self, state, images_bgr):
        """images_bgr: {短名: BGR ndarray}"""
        payload = {}
        for name, img in images_bgr.items():
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f"JPEG 编码失败: {name}")
            payload[name] = base64.b64encode(jpg.tobytes()).decode()
        self._send({"type": "act", "state": [float(v) for v in state],
                    "images": payload})
        rsp = self._recv()
        if "action" not in rsp:
            raise RuntimeError(f"服务端返回异常: {rsp}")
        return np.asarray(rsp["action"], dtype=np.float32)


def get_obs(node, cameras):
    """观测 = state(12) + 各相机最新帧 {短名: BGR}。布局与训练数据一致。"""
    if node.arm_q is None or node.torso_q is None:
        return None, None
    state = list(node.torso_q[:4]) + list(node.arm_q[:7]) + [node.grip_state]
    images = {}
    for topic in cameras:
        img = node.images.get(topic)
        if img is None:
            return state, None                     # 任一路缺帧则等待
        images[cam_short_name(topic)] = img
    return state, images


def run_episode(node, link, args, ball_name):
    link.reset()
    period = 1.0 / CTRL_HZ
    for step in range(MAX_STEPS):
        state, images = get_obs(node, args.cameras)
        if state is None or images is None:
            node.pump(0.05)
            continue
        action = link.act(state, images)
        node.send_torso(action[0:4])
        node.send_arm(action[4:11])
        node.send_grip(float(np.clip(action[11], 0.0, GRIP_MAX)))
        # 每秒查一次是否已成功（提前结束）
        if step % int(CTRL_HZ) == 0 and step > 0 and check_success(node, ball_name):
            return True, step
        node.sim_sleep(period)     # 按仿真时钟定步，与训练数据帧间隔一致
    return check_success(node, ball_name), MAX_STEPS


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    link = PolicyLink(args.server)
    print(f"[link] 已连接策略服务 {args.server}")

    rclpy.init()
    node = CollectorNode(args)
    head_kin = Kinematics(MJCF_PATH, args.site)
    node.start_head_track(head_kin)      # 头部跟踪与采集时一致，保证观测分布相同
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.external_spin = True

    n_ok = 0
    results = []
    try:
        node.wait_ready()
        for ep in range(args.episodes):
            print(f"\n===== Eval {ep+1}/{args.episodes} =====")
            node.track_object = None
            node._head_cmd = None
            node.move_to_start(3.0)
            node.send_grip(0.0)
            random_scene(node)
            time.sleep(1.0)
            color = args.color or random.choice(list(BALLS))
            ball_name = BALLS[color]
            print(f"[task] {PROMPT_TPL.format(color=color)}")
            node.track_object = ball_name    # 与采集一致：头部相机跟踪目标球
            time.sleep(1.0)

            t0 = time.time()
            success, steps = run_episode(node, link, args, ball_name)
            n_ok += int(success)
            results.append({"episode": ep, "color": color, "success": success,
                            "steps": steps, "seconds": round(time.time() - t0, 1)})
            print(f"[result] success={success}  步数{steps}  "
                  f"用时{time.time()-t0:.1f}s  累计成功率 {n_ok}/{ep+1}")
    finally:
        rate = n_ok / max(1, len(results))
        print(f"\n[eval done] {n_ok}/{len(results)}  成功率 {rate:.1%}")
        with open("eval_results.json", "w") as f:
            json.dump({"success_rate": rate, "episodes": results}, f,
                      ensure_ascii=False, indent=2)
        print("[saved] eval_results.json")
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
