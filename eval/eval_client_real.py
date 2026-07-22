#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_client_real.py — 真机闭环评测客户端（在机器人本体电脑上运行，aarch64 ROS2 SDK）

与 serve_policy.py（GPU 机器）配对，协议与仿真版 eval_client.py 完全相同。
IO 层走官方 SDK 高层接口（不直接发 joint_space_command topic——真机严禁）：
  观测: get_current_joints_position + register_image_callback
  动作: set_joints_position(control_way="filter")   # 250Hz 滤波吸收 30Hz 指令

用法（机器人本体电脑，先 source astribot_sdk_aarch64/env.sh）:
  python3 eval_client_real.py --server <GPU机器IP>:9000 --dry-run   # 链路验证，机器人不动
  python3 eval_client_real.py --server <GPU机器IP>:9000             # 正式执行

安全须知:
  - 首次必须 --dry-run 验证观测/推理链路
  - 确认机器人为 safe 模式，急停按钮在手边，999-stop_robot.py 随时可用
  - 本脚本对相邻两步动作做了限幅（--max-delta），异常/Ctrl+C 自动 stop_robot
  - 注意：仿真训练的策略直接上真机大概率因视觉域差距失败，本脚本首要作用是
    打通部署链路；真机成功需真机数据微调或做 sim2real 迁移
"""

import argparse
import base64
import json
import socket
import struct
import threading
import time

import cv2
import numpy as np

from astribot_sdk.core.astribot_api.astribot_client import Astribot

CTRL_HZ = 30.0            # 与训练数据频率一致
GRIP_MAX = 85.0           # 夹爪上限，与采集一致
MAX_STEPS = 1200          # 单条上限 40s

# 官方 VLA 初始姿态（astribot_eval，与仿真采集一致）
START_TORSO = [0.29, -0.58, 0.29, 0.0]
START_ARM_LEFT = [0.159, -0.022, -1.42, 1.66, -0.345, 0.115, 0.125]
START_ARM_RIGHT = [-0.159, -0.022, 1.42, 1.66, 0.345, 0.115, -0.125]
START_HEAD = [0.0, 0.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="策略服务地址 IP:PORT")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--camera", default="head_rgbd",
                   choices=["head_rgbd", "torso_rgbd", "left_wrist_rgbd",
                            "right_wrist_rgbd"],
                   help="观测相机（训练用什么就选什么）")
    p.add_argument("--dry-run", action="store_true",
                   help="只走观测->推理链路并打印动作，不下发指令")
    p.add_argument("--max-delta", type=float, default=0.06,
                   help="相邻两步关节指令最大变化(rad)，安全限幅")
    p.add_argument("--prompt", default="pick the red ball and place it into the box")
    return p.parse_args()


class PolicyLink:
    """TCP + 长度前缀 JSON，与 serve_policy.py 配对。"""

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
            c = self.sock.recv(4 - len(head))
            if not c:
                raise ConnectionError("服务端断开")
            head += c
        (n,) = struct.unpack(">I", head)
        body = b""
        while len(body) < n:
            c = self.sock.recv(min(65536, n - len(body)))
            if not c:
                raise ConnectionError("服务端断开")
            body += c
        return json.loads(body.decode())

    def reset(self):
        self._send({"type": "reset"})
        return self._recv()

    def act(self, state, image_bgr):
        ok, jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        self._send({"type": "act", "state": [float(v) for v in state],
                    "image_jpeg_b64": base64.b64encode(jpg.tobytes()).decode()})
        rsp = self._recv()
        if "action" not in rsp:
            raise RuntimeError(f"服务端返回异常: {rsp}")
        return np.asarray(rsp["action"], dtype=np.float32)


class CameraBuffer:
    """保存指定相机最新一帧 BGR 图。"""

    def __init__(self):
        self.frame = None
        self.stamp = 0.0
        self._lock = threading.Lock()

    def cb(self, topic_name, msg, width, height, array):
        if msg.format.lower() != "jpeg":       # 只要 color 流
            return
        with self._lock:
            self.frame = array                  # need_decode=True -> BGR HWC
            self.stamp = time.time()

    def latest(self, max_age=0.5):
        with self._lock:
            if self.frame is None or time.time() - self.stamp > max_age:
                return None
            return self.frame.copy()


def get_state(bot, names):
    """state(12) = torso4 + arm_left7 + gripper_left1，布局与训练数据一致。"""
    pos = bot.get_current_joints_position(names)
    return list(pos[0][:4]) + list(pos[1][:7]) + [float(pos[2][0])]


def main():
    args = parse_args()

    link = PolicyLink(args.server)
    print(f"[link] 已连接策略服务 {args.server}")

    bot = Astribot(freq=250.0)
    names = [bot.torso_name, bot.arm_left_name, bot.effector_left_name]

    # 相机
    cam = CameraBuffer()
    bot.activate_camera()
    for _ in range(20):
        info = bot.get_cameras_info()
        if info.get(args.camera, {}).get("activate"):
            break
        print("[cam] 等待相机激活...")
        time.sleep(1.0)
    bot.register_image_callback(args.camera, "color", cam.cb, need_decode=True)
    print(f"[cam] {args.camera} 已就绪")

    n_ok = 0
    results = []
    try:
        for ep in range(args.episodes):
            print(f"\n===== 真机评测 {ep+1}/{args.episodes} =====")
            print("[step] 回官方初始姿态（5s）")
            if not args.dry_run:
                bot.move_joints_position(
                    [bot.torso_name, bot.arm_left_name, bot.arm_right_name,
                     bot.head_name, bot.effector_left_name],
                    [START_TORSO, START_ARM_LEFT, START_ARM_RIGHT,
                     START_HEAD, [0.0]], duration=5.0)
            input("[operator] 摆好物体后按回车开始本条评测...")

            link.reset()
            prev_cmd = None
            period = 1.0 / CTRL_HZ
            t_ep = time.time()
            for step in range(MAX_STEPS):
                t0 = time.time()
                state = get_state(bot, names)
                img = cam.latest()
                if img is None:
                    print("[warn] 相机帧过期，跳过本步")
                    time.sleep(period)
                    continue
                action = link.act(state, img)

                # 安全限幅：相邻两步指令变化不超过 max-delta
                if prev_cmd is not None:
                    action[:11] = np.clip(action[:11],
                                          prev_cmd[:11] - args.max_delta,
                                          prev_cmd[:11] + args.max_delta)
                action[11] = float(np.clip(action[11], 0.0, GRIP_MAX))
                prev_cmd = action.copy()

                if args.dry_run:
                    if step % 30 == 0:
                        print(f"[dry] step {step}  action={np.round(action, 3).tolist()}")
                else:
                    bot.set_joints_position(
                        names,
                        [action[0:4].tolist(), action[4:11].tolist(), [action[11]]],
                        control_way="filter")
                time.sleep(max(0.0, period - (time.time() - t0)))

            ans = input("[operator] 本条是否成功? [y/N] ").strip().lower()
            success = ans == "y"
            n_ok += int(success)
            results.append({"episode": ep, "success": success,
                            "seconds": round(time.time() - t_ep, 1)})
            print(f"[result] success={success}  累计 {n_ok}/{ep+1}")
    except KeyboardInterrupt:
        print("\n[abort] 手动中断")
    finally:
        if not args.dry_run:
            try:
                bot.stop_robot()
                print("[safe] 已调用 stop_robot()")
            except Exception as e:
                print(f"[safe] stop_robot 异常: {e}，请立即使用物理急停")
        rate = n_ok / max(1, len(results))
        print(f"\n[eval done] {n_ok}/{len(results)}  成功率 {rate:.1%}")
        with open("eval_results_real.json", "w") as f:
            json.dump({"success_rate": rate, "episodes": results}, f,
                      ensure_ascii=False, indent=2)
        print("[saved] eval_results_real.json")


if __name__ == "__main__":
    main()
