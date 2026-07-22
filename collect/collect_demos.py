#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_demos.py — 批量采集"抓球入箱"演示数据（脚本专家策略）

每条 episode 流程：
  复位场景 -> 随机摆放三球 -> 随机选目标颜色 -> 30Hz 录制(相机+状态+动作)
  -> 执行抓放 -> 按球是否入箱自动判成功 -> 存 HDF5

用法（容器内，先 source env.sh，仿真需运行 with_camera 配置 rgb_array 模式）：
  pip install h5py                       # 首次
  python3 collect_demos.py --episodes 50 --out /workspace/demos
  python3 collect_demos.py --episodes 200 --only-success   # 只保存成功的

数据结构（每个 episode 一个 .h5）：
  /images/<相机名>      (T,) JPEG bytes(vlen)
  /state                (T, 12)  torso4 + arm_left7 + gripper_left1 实际值
  /action               (T, 12)  同布局的指令值
  /tcp_pose             (T, 3)   arm_left_tool 世界系位置(FK)
  attrs: prompt, target, success, fps, ...
"""

import argparse
import json
import os
import random
import threading
import time

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.time import Time as RclTime
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from astribot_msgs.srv import RawRequest
from astribot_msgs.msg import RobotJointState

from scripted_grasp import (Kinematics, GraspNode, START_ARM_LEFT, START_TORSO,
                            CTRL_HZ, GRASP_HOVER, LIFT_HEIGHT, MJCF_PATH)

RECORD_HZ = 30.0
BALLS = {"red": "ball_red", "green": "ball_green", "blue": "ball_blue"}
PROMPT_TPL = "pick the {color} ball and place it into the box"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", default="/workspace/demos")
    p.add_argument("--cameras", nargs="+",
                   default=["/astribot_whole_body/camera/third_person/image_raw"],
                   help="要录制的图像 topic 列表")
    p.add_argument("--site", default="arm_left_tool")
    p.add_argument("--open", type=float, default=0.0, dest="grip_open")
    p.add_argument("--preclose", type=float, default=35.0)
    p.add_argument("--close", type=float, default=70.0, dest="grip_close")
    p.add_argument("--seg-time", type=float, default=2.5)
    p.add_argument("--only-success", action="store_true", help="失败的 episode 不落盘")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


class CollectorNode(GraspNode):
    def __init__(self, args):
        super().__init__(args)
        self.images = {}                       # topic -> 最新一帧 np(H,W,3)
        for topic in args.cameras:
            self.create_subscription(Image, topic,
                                     lambda m, t=topic: self._img_cb(t, m), 1)
        self.grip_state = 0.0
        self.create_subscription(RobotJointState, "/astribot_gripper_left/joint_space_states",
                                 lambda m: setattr(self, "grip_state",
                                                   float(m.position[0]) if m.position else 0.0), 10)
        # 最近一次指令（作为 action 记录）
        self.last_cmd = {"torso": list(START_TORSO), "arm": [0.0] * 7, "grip": 0.0}
        self.reset_cli = self.create_client(Empty, "/simulation/reset")
        self.pose_cli = self.create_client(RawRequest, "/simulation/reset_object_pose")
        # 录制器
        self._rec_on = False
        self._rec_buf = None
        self._rec_thread = None

    # ---- 覆写发布函数，顺带记录 action ----
    def send_arm(self, q):
        self.last_cmd["arm"] = [float(v) for v in q]
        super().send_arm(q)

    def send_torso(self, q):
        self.last_cmd["torso"] = [float(v) for v in q]
        super().send_torso(q)

    def send_grip(self, v):
        self.last_cmd["grip"] = float(v)
        super().send_grip(v)

    def _img_cb(self, topic, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]              # 存 BGR 供 imencode
        self.images[topic] = img.copy()

    def reset_tf(self):
        """已改为直接订阅 /tf 原始消息解析物体位姿，无需重建监听器。"""
        pass

    # ---- 服务 ----
    def _wait_future(self, fut, timeout=5.0):
        if getattr(self, "external_spin", False):
            t0 = time.time()
            while not fut.done() and time.time() - t0 < timeout:
                time.sleep(0.02)
        else:
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)

    def call_reset(self):
        self.reset_cli.wait_for_service(timeout_sec=5.0)
        self._wait_future(self.reset_cli.call_async(Empty.Request()))

    def set_object_poses(self, pose_dict):
        req = RawRequest.Request()
        req.request = json.dumps(pose_dict)
        self.pose_cli.wait_for_service(timeout_sec=5.0)
        self._wait_future(self.pose_cli.call_async(req))

    # ---- 录制 ----
    def start_record(self, kin):
        self._rec_buf = {"t": [], "state": [], "action": [], "tcp": [],
                         "images": {t: [] for t in self.images_topics()}}
        self._rec_on = True
        self._rec_thread = threading.Thread(target=self._rec_loop, args=(kin,), daemon=True)
        self._rec_thread.start()

    def images_topics(self):
        return list(self.args.cameras)

    def _rec_loop(self, kin):
        period = 1.0 / RECORD_HZ
        while self._rec_on:
            t0 = time.time()
            if self.arm_q is not None and self.torso_q is not None:
                state = list(self.torso_q[:4]) + list(self.arm_q[:7]) + [self.grip_state]
                action = list(self.last_cmd["torso"]) + list(self.last_cmd["arm"]) + [self.last_cmd["grip"]]
                kin.sync(self.arm_q, self.torso_q, self.chassis_q)
                tcp, _ = kin.ee_pose()
                self._rec_buf["t"].append(time.time())
                self._rec_buf["state"].append(state)
                self._rec_buf["action"].append(action)
                self._rec_buf["tcp"].append(list(tcp))
                for topic in self._rec_buf["images"]:
                    img = self.images.get(topic)
                    if img is not None:
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self._rec_buf["images"][topic].append(jpg.tobytes() if ok else b"")
                    else:
                        self._rec_buf["images"][topic].append(b"")
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def stop_record(self):
        self._rec_on = False
        if self._rec_thread:
            self._rec_thread.join(timeout=2.0)
        return self._rec_buf


def check_success(node, ball_name, container_name="container"):
    try:
        ball = node.object_pos(ball_name, timeout=2.0)
        box = node.object_pos(container_name, timeout=2.0)
    except RuntimeError:
        return False
    return (abs(ball[0] - box[0]) < 0.07 and abs(ball[1] - box[1]) < 0.07
            and ball[2] < box[2] + 0.06)


def random_scene(node, max_retry=3):
    """随机摆三球（互不重叠，左臂可达区）。贴桌生成避免弹跳，摆完验证可查且静止。"""
    for attempt in range(max_retry):
        poses, taken = {}, []
        for name in BALLS.values():
            for _ in range(50):
                x = random.uniform(0.62, 0.76)
                y = random.uniform(0.02, 0.24)
                if all((x - a) ** 2 + (y - b) ** 2 > 0.07 ** 2 for a, b in taken):
                    taken.append((x, y))
                    poses[name] = [x, y, 0.80, 0, 0, 0, 1]     # 贴桌生成，不空投
                    break
        node.set_object_poses(poses)
        time.sleep(1.0)
        # 验证：三球 TF 可查、位置与目标接近且两次采样静止
        ok = True
        for name, p in poses.items():
            try:
                p1 = node.object_pos(name, timeout=3.0)
                time.sleep(0.3)
                p2 = node.object_pos(name, timeout=2.0)
            except RuntimeError:
                ok = False; break
            if (np.linalg.norm(np.asarray(p1[:2]) - np.asarray(p[:2])) > 0.05
                    or np.linalg.norm(np.asarray(p2) - np.asarray(p1)) > 0.005):
                ok = False; break
        if ok:
            return
        print(f"[scene] 摆放验证失败(第{attempt+1}次)，重建TF监听后重摆...", flush=True)
        node.reset_tf()
        time.sleep(1.0)
    print("[scene] 多次摆放失败，沿用当前场景", flush=True)


def run_grasp(node, kin, args, ball_name):
    """复用验证过的六段抓放序列。成功与否由外层用 TF 判定。"""
    ball = node.object_pos(ball_name)
    box = node.object_pos("container")
    q_now, torso_now = list(START_ARM_LEFT), list(START_TORSO)

    # 姿态锁定（同 scripted_grasp：以初始姿态工具朝向为 6D IK 目标）
    kin.sync(q_now, torso_now, node.chassis_q)
    _, m0 = kin.ee_pose()
    if m0 is not None:
        kin.target_mat = np.asarray(m0).reshape(3, 3).copy()

    def goto(pos, T=None):
        nonlocal q_now, torso_now
        ok, t_t, q_t = kin.ik_position(pos, q_now, torso_now)
        if not ok:
            raise RuntimeError(f"IK失败 目标{np.round(pos,3)}")
        node.move_smooth(q_now, q_t, torso_now, t_t, T or args.seg_time)
        q_now, torso_now = list(q_t), list(t_t)

    print("[step] 抓取序列开始", flush=True)
    node.hold_grip(args.grip_open, 0.8)
    goto(ball + np.array([0, 0, GRASP_HOVER]))
    print("[step] 下探", flush=True)
    node.hold_grip(args.preclose, 0.6)          # 预收窄，避免下探蹭跑球
    goto(ball + np.array([0, 0, 0.05]))
    goto(ball, T=2.5)                           # 慢速下探
    # 闭环修正：实测抓取中心与球心偏差并反向补偿（稳态误差随构型变化）
    for _ in range(20):
        node.pump(0.02)
    kin.sync(node.arm_q, node.torso_q, node.chassis_q)
    real_ee, _ = kin.ee_pose()
    delta = real_ee - node.object_pos(ball_name)
    if np.linalg.norm(delta) > 0.008:
        goto(np.asarray(node.object_pos(ball_name)) - delta, T=1.2)
    node.adaptive_close(start=args.preclose, max_close=85.0)
    print("[step] 提起并移到容器", flush=True)
    goto(ball + np.array([0, 0, LIFT_HEIGHT]))
    goto(box + np.array([0, 0, LIFT_HEIGHT]))
    goto(box + np.array([0, 0, 0.10]), T=1.5)   # 低位释放，避免球弹出/甩飞
    node.hold_grip(args.grip_open, 1.2)
    goto(box + np.array([0, 0, LIFT_HEIGHT + 0.05]), T=1.5)


def save_episode(path, buf, meta):
    with h5py.File(path, "w") as f:
        f.create_dataset("state", data=np.asarray(buf["state"], dtype=np.float32))
        f.create_dataset("action", data=np.asarray(buf["action"], dtype=np.float32))
        f.create_dataset("tcp_pose", data=np.asarray(buf["tcp"], dtype=np.float32))
        f.create_dataset("timestamp", data=np.asarray(buf["t"], dtype=np.float64))
        g = f.create_group("images")
        vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
        for topic, frames in buf["images"].items():
            key = topic.strip("/").replace("/", "_")
            ds = g.create_dataset(key, (len(frames),), dtype=vlen)
            for i, b in enumerate(frames):
                ds[i] = np.frombuffer(b, dtype=np.uint8)
        for k, v in meta.items():
            f.attrs[k] = v


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    kin = Kinematics(MJCF_PATH, args.site)
    rec_kin = Kinematics(MJCF_PATH, args.site)   # 录制线程专用，避免与主线程IK共用MjData产生竞态
    rclpy.init()
    node = CollectorNode(args)
    # 后台独立执行器：持续处理 TF/状态/图像回调，主线程只做运动逻辑（修复长跑后 TF 监听失灵）
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.external_spin = True
    n_ok = n_all = 0
    try:
        node.wait_ready()
        for ep in range(args.episodes):
            print(f"\n===== Episode {ep+1}/{args.episodes} =====")
            print("[step] 回初始姿态并重摆场景", flush=True)
            node.move_to_start(3.0)           # 初始姿态即"零位"，不再回真零位（姿态怪异）
            node.send_grip(0.0)               # 开爪备用
            random_scene(node); time.sleep(1.0)

            color = random.choice(list(BALLS))
            ball_name = BALLS[color]
            prompt = PROMPT_TPL.format(color=color)
            print(f"[task] {prompt}")
            if not hasattr(node, "empty_tq"):
                node.calibrate_gripper()      # 空载力矩曲线标定（每进程一次，手举空中时）
            if node.chassis_q is not None:
                kin.chassis_now = list(node.chassis_q)
                rec_kin.chassis_now = list(node.chassis_q)

            node.start_record(rec_kin)
            t0 = time.time()
            try:
                run_grasp(node, kin, args, ball_name)
                error = ""
            except Exception as e:
                error = str(e)
                print(f"[warn] 执行异常: {e}")
            buf = node.stop_record()
            time.sleep(0.5)

            success = (error == "") and check_success(node, ball_name)
            n_all += 1; n_ok += int(success)
            try:
                bp = np.round(node.object_pos(ball_name, timeout=2.0), 3)
                cp = np.round(node.object_pos("container", timeout=2.0), 3)
                where = f"球@{bp.tolist()} 容器@{cp.tolist()}"
            except RuntimeError:
                where = "球/容器TF不可查"
            print(f"[result] success={success}  时长{time.time()-t0:.1f}s "
                  f"帧数{len(buf['state'])}  {where}  累计成功率 {n_ok}/{n_all}")

            if success or not args.only_success:
                fname = os.path.join(args.out, f"episode_{ep:04d}_{color}_"
                                     f"{'ok' if success else 'fail'}.h5")
                save_episode(fname, buf, {
                    "prompt": prompt, "target": ball_name, "success": success,
                    "fps": RECORD_HZ, "error": error,
                    "state_layout": "torso4+arm_left7+gripper_left1",
                    "gripper_scale": "0-100, 0=open",
                })
                print(f"[saved] {fname}")
    finally:
        print(f"\n[done] 共 {n_all} 条, 成功 {n_ok} 条")
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
