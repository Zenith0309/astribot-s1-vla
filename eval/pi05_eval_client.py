#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pi05_eval_client.py — pi05 仿真闭环评测客户端（ROS2 容器内运行，先 source env.sh）

与官方 pi05 推理服务（serve_policy.py，WebSocket :8000）配对：
  1. 从仿真读观测 → 组装 31 维 SO3 state + 三相机 224 图像 + prompt
  2. msgpack 发给服务 → 收 action chunk (H, 31)（笛卡尔 SO3，已反归一化）
  3. 把 left_arm 的 SO3 位姿反解成 torso+arm 关节（复用 6D IK），下发仿真
  4. 滚动执行 n 步后重新观测（receding horizon）→ TF 判成功

依赖仅 msgpack + websockets（容器内 pip install msgpack websockets），
不需要装 pi05/torch/lerobot——推理在 GPU 端的服务里。

用法（服务已在 GPU 端启动，监听 8000）：
  pip install msgpack websockets
  python3 eval/pi05_eval_client.py --server <GPU_IP>:8000 --episodes 10
"""

import argparse
import functools
import os
import random
import re
import sys
import threading
import time

import cv2
import numpy as np
import msgpack
import rclpy
from websockets.sync.client import connect as ws_connect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "collect"))
from collect_demos import CollectorNode, random_scene, check_success, BALLS, PROMPT_TPL  # noqa: E402
from scripted_grasp import (Kinematics, MJCF_PATH, START_ARM_RIGHT,  # noqa: E402
                            START_ARM_LEFT, START_TORSO)

# ---- pi05 观测键（与训练数据列名一致） ----
STATE_KEY = "cartesian_so3_dict.cartesian_pose_state"
CAM_KEYS = {
    "head_rgbd": "images_dict.head.rgb",
    "left_wrist_rgbd": "images_dict.left_wrist.rgb",
    "right_wrist_rgbd": "images_dict.right_wrist.rgb",
}
ARM_L_JOINTS = [f"astribot_arm_left_joint_{i}" for i in range(1, 8)]
ARM_R_JOINTS = [f"astribot_arm_right_joint_{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"astribot_torso_joint_{i}" for i in range(1, 5)]
GRIP_MAX = 85.0
MAX_STEPS = 1500


# ================= msgpack-numpy（与服务端 msgpack_numpy.py 线兼容） =================
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class PI05Link:
    """与官方 serve_policy.py 的 WebSocket 连接。协议：连上收 metadata，
    之后每次 send(obs) → recv(action_dict)。"""

    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.ws = ws_connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self.metadata = _unpackb(self.ws.recv())     # 首帧是服务端元信息
        print(f"[link] 已连接 pi05 服务，metadata={self.metadata}")

    def infer(self, obs):
        self.ws.send(_packb(obs))
        rsp = self.ws.recv()
        if isinstance(rsp, str):
            raise RuntimeError(f"服务端推理错误:\n{rsp}")
        return _unpackb(rsp)                          # {"state":.., "actions":(H,31)}


# ================= SO3 <-> 矩阵（与官方 geometry.py 一致） =================
def so3_to_matrix(so3):
    r1, r2 = so3[:3], so3[3:6]
    v1 = r1 / (np.linalg.norm(r1) + 1e-8)
    v2 = r2 - np.dot(r2, v1) * v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    v3 = np.cross(v1, v2)
    return np.stack([v1, v2, v3])


def mat_to_so3(R):
    return R[:2, :].reshape(6)


# ================= 全身 FK：组装 31 维 SO3 state =================
class StateFK:
    def __init__(self):
        root = os.environ.get("ASTRIBOT_SIMU_ROOT", os.getcwd())
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(root + "/" + MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        q = lambda n: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.q_l = [q(n) for n in ARM_L_JOINTS]
        self.q_r = [q(n) for n in ARM_R_JOINTS]
        self.q_t = [q(n) for n in TORSO_JOINTS]
        s = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
        self.site_l, self.site_r = s("arm_left_tool"), s("arm_right_tool")
        self.torso_kind, self.torso_id = self._torso()

    def _torso(self):
        for n in ("torso_tool", "torso_link", "astribot_torso_tool"):
            sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, n)
            if sid >= 0:
                return "site", sid
        for i in (4, 3, 2, 1):
            bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                     f"astribot_torso_link{i}")
            if bid >= 0:
                return "body", bid
        raise RuntimeError("找不到躯干参考系")

    def _pose9(self, kind, idx):
        if kind == "site":
            return np.concatenate([self.data.site_xpos[idx],
                                   mat_to_so3(self.data.site_xmat[idx].reshape(3, 3))])
        return np.concatenate([self.data.xpos[idx],
                               mat_to_so3(self.data.xmat[idx].reshape(3, 3))])

    def state31(self, torso4, arm_l7, grip, head2=(0.0, 0.0)):
        for adr, v in zip(self.q_t, torso4):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_l, arm_l7):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_r, START_ARM_RIGHT):
            self.data.qpos[adr] = v
        self.mj.mj_forward(self.model, self.data)
        v = np.zeros(31, dtype=np.float32)
        v[0:9] = self._pose9(self.torso_kind, self.torso_id)
        v[9:18] = self._pose9("site", self.site_l)
        v[18] = grip / 100.0
        v[19:28] = self._pose9("site", self.site_r)
        v[28] = 0.0
        v[29:31] = head2
        return v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="pi05 服务 IP:PORT")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--color", default=None, choices=[None, "red", "green", "blue"])
    p.add_argument("--exec-steps", type=int, default=8,
                   help="每个 action chunk 执行几步后重新观测（receding horizon）")
    p.add_argument("--seed", type=int, default=None)
    # CollectorNode 占位参数
    p.add_argument("--site", default="arm_left_tool")
    p.add_argument("--open", type=float, default=0.0, dest="grip_open")
    p.add_argument("--preclose", type=float, default=35.0)
    p.add_argument("--close", type=float, default=70.0, dest="grip_close")
    p.add_argument("--seg-time", type=float, default=2.5)
    p.add_argument("--hold-min", type=float, default=62.0)
    args = p.parse_args()
    args.cameras = [f"/astribot_whole_body/camera/{c}/image_raw" for c in CAM_KEYS]
    return args


def cam_short(topic):
    m = re.search(r"camera/(.+?)/image", topic)
    return m.group(1) if m else topic


def build_obs(node, fk, prompt):
    if node.arm_q is None or node.torso_q is None:
        return None
    imgs = {}
    for topic in node.args.cameras:
        img = node.images.get(topic)
        if img is None:
            return None
        short = cam_short(topic)
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        imgs[CAM_KEYS[short]] = rgb.astype(np.uint8)
    state = fk.state31(node.torso_q[:4], node.arm_q[:7], node.grip_state)
    obs = {STATE_KEY: state, "prompt": prompt}
    obs.update(imgs)
    return obs


def run_episode(node, kin, fk, link, args, ball_name, prompt):
    q_now, torso_now = list(START_ARM_LEFT), list(START_TORSO)
    step = 0
    while step < MAX_STEPS:
        obs = build_obs(node, fk, prompt)
        if obs is None:
            node.pump(0.05)
            continue
        actions = link.infer(obs)["actions"]          # (H, 31)
        actions = np.asarray(actions, dtype=np.float32)
        for a in actions[:args.exec_steps]:
            # left_arm 段：pos[9:12] + 6D 旋转[12:18] → 6D IK
            kin.target_mat = so3_to_matrix(a[12:18])
            ok, t_t, q_t = kin.ik_position(a[9:12], q_now, torso_now)
            if ok:
                node.send_torso(t_t)
                node.send_arm(q_t)
                q_now, torso_now = list(q_t), list(t_t)
            node.send_grip(float(np.clip(a[18] * 100.0, 0.0, GRIP_MAX)))
            node.sim_sleep(1.0 / 30.0)
            step += 1
            if step % 30 == 0 and check_success(node, ball_name):
                return True, step
    return check_success(node, ball_name), step


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    link = PI05Link(args.server)
    fk = StateFK()
    kin = Kinematics(MJCF_PATH, args.site)

    rclpy.init()
    node = CollectorNode(args)
    head_kin = Kinematics(MJCF_PATH, args.site)
    node.start_head_track(head_kin)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.external_spin = True

    n_ok, results = 0, []
    try:
        node.wait_ready()
        for ep in range(args.episodes):
            print(f"\n===== pi05 Eval {ep+1}/{args.episodes} =====")
            node.track_object = None
            node.move_to_start(3.0)
            node.send_grip(0.0)
            random_scene(node)
            node.sim_sleep(1.0)
            color = args.color or random.choice(list(BALLS))
            ball_name = BALLS[color]
            prompt = PROMPT_TPL.format(color=color)
            print(f"[task] {prompt}")
            node.track_object = ball_name
            node.sim_sleep(1.0)
            if node.chassis_q is not None:
                kin.chassis_now = list(node.chassis_q)

            t0 = time.time()
            success, steps = run_episode(node, kin, fk, link, args, ball_name, prompt)
            n_ok += int(success)
            results.append({"episode": ep, "color": color, "success": success,
                            "steps": steps})
            print(f"[result] success={success} 步数{steps} 用时{time.time()-t0:.1f}s "
                  f"累计 {n_ok}/{ep+1}")
    finally:
        import json
        rate = n_ok / max(1, len(results))
        print(f"\n[eval done] {n_ok}/{len(results)} 成功率 {rate:.1%}")
        with open("pi05_eval_results.json", "w") as f:
            json.dump({"success_rate": rate, "episodes": results}, f,
                      ensure_ascii=False, indent=2)
        try:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
