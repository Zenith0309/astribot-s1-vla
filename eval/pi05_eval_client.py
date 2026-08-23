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
                            START_ARM_LEFT, START_TORSO, STATE_HZ)

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
        for n in ("astribot_torso_link_4", "astribot_torso_link_3",
                  "astribot_torso_link_2", "astribot_torso_link_1"):
            bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, n)
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
    p.add_argument("--trace-every", type=int, default=0, metavar="N",
                   help="每 N 步记录一次预测目标（配合 --dump-obs 同时存图），"
                        "用于定位闭环从第几步开始发散")
    p.add_argument("--dump-obs", default=None, metavar="DIR",
                   help="把每条 episode 首帧实际发给模型的三路图像存到 DIR，用于核对"
                        "评测观测是否与训练分布一致（排查'模型输出恒定'类问题）")
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
        # processor 期望 CHW float[0,1]（与训练时 LeRobot 输出一致，不是 HWC uint8）
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
        imgs[CAM_KEYS[short]] = chw
    # head 两维必须发真实值，与训练侧一致（v3 起数据里记录了真实 head 角度）。
    # 训练时若 head 是常量、评测时给真实值（或反之）都会造成分布不匹配。
    head2 = tuple(node.head_q[:2]) if getattr(node, "head_q", None) is not None else (0.0, 0.0)
    state31 = fk.state31(node.torso_q[:4], node.arm_q[:7], node.grip_state, head2)
    # 官方 output_config 的 state 规范是 34 维：末尾追加 chassis [x,y,yaw]。
    # select_pad 只读 0:31（模型输入不变），post_process 的 output_extract 需要满 34 维。
    chassis = node.chassis_q if node.chassis_q is not None else [0.0, 0.0, 0.0]
    state = np.concatenate([state31, np.asarray(chassis[:3], dtype=np.float32)]).astype(np.float32)
    obs = {STATE_KEY: state, "prompt": prompt}
    # obs_timestamp（仿真秒）：RtcPolicy 靠它算「上次推理后实际执行了几步」。
    # 不发这个字段时 _calculate_execute_steps() 直接 return 1，RTC 前缀只按 1 步偏移，
    # 而我们实际执行 exec_steps 步 —— 每次错位 (exec_steps-1) 步并不断累积，
    # 预测会彻底跑偏（实测中位误差 34cm）。用仿真时钟而非墙钟，与训练时序一致。
    obs["obs_timestamp"] = float(node._tick) / STATE_HZ
    obs.update(imgs)
    return obs


def run_episode(node, kin, fk, link, args, ball_name, prompt, probe=None):
    """probe: 可选 dict，回填第一次推理的动作特征，用于诊断因果混淆。

    只看成功率会被"球恰好落在记忆的平均动作附近"骗到。真正的判据是
    【模型输出是否随球位变化】——所以这里记录首个 action chunk 的左臂
    目标位置 a[9:12]，配合 episode 记录的球位，事后算相关性即可判定。
    """
    q_now, torso_now = list(START_ARM_LEFT), list(START_TORSO)
    step = 0
    ik_ok = ik_fail = 0          # IK 成败计数：全失败 -> 机器人不动 -> 状态不变 -> 死锁
    tcp_trace = []               # 模型指令目标的游走范围
    # 抓取判据要看【实际到达】而非【指令】：训练数据的指令位置本身相对球心偏了一个
    # 跟踪误差（采集时 goto(aim - delta) 的闭环修正），模型学到的就是带偏移的指令。
    best_d = 1e9                 # 实际末端到球的最小距离
    best_at = None
    while step < MAX_STEPS:
        obs = build_obs(node, fk, prompt)
        if obs is None:
            node.pump(0.05)
            continue
        actions = link.infer(obs)["actions"]          # (H, 31)
        actions = np.asarray(actions, dtype=np.float32)
        # 每隔 --trace-every 步记一次「预测目标 + 图像」，用于定位闭环从第几步开始发散
        if probe is not None and probe.get("trace_every"):
            _te = probe["trace_every"]
            if step % _te < args.exec_steps:
                probe.setdefault("trace", []).append(
                    [step, [round(float(v), 4) for v in actions[0, 9:12]]])
                if probe.get("dump_dir"):
                    import os as _os
                    _d = _os.path.join(probe["dump_dir"], f"ep{probe.get('ep',0):02d}")
                    _os.makedirs(_d, exist_ok=True)
                    _hk = "images_dict.head.rgb"
                    if _hk in obs:
                        _im = (np.asarray(obs[_hk]).transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(_os.path.join(_d, f"step{step:04d}_head.png"),
                                    cv2.cvtColor(_im, cv2.COLOR_RGB2BGR))
        if probe is not None and probe.get("dump_dir") and "first_action_xyz" not in probe:
            # 存下真正发给模型的图（CHW float[0,1] -> HWC uint8）
            import os as _os
            _os.makedirs(probe["dump_dir"], exist_ok=True)
            for _k, _v in obs.items():
                if _k.startswith("images_dict"):
                    _im = (np.asarray(_v).transpose(1, 2, 0) * 255).astype(np.uint8)
                    _n = _k.replace("images_dict.", "").replace(".rgb", "")
                    cv2.imwrite(_os.path.join(probe["dump_dir"],
                                f"ep{probe.get('ep',0):02d}_{_n}.png"),
                                cv2.cvtColor(_im, cv2.COLOR_RGB2BGR))
        if probe is not None and "first_action_xyz" not in probe:
            # 首个 chunk 的左臂目标位置（模型"想去哪"），以及该 chunk 内的位移幅度
            probe["first_action_xyz"] = [round(float(v), 4) for v in actions[0, 9:12]]
            probe["chunk_span_cm"] = round(
                float(np.linalg.norm(actions[-1, 9:12] - actions[0, 9:12]) * 100), 2)
        for a in actions[:args.exec_steps]:
            # left_arm 段：pos[9:12] + 6D 旋转[12:18] → 6D IK
            kin.target_mat = so3_to_matrix(a[12:18])
            ok, t_t, q_t = kin.ik_position(a[9:12], q_now, torso_now)
            if ok:
                ik_ok += 1
                node.send_torso(t_t)
                node.send_arm(q_t)
                q_now, torso_now = list(q_t), list(t_t)
            else:
                ik_fail += 1
            if step % 60 == 0:
                tcp_trace.append([round(float(v), 4) for v in a[9:12]])
            if step % 10 == 0:      # 实际末端 vs 球：抓取到底有没有到位
                try:
                    _b = np.asarray(node.object_pos(ball_name, timeout=0.5))
                    kin.sync(node.arm_q, node.torso_q, node.chassis_q)
                    _ee, _ = kin.ee_pose()
                    if _ee is not None:
                        _d = float(np.linalg.norm(np.asarray(_ee) - _b))
                        if _d < best_d:
                            best_d, best_at = _d, step
                except (RuntimeError, TypeError):
                    pass
            node.send_grip(float(np.clip(a[18] * 100.0, 0.0, GRIP_MAX)))
            node.sim_sleep(1.0 / 30.0)
            step += 1
            if step % 30 == 0 and check_success(node, ball_name):
                _fin(probe, ik_ok, ik_fail, tcp_trace, best_d, best_at)
                return True, step
    _fin(probe, ik_ok, ik_fail, tcp_trace, best_d, best_at)
    return check_success(node, ball_name), step


def _fin(probe, ik_ok, ik_fail, tcp_trace, best_d=None, best_at=None):
    """回填执行侧诊断：IK 成功率 + 模型目标点的实际游走范围。"""
    if probe is None:
        return
    probe["ik_ok"] = ik_ok
    probe["ik_fail"] = ik_fail
    if best_d is not None and best_d < 1e8:
        probe["closest_cm"] = round(best_d * 100, 2)     # 实际末端最接近球的距离
        probe["closest_at_step"] = best_at
    if len(tcp_trace) >= 2:
        t = np.asarray(tcp_trace, dtype=float)
        probe["target_range_cm"] = [round(float(v) * 100, 2) for v in (t.max(0) - t.min(0))]


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # 每条 episode 重连一次：服务端 _handler 在新连接时会 self._policy.reset()，
    # 清掉上一条 episode 残留的策略状态（如 RTC 动作前缀）。
    # 官方注释原话："New connection = new episode"。复用一条连接跑多个 episode，
    # 会让第 2 条起被上一条的动作前缀污染。
    link = None
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
            if link is not None:            # 新 episode 换新连接 -> 服务端 reset 策略状态
                try:
                    link.ws.close()
                except Exception:
                    pass
            link = PI05Link(args.server)
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

            # 记录球的初始位置：判定"动作是否随球位变化"的自变量
            try:
                ball_pos = [round(float(v), 4)
                            for v in node.object_pos(ball_name, timeout=2.0)]
            except RuntimeError:
                ball_pos = None

            t0 = time.time()
            probe = {"ep": ep, "dump_dir": args.dump_obs,
                     "trace_every": args.trace_every}
            success, steps = run_episode(node, kin, fk, link, args, ball_name,
                                         prompt, probe=probe)
            n_ok += int(success)
            # bool(): check_success 返回 numpy.bool_，json 不认
            for _k in ("dump_dir", "ep", "trace_every"):         # 仅内部用
                probe.pop(_k, None)
            results.append({"episode": ep, "color": color, "success": bool(success),
                            "steps": int(steps), "ball_pos": ball_pos, **probe})
            print(f"[result] success={success} 步数{steps} 用时{time.time()-t0:.1f}s "
                  f"累计 {n_ok}/{ep+1}")
    finally:
        import json
        rate = n_ok / max(1, len(results))
        print(f"\n[eval done] {n_ok}/{len(results)} 成功率 {rate:.1%}")

        # ---- 因果混淆判据：模型的目标位置是否随球位变化 ----
        diag = {}
        pts = [(r["ball_pos"], r.get("first_action_xyz")) for r in results
               if r.get("ball_pos") and r.get("first_action_xyz")]
        if len(pts) >= 3:
            b = np.array([p[0] for p in pts], dtype=float)[:, :2]   # 球 xy
            a = np.array([p[1] for p in pts], dtype=float)[:, :2]   # 目标 xy
            act_std = a.std(axis=0)
            diag = {
                "n": len(pts),
                "ball_xy_std_cm": [round(float(v) * 100, 2) for v in b.std(axis=0)],
                "action_xy_std_cm": [round(float(v) * 100, 2) for v in act_std],
                "corr_x": round(float(np.corrcoef(b[:, 0], a[:, 0])[0, 1]), 3)
                          if act_std[0] > 1e-6 else None,
                "corr_y": round(float(np.corrcoef(b[:, 1], a[:, 1])[0, 1]), 3)
                          if act_std[1] > 1e-6 else None,
            }
            print(f"[diag] 球位 xy 标准差 {diag['ball_xy_std_cm']} cm")
            print(f"[diag] 模型目标 xy 标准差 {diag['action_xy_std_cm']} cm")
            print(f"[diag] 相关系数 x={diag['corr_x']} y={diag['corr_y']}")
            if max(act_std) * 100 < 1.0:
                print("[diag] ⚠️ 目标位置几乎不变（<1cm）→ 仍在输出记忆的平均动作，"
                      "因果混淆未破解")
            elif (diag["corr_x"] or 0) > 0.5 or (diag["corr_y"] or 0) > 0.5:
                print("[diag] ✅ 目标位置随球位变化且正相关 → 模型在用视觉定位")
            else:
                print("[diag] ⚠️ 目标位置有变化但与球位不相关 → 可能在跟随别的信号")

        with open("pi05_eval_results.json", "w") as f:
            json.dump({"success_rate": rate, "diagnosis": diag,
                       "episodes": results}, f, ensure_ascii=False, indent=2)
        try:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
