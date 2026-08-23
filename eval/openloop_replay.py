#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""openloop_replay.py — 把训练数据里记录的动作在仿真里开环重放，隔离"执行链路"故障。

背景：v3 模型单步预测误差 0.3cm（准）、IK 100% 成功，但闭环评测 20/20 跑满步数不成功，
一条 episode 内目标点游走 60~100cm。此时"模型闭环发散"和"执行链路本身不忠实"分不开。

本脚本完全不用模型，只重放**记录过的专家动作**，分两条路径：

  A. 关节直下发  —— send_arm(记录的关节角)。这就是 collect_demos 当初做的事，必须成功。
  B. 经 IK 层    —— 先 FK 记录的关节角得到笛卡尔位姿，再用评测客户端的
                     kin.ik_position() 反解回关节角下发。这是**评测走的路**。

判读：
  - A 成功、B 失败       -> IK 层（或其调用方式）是罪魁，模型无罪
  - A、B 都成功          -> 执行链路没问题，问题在模型闭环（误差累积/观测漂移）
  - A 也失败             -> 仿真状态/场景摆放/时序有问题，先修这个

用法（容器内，仿真运行中）：
  python3 eval/openloop_replay.py --h5 /workspace/demos_v2_clean296/episode_0000_red_ok.h5 --path both
"""

import argparse
import os
import sys
import threading

import h5py
import numpy as np
import rclpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collect"))
sys.path.insert(0, os.environ.get("ASTRIBOT_SIMU_ROOT", "."))

from collect_demos import CollectorNode, check_success, BALLS  # noqa: E402
from scripted_grasp import Kinematics, MJCF_PATH, START_ARM_RIGHT  # noqa: E402

# MJCF_PATH 在 scripted_grasp 里是相对路径，依赖 cwd。解析成绝对路径，
# 这样脚本在任何目录下都能跑（只要设了 ASTRIBOT_SIMU_ROOT）。
if not os.path.isabs(MJCF_PATH):
    _root = os.environ.get("ASTRIBOT_SIMU_ROOT", ".")
    _abs = os.path.join(_root, MJCF_PATH)
    if os.path.exists(_abs):
        MJCF_PATH = _abs


def mat_to_so3(R):
    return R[:2, :].reshape(6)


def so3_to_matrix(so3):
    r1, r2 = so3[:3], so3[3:6]
    v1 = r1 / (np.linalg.norm(r1) + 1e-8)
    v2 = r2 - np.dot(r2, v1) * v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.stack([v1, v2, np.cross(v1, v2)])


class FK:
    """由关节角算左臂 TCP 位姿（与转换脚本同一套 MJCF）。"""

    def __init__(self, site="arm_left_tool"):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        jn = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.q_t = [self.model.jnt_qposadr[jn(f"astribot_torso_joint_{i}")] for i in range(1, 5)]
        self.q_l = [self.model.jnt_qposadr[jn(f"astribot_arm_left_joint_{i}")] for i in range(1, 8)]
        self.q_r = [self.model.jnt_qposadr[jn(f"astribot_arm_right_joint_{i}")] for i in range(1, 8)]
        self.site_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)

    def tcp(self, torso4, arm7):
        for adr, v in zip(self.q_t, torso4):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_l, arm7):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_r, START_ARM_RIGHT):
            self.data.qpos[adr] = v
        self.mj.mj_forward(self.model, self.data)
        return (self.data.site_xpos[self.site_l].copy(),
                self.data.site_xmat[self.site_l].reshape(3, 3).copy())


def replay(node, kin, fk, act, ball_name, use_ik, grip_max=85.0, grasp_idx=None):
    """重放一条动作序列。use_ik=False 直接下发关节；True 走 FK->IK 往返。"""
    q_now = list(act[0, 4:11])
    torso_now = list(act[0, :4])
    err = []
    trace = []                      # 球的位置随时间；判断有没有被抓起来
    for t in range(len(act)):
        if t % 60 == 0 or (grasp_idx is not None and abs(t - grasp_idx) < 2):
            try:
                bp = node.object_pos(ball_name, timeout=1.0)
                pos_now, _ = fk.tcp(act[t, :4], act[t, 4:11])
                trace.append((t, [round(float(v), 4) for v in bp],
                              round(float(np.linalg.norm(np.asarray(bp) - pos_now)) * 100, 2)))
            except RuntimeError:
                pass
        torso4, arm7, grip = act[t, :4], act[t, 4:11], act[t, 11]
        if use_ik:
            pos, R = fk.tcp(torso4, arm7)              # 记录动作 -> 笛卡尔
            kin.target_mat = R
            ok, t_t, q_t = kin.ik_position(pos, q_now, torso_now)   # 笛卡尔 -> 关节
            if ok:
                node.send_torso(t_t); node.send_arm(q_t)
                q_now, torso_now = list(q_t), list(t_t)
                p2, _ = fk.tcp(t_t, q_t)
                err.append(np.linalg.norm(p2 - pos))   # IK 往返误差
            else:
                err.append(np.nan)
        else:
            node.send_torso(list(torso4)); node.send_arm(list(arm7))
        node.send_grip(float(np.clip(grip, 0.0, grip_max)))
        node.sim_sleep(1.0 / 30.0)
    node.sim_sleep(0.5)
    ok_final = bool(check_success(node, ball_name))
    e = np.array([v for v in err if not np.isnan(v)]) if err else None
    return ok_final, e, (np.isnan(err).sum() if err else 0), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="一条采集好的 episode")
    ap.add_argument("--path", default="both", choices=["joint", "ik", "both"])
    ap.add_argument("--site", default="arm_left_tool")
    ap.add_argument("--open", type=float, default=0.0, dest="grip_open")
    ap.add_argument("--preclose", type=float, default=35.0)
    ap.add_argument("--close", type=float, default=70.0, dest="grip_close")
    ap.add_argument("--seg-time", type=float, default=2.5)
    ap.add_argument("--hold-min", type=float, default=62.0)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    args.cameras = [f"/astribot_whole_body/camera/{c}/image_raw"
                    for c in ("head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd")]

    with h5py.File(args.h5, "r") as f:
        act = f["action"][:]
        tcp = f["tcp_pose"][:]
        target = str(f.attrs["target"])
        prompt = str(f.attrs["prompt"])
    # 抓取瞬间（下探最低点）的 xy 即球的原始位置
    gi = int(np.argmin(tcp[:, 2]))
    ball_xy = tcp[gi, :2]
    print(f"[info] {os.path.basename(args.h5)}  帧数={len(act)}  目标={target}")
    print(f"[info] prompt: {prompt}")
    print(f"[info] 由 tcp 最低点反推球位 xy = {np.round(ball_xy, 4).tolist()}")

    rclpy.init()
    node = CollectorNode(args)
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.external_spin = True
    node.wait_ready()

    kin = Kinematics(MJCF_PATH, args.site)
    fk = FK(args.site)
    results = {}
    try:
        paths = ["joint", "ik"] if args.path == "both" else [args.path]
        for p in paths:
            print(f"\n===== 路径 {p.upper()} =====")
            node.move_to_start(3.0)
            node.send_grip(0.0)
            # 把三球摆回记录时的位置（目标球放原位，另两球挪开避免干扰）
            poses = {}
            others = [b for b in BALLS.values() if b != target]
            poses[target] = [float(ball_xy[0]), float(ball_xy[1]), 0.80, 0, 0, 0, 1]
            for k, b in enumerate(others):
                poses[b] = [0.45, -0.45 + 0.15 * k, 0.80, 0, 0, 0, 1]
            node.set_object_poses(poses)
            node.sim_sleep(1.5)
            if node.chassis_q is not None:
                kin.chassis_now = list(node.chassis_q)

            b0 = node.object_pos(target, timeout=2.0)
            print(f"  摆放后球实际位置 {np.round(b0,4).tolist()}（期望 xy {np.round(ball_xy,4).tolist()}）")
            ok, err, nfail, trace = replay(node, kin, fk, act, target,
                                           use_ik=(p == "ik"), grasp_idx=gi)
            results[p] = ok
            msg = f"  成功={ok}"
            if err is not None and len(err):
                msg += (f"  IK往返误差 中位{np.median(err)*100:.3f}cm "
                        f"最大{err.max()*100:.3f}cm  IK失败{nfail}次")
            print(msg)
            print(f"  球位追踪 (帧, 球xyz, 球到TCP距离cm):")
            for t, bp, d in trace:
                mark = "  <-- 记录的抓取时刻" if abs(t - gi) < 3 else ""
                print(f"    f{t:4d}  {bp}  {d:6.2f}cm{mark}")
    finally:
        try:
            ex.shutdown(timeout_sec=1.0); node.destroy_node(); rclpy.shutdown()
        except Exception:
            pass

    print("\n===== 判读 =====")
    j, i = results.get("joint"), results.get("ik")
    if j and i:
        print("  A、B 都成功 -> 执行链路没问题，问题在【模型闭环】（误差累积/观测漂移）")
    elif j and i is False:
        print("  A 成功、B 失败 -> 【IK 层】是罪魁，模型无罪，去查 ik_position 的调用方式")
    elif j is False:
        print("  A 也失败 -> 【仿真状态/场景摆放/时序】有问题，先修这个再谈模型")
    else:
        print(f"  结果: {results}")


if __name__ == "__main__":
    main()
