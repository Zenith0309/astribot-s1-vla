#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripted_grasp.py — Astribot S1 仿真脚本化抓取演示（无需模型权重）

原理：
  1. 从 TF 读取目标球和容器的世界位姿（仿真按 object_names 发布）
  2. 本地加载同一份 MJCF 只做运动学（不跑物理），用阻尼最小二乘 IK 解手臂关节角
  3. 状态机：回home上方 -> 球上方 -> 下探 -> 合爪 -> 提起 -> 容器上方 -> 开爪 -> 收回
  4. 以 50Hz 向 /astribot_arm_left(gripper_left)/joint_space_command 发平滑插值指令

用法（容器内，先 source env.sh）：
  python3 scripted_grasp.py --list                 # 第一步：打印模型部件名，确认命名
  python3 scripted_grasp.py --ball ball_red        # 抓红球放进容器
  python3 scripted_grasp.py --ball ball_green --open 0.8 --close 0.0

依赖：仿真进程正在运行（with_camera 配置，object_names 含球和容器）
"""

import argparse
import math
import os
import threading
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from astribot_msgs.msg import RobotJointController, RobotJointState

# ---------------- 配置 ----------------
MJCF_PATH = "astribot_descriptions/mjcf/astribot_s1_mjcf/astribot_s1_for_aloha_with_gripper.xml"
ARM_JOINTS = [f"astribot_arm_left_joint_{i}" for i in range(1, 8)]
ARM_CMD_TOPIC = "/astribot_arm_left/joint_space_command"
GRIPPER_CMD_TOPIC = "/astribot_gripper_left/joint_space_command"
TORSO_CMD_TOPIC = "/astribot_torso/joint_space_command"

# 官方 VLA 执行系统的初始姿态（astribot_eval config）：躯干前倾、双臂抬起备操作
START_TORSO = [0.29, -0.58, 0.29, 0.0]
START_ARM_LEFT = [0.159, -0.022, -1.42, 1.66, -0.345, 0.115, 0.125]
START_ARM_RIGHT = [-0.159, -0.022, 1.42, 1.66, 0.345, 0.115, -0.125]
START_HEAD = [0.0, 0.0]
ARM_RIGHT_CMD_TOPIC = "/astribot_arm_right/joint_space_command"
HEAD_CMD_TOPIC = "/astribot_head/joint_space_command"
ARM_RIGHT_STATE_TOPIC = "/astribot_arm_right/joint_space_states"
HEAD_STATE_TOPIC = "/astribot_head/joint_space_states"
ARM_STATE_TOPIC = "/astribot_arm_left/joint_space_states"
GRIPPER_STATE_TOPIC = "/astribot_gripper_left/joint_space_states"
GRIPPER_JOINT_RANGE = 0.93          # 夹爪关节限位(rad)，用于 rad->百分比换算
TORSO_STATE_TOPIC = "/astribot_torso/joint_space_states"
TORSO_JOINTS = [f"astribot_torso_joint_{i}" for i in range(1, 5)]
CHASSIS_JOINTS = ["astribot_chassis_x", "astribot_chassis_y", "astribot_chassis_zrot"]
CHASSIS_STATE_TOPIC = "/astribot_chassis/joint_space_states"
CHASSIS_CMD_TOPIC = "/astribot_chassis/joint_space_command"
# 站位标定：球应落在底盘前向/左向的这个相对位置（当前固定站位抓取成功区间的中心，
# 即手臂已验证的甜点相对位置）。任何球位都把底盘移到复现此相对位置的站位。
STANDOFF_FWD = 0.69       # 球相对底盘的前向(x)距离
STANDOFF_LAT = 0.13       # 球相对底盘的左向(y)距离
WORLD_FRAME = "simulation"

CTRL_HZ = 50.0            # 指令发布频率（仿真时间）
STATE_HZ = 50.0           # 仿真每仿真秒发布的关节状态条数（实测：主循环50Hz每步一条）
GRASP_HOVER = 0.12        # 球上方悬停高度 (m)
GRASP_DEPTH = 0.0         # 下探目标=抓取中心对准球心 (m)
LIFT_HEIGHT = 0.20        # 提起高度 (m)
IK_ITERS = 80
IK_DAMPING = 1e-3
IK_TOL = 2e-3             # 位置收敛容差 (m)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="打印模型中的 body/site/joint 名后退出")
    p.add_argument("--ball", default="ball_red", help="目标物体 body 名 (ball_red/green/blue)")
    p.add_argument("--container", default="container", help="容器 body 名")
    p.add_argument("--site", default="arm_left_tool", help="末端参考 site（官方左臂TCP）")
    p.add_argument("--open", type=float, default=0.0, dest="grip_open", help="夹爪张开指令值(0-100, 0=全开)")
    p.add_argument("--preclose", type=float, default=35.0,
                   help="下探前预收窄开度(0-100)。收到略大于球径，避免指根/腕部下探时蹭跑球")
    p.add_argument("--close", type=float, default=70.0, dest="grip_close", help="夹爪闭合指令值(0-100, 70=适度夹持,100全闭会挤出球)")
    p.add_argument("--mjcf", default=MJCF_PATH, help="MJCF 路径")
    p.add_argument("--seg-time", type=float, default=2.5, help="每段运动时长(秒)")
    p.add_argument("--hold-min", type=float, default=62.0,
                   help="自适应闭合的保持下限(0-100)。滑落调大，挤飞调小。实测62为甜点")
    p.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("DX", "DY", "DZ"),
                   help="抓取点相对球心的补偿(m)。使用官方TCP(arm_left_tool)后默认无需补偿")
    p.add_argument("--no-base", action="store_true", help="禁用底盘站位（回到固定底盘旧行为）")
    p.add_argument("--standoff", type=float, nargs=2, default=[STANDOFF_FWD, STANDOFF_LAT],
                   metavar=("FWD", "LAT"),
                   help="站位时球相对底盘的前向/左向距离(m)，默认为甜点标定值")
    return p.parse_args()


class Kinematics:
    """本地 MuJoCo 模型，只做 FK/IK，不参与物理仿真。"""

    def __init__(self, mjcf_path, ee_hint=None):
        # 与官方 factory 一致：ASTRIBOT_SIMU_ROOT + 绝对路径，否则 include 链解析失败
        if not mjcf_path.startswith("/"):
            root = os.environ.get("ASTRIBOT_SIMU_ROOT", os.getcwd())
            mjcf_path = root + "/" + mjcf_path
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 手臂关节 -> qpos/dof 索引
        self.jnt_qpos, self.jnt_dof = [], []
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"模型中找不到关节 {name}，用 --list 检查命名")
            self.jnt_qpos.append(self.model.jnt_qposadr[jid])
            self.jnt_dof.append(self.model.jnt_dofadr[jid])
        self.chassis_qpos = []
        for name in CHASSIS_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.chassis_qpos.append(self.model.jnt_qposadr[jid] if jid >= 0 else None)
        self.torso_qpos, self.torso_dof = [], []
        for name in TORSO_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"模型中找不到躯干关节 {name}")
            self.torso_qpos.append(self.model.jnt_qposadr[jid])
            self.torso_dof.append(self.model.jnt_dofadr[jid])

        # 头部关节与头部相机（视线跟踪用）：按名字模糊匹配，找不到则禁用该功能
        head_j = []
        for j in range(self.model.njnt):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "head" in nm:
                head_j.append(self.model.jnt_qposadr[j])
        self.head_qpos = sorted(head_j)[:2]
        self.head_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgbd")
        if self.head_cam_id < 0:
            for c in range(self.model.ncam):
                cn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) or ""
                if "head" in cn:
                    self.head_cam_id = c
                    break

        # 末端：优先用左右指尖中点（真实抓取中心）；退化到单 site/body
        self.ee_pair = None
        idL = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "astribot_gripper_left_Link_L2")
        idR = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "astribot_gripper_left_Link_R2")
        if ee_hint is None and idL >= 0 and idR >= 0:
            self.ee_pair = (idL, idR)
            self.ee_is_site, self.ee_id = False, idR
            print("[kin] 末端参考: 左右指尖(Link_L2/Link_R2)中点 = 抓取中心")
        else:
            self.ee_is_site, self.ee_id = self._find_ee(ee_hint)
            kind = "site" if self.ee_is_site else "body"
            name = mujoco.mj_id2name(self.model,
                                     mujoco.mjtObj.mjOBJ_SITE if self.ee_is_site else mujoco.mjtObj.mjOBJ_BODY,
                                     self.ee_id)
            print(f"[kin] 末端参考: {kind} '{name}'")

    def _find_ee(self, hint):
        def names(objtype, n):
            return [(i, mujoco.mj_id2name(self.model, objtype, i)) for i in range(n)]
        sites = names(mujoco.mjtObj.mjOBJ_SITE, self.model.nsite)
        bodies = names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        if hint:
            for i, n in sites:
                if n == hint:
                    return True, i
            for i, n in bodies:
                if n == hint:
                    return False, i
            raise RuntimeError(f"找不到指定末端 '{hint}'")
        # 自动探测：优先 site，名字含 gripper_left 或 effector_left
        for key in ("gripper_left", "left_gripper", "effector_left", "left_effector"):
            for i, n in sites:
                if n and key in n:
                    return True, i
        for key in ("gripper_left", "left_gripper"):
            cands = [(i, n) for i, n in bodies if n and key in n]
            if cands:
                # 取运动链最深的（child 最多层）近似为指尖
                return False, cands[-1][0]
        raise RuntimeError("自动探测末端失败，用 --list 查看命名后通过 --site 指定")

    def list_names(self):
        for label, objtype, n in [("BODY", mujoco.mjtObj.mjOBJ_BODY, self.model.nbody),
                                  ("SITE", mujoco.mjtObj.mjOBJ_SITE, self.model.nsite),
                                  ("JOINT", mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)]:
            print(f"===== {label} =====")
            for i in range(n):
                print(" ", mujoco.mj_id2name(self.model, objtype, i))

    def sync(self, arm_q, torso_q=None, chassis_q=None):
        for adr, q in zip(self.jnt_qpos, arm_q):
            self.data.qpos[adr] = q
        if torso_q is not None:
            for adr, q in zip(self.torso_qpos, torso_q):
                if adr is not None:
                    self.data.qpos[adr] = q
        if chassis_q is not None:
            for adr, q in zip(self.chassis_qpos, chassis_q):
                if adr is not None:
                    self.data.qpos[adr] = q
        mujoco.mj_forward(self.model, self.data)

    def ee_pose(self):
        if self.ee_pair is not None:
            pos = 0.5 * (self.data.xpos[self.ee_pair[0]] + self.data.xpos[self.ee_pair[1]])
            return pos.copy(), None
        if self.ee_is_site:
            pos = self.data.site_xpos[self.ee_id].copy()
            mat = self.data.site_xmat[self.ee_id].copy()
        else:
            pos = self.data.xpos[self.ee_id].copy()
            mat = self.data.xmat[self.ee_id].copy()
        return pos, mat

    def ee_jacp(self, jacp, jacr):
        if self.ee_pair is not None:
            j1 = np.zeros_like(jacp); j2 = np.zeros_like(jacp)
            jr = np.zeros_like(jacr)
            mujoco.mj_jacBody(self.model, self.data, j1, jr, self.ee_pair[0])
            mujoco.mj_jacBody(self.model, self.data, j2, jr, self.ee_pair[1])
            jacp[:] = 0.5 * (j1 + j2)
        elif self.ee_is_site:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

    def ik_position(self, target_pos, arm_q_init, torso_q_init):
        """位置 IK：躯干4 + 手臂7 共 11 自由度（躯干步长减半，类似 WBC 让躯干配合前倾）。
        返回 (成功, 躯干4关节, 手臂7关节)。"""
        qa = np.array(arm_q_init, dtype=float)
        qt = np.array(torso_q_init, dtype=float)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        dofs = list(self.torso_dof) + list(self.jnt_dof)
        chassis = getattr(self, "chassis_now", None)
        tgt_mat = getattr(self, "target_mat", None)   # 3x3, 姿态锁定目标（None=仅位置IK）
        for _ in range(IK_ITERS):
            self.sync(qa, qt, chassis)
            pos, mat = self.ee_pose()
            err_p = np.asarray(target_pos) - pos
            use_ori = tgt_mat is not None and mat is not None
            if use_ori:
                R_c = np.asarray(mat).reshape(3, 3)
                ori_err = Rotation.from_matrix(tgt_mat @ R_c.T).as_rotvec()
                done = np.linalg.norm(err_p) < IK_TOL and np.linalg.norm(ori_err) < 0.1
            else:
                done = np.linalg.norm(err_p) < IK_TOL
            if done:
                return True, qt, qa
            self.ee_jacp(jacp, jacr)
            if use_ori:
                err = np.concatenate([err_p, 0.5 * ori_err])
                J = np.vstack([jacp[:, dofs], jacr[:, dofs]])   # 6x11
                JJT = J @ J.T + IK_DAMPING * np.eye(6)
            else:
                err = err_p
                J = jacp[:, dofs]                               # 3x11
                JJT = J @ J.T + IK_DAMPING * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            dq[:4] *= 0.5                           # 躯干动得保守些
            qt += np.clip(dq[:4], -0.1, 0.1)
            qa += np.clip(dq[4:], -0.2, 0.2)
        self.sync(qa, qt)
        pos, _ = self.ee_pose()
        final_err = float(np.linalg.norm(np.asarray(target_pos) - pos))
        print(f"[ik] 未完全收敛, 残差 {final_err*1000:.1f} mm, 末端位于 {np.round(pos,3)}, 目标 {np.round(np.asarray(target_pos),3)}")
        return final_err < 0.03, qt, qa


    # ---- 头部视线跟踪（2-DoF 数值解） ----
    def _head_gaze_err(self, hq, target):
        """误差 = 1 - cos(相机光轴, 指向目标向量)。"""
        for adr, v in zip(self.head_qpos, hq):
            self.data.qpos[adr] = v
        mujoco.mj_forward(self.model, self.data)
        p = self.data.cam_xpos[self.head_cam_id]
        mat = self.data.cam_xmat[self.head_cam_id].reshape(3, 3)
        f = -mat[:, 2]                      # MuJoCo 相机朝 -Z 看
        d = np.asarray(target, float) - p
        n = np.linalg.norm(d)
        if n < 1e-6:
            return 0.0
        return 1.0 - float(np.dot(f, d / n))

    def head_look_at(self, target, head_now):
        """求让头部相机光轴指向 target 的 [yaw, pitch]（坐标下降，热启动后很快）。"""
        if self.head_cam_id < 0 or len(self.head_qpos) < 2:
            return list(head_now[:2])
        lim = np.array([[-1.57, 1.57], [-1.22, 1.22]])   # 头部关节限位
        q = np.clip(np.asarray(head_now[:2], float), lim[:, 0], lim[:, 1])
        step = 0.12
        base = self._head_gaze_err(q, target)
        for _ in range(14):        # 热启动下足够收敛；限制次数避免吃满CPU干扰控制时序
            improved = False
            for j in (0, 1):
                for s in (step, -step):
                    qt = q.copy()
                    qt[j] = np.clip(qt[j] + s, lim[j, 0], lim[j, 1])
                    e = self._head_gaze_err(qt, target)
                    if e < base - 1e-7:
                        q, base, improved = qt, e, True
            if not improved:
                step *= 0.5
                if step < 0.008:
                    break
        return q.tolist()


class GraspNode(Node):
    def __init__(self, args):
        super().__init__("scripted_grasp")
        self.args = args
        self.arm_pub = self.create_publisher(RobotJointController, ARM_CMD_TOPIC, 10)
        self.grip_pub = self.create_publisher(RobotJointController, GRIPPER_CMD_TOPIC, 10)
        self.torso_pub = self.create_publisher(RobotJointController, TORSO_CMD_TOPIC, 10)
        self.arm_r_pub = self.create_publisher(RobotJointController, ARM_RIGHT_CMD_TOPIC, 10)
        self.head_pub = self.create_publisher(RobotJointController, HEAD_CMD_TOPIC, 10)
        self.chassis_pub = self.create_publisher(RobotJointController, CHASSIS_CMD_TOPIC, 10)
        self.arm_r_q, self.head_q = None, None
        self.create_subscription(RobotJointState, ARM_RIGHT_STATE_TOPIC,
                                 lambda m: setattr(self, "arm_r_q", list(m.position)[:7]), 10)
        self.create_subscription(RobotJointState, HEAD_STATE_TOPIC,
                                 lambda m: setattr(self, "head_q", list(m.position)[:2]), 10)
        self.arm_q = None
        self.torso_q = None
        self.chassis_q = None
        self.create_subscription(RobotJointState, ARM_STATE_TOPIC, self._arm_cb, 10)
        self.create_subscription(RobotJointState, TORSO_STATE_TOPIC, self._torso_cb, 10)
        self.create_subscription(RobotJointState, CHASSIS_STATE_TOPIC,
                                 lambda m: setattr(self, "chassis_q", list(m.position)[:3]), 10)
        self.grip_state = 0.0     # 夹爪关节实际位置
        self.grip_torque = 0.0    # 夹爪关节实际力矩
        self.create_subscription(RobotJointState, GRIPPER_STATE_TOPIC, self._grip_cb, 10)
        # 物体位姿：直接订阅 /tf 原始消息自行解析（绕开 tf2 监听器机制）
        self.obj_pose = {}        # child_frame_id -> ([x,y,z], 收到时间)
        self.create_subscription(TFMessage, "/tf", self._tf_cb, 50)
        # 头部视线跟踪
        self.track_object = None
        self._head_cmd = None
        # 仿真时钟：以关节状态消息计数（共享服务器上实时率会波动，
        # 所有轨迹按仿真时间定步，不受墙钟影响）
        self._tick = 0

    def start_head_track(self, kin_head, hz=10.0, max_step=0.08):
        """10Hz 后台线程：track_object 置为物体名即让头部相机跟踪它，置 None 停止。
        kin_head 须为独立 Kinematics 实例（避免与主线程 IK 共用 MjData 竞态）。
        惰性求解：仅当目标或躯干明显移动时才重解视线 IK，球静止时开销≈0，
        避免与 50Hz 控制循环/录制线程抢 CPU 导致控制时序抖动。"""
        def _loop():
            period = 1.0 / hz
            last_pt = last_torso = last_sol = None
            while getattr(self, "_head_on", True):
                t0 = time.time()
                name = self.track_object
                if name and self.torso_q is not None:
                    ent = self.obj_pose.get(name)
                    if ent is not None and time.time() - ent[1] < 1.0:
                        try:
                            pt = np.asarray(ent[0])
                            torso = np.asarray(self.torso_q[:4])
                            cur = np.asarray(self._head_cmd if self._head_cmd is not None
                                             else (self.head_q or [0.0, 0.0]))[:2]
                            need = (last_sol is None
                                    or np.linalg.norm(pt - last_pt) > 0.01
                                    or np.linalg.norm(torso - last_torso) > 0.02)
                            if need:
                                kin_head.sync(self.arm_q, self.torso_q, self.chassis_q)
                                last_sol = np.asarray(kin_head.head_look_at(pt, cur))
                                last_pt, last_torso = pt, torso
                            cmd = np.clip(last_sol, cur - max_step, cur + max_step)
                            if np.linalg.norm(cmd - cur) > 1e-4:   # 未到位才继续发
                                self._head_cmd = cmd.tolist()
                                self._pub_part(self.head_pub, self._head_cmd)
                        except Exception as e:
                            print(f"[head] 跟踪异常: {e}", flush=True)
                else:
                    last_pt = last_torso = last_sol = None
                time.sleep(max(0.0, period - (time.time() - t0)))
        self._head_on = True
        threading.Thread(target=_loop, daemon=True).start()

    def _grip_cb(self, m):
        if m.position:
            self.grip_state = float(m.position[0])
        if m.torque:
            self.grip_torque = float(m.torque[0])

    def grip_pct(self):
        """夹爪实际位置，统一为百分比（量纲在标定时确定，不再逐次猜测）。"""
        v = self.grip_state
        if getattr(self, "_grip_is_pct", None) is False:
            return v / GRIPPER_JOINT_RANGE * 100.0
        return v

    def calibrate_gripper(self, max_close=85.0, step=2.0):
        """空载标定：闭合扫描记录 torque_empty(cmd) 曲线，同时确定状态量纲。
        须在夹爪空载（手举空中）时调用一次。"""
        print("[grip] 空载标定中...")
        self.empty_tq = {}
        self.send_grip(0.0)
        self.sim_sleep(0.5)
        cmd = 0.0
        while cmd <= max_close:
            self.send_grip(cmd)
            self.sim_sleep(0.05)
            self.empty_tq[round(cmd)] = self.grip_torque
            cmd += step
        self._grip_is_pct = self.grip_state > 2.0   # 满程时观测原始值确定量纲
        self.send_grip(0.0)
        self.sim_sleep(0.5)
        vals = list(self.empty_tq.values())
        print(f"[grip] 标定完成: 量纲={'百分比' if self._grip_is_pct else '弧度'}, "
              f"空载力矩范围 [{min(vals):.3f}, {max(vals):.3f}]")

    def empty_torque_at(self, cmd):
        tbl = getattr(self, "empty_tq", None)
        if not tbl:
            return 0.0
        key = min(tbl.keys(), key=lambda k: abs(k - cmd))
        return tbl[key]

    def _tf_cb(self, msg):
        now = time.time()
        for tr in msg.transforms:
            t = tr.transform.translation
            self.obj_pose[tr.child_frame_id] = ([t.x, t.y, t.z], now)

    def pump(self, timeout=0.0):
        """回调泵：外部执行器接管时仅休眠，否则手动 spin 一次。"""
        if getattr(self, "external_spin", False):
            if timeout > 0:
                time.sleep(timeout)
            return
        rclpy.spin_once(self, timeout_sec=timeout)

    def _arm_cb(self, msg):
        self.arm_q = list(msg.position)[:7]
        self._tick += 1

    def sim_sleep(self, dt):
        """按仿真时间等待 dt 秒（以状态消息为时钟）。仿真变慢时自动拉长墙钟等待，
        保证轨迹在仿真世界里始终是设计速度。带墙钟兜底防状态流中断时卡死。"""
        target = self._tick + max(1, int(dt * STATE_HZ))
        deadline = time.time() + max(1.0, dt * 30)
        while self._tick < target and time.time() < deadline:
            self.pump(0.002)

    def _torso_cb(self, msg):
        self.torso_q = list(msg.position)[:4]

    def wait_ready(self, timeout=10.0):
        t0 = time.time()
        while rclpy.ok() and (self.arm_q is None or self.torso_q is None):
            self.pump(0.1)
            if time.time() - t0 > timeout:
                raise RuntimeError("等待关节状态超时，确认仿真在跑且 ROS_DOMAIN_ID=25")
        print(f"[node] 关节状态就绪 arm={np.round(self.arm_q,3)}")

    def object_pos(self, name, timeout=5.0):
        t0 = time.time()
        while rclpy.ok():
            ent = self.obj_pose.get(name)
            if ent is not None and time.time() - ent[1] < 1.0:   # 1秒内的新鲜数据
                return np.array(ent[0])
            self.pump(0.05)
            if time.time() - t0 > timeout:
                raise RuntimeError(f"TF 查不到 '{name}'（frame {WORLD_FRAME}），"
                                   f"确认 yaml 的 object_names 包含它")

    def send_arm(self, q):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v) for v in q]
        self.arm_pub.publish(msg)

    def send_grip(self, v):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v)]
        self.grip_pub.publish(msg)

    def send_torso(self, q):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v) for v in q]
        self.torso_pub.publish(msg)

    def _pub_part(self, pub, q):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v) for v in q]
        pub.publish(msg)

    def _move_all(self, targets, duration):
        """全身多部件同步 min-jerk 插值。targets: {"arm":..,"torso":..,"arm_r":..,"head":..}"""
        cur = {
            "arm":   np.asarray(self.arm_q if self.arm_q else [0.0]*7),
            "torso": np.asarray(self.torso_q if self.torso_q else [0.0]*4),
            "arm_r": np.asarray(self.arm_r_q if self.arm_r_q else [0.0]*7),
            "head":  np.asarray(self.head_q if self.head_q else [0.0]*2),
        }
        tgt = {k: np.asarray(v) for k, v in targets.items()}
        steps = max(2, int(duration * CTRL_HZ))
        for i in range(steps + 1):
            s = i / steps
            k = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.send_arm(cur["arm"] + k * (tgt["arm"] - cur["arm"]))
            self.send_torso(cur["torso"] + k * (tgt["torso"] - cur["torso"]))
            self._pub_part(self.arm_r_pub, cur["arm_r"] + k * (tgt["arm_r"] - cur["arm_r"]))
            self._pub_part(self.head_pub, cur["head"] + k * (tgt["head"] - cur["head"]))
            self.sim_sleep(1.0 / CTRL_HZ)

    def move_to_start(self, duration=4.0):
        """全身（躯干前倾+双臂抬起+头部归零）移动到官方初始姿态。"""
        self._move_all({"arm": START_ARM_LEFT, "torso": START_TORSO,
                        "arm_r": START_ARM_RIGHT, "head": START_HEAD}, duration)

    def go_home(self, duration=3.0):
        """全身平滑回零位（与仿真 keyframe 一致），复位前调用可避免执行器目标跳变打爆物理。"""
        self._move_all({"arm": [0.0]*7, "torso": [0.0]*4,
                        "arm_r": [0.0]*7, "head": [0.0]*2}, duration)
        self.send_grip(0.0)

    def send_chassis(self, q):
        msg = RobotJointController()
        msg.mode = 1
        msg.command = [float(v) for v in q]     # [x, y, zrot]
        self.chassis_pub.publish(msg)

    def move_base_to(self, target_xyyaw, duration=3.0):
        """底盘 min-jerk 平移/转向到目标 [x, y, yaw]（世界系）。"""
        cur = np.asarray(self.chassis_q[:3] if self.chassis_q is not None else [0.0, 0.0, 0.0])
        tgt = np.asarray(target_xyyaw, dtype=float)
        steps = max(2, int(duration * CTRL_HZ))
        for i in range(steps + 1):
            s = i / steps
            k = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.send_chassis(cur + k * (tgt - cur))
            self.sim_sleep(1.0 / CTRL_HZ)

    def station_for_ball(self, ball_xy, fwd=STANDOFF_FWD, lat=STANDOFF_LAT):
        """算把球摆到手臂甜点相对位置所需的底盘站位 [x, y, yaw=0]。
        底盘世界系：base_x = ball_x - fwd（球在前方 fwd 处），base_y = ball_y - lat。"""
        return [float(ball_xy[0]) - fwd, float(ball_xy[1]) - lat, 0.0]

    def move_smooth(self, a_from, a_to, t_from, t_to, duration):
        """手臂+躯干同步 min-jerk 插值，50Hz 发布。"""
        a_from, a_to = np.asarray(a_from), np.asarray(a_to)
        t_from, t_to = np.asarray(t_from), np.asarray(t_to)
        steps = max(2, int(duration * CTRL_HZ))
        for i in range(steps + 1):
            s = i / steps
            k = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.send_arm(a_from + k * (a_to - a_from))
            self.send_torso(t_from + k * (t_to - t_from))
            self.sim_sleep(1.0 / CTRL_HZ)

    def adaptive_close(self, start=0.0, max_close=100.0, step=2.0,
                       squeeze=8.0, hold_min=60.0, force_th=0.15,
                       stall_gap=6.0, stall_n=4, verbose=True):
        """自适应闭合：以"相对空载曲线的力矩增量"为主接触信号（等效电机电流检测），
        位置停滞为兜底。接触需连续 3 个周期确认，避免启动瞬态/噪声误触发。
        依赖 calibrate_gripper() 预先标定的空载曲线（未标定则退化为零基线）。"""
        cmd, stall, contact_count = start, 0, 0
        while cmd < max_close:
            cmd = min(cmd + step, max_close)
            self.send_grip(cmd)
            self.sim_sleep(0.05)          # 仿真时间等待，确保力矩反馈与指令同步
            actual_pct = self.grip_pct()
            dtq = abs(self.grip_torque - self.empty_torque_at(cmd))
            contact_count = contact_count + 1 if dtq > force_th else 0
            if verbose and int(cmd) % 10 == 0:
                print(f"[grip] cmd={cmd:5.1f}%  actual={actual_pct:5.1f}%  Δtorque={dtq:.3f}")
            if cmd - actual_pct > stall_gap:
                stall += 1
            else:
                stall = 0
            if contact_count >= 3 or stall >= stall_n:
                reason = "力矩" if contact_count >= 3 else "位置停滞"
                # 保持值 = max(接触点+余量, 下限60)：实测夹稳球需要约60%
                hold = min(max(cmd + squeeze, hold_min), max_close)
                self.last_contact_dtq = max(dtq, force_th)   # 供握力守护做参考
                print(f"[grip] 接触({reason}) cmd={cmd:.0f}% actual={actual_pct:.0f}% "
                      f"Δtorque={dtq:.3f} -> 保持 {hold:.0f}%")
                self.hold_grip(hold, 0.8)
                return hold
        print(f"[grip] 未检测到接触，闭合到 {max_close:.0f}%（可能没夹到东西）")
        self.hold_grip(max_close, 0.5)
        return max_close

    def hold_grip(self, value, duration=1.0):
        steps = int(duration * CTRL_HZ)
        for _ in range(steps):
            self.send_grip(value)
            self.sim_sleep(1.0 / CTRL_HZ)

    # ---- 握力守护：只救明确滑移，限幅防挤飞 ----
    def start_grip_guard(self, hold, ref_dtq=None, max_bump=6.0, hz=10.0):
        """仅位置信号：实际开度较基准继续收拢 >2.5%（球明确在外滑）时 +3%。
        总加紧不超过初始保持值 +max_bump——提起后的力矩/开度正常波动不应触发
        （力矩信号在运动中噪声大、会把好球挤飞，已弃用）。"""
        self._guard_on = True
        self._guard_cmd = float(hold)
        cap = min(float(hold) + max_bump, 85.0)
        def _loop():
            base_pct = self.grip_pct()
            while getattr(self, "_guard_on", False):
                try:
                    pct = self.grip_pct()
                    if pct - base_pct > 2.5 and self._guard_cmd < cap:
                        self._guard_cmd = min(self._guard_cmd + 3.0, cap)
                        self.send_grip(self._guard_cmd)
                        base_pct = pct
                        print(f"[grip-guard] 位置滑移 pct={pct:.0f}% "
                              f"-> 加紧至 {self._guard_cmd:.0f}% (上限{cap:.0f})", flush=True)
                except Exception:
                    pass
                time.sleep(1.0 / hz)
        threading.Thread(target=_loop, daemon=True).start()

    def stop_grip_guard(self):
        self._guard_on = False


def main():
    args = parse_args()
    kin = Kinematics(args.mjcf, args.site)
    if args.list:
        kin.list_names()
        return

    rclpy.init()
    node = GraspNode(args)
    try:
        node.wait_ready()
        ball = node.object_pos(args.ball)
        box = node.object_pos(args.container)
        print(f"[task] {args.ball} @ {np.round(ball,3)}  ->  {args.container} @ {np.round(box,3)}")

        if node.chassis_q is not None:
            kin.chassis_now = list(node.chassis_q)
            print(f"[info] 底盘位姿 (x,y,rot) = {np.round(node.chassis_q,4)} 已纳入运动学")
        print("[state] 0/6 移动到初始姿态（躯干前倾+抬臂）")
        node.move_to_start(4.0)
        q_now = list(START_ARM_LEFT)
        torso_now = list(START_TORSO)

        # 底盘站位：根据球位移动底盘，把球摆到手臂甜点相对位置（近身球不再大幅弯腰）
        if not args.no_base:
            station = node.station_for_ball(ball[:2], args.standoff[0], args.standoff[1])
            print(f"[state] 0b 底盘站位 -> (x,y,yaw)={np.round(station,3)} (球@{np.round(ball[:2],3)})")
            node.move_base_to(station, duration=3.0)
            node.sim_sleep(0.5)
            if node.chassis_q is not None:
                kin.chassis_now = list(node.chassis_q)
                print(f"[info] 站位后底盘实际 = {np.round(node.chassis_q,3)}，已纳入运动学")
            ball = node.object_pos(args.ball)   # 底盘动后球的世界位姿不变，刷新一下取新鲜值
            box = node.object_pos(args.container)

        if not hasattr(node, "empty_tq"):
            node.calibrate_gripper()          # 空载力矩曲线标定（每进程一次，手举空中时）

        # 头部相机跟踪目标球（独立 Kinematics 实例避免与 IK 竞态）
        head_kin = Kinematics(args.mjcf, args.site)
        node.start_head_track(head_kin)
        node.track_object = args.ball
        print(f"[head] 头部相机开始跟踪 {args.ball}")
        time.sleep(1.0)                       # 等视线到位

        # 姿态锁定：以初始姿态下工具的朝向作为所有抓取动作的目标姿态（保证竖直下探不斜碰桌面）
        kin.sync(q_now, torso_now, getattr(kin, "chassis_now", None))
        _, m0 = kin.ee_pose()
        if m0 is not None:
            kin.target_mat = np.asarray(m0).reshape(3, 3).copy()
            print("[kin] 已锁定工具姿态（6D IK）")
        T = args.seg_time

        def goto(pos, label, T_seg=None):
            nonlocal q_now, torso_now
            ok, t_target, q_target = kin.ik_position(pos, q_now, torso_now)
            if not ok:
                raise RuntimeError(f"IK 失败于阶段 [{label}] 目标 {np.round(pos,3)}")
            print(f"[state] {label}")
            node.move_smooth(q_now, q_target, torso_now, t_target, T_seg or T)
            q_now = list(q_target)
            torso_now = list(t_target)

        off = np.asarray(args.offset)
        node.hold_grip(args.grip_open, 0.8)                                  # 预张开
        goto(ball + off + [0, 0, GRASP_HOVER], "1/6 移动到球上方")
        node.hold_grip(args.preclose, 0.6)                                   # 预收窄，缩小手指轮廓
        goto(ball + off + [0, 0, 0.05], "2/6 下探(快段)")
        goto(ball + off, "2/6 下探(慢段)", T_seg=2.5)

        # --- 实测诊断：用真实关节状态算抓取中心，和瞄准点（球心+补偿）对比 ---
        node.sim_sleep(0.6)
        kin.sync(node.arm_q, node.torso_q, node.chassis_q)
        real_ee, _ = kin.ee_pose()
        ball_now = node.object_pos(args.ball)
        aim = ball_now + off                  # 瞄准点含补偿，修正不应把补偿抵消
        delta = real_ee - aim
        print(f"[debug] 抓取中心实际 {np.round(real_ee,3)} | 瞄准点 {np.round(aim,3)} | 偏差(x,y,z) {np.round(delta,3)}")

        # 闭环修正：稳态误差随构型变化，实测后反向补偿一步
        if np.linalg.norm(delta) > 0.008:
            print(f"[state] 2b/6 闭环修正 (补偿 {np.round(-delta,3)})")
            goto(aim - delta, "2b/6 修正下探")
            node.sim_sleep(0.4)
            kin.sync(node.arm_q, node.torso_q, node.chassis_q)
            real_ee2, _ = kin.ee_pose()
            print(f"[debug] 修正后抓取中心 {np.round(real_ee2,3)} | 残余偏差 {np.round(real_ee2-aim,3)}")

        print("[state] 3/6 自适应闭合夹爪")
        hold = node.adaptive_close(start=args.preclose, max_close=85.0,
                                   hold_min=args.hold_min)
        node.start_grip_guard(hold, getattr(node, "last_contact_dtq", 0.5))
        goto(ball + off + [0, 0, LIFT_HEIGHT], "4/6 提起")   # 带补偿竖直提起
        goto(box + [0, 0, LIFT_HEIGHT], "5/6 移到容器上方")
        goto(box + [0, 0, 0.10], "6a/6 下放至容器口", T_seg=1.5)   # 低位释放，避免球弹出/甩飞
        print("[state] 6/6 松开夹爪")
        node.stop_grip_guard()
        node.hold_grip(args.grip_open, 1.2)
        goto(box + [0, 0, LIFT_HEIGHT + 0.05], "收回")
        print("[done] 抓取流程结束，看 Foxglove/窗口验证球是否入箱")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
