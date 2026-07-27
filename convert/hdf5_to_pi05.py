#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_to_pi05.py — 把采集的 HDF5 转成官方 pi05 要求的 LeRobot 数据集（笛卡尔 SO3）

与 ACT 版转换（hdf5_to_lerobot.py）的区别：pi05 的动作空间是 31 维全身笛卡尔
SO3，需要用 FK 把关节动作转成末端位姿。本脚本在【训练环境】运行，且需要能加载
仿真 MJCF（设 ASTRIBOT_SIMU_ROOT 指向 astribot_simulation 根目录）。

31 维布局（官方 base_processor 固定值，state 与 command 同布局）：
  [0:9]   torso        SO3 位姿 (x,y,z + 旋转矩阵前两行6D)
  [9:18]  left_arm     SO3 位姿
  [18:19] left_gripper 标量 (0-1)
  [19:28] right_arm    SO3 位姿
  [28:29] right_gripper标量
  [29:31] head         2 关节

采集数据只动左臂+躯干，右臂全程恒为官方 start pose、右爪常开、头部跟踪：
  - torso / left_arm 位姿：由记录的关节值做 FK
  - right_arm 位姿：由常量 START_ARM_RIGHT 做 FK（右臂采集期间未动，精确）
  - left_gripper：记录值 /100；right_gripper：0；head：见 --head 选项

数据列（标准 LeRobot，pi05 dataloader v20 直接读取）：
  cartesian_so3_dict.cartesian_pose_state / .cartesian_pose_command  (31,)
  images_dict.head.rgb / .left_wrist.rgb / .right_wrist.rgb          (224,224,3)
  prompt (task)

用法：
  export ASTRIBOT_SIMU_ROOT=/data0/tzs/astribot/astribot_simulation
  python convert/hdf5_to_pi05.py --src /workspace/demos \
      --root ./datasets/pi05_pick_ball --repo-id local/astribot_pick_ball
"""

import argparse
import glob
import os
import re
import sys

import cv2
import h5py
import numpy as np
import mujoco

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 采集时的固定量（与 scripted_grasp.py 一致）
MJCF_REL = "astribot_descriptions/mjcf/astribot_s1_mjcf/astribot_s1_for_aloha_with_gripper.xml"
START_ARM_RIGHT = [-0.159, -0.022, 1.42, 1.66, 0.345, 0.115, -0.125]
ARM_L_JOINTS = [f"astribot_arm_left_joint_{i}" for i in range(1, 8)]
ARM_R_JOINTS = [f"astribot_arm_right_joint_{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"astribot_torso_joint_{i}" for i in range(1, 5)]

CAM_RE = re.compile(r"camera_(.+?)_image")
# HDF5 相机短名 -> pi05 列名
CAM_MAP = {
    "head_rgbd": "images_dict.head.rgb",
    "left_wrist_rgbd": "images_dict.left_wrist.rgb",
    "right_wrist_rgbd": "images_dict.right_wrist.rgb",
}
STATE_KEY = "cartesian_so3_dict.cartesian_pose_state"
ACTION_KEY = "cartesian_so3_dict.cartesian_pose_command"
POSE_DIM = 9


def mat_to_so3(R):
    """3x3 旋转矩阵 -> 6D（前两行展平），与官方 geometry.matrix_to_so3 一致。"""
    return R[:2, :].reshape(6)


class FK:
    """本地 MuJoCo 前向运动学：给定关节角，读取各末端 site 世界位姿。
    固定底盘（chassis=0）下世界系≈base 系，符合 pi05 state frame_ref='base'。"""

    def __init__(self, mjcf_path=None):
        if mjcf_path is None:
            root = os.environ.get("ASTRIBOT_SIMU_ROOT")
            if not root:
                sys.exit("[err] 请设置 ASTRIBOT_SIMU_ROOT 指向 astribot_simulation 根目录")
            mjcf_path = root + "/" + MJCF_REL
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.q_l = [self._qadr(n) for n in ARM_L_JOINTS]
        self.q_r = [self._qadr(n) for n in ARM_R_JOINTS]
        self.q_t = [self._qadr(n) for n in TORSO_JOINTS]
        self.site_l = self._site(["arm_left_tool"])
        self.site_r = self._site(["arm_right_tool"])
        # 躯干参考系：优先专用 site，退化到躯干末节 body
        self.torso_kind, self.torso_id = self._torso_frame()
        print(f"[fk] 躯干参考系: {self.torso_kind} id={self.torso_id} | "
              f"left_site={self.site_l} right_site={self.site_r}")

    def _qadr(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            sys.exit(f"[err] MJCF 找不到关节 {name}")
        return self.model.jnt_qposadr[jid]

    def _site(self, names):
        for n in names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            if sid >= 0:
                return sid
        sys.exit(f"[err] MJCF 找不到 site {names}")

    def _torso_frame(self):
        for n in ("torso_tool", "torso_link", "astribot_torso_tool"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            if sid >= 0:
                return "site", sid
        # 退化：躯干最后一节 body（本模型命名为 astribot_torso_link_4，带下划线）
        for n in ("astribot_torso_link_4", "astribot_torso_link_3",
                  "astribot_torso_link_2", "astribot_torso_link_1",
                  "astribot_torso_link4", "astribot_torso_link3"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                return "body", bid
        sys.exit("[err] MJCF 找不到躯干参考系（torso site/body），请手工指定")

    def pose9(self, torso4, arm_l7, arm_r7):
        """返回 (torso_pose9, left_pose9, right_pose9)，均为世界系 SO3。"""
        for adr, v in zip(self.q_t, torso4):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_l, arm_l7):
            self.data.qpos[adr] = v
        for adr, v in zip(self.q_r, arm_r7):
            self.data.qpos[adr] = v
        mujoco.mj_forward(self.model, self.data)
        if self.torso_kind == "site":
            tp = self.data.site_xpos[self.torso_id]
            tR = self.data.site_xmat[self.torso_id].reshape(3, 3)
        else:
            tp = self.data.xpos[self.torso_id]
            tR = self.data.xmat[self.torso_id].reshape(3, 3)
        lp = self.data.site_xpos[self.site_l]
        lR = self.data.site_xmat[self.site_l].reshape(3, 3)
        rp = self.data.site_xpos[self.site_r]
        rR = self.data.site_xmat[self.site_r].reshape(3, 3)
        return (np.concatenate([tp, mat_to_so3(tR)]),
                np.concatenate([lp, mat_to_so3(lR)]),
                np.concatenate([rp, mat_to_so3(rR)]))


def build_vec(fk, torso4, arm7, grip, head2):
    """组装 31 维 SO3 向量。右臂恒为 start pose，右爪 0。"""
    tp, lp, rp = fk.pose9(torso4, arm7, START_ARM_RIGHT)
    v = np.zeros(31, dtype=np.float32)
    v[0:9] = tp
    v[9:18] = lp
    v[18] = grip / 100.0
    v[19:28] = rp
    v[28] = 0.0
    v[29:31] = head2
    return v


def load_episode(path, resize, head_val):
    with h5py.File(path, "r") as f:
        state = np.asarray(f["state"], dtype=np.float32)      # (T,12) torso4+arm7+grip1
        action = np.asarray(f["action"], dtype=np.float32)
        imgs = {}
        for key in f["images"]:
            m = CAM_RE.search(key)
            short = m.group(1) if m else key
            if short not in CAM_MAP:
                continue
            frames = []
            for raw in f["images"][key]:
                buf = np.asarray(raw, dtype=np.uint8)
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
                if im is None:
                    im = frames[-1].copy() if frames else np.zeros((resize, resize, 3), np.uint8)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (resize, resize), interpolation=cv2.INTER_AREA)
                frames.append(im)
            imgs[CAM_MAP[short]] = frames
        prompt = f.attrs.get("prompt", "pick the ball and place it into the box")
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
    T = min([len(state), len(action)] + [len(v) for v in imgs.values()])
    head2 = np.asarray(head_val, dtype=np.float32)
    return state[:T], action[:T], {k: v[:T] for k, v in imgs.items()}, str(prompt), head2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default="./datasets/pi05_pick_ball")
    ap.add_argument("--repo-id", default="local/astribot_pick_ball")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--include-fail", action="store_true")
    ap.add_argument("--head", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("YAW", "PITCH"),
                    help="头部关节常量（采集未记录头部；抓放任务头部动作无关，默认0）")
    args = ap.parse_args()

    pattern = "*.h5" if args.include_fail else "*_ok.h5"
    files = sorted(glob.glob(os.path.join(args.src, pattern)))
    if not files:
        sys.exit(f"[err] {args.src} 下没有 {pattern}")
    print(f"[info] 待转换 {len(files)} 条")

    fk = FK()

    features = {
        STATE_KEY: {"dtype": "float32", "shape": (31,), "names": None},
        ACTION_KEY: {"dtype": "float32", "shape": (31,), "names": None},
    }
    for col in CAM_MAP.values():
        features[col] = {"dtype": "video", "shape": (args.resize, args.resize, 3),
                         "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps,
                                    root=args.root, features=features, use_videos=True)

    n_frames = 0
    for i, path in enumerate(files):
        state, action, imgs, prompt, head2 = load_episode(path, args.resize, args.head)
        if set(imgs) != set(CAM_MAP.values()):
            print(f"[warn] {os.path.basename(path)} 相机不齐，跳过")
            continue
        T = len(state)
        if T == 0:
            print(f"[warn] {os.path.basename(path)} 空 episode，跳过")
            continue
        for t in range(T):
            svec = build_vec(fk, state[t, :4], state[t, 4:11], state[t, 11], head2)
            avec = build_vec(fk, action[t, :4], action[t, 4:11], action[t, 11], head2)
            # 严格 dtype：state/action float32，图像 uint8 连续内存（bundled lerobot 敏感）
            frame = {STATE_KEY: np.asarray(svec, dtype=np.float32),
                     ACTION_KEY: np.asarray(avec, dtype=np.float32)}
            for col in CAM_MAP.values():
                frame[col] = np.ascontiguousarray(imgs[col][t], dtype=np.uint8)
            # sub_task_index=[距子任务起点帧数, 距终点帧数]。单任务 episode 即整条：
            # add_frame 不自动填此列，save_episode 会 np.stack 它，不填会报"空"
            frame["sub_task_index"] = np.array([t, T - 1 - t], dtype=np.int64)
            # pi05 内置 lerobot 签名：add_frame(frame, tasks:list[str], coarse_task:str)
            dataset.add_frame(frame, tasks=[prompt], coarse_task=prompt)
        dataset.save_episode(task=prompt)
        n_frames += T
        print(f"[{i+1}/{len(files)}] {os.path.basename(path)}  帧数 {T}")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        # pi05 用自己的 compute_norm_stats 生成归一化，不需要 lerobot 内置 stats；
        # 且 create() 构建的 dataset 缺 load_images 属性，compute_stats 会崩。
        # run_compute_stats=False：完成必要整理(episode_data_index/视频/校验)，跳过统计。
        try:
            dataset.consolidate(run_compute_stats=False)
        except TypeError:
            dataset.consolidate()
    print(f"[done] {len(files)} 条 / {n_frames} 帧 -> {args.root}")
    print("下一步: 计算归一化统计 (compute_norm_stats)，再训练。见 pi05_README.md")


if __name__ == "__main__":
    main()
