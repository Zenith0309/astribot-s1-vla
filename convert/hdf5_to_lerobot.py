#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_to_lerobot.py — 把 collect_demos.py 产出的 HDF5 episode 转成 LeRobot 数据集

在【训练环境】（非 ROS 容器）运行：
  python convert/hdf5_to_lerobot.py \
      --src /data0/tzs/astribot/demos \
      --repo-id local/astribot_pick_ball \
      --root ./datasets/astribot_pick_ball

  # 只转成功样本（文件名 *_ok.h5），是默认行为；--include-fail 可带上失败样本

输入 HDF5 结构（collect_demos.py 约定）：
  /state    (T,12) float32   torso4 + arm_left7 + gripper_left1（实际值）
  /action   (T,12) float32   同布局（指令值）
  /images/<key>  (T,) vlen uint8 JPEG
  attrs: prompt / target / success / fps

输出：LeRobot v3 数据集（features 命名与 eval/serve_policy.py 保持一致）
"""

import argparse
import glob
import os
import sys

import cv2
import h5py
import numpy as np

try:                      # lerobot >= 0.2 扁平化了包路径
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

STATE_DIM = 12
STATE_NAMES = ([f"torso_{i}" for i in range(4)]
               + [f"arm_left_{i}" for i in range(7)] + ["gripper_left"])
CAM_FEATURE = "observation.images.third_person"     # 与评测端约定一致


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="HDF5 episode 目录")
    p.add_argument("--repo-id", default="local/astribot_pick_ball")
    p.add_argument("--root", default="./datasets/astribot_pick_ball",
                   help="数据集输出目录")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--include-fail", action="store_true")
    p.add_argument("--resize", type=int, nargs=2, default=None,
                   metavar=("H", "W"), help="可选：统一缩放图像，如 224 224")
    p.add_argument("--camera-key", default=None,
                   help="HDF5 images/ 组下的键名；默认取第一个")
    return p.parse_args()


def load_episode(path, cam_key, resize):
    with h5py.File(path, "r") as f:
        state = np.asarray(f["state"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)
        keys = list(f["images"].keys())
        if not keys:
            raise RuntimeError(f"{path} 没有图像数据")
        key = cam_key or keys[0]
        frames = []
        for raw in f["images"][key]:
            buf = np.asarray(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)          # BGR
            if img is None:
                img = frames[-1].copy() if frames else None
                if img is None:
                    raise RuntimeError(f"{path} 首帧解码失败")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if resize is not None:
                img = cv2.resize(img, (resize[1], resize[0]),
                                 interpolation=cv2.INTER_AREA)
            frames.append(img)
        prompt = f.attrs.get("prompt", "pick the ball and place it into the box")
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
    T = min(len(state), len(action), len(frames))
    return state[:T], action[:T], frames[:T], str(prompt)


def main():
    args = parse_args()
    pattern = "*.h5" if args.include_fail else "*_ok.h5"
    files = sorted(glob.glob(os.path.join(args.src, pattern)))
    if not files:
        sys.exit(f"[err] {args.src} 下没有匹配 {pattern} 的文件")
    print(f"[info] 待转换 {len(files)} 条 episode")

    # 用第一条确定图像尺寸
    _, _, frames0, _ = load_episode(files[0], args.camera_key, args.resize)
    H, W, _ = frames0[0].shape
    print(f"[info] 图像尺寸 {H}x{W}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (STATE_DIM,),
                   "names": STATE_NAMES},
        CAM_FEATURE: {"dtype": "video", "shape": (H, W, 3),
                      "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, root=args.root,
        features=features, use_videos=True)

    n_frames = 0
    for i, path in enumerate(files):
        state, action, frames, prompt = load_episode(path, args.camera_key, args.resize)
        for t in range(len(state)):
            frame = {"observation.state": state[t], "action": action[t],
                     CAM_FEATURE: frames[t]}
            try:                                   # 新版：task 放在 frame 里
                dataset.add_frame({**frame, "task": prompt})
            except (KeyError, ValueError, TypeError):
                dataset.add_frame(frame)
        try:
            dataset.save_episode()
        except TypeError:                          # 旧版：task 作为参数传入
            dataset.save_episode(task=prompt)
        n_frames += len(state)
        print(f"[{i+1}/{len(files)}] {os.path.basename(path)}  帧数 {len(state)}")

    if hasattr(dataset, "finalize"):               # 数据集 v3 需要收尾
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):          # 数据集 v2 的等价操作
        dataset.consolidate()
    print(f"[done] 共 {len(files)} 条 / {n_frames} 帧 -> {args.root}")
    print("下一步: bash train/train_act.sh")


if __name__ == "__main__":
    main()
