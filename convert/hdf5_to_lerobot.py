#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_to_lerobot.py — 把 collect_demos.py 产出的 HDF5 episode 转成 LeRobot 数据集

在【训练环境】（非 ROS 容器）运行：
  python convert/hdf5_to_lerobot.py \
      --src /data0/tzs/astribot/demos \
      --root ./datasets/astribot_pick_ball

  # 默认转换 HDF5 内的全部相机；--resize 224 224 可统一缩放
  # 默认只转成功样本（*_ok.h5）；--include-fail 带上失败样本

输入 HDF5 结构（collect_demos.py 约定）：
  /state    (T,12) float32   torso4 + arm_left7 + gripper_left1（实际值）
  /action   (T,12) float32   同布局（指令值）
  /images/<key>  (T,) vlen uint8 JPEG      key 形如 astribot_whole_body_camera_head_rgbd_image_raw
  attrs: prompt / target / success / fps

输出：LeRobot v3 数据集。相机特征命名 observation.images.<相机名>
（如 head_rgbd / left_wrist_rgbd / right_wrist_rgbd），与真机及官方约定对齐。
"""

import argparse
import glob
import os
import re
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


def cam_short_name(h5_key):
    """astribot_whole_body_camera_head_rgbd_image_raw -> head_rgbd"""
    m = re.search(r"camera_(.+?)_image", h5_key)
    return m.group(1) if m else h5_key


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="HDF5 episode 目录")
    p.add_argument("--repo-id", default="local/astribot_pick_ball")
    p.add_argument("--root", default="./datasets/astribot_pick_ball")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--include-fail", action="store_true")
    p.add_argument("--resize", type=int, nargs=2, default=None,
                   metavar=("H", "W"), help="可选：统一缩放图像，如 224 224")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="只转这些相机（短名，如 head_rgbd）；默认全部")
    return p.parse_args()


def decode_frames(dset, resize, prev_ok=None):
    frames = []
    for raw in dset:
        buf = np.asarray(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None   # BGR
        if img is None:
            if frames:
                img = frames[-1].copy()
            elif prev_ok is not None:
                img = prev_ok.copy()
            else:
                raise RuntimeError("首帧解码失败且无可回退帧")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if resize is not None:
                img = cv2.resize(img, (resize[1], resize[0]),
                                 interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


def load_episode(path, cameras, resize):
    with h5py.File(path, "r") as f:
        state = np.asarray(f["state"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)
        images = {}
        for key in f["images"]:
            short = cam_short_name(key)
            if cameras is not None and short not in cameras:
                continue
            images[short] = decode_frames(f["images"][key], resize)
        if not images:
            raise RuntimeError(f"{path} 没有匹配的相机数据")
        prompt = f.attrs.get("prompt", "pick the ball and place it into the box")
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
    T = min([len(state), len(action)] + [len(v) for v in images.values()])
    return state[:T], action[:T], {k: v[:T] for k, v in images.items()}, str(prompt)


def main():
    args = parse_args()
    pattern = "*.h5" if args.include_fail else "*_ok.h5"
    files = sorted(glob.glob(os.path.join(args.src, pattern)))
    if not files:
        sys.exit(f"[err] {args.src} 下没有匹配 {pattern} 的文件")
    print(f"[info] 待转换 {len(files)} 条 episode")

    # 用第一条确定相机清单与图像尺寸
    _, _, images0, _ = load_episode(files[0], args.cameras, args.resize)
    cam_shapes = {k: v[0].shape for k, v in images0.items()}
    for k, shp in cam_shapes.items():
        print(f"[info] 相机 {k}: {shp[0]}x{shp[1]}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (STATE_DIM,),
                   "names": STATE_NAMES},
    }
    for k, (H, W, _) in cam_shapes.items():
        features[f"observation.images.{k}"] = {
            "dtype": "video", "shape": (H, W, 3),
            "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, root=args.root,
        features=features, use_videos=True)

    n_frames = 0
    for i, path in enumerate(files):
        state, action, images, prompt = load_episode(path, args.cameras, args.resize)
        if set(images) != set(cam_shapes):
            print(f"[warn] {os.path.basename(path)} 相机不齐 {set(images)}，跳过")
            continue
        for t in range(len(state)):
            frame = {"observation.state": state[t], "action": action[t]}
            for k in images:
                frame[f"observation.images.{k}"] = images[k][t]
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

    if hasattr(dataset, "finalize"):               # 数据集 v3 收尾
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):          # 数据集 v2 等价操作
        dataset.consolidate()
    print(f"[done] 共 {len(files)} 条 / {n_frames} 帧 -> {args.root}")
    print("下一步: bash train/train_act.sh")


if __name__ == "__main__":
    main()
