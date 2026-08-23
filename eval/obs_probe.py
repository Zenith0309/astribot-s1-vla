#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""obs_probe.py — 对比【评测时实时观测】与【训练数据观测】，定位二者的分布差异。

背景：同一个 checkpoint，用训练集的帧回放预测误差 0.3cm（完美），
但闭环评测时对任何球位都输出同一个动作。差别只可能在送进去的观测上。
本脚本在仿真容器内抓取评测客户端实际会发送的图像，检查：

  1. 图像是否真的在变（不同球位应给出不同画面；若几乎不变说明相机数据是死的）
  2. 与训练集图像的像素统计是否对得上（均值/标准差差太多说明域不匹配）
  3. 顺带把两边的图存成 png，可直接看

用法（容器内，仿真运行中）：
  python3 eval/obs_probe.py --shots 4 --train-root /workspace/datasets/pi05_pick_ball_v2
"""

import argparse
import os
import sys
import threading

import cv2
import numpy as np
import rclpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collect"))
sys.path.insert(0, os.environ.get("ASTRIBOT_SIMU_ROOT", "."))

from collect_demos import CollectorNode, random_scene, BALLS  # noqa: E402

CAM_KEYS = {"head_rgbd": "head", "left_wrist_rgbd": "left_wrist",
            "right_wrist_rgbd": "right_wrist"}


def grab(node):
    """完全照搬 pi05_eval_client.build_obs 的图像处理。"""
    out = {}
    for topic in node.args.cameras:
        img = node.images.get(topic)
        if img is None:
            return None
        short = [k for k in CAM_KEYS if k in topic]
        if not short:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        out[CAM_KEYS[short[0]]] = rgb
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4, help="换几次球位取样")
    ap.add_argument("--head-track", action="store_true",
                    help="启用头部视线跟踪（采集与评测客户端都会开；不开则头朝默认方向，"
                         "很可能拍不到桌面）")
    ap.add_argument("--train-root", default=None, help="训练集根，用于像素统计对比")
    ap.add_argument("--outdir", default="/workspace/obs_probe")
    # CollectorNode 占位参数
    ap.add_argument("--site", default="arm_left_tool")
    ap.add_argument("--open", type=float, default=0.0, dest="grip_open")
    ap.add_argument("--preclose", type=float, default=35.0)
    ap.add_argument("--close", type=float, default=70.0, dest="grip_close")
    ap.add_argument("--seg-time", type=float, default=2.5)
    ap.add_argument("--hold-min", type=float, default=62.0)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    args.cameras = [f"/astribot_whole_body/camera/{c}/image_raw" for c in CAM_KEYS]
    os.makedirs(args.outdir, exist_ok=True)

    rclpy.init()
    node = CollectorNode(args)
    if args.head_track:                       # 与 pi05_eval_client 的做法一致
        from scripted_grasp import Kinematics, MJCF_PATH
        node.start_head_track(Kinematics(MJCF_PATH, args.site))
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.external_spin = True
    node.wait_ready()

    shots = []
    try:
        for i in range(args.shots):
            node.move_to_start(2.0)
            random_scene(node)
            node.sim_sleep(1.5)
            ball = BALLS[list(BALLS)[i % len(BALLS)]]
            if args.head_track:               # 与评测客户端一致：先让头转向目标球再取图
                node.track_object = ball
                node.sim_sleep(1.5)
            try:
                bp = np.round(node.object_pos(ball, timeout=2.0), 3).tolist()
            except RuntimeError:
                bp = None
            imgs = grab(node)
            if imgs is None:
                print(f"  [{i}] 没拿到图像"); continue
            shots.append((bp, imgs))
            for k, v in imgs.items():
                cv2.imwrite(os.path.join(args.outdir, f"eval_{i}_{k}.png"),
                            cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            print(f"  [{i}] 球@{bp}  head 像素均值={imgs['head'].mean():.1f}")
    finally:
        try:
            ex.shutdown(timeout_sec=1.0); node.destroy_node(); rclpy.shutdown()
        except Exception:
            pass

    if len(shots) < 2:
        raise SystemExit("[err] 取样不足")

    print(f"\n===== 1. 评测图像是否随球位变化（n={len(shots)}）=====")
    for cam in ["head", "left_wrist", "right_wrist"]:
        a = np.stack([s[1][cam].astype(np.float32) for s in shots])
        # 两两之间的平均绝对像素差
        diffs = [np.abs(a[i] - a[j]).mean()
                 for i in range(len(a)) for j in range(i + 1, len(a))]
        print(f"  {cam:12s} 两两平均像素差 = {np.mean(diffs):6.2f} "
              f"(0 表示画面完全一样 -> 相机数据是死的)")

    if args.train_root:
        import glob
        print(f"\n===== 2. 与训练集图像的像素统计对比 =====")
        for cam, col in [("head", "images_dict.head.rgb"),
                         ("left_wrist", "images_dict.left_wrist.rgb"),
                         ("right_wrist", "images_dict.right_wrist.rgb")]:
            vids = sorted(glob.glob(os.path.join(
                args.train_root, "videos", "chunk-000", col, "*.mp4")))
            if not vids:
                print(f"  {cam}: 找不到训练视频"); continue
            cap = cv2.VideoCapture(vids[0])
            frames = []
            while len(frames) < 20:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                print(f"  {cam}: 训练视频解不出帧"); continue
            tr = np.stack(frames).astype(np.float32)
            ev = np.stack([s[1][cam].astype(np.float32) for s in shots])
            cv2.imwrite(os.path.join(args.outdir, f"train_{cam}.png"),
                        cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            print(f"  {cam:12s} 训练: 均值{tr.mean():6.1f} 标准差{tr.std():6.1f} | "
                  f"评测: 均值{ev.mean():6.1f} 标准差{ev.std():6.1f} | "
                  f"均值差 {abs(tr.mean()-ev.mean()):5.1f}")
            for c, name in enumerate("RGB"):
                print(f"      {name}: 训练{tr[...,c].mean():6.1f}  评测{ev[...,c].mean():6.1f}")

    print(f"\n图像已存到 {args.outdir}（eval_*.png 与 train_*.png 可直接比对）")


if __name__ == "__main__":
    main()
