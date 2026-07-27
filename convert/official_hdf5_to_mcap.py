#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
official_hdf5_to_mcap.py — 把官方 Astribot HDF5（command_poses_dict/poses_dict/
images_dict/joints_dict 结构）转成 MCAP，供 Foxglove 回放。

与自采格式(hdf5_to_mcap.py)不同：官方格式图像是拼接 JPEG 字节流 + rgb_size 数组，
关节用 joints_dict/joints_position_state|command (25维)，时间戳用顶层 time。

用法（任何机器，无需 ROS）：
  pip install mcap h5py numpy opencv-python
  python official_hdf5_to_mcap.py --src <dir> --episode 111_episode_0.hdf5 --out ./mcap
  python official_hdf5_to_mcap.py --src <dir> --out ./mcap        # 转目录下全部 .hdf5

Foxglove 打开 .mcap：
  /camera/head|left|right  三路图像
  /joints/state /joints/command  25 维关节曲线（Plot 面板）
  /meta  帧数等
"""

import argparse
import base64
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np

try:
    from mcap.writer import Writer
except ImportError:
    sys.exit("请先: pip install mcap")

CAMS = ["head", "left", "right"]
IMG_SCHEMA = {"title": "foxglove.CompressedImage", "type": "object",
              "properties": {"timestamp": {"type": "object"}, "frame_id": {"type": "string"},
                             "data": {"type": "string", "contentEncoding": "base64"},
                             "format": {"type": "string"}}}
VEC_SCHEMA = {"title": "JointVector", "type": "object",
              "properties": {"data": {"type": "array", "items": {"type": "number"}}}}
META_SCHEMA = {"title": "EpisodeMeta", "type": "object"}


def decode_frame(rgb_flat, sizes, i):
    """从拼接 JPEG 字节流里取第 i 帧并解码为 BGR。"""
    off = int(np.sum(sizes[:i]))
    n = int(sizes[i])
    buf = np.asarray(rgb_flat[off:off + n], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), buf


def convert(h5_path, out_path):
    with h5py.File(h5_path, "r") as f, open(out_path, "wb") as fo:
        w = Writer(fo)
        w.start()

        def ch(topic, schema, name):
            sid = w.register_schema(name=name, encoding="jsonschema",
                                    data=json.dumps(schema).encode())
            return w.register_channel(topic=topic, message_encoding="json", schema_id=sid)

        ts = np.asarray(f["time"], dtype=np.float64)
        T = len(ts)
        jstate = np.asarray(f["joints_dict/joints_position_state"], dtype=np.float64)
        jcmd = np.asarray(f["joints_dict/joints_position_command"], dtype=np.float64)

        ch_state = ch("/joints/state", VEC_SCHEMA, "JointVector")
        ch_cmd = ch("/joints/command", VEC_SCHEMA, "JointVector")
        ch_meta = ch("/meta", META_SCHEMA, "EpisodeMeta")
        cam_ch, cam_data = {}, {}
        for c in CAMS:
            key = f"images_dict/{c}/rgb"
            if key in f:
                cam_ch[c] = ch(f"/camera/{c}", IMG_SCHEMA, "foxglove.CompressedImage")
                cam_data[c] = (np.asarray(f[key]),
                               np.asarray(f[f"images_dict/{c}/rgb_size"], dtype=np.int64))

        t0 = int(ts[0] * 1e9)
        w.add_message(ch_meta, log_time=t0, publish_time=t0,
                      data=json.dumps({"frames": T, "cameras": list(cam_ch),
                                       "joint_dim": jstate.shape[1]}).encode())

        for i in range(T):
            ns = int(ts[i] * 1e9)
            stamp = {"sec": ns // 10**9, "nsec": ns % 10**9}
            w.add_message(ch_state, log_time=ns, publish_time=ns,
                          data=json.dumps({"data": jstate[i].tolist()}).encode())
            w.add_message(ch_cmd, log_time=ns, publish_time=ns,
                          data=json.dumps({"data": jcmd[i].tolist()}).encode())
            for c, (flat, sizes) in cam_data.items():
                _, raw = decode_frame(flat, sizes, i)
                if raw.size == 0:
                    continue
                msg = {"timestamp": stamp, "frame_id": c,
                       "data": base64.b64encode(raw.tobytes()).decode(), "format": "jpeg"}
                w.add_message(cam_ch[c], log_time=ns, publish_time=ns,
                              data=json.dumps(msg).encode())
        w.finish()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HDF5 目录或单文件")
    ap.add_argument("--out", default="./mcap")
    ap.add_argument("--episode", default=None, help="只转这一个文件名")
    args = ap.parse_args()

    if os.path.isfile(args.src):
        files = [args.src]
    elif args.episode:
        files = [os.path.join(args.src, args.episode)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "*.hdf5")))
    if not files:
        sys.exit("[err] 没找到 .hdf5")
    os.makedirs(args.out, exist_ok=True)

    for p in files:
        dst = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".mcap")
        T = convert(p, dst)
        print(f"[ok] {os.path.basename(dst)}  {T}帧  {os.path.getsize(dst)/1e6:.1f}MB")
    print(f"[done] {len(files)} 个 -> {args.out}，Foxglove 打开即可回放")


if __name__ == "__main__":
    main()
