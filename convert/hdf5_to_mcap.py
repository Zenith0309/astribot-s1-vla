#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_to_mcap.py — 把采集的 HDF5 episode 转成 MCAP，供 Foxglove 直接回放

用法（任何装了 python 的机器，无需 ROS）：
  pip install mcap h5py numpy
  python convert/hdf5_to_mcap.py --src /data0/tzs/astribot/demos --out ./mcap
  python convert/hdf5_to_mcap.py --src demos --episode episode_0003_green_ok.h5

然后 Foxglove -> Open local file -> 选 .mcap：
  - 图像面板: /camera/<相机名>  （jpeg 原样封装，无转码）
  - Plot 面板: /state.data[0..11] 或 /action.data[0..11] 画关节/夹爪曲线
  - Raw Messages: /meta 查看 prompt / target / success
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import time

import h5py
import numpy as np

try:
    from mcap.writer import Writer
except ImportError:
    sys.exit("请先: pip install mcap")

CAM_RE = re.compile(r"camera_(.+?)_image")

IMG_SCHEMA = {
    "title": "foxglove.CompressedImage",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object",
                      "properties": {"sec": {"type": "integer"},
                                     "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}
VEC_SCHEMA = {
    "title": "JointVector", "type": "object",
    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
}
META_SCHEMA = {"title": "EpisodeMeta", "type": "object"}

STATE_NAMES = ([f"torso_{i}" for i in range(4)]
               + [f"arm_left_{i}" for i in range(7)] + ["gripper_left"])


def convert(h5_path, out_path):
    with h5py.File(h5_path, "r") as f, open(out_path, "wb") as fo:
        w = Writer(fo)
        w.start()

        def add_channel(topic, schema, schema_name):
            sid = w.register_schema(name=schema_name, encoding="jsonschema",
                                    data=json.dumps(schema).encode())
            return w.register_channel(topic=topic, message_encoding="json",
                                      schema_id=sid)

        ts = np.asarray(f["timestamp"], dtype=np.float64)
        state = np.asarray(f["state"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)

        ch_state = add_channel("/state", VEC_SCHEMA, "JointVector")
        ch_action = add_channel("/action", VEC_SCHEMA, "JointVector")
        ch_meta = add_channel("/meta", META_SCHEMA, "EpisodeMeta")
        cams = {}
        for key in f["images"]:
            m = CAM_RE.search(key)
            short = m.group(1) if m else key
            cams[key] = add_channel(f"/camera/{short}", IMG_SCHEMA,
                                    "foxglove.CompressedImage")

        # 元信息发一条
        meta = {k: (v.item() if hasattr(v, "item") else
                    v.decode() if isinstance(v, bytes) else v)
                for k, v in f.attrs.items()}
        meta["state_names"] = STATE_NAMES
        t0 = int(ts[0] * 1e9)
        w.add_message(ch_meta, log_time=t0, publish_time=t0,
                      data=json.dumps(meta, default=str).encode())

        T = len(ts)
        for t in range(T):
            ns = int(ts[t] * 1e9)
            stamp = {"sec": ns // 10**9, "nsec": ns % 10**9}
            w.add_message(ch_state, log_time=ns, publish_time=ns,
                          data=json.dumps({"data": state[t].tolist()}).encode())
            w.add_message(ch_action, log_time=ns, publish_time=ns,
                          data=json.dumps({"data": action[t].tolist()}).encode())
            for key, ch in cams.items():
                raw = np.asarray(f["images"][key][t], dtype=np.uint8)
                if raw.size == 0:
                    continue
                msg = {"timestamp": stamp, "frame_id": "camera",
                       "data": base64.b64encode(raw.tobytes()).decode(),
                       "format": "jpeg"}
                w.add_message(ch, log_time=ns, publish_time=ns,
                              data=json.dumps(msg).encode())
        w.finish()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HDF5 目录或单个文件")
    ap.add_argument("--out", default="./mcap", help="输出目录")
    ap.add_argument("--episode", default=None, help="只转这一个文件名")
    args = ap.parse_args()

    if os.path.isfile(args.src):
        files = [args.src]
    elif args.episode:
        files = [os.path.join(args.src, args.episode)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "*.h5")))
    if not files:
        sys.exit("[err] 没找到 .h5 文件")
    os.makedirs(args.out, exist_ok=True)

    for p in files:
        name = os.path.splitext(os.path.basename(p))[0] + ".mcap"
        dst = os.path.join(args.out, name)
        t0 = time.time()
        T = convert(p, dst)
        print(f"[ok] {name}  {T}帧  {os.path.getsize(dst)/1e6:.1f}MB  "
              f"{time.time()-t0:.1f}s")
    print(f"[done] {len(files)} 个文件 -> {args.out}，Foxglove 打开即可回放")


if __name__ == "__main__":
    main()
