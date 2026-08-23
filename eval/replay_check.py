#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_check.py — 用训练数据本身回放，判断"模型没学会"还是"评测管线接错了"。

闭环评测成功率为 0 时，有两种完全不同的原因，靠成功率区分不了：
  A. 模型没学会      -> 它连训练集里见过的帧都预测不对
  B. 评测管线接错了  -> 模型预测得很准，但评测侧的观测组装/动作解释与训练不一致

本脚本从训练用的 LeRobot 数据集里取若干帧，按【评测客户端的格式】发给推理服务，
再和 parquet 里记录的 ground-truth 动作比对：
  - 误差很小  -> 模型是好的，去查评测客户端（观测组装 / 动作解码 / IK）
  - 误差很大  -> 模型确实没学会，回到数据与训练配方

用法（训练环境，推理服务已起）：
  python eval/replay_check.py --server 127.0.0.1:8000 \
      --root /data0/tzs/astribot/datasets/pi05_pick_ball_v2 --episodes 3 --frames 5
"""

import argparse
import functools
import glob
import json
import os

import cv2
import msgpack
import numpy as np
import pyarrow.parquet as pq
from websockets.sync.client import connect as ws_connect

STATE_KEY = "cartesian_so3_dict.cartesian_pose_state"
ACTION_KEY = "cartesian_so3_dict.cartesian_pose_command"
CAM_COLS = {
    "images_dict.head.rgb": "head",
    "images_dict.left_wrist.rgb": "left_wrist",
    "images_dict.right_wrist.rgb": "right_wrist",
}


# msgpack-numpy 线兼容（与 pi05_eval_client.py 完全相同，必须一致否则服务端解不开）
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


class Link:
    """与 pi05_eval_client 相同的 msgpack/WebSocket 协议。"""

    def __init__(self, server):
        self.ws = ws_connect(f"ws://{server}", compression=None, max_size=None)
        self.metadata = _unpackb(self.ws.recv())    # 首帧是服务端元信息

    def infer(self, obs):
        self.ws.send(_packb(obs))
        rsp = self.ws.recv()
        if isinstance(rsp, str):
            raise RuntimeError(f"服务端推理错误:\n{rsp}")
        return _unpackb(rsp)


def load_video_frames(video_path, indices):
    """按帧号取出若干帧，返回 {idx: RGB uint8 (224,224,3)}。"""
    cap = cv2.VideoCapture(video_path)
    want, out, i = set(indices), {}, 0
    while cap.isOpened() and len(out) < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--root", required=True, help="LeRobot 数据集根（含 data/ videos/）")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=5, help="每条 episode 抽几帧")
    ap.add_argument("--prompt", default=None,
                    help="不给则从 meta/tasks.jsonl 按 task_index 取")
    ap.add_argument("--fresh-conn", action="store_true",
                    help="每帧新建连接。服务端可能跨请求保留 RTC 动作前缀等隐藏状态，"
                         "开这个可判断预测偏差是模型问题还是历史状态污染")
    args = ap.parse_args()

    pq_files = sorted(glob.glob(os.path.join(args.root, "data", "chunk-*", "*.parquet")))
    if not pq_files:
        raise SystemExit(f"[err] {args.root} 下没有 parquet")
    print(f"[info] 数据集 {len(pq_files)} 条 episode，取前 {args.episodes} 条")

    tasks = {}
    tj = os.path.join(args.root, "meta", "tasks.jsonl")
    if os.path.exists(tj):
        for line in open(tj, encoding="utf-8"):
            d = json.loads(line)
            tasks[d.get("task_index")] = d.get("task")
    print(f"[info] tasks: {tasks}")

    link = None if args.fresh_conn else Link(args.server)
    rows = []

    for ep_path in pq_files[:args.episodes]:
        t = pq.read_table(ep_path)
        n = t.num_rows
        ep_idx = int(t.column("episode_index")[0].as_py())
        idxs = np.linspace(0, n - 1, args.frames, dtype=int).tolist()

        # 三路相机视频
        vids = {}
        for col, short in CAM_COLS.items():
            vp = os.path.join(args.root, "videos", "chunk-000", col,
                              f"episode_{ep_idx:06d}.mp4")
            if not os.path.exists(vp):
                raise SystemExit(f"[err] 缺视频 {vp}")
            vids[col] = load_video_frames(vp, idxs)

        states = np.array(t.column(STATE_KEY).to_pylist(), dtype=np.float32)
        acts = np.array(t.column(ACTION_KEY).to_pylist(), dtype=np.float32)

        ti = t.column("task_index")[0].as_py()
        ti = ti[0] if isinstance(ti, list) else ti
        prompt = args.prompt or tasks.get(ti) or "pick the ball and place it into the box"

        for fi in idxs:
            obs = {"prompt": prompt}
            # state: 训练集是 31 维，评测协议要 34 维（末尾补 chassis）
            s = states[fi]
            obs[STATE_KEY] = np.concatenate(
                [s, np.zeros(34 - len(s), dtype=np.float32)]).astype(np.float32) \
                if len(s) < 34 else s.astype(np.float32)
            for col in CAM_COLS:
                img = vids[col].get(fi)
                if img is None:
                    break
                obs[col] = np.ascontiguousarray(
                    img.transpose(2, 0, 1)).astype(np.float32) / 255.0
            if len(obs) < 5:
                continue

            conn = Link(args.server) if args.fresh_conn else link
            pred = np.asarray(conn.infer(obs)["actions"], dtype=np.float32)
            if args.fresh_conn:
                try:
                    conn.ws.close()
                except Exception:
                    pass
            gt = acts[fi]

            # 左臂目标位置（第 9:12 维）是最能说明问题的量
            p_xyz, g_xyz = pred[0, 9:12], gt[9:12]
            err_cm = float(np.linalg.norm(p_xyz - g_xyz) * 100)
            # 与"当前状态"的距离作参照：若预测≈当前状态，说明模型只是原地不动
            s_xyz = s[9:12]
            base_cm = float(np.linalg.norm(g_xyz - s_xyz) * 100)
            rows.append(dict(ep=ep_idx, frame=int(fi), err_cm=err_cm,
                             gt_step_cm=base_cm,
                             pred=[round(float(v), 4) for v in p_xyz],
                             gt=[round(float(v), 4) for v in g_xyz],
                             state=[round(float(v), 4) for v in s_xyz]))
            print(f"  ep{ep_idx} f{fi:4d}  预测={np.round(p_xyz,4).tolist()}  "
                  f"真值={np.round(g_xyz,4).tolist()}  误差={err_cm:6.2f}cm  "
                  f"(真值相对当前状态位移 {base_cm:.2f}cm)")

    if not rows:
        raise SystemExit("[err] 没有可比对的帧")

    e = np.array([r["err_cm"] for r in rows])
    b = np.array([r["gt_step_cm"] for r in rows])
    print(f"\n===== 回放比对（n={len(rows)}）=====")
    print(f"  预测 vs 真值 误差: 中位数 {np.median(e):.2f}cm  平均 {e.mean():.2f}cm  最大 {e.max():.2f}cm")
    print(f"  参照：真值相对当前状态的位移 中位数 {np.median(b):.2f}cm")
    if np.median(e) < 2.0:
        print("  ✅ 模型能复现训练数据 -> 模型没问题，去查【评测客户端】"
              "（观测组装 / 动作解码 / IK 下发）")
    elif np.median(e) < np.median(b):
        print("  ⚠️ 预测有方向但不准 -> 模型学得不充分，可加训练步数/数据")
    else:
        print("  ❌ 误差不小于「原地不动」的基线 -> 模型没学会这个映射，"
              "回到数据与训练配方")

    with open("replay_check_results.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "median_err_cm": float(np.median(e)),
                   "median_gt_step_cm": float(np.median(b))},
                  f, ensure_ascii=False, indent=2)
    print("  结果已写入 replay_check_results.json")


if __name__ == "__main__":
    main()
