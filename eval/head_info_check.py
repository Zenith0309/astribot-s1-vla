#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""head_info_check.py — 验证「加入头部角度后，观测里是否真的能定出物体世界位置」。

这是 B 方案（把 head 关节角写进 state[29:31]）成立与否的关键判据。直接读原始 h5，
不需要等转换，所以采到几十条就能提前验证，避免白跑 2 小时转换 + 17 小时训练。

原理：头部相机跟踪目标球时
        球的像素位置 = f(球的世界位置, 头部朝向)
    只给像素位置 -> 欠定，解不出世界位置（v2 实测相关性 |r|<0.17）
    像素位置 + 头部朝向 -> 应该可解

做法：对每条 episode 取同一相位的一帧，提取
        输入 X = [球像素x, 球像素y, head0, head1]（可选加二次项）
        标签 y = [球世界x, 球世界y]
    做最小二乘并留一法交叉验证，看能否预测出世界位置。

判读：
    留一法 RMSE < 3cm  -> 信息足够，B 方案成立，可以投入训练
    仅像素 vs 像素+头部 的对比也会打印，用来证明「是头部角度补上了这块信息」

用法（训练环境，纯离线，读 h5）：
  python eval/head_info_check.py --demos <demos_v3 目录> --episodes 60 --frame 124
"""

import argparse
import glob
import os
import re

import cv2
import h5py
import numpy as np

COLOR_HSV = {
    "red":   [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (180, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue":  [((100, 120, 50), (130, 255, 255))],
}
HEAD_TOPIC_HINT = "head_rgbd"


def ball_centroid(rgb, color):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None
    for lo, hi in COLOR_HSV[color]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 12:
        return None
    return float(cent[i][0]), float(cent[i][1])


def loo_rmse(X, y):
    """留一法交叉验证 RMSE（cm）。样本少时比训练集残差可信得多。"""
    n = len(X)
    Xa = np.hstack([X, np.ones((n, 1))])
    errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        try:
            w, *_ = np.linalg.lstsq(Xa[m], y[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        errs.append(np.linalg.norm(Xa[i] @ w - y[i]))
    return float(np.mean(errs) * 100) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--frame", type=int, default=124)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.demos, "*_ok.h5")))[:args.episodes]
    if not files:
        raise SystemExit(f"[err] {args.demos} 下没有 *_ok.h5")
    print(f"[info] {len(files)} 条, 取第 {args.frame} 帧")

    rows, nohead = [], 0
    for p in files:
        m = re.search(r"_(red|green|blue)_", os.path.basename(p))
        if not m:
            continue
        color = m.group(1)
        with h5py.File(p, "r") as f:
            st = np.asarray(f["state"], dtype=np.float32)
            if st.shape[1] < 14:
                nohead += 1
                continue
            tcp = np.asarray(f["tcp_pose"], dtype=np.float32)
            fi = min(args.frame, len(st) - 1)
            head = st[fi, 12:14].astype(float)
            gi = int(np.argmin(tcp[:, 2]))
            wxy = tcp[gi, :2].astype(float)      # 球世界位置（近似）
            key = next((k for k in f["images"] if HEAD_TOPIC_HINT in k), None)
            if key is None:
                continue
            raw = np.asarray(f["images"][key][fi], dtype=np.uint8)
            if raw.size == 0:
                continue
            im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        c = ball_centroid(rgb, color)
        if c is None:
            continue
        rows.append(dict(px=c[0], py=c[1], h0=head[0], h1=head[1],
                         wx=wxy[0], wy=wxy[1]))

    if nohead:
        print(f"[warn] {nohead} 条是旧格式(无 head)，已跳过")
    if len(rows) < 10:
        raise SystemExit(f"[err] 有效样本 {len(rows)} 条，太少")

    P = np.array([[r["px"], r["py"]] for r in rows])
    H = np.array([[r["h0"], r["h1"]] for r in rows])
    W = np.array([[r["wx"], r["wy"]] for r in rows])

    print(f"\n===== 样本 n={len(rows)} =====")
    print(f"  球世界位置 x 跨度 {W[:,0].ptp()*100:.1f}cm  y 跨度 {W[:,1].ptp()*100:.1f}cm")
    print(f"  球像素位置 标准差 {P[:,0].std():.1f} / {P[:,1].std():.1f} px")
    print(f"  头部角度   第一维 [{H[:,0].min():.3f},{H[:,0].max():.3f}]  "
          f"第二维 [{H[:,1].min():.3f},{H[:,1].max():.3f}]")

    # 基线：预测成常数（世界位置的均值）——任何有用的模型必须显著优于它
    base = float(np.mean(np.linalg.norm(W - W.mean(0), axis=1)) * 100)

    print(f"\n  === 能否从观测反解世界位置（留一法 RMSE，越小越好）===")
    print(f"    常数基线（永远猜均值）      : {base:6.2f} cm")
    r_px = loo_rmse(P, W)
    print(f"    只用像素位置                : {r_px:6.2f} cm   <- v2 的情形")
    r_h = loo_rmse(H, W)
    print(f"    只用头部角度                : {r_h:6.2f} cm")
    r_both = loo_rmse(np.hstack([P, H]), W)
    print(f"    像素 + 头部角度             : {r_both:6.2f} cm   <- v3 的情形")
    # 加交叉项/二次项：像素与头部朝向的关系本质是非线性的
    feat = np.hstack([P, H, P * H, P**2, H**2,
                      (P[:, [0]] * H[:, [1]]), (P[:, [1]] * H[:, [0]])])
    r_poly = loo_rmse(feat, W)
    print(f"    像素 + 头部 + 二次/交叉项   : {r_poly:6.2f} cm")

    best = np.nanmin([r_both, r_poly])
    print(f"\n  === 判读 ===")
    if best < 3.0:
        print(f"  ✅ 加入头部角度后可反解到 {best:.2f}cm -> 观测里已有足够的位置信息，")
        print(f"     B 方案成立，可以投入转换与训练。")
    elif best < base * 0.5:
        print(f"  ⚠️ 有改善（{base:.1f} -> {best:.1f}cm）但精度仍不足抓取所需（约 1-2cm）。")
        print(f"     可考虑同时关掉头部跟踪（--no-head-track），让映射更直接。")
    else:
        print(f"  ❌ 仍无法反解（{best:.1f}cm vs 基线 {base:.1f}cm）-> B 方案不够，")
        print(f"     改用 A 方案（采集时 --no-head-track，头部固定）。")
    if not np.isnan(r_px) and not np.isnan(r_both):
        print(f"  头部角度的贡献：{r_px:.1f}cm -> {r_both:.1f}cm"
              f"（{'有效' if r_both < r_px * 0.8 else '不明显'}）")


if __name__ == "__main__":
    main()
