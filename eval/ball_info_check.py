#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ball_info_check.py — 检验训练观测里到底有没有「球的绝对位置」信息。

动机：采集时头部相机会跟踪目标球（node.track_object = ball_name），球因此始终
落在画面中心附近；而 state 的 head 两维在转换时被硬编码为 [0,0]
（hdf5_to_pi05.py 的 --head 默认值）。若两者同时成立，则：

    球在世界系里的位置，既不在图像里（被跟踪抵消了），也不在 state 里（被清零了）

模型就只能靠桌沿透视、物体相对大小等二阶线索去猜位置——这正好解释闭环里
「x/z 大致准、y 系统性偏」的现象（y 是左右方向，恰是头部偏航补偿掉的那一维）。

做法：对若干训练 episode，在悬停阶段的 head 图像里用颜色阈值找目标球的像素质心，
与该 episode 的球世界坐标（由 tcp 最低点反推）做相关。

  - 像素质心几乎不随世界位置变化（标准差很小 / 相关系数接近 0）
        -> 位置信息确实被抹掉了，是数据设计问题，要改采集（关掉头部跟踪或把 head 角度写进 state）
  - 像素质心随世界位置明显变化
        -> 信息在图像里，问题在别处

用法（训练环境，纯离线）：
  python eval/ball_info_check.py --root <datasets>/pi05_pick_ball_v2 \
      --demos <demos_v2 目录> --episodes 30 --frame 124
"""

import argparse
import glob
import json
import os
import re

import cv2
import h5py
import numpy as np

# HSV 阈值：仿真里的三色球饱和度高，好分割
COLOR_HSV = {
    "red":   [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (180, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue":  [((100, 120, 50), (130, 255, 255))],
}


def ball_centroid(rgb, color):
    """返回目标色球的像素质心 (cx, cy) 与像素数；找不到返回 None。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None
    for lo, hi in COLOR_HSV[color]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    # 去掉零散噪点，只留最大连通域（球）
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 12:
        return None
    return float(cent[i][0]), float(cent[i][1]), int(stats[i, cv2.CC_STAT_AREA])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot 数据集根")
    ap.add_argument("--demos", required=True, help="原始 h5 目录（取球世界坐标）")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--frame", type=int, default=124,
                    help="取哪一帧（悬停阶段最能体现「球在画面何处」）")
    args = ap.parse_args()

    tasks = {}
    tj = os.path.join(args.root, "meta", "tasks.jsonl")
    if os.path.exists(tj):
        for line in open(tj, encoding="utf-8"):
            d = json.loads(line)
            tasks[d.get("task_index")] = d.get("task")

    h5s = sorted(glob.glob(os.path.join(args.demos, "*_ok.h5")))
    vids = sorted(glob.glob(os.path.join(
        args.root, "videos", "chunk-000", "images_dict.head.rgb", "*.mp4")))
    print(f"[info] h5={len(h5s)} 条, 视频={len(vids)} 条, 取前 {args.episodes} 条, 帧 {args.frame}")

    rows = []
    for k, (h5p, vp) in enumerate(zip(h5s, vids)):
        if k >= args.episodes:
            break
        m = re.search(r"_(red|green|blue)_", os.path.basename(h5p))
        if not m:
            continue
        color = m.group(1)
        with h5py.File(h5p, "r") as f:
            tcp = f["tcp_pose"][:]
        gi = int(np.argmin(tcp[:, 2]))
        ball_xy = tcp[gi, :2]              # 世界系球位（近似）

        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        c = ball_centroid(rgb, color)
        if c is None:
            continue
        rows.append(dict(ep=k, color=color, wx=float(ball_xy[0]), wy=float(ball_xy[1]),
                         px=c[0], py=c[1], area=c[2]))

    if len(rows) < 6:
        raise SystemExit(f"[err] 有效样本只有 {len(rows)} 条，无法判断")

    w = np.array([[r["wx"], r["wy"]] for r in rows])
    p = np.array([[r["px"], r["py"]] for r in rows])

    print(f"\n===== 样本 n={len(rows)} =====")
    print(f"  球世界位置  x 跨度 {w[:,0].ptp()*100:6.1f}cm  y 跨度 {w[:,1].ptp()*100:6.1f}cm")
    print(f"  球像素位置  x 跨度 {p[:,0].ptp():6.1f}px   y 跨度 {p[:,1].ptp():6.1f}px")
    print(f"              标准差 {p[:,0].std():.1f} / {p[:,1].std():.1f} px  (画面 224px)")

    print(f"\n  === 世界位置 vs 像素位置 的相关性 ===")
    names = [("世界x", 0), ("世界y", 1)]
    for wn, wi in names:
        for pn, pi in [("像素x", 0), ("像素y", 1)]:
            if p[:, pi].std() < 1e-6:
                print(f"    {wn} ~ {pn}: 像素无变化"); continue
            r = np.corrcoef(w[:, wi], p[:, pi])[0, 1]
            print(f"    {wn} ~ {pn}: r = {r:+.3f}")

    # 判读要看【相关性】而不是【是否变化】：像素位置变化大但与世界位置不相关，
    # 恰恰是信息被破坏的情形（比"球始终在画面中心"更隐蔽）。
    best_r = max(abs(np.corrcoef(w[:, wi], p[:, pi])[0, 1])
                 for wi in (0, 1) for pi in (0, 1) if p[:, pi].std() > 1e-6)
    print(f"\n  === 判读（最大 |r| = {best_r:.3f}）===")
    if best_r < 0.4:
        print("  ❌ 球的像素位置与世界位置【不相关】-> 图像里没有可用的位置信息。")
        print("     机理：头部相机跟踪球，像素位置 = f(世界位置, 头部朝向)；")
        print("           而头部朝向既没记进 hdf5（state_layout 只有 torso4+arm7+grip1），")
        print("           转换时 head 两维又被写死 [0,0]（--head 默认值）。")
        print("           两条路都断了，模型无法反解球在世界何处。")
        print("     注意：脚本专家不受影响——它直接读 TF 真值 object_pos()，从不看图像。")
        print("     结论：【数据设计问题】，调训练参数无法解决。两个修法：")
        print("       A. 采集时加 --no-head-track（头部固定 -> 像素位置唯一对应世界位置）")
        print("       B. 采集时记录 head 关节角，转换时写进 state[29:31]（官方真机数据就是这样）")
    elif p[:, 0].std() < 12 and p[:, 1].std() < 12:
        print("  ⚠️ 球在画面里几乎固定（标准差 <12px / 224px）-> 同样是信息缺失")
    else:
        print("  ✅ 像素位置与世界位置相关，图像含位置信息 -> 问题不在这里")

    with open("ball_info_check.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("  明细已写入 ball_info_check.json")


if __name__ == "__main__":
    main()
