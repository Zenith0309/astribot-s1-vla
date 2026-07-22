#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_policy.py — ACT 策略推理服务（训练环境运行，GPU）

与 eval_client.py（ROS 容器内）配对使用，协议为 TCP + 长度前缀 JSON：
  {"type":"reset"}                                    -> {"ok":true}
  {"type":"act","state":[12],"image_jpeg_b64":"..."}  -> {"action":[12]}

用法:
  python eval/serve_policy.py --ckpt outputs/train/act_pick_ball/checkpoints/last/pretrained_model
  # 默认监听 0.0.0.0:9000（客户端在同机 Docker --network=host 内，走 127.0.0.1）
"""

import argparse
import base64
import json
import socket
import struct

import cv2
import numpy as np
import torch

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
except ImportError:
    from lerobot.common.policies.act.modeling_act import ACTPolicy

CAM_FEATURE = "observation.images.third_person"     # 与转换脚本约定一致


def recv_msg(conn):
    head = b""
    while len(head) < 4:
        chunk = conn.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    (length,) = struct.unpack(">I", head)
    body = b""
    while len(body) < length:
        chunk = conn.recv(min(65536, length - len(body)))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode())


def send_msg(conn, obj):
    body = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(body)) + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model 目录")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[load] {args.ckpt} -> {args.device}")
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.to(args.device)
    policy.eval()
    policy.reset()
    print("[load] 完成，等待客户端连接...")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[conn] 客户端 {addr}")
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                if msg["type"] == "reset":
                    policy.reset()                  # 清空 action 队列（每 episode 一次）
                    send_msg(conn, {"ok": True})
                elif msg["type"] == "act":
                    state = torch.tensor(msg["state"], dtype=torch.float32,
                                         device=args.device).unsqueeze(0)
                    jpg = base64.b64decode(msg["image_jpeg_b64"])
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                    img_t = img_t.unsqueeze(0).to(args.device)
                    batch = {"observation.state": state, CAM_FEATURE: img_t}
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    send_msg(conn, {"action": action.squeeze(0).cpu().tolist()})
                else:
                    send_msg(conn, {"error": f"unknown type {msg['type']}"})
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("[conn] 客户端断开，继续等待...")


if __name__ == "__main__":
    main()
