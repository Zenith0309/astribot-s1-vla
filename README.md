# astribot-act

Astribot S1 仿真「抓球入箱」任务的 ACT 小型策略验证管线：**数据采集 → LeRobot 格式转换 → ACT 训练 → 仿真闭环评测**。

目的：在 pi05 预训练权重到位前，用从零训练的 ACT 先打通「数据 → 训练 → 评测」全闭环，验证数据质量与评测基建。后续换 pi05 只需替换推理服务中的模型。

## 目录结构

```
astribot-act/
├── collect/                  # 数据采集（ROS2 容器内运行）
│   ├── scripted_grasp.py     #   脚本化专家：IK + 力矩自适应夹取
│   ├── collect_demos.py      #   批量采集 -> HDF5
│   └── go_start_pose.py      #   回官方 VLA 初始姿态
├── convert/
│   └── hdf5_to_lerobot.py    # HDF5 -> LeRobot v3 数据集（训练环境运行）
├── train/
│   └── train_act.sh          # lerobot-train ACT 训练脚本
├── eval/
│   ├── serve_policy.py       # 推理服务：TCP + JSON（训练环境，GPU）
│   └── eval_client.py        # 评测客户端：30Hz 闭环（ROS2 容器内）
└── requirements_train.txt    # 训练环境依赖
```

架构上训练/推理环境（新版 PyTorch）与控制环境（ROS2 容器）分离，二者通过 TCP 通信——与官方 pi05 的 WebSocket 推理服务架构同构。

## 前置条件

- 仿真环境：Docker 容器 `astribot_sim`（ROS2 Humble + MuJoCo），`with_camera` 配置、`rgb_array` 模式，场景含三色球 + 容器 + `third_person` 相机（搭建见项目文档《星尘VLA仿真抓取与数据采集链路.md》）
- 训练环境：宿主机单独 conda/venv（Python ≥ 3.10），单卡 ≥ 12G 显存

```bash
# 训练环境（宿主机）
conda create -n act python=3.10 -y && conda activate act
pip install -r requirements_train.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 第 1 步：采集数据（ROS 容器内）

```bash
# 终端1: 仿真   终端2(可选): foxglove   —— 均先 source env.sh
cd /workspace/astribot_simulation && source env.sh
python3 collect/collect_demos.py --episodes 150 --out /workspace/demos --only-success
```

单条约 30s，150 条约一个晚上。ACT 单任务建议 ≥ 50 条成功演示，100-150 条更稳。
容器内 `/workspace/demos` 对应宿主机挂载目录（按你的 `docker run -v` 映射找）。

## 第 2 步：格式转换（训练环境）

```bash
python convert/hdf5_to_lerobot.py \
    --src /data0/tzs/astribot/demos \
    --root ./datasets/astribot_pick_ball
```

默认只转成功样本（`*_ok.h5`）。特征布局（与评测端严格一致，勿单独改动）：

| 特征 | 形状 | 含义 |
|---|---|---|
| `observation.state` | (12,) | torso4 + arm_left7 + gripper_left1（实际值） |
| `action` | (12,) | 同布局指令值；夹爪 0-100 百分比，0=全开 |
| `observation.images.third_person` | (H,W,3) | 第三人称相机 RGB，30fps |

## 第 3 步：训练（训练环境）

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 境内下载 ResNet backbone
bash train/train_act.sh                       # 默认 5w 步，batch 8，约需 10G 显存
```

checkpoint 输出至 `outputs/train/act_pick_ball/checkpoints/<step>/pretrained_model/`。

## 第 4 步：闭环评测（两个终端）

```bash
# 终端 A（训练环境，宿主机）：起推理服务
python eval/serve_policy.py \
    --ckpt outputs/train/act_pick_ball/checkpoints/last/pretrained_model

# 终端 B（ROS 容器内，仿真运行中）：起评测客户端
cd <本仓库> && source /workspace/astribot_simulation/env.sh
python3 eval/eval_client.py --episodes 20 --server 127.0.0.1:9000
```

容器为 `--network=host`，客户端直接连宿主机 `127.0.0.1:9000`。
评测复用采集时的场景随机化与 TF 成功判定（球与容器水平距离 <7cm 且落底），结果写入 `eval_results.json`。

## 常见问题

- **`lerobot-train` 不存在**：老版本改用 `python -m lerobot.scripts.train`，参数一致。
- **转换脚本 import 报错**：lerobot 不同版本包路径不同，脚本已做双路径兼容；仍报错时 `pip show lerobot` 确认版本并升级。
- **评测时机器人乱动**：确认转换与评测的 state/action 布局一致、相机 topic 一致（`--camera`）、仿真为同一场景配置。
- **成功率低**：先看训练 loss 是否收敛；再检查数据量（<50 条基本不行）；ACT 对相机视角一致性敏感，确认采集与评测的相机位姿相同。
- **境内网络**：pip 用 aliyun 镜像；HuggingFace 一律 `HF_ENDPOINT=https://hf-mirror.com`。

## 版本说明

- LeRobot 数据集 v3 API（`create / add_frame / save_episode / finalize`），转换脚本对 v2（`consolidate`）做了回退兼容。
- ACT 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)（ALOHA）。
- LeRobot: <https://github.com/huggingface/lerobot>
