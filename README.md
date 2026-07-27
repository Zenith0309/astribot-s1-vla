# astribot-s1-vla

Astribot S1 人形机器人的 **Vision-Language-Action (VLA) 数据与训练全流程工具集**：从 MuJoCo 仿真采集专家演示、到真机数据处理、双路格式转换（ACT / pi0.5）、模型训练、闭环评测，一站式打通。

> 目标：为 Astribot S1 提供一套可复现的 VLA 数据生产 + 模型微调 + 评测管线，覆盖**仿真验证**与**真机部署**两条路径。

## 这个项目解决什么

VLA 模型（pi0/pi0.5 等）无法对未见过的本体零样本工作，正确路径是**当预训练底座微调**。微调需要三样东西：高质量演示数据、正确的数据格式、可靠的训练评测管线。本项目把这三样全部做齐并实测跑通：

- **数据生产**：仿真侧脚本专家批量采集（无需人工遥操作即可产出演示），真机侧遥操作数据的审查与转换。
- **格式对齐**：把采集数据转成官方 pi0.5 要求的 34 维笛卡尔 SO3 LeRobot 格式，趟平官方 fork 的一批隐藏坑。
- **训练评测**：多卡 DeepSpeed 微调 + 仿真/真机闭环评测客户端。

## 目录结构

```
astribot-s1-vla/
├── collect/                       # 仿真数据采集（ROS2 容器内）
│   ├── scripted_grasp.py          #   脚本专家：11-DOF IK + 力矩自适应夹取 + 底盘站位
│   ├── collect_demos.py           #   批量采集 -> HDF5（三相机 + 头部视线跟踪）
│   ├── go_start_pose.py           #   回官方 VLA 初始姿态（最小控制示例）
│   └── scene_*.xml                #   自定义场景（三色球 + 容器 + 第三人称相机）
├── convert/                       # 数据格式转换
│   ├── hdf5_to_pi05.py            #   自采 HDF5 -> pi0.5 LeRobot（34维 SO3，FK 生成位姿）
│   ├── hdf5_to_lerobot.py         #   自采 HDF5 -> ACT LeRobot（关节空间，快速基线）
│   ├── official_hdf5_to_mcap.py   #   官方真机 HDF5 -> MCAP（Foxglove 审查用）
│   └── hdf5_to_mcap.py            #   自采 HDF5 -> MCAP
├── train/
│   └── train_act.sh               # ACT 训练（LeRobot 自带实现，快速基线）
├── eval/
│   ├── serve_policy.py            #   ACT 推理服务（TCP + JSON）
│   ├── eval_client.py             #   ACT 仿真评测客户端
│   ├── pi05_eval_client.py        #   pi0.5 仿真评测客户端（msgpack/WebSocket + SO3->IK）
│   └── eval_client_real.py        #   真机评测客户端（官方 SDK 高层接口）
├── pi05/
│   ├── train_pick_ball.yaml       # pi0.5 训练配置（多卡 4090 档）
│   ├── infer_pick_ball.yaml       # pi0.5 推理服务配置
│   └── README_pi05.md             # pi0.5 微调 Runbook
├── docs/
│   └── pi05微调完整指南.md         # ⭐ 完整微调指南（数据采集/标注/转换/训练/评测/踩坑）
└── requirements_train.txt
```

## 两条路径

### 路径一：仿真验证（快速迭代，零真机成本）

用脚本专家在 MuJoCo 里批量采集演示，验证整条"数据→训练→评测"管线。适合快速试错、验证工具链、对比不同模型/checkpoint。

```bash
# 1. 采集（ROS2 容器内，仿真运行中）
python3 collect/collect_demos.py --episodes 150 --out demos --only-success \
    --cameras /astribot_whole_body/camera/head_rgbd/image_raw \
              /astribot_whole_body/camera/left_wrist_rgbd/image_raw \
              /astribot_whole_body/camera/right_wrist_rgbd/image_raw

# 2. 转 pi0.5 格式（训练环境）
export ASTRIBOT_SIMU_ROOT=<astribot_simulation 根>
python convert/hdf5_to_pi05.py --src demos --root datasets/pick_ball

# 3. 归一化 + 训练 + 评测：见 docs/pi05微调完整指南.md
```

> 注意：纯仿真数据训的策略因视觉域差距无法直接上真机——仿真路径的价值是**验证管线与方法**，真机部署需真机数据。

### 路径二：真机数据微调（可部署策略）

用官方链路遥操作采集真机数据，微调 pi0.5 得到能直接上真机的策略。数据采集、标注、转换、训练的完整步骤见 **[docs/pi05微调完整指南.md](docs/pi05微调完整指南.md)**。真机数据原生带底盘/力矩/三相机，无 sim2real 鸿沟。

## 核心文档

**[docs/pi05微调完整指南.md](docs/pi05微调完整指南.md)** 是本项目的核心 —— 从数据采集到部署评测的完整 pi0.5 微调指南，包含：

- 环境搭建（境内 tokenizer 离线化、ffmpeg）
- 数据采集原则（随机化防因果混淆）与官方 HDF5 格式
- 数据标注（单任务 `--coarse-prompt` / 多任务 JSON 标注 / 成功标签）
- 转换、归一化、多卡训练、推理服务、闭环评测
- **官方框架一批隐藏坑的解法**（add_frame 签名、sub_task_index、consolidate、离线 tokenizer、归一化路径、图像 CHW、state 34 维、sync 服务、因果混淆）
- 34 维 SO3 动作空间布局附录

## 技术要点

- **动作空间**：官方 34 维全身笛卡尔 SO3 = `torso9 + left_arm9 + left_gripper1 + right_arm9 + right_gripper1 + head2 + chassis3`，底盘天然可学。
- **脚本专家**：11-DOF（躯干4+手臂7）阻尼最小二乘 IK + 官方 TCP 参考系 + 力矩自适应夹取 + 仿真时钟定步（抗共享服务器实时率波动）+ 底盘按物体位置站位。
- **观测格式**：三相机 CHW float[0,1] + 34 维 state + prompt，msgpack/WebSocket 与官方 pi0.5 推理服务同构。
- **训练**：多卡 DeepSpeed ZeRO-2，小数据集冻结主干只训动作 expert 防过拟合。

## 环境

- 采集侧：Docker + ROS2 Humble + MuJoCo（仿真环境搭建见配套的 astribot-mujoco-tutorial）
- 训练侧：Python 3.11 + CUDA 12.1+ + `astribot_pi05` 框架，`pip install -r requirements_train.txt`

## 参考

- 官方框架：`astribot_pi05`（训练/推理）、`astribot_eval`（真机执行）、`astribot_simulation`（仿真）
- LeRobot：<https://github.com/huggingface/lerobot>
- 支持：support@astribot.com
