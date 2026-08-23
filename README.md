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
│   ├── hdf5_to_pi05.py            #   自采 HDF5 -> pi0.5 LeRobot（31维 SO3，FK 生成位姿）
│   ├── hdf5_to_lerobot.py         #   自采 HDF5 -> ACT LeRobot（关节空间，快速基线）
│   ├── official_hdf5_to_mcap.py   #   官方真机 HDF5 -> MCAP（Foxglove 审查用）
│   └── hdf5_to_mcap.py            #   自采 HDF5 -> MCAP
├── train/
│   └── train_act.sh               # ACT 训练（LeRobot 自带实现，快速基线）
├── eval/                          # 评测 + ⭐诊断工具（排障顺序见 docs 指南 §11）
│   ├── pi05_eval_client.py        #   pi0.5 闭环评测（记录 closest_cm / IK 成败 / 预测轨迹）
│   ├── head_info_check.py         #   ⭐⭐ 判观测里有没有物体位置信息（改观测设计后必跑）
│   ├── replay_check.py            #   ⭐ 拿训练帧回放，分离"模型问题"与"评测管线问题"
│   ├── obs_probe.py               #   对比评测/训练两端的观测分布（含存图肉眼比对）
│   ├── openloop_replay.py         #   开环重放专家动作，隔离 IK/下发/时序故障
│   ├── ball_info_check.py         #   从转换后数据集查像素位置与世界位置的相关性
│   ├── serve_policy.py            #   ACT 推理服务（TCP + JSON）
│   ├── eval_client.py             #   ACT 仿真评测客户端
│   └── eval_client_real.py        #   真机评测客户端（官方 SDK 高层接口）
├── patches/                       # 打在官方框架上的补丁（重装框架后要重新 git apply）
├── pi05/
│   ├── train_pick_ball.yaml       # v1 训练配置（历史存档，勿覆盖其 norm/checkpoint）
│   ├── train_pick_ball_v2.yaml    # 历史（裸底座 + 数据集v2）
│   ├── train_pick_ball_v3.yaml    # 历史（pnp起点 + 数据集v2，闭环0%：观测缺信息）
│   ├── train_pick_ball_v4.yaml    # ⭐ 当前用这个（pnp起点 + 数据集v3，state 含真实 head）
│   ├── infer_pick_ball.yaml       # pi0.5 推理服务配置（server_type 必须 sync）
│   └── README_pi05.md             # pi0.5 微调 Runbook
├── docs/
│   ├── pi05微调完整指南.md         # ⭐ 真机主线：采集/转换/训练/推理服务/astribot_eval 闭环/踩坑
│   └── pi05微调完整指南_仿真版.md  # 仿真链路存档（三轮失败的量化根因 + 框架语义坑）
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

用官方链路遥操作采集真机数据，微调 pi0.5 得到能直接上真机的策略。**完整步骤见 [docs/pi05微调完整指南.md](docs/pi05微调完整指南.md)**（真机版，覆盖采集→转换→训练→推理服务→`astribot_eval` 闭环执行）。真机数据原生带底盘/力矩/三相机、`state[29:31]` 天然含真实 head 角度，无 sim2real 鸿沟。

## 核心文档

**[docs/pi05微调完整指南.md](docs/pi05微调完整指南.md)** 是本项目的核心 —— **真机**微调与闭环部署的完整指南，包含：

- **真机 vs 仿真差异速查**（`async`/`rtc_delay:3` 与仿真的 `sync`/`rtc_delay:0` 是反的，最容易翻车）
- 两套环境搭建（GPU 机训练推理 + 本体电脑执行系统）与安全规程
- 真机遥操作采集、官方 HDF5 实测结构、**观测信息量自检**（开训前必做）
- `--replace-state-with-action`：训练与推理的 state 语义对齐
- 转换、归一化、多卡训练、推理服务、**`astribot_eval` 闭环 runbook**
- 观测/动作协议精确规格 + RTC 如何用 `obs_timestamp`
- **官方框架与执行系统的隐藏坑**（`prompt_way=None` 静默不动、共享缓冲尺寸写死、`model_kwargs` 覆盖语义、离线 tokenizer、图像 CHW…）
- 34 维 SO3 布局 / 真机 HDF5 结构 / 仿真侧经验三个附录

仿真链路的完整记录存档在 **[docs/pi05微调完整指南_仿真版.md](docs/pi05微调完整指南_仿真版.md)**。

## 技术要点

- **动作空间**：**训练 31 维**（`torso9 + left_arm9 + left_gripper1 + right_arm9 + right_gripper1 + head2`），**推理输出 34 维**（末尾 chassis3 由框架从 state 原样复制，不是模型预测的）。也就是说**底盘不参与学习**，想让模型自己移底盘必须改框架——详见 `docs/pi05微调完整指南.md` §6.4。
- **脚本专家**：11-DOF（躯干4+手臂7）阻尼最小二乘 IK + 官方 TCP 参考系 + 力矩自适应夹取 + 仿真时钟定步（抗共享服务器实时率波动）+ 底盘按物体位置站位。
- **观测格式（评测侧）**：三相机 CHW float[0,1] + 34 维 state + prompt，msgpack/WebSocket 与官方 pi0.5 推理服务同构。
- **训练**：多卡 DeepSpeed ZeRO-2，小数据集冻结主干只训动作 expert 防过拟合。

## 环境

- 采集侧：Docker + ROS2 Humble + MuJoCo（仿真环境搭建见配套的 astribot-mujoco-tutorial）
- 训练侧：Python 3.11 + CUDA 12.1+ + `astribot_pi05` 框架，`pip install -r requirements_train.txt`

## 参考

- 官方框架：`astribot_pi05`（训练/推理）、`astribot_eval`（真机执行）、`astribot_simulation`（仿真）
- LeRobot：<https://github.com/huggingface/lerobot>
- 支持：support@astribot.com
