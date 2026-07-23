# pi05 微调 Runbook — 星尘 S1 仿真抓球

用官方 `astribot_pi05` 框架 + 官方预训练权重 `pi05_base_bfloat16`，在自采的仿真抓球数据上微调，再仿真闭环评测。

相关路径（服务器）：
- 训练框架：`/data0/tzs/astribot_pi05/astribot_pi05-main`
- 预训练权重（压缩包）：`/data0/tzs/astribot/cheakpoint/pi05_base_bfloat16.zip`
- 采集数据（HDF5）：`/workspace/demos`（容器内）→ 宿主机对应挂载目录

## 与 ACT 版的关系

同一批 HDF5，两条转换路线：`convert/hdf5_to_lerobot.py`（ACT，关节空间）与
`convert/hdf5_to_pi05.py`（pi05，笛卡尔 SO3）。ACT 是快速基线兼评测基建验证，
pi05 是主线。评测客户端 `eval/eval_client.py` 两者复用（换推理服务地址即可）。

## 数据格式对齐（关键）

pi05 是标准 LeRobot 数据集，但列名和动作空间固定为官方 31 维全身笛卡尔 SO3：

| 段 | 索引 | 内容 |
|---|---|---|
| torso | 0:9 | SO3 位姿（xyz + 旋转矩阵前两行 6D） |
| left_arm | 9:18 | SO3 位姿 |
| left_gripper | 18:19 | 标量 0-1 |
| right_arm | 19:28 | SO3 位姿 |
| right_gripper | 28:29 | 标量 |
| head | 29:31 | 2 关节 |

`hdf5_to_pi05.py` 用 MJCF 做 FK 从记录的关节值生成上述位姿。采集只动左臂+躯干，
所以：右臂位姿由常量 `START_ARM_RIGHT` FK（右臂全程未动，精确）；右爪=0；
头部采集未记录，抓放任务与头部无关，默认常量 `[0,0]`（`--head` 可改）。delta 与
坐标系变换由官方 processor 在训练时完成，本脚本只存 base 系绝对位姿。

## 步骤

### 0. 解压权重 + 装训练环境

```bash
cd /data0/tzs/astribot/cheakpoint
unzip pi05_base_bfloat16.zip                 # 得到 pi05_base_bfloat16/ 目录

cd /data0/tzs/astribot_pi05/astribot_pi05-main/astribot_pi05
conda create -n pi05 python=3.11 -y && conda activate pi05
bash install.sh                              # 装 torch2.7/transformers 补丁等
# 境内：export HF_ENDPOINT=https://hf-mirror.com（下载 paligemma tokenizer 等）
```

### 1. 转换数据

```bash
conda activate pi05
export ASTRIBOT_SIMU_ROOT=/data0/tzs/astribot/astribot_simulation   # FK 需加载 MJCF
python /path/to/astribot-act/convert/hdf5_to_pi05.py \
    --src <宿主机上的 demos 目录> \
    --root /data0/tzs/astribot/datasets/pi05_pick_ball \
    --repo-id local/astribot_pick_ball
```

转出后目录名 `pi05_pick_ball`，训练配置的 `dataset_root` 指其上级、`repo_id_list` 填 `pi05_pick_ball`。

### 2. 放置配置 + 计算归一化统计

```bash
cd /data0/tzs/astribot_pi05/astribot_pi05-main/astribot_pi05
cp /path/to/astribot-act/pi05/train_pick_ball.yaml configs/train_example/
# 按你的实际路径核对 yaml 里 model_name_or_path / dataset_root

bash scripts/train/compute_norm_stats.sh -c configs/train_example/train_pick_ball.yaml
# 会在 normalization_base_path (outputs/pi05_pick_ball/norm) 下生成 norm_stats.json
```

### 3. 训练

```bash
# 单卡
bash scripts/train/train.sh -c configs/train_example/train_pick_ball.yaml
# 多卡（如 4 卡）
bash scripts/train/train.sh -c configs/train_example/train_pick_ball.yaml -n 4
# 断点续训
bash scripts/train/train.sh -c configs/train_example/train_pick_ball.yaml -r latest
```

checkpoint 在 `outputs/pi05_pick_ball/checkpoint-*/`，TensorBoard 日志在 `outputs/pi05_pick_ball/runs/`。

### 4. 推理服务 + 仿真评测

```bash
# 训练机：起推理服务（WebSocket :8000）
cp /path/to/astribot-act/pi05/infer_pick_ball.yaml configs/infer_example/
# 改 checkpoint 指向某个 checkpoint-XXXXX
bash scripts/infer/infer.sh -c configs/infer_example/infer_pick_ball.yaml

# 评测客户端见 eval/pi05_eval_client.py（本仓库，ROS 容器内跑）
```

## 前置检查与坑

- **显存**：pi05 微调官方推荐 80G。单卡 24-48G 要靠 `train_expert_only: true` +
  `GPU_memory_opt_level: 2` + 小 batch + 梯度累积，速度慢但能跑。
- **共享服务器**：你的服务器 144 核常年满载，训练前 `nvidia-smi` 看清空闲卡，
  用 `CUDA_VISIBLE_DEVICES` 绑定，别和别人抢同一张卡。
- **两个待核对项**（首次转换后抽验）：
  1. 躯干参考系：脚本自动探测 torso site/body 并打印，若官方给了 golden 样本，
     比对 torso 段 9 维数值确认选对了 frame。
  2. 夹爪量纲：脚本按 /100 归一到 0-1；若官方 base 权重用别的约定，训练能收敛但
     真机迁移前需对齐。
- **数据量**：150 条对单任务微调偏少，先看验证 loss 是否平稳下降；不行就加采到 300+。
- **flash-attn**：没装成功就把两个 yaml 的 `attn_implementation` 改 `eager`。
