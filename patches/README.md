# 官方框架 `astribot_pi05` 的本地补丁

这些补丁打在**官方框架仓库**（`astribot_pi05`）上，不属于本仓库代码。
换机器 / 重装框架后需要重新应用，否则会踩回原来的坑。

## 应用方式

```bash
cd <astribot_pi05 仓库根>          # 含 .git 的那一层
git apply --check patches/0001-tokenizer-path-resolution.patch   # 先干跑检查
git apply         patches/0001-tokenizer-path-resolution.patch
```

`git apply --check` 通过再真正应用。若上游更新导致冲突，看下面的意图说明手工改。

---

## 0001 — tokenizer 路径解析（消除硬编码绝对路径）

**改的文件**：`astribot_pi05/pi05/models/pi05/processor_pi05.py`

**为什么要改**

上游代码是：

```python
if tokenizer is None:
    tokenizer = GemmaTokenizer.from_pretrained("google/paligemma-3b-pt-224")
```

境内直连 HuggingFace 常超时，所以早期改成了从魔搭下载后的**绝对路径**
（`/home/<user>/.cache/modelscope/...`）。这带来两个问题：

1. **换机器或换账号直接崩**，且报错发生在 processor 构造阶段，不容易一眼看出根因。
2. 路径里带具体用户名，不可移植，也不适合提交。

**改成什么**

按优先级解析，四级回退：

| 优先级 | 来源 | 用途 |
|---|---|---|
| 1 | `tokenizer_path` 参数（可从 YAML 的 `processor_kwargs.tokenizer_path` 传入） | 单次实验显式指定 |
| 2 | 环境变量 `ASTRIBOT_TOKENIZER_PATH` | **换机器时最省事的办法** |
| 3 | 本地常见位置（`~` 会展开，故对任何用户都成立） | 默认无感可用 |
| 4 | 回退 hub id `google/paligemma-3b-pt-224` | 有网络时仍可工作 |

前两级若给了路径但目录不可用，会**立刻抛 `FileNotFoundError` 并说明缺什么文件**，
而不是静默回退到别处——静默回退会让人以为用的是自己指定的词表。
每次解析成功都会 `logging.info` 打出实际来源，避免"日志说一套、实际用另一套"。

**新机器上的最小操作**

```bash
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
python3 -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/paligemma-3b-pt-224', allow_patterns=['tokenizer*','*.json','special_tokens*']))"
# 若下载目录不在候选列表里：
export ASTRIBOT_TOKENIZER_PATH=<上面打印的目录>
```

**兼容性说明**

- `tokenizer_path` **不写入** `init_config`，因此不会被 `save_pretrained` 固化进
  checkpoint 的 `processor_config.json`，checkpoint 保持可移植。
- 推理路径不受影响：`Pi05Processor.from_pretrained()` 本来就从 checkpoint 的
  `processor/` 目录加载 tokenizer，显式传入，不走这条解析链。
- 官方 `pi05_pnp` checkpoint 自带的 `processor/tokenizer.model` 与魔搭副本
  **md5 完全相同**，两者编码同一句话得到的 token id 一致，可互换。

**验证过的行为**（9 项，全部通过）

1. 无任何指定 → 命中本地候选（`~` 正确展开）
2. 环境变量优先于本地候选
3. 显式参数优先于环境变量
4. 显式参数路径写错 → 抛 `FileNotFoundError`，不静默回退
5. 环境变量路径写错 → 同样抛错
6. 两个来源真正加载 tokenizer，`vocab=257152`，编码结果逐 token 一致
7. 走默认来源构造完整 `Pi05Processor` 成功（模拟训练路径）
8. 用 `tokenizer_path` 参数构造成功（模拟 YAML 传入）
9. `init_config` 不含 `tokenizer_path`，checkpoint 不被污染
