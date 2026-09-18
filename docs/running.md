# 运行与复现

## 本地轻量检查

```bash
git submodule update --init --recursive
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
.venv/bin/evo-wam doctor
.venv/bin/evo-wam check
```

源码检查也可用 `PYTHONPATH=src .venv/bin/python -m evo_wam ...`。标准库 unittest 不另需测试框架。没有 CUDA 或原生依赖时，native 测试明确跳过；这不算 native 验证成功。

## 原生 CUDA 环境

上游验证栈为 Python 3.10、PyTorch 2.9.0/CUDA12.6、diffusers0.36.0、transformers4.55.2。先安装匹配 CUDA 的 torch/torchvision，再装原生额外依赖：

```bash
uv pip install --python .venv/bin/python torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e '.[native]'
.venv/bin/evo-wam check-native
```

ICL 实际使用原生 PyTorch FlexAttention，运行上述检查无需构建未用到的 flash-attn。加载器只对固定源码中一处 legacy 导入块设置显式缺依赖保护，不伪造 flash 模块、不替代 attention、不修改 submodule；真正调用 legacy FlashAttention 时仍会报缺依赖。原生小模型使用随机权重，仅检验真实架构的条件路由、MCP、采样和梯度。

仅在需要 legacy FlashAttention 路径时另行安装真实库（源码构建可能较慢）：

```bash
uv pip install --python .venv/bin/python ninja wheel
MAX_JOBS=2 FLASH_ATTN_CUDA_ARCHS=80 uv pip install --python .venv/bin/python flash-attn==2.8.3.post1 --no-build-isolation
```

## 数据

每样本采用审计过的 JSON+NPZ，详见 `data.md`。数据集索引：

```json
{"samples":[{"manifest":"episode_001/sample.json","split":"train"}]}
```

索引仅读元数据，不将全数据集张量一次载入内存。训练按需加载单个样本。训练、验证、测试按原始人类示范和机器人轨迹的连通组隔离。

```bash
.venv/bin/evo-wam validate-data --index /absolute/dataset/index.json
.venv/bin/evo-wam make-fixture --output outputs/fixture
```

fixture 是显式标记的数值输入，不是采集的机器人数据。它覆盖所有四个 IFP 目标的有效窗口，提供原生 clean latent/grid、物体观测特征和纯观测实体池化权重。真实数据的相机标定、对象跟踪、接触标注、动作归一化需由采集流程提供；代码不从视频或文件名编造这些真值。

Native训练、候选校准和推理都要求 `action_space` 元数据：`representation="zero-wam-normalized"`、固定 `normalization_id`、`dimension` 及 Boolean `valid_channels`。F 的动作必须与展平后的 native 动作标签一致，不能混用米/弧度等物理单位和归一化单位。未用通道必须为0；推理在初噪声和每一步去噪后都清零。训练 artifact 记录并验证归一化注册表。

native 训练窗口必须真实完整执行。仅执行前缀的候选记录用于 `calibrate-f`，其未执行部分由 label_valid 屏蔽。不能用计划动作后半段去解释续接策略结果。

## 分阶段训练

原生随机小模型的一步集成路径（GPU/原生依赖齐备后）：

```bash
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture/index.json --tiny-native --stage interface --steps 1 --output outputs/interface
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture/index.json --tiny-native --stage reader --initialize outputs/interface/adapter.pt --steps 1 --output outputs/reader
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture/index.json --tiny-native --stage joint --initialize outputs/reader/adapter.pt --steps 1 --output outputs/joint
```

正式训练用本地发布权重替换 `--tiny-native`：

```bash
.venv/bin/evo-wam train --config /absolute/audited-config.json --index /absolute/dataset/index.json --checkpoint /absolute/zero-wam-pretrain --stage interface --steps 1000 --output outputs/interface-real
```

正式配置必须根据真实输入修改维度并将 `synthetic_dimensions_only` 设为 false；源码版本、动作维度与 IFP 架构均检查。单卡入口沿用上游每rank一个样本约束，不假定16GB显存能装完整10.8B模型。`native_config.json` 记录实际架构，tiny 的层号覆盖不会伪装成原30层配置。

每次更新记录 losses、条件/无条件标记、阶段、梯度范数、实际计算时间与模型来源到 `metrics.jsonl`。无条件分支不运行任务读取、真实要求教师或成对损失。

## 恢复与阶段切换

```bash
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture/index.json --tiny-native --stage interface --resume outputs/interface/adapter.pt --steps 1 --output outputs/interface
```

`--resume` 要求相同阶段及完整配置，恢复 optimizer、更新步数、采样generator与torch CPU/CUDA RNG。累计尝试步数不得超过登记预算。`--initialize` 用于阶段切换：复用模型参数，重新建立该阶段 optimizer。

tiny artifact 保存包括冻结基干在内的全部权重，因此更换进程的初始化随机数不会悄悄换模型。正式 artifact 只存接口、LoRA和MCP等增量，保留原始 safetensors/config 的 SHA256 身份；恢复前核验。两种模式不能混用。

保存采用同目录临时文件再替换；权重和运行日志默认忽略，不推送 Git。

## 候选分布适配与推理

用已冻结的读取器/策略在训练任务上产生候选并记录实际结果。每条记录的元数据须包含生成策略 artifact 的 `policy_artifact_sha256`；前缀之后改用其他动作，原候选标签不得跨过 executed_steps。

```bash
.venv/bin/evo-wam calibrate-f --artifact outputs/joint/adapter.pt --index /absolute/candidate-data/index.json --steps 100 --output outputs/calibrated/adapter.pt
.venv/bin/evo-wam predict --artifact outputs/joint/adapter.pt --manifest outputs/fixture/observation.json --candidates 1 --diagnostic --output outputs/prediction
```

`calibrate-f` 只更新 F，记录数据窗口和实际交互量，不反复开启新一轮适配，也不校准成功概率。四候选预测要求已完成这一有限适配：

```bash
.venv/bin/evo-wam predict --artifact outputs/calibrated/adapter.pt --manifest /absolute/evaluation/observation.json --checkpoint /absolute/zero-wam-pretrain --candidates 4 --scoring-config /absolute/validation/scoring.json --output outputs/candidates
```

正式预测需 artifact 配置的 `validation_locked=true`；正式四候选另需 `scoring.json` 提供 `validation_locked=true`、校准后策略的 `policy_artifact_sha256`、全部 `field_weights`、`uncertainty_weight` 和有限 `max_cost`，不得通过命令行覆盖。`--diagnostic` 明确排除正式测试主张，允许临时 `--max-cost`。

部署输入采用独立 `kind="observation"` 的JSON+NPZ，仅含 `entity_ids`、`robot_history`、`proprio_history`、`embodiment`、`robot_latent` 和按时间/空间排序的 `demo_view_0` 等示范。JSON声明 `view_ids`、`chunk_size`、`actions_per_frame` 和 `action_space`。推理既不要求也不读取未来标签、真实要求或记录动作；`make-fixture` 同时生成此观测示例。空当前要求会拒绝，不作为任务完成。

输出为归一化动作数组和诊断 JSON，`commands_sent=0`。它不会直接发送实机命令。真实控制器须提供经标定的动作解码、控制频率和执行接口；闭环及快照回放接法见 `evaluation.md`。

## 结果应如何解释

测试必须分别记录：CPU契约、小计算图、原生随机小模型、发布权重、RoboTwin、真实机器人。toy四条件测试验证评测器能够区分左右任务和保持/释放，不说明训练模型已经学会这些任务。公开检查点复评和公开子集重训不同；两套RoboTwin成功判据分别报告。HOST是后置独立验证。
