# 验证记录

日期：2026-09-19。所有下列结果区分代码/计算图验收与研究实验结果。

## 本轮 v2 已实际运行

| 检查 | 结果 | 能支持的结论 |
|---|---|---|
| 完整 unittest 回归 | **146 项通过，无跳过** | 包括契约、模型、成对训练、数据、CLI、真实原生小模型、视觉及环境桥检查 |
| 三种绑定状态、容差与事件约束 | 通过 | 缺失/未知对象保留要求；不相容的事件发生时刻不能拼凑满足顺序 |
| 遮挡监督 | 通过 | 单视角要求、latent、交互及蒸馏受数据证据控制；改变隐藏真值不改变对应可见损失和梯度 |
| 暖启动调用与梯度 | 通过 | 启用前零采样/零教师/零学生蒸馏；启用后仍有直接条件梯度；execution=0 保持关闭 |
| 原生历史缓存和 teacher-forcing | 通过 | 真正过去动作影响主视频及 Phi；padding 不影响输出；历史 KV 可反向；原始成对网格不被修改 |
| 原生 FlexAttention | 通过 | 实际原生前向/反向，禁止编译次数耗尽后静默稠密回退 |
| 三阶段命令行条件训练 | 通过 | interface、reader、joint 各一次成功更新；joint 含4个IFP目标、交互和成对监督；暖启动期间无执行蒸馏 |
| 无条件分支 | 通过 | interface/joint 的原生目标可更新；reader 无可训练条件目标时不更新 |
| 准确要求的原生采样 | 通过 | 真实 tiny-native G→视频采样→动作采样产生4步×3维有限候选，commands_sent=0 |
| 示范读取诊断 | 正确拒绝 | 仅一步训练的小模型产生不确定绑定，采样前返回 unresolved_required_binding，不强行选实体 |
| 原始视频预处理 | 通过 | 真实 FFV1 视频与真实随机微型 Wan VAE，RGB/因果端点/归一化/相机与ROI顺序及v2再加载正确 |
| RoboTwin 桥 | 通过 API 检查 | 与固定上游的动作变换数值一致，fake环境验证前缀/实执行历史/双判据/终止；不是仿真任务结果 |

环境：Python 3.10.19、PyTorch 2.9.0+cu126、diffusers 0.36.0、transformers 4.55.2，RTX 3080 Laptop GPU（16GB）。完整回归约23秒，时间不作为性能结论。

CUDA检查使用真实上游类的随机小配置及原生融合 FlexAttention，没有用另一套attention替代。
训练命令行用1层、2 heads、4组MCP；历史梯度检查还使用真实2层小模型，视觉测试使用随机微型Wan VAE。
FlashAttention未安装；固定上游的未用legacy导入仍通过明确加载保护处理，submodule未修改。

本地可复跑：

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/evo-wam make-fixture --output outputs/fixture-v2
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture-v2/index.json --tiny-native --seed 1 --stage interface --steps 1 --output outputs/interface-v2
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture-v2/index.json --tiny-native --seed 1 --stage reader --initialize outputs/interface-v2/adapter.pt --steps 1 --output outputs/reader-v2
.venv/bin/evo-wam train --config configs/V1.json --index outputs/fixture-v2/index.json --tiny-native --seed 1 --stage joint --initialize outputs/reader-v2/adapter.pt --steps 1 --output outputs/joint-v2
```

seed 1 的该数值 fixture 进入条件分支；seed 0 的首步进入无条件分支，两条路径本轮均实际运行。暖启动计数是当前阶段的成功优化更新数；联合阶段无条件原生更新也计数，reader阶段无条件批次不更新、不推进计数。

这批结果仅证明代码和计算图路径，**不证明机器人已经学会任务、控制关系或跨视角迁移**。完整权重按用户要求留作服务器路径，本轮没有下载。

`353eb016` 初版记录过81项测试、v1合成F适配和四候选CLI链路；它们是历史记录，不作为v2真实模型成绩。v2正式候选分布适配与成功率需要服务器数据后重新执行。

## 完整公开配置检查

只读取官方模型配置及revision，在 `torch.device('meta')` 上成功构造全部架构，没有下载或加载完整权重。

- 模型：`Robbyant-Research/zero-wam-pretrain`
- 官方revision：`7040c4195df216c900334ef62d5fdcf05c0601aa`
- 唯一参数：10,789,086,430，约10.8B；共享别名不能重复计数。
- 30层、隐藏维度3072、24 heads、动作维度30。
- 4个MCP组，每组1层，收集层 `[3, 11, 19, 29]`。
- 所有参数/缓冲均为meta；state dict 1880项。

[已核对的官方配置](https://huggingface.co/Robbyant-Research/zero-wam-pretrain/resolve/7040c4195df216c900334ef62d5fdcf05c0601aa/transformer/config.json)

## 尚未验证

- 完整10.8B已训练权重加载、显存适配和训练；本机16GB GPU不能代表所需训练资源。
- RoboTwin真实环境闭环、原始/修改成功判据的模型成绩。
- 真实机器人执行、真实人类跨视角数据及正式统计实验。
- 相机、动作归一化、对象跟踪和真实接触标签质量。
- HOST检查点适配与独立验证。

这些条目不能由toy环境测试、数值fixture或完整配置的meta构造替代。正式实验需在独立验证集锁定配置和评分参数，再分别报告绑定错误、控制失败、前缀进展、局部Oracle及闭环成功率。
