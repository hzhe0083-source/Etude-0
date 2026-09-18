# 验证记录

日期：2026-09-19。所有下列结果区分代码/计算图验收与研究实验结果。

## 已实际运行

| 检查 | 结果 | 能支持的结论 |
|---|---|---|
| `evo-wam check` | 81 项通过，无跳过 | 契约、损失、数据、模型、训练、评测、CLI与原生封装回归通过 |
| 原生 Zero-WAM CUDA smoke | 通过 | 实际 FlexAttention 视频/动作/IFP前向反向与采样可运行 |
| current/remaining 路由 | 通过 | 视频可区分两类条件；动作只直接读取 current |
| MCP固定Phi换任务 | 通过 | 辅助分支无独立任务条件捷径；纯动作上下文不携带 g |
| 执行一致性梯度 | 通过 | 读取器获得直接条件梯度，视频采样链不接受该梯度 |
| 原生无效动作通道 | 通过 | 初始噪声、每个采样步及最终输出的无效通道为0 |
| 三阶段命令行训练 | 通过 | interface、reader、joint实际更新；joint启用4个IFP时域、交互与成对损失 |
| 保存、恢复与阶段初始化 | 通过 | 冻结tiny基干、增量参数、optimizer与随机状态可保存/恢复 |
| 仅观测推理 | 通过 | 输入不含未来标签、真实要求或记录动作，生成单候选归一化动作 |
| F有限适配及四候选 | 通过 | 合成前缀标签、策略身份核对、F-only更新和固定候选排序链路运行 |
| 空当前要求 | 正确拒绝 | 不把空要求作为已完成任务或零代价可执行目标 |
| 未来池化泄漏检查 | 通过 | 交互头只能池化当前预测chunk，拒绝后续teacher-forced tokens |

运行环境：Python 3.10.19、PyTorch 2.9.0+cu126、diffusers 0.36.0、transformers 4.55.2，RTX 3080 Laptop GPU（16GB）。

原生小模型为实际上游类的随机小配置（1层、2 heads、head dimension 18），不是用另一个attention近似的模型。独立smoke使用1个MCP头，命令行联合训练使用4个。已核对原生编译attention的原始调用对象确为 `torch.nn.attention.flex_attention.flex_attention`。

FlashAttention未安装。固定上游一处未用于ICL的硬导入通过显式内存加载保护处理；未伪造依赖模块、未改动submodule，真正调用legacy FlashAttention仍报缺依赖。

三阶段、F适配、候选与空要求检查使用标为synthetic的数值输入及随机tiny基干。`commands_sent=0`。它们证明软件路径可运行，**不证明机器人已经学会任务、控制关系或跨视角迁移**。损失值不是论文结果。

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
