# 非配对视频作用表示

主训练现在接受各自只有一个视角的独立视频。同步多视角是可选的校准、验证或一致性数据，不能从“有两份视频”推断二者可靠配对。

按本次确认，新增视频损失只更新轻量作用编码器 B 和训练期预测器 P。
导出时冻结 B，删除 P 的使用，将窗口作用 tokens 按时间拼接为 `Z_D`，交给已有读取器 R。
R 仍结合机器人当前场景预测 `g_current/g_remaining`；WAM 在机器人阶段按现有方案更新。
`Z_D` 描述示范中观察到的过程，`g` 描述机器人现在需要实现的作用，二者没有合并。

## 模块与监督

B 读取整个过程窗口的冻结视觉特征和数据有效掩码，经时间编码与固定数量查询聚合为连续 tokens。
P 只读取过去的逐实体／patch特征、B 输出和未来时间查询；没有干净未来特征或未来缓存的直接入口。
过去特征承担外观预测的残差基线。每窗至少有一个过去时刻和两个未来时刻，监督中间过程及终点。

基础目标为有效未来特征的均方误差。有可靠标注时，再加几何误差、独立 Bernoulli 关系／事件损失。
短视频不需要机器人动作、任务要求或完整关系图就能进入预训练。未知字段不生成负标签；全无监督时也不单独训练容量正则。
连续 tokens 有固定数量与宽度，可加入训练期噪声和小幅能量正则；这些约束不是正式的信息率界，也不保证相机不变性。

`effect_fields=()` 的纯视觉窗口不构造实体两两关系张量；patch数量较大时不会为缺失关系标签支付二次规模成本。
几何标签必须使用共同、经过审核的坐标约定和单位；单目估计不能伪装成精确三维真值。
仅含关系／事件的样本不需要向几何头提供标签。机器人 replay 和人类视频共用 B/P 与一致的特征／作用词汇身份。

## 输入与预处理

`video_pretrain` 数据格式见 [video_data.md](video_data.md)。它与包含机器人执行监督的 `training_sample` 是独立类型。

```bash
evo-wam preprocess-video --manifest /server/raw-human-video.json --output /server/windows/human-001
evo-wam pretrain-video --config configs/video/U2_effect_constraints.json \
  --index /server/video-index.json --steps 900 --device cuda --output /server/runs/video-effects
evo-wam pretrain-video --config configs/video/U2_effect_constraints.json \
  --index /server/video-index.json --steps 100 --device cuda \
  --resume /server/runs/video-effects/video_encoder.pt --output /server/runs/video-effects
```

续训的累计步数仍受预算限制，上例分900+100步完成登记预算。
预处理复用本地 Wan VAE，权重路径仍由服务器填写。原始片段须已筛选为连续操作；本版不自动识别镜头切换。
相机运动、演员外观与接触稳定性仍需要独立诊断，裁剪不能充当真实三维视角标签。

索引先检查原始视频、相邻窗口、转载来源及机器人轨迹的连通组，禁止跨训练／验证／测试划分。
`bridge_sources` 可将有限的人机任务对应数据加入同一来源审计，不会自动生成训练配对。
检查点保存来源记录、特征版本、窗口策略、各领域实际窗口数和更新数；用于下游审计的训练来源包含已消费域的完整训练来源组件及相关人机桥边，独立未使用域不冒充已训练。只按需读取特征数组，已消费数组的校验值用于续训一致性检查。

## 接入现有机器人训练

已有原始 `demo_view_i` 特征需要提供 `demo_feature_space_id` 和
`demonstration_layouts: [{frames, tokens_per_frame, frame_times}]`。不能从展平序列猜测视频时间轴。

```bash
evo-wam encode-demonstrations --artifact /server/runs/video-effects/video_encoder.pt \
  --manifest /server/robot-sample/sample.json --output /server/robot-sample-encoded
evo-wam train --config /server/audited-robot-config.json --index /server/encoded-index.json \
  --demo-encoder /server/runs/video-effects/video_encoder.pt --checkpoint /server/zero-wam \
  --stage interface --steps 1000 --output /server/runs/interface
```

随后按已有 `reader`、`joint` 路线训练，继续提供相同 `--demo-encoder`。
真实配置的 `dimensions.demo_dim` 必须等于 B 的 `latent_dim`。`raw_features` 与 `video_effect_tokens` 明确区分；后者记录编码器 SHA256、特征空间、token宽度和窗口策略。
机器人训练把该编码器实际训练来源与所有下游划分联查，防止预训练已经读过所谓“未见示范”。
检查点和运行记录保留同一身份，推理时换错编码器会拒绝。

从原始视频直接推理可在 `preprocess-visual` 配置中提供 `demo_encoder_artifact`、`demo_feature_space_id`，可另给 `demo_encoder_sha256`。
原生 WAM 仍读取 R 生成的作用要求；不会把原始示范或 B 输出直接塞给辅助头绕过主干。

## 最小对照与诊断

| 配置 | 机器人窗口预算 | 人类窗口预算 | 目标 |
|---|---:|---:|---|
| U0_robot_only | 250 | 0 | 相同 B/P，只用机器人特征预测建立基线 |
| U1_feature_prediction | 250 | 750 | 增加同一批网络视频，只监督多时域特征预测 |
| U2_effect_constraints | 250 | 750 | 同样视频与参数容量，增加有效作用监督、瓶颈噪声和能量正则 |

通用入口也允许 `domain_schedule=["human"]`，可在机器人数据就绪前独立预训练编码器；这不建立执行能力。上表正式对照保留固定机器人回放。

三组使用同一架构和初始化种子；U1/U2的数据调度与总步数相同。U0的总计算更少，须单独报告，不能称为总算力完全匹配。
这里的“普通预测”对照仍经过同一小瓶颈以控制参数容量；U1/U2的差值检验整组约束的增量，不能单独归因于控制关系。
若要进一步归因，可在固定噪声和容量设置下，仅改变关系／事件损失权重。
所有默认维度是数值验收用，真实特征维度需要服务器配置。

```bash
evo-wam evaluate-video --artifact /server/runs/video-effects/video_encoder.pt \
  --index /server/heldout-video-index.json --split validation --output /server/results/video-diagnostics.json
```

该诊断在留出窗口上报告原始、置零、跨窗口打乱 z 的多时域特征误差。它能检查预测器是否利用瓶颈，不能替代机器人执行证据。
最终仍需删除 P，固定机器人状态换示范，在未见视角和任务组合上测绑定、控制关系及闭环行为；网络视频损失没有直接更新 WAM 的主张也保持明确。

相关技术依据包括 [LIT](https://arxiv.org/abs/2609.12641)、[LAPA](https://arxiv.org/abs/2410.11758)、
[UniVLA](https://arxiv.org/abs/2505.06111) 和 [GeoLAM](https://arxiv.org/abs/2609.17099)。本实现的连续瓶颈与部分作用监督是可检验的实现选择，研究增量仍需靠迁移与执行实验建立。
