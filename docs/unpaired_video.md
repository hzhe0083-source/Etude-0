# 历史可选实验：非配对视频作用表示

当前主线是 [原生 Zero-WAM 人类跨视频 ICL](human_icl.md)：完整示范 A 通过原有 ICL 通路参与独立执行 B 的视频预测，视频损失直接更新部署主干中的适配参数。它不要求先训练或导出下文的 B/P、`Z_D`、R/G/Q，也不要求作用标签或 F 排序。

本文保留此前独立 B/P 预训练的实现、容量候选与命令，供可选对照使用。本文中“主训练”“本次确认”等表述仅指这条历史分支；B/P 的宽度和实验结果不能当作原生 ICL 的配置或验证记录。

主训练现在接受各自只有一个视角的独立视频。同步多视角是可选的校准、验证或一致性数据，不能从“有两份视频”推断二者可靠配对。

按本次确认，新增视频损失只更新轻量作用编码器 B 和训练期预测器 P。
导出时冻结 B，删除 P 的使用，将窗口作用 tokens 按时间拼接为 `Z_D`，交给已有读取器 R。
R 仍结合机器人当前场景预测 `g_current/g_remaining`；WAM 在机器人阶段按现有方案更新。
`Z_D` 描述示范中观察到的过程，`g` 描述机器人现在需要实现的作用，二者没有合并。

## 模块与监督

B 读取整个过程窗口的冻结视觉特征和数据有效掩码。对于规则 patch，B 在聚合前用非线性映射联合编码内容、有效掩码、真实归一化 patch 中心坐标和时间，再经固定数量查询聚合为连续 tokens。先汇总内容再加位置无法恢复对应关系，因此不采用这种顺序。
对于跟踪实体，B 先用共享 GRU 沿稳定实体槽位编码时间轨迹，再聚合实体集合；实体编号不作为物理位置或输入特征。同步重排特征、掩码和实体表不应改变表示。
P 只读取过去的逐实体／patch 特征、B 输出和未来时间查询；patch 路径也读取相同的静态空间坐标，明确自己预测的位置。没有干净未来特征或未来缓存的直接入口。
过去特征承担外观预测的残差基线。每窗至少有一个过去时刻和两个未来时刻，监督中间过程及终点。

基础目标为有效未来特征的均方误差。有可靠标注时，再加几何误差、独立 Bernoulli 关系／事件损失。
短视频不需要机器人动作、任务要求或完整关系图就能进入预训练。未知字段不生成负标签；全无监督时也不单独训练容量正则。
连续 tokens 有固定数量与宽度，可加入训练期噪声和小幅能量正则 `mean(z²)`；这不是 KL 正则或正式的信息率界，也不保证相机不变性。

`effect_fields=()` 的纯视觉窗口不构造实体两两关系张量；patch数量较大时不会为缺失关系标签支付二次规模成本。
几何标签必须使用共同、经过审核的坐标约定和单位；单目估计不能伪装成精确三维真值。
仅含关系／事件的样本不需要向几何头提供标签。机器人 replay 和人类视频共用 B/P 与一致的特征／作用词汇身份。

## 容量候选与测试配置

`configs/video/U0_robot_only.json`、`U1_feature_prediction.json`、`U2_effect_constraints.json`
及现有机器人 JSON 保留微型维度和 `synthetic_dimensions_only: true`，用于测试。
B 的类默认值改为每窗口64个768维 tokens、内部宽度512；P默认使用同样的768／512宽度。
这些是尚未经正式实验验证的容量起点，不能仅凭接口变宽就宣称细节保留充分或迁移改善。

容量实验同时将读取器的 `demo_dim` 和 R／G／Q 共用的 `token_dim` 设为768：

| 容量候选 | B每窗口token数 | B/P latent宽度 | B/P内部宽度 | R/G/Q宽度 |
|---|---:|---:|---:|---:|
| K16 | 16 | 768 | 512 | 768 |
| K64（主候选） | 64 | 768 | 512 | 768 |
| K100 | 100 | 768 | 512 | 768 |

已有配置加载器要求明确填写模型维度，因此不会静默采用类默认值。数据就绪后，从审核过的视频和机器人配置生成上述对照：

```bash
evo-wam make-capacity-configs \
  --video-config /server/audited-video-config.json \
  --robot-config /server/audited-robot-config.json \
  --output /server/capacity-configs
```

输出 `video_K16.json`、`video_K64.json`、`video_K100.json` 和共用的 `robot_768.json`。
生成器改变上述容量字段，并清除旧配置的实验与绑定阈值验证锁，要求新模型重新验证；阈值数值本身不变。输入特征维度、作用标签词汇、窗口和查询时域、噪声与所有损失权重、调度、种子和预算保持不变。
它也保留原配置的合成标记，不会把测试配置自动认证为正式数据配置；尚无真实数据时，可用仓库的U2与V1生成扩大容量的数值检查配置，但不能将其称为正式实验。

`g_current/g_remaining` 的数量仍由角色数乘查询时刻数决定，生成器不将它们改成16／64／100。
这里的 K 只指 B 每窗口的输出数量，十个窗口会产生160／640／1,000个示范 tokens。
三个 K 使用独立的预训练、导出和机器人训练运行，保持同一数据索引与训练预算，并分别记录真实窗口数、示范序列长度、耗时和显存；相同步数不表示相同计算量。
内部宽度512不改变现有时空编码与查询汇总架构。`capacity` 仍乘 `mean(z²)`，不是位姿重建权重或KL；扩容时不同时增加压缩正则。
先检验目标对象、移动方向和中间事件的保留，再测下游绑定与执行，不能只报告特征重建误差。

## 输入与预处理

`video_pretrain` 数据格式见 [video_data.md](video_data.md)。它与包含机器人执行监督的 `training_sample` 是独立类型。
窗口使用 format v2，patch 数据必须保存真实 `patch_grid=[H,W]`、
`patch_coordinate_system="normalized_xy_patch_centers"` 和逐槽位的 `patch_coordinates[N,2]`；不能从 patch 数量猜测二维布局。索引仍为 format v1。

```bash
evo-wam preprocess-video --manifest /server/raw-human-video.json --output /server/windows/human-001
evo-wam pretrain-video --config /server/capacity-configs/video_K64.json \
  --index /server/video-index.json --steps 900 --device cuda --output /server/runs/video-effects
evo-wam pretrain-video --config /server/capacity-configs/video_K64.json \
  --index /server/video-index.json --steps 100 --device cuda \
  --resume /server/runs/video-effects/video_encoder.pt --output /server/runs/video-effects
```

续训的累计步数仍受预算限制；上例假设数据配置登记了1,000步预算，分900+100步完成。
预处理复用本地 Wan VAE，权重路径仍由服务器填写。原始片段须已筛选为连续操作；本版不自动识别镜头切换。
相机运动、演员外观与接触稳定性仍需要独立诊断，裁剪不能充当真实三维视角标签。

索引先检查原始视频、相邻窗口、转载来源及机器人轨迹的连通组，禁止跨训练／验证／测试划分。
`bridge_sources` 可将有限的人机任务对应数据加入同一来源审计，不会自动生成训练配对。
检查点保存来源记录、特征版本、窗口策略、各领域实际窗口数和更新数；用于下游审计的训练来源包含已消费域的完整训练来源组件及相关人机桥边，独立未使用域不冒充已训练。只按需读取特征数组，已消费数组的校验值用于续训一致性检查。

artifact 的 `feature_kind_updates` 分别记录 `patches` 和 `tracked_entities` 的成功训练更新数。两条路径含各自的空间位置或轨迹 GRU 参数，因此导出、评估及原始视频预处理均拒绝成功更新数为0的输入类型。U0 仍可只使用机器人数据，但若要导出人类 patch 示范，其机器人回放必须包含 patch 窗口；只训练跟踪实体不能视为已训练的 patch 编码器。

## 接入现有机器人训练

已有原始 `demo_view_i` 特征需要提供 `demo_feature_space_id` 和
`demonstration_layouts`。每个视角显式记录 `frames`、`tokens_per_frame`、
`frame_times`、`feature_kind="patches"`、`patch_grid`、
`patch_coordinate_system="normalized_xy_patch_centers"` 和
`token_order="time,height,width,channel"`。
展平顺序固定为时间、高度、宽度、通道；不能从展平序列猜测视频时间轴或空间布局。

```bash
evo-wam encode-demonstrations --artifact /server/runs/video-effects/video_encoder.pt \
  --manifest /server/robot-sample/sample.json --output /server/robot-sample-encoded
evo-wam train --config /server/capacity-configs/robot_768.json --index /server/encoded-index.json \
  --demo-encoder /server/runs/video-effects/video_encoder.pt --checkpoint /server/zero-wam \
  --stage interface --steps 1000 --output /server/runs/interface
```

随后按已有 `reader`、`joint` 路线训练，继续提供相同 `--demo-encoder`。
真实配置的 `dimensions.demo_dim` 必须等于 B 的 `latent_dim`。`raw_features` 与 `video_effect_tokens` 明确区分；后者记录 `encoder_version: 2`、编码器 SHA256、特征空间、token宽度和窗口策略。
机器人训练把该编码器实际训练来源与所有下游划分联查，防止预训练已经读过所谓“未见示范”。
检查点和运行记录保留同一身份，推理时换错编码器会拒绝。

本次空间／时间对应修复采用 v2 编码器 artifact。旧 v1 artifact 不能续训到新结构，旧导出缓存也不能继续使用：须按新布局预训练 B、重新导出 `Z_D`，再训练读取器及所需 WAM 适配参数。新增坐标只描述视频内部的空间位置，不要求人机相机标定，也不证明跨视角不变性。

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
这里的“普通预测”对照仍经过同容量瓶颈以控制参数容量；U1/U2的差值检验整组约束的增量，不能单独归因于控制关系。
若要进一步归因，可在固定噪声和容量设置下，仅改变关系／事件损失权重。
上述U系列JSON中的微型维度是数值验收用，真实特征维度需要服务器配置。U系列的数据／监督对照与K16／K64／K100容量对照是两个独立实验轴。

```bash
evo-wam evaluate-video --artifact /server/runs/video-effects/video_encoder.pt \
  --index /server/heldout-video-index.json --split validation --output /server/results/video-diagnostics.json
```

该诊断在留出窗口上报告原始、置零、跨窗口打乱 z 的多时域特征误差。它能检查预测器是否利用瓶颈，不能替代机器人执行证据。
最终仍需删除 P，固定机器人状态换示范，在未见视角和任务组合上测绑定、控制关系及闭环行为；网络视频损失没有直接更新 WAM 的主张也保持明确。

相关技术依据包括 [LIT](https://arxiv.org/abs/2609.12641)、[LAPA](https://arxiv.org/abs/2410.11758)、
[UniVLA](https://arxiv.org/abs/2505.06111) 和 [GeoLAM](https://arxiv.org/abs/2609.17099)。本实现的连续瓶颈与部分作用监督是可检验的实现选择，研究增量仍需靠迁移与执行实验建立。
