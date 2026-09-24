# G/π 阶段 B：目的分组、冻结探针与离线评测

本页接续 [阶段 A 的数据和部署契约](g_pi.md)。只扩展 G 的意图学习与离线评测；π 仍读语言，`p_drop=0`，不读示范。没有增加 done 头、人机阶段对齐、未来视频损失或在线执行。

## B1：两步 G 与两条示范路径

第一步以独立查询栈读取完整示范的冻结特征，得到有序 `u [B,M_u,D]`；每层只在这些查询内部做 self-attention，只向示范做 cross-attention。第二步以目标查询读取 u、机器人特征及可选当前 state，输出原来的 `(z,p)`。目标查询不能反向成为意图查询的记忆。

`demo_route` 有两种设置：

| 设置 | 冻结 DiT 内的信息流 | 适用范围 |
| --- | --- | --- |
| `one_way`（保留的默认值） | 示范只读示范；机器人读完整示范及因果机器人历史 | 保留原生人机视觉交互；u 是目标头的额外输入，不能宣称是唯一示范通路 |
| `via_u_only` | 示范和机器人分别编码，目标头通过 u 获得示范信息 | 连接式意图头的通路消融，排除 demo→robot→goal 旁路；可能损失细粒度跨场景匹配 |

默认不根据 fixture 结果自动改变。两种路线的选择应结合真实留出数据、旁路诊断和任务表现再决定。默认 `one_way` 下，机器人特征依然含示范信息，这是有意保留的设计。

目标记忆对 u 加可学习槽位位置嵌入，否则 cross-attention 本身会对记忆排列不变。对比相似度逐槽位归一化后按固定顺序展开，保留位置；不先平均 u。槽位顺序敏感只是一项结构保证，不等于已经学到某个具体物体角色或阶段。

提供以下配置；所有相机尺寸仍是阶段 A 的 tiny 示例，实际训练必须先审计：

| 配置 | 数据 | 消融 |
| --- | --- | --- |
| [g_translator.json](../configs/se3/g_translator.json) | v1 | connected，u 在目标路径中，单向 native 交互 |
| [g_translator_v2.json](../configs/se3/g_translator_v2.json) | v2 | 同上，跨物体目的组 |
| [g_translator_regression.json](../configs/se3/g_translator_regression.json) | v1 | 只有目标回归，直接读取示范特征 |
| [g_translator_independent.json](../configs/se3/g_translator_independent.json) | v1 | 独立对比头，u 不进入目标路径 |
| [g_translator_via_u_only.json](../configs/se3/g_translator_via_u_only.json) | v1 | connected，但关闭 native 的 demo→robot 通路 |

默认 `M_u=4`，意图栈深度与目标栈相同，可单独配置。训练 `contrastive_weight` 默认配置为 1、温度 .1，均可调。非零对比权重必须提供目的分组表，不会静默退化成纯回归。回归按本批有机器人配对的样本取均值；纯人手示范仅进入对比项。整批都是纯人手时只有对比梯度。

## 离线目的分组表

表为 `format_version:1`、`kind:g_pi_intent_groups`，顶层包含：

- `data_version`：v1 或 v2，必须与训练配置一致。
- `feature_space_id`、`latent_normalization`：与已有 Wan 缓存一致。
- `grouping_evidence`：人/AI 审核的研究依据；不是模型输入。
- `entries`：完整示范记录；可选 `source_aliases` 与 `uncertain_pairs`。

每条记录包含 `demo_id/purpose_group/split/source_id/source_group/person_id/scene_id/view_id/object_ids/arrays/complete_demo`；`complete_demo` 必须为 true。`purpose_group` 是离线不透明 ID，不要求等于关系谓词。NPZ 复用已有 `latent [C,T,H,W]` 和 `frame_times [T]` 契约，不存语言或目的嵌入。

可选 `paired_task` 指向成功机器人 task，必须与表中的示范路径、来源身份完全一致。无配对的视频省略该字段；不会根据同组关系虚构机器人目标。

v1 另有 `object_family`，同目的组的物体属于经审核的相近类别。v2 另有 `operation` 和 `role_candidates`，训练记录每个角色的候选数必须为 1；正样本包含不同物体。操作/角色字段只用于离线审计和采样。

每批至少两个目的组、每组至少两个独立来源；优先跨人物、场景和视角，优先选择同物体而目的不同的难负例，有条件时加入同场景不同目的。`relation_signature/role_signature/order_signature` 是可选难负例标签，不作为目的词表或模型输入。`uncertain_pairs` 中的样本对从正例、负例及对比分母中排除。

来源别名、内容完全相同的 latent 视频均合并为同一来源分量；重复 HumanGen 视频不能冒充独立正例。分组表与机器人索引还会联合检查训练/验证/测试泄漏。完全相同的内容检测无法识别重新裁切、重新编码的近重复，仍须提供可靠的来源元数据。

```bash
.venv/bin/python -m evo_wam.cli train-goal-interface \
  --config configs/se3/g_translator.json --index /data/g-index.json \
  --intent-groups /data/purpose-v1.json --stage g --tiny-native \
  --steps 2 --device cuda --output /tmp/evo-g-b
```

分组表、示范和配对机器人文件参与训练指纹；目的采样有独立 RNG，保存并精确续训。G 不加载 task 的语言文件，允许省略其 language 字段；π 的训练语言契约保持不变。G 的 checkpoint/export 为 v4，明确拒绝旧 G v3 权重静默续训；π 工件保持 v3，数据及观测仍为 v3，E 身份没有因为新增意图头而改变。

## B2：冻结完整示范特征探针

探针只执行示范一路冻结 DiT，用固定空提示词。按时间、行、列做固定 3D 池化并保持槽位顺序，训练一个线性读出。标准化只拟合训练集；人物、场景和来源分量跨划分必须不相交，多个视角按来源加权。输出测试准确率、按目的平均的准确率、均匀随机与训练多数类基线、未见目的列表。

探针 manifest：`format_version:1`、`kind:g_pi_intent_probe`，包括 `purpose_table`、`layer`、`pool_grid:[4,2,2]`，可配置 `steps/learning_rate/seed`。可选 `feature_cache` 用于读取已有特征，或 `save_feature_cache` 保存本次提取；缓存保留基座、空提示词、层、池化和输入文件身份。

```bash
.venv/bin/python -m evo_wam.cli probe-g-pi-intent \
  --manifest /data/probe.json --artifact /tmp/evo-g-b/goal_interface.pt \
  --device cuda --output /tmp/intent-probe.json
```

目的分类是离线探针专用的监督，不是 G 的输出头。这个线性探针失败不能单独证明冻结基座没有信息，成功也不能证明 G 使用了该信息。

## B3：基于测量的离线评测

```bash
.venv/bin/python -m evo_wam.cli evaluate-g-pi \
  --manifest /data/evaluation.json --device cuda --output /tmp/g-pi-evaluation.json
```

工具比较 G、当前状态基线及同场景训练集众数目标基线，报告目标变化误差、有效区域、各机器人子目标、示范交换、意图×布局四元组，以及按场景/任务分组的统计。有效区域同时要求靠近本意图真值且远离其他意图真值；缺少独立对象证据时不会报告物体任务成功。

评测 manifest 为 `format_version:1`、`kind:g_pi_evaluation`、`split:validation|test`。必需字段还包括 `g_policy`（相对的导出目录）、`encoder_identity/registry`（与 G 工件完全一致）、`thresholds`（阶段 A 标定得到或显式给出的四项距离）、`training_goals/cases/pairs/quadruples`；可选 `pi_policy`、`seed`。文件引用均相对于 manifest 目录，不能越出该目录。

`training_goals` 每项含 `scene/task/source_id/subgoal/split/goal`，split 必须为 train，goal 指向带 E 身份的目标 JSON。众数按 scene+subgoal 的独立来源支持数选择代表目标，不读取当前目的标签。`cases` 每项含 `id/scene/task/layout/intent/source_id/subgoal/observation/truth/current`；observation 是阶段 A 的 G 观测 JSON，truth/current 是目标 JSON。可选 `language`、`wrong_goal`、`action_replay:{arrays,subgoal_time}` 和独立 `object_evidence`。回放 NPZ 只含 actions/actions_mask，必须遵守 `[1,A,F,N,1]` 及切换之后屏蔽的约定。

`pairs` 为 `{kind,ids:[a,b]}`，kind 为 `demo_swap`、`performer_viewpoint` 或 `grasp_speed`；后两种必须保持目的和真值不变，所有配对必须固定机器人观测。`quadruples` 为四个 case ID 的列表，要求恰好两个意图×两个布局，换布局时示范不变。任何分组标签只送给指标代码；G 回调仅收到示范、机器人历史和 state，π 回调不包含示范或离线标签。

交换示范必须两边都更接近各自真值才计成功。安慰剂按执行者/视角变化和抓法/速度变化分别统计。输出 JSON 与同名 SVG，包含真值距离分辨率曲线及示范前缀 10%–100% 曲线；前缀按 latent 帧数比例向上取整，确实切短输入并记录实际帧数，短示范可有重复点。这不是原始视频时间的精确百分位采样。

命令中的旁路诊断分别置换 u 的槽位，以及保留 u 而重新计算不读取示范的机器人特征。Python 的 `GTranslator.diagnose_intent(..., replacement_u=...)` 还支持显式提供另一示范的 u；命令暂不提供该字段。它报告预测变化与真值误差，不能把扰动敏感性单独解释为语义理解或某条通路的贡献比例。只有 connected 模式适用 u 旁路诊断，其他两种消融会记录不适用，照常完成其他指标。

可选 π 离线回放比较真值目标与 G 目标的动作误差；语言默认空提示词，显式指令需记录。另做「正确/空指令 × 正确/错误目标」四格诊断，统一采样随机性。回放只比较已记录状态上的动作，不模拟执行后新状态；动作误差不能替代闭环任务成功率。

联合评测先加载 π，再将 G 的解码器加载到同一 native/E 上；校验基座引用、实时校验和、空提示词和 E 身份，避免重复构建冻结视频基座。G 的权重加载不会覆盖 π 的动作参数。

阶段 C 已加入独立前缀缓存、目标缓存训练加速、π FSDP 和 HumanGen 转换/预检，见 [阶段 C 使用说明](g_pi_scaling.md)。真实多卡与训练数据就绪条件仍需分别验证。
