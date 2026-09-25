# G（做什么）与 π（怎么做）：阶段 A–D

阶段 A 已实现 A2–A7；**A1 已取消**。G 不读任务文字，视觉 DiT 始终使用原生空提示词。π 保留 `main@88df563` 的语言指令通路：LIT 读 `[L,s,g]`，动作专家读 `[L,s,Z_l]`。A7 新增可关闭的语言置空消融与无语言推理，默认 `p_drop=0`；默认语言选择与传递路径不变。

阶段 B 已加入目的分组对比学习、冻结探针和离线评测，见 [阶段 B 使用说明](g_pi_intent.md)。G 保留单向示范→机器人交互作为默认，另提供 `via_u_only` 消融；π 的阶段 A 语言行为不变。

阶段 C 的独立前缀缓存、E 离线目标缓存、π FSDP 与 HumanGen 转换/预检见 [阶段 C 使用说明](g_pi_scaling.md)。

示范只进入 G；π 的动作学习只使用成功机器人录像。不做人机动作、阶段或帧对齐，没有 done 头、未来视频损失或联合训练。关系名、物体状态、关系序号和标定分组均为离线标签，不能进入模型。旧 `latent`、`direct_features`、`observed_dual`、H0–H3 路线保留原契约。

## D：子目标输入与块终点监督

π 现在分成 `pi_prior` 和 `pi` 两个训练阶段。**g=(z,p) 与 q_t 不是同一个角色**：g 是下一子目标 t′ 的输入条件，q_t 是当前实际监督动作块执行结束后的记录状态，仅用于训练。D 不修改 G 的阶段和输入。

q 的控制索引是 `last_nonzero_action_mask_step + 1`，在有效动作通道和 t′ 截断完成之后计算，不从动作向量反推位姿。若块在 t′ 前结束，q 是中间状态；若最后受监督动作是 t′ 前一步，q 就是 t′ 状态。数据样本额外返回 `block_end_poses/gripper/time/index/valid`、`reaches_subgoal`，无需改现有 task v3 数组。

选择的越界策略是 **屏蔽 L_q**，保留动作监督，并令 q 的位姿/夹爪为 None；prior 因需要 q 作条件会明确拒绝这种样本。现有合法 v3 事件/terminal 掩码通常已排除越界，保护分支仍独立测试。动作 mask 的空洞和尾部无效步会改变真实块终点，不能一律用 t+H 代替。

| 阶段 | 动作条件 | 训练目标 | 更新范围 |
| --- | --- | --- | --- |
| `pi_prior` | `[L,s,E_eta(q)]`，每层同一组 token | 原掩码 flow action loss | 完整动作专家、独立 endpoint_encoder、共享 state/condition 投影及动作侧语言投影 |
| `pi` | `[L,s,Z_l]`；Z_l 顺序读取 `[L,s,noisy_g]` 与机器人特征 | `L_action + lambda * goal_pose_loss(q_hat,q)` | 完整动作专家、LIT、子目标编码/投影、共享投影、端点解码器 |

E_eta 是与语义子目标 `goal_encoder` **参数完全分开的 MLP**，输出与 Z 相同的 `[B,num_tokens,dim]`；默认 100 个。阶段一不读取视觉数组内容、人手示范或 g，不运行 E、视频分支或目标噪声；仍加载共享 native/E 对象以复用动作专家和基座身份，未另做纯动作权重加载器。

阶段二从成功的 prior 工件迁移动作专家与共享 state/condition 投影，并创建新优化器；LIT、语义子目标编码器和 D_omega 新初始化。E_eta 不进入阶段二预测、优化器或 checkpoint，原 prior 工件哈希保留来源。D_omega 复用 `_PoseDecoder`，每末端一个查询，读取最后一层**全部 Z token**，避免单 key 双臂耦合；旧 SE(3)/G 解码方式保持不变。

λ 配置为 `pi_training.pose_weight`，默认 **0.3**。损失复用现有米制平移缩放 MSE、旋转矩阵距离和夹爪 MSE；不是 LIT 的分位数归一化目标损失。日志给出 `lit_pose_before/reaches` 及各自 count，还有端点位置误差（米）、角度误差（度）和夹爪误差。FSDP 对有效 q 数量归一，before/reaches 分别按对应计数归一。

```json
"pi_training": {
  "pose_weight": 0.3,
  "goal_source": "hindsight",
  "ablations": {"no_stage1": false, "no_lit_pose": false, "exact_goal": false}
}
```

三个开关默认全关。`no_stage1` 允许明确跳过 prior；`no_lit_pose` 将有效 λ 置 0；`exact_goal` 跳过输入噪声。没有 prior 初始化且未声明 no_stage1，或阶段二噪声全零却未声明 exact_goal，均报错。对应示例配置为 `pi_goal_no_stage1.json`、`pi_goal_no_lit_pose.json`、`pi_goal_exact_goal.json`。`goal_source` 预留交叉拟合来源入口，当前只接受 hindsight；其他值明确报未实现。

## D4：带噪 g，干净 q

`goal_noise` 保留四个尺度：z_std、translation_std（米）、rotation_std（弧度）、gripper_std；增加 translation_max_m 和 rotation_max_deg（度）。z 先逐 token 归一化再加高斯噪声，沿用原有行为，不再二次归一化；夹爪加噪后限制在 `[0,1]`。平移和旋转轴角采用真正的球内截断高斯，不把大样本直接裁到边界；采用独立可恢复的 CPU goal RNG。

正平移噪声必须同时声明 `candidate_separation_m` 与半径上限，且严格满足 `translation_max_m < candidate_separation_m / 2`；缺失或不满足就报错。candidate_separation_m 应为数据集中相关候选的经审核最小间距，不从末端轨迹猜测。旋转标准差为正时也必须指定有效角度上限。

示例配置的尺度为 z=.03、平移=.005 m、旋转=.03 rad、夹爪=.02，上限 .02 m/10°，候选间距 .1 m。**这些只是 synthetic_dimensions_only 配置的手工示例，不是 G 的实测误差，也不适用于未经核查的真实数据。** 干净 q 永远不加噪、不进入阶段二网络。

```bash
.venv/bin/python -m etude.cli calibrate-g-pi-noise \
  --manifest /data/g-validation-residuals.json --output /tmp/pi-noise.json
```

标定 manifest 为 `g_pi_noise_validation` v1、`split:validation`，含 encoder_identity、registry、candidate_separation_m 和 records；每条 record 为 sample_id/source_group/prediction/reference，后两项指向已带 E/坐标身份的目标 JSON。至少两个独立来源组，可选 scale_quantile（默认 .95）。工具读取位置米、角度度、夹爪和 z 距离；各来源组等权，分位数除以维度平方根作为每坐标经验尺度，最大观测残差作为硬上限。任何位置残差达到候选间距的一半都会拒绝标定，不静默裁小错误物体预测。输出 `config` 字段可合入 π 配置，保留单位、原始残差与文件哈希；它不是高斯拟合或置信度保证。

## D：两阶段命令、工件与换 g 诊断

```bash
.venv/bin/python -m etude.cli train-goal-interface \
  --config configs/se3/pi_prior.json --index /data/pi-index.json \
  --stage pi_prior --tiny-native --steps 2 --device cuda --output /tmp/evo-pi-prior
.venv/bin/python -m etude.cli train-goal-interface \
  --config configs/se3/pi_goal.json --index /data/pi-index.json \
  --stage pi --initialize /tmp/evo-pi-prior/goal_interface.pt \
  --tiny-native --steps 2 --device cuda --output /tmp/evo-pi
```

两阶段各自用 `--resume` 精确续训；初始化和续训互斥。prior 当前只支持单卡，启用 FSDP 明确报错。阶段二可使用 `pi_goal_fsdp.json` 和原 torchrun 入口，同样需 `--initialize`；D_omega 与 LIT 位于同一 FSDP 前向范围，包含在轻量保存与导出中。prior 工件只能续训/初始化，不能导出成需要输入 q 的部署策略。

π 工件升为 v4，记录 stage、有效 λ、噪声配置、candidate_separation_m、消融及 prior 来源哈希；拒绝旧 π v3 静默加载。每阶段只存参与该阶段学习的接口/动作权重，冻结视频和 E 不入文件。G 仍用原 v4，数据/配置/观测仍为 v3。

普通 `pi.predict(...)` 不执行 E_eta 或 D_omega。`pi.diagnose_endpoint(history,state,language,goal,frame_times=...)` 可仅解码 q_hat，不采动作、不消耗动作 RNG、不接收 q 标签；也可用 `predict(..., return_endpoint=True)` 请求附带诊断。

现有 `evaluate-g-pi` 在 case 提供 wrong_goal 且加载 π 时增加换 g 读出：固定画面、状态、语言和 seed，报告每个末端 `q_hat(new)-q_hat(original)` 的位移米数、与目标 `p(new)-p(original)` 的方向余弦，以及分组汇总。零目标位移或零预测位移时余弦为 None，并记录有效性标志；不制造 NaN 或完美得分。不需要动作标签或真实机器人。λ=0 时标记 untrained_readout，不能把未监督头的位移解释成经过标定的块终点。

与 [LIT 原文](https://arxiv.org/html/2609.12641v1)仍有差异：这里共享且冻结视频/E，π 额外接收子目标 g，动作专家直接读取语言和 state；阶段一从 Zero-WAM 动作权重初始化，而不是从头训练动作专家。L_q 的量纲也不同。输入噪声和辅助重建不能单独保证消除视觉捷径，效果仍需真实留出数据与诊断验证。

## 两套时间网格与 v3 数据

G/π task、observation、实验配置为 **version/schema 3**；G 和阶段 D 的 π 工件均为 **version 4**，分别记录各自架构。产品路线称 v2，与磁盘格式编号不同。旧工件不静默续训，旧的一维 E 缓存也因身份不一致被拒绝。本轮不提供旧模型权重初始化转换器。`g_pi_index` 仍为 version 1，保留 `samples`、source aliases 和 bridge split 审计。

机器人 NPZ 的基础字段保持如下：

| 数组 | 形状 | 约定 |
| --- | --- | --- |
| `control_times` | `[T_c]` | 从任务 0 开始、间隔 control_dt |
| `states` | `[T_c,S]` | 测量机器人状态 |
| `poses` | `[T_c,E,4,4]` | robot-base→tool、米、固定末端顺序 |
| `gripper` | `[T_c,E]` | 测量夹爪，closed=0/open=1 |
| `actions`、`actions_mask` | 各 `[A,T_c]` | 未填充的规范化控制命令及 Boolean 有效性 |
| `latent` | `[C,T_l,H,W]` | 原生因果 VAE 视频 latent |
| `latent_available_times` | `[T_l]` | 各 latent 所含最后原始帧的拍摄时间 |
| `subgoal_times` | `[K]` | 重算后的边界及 terminal 的有序并集 |
| `subgoal_latents` | `[K,C,1,H,W]` | 各边界原始图像分别经 VAE 首帧路径编码 |

manifest 保留 `robot_source`、action/state/坐标/工具/夹爪约定、`feature_space_id`，并要求 `goal_source: measured_endpoint`、`task_start_time: 0`、`success: true`、`latent_normalization: (posterior_mode-mean)/std`。π 训练保留 `language` 缓存及其身份；G 不加载语言文件，允许省略此字段。G 额外要求完整 human demonstration 和 `audited_semantic_task` 配对；π 不加载人手视频。

时间元数据为 `frame_stride=r`、`temporal_down_rate=4`、`actions_per_frame=N=4*r`、`action_frames=F`、`alignment=zerowam_causal_first_then_four`、`subgoal_encoding=wan_vae_single_frame`。VAE 输入原始控制帧为 `0,r,2r,…`，首帧单独编码，随后每四个采样帧产生一枚 latent；可用控制步为 `0,N,2N,…`。`T_l=floor((T_c-1)/N)+1`，历史时间必须等于 `control_times[::N]`。

这对应 [lerobot_latent_dataset.py:581–600](../third_party/Zero-WAM/wan_va/dataset/lerobot_latent_dataset.py#L581) 的 `history_size=frame_stride*4` 和 `(f n)` 布局。上游 [robotwin_action.py:93–106](../third_party/Zero-WAM/wan_va/dataset/robotwin_action.py#L93) 会先放 N 行零历史，本契约不把这些占位动作当未来监督。

只在 latent 可用且早于 terminal 的控制步 i 采样；动作列 i 表示接下来 `(t_i,t_{i+1}]` 的命令，窗口为 `actions[:,i:i+F*N]`，重排 `[1,A,F,N,1]`。下一边界 i′ 必须严格晚于 i，所有 k≥i′ 的动作屏蔽。p 精确取该控制步状态，z 只取对应的独立单帧 latent。目标时间严格匹配重算结果；网格容差小于控制周期，不能误选邻步。

`icl_preprocess.encode_g_pi_frames` 复用已有 `vision.encode_rgb`。非 gripper 来源可传 `subgoal_metadata` 与 `offline_arrays`（含 poses 及对应证据），由同一解析器确定边界后单帧编码。历史序列丢弃不足四帧的尾组，但目标原始图像仍保留 terminal。不要从整段时序 latent 中替代抽取单帧目标。

## A2：逐视角二维目标网格

`goal_encoder` 配置示例：

```json
{
  "layer": 1,
  "grid_size": [4, 4],
  "camera_layout": [
    {"name": "head", "token_width": 20},
    {"name": "left_wrist", "token_width": 20},
    {"name": "right_wrist", "token_width": 20}
  ]
}
```

`camera_layout` 必须显式提供；`token_width` 是 **DiT patch 之后的列数**，不是原始像素或 VAE latent 列数。示例的 20 只是布局示例，必须按实际相机缓存与 native patch size 审计。各宽度之和必须等于机器人画布的 patch 宽度。不能从张量内容自动猜测视角边界。

E 单独编码目标帧，在固定层将特征恢复成二维空间网格，然后沿宽度按布局切片，每个相机分别做 adaptive average pooling，最后按「相机顺序→相机内行优先」拼接并逐 token L2 归一化。默认每相机 4×4，因此两相机 K_z=32，三相机 K_z=48。池化区域不跨相机边界；DiT 原有的跨画布注意力没有禁止。

E 身份包含 grid_size、camera_layout、num_views、token_order=`camera_then_row_major`、K_z、d_z、层、clean timestep=0、归一化、空提示词来源、冻结视频哈希和精度。换相机顺序、列宽、网格，或者加载旧一维缓存，都会被身份校验拒绝。π 的 z 投影增加 `[1,K_z,native_dim]` 可学习位置嵌入。

[G 配置](../configs/se3/g_translator.json)和 [π 配置](../configs/se3/pi_goal.json)中的 `front/wrist` 各 1 列仅用于两列 tiny 画布，仍标记 `synthetic_dimensions_only: true`，不是 RoboTwin/SO-101 的实际分辨率配置。

## A6：每个末端独立位姿解码

G 使用 K_z 个视觉目标查询和 E 个位姿查询，共 K_z+E 个。第 e 个末端只从第 e 个位姿 token 解码；复用 `_PoseDecoder` 的 `per_effector` 分支，避免所有手臂读同一个键导致平移常数差、夹爪单调耦合。共享 decoder 权重允许跨样本泛化，不限制两只手必须同向变化。单末端行为与旧分支严格等价，旧路线默认解码分支不变。

## A3：离线子目标来源

目标状态来源仍是机器人；边界来源由 `subgoal_source` 选择，省略时为 `gripper`。每条样本返回审计 metadata，但不把离线证据增加为模型 Tensor 输入。

| 来源 | 边界重算与证据 |
| --- | --- |
| `gripper` | 测量夹爪迟滞/消抖，默认阈值 .25/.75、持续 2 步；自动记录规则、确认索引、证据和版本 |
| `sim_relation` | 保存的 `object_positions [Tc,O,3]`、`object_extents [O,3]`，开合任务另有 `object_joint_fractions [Tc,O]`；按固定视觉几何谓词重算 |
| `pedal` | 原始 `pedal_times [R]`、scale/offset 时钟映射、annotation_version、confirmation_delay；映射后向后取首个控制步，不倒扣延迟、不回填 |
| `candidate_match` | 重算测量夹爪候选与末端速度候选，再按任务模板单调匹配；强制 `weak_label: true`，拒绝错序、缺段、额外候选和超时段 |

非默认来源需显式 `subgoal_annotation`，包括 `detector_version`、完整 `thresholds`、`stable_steps`、`evidence`、`weak_label`、`control_indices`、`relations` 和 `relation_registry: visual_relations_v1`，并带来源专用字段。解析器另记录 `relation_control_indices`，供机器人子目标匹配与标定使用。控制索引始终包含 terminal；同一时刻只存一张完整目标图像。

默认关系：`in_hand`、`placed`、`opened`、`closed`，不含 pressed。实例字段为 `predicate/object_role/effector/occurrence`；placed 另有 `relation`（on/in）和 `target_role`。occurrence 从 1 开始，表示相同角色实例在程序中的重复编号。扩展关系必须增加固定实现并升级 registry 版本，不能在模板里塞入任意执行代码。

sim 模式还需 `object_roles` 有序列表和 `geometry: axis_aligned_robot_base_boxes`；extents 是米制半尺寸。in_hand 检查末端距离、夹爪闭合区及相对初始抬升；placed 检查 AABB 几何关系、释放和物体低速；opened/closed 检查可见关节开度。**这些是明确的几何近似，不宣称适配任意形状或代表真实接触/attachment。** 连续成立 k 步后，在当前步确认；重复关系必须经过重置，不能用未来确认后回填。

候选工具不使用力、触觉、摩擦或质量。它检测：测量夹爪开合、闭合后在非零宽度持续停住、末端平移速度低谷。低谷需看到后续上升才在当前步确认，或持续低速达到稳定窗口才确认。opened/closed 的本体候选不能验证真实视觉关系，因此整个匹配结果保持弱标签。严格匹配可能保守丢弃正常录像，需在真实 SO-101 数据上审计保留率。

```bash
.venv/bin/python -m etude.cli generate-g-pi-candidates \
  --manifest /data/candidate-spec.json --output /data/candidate-labels.json
```

spec 为 version 1、kind=`g_pi_candidate_spec`，含本地 evidence NPZ 的 `arrays`、`event_rules`、`relations`、`thresholds`、`stable_steps`、`evidence`、`template_version`。NPZ 至少有 control_times、poses、gripper；thresholds 精确包含 `blocked_width_min/blocked_width_max/width_stability/speed_max/min_duration/max_duration`。输出提供可并入 task 的来源和审计字段，以及用于重新编码目标图像的 subgoal_times。

## 共享基座与 A7 语言消融

E 仅引用 shared native，不注册或复制其权重。CUDA 冻结视频参数为 BF16；可训练 decoder/LIT/动作分支保持 FP32 主权重，使用 BF16 autocast。G 与视频/E 使用固定原生空提示词。π 仍读语言指令，不读人手示范；动作 K/V 保持隔离，MCP 与视频冻结。

`p_drop` **默认 0**，是后续诊断开关。显式设为 .4 时，每个 π 训练样本只抽一次随机数，选择原指令或 native.g_pi_empty_text，随后只投影一次，LIT 与动作专家使用同一个 hidden。不使用全零替代，不跨优化步骤缓存可训练语言投影。独立 CPU language RNG 保存进 checkpoint，精确续训恢复；p_drop=0 不消耗该 RNG，也不扰动原来的 sample/action/goal 随机流。G 不执行这项随机选择。

推理可调用 `pi.predict(history, state, language, goal)`，或省略语言后调用 `pi.predict(history, state, goal=goal)`；后者使用同一个内部空提示词。控制器的 language 同样可省略。训练 task 保留语言缓存引用与身份，观测 manifest 的 language 可选；不给语言时不会读取语言文件。

## A4/A5：切换、停止保护与标定

控制器每控制步检测事件；只有新 latent 可用时传 frame 和绝对 frame_available_time，否则 frame=None。目标达到才增加 completed_subgoals；夹爪事件只中断剩余 chunk 并重新调用一次 G，不直接宣告子目标完成。计数仅用于日志，不作为 G/π 参数。

若刷新后的目标已接近当前状态，暂不发新动作（action=None），也不立即停止。必须等到**拍摄可用时刻严格晚于刷新控制时刻**的新 latent，再调用 G 确认；仍接近才停止，否则继续新目标。刷新前拍摄、刷新后才送达的积压帧不能通过保护。换示范清空历史、计数、等待状态和任务时钟。等待新画面期间不实现任何额外物理保持控制器。

z 停止阈值没有默认值。程序调用需显式 GoalThresholds(z=...)；policy 导出必须提供完整四维显式阈值，或标定 artifact。四维度量为最差 token 的 L2 距离、最差末端平移米、旋转度、夹爪绝对误差，停止是四项同时满足。

标定输入 version 1、kind=`g_pi_validation`、split=`validation`，包含训练 artifact 的 encoder_identity 与 registry，以及：

```json
{
  "recordings": [
    {"task": "robot-a.json", "intent_group": "opaque-intent", "target_cache": "a-z.npz"},
    {"task": "robot-b.json", "intent_group": "opaque-intent", "target_cache": "b-z.npz"}
  ],
  "margin_fraction": 0.5
}
```

同一意图需至少两条独立 robot source_group；重复 source/trajectory 不能冒充独立录像。相同机器人子目标按关系实例或夹爪事件顺序匹配，不做人机对齐。`target_cache` 可省略，此时通过 `--artifact` 重建 E 并逐单帧编码。可选 `oracle_goal_arrivals` 是 `reference/achieved` 两个目标 JSON 路径的列表，用于加入 π 真值目标执行的到达误差。

标定计算同意图、对应子目标的跨录像离散度、相邻目标距离和可选到达误差。各维正样本最大值构成下界，再在可分离范围内取 margin_fraction（默认一半）。每个相邻对只需至少一维超出阈值，符合控制器的 AND 条件；并非要求四个维度都能区分。最终重验所有正样本被接受、所有相邻目标被拒绝；无解则 ValueError，不自动放宽。跨场景离散是保守的验证分布容差，不等于纯传感噪声，也不是统计置信保证。

```bash
.venv/bin/python -m etude.cli calibrate-g-pi-stop \
  --manifest /data/validation.json --artifact /tmp/evo-pi/goal_interface.pt \
  --device cuda --output /data/stop-calibration.json
.venv/bin/python -m etude.cli export-goal-policy \
  --artifact /tmp/evo-pi/goal_interface.pt --calibration /data/stop-calibration.json \
  --output /tmp/evo-pi-policy
# 或用 --stop-thresholds /data/thresholds.json；文件需恰含
# z、position_m、rotation_deg、gripper 四个有限非负值，两种选项互斥。
```

标定 artifact 保存完整距离证据、验证文件哈希和 E/视角/registry 身份；policy 内嵌阈值及标定证据，加载重新核验。已有目标缓存可做 cache-only 标定，不需要载入模型；首次标定从训练 artifact 读取 E，避免依赖尚未导出的 policy。

## 训练、部署及范围

```bash
.venv/bin/python -m etude.cli train-goal-interface \
  --config configs/se3/g_translator.json --index /data/g-index.json \
  --intent-groups /data/purpose-v1.json \
  --stage g --tiny-native --steps 2 --device cuda --output /tmp/evo-g
.venv/bin/python -m etude.cli train-goal-interface \
  --config configs/se3/pi_goal.json --index /data/pi-index.json \
  --stage pi --initialize /tmp/evo-pi-prior/goal_interface.pt \
  --tiny-native --steps 2 --device cuda --output /tmp/evo-pi
# 续训保留同配置/index/seed并添加 --resume 对应 goal_interface.pt。
# 本轮仅验证 tiny；真实训练还需审计配置并提供本地 --checkpoint。
.venv/bin/python -m etude.cli predict-goal-policy \
  --policy /tmp/evo-pi-policy --observation /data/pi-observation.json \
  --device cuda --seed 0 --output /data/actions.npz
```

checkpoint（G v4、π v4）仍只保存各阶段可训练 interface/action 权重、优化器/RNG/数据指纹与原始基座引用；E 和所有冻结参数不复制进文件。conditioning_mode 为 G 的 demo_robot_state_internal_empty、π 的 language_state_goal，π 另有明确 stage 和监督角色，p_drop 独立记录。原始基座与空提示词文件必须保留，--checkpoint 可迁移同哈希基座路径。旧工件因结构或身份不匹配被明确拒绝，不静默恢复。

v3 observation 的公共元数据保留机器人动作/坐标/两网格约定；NPZ 只有 state、history_latent、latent_available_times。当前控制步可以位于两个 latent 之间，但历史只能含完整已可用前缀。G 观测提供 demonstration，π 提供 goal 文件及可选 language；关系、物体状态、阶段索引等字段均被拒绝。

阶段 B 的换示范/错误目标评测默认使用空语言，并记录语言条件；具体工具见 [阶段 B 使用说明](g_pi_intent.md)。缓存、FSDP 与数据转换的可用范围及真实数据缺项见 [阶段 C](g_pi_scaling.md)。

```bash
taskset -c 0 .venv/bin/python -m unittest discover -s tests -q
```

仅使用 fixture、tiny native 和 tiny VAE 验证；没有真实 Wan 权重、SO-101 或 RoboTwin 实际执行质量结论。共享 BF16 基座节省模型存储，但原生完整动作专家仍很大，16GB 的 tiny 测试不构成真实模型显存保证。
