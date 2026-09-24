# G（做什么）与 π（怎么做）

第一版是两条独立路线：`g_translator` 从完整人手示范和当前任务的机器人历史预测目标，`pi_goal` 只从机器人历史、state、语言和目标生成动作。没有阶段 1、done 头、未来视频生成、主干训练或 G→π 联合训练。旧的 `latent`、`direct_features`、`observed_dual` 和 H0–H3 不变。

目标为 `{"z": [1,K_z,d_z], "goal_poses": [1,E,4,4], "goal_gripper": [1,E]}`。位姿沿用 robot-base→tool、米和声明的 `end_effectors` 顺序；夹爪 closed=0、open=1。E 单独编码一帧，取固定层的空间 token，经固定一维 adaptive average pooling 得到默认 8 个 token，再逐 token L2 归一化。E 是独立冻结 Wan 快照；目标缓存、G/π checkpoint 都记录层、clean timestep=0、池化、归一化、维度、权重哈希和空提示词 embedding 身份，身份不一致拒绝加载。

冻结视觉分支使用 Zero-WAM **空提示词 embedding**，不使用全零文本代替。训练配置可给 `empty_text_emb_path` 或 `empty_emb_path`；未指定时使用固定上游 `wan_va/assets/empty_text_emb.pt`。两个路径若同时给出必须指向同一文件。checkpoint 保存 embedding 本身、来源和文件/张量哈希，独立导出不依赖原路径；tiny 测试明确使用 fixture。任务语言和 state 不进入 Wan。

G 的 mask 是示范双向只看示范，机器人看完整示范并按 chunk 因果看机器人历史。示范沿用原生 `icl_rope_h` 空间偏移，机器人 RoPE 从任务开始重新计数。G 的示范逐层 K/V 可按任务缓存。唯一可训练部分是 goal decoder：`K_z+1` 个查询，经 self-attention、cross-attention、FFN，再输出 z 和复用的 pose/gripper decoder。

π 复用逐层 `_RecurrentGroup`，语义内存为语言、state、z 投影、`encode_goal(p)`，视觉内存仅为该层机器人特征。动作专家每层读语言、state、LIT tokens，继续使用独立 action K/V。训练参数包括 LIT queries/groups、state encoder、pose goal encoder、condition adapter、z projection，以及 `action_named_parameters` 选择的原生动作分支（动作嵌入、动作文本/时间条件、各层动作 attention/FFN/modulation、动作输出）；MCP 和视频分支冻结。π 的视觉分支复用 E 的冻结快照，动作分支为独立对象。

## 数据契约

新 manifest 为 `format_version: 1, kind: "g_pi_task"`，index 为 `kind: "g_pi_index"`、`format_version: 1`，`samples` 仍为 `{"manifest": "task.json", "split": "train"}` 列表，支持现有 `source_aliases` / `bridge_sources` 跨划分审计。

任务 manifest 复用 [SE(3) 契约](se3_interface.md) 的 robot_source、action_space、state_space_id、coordinate_frame、pose_units、pose_representation、tool_frames、end_effectors、gripper_space、language。新增/固定字段：

- `sample_id`、`arrays`；`feature_space_id` 和 `latent_normalization: "(posterior_mode-mean)/std"`。
- `goal_source: "measured_endpoint"`、`task_start_time: 0`、`success: true`、正数 `control_dt`。
- `action_frames` 为原有 chunk 长度 F；`actions_per_frame` 为 N，动作窗口 H=F·N 个控制步。
- `event_rules: {"signal_source":"measured", "close_threshold":0.25, "open_threshold":0.75, "debounce_steps":2}`；必须与训练配置一致。
- G 需要 `demonstration: {"source_id":..., "source_group":..., "domain":"human", "arrays":"demo.npz"}` 和 `compatibility: {"kind":"audited_semantic_task", "evidence":...}`。demo NPZ 仅含 `latent [C,Td,Hd,Wd]`、`frame_times [Td]`，无需机器人时间对齐。π 不打开示范数组，也不向模型提供示范字段。

机器人 NPZ 内容如下，均不含人手动作或物体位姿：

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `latent` | `[C,T,H,W]` | 干净机器人画面 latent |
| `frame_times` | `[T]` | 从 0 开始、间隔 control_dt |
| `states` | `[T,S]` | 测量状态 |
| `poses` | `[T,E,4,4]` | 测量末端 SE(3) |
| `gripper` | `[T,E]` | 归一化测量夹爪 |
| `actions`、`actions_mask` | `[A,T]` | 动作和 Boolean 有效性 |

第一版要求机器人各数组按**每个控制步**同步提供；不会替数据做插值或轨迹重定向。实际相机/VAE 帧率不同的数据需要先明确机器人内部采样契约，再接入；人手与机器人始终仅任务配对。

事件采用迟滞和连续消抖，时间为确认步，绝不回溯到开始变化的时刻；初始稳定状态建立不算事件，同步多夹爪事件按末端顺序合并。成功录像最后一帧总是 terminal。随机取 t<terminal，以严格晚于 t 的首个事件为 t′，否则取 terminal。loader 返回 `events`、`terminal_time`、`subgoal_time`，物理复制历史 `[0,t]` 和单独的目标帧，动作输出 `[1,A,F,N,1]`；所有时间 ≥t′ 的动作 mask=False。旧路线的固定 `goal_time-current_time` 约束保持不变。

## 训练、续训与导出

[G 配置](../configs/se3/g_translator.json)和 [π 配置](../configs/se3/pi_goal.json)是待审计的维度示例；真实训练需把 `synthetic_dimensions_only` 设为 false，核对状态宽度、层号、分组和数据。默认目标噪声关闭；可分别配置 z、平移（米）、旋转（弧度）、夹爪噪声。

```bash
.venv/bin/python -m evo_wam.cli train-goal-interface \
  --config configs/se3/g_translator.json --index /data/g-index.json \
  --stage g --tiny-native --steps 2 --device cuda --output /tmp/evo-g
.venv/bin/python -m evo_wam.cli train-goal-interface \
  --config configs/se3/pi_goal.json --index /data/pi-index.json \
  --stage pi --tiny-native --steps 2 --device cuda --output /tmp/evo-pi

# 真实训练用 --checkpoint /models/zero-wam 替换 --tiny-native。
# 续训保留相同配置/index/seed，并添加：
# --resume /tmp/evo-g/goal_interface.pt

.venv/bin/python -m evo_wam.cli export-goal-policy \
  --artifact /tmp/evo-g/goal_interface.pt --output /tmp/evo-g-policy
.venv/bin/python -m evo_wam.cli export-goal-policy \
  --artifact /tmp/evo-pi/goal_interface.pt --output /tmp/evo-pi-policy
```

两套训练各自保存优化器、随机状态、随机 t 采样状态、数据指纹和完整模型，禁止跨路线 `--initialize`。每次训练核对冻结 E/Video DiT 前后校验和。E 权重始终以 FP32 保存，即使动作/decoder 以 BF16 导出；CUDA native attention 计算精度也在 E 身份中声明。

## 离线推理与控制器

`g_pi_observation` version 1 的公共字段为上述机器人动作/坐标约定、`feature_space_id`、`latent_normalization`、`current_time`、`control_dt`、`actions_per_frame`、`arrays`。不含任何训练标签、terminal 或未来序列。观测 NPZ 必须恰好包含 `state [S]`、`history_latent [C,T,H,W]`、`history_times [T]`，历史从任务 0 开始并结束于 current_time。

G 观测额外含 `demonstration` 人手视频记录；π 观测额外含 `language` 文本缓存路径和 `goal` 目标 JSON 路径，禁止 demonstration 字段。G 输出 `.npz` 的同时生成 `.json` 目标侧文件，包含 E 身份、数组哈希和位姿/夹爪坐标约定，π 可直接读取。

```bash
.venv/bin/python -m evo_wam.cli predict-goal-policy \
  --policy /tmp/evo-g-policy --observation /data/g-observation.json \
  --device cuda --output /data/goal.npz
# pi-observation.json 的 goal 指向上一步 goal.json
.venv/bin/python -m evo_wam.cli predict-goal-policy \
  --policy /tmp/evo-pi-policy --observation /data/pi-observation.json \
  --device cuda --seed 0 --output /data/actions.npz
```

Python 接口为 `GTranslator.predict(demo, robot_frames_t0_to_t, state)` 与 `PiGoalPolicy.predict(robot_frames_t0_to_t, state, language, goal)`。`GTranslator.cache_demo` 可显式预缓存；新示范自动重新编码。π 可提供 `frame_times`，部署 loader 使用真实任务局部时间。

`g_pi_controller.controller_step` 是无设备命令的纯逻辑状态转换：每个 chunk 调一次 π，逐控制步检查同一事件检测器和目标误差；事件或目标达成时丢弃剩余动作并调用 G。新目标与当前 E/位姿/夹爪均在阈值内才停止；换示范重置任务起点、历史、事件状态和动作。阈值需在机器人验证集上确定。第一版不接机器人或仿真闭环，fixture 测试不能证明跨人机泛化。

```bash
taskset -c 0 .venv/bin/python -m unittest discover -s tests -q
```

native 验证使用至少两个 tiny Wan block；无 CUDA 时显式 skip。CPU 验证数据、事件、mask 规则、接口、参数归属、缓存身份、checkpoint 契约及控制器逻辑。
