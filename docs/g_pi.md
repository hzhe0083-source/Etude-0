# G（做什么）与 π（怎么做）

`g_translator` 从完整人手示范与机器人任务历史预测 `(z,p)`；`pi_goal` 只读机器人历史、state、语言和目标，生成动作。没有主干训练、未来视频损失、阶段 1、done 头或 G→π 联合训练。旧的 `latent`、`direct_features`、`observed_dual`、H0–H3 保持原接口。

## 两套时间网格：task v2

`g_pi_task` 使用 `format_version: 2`。v1 的每控制步一帧 latent 假设不再接受，必须重新生成缓存。`g_pi_index` 仍为 version 1，保留 `samples: [{"manifest":"task.json","split":"train"}]`、source aliases 和 bridge split 审计。

机器人 NPZ 必须恰好包含以下十个数组：

| 数组 | 形状 | 约定 |
| --- | --- | --- |
| `control_times` | `[T_c]` | 从任务 0 开始，间隔 `control_dt` |
| `states` | `[T_c,S]` | 测量状态 |
| `poses` | `[T_c,E,4,4]` | 控制步上的 robot-base→tool SE(3)，米 |
| `gripper` | `[T_c,E]` | 测量夹爪，closed=0、open=1 |
| `actions`、`actions_mask` | 各 `[A,T_c]` | 未加历史 padding 的规范化控制命令及 Boolean 有效性 |
| `latent` | `[C,T_l,H,W]` | 按原生 VAE 因果顺序编码的机器人历史 |
| `latent_available_times` | `[T_l]` | 每个 latent 所含最后一个原始帧的拍摄时刻 |
| `subgoal_times` | `[K]` | 全部确认事件时刻与 terminal 的有序并集 |
| `subgoal_latents` | `[K,C,1,H,W]` | 各目标时刻原始图像**分别**走 VAE 首帧路径的编码 |

manifest 沿用 [SE(3) 契约](se3_interface.md) 的 robot_source、action_space、state_space_id、坐标/工具/末端顺序、gripper_space、language，并要求：

- `sample_id`、`arrays`、`feature_space_id`、`latent_normalization: "(posterior_mode-mean)/std"`。
- `goal_source: "measured_endpoint"`、`task_start_time: 0`、`success: true`、正数 `control_dt`。
- `frame_stride: r`、`temporal_down_rate: 4`、`actions_per_frame: N=4*r`、`action_frames: F`。
- `alignment: "zerowam_causal_first_then_four"`、`subgoal_encoding: "wan_vae_single_frame"`。
- `event_rules: {"signal_source":"measured", "close_threshold":0.25, "open_threshold":0.75, "debounce_steps":2}`，与训练配置一致。
- G 额外需要 human `demonstration` 视频记录和 `compatibility: {"kind":"audited_semantic_task", "evidence":...}`。demo NPZ 仍为 `latent`、`frame_times`；它与机器人仅任务级配对。π 不打开示范数组。

这里 r 是原始机器人控制/视频帧索引的采样步长。VAE 输入索引为 `0,r,2r,…`：latent 0 单独编码帧 0，latent j≥1 新增四个采样帧 `(4j−3)r,…,4jr`，其因果感受野可包含更早帧；可用控制步为 `j*N`。因此 `T_l=floor((T_c−1)/N)+1`，可用时刻为 `control_times[::N]`。不完整的最后一组只从历史视频编码中舍弃，末尾的事件和 terminal 图像仍须单帧编码。例：r=4 时 N=16，历史可用控制步是 0、16、32……。

该约定对应 Zero-WAM [lerobot_latent_dataset.py:563](../third_party/Zero-WAM/wan_va/dataset/lerobot_latent_dataset.py#L563) 的 temporal_down_rate=4、latent_frame_num，以及 [581–600 行](../third_party/Zero-WAM/wan_va/dataset/lerobot_latent_dataset.py#L581) 的 frame_stride、history_size 和 `(f n) c -> c f n 1`。上游 [robotwin_action.py:93–106](../third_party/Zero-WAM/wan_va/dataset/robotwin_action.py#L93) 会先插 N 行零历史：aligned 第 0 块是占位，第 f≥1 块对应原动作 `[(f−1)N,fN)`。本契约存原始未 padding 动作，绝不把这 N 行占位用作未来监督。

只在某个 latent 刚可用且严格早于 terminal 的控制步 i 采样 t。历史物理复制到 `i/N+1` 个 latent，目标时刻 t′ 是 t 后第一个确认事件，否则为成功录像最后控制步。事件按测量夹爪逐控制步迟滞/消抖，时间取确认步；初始状态建立不算事件。同一时刻多个末端事件保留事件顺序，但只缓存一个完整目标；terminal 若与事件同刻也只缓存一次。`subgoal_times` 必须与重新检测出的控制时刻**完全一致**，不做时间容差或最近邻匹配。

动作列 k 表示在 `(control_times[k],control_times[k+1]]` 执行的命令。t 对应 i 时，未来窗口取 `actions[:,i:i+F*N]`，重排为 `[1,A,F,N,1]`；其顺序等价于上游 `(f n)`。t′ 对应 i′ 时，仅 `i≤k<i′` 的命令参与 flow loss。事件时刻及其后的命令、末尾不足 chunk 的 padding 全部屏蔽。p 精确取 `poses[i′]`、`gripper[i′]`，z 的输入只取匹配的单帧 `subgoal_latents`。

## 单帧目标预处理

现有 `vision.encode_rgb` 支持 `T=1`，复用本地 Wan VAE、posterior mode 和原生归一化。新增 `icl_preprocess.encode_g_pi_frames` 将完整控制网格 RGB 转为上述四个视觉数组；每个目标分别调用一次 `encode_rgb`，不会与其他目标或历史一起编码，也不会从历史视频 latent 中替代取值。真实 tiny Wan VAE 测试验证了该路径及尾部 terminal 的保留。

```python
from evo_wam.g_pi_data import EventRules
from evo_wam.icl_preprocess import encode_g_pi_frames

visual_arrays, alignment = encode_g_pi_frames(
    vae, rgb, control_times, measured_gripper,
    frame_stride=4, control_dt=0.04,
    event_rules=EventRules(), size=[256, 256],
)
# rgb 为 uint8 [T_c,H,W,3]；保留完整原始帧，不能先用 read_video 截掉尾部。
# 将 visual_arrays 与六个控制数组合并写入 task.npz，alignment 写入 task.json。
# 继续记录 VAE 权重身份、feature_space_id 和源录像 provenance。
```

该 helper 接受已加载的本地冻结 VAE，不下载权重。异步相机/不等间隔控制数据需要显式转换为这套原生网格；工具不会自行插值或伪造事件时刻图像。

## 单一基座、精度和参数归属

E 是共享 native 的固定读取规则，不持有或注册第二份权重。G、π 视觉读取和 E 可引用同一 native；同一模型调用须顺序执行。冻结校验和排除已隔离的动作分支，动作更新不会改变 E 身份或使 demo K/V 缓存失效。E 单独读一帧，固定层、clean timestep=0、空间 adaptive-average-pool 到默认 8 个 token、逐 token L2 归一化；身份包含规则、视频存储/计算精度、空提示词身份和冻结基座哈希。

CUDA 上冻结参数以 BF16 存储；G decoder、π LIT/目标投影和动作分支为 FP32 主权重，forward 使用 BF16 autocast。基座先在 CPU 按原 FP32 权重构建，只转换冻结参数，之后搬到 GPU，避免 GPU 同时容纳整份 FP32 临时基座，也避免把动作主权重先舍入再升回 FP32。视频和 E 始终 no_grad。CPU 构造仍可能需要一份完整 FP32 模型的峰值内存。

视觉分支使用原生空提示词 embedding。配置可指定 `empty_text_emb_path` 或 `empty_emb_path`，默认沿用上游 `wan_va/assets/empty_text_emb.pt`；来源及文件/张量哈希均记录。语言和 state 不进入 Wan。

G 的示范只看示范，机器人读示范并对自己的 latent 历史做 chunk 因果 attention；机器人 RoPE 从任务 0 重计，demo 使用原生 icl_rope_h 偏移。G 只训练 goal decoder。π 复用 `_RecurrentGroup` 的 self→semantic→visual 顺序，语义为 `[L,s,z投影,encode_goal(p)]`，动作条件为 `[L,s,Z_l]`；训练现有 `action_named_parameters` 完整动作分支以及 LIT/目标接口，MCP 与视频冻结。

## 训练、轻量 checkpoint 与部署

[G 配置](../configs/se3/g_translator.json)、[π 配置](../configs/se3/pi_goal.json)继续使用 schema 2。`base_seed` 默认 0，只用于 tiny 随机基座，与训练 `--seed` 解耦。真实训练必须审计配置维度并设 `synthetic_dimensions_only: false`。目标噪声默认关闭。

```bash
.venv/bin/python -m evo_wam.cli train-goal-interface \
  --config configs/se3/g_translator.json --index /data/g-index.json \
  --stage g --tiny-native --steps 2 --device cuda --output /tmp/evo-g
.venv/bin/python -m evo_wam.cli train-goal-interface \
  --config configs/se3/pi_goal.json --index /data/pi-index.json \
  --stage pi --tiny-native --steps 2 --device cuda --output /tmp/evo-pi
# 真实训练：将 --tiny-native 换成 --checkpoint /models/zero-wam。
# 续训：添加 --resume /tmp/evo-g/goal_interface.pt，保持相同配置/index/seed。
.venv/bin/python -m evo_wam.cli export-goal-policy \
  --artifact /tmp/evo-g/goal_interface.pt --output /tmp/evo-g-policy
.venv/bin/python -m evo_wam.cli export-goal-policy \
  --artifact /tmp/evo-pi/goal_interface.pt --output /tmp/evo-pi-policy
```

训练 artifact 和导出 policy 均升为 version 2。只保存 FP32 可训练 `interface` / `action` 权重，不保存任何冻结 native/E 参数或空提示词张量。训练 artifact 另存独立优化器、RNG、采样 cursor 和已消费数据指纹。基座引用含原始 checkpoint 文件标识/哈希、空提示词来源、E 身份与精度。加载先重建并校验基座，再恢复可训练参数；错误基座或混入冻结参数的 state 均拒绝。旧 v1 完整快照不自动迁移。

真实部署必须保留原始基座和空提示词文件；`--checkpoint` 可覆盖基座路径，但内容哈希必须相同。tiny 用固定 base_seed 重建，无需真实权重。导出的 legacy `--dtype bfloat16` 参数仍接受并记录 `requested_dtype`，但 v2 保存/加载的可训练主权重始终 FP32，冻结视频精度取基座引用，不通过导出改变 E 身份。

`g_pi_observation` 同样升为 version 2，公共元数据包含机器人动作/坐标约定、`current_time`、`control_dt`、`actions_per_frame` 和上述四个网格/编码声明。NPZ 仅有 `state [S]`、`history_latent [C,T_l,H,W]`、`latent_available_times [T_l]`；当前控制时刻可以处于两个 latent 可用时刻之间，历史必须是截至该时刻的完整可用前缀。G 额外包含 demonstration；π 额外包含 language 和 goal 文件路径，禁止 demonstration。

```bash
.venv/bin/python -m evo_wam.cli predict-goal-policy \
  --policy /tmp/evo-g-policy --observation /data/g-observation.json \
  --device cuda --output /data/goal.npz
# π 观测的 goal 指向上一步的 goal.json（含 E 身份、坐标约定和数组哈希）。
.venv/bin/python -m evo_wam.cli predict-goal-policy \
  --policy /tmp/evo-pi-policy --observation /data/pi-observation.json \
  --device cuda --seed 0 --output /data/actions.npz
# 真实基座迁移路径时，两条命令都可添加 --checkpoint /models/zero-wam。
```

`controller_step` 每个控制步检测夹爪事件。只有新 latent 可用时传 `frame` 和绝对 `frame_available_time`，否则传 `frame=None`；不复制旧画面充数。首次示范需同时给 t0 的首 latent；换示范重置任务时间、控制时钟、历史和事件状态。事件/达成打断 chunk 后，G 读取最近可用历史；停止检查的 `current_goal.z` 必须由调用方将当前原始图像独立编码后过 E。没有机器人命令或在线仿真。

## 参数量资源估算

以下是十进制 GB，假设**整份 native P=5B（含动作分支）**，不是视频主干另加动作专家。参考上游默认结构的参数比例估计动作 A≈0.440205P；双末端配置的 G decoder I_G≈26.01M、π interface I_π≈86.32M。Adam FP32 双状态占 8T 字节，梯度占 4T，T 为可训练参数数。忽略激活、demo K/V、CUDA workspace、加载临时副本及元数据；原始共享基座文件另行保留。

| 项目 | G 旧→新 | π 旧→新 |
| --- | ---: | ---: |
| 模型权重显存 | 40.10 → 10.10 GB | 40.35 → 14.75 GB |
| 权重＋梯度＋Adam 状态 | 40.42 → 10.42 GB | 67.79 → 42.20 GB |
| 可精确续训的 checkpoint | 40.31 → 0.31 GB | 58.64 → 27.45 GB |
| 导出权重 | 40.10 → 0.10 GB | 40.35 → 9.15 GB |

公式：旧权重为 `8P+4I`；新 G 权重为 `2P+4I_G`，新 π 为 `2(P−A)+4(A+I_π)`；训练常驻再加 `12T`。旧 checkpoint 为 `8P+4I+8T`，新为 `12T`；新导出为 `4T`。

仅在 meta device 实例化上游默认结构并计数，安装动作 K/V 隔离后实际 P≈11.356B、动作约4.999B（还含冻结的未执行 MCP）。按此结构，新 G 权重约22.82GB、新 π 权重/梯度/Adam 常驻约94.08GB；这不是已检查的发布权重。**16GB GPU 本轮只能验证 tiny native，不能据此宣称可训练真实 π。** 若“Wan 5B”特指视频主干，还需额外计入动作专家。

## 验证与范围

```bash
taskset -c 0 .venv/bin/python -m unittest discover -s tests -q
```

测试覆盖两网格、上游动作 padding/布局、单帧 VAE、共享对象与混合存储精度、真实更新后 E/冻结校验不变、无冻结权重的 checkpoint、错基座拒绝、精确续训、分片导出及控制器异步频率。native 使用至少两个 tiny block；没有真实 Wan 权重验证或执行成功率结论。

可选的批量 E 目标缓存命令/训练开关本轮未新增；已有 `save_target_cache` / `load_target_cache` 仍按 E 身份工作。训练按要求只采 latent 边界，控制器可在中间事件步刷新 G；这种中间 state 的泛化仍需真实数据评估。
