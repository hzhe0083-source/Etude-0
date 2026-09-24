# G/π 阶段 C：缓存、FSDP 与 HumanGen 转换

接续 [基础接口](g_pi.md)与[意图训练和评测](g_pi_intent.md)。默认仍是 `one_way`，G 不读任务文字，π 保留语言且 `p_drop=0`。本轮不改变训练目标、不增加 LoRA、done 头或在线执行。

阶段 D 已为 π 增加非视觉 prior 和干净块终点监督，π 工件升级 v4；本页缓存/FSDP机制继续适用，新的初始化和噪声契约见 [阶段 D](g_pi.md#d子目标输入与块终点监督)。

## C1：任务内前缀与 E 目标缓存

连续离线推理可以显式开启：

```python
g.clear_context_cache()
pi.clear_context_cache()
goal = g.predict(demo, history, state, use_prefix_cache=True)
actions = pi.predict(history, state, language, goal, use_prefix_cache=True)
```

`RobotPrefixCache` 持久化已闭合机器人 chunk 的每层特征、旋转后的 K/V；只把闭合前缀之后的后缀重新送入 native。未闭合 chunk 内的旧帧也要重算，因为它们可以读取同 chunk 新增的帧。RoPE 始终使用从本任务 0 开始的原始位置，128 padding 和 mask 使用全局 token 索引。

G、π 各有自己的缓存，不能互换。G 的 `one_way` 机器人 K/V 含示范信息；`via_u_only` 虽然隔离这条通路，仍采用 G 的独立缓存身份。基座版本、设备、dtype、示范、路由、chunk 配置或既有历史内容变化都会拒绝复用。t 之后的帧仍先物理截断，不进入哈希或 native。

**每次新任务都要清 G 和 π 缓存，包括再次使用同一段示范。** 不同示范会自动清 G，但 π 不读示范，不能据此自动重置。纯逻辑控制器只重置自己的历史，调用方同时负责模型缓存。缓存只存在内存，不进入 checkpoint。默认 `use_prefix_cache=False`，随机 t 训练保持原截断路径；连续 t 调用才适合此前缀缓存。

数学可见范围与独立截断前向相同；bf16 下不同矩阵形状/归约顺序不承诺逐位相等。tiny 测试沿用既有 native bf16 容差 `rtol=atol=.015`；已缓存前缀自身、未来隔离采用逐位相等检查，并用 hook 验证确实只计算后缀。缓存按历史×层数增长，严格前缀哈希有 CPU 同步成本，尚未测真实 Wan 的净速度与显存收益。

E 目标可在训练前离线生成，不需要先训练一个模型：

```bash
.venv/bin/python -m evo_wam.cli cache-g-pi-targets \
  --config configs/se3/pi_goal.json --tiny-native --index /data/index.json \
  --split train --device cuda --output /tmp/evo-targets
# 有训练工件时可将 --config/--tiny-native 换成 --artifact /data/goal_interface.pt。
# 真正的基座使用本地 --checkpoint，本轮未加载真实权重。
```

缓存配置增加 `"target_cache_index":"/tmp/evo-targets/target-index.json"` 即可供 G、意图混合 G 或 π 训练使用；不设置时继续在线计算 E。目标仍是一张一张单独编码，索引包含完整 E 身份、源索引/任务/数组哈希、精确 subgoal_times、每张目标 latent 哈希和缓存文件哈希。训练选择准确的子目标缓存，不调用 E；对应文件进入续训指纹。缓存按任务从磁盘读取，避免全数据集 token 常驻内存。输出目录必须为空，移动源数据后需要重建这个严格绑定路径的索引。

## C2：π FSDP2

[pi_goal_fsdp.json](../configs/se3/pi_goal_fsdp.json) 在原 π 配置上增加：

```json
"distributed": {"enabled": true, "activation_checkpointing": true}
```

```bash
taskset -c 0 .venv/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=1 -m evo_wam.cli train-goal-interface \
  --config configs/se3/pi_goal_fsdp.json --index /data/pi-index.json \
  --stage pi --initialize /tmp/evo-pi-prior/goal_interface.pt \
  --tiny-native --steps 2 --device cuda --output /tmp/evo-pi-fsdp
# 四卡机器使用 --nproc_per_node=4，实际维度配置和本地 --checkpoint；去掉 --tiny-native。
```

复用 Zero-WAM 的 `shard_model` 和 `apply_ac`。完整训练 LIT、目标投影与动作专家的参数；视频/E 继续冻结。使用 FP32 可训练主权重、BF16 冻结视频存储和 BF16 autocast；上游 sharding 的 `param_dtype=None` 保留这种混合存储，梯度归约为 FP32。

手动动作前向和 LIT `conditions` 注册为 FSDP 前向入口。激活重计算显式保存该次 FlexAttention mask，避免动作前向恢复旧 mask 后，反向重算读到错误可见范围。

每个 rank 每步一个样本，全局 batch 大小等于 world size，按全局步轮转划分；极小数据集允许环回，并不是独立数据的证明。模型初始化相同，随后样本/动作/目标/语言随机流按 rank 分离。开始训练前各 rank 对配置、数据、基座和 E 身份达成一致；逐步合并输入哈希。精确续训要求 world size、配置、种子和输入一致，改变 GPU 数量会明确拒绝。

所有 rank 参与模型/优化器状态收集，只有 rank 0 写文件。checkpoint 仍只含完整可训练参数、优化器、各 rank RNG 和基座引用；不保存冻结权重。导出仍使用原来的 `export-goal-policy`，分片 safetensors 不含优化器或 rank RNG，可在普通单卡 loader 中重建。G 工件仍 v4，阶段 D 的 π 工件为 v4；启用 FSDP 的 π 工件额外校验分布式元数据。

单卡 tiny 已验证真实 FSDP、AC、有无 AC 更新一致、训练更新后 E 不变、精确续训和分片导出。**没有验证 4×A800、多节点性能或真实 Wan 峰值显存**；单卡不能证明多 rank 的通信与数值表现。当前完整状态收集需要主机容纳可训练权重和优化器状态。

## C3：HumanGen RoboTwin

两个命令均只读本地文件，不下载、不安装依赖、不加载模型：

```bash
.venv/bin/python -m evo_wam.cli audit-humangen-g-pi \
  --manifest /data/audit.json --output /tmp/humangen-preflight.json
.venv/bin/python -m evo_wam.cli convert-humangen-g-pi \
  --manifest /data/conversion.json --output /tmp/humangen-g-pi
```

预检 manifest 是 `humangen_robotwin_audit` v1，含 `fields:{state,action,timestamp}`、control_dt、frame_stride、episodes，可选 info.json 路径 `info`。每条 episode 含 episode_id、raw、按相机顺序排列的 robot_latents `.pth` 和 human_latent `.pth`；如果 raw 是从 Parquet 原样导出的 NPZ，可带 original_parquet 与 raw_extraction_evidence。发布 `.pth` 的 `(f h w)c` 被还原为 `[C,F,H,W]`，根据 frame_ids 的每四帧末端计算可用时间；所有文本和文本嵌入都被丢弃。

转换 manifest 是 `humangen_robotwin_conversion` v1，主要字段为 repository、fields、pose_convention、gripper_signal、normalization、robot、event_rules、episodes。完整 fixture 例子见 [测试构造器](../tests/test_g_pi_humangen.py)。它显式记录：

- 仓库 revision/来源依据、每条 episode 的成功和任务级配对证据；不把整段 action_config 文案当作关系或阶段标签。
- 四元数 XYZW/WXYZ、位置单位、源坐标到 robot-base 的刚体变换、工具帧和末端测量依据。
- 夹爪的 measured/command 来源、字段、闭合/张开量程及证据；unknown 会拒绝转成训练工件。
- 原生动作统计文件及哈希、VAE/特征身份、各相机宽度及顺序、精确控制和 latent 两套网格。

动作复用 `RobotwinActionTransform` 的相对位姿、30 维通道、分位数归一化和 `[-2,2]` 截断。输入 actions 已是下一步目标，不再后移；末尾没有下一张观测的动作由原有子目标窗口规则屏蔽。state/action 的原夹爪单位供动作统计归一化，目标 p 的夹爪单独按明确量程映射到 `[0,1]`，避免重复归一化。

允许 weak `gripper` 或 `candidate_match`。命令夹爪必须显式带 `gripper_signal_source=command` 与 `gripper_source_evidence`，事件版本为 command_gripper_v1；它只能产生弱开合事件，不能作为 blocked-close 抓取证据。candidate_match 必须使用测量夹爪。原来的 measured 默认不变。

visual_cache 为 `humangen_robotwin_visuals` v1，必须绑定原始数据、VAE、camera layout、序列缓存，以及独立编码的全部子目标/terminal 单帧。人手缓存可直接读取 `.pth`，原始人手视频哈希可选；其清单视频身份、latent 内容摘要、已知生成父轨迹构成来源闭包，同源不能跨划分。不要把每条配对中的机器人路径误当作生成人手视频的父录像。输出先完整验证再原子落盘，拒绝覆盖。

本轮对用户提供的 place_empty_cup episode 573/92 做了真实预检：170/181 控制行，三相机分别得到 `[48,11,14,54]` / `[48,12,14,54]`，人手为 `[48,16,20,28]`；严格满足 `action[:-1] == observation.state[1:]`。使用机器上已有 Python 3.12/pyarrow 将 Parquet 数值原样导出到独立 `/tmp`，没有改项目环境或样本目录。

**info.json 和 parquet 不能证明夹爪是测量值还是命令值，因此预检元数据记为 unknown，training_ready=false。** 原生 Zero-WAM 消费者采用 XYZW，但发布 schema 仅标 q1…q4；实际转换仍需明确约定和坐标/成功证据。该样本也缺独立单帧目标、全局动作统计和完整 VAE 身份；不能用视频 latent 切片补齐。人手来源 073 配 robot 573/152，723 配 92/389/108，必须保持同组。place_empty_cup 属于原生零样本保留任务，本轮只做审核，未放入训练集。

## 整个 v2 的已知边界

- A 的关系检测仍是视觉几何近似或弱本体标签；真实 SO-101 的候选保留率、停止阈值与执行质量未验证。
- B 的目的分组依赖离线审计，不能自动保证组合泛化；one_way 保留绕过 u 的通路，已有诊断不等于证明语义理解。
- C 的前缀缓存只适合同任务递增历史，未加历史淘汰或真实性能基准；bf16 的独立前向不承诺逐位等价。
- 真实 5B 权重、多卡训练与闭环机器人均未在本机验证；离线动作误差不能替代任务成功率。
- HumanGen 转换需要上述缺项，重新编码/裁切的近重复仍依赖可靠来源记录。旧 G v3、一维 E 缓存等不提供自动迁移。
