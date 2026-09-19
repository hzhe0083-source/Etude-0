# 原始视觉输入与服务器资源

权重按用户要求留空，由服务器提供本地路径；不自动下载。复制
`configs/server/resources.example.json` 到服务器工作目录填写路径和标定信息。
训练仍通过 `train --checkpoint /server/zero-wam` 显式加载；视觉入口通过
`vae_path` 和覆盖 config、所有 safetensors 分片及索引的 `vae_sha256` 验证来源。
资源清单是交接模板，不能直接作为实验配置。

```bash
evo-wam preprocess-visual --manifest /server/episode/raw.json --output /server/episode/encoded
evo-wam predict --artifact /server/runs/joint/adapter.pt --checkpoint /server/zero-wam \
  --manifest /server/episode/encoded/observation.json --output /server/prediction
```

`raw.json` 为 `format_version: 1, kind: "raw_visual_observation"`，包含：

- `arrays`：原始机器人 NPZ；`camera_order`：固定相机名称数组。
- `robot_size`、`demo_size`：显式 `[height,width]`，16 的倍数；首版所有机器人相机采用相同分辨率，按宽度拼接。
- `demonstrations: [{view_id, video}]`：本地人类视频路径；`demo_fps`：明确采样帧率。视频按时间采样，保留 `1+4k` 帧，丢弃不足一个编码组的尾部，记录原始帧索引。
- `vae_path`、`vae_sha256`；`entity_source` 为 `visual_tracks` 或 `simulator_oracle`；`tracker_identity` 和 `coordinate_frame` 必填。
- v2 观测的 `chunk_size`、`actions_per_frame`、`action_space`、`observed_action_space`、`observation_step`、`control_dt`、`history_chunks`，含义见 `data.md`。

原始 NPZ 只允许以下字段，不能含未来标签：

| 字段 | 形状与含义 |
|---|---|
| `rgb_<camera>` | uint8 `[T,H,W,3]`，RGB，同步相机，T=1+4k |
| `rgb_step_offsets` | int64 `[T]`，相对当前观测的控制步，严格递增且最后为0 |
| `entity_ids` | int64 `[N]`，稳定非负跟踪ID |
| `entity_masks` | `[F,N,C,H,W]`，在编码端点时刻的物体跟踪区域，F=1+(T-1)/4 |
| `proprio_history` | float `[F,Dp]`，与端点时刻同步 |
| `embodiment` | float `[De]` |
| `observed_action_history` | float `[K,Da]`，已实际提交的归一化动作，可显式为空 |
| `observed_action_step_offsets` | int64 `[K]`，已执行动作的结束时间 |

使用冻结的真实 `AutoencoderKLWan` 后验均值和 `(latent-mean)/std`。
人类 tokens 按时间、高度、宽度展开；机器人相机按声明顺序横向拼接，实体特征是纯观测区域的加权池化。
配置中的 `demo_dim/entity_dim` 应与实际 VAE latent 通道数一致，不能继续使用数值 fixture 的维度。
缓存身份包括权重、源视频、原始数组、采样配置、相机顺序与跟踪来源。

实体检测、跨帧跟踪和几何标定仍由显式输入提供，不能从这个入口宣称已实现无标注的任意对象感知。
ROI 完全不可见或时间不齐会报错；未来可接已验证的状态估计器，本版不会编造特征。
`visual_tracks` 与 `simulator_oracle` 的实验必须分开记录。
离线训练可复用相同编码函数生成 clean latent 和观测特征；作用标签与视角证据仍按 v2 数据契约由标注流程给出。
这里没有把从原始视频估计接触和必要事件的问题伪装成自动标注。

本地检查只使用随机微型 VAE 验证真实编码接口，不代表服务器完整 VAE、模型权重或视觉任务准确率已经验收。
