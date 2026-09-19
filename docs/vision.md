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
`raw_features` 模式的 `demo_dim/entity_dim` 应与实际 VAE latent 通道数一致；
使用 B 时 `demo_dim` 改为 artifact 的 `token_dim`，机器人 `entity_dim` 仍为 VAE 通道数。
不能继续使用数值 fixture 的维度。
缓存身份包括权重、源视频、原始数组、采样配置、相机顺序与跟踪来源。

实体检测、跨帧跟踪和几何标定仍由显式输入提供，不能从这个入口宣称已实现无标注的任意对象感知。
ROI 完全不可见或时间不齐会报错；未来可接已验证的状态估计器，本版不会编造特征。
`visual_tracks` 与 `simulator_oracle` 的实验必须分开记录。
离线训练可复用相同编码函数生成 clean latent 和观测特征；作用标签与视角证据仍按 v2 数据契约由标注流程给出。
这里没有把从原始视频估计接触和必要事件的问题伪装成自动标注。

本地检查只使用随机微型 VAE 验证真实编码接口，不代表服务器完整 VAE、模型权重或视觉任务准确率已经验收。

## 单视角、非配对视频预训练窗口

`preprocess_video(manifest_path, output, device="cuda")` 接受独立的
`format_version: 1, kind: "raw_video_pretrain"` JSON。它不需要机器人轨迹、
第二视角、物体跟踪、动作或接触标签；只提取冻结 Wan VAE 的观测 patch 特征。
这些窗口训练小型视频作用编码器 B，不直接对 W 计算人类视频损失。

```json
{
  "format_version": 1,
  "kind": "raw_video_pretrain",
  "video": "screened-continuous-clip.mp4",
  "vae_path": "/server/models/wan/vae",
  "vae_sha256": {"config.json": "填入真实SHA256", "diffusion_pytorch_model.safetensors": "填入真实SHA256"},
  "size": [320, 480],
  "fps": 12,
  "domain": "human",
  "source_id": "original-clip-001",
  "source_group": "original-recording-001",
  "feature_space_id": "audited-wan-feature-space-v1",
  "split": "train",
  "window_frames": 5,
  "context_frames": 2,
  "continuous_segment_verified": true
}
```

视频必须预先剪成已筛选的连续片段。入口不自动检测镜头切换，不接受隐式
起止裁剪参数。`window_frames/context_frames` 指编码后的时间帧数：每窗至少
1 个上下文帧和 2 个未来观测帧。默认相邻窗口不重叠；可显式设置
`window_stride_frames`，其值须在 1 到窗口长度之间。不足一个完整窗口的尾部
丢弃并记录，不能补成虚假的未来监督。机器人视频可另填真实 `trajectory_id`。

输出 format v1 的 `index.json` 与多个 format v2 的 `window_*.json/.npz`。
窗口元数据记录真实 latent `patch_grid=[H,W]` 和
`patch_coordinate_system="normalized_xy_patch_centers"`；每个 NPZ 仅含
`features[T,H*W,C]`、同形 Boolean `feature_valid`、`frame_times[T]` 和
`patch_coordinates[H*W,2]`。坐标按高度、宽度展开后的槽位顺序保存，不能从 token 数量猜测网格。
第 `r` 行、第 `c` 列的中心坐标为 `x=2*(c+0.5)/W-1`、`y=2*(r+0.5)/H-1`，行列从0计数。
时间使用实际采样原始帧索引的编码端点 `frame_indices[::4] / source_fps`，
保留源视频秒数，不伪装成机器人控制步。先因果编码完整片段，再切分特征窗口；
每窗保留源文件哈希、VAE 哈希、分辨率、采样率、端点索引和窗口策略。
所有窗口继承同一个 `source_id/source_group/split`；跨文件汇总时索引加载器还会
检查源组及下游机器人来源是否跨训练／验证／测试划分泄漏。

生成的窗口不含自动作用标签。patch 可观测性不等于目标身份或接触状态已经标注。

## 将已训练 B 用于机器人示范入口

原 `raw_visual_observation` 可增加 `demo_encoder_artifact` 本地路径、必填的
`demo_feature_space_id`，以及可选 `demo_encoder_sha256`。预处理先核对实际 artifact
哈希与特征空间，再将每段归一化 Wan 特征 `[1,T,H*W,C]`、全有效观测掩码和源视频
端点秒数及真实 patch 布局送入冻结编码器的 `encode_demo`，按 artifact 的 `window_frames` 生成有序
作用 tokens；无随机扰动，不更新 W。

输出的 `demonstration_encoding` 记录 `encoder_version: 2`、编码器 SHA256、特征空间、token 维度、窗口长度
和每窗 token 数，供下游机器人 reader artifact 核对。完整窗口策略另记入视觉来源：
非重叠分窗，尾部只重复数值并标记为无效，绝不把补齐项当作观测。

未提供 B artifact 时继续输出 `raw_features`，并新增 `demonstration_layouts`，为每个
视角记录 `frames/tokens_per_frame/frame_times`、`feature_kind="patches"`、
`patch_grid`、`patch_coordinate_system="normalized_xy_patch_centers"` 和
`token_order="time,height,width,channel"`。
顺序固定为时间、高度、宽度、通道。转换既有数据时必须使用这些时间和布局，
不能根据扁平 token 总数猜测帧数或网格。B 已编码模式不会把作用 tokens 冒充原始 patch 布局。

空间／时间对应修复后的 B artifact 为 v2。v1 artifact 和旧作用 token 缓存须重新预训练／导出，再训练对应读取器；不能把旧编码器身份用于新结构或从旧 artifact 续训。
