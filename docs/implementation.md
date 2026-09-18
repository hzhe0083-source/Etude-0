# 分阶段交付

每阶段单独 commit、push 到 `main`，核对远程 HEAD；不强制推送。硬件或数据相关检查无法运行时记录实际限制，不用模拟结果替代。

| 阶段 | 提交 | 验收 |
|---|---|---|
| 1 | docs: freeze specification and implementation milestones | 完整规格、版本与环境说明 |
| 2 | feat: add effect contracts and paired training losses | 类型、索引、概率、掩码、JS、监督均值 |
| 3 | feat: add executable effect interfaces | 要求编码/解码、因果物理预测、任务隔离与评分 |
| 4 | feat: integrate Zero-WAM conditioning and training paths | 实际上游接口、小规模前向、冻结及梯度路径 |
| 5 | feat: add paired data and experiment configurations | 配对随机性、统一条件丢弃、分组划分及公平对照 |
| 6 | feat: add rollout diagnostics and candidate evaluation | 前缀/续接标签、局部 Oracle、缓存与行为检查 |
| 7 | test: validate integrated workflows and document execution | 可运行入口、集成验证与限制记录 |

## 实施边界

- 主底座的可调用集成必须使用固定版本 Zero-WAM 的真实实现；小模型仅用来验证相同接口的计算和梯度。
- 代码、配置、文档和小型非敏感测试输入入库。权重、正式数据、生成视频、日志和凭据不入库。
- 真实机器人型号、相机/动作标定和训练算力待定；不虚构这些配置或实际成功率。
- HOST 是第一底座验收后的独立验证，不构成首轮实现的前置依赖。
