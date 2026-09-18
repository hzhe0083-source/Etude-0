# Evo-WAM

通过可执行物体作用接口连接人类示范、视频世界模型与机器人控制。

研究主线固定为：**建立作用接口 → 跨视角、多时域训练部署主干 → 独立验证候选排序**。

- [冻结方法与实现约定](docs/spec.md)
- [分阶段提交与验收](docs/implementation.md)
- [安装、训练、恢复与推理](docs/running.md)
- [数据格式与划分](docs/data.md)
- [评测协议及真实机器人边界](docs/evaluation.md)

主底座为 [Zero-WAM](https://github.com/robbyant-research/Zero-WAM)，以 Git submodule 固定到 `08e2c4ae41e2b63573a299825cebe6753481407c`。上游代码保留其原有许可与归属；模型权重与数据不进入本仓库。

## 环境

轻量检查使用 Python 3.10、PyTorch 2.9 和标准库 `unittest`。真实底座另需上游依赖；其官方测试环境为 PyTorch 2.9.0 / CUDA 12.6。CPU 或小模型检查不能替代真实检查点、仿真或实机验证。

```bash
git clone --recurse-submodules git@github.com:hzhe0083-source/Evo-WAM.git
cd Evo-WAM
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

## 运行入口

```bash
evo-wam doctor
evo-wam check
evo-wam make-fixture --output outputs/fixture
evo-wam validate-data --index outputs/fixture/index.json
evo-wam check-native
```

训练支持 `interface`、`reader`、`joint` 三种运行阶段（对应两个研究阶段中的接口学习、读取器暖启动及有限联合训练）。`train`、`predict` 和 `calibrate-f` 的完整命令见运行文档。

`configs/` 提供 T0/T1/T2、V0/V1、geometry/full 的匹配配置。默认维度明确标为**合成检查用**；真实检查点训练要求根据数据填写并审计维度、动作表示和时域，不能直接冒充真实机器人设置。

## 验证边界

代码按七个阶段提交和推送。测试区分契约／小计算图、原生 Zero-WAM 随机小模型、完整已训练检查点、模拟器与实机。见 [验证记录](docs/validation.md)。大规模训练、正式统计实验和实机结果须在资源落实后测量；合成检查不代表机器人成功率。
