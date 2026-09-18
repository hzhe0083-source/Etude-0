# Evo-WAM

通过可执行物体作用接口连接人类示范、视频世界模型与机器人控制。

研究主线固定为：**建立作用接口 → 跨视角、多时域训练部署主干 → 独立验证候选排序**。

- [冻结方法与实现约定](docs/spec.md)
- [分阶段提交与验收](docs/implementation.md)

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

## 交付状态

代码按 `docs/implementation.md` 的七个阶段交付，每阶段通过对应验收后提交和推送。大规模训练、正式统计实验和实机结果须在资源落实后测量；此仓库不把合成检查报告为机器人实验结果。
