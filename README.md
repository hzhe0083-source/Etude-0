# Evo-WAM

Training native Zero-WAM with additional human demonstration videos for in-context robot control.

The current experiment adds **human-to-human cross-video prediction** to the existing robot ICL objective: watch demonstration A, observe the earlier part of an independent execution B, and predict B's continuation through Zero-WAM's native video pathway. A and B need audited task compatibility; each additional human video does not need a matching robot trajectory or human action labels. At deployment, the original demonstration-to-robot execution path runs with fixed parameters.

- [Native human-video ICL: data, training, controls, and export](docs/human_icl.md)
- [Current scope and archived effect-interface specification](docs/spec.md)
- [Validation record and unmeasured results](docs/validation.md)

Earlier effect-interface experiments remain available as optional comparisons. B/P pretraining, G/Q requirements, relationship labels, and F ranking are not prerequisites for the current route:

- [Archived B/P pretraining and reader integration](docs/unpaired_video.md)
- [Effect-interface stages and acceptance checks](docs/implementation.md)
- [Effect-interface training and inference](docs/running.md)
- [Effect-interface data formats and splits](docs/data.md)
- [Raw visual preprocessing and model weight paths](docs/vision.md)
- [RoboTwin closed-loop execution with oracle requirements](docs/robotwin.md)
- [Evaluation protocol and real-robot limitations](docs/evaluation.md)

The base model is [Zero-WAM](https://github.com/robbyant-research/Zero-WAM), pinned as a Git submodule at `08e2c4ae41e2b63573a299825cebe6753481407c`. Upstream code retains its original license and attribution. Model weights and datasets are not included in this repository.

## Environment

Lightweight checks use Python 3.10, PyTorch 2.9, and the standard-library `unittest` module. Running the native backbone also requires the upstream dependencies; its documented test environment uses PyTorch 2.9.0 / CUDA 12.6. CPU and small-model checks do not replace validation with trained checkpoints, simulation, or real robots.

```bash
git clone --recurse-submodules git@github.com:hzhe0083-source/Evo-WAM.git
cd Evo-WAM
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

## Commands

```bash
evo-wam doctor
evo-wam check
evo-wam make-fixture --output outputs/fixture
evo-wam validate-data --index outputs/fixture/index.json
evo-wam check-native
```

The current route uses `preprocess-icl-video`, `train-native-icl`, and `export-native-icl`. It feeds normalized Wan latents into the native ICL context path, trains video/context attention LoRA, and exports merged weights in the original model format. Robot batches keep the native video/action objectives; human batches have no action branch or action loss. The action expert and base weights stay frozen during this adaptation.

`configs/icl/` contains R0 (robot-only adaptation), H0 (the same robot data plus ordinary human-video continuation), and H1 (the same videos with human cross-video ICL). These are explicitly marked **for synthetic checks only**. They keep the robot update count fixed; H0/H1 additionally match human targets, losses, and update counts. Real data, semantic pair audits, full-checkpoint validation, and measured resource budgets are still required. See the [native ICL guide](docs/human_icl.md) for the exact comparison and input contracts.

The earlier `interface`, `reader`, and `joint` stages and `pretrain-video`/`encode-demonstrations` commands continue to implement the separate effect-interface experiments. Their T0/T1/T2, V0/V1, geometry/full, and U0/U1/U2 configurations also retain synthetic dimensions.

`make-capacity-configs` belongs to those earlier B/P experiments. Its 16/64/100-token candidates are not a new bottleneck in the native ICL route.

## Validation scope and limitations

Model weight paths are specified on the server using `configs/server/resources.example.json`. Validation distinguishes contract and small-computation-graph checks, randomly initialized native Zero-WAM small models, full trained checkpoints, simulation, and real robots. See the [validation record](docs/validation.md). No real human-video dataset has been collected for this experiment yet. Large-scale training, cross-view transfer gains, and robot execution results remain unmeasured; neither code tests nor the addition of cross-prediction establish a novel method or improved task success.
