# Evo-WAM

Connecting human demonstrations, video world models, and robot control through executable object-effect interfaces.

The research workflow is: **learn the effect interface → train the deployed backbone with cross-view, multi-horizon supervision → independently validate candidate ranking**.

- [Method specification and implementation contract](docs/spec.md)
- [Implementation stages and acceptance checks](docs/implementation.md)
- [Installation, training, checkpoint recovery, and inference](docs/running.md)
- [Data formats and splits](docs/data.md)
- [Unpaired video pretraining and reader integration](docs/unpaired_video.md)
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

Training supports three runtime stages: `interface`, `reader`, and `joint`. These implement interface learning, reader warmup, and limited joint training within the two research stages. See the running guide for complete `train`, `predict`, and `calibrate-f` commands.

`preprocess-video`, `pretrain-video`, and `encode-demonstrations` provide the path from unpaired single-view videos to effect tokens and the existing reader. Synchronized multi-view data supplies optional supervision. The video objective trains only the lightweight encoder and prediction heads; WAM is updated during the subsequent robot training stage.

`configs/` provides matched T0/T1/T2, V0/V1, and geometry/full configurations. Default dimensions are explicitly marked **for synthetic checks only**. Training with real checkpoints requires data-specific, audited dimensions, action representations, and time horizons; the defaults are not validated robot configurations.

## Validation scope and limitations

The initial implementation was committed in seven stages; review fixes continue as separate commits and pushes. Model weight paths are specified on the server using `configs/server/resources.example.json`. Validation distinguishes contract and small-computation-graph checks, randomly initialized native Zero-WAM small models, full trained checkpoints, simulation, and real robots. See the [validation record](docs/validation.md). Large-scale training, formal statistical experiments, and real-robot results remain to be measured once the required resources are available. Synthetic checks do not establish robot task success rates.
