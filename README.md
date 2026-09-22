# Evo-WAM

Training native Zero-WAM to use human or robot demonstrations through a robot-goal-supervised action interface.

The current experiment uses **observed robot history followed by a complete demonstration** as Wan context. A single clean context pass produces hidden features at each layer. A pose decoder reads those features to predict the target robot's next action-block endpoint pose and gripper state; its hidden features and the corresponding Wan features jointly condition the Action Expert. Joint training uses only robot action, endpoint pose, and gripper supervision. There is **no future-video sampling, video reconstruction loss, or IFP loss** in this route. The external visual/text encoders remain frozen; the Wan backbone, action expert, and pose interface are updated. Deployment freezes the complete policy.

This is a supervised dual-feature experiment initialized from the local Zero-WAM checkpoint, not proof of appearance invariance, demonstration dependence, or robot success. Feeding demonstrations into context does not by itself prevent a policy from ignoring them. The former two-stage, generated-future recurrent interface remains available as a separate comparison with its original checkpoint format and commands.

- [Direct observed-context training: features, supervision, and commands](docs/observed_context.md)
- [Retained SE(3) interface: two-stage generated-future comparison](docs/se3_interface.md)
- [Native human-video ICL: data, training, controls, and export](docs/human_icl.md)
- [Current scope and archived effect-interface specification](docs/spec.md)
- [Validation record and unmeasured results](docs/validation.md)

The independent H1/H2/H3 route remains available: H1 adds human-to-human cross-video prediction, H2 adds temporal demonstration compression, and H3 adds appearance consistency. Those extra human targets need no robot action labels. The new SE(3) route instead requires semantically compatible reference-to-robot pairs, with endpoint and action labels supplied by the target robot. Human references need no human 3D labels. The two interface designs are separate experiments.

Earlier effect-interface experiments also remain available as optional comparisons. B/P pretraining, G/Q requirements, relationship labels, and F ranking are not prerequisites for the current route:

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

The current route uses `cache-goal-language` and `preprocess-icl-video`, then `train-goal-interface --stage joint` with [`configs/se3/observed_dual.json`](configs/se3/observed_dual.json), followed by `export-goal-policy` and `predict-goal-policy`. It starts from the pretrained checkpoint without requiring the earlier nonvisual stage. No LoRA is installed: training keeps FP32 master parameters/optimizer state with BF16 CUDA autocast. Each query supervises one action block. The example sets `state_dim: 12` but remains **synthetic_dimensions_only**; SO101 action mapping, geometric targets, data, and resources still require validation. The output contains actions and predicted robot endpoint pose/gripper, without `generated_future`.

Goal data and artifacts retain version 2, with a distinct `observed_goal_dual_v1` architecture marker. Full training checkpoints support exact continuation; complete safetensors deployment bundles load through the dedicated policy loader without an external base checkpoint. The retained `latent` and `direct_features` routes use `--stage goal` then `--stage visual --initialize ...`; their original 100-token recurrent interface and example configurations remain documented in the [SE(3) guide](docs/se3_interface.md).

The independent ICL route uses `train-native-icl` and `export-native-icl`. It feeds normalized Wan latents into the native ICL context path and trains video/context attention LoRA. H2/H3 also train a shared demonstration compressor and adapter, while leaving target observations uncompressed. H1 exports merged weights in the original model format; enabled demonstration-bottleneck runs export a deployment bundle containing the merged backbone and the required interface weights. Robot batches keep the native video/action objectives; human batches have no action branch or action loss. The action expert and base weights stay frozen during that adaptation.

`configs/icl/` contains R0 (robot-only adaptation), H0 (the same robot data plus ordinary human-video continuation), H1 (the same videos with raw human cross-video ICL), H2 (H1 plus temporal demonstration compression), and H3 (H2 plus appearance consistency). These are explicitly marked **for synthetic checks only**. They keep the robot update count fixed; H0/H1 additionally match human targets, losses, and update counts. H2 preserves H1's predictive losses; H3 changes only the consistency weight and requires an audited appearance variant for each conditioned sample. Candidate bottleneck dimensions are unvalidated. Real data, semantic pair audits, full-checkpoint validation, and measured resource budgets are still required. See the [native ICL guide](docs/human_icl.md) for the exact comparison and input contracts.

The earlier `interface`, `reader`, and `joint` stages and `pretrain-video`/`encode-demonstrations` commands continue to implement the separate effect-interface experiments. Their T0/T1/T2, V0/V1, geometry/full, and U0/U1/U2 configurations also retain synthetic dimensions.

`make-capacity-configs` belongs to those earlier B/P experiments. Its 16/64/100-token candidates remain separate from the native temporal bottleneck, which produces `ceil(T / group_frames) * tokens_per_group` ordered tokens per demonstration.

## Validation scope and limitations

Model weight paths are specified on the server using `configs/server/resources.example.json`. Validation distinguishes contract and small-computation-graph checks, randomly initialized native Zero-WAM small models, full trained checkpoints, simulation, and real robots. See the [validation record](docs/validation.md). No real human-video dataset has been collected for this experiment yet. Large-scale training, cross-view transfer gains, and robot execution results remain unmeasured; neither code tests nor the addition of cross-prediction establish a novel method or improved task success.
