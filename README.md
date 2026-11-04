# BlinkOPD

**Teach with Blinks: Breaking Linguistic Momentum in On-Policy Distillation for Fine-Grained Multimodal Perception**

This release contains training code, reproducible baseline launchers, the corrected six-benchmark evaluator, ten held-out benchmark evaluators, and the experiments and plots for Figures 2, 3 and 5.

## Models and data

| Artifact | Download |
| --- | --- |
| BlinkOPD-4B | [Hugging Face](https://huggingface.co/Danny-Lian/BlinkOPD-4B) |
| BlinkOPD-9B | [Hugging Face](https://huggingface.co/Danny-Lian/BlinkOPD-9B) |
| Decontaminated training pool | [Hugging Face](https://huggingface.co/datasets/Danny-Lian/Vision-OPD-6K-Decontaminated) |

The data derives from [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K). We removed 21 of its 6,241 questions whose original images have decoded RGB pixels identical to V*Bench evaluation images. The resulting pool has 6,220 questions. The dataset includes the removal record. This exact-match check against V*Bench does not establish the absence of every possible form of contamination.

All reproduced six-benchmark results use the **corrected judge** protocol described below. This concerns evaluation, not relabeling of the training dataset.

## Installation

Use Linux, Python 3.12, a CUDA-compatible PyTorch installation and Bash 4 or newer. Training uses CUDA, FSDP, Ray and vLLM. GPU memory requirements depend on image sizes, model size and the chosen offload and token budgets.

```bash
conda create -n blinkopd python=3.12 -y
conda activate blinkopd
pip install -r requirements.txt
pip install -e . --no-deps
pip install flash-attn==2.8.3.post1 --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

The principal training versions are PyTorch 2.10.0, Transformers 5.5.0, vLLM 0.18.0 and Ray 2.53.0. Install PyTorch for your CUDA platform before the compiled extensions. Baselines fetch fixed official revisions into `third_party/`; they use that method's implementation rather than silently substituting BlinkOPD.

If a compiled extension reports a `CXXABI` version error under Conda, ensure the environment's C++ runtime is selected, for example `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`, and rebuild the extension if necessary.

Run commands from the repository root. Defaults are relative to this repository: downloaded data goes to `data/`, outputs to `outputs/`, and cloned baselines to `third_party/`. These directories are ignored by Git. Hugging Face caching follows `HF_HOME`; choose a local disk with sufficient free space. `MODEL_PATH`, `DATA_ROOT`, `TRAIN_FILE` and `OUTPUT_DIR` can override the main training paths. Generated Parquet files contain absolute paths for the current machine; regenerate them after moving the dataset.

## Prepare the training data

```bash
python scripts/prepare_data.py
```

Training uses **500 images randomly sampled from the decontaminated 6.2K training pool**. The preparation script downloads and verifies the data, extracts images, and prepares the training files. The selected image index and random sampling code are included for reproducibility.

The downloader pins dataset revision `51bc6e1a2530f983a6b2f25192c1037f89f564df`. The three archives total about 37.4 GB; allow approximately 75 GB for cached archives plus extracted files, with additional space for models and training checkpoints.

The 4B recipe uses the original tight teacher crops. The 9B recipe uses a window with three times the bounding-box width and height, shifted inside the image, and resized with the original capped Lanczos rule. The released frozen first-glance decisions for both sizes were computed on tight crops; the 9B training crop changes only the later teacher view.

`indices/k500.csv` gives the zero-based pool row, student/original image names and question SHA-256 for each selected example. `indices/first_glance.csv` contains the corresponding cached base-model decisions. Probe indices refer to the same retained pool row order.

To reproduce the released random selection and verify the image index:

```bash
python scripts/select_k500.py --stages 3000 1000 --check
```

The command reproduces the released selection with seed 42. Training uses the released image index.

## Train BlinkOPD

One command per size trains and merges the final checkpoint:

```bash
bash scripts/train_blinkopd.sh 4b --gpus 0,1,2,3
bash scripts/train_blinkopd.sh 9b --gpus 0,1,2,3
```

The recipe uses 38 optimizer steps, batch size 96, eight rollouts per question, a frozen base teacher, learning rate `2e-6`, five warmup steps and a constant schedule. It uses top-100-plus-tail JSD, importance clipping at 2, eyes-closed extrapolation with gain 3.4, and first-glance conflict scales 2/1. Packed sequences remain isolated. The 9B recipe excludes the first response token from distillation. Prefix sharing is disabled.

The 4B defaults keep actor parameters and optimizer state resident and use a 12,288-token actor micro-batch budget. The 9B defaults offload them and use 9,216 tokens. These are fixed defaults, not automatic GPU handoff or adaptive memory-occupancy rules. Set `ACTOR_PARAM_OFFLOAD`, `ACTOR_OPTIMIZER_OFFLOAD`, `PPO_MAX_TOKEN_LEN_PER_GPU` and `ROLLOUT_GPU_MEMORY_UTILIZATION` for your hardware. Changing these infrastructure settings can affect throughput and floating-point execution order.

Additional arguments are Hydra overrides. Print a resolved configuration without loading models or starting Ray:

```bash
N_GPUS=7 bash scripts/train_blinkopd.sh 4b --print-config
```

Outputs are under `outputs/blinkopd_<size>_k500/`. The merged inference model is in `model/`; resumable training state is in `checkpoints/`. To resume, use the same output directory and recipe; the trainer's default resume mode is `auto`.

The BlinkOPD FSDP path supports GPU counts that do not divide the logical batch: transport padding is excluded from the objective and real-row loss normalization is preserved. FSDP1 resume can reconstruct saved shards when the GPU count changes; all saved model, optimizer and extra-state shard files must be available. A changed GPU count does not guarantee bitwise-identical continuation. FSDP2 resharding is not implemented.

For multiple nodes, first create a Ray cluster with your own head address and GPU allocation:

```bash
ray start --head --port=6379 --num-gpus=4
ray start --address=<head-address>:6379 --num-gpus=4
NNODES=2 RAY_ADDRESS=<head-address>:6379 bash scripts/train_blinkopd.sh 9b --gpus 0,1,2,3
```

Run the head command on the head, the worker command on each additional node, and the training command once on the head. Make code, model and generated data paths accessible at the same locations on every node. These commands join an existing GPU allocation; they do not request machines.

## Reproduce the baselines

After data preparation, each command trains the official method and evaluates its merged model with the common corrected six-benchmark scorer. The base-model command performs evaluation only. Change `--size` to `9b` for the other size; `--no-eval` stops after training and merging.

```bash
bash baselines/base_model.sh --size 4b --gpus 0,1,2,3
bash baselines/vision_opd.sh --size 4b --gpus 0,1,2,3
bash baselines/opd_v.sh --size 4b --gpus 0,1,2,3
bash baselines/cv_opd.sh --size 4b --gpus 0,1,2,3
bash baselines/vad.sh --size 4b --gpus 0,1,2,3
bash baselines/opd_aha.sh --size 4b --gpus 0,1,2,3
```

Baselines use the full 6,220-question decontaminated pool. They retain their method-specific training objectives. VAD's online benchmark monitoring is disabled for the reproduced training command; final evaluation uses the common scorer. OPD-Aha runs 70 steps by default; use `BASELINE_STEPS=40` to produce its 40-step variant. Other baseline defaults follow their epoch/batch settings. Outputs are in `outputs/baselines/` and `outputs/eval/`.

The external baseline implementations require the total rollout batch to be divisible by the total GPU count. The launchers validate this; BlinkOPD's arbitrary-count padding changes are not injected into those independent official repositories. For multi-node baselines, set `NNODES` and `RAY_ADDRESS` as above. `POOL_DIR`, `THIRD_PARTY_DIR`, `MODEL_PATH` and `OUTPUT_DIR` configure baseline paths. Hydra overrides follow `--`.

| Method / reference | Official implementation | Pinned revision |
| --- | --- | --- |
| Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation | [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD), [paper](https://arxiv.org/abs/2605.18740) | `c8a8fdd` |
| OPD-V | [aniri15/OPD-V](https://github.com/aniri15/OPD-V) | `a780c15` |
| Contrastive Vision-OPD | [HaoLin3419/Contrastive-Vision-OPD](https://github.com/HaoLin3419/Contrastive-Vision-OPD) | `8be9a55` |
| VAD | [DeepExperience/VAD_Multimodal_OPD](https://github.com/DeepExperience/VAD_Multimodal_OPD) | `4f2dd54` |
| OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation | [Echochef/OPD-Aha](https://github.com/Echochef/OPD-Aha) | `66982fb` |

Use the linked authors' current citation instructions when citing each baseline. The launchers restore missing configuration/model support files from the pinned Vision-OPD revision where an official repository's ignore rules omitted them.

## Evaluate the released models

```bash
bash eval/run_eval.sh --model Danny-Lian/BlinkOPD-4B --gpus 0,1,2,3 --name blinkopd-4b
bash eval/run_eval.sh --model Danny-Lian/BlinkOPD-9B --gpus 0,1,2,3 --name blinkopd-9b
```

The runner downloads benchmark data from pinned official sources, launches vLLM inference replicas, then frees those replicas and serves `openai/gpt-oss-120b` for unresolved judge requests. A single selected GPU is supported if both models individually fit. `--judge-gpus` can reserve a separate, disjoint judge allocation. `MODEL_TP` and `JUDGE_TP` configure tensor parallelism. Use `DATA_DIR`, `OUTPUT_ROOT`, `MODEL_PORT` and `JUDGE_PORT` to configure storage and ports.

The six benchmarks are [V*Bench](https://huggingface.co/datasets/lmms-lab-encoder/vstar-bench), [ZoomBench](https://huggingface.co/datasets/inclusionAI/ZoomBench), [HR-Bench 4K/8K](https://huggingface.co/datasets/DreamMr/HR-Bench), [MME-RealWorld EN](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval) and [MME-RealWorld CN](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-CN-Lmms-eval). `eval/prepare_data.py` records exact revisions and reproduces the prompts and retained question order. Benchmark images are downloaded by the user and are not redistributed here.

The original evaluator can accept a response when its first option-like letter matches the ground truth, even if the final answer differs. The corrected path checks parsed final choices before accepting that shortcut. For example, a response ending `Answer: (C)` must not be accepted for ground truth A merely because the word “Answer” begins with A. Both original and corrected scores are saved. Remaining judge requests use temperature zero; malformed verdicts are retried, persistent unresolved outputs are recorded as incorrect, and API failures cause an incomplete evaluation to exit. Resume with the same command to reuse completed outputs and cached judgments.

`results.json` reports per-benchmark scores and their unweighted mean. Full answers, judge records and logs remain under the chosen output directory.

## Ten held-out benchmarks

One command evaluates all ten:

```bash
bash general_eval/run.sh --model Danny-Lian/BlinkOPD-9B --gpus 0,1,2,3 --name blinkopd-9b
```

Use `--benchmarks` to choose a comma-separated subset. The runner uses official prompts, greedy decoding with thinking disabled, benchmark-specific rules and a local GPT-OSS-120B judge for unresolved cases. `--judge-gpu` permits a separate judge allocation; without it, inference and judging run in sequence on the selected GPUs. Downloads and results default to `data/general/` and `outputs/general_eval/`.

| Benchmark | Download source |
| --- | --- |
| MMStar | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| CV-Bench | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| MMVP | [MMVP/MMVP](https://huggingface.co/datasets/MMVP/MMVP) |
| POPE | [lmms-lab-encoder/POPE](https://huggingface.co/datasets/lmms-lab-encoder/POPE) |
| HallusionBench | [lmms-lab-encoder/HallusionBench](https://huggingface.co/datasets/lmms-lab-encoder/HallusionBench) |
| MathVista testmini | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) |
| MathVerse testmini | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) |
| MATH-Vision test | [MathLLMs/MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) |
| We-Math testmini | [We-Math/We-Math](https://huggingface.co/datasets/We-Math/We-Math) |
| DynaMath | [DynaMath/DynaMath_Sample](https://huggingface.co/datasets/DynaMath/DynaMath_Sample) |

The converter pins dataset revisions and verifies question counts. Dataset licenses and access conditions remain those of their sources.

## Figures 2, 3 and 5

After preparing the training pool, run the readout experiments, statistics and plots with:

```bash
python probe/run.py --gpus 0,1,2,3
```

The default model is the frozen Qwen3.5-4B base model. `probe/items.csv` identifies the exact 768 pool questions and mismatched-image partners used for the prefix readouts. It includes the original tight-teacher image name so row order can be checked. Four student rollouts per question are probed at 0/25/50/75/100% of the generated prefix.

Figure 2 measures follow rate and survival score for Vision-OPD, OPD-V, VAD, CV-OPD, OPD-Aha and BlinkOPD. Figure 3 measures reference residual drift and false-belief probability; its second panel uses 300 MME-RealWorld-CN questions identified in `probe/mmerw_cn_300.csv`, not additional training examples. Figure 5 measures corrective density. The released targets and readouts follow the definitions in the experiment code; OPD-V's weighted target falls back to the student when its answer-token weight is zero.

Outputs in `outputs/probe/` include resumable per-shard records, compressed merged readouts, `numbers.json`, `numbers.csv`, and PDF/PNG figures. Rerunning the same command resumes completed questions. Generation is stochastic: recomputing rollouts across different inference libraries, hardware or versions need not exactly reproduce published point estimates. To reuse an existing rollout file, pass `--rollouts`; to redraw existing numbers, use `--plot-only`. `--skip-etoken` omits Figure 3's benchmark panel.

## Files and implementation

| Path | Purpose |
| --- | --- |
| `scripts/train_blinkopd.sh` | Training entry point and final model merge |
| `scripts/prepare_data.py` | Verified data download, local paths and size-specific teacher views |
| `scripts/select_k500.py` | Deterministic image-level sampling and index verification |
| `scripts/print_config.py` | Resolve Hydra configuration without initializing the training runtime |
| `scripts/merge_checkpoint.sh` | Merge actor shards and install the perception chat template |
| `indices/k500.csv` | Exact selected pool rows and question hashes |
| `indices/first_glance.csv` | Frozen first-glance decisions for selected rows |
| `baselines/common.sh` | Fixed-revision fetching, data setup, checkpoint merge and evaluation |
| `baselines/base_model.sh` | Evaluate an untrained base model |
| `baselines/{vision_opd,opd_v,cv_opd,vad,opd_aha}.sh` | One-command official baseline recipes |
| `eval/{prepare_data,infer,score}.py`, `eval/run_eval.sh` | Six-benchmark downloads, inference and corrected scoring |
| `general_eval/{prepare_data,evaluate,scoring}.py`, `general_eval/run.sh` | Ten-benchmark preparation, inference and scoring |
| `general_eval/official/` | Benchmark-specific parsing and scoring helpers; source attribution is retained |
| `probe/{run,readout,stats,plot}.py` | Probe orchestration, GPU readouts, numerical summaries and figures |
| `probe/items.csv`, `probe/mmerw_cn_300.csv` | Exact probe and benchmark question indices |
| `chat_templates/perception_chat_template_qwen35.jinja` | Perception template used for training and inference |
| `verl/trainer/ppo/{ray_trainer,core_algos}.py` | Teacher-input construction, training loop and distillation objective |
| `verl/workers/actor/dp_actor.py` | Actor scoring, distillation and masked-row normalization |
| `verl/models/transformers/qwen3_5{,_packed}.py` | Qwen3.5 forward integration and isolated packed linear attention |
| `verl/utils/{cut_reference,first_glance_gate}.py` | Eyes-closed inputs and first-glance gain |
| `verl/utils/{teacher_prefix_cache,image_processor_memo,mm_transport}.py` | Teacher-prefix reuse, image processing reuse and multimodal transport |
| `verl/utils/dp_padding.py` | Arbitrary-count transport padding and objective normalization |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | Save/resume and FSDP1 shard reconstruction |
| Remaining `verl/` modules | Required distributed runtime, configuration, data loading, rollout and merge support inherited from OPD-Aha/verl |
| `requirements.txt`, `pyproject.toml` | Environment versions and package installation |
| `LICENSE`, `NOTICE` | License and third-party attribution |
| `RELEASE_MANIFEST.json` | SHA-256 inventory of the released files |

The included acceleration preserves the objective: teacher-input prefetch and caching, per-prompt/image reuse, multimodal NumPy transport, padding for indivisible GPU counts, and epoch-boundary resume. It does not include site-specific launchers, scheduler policies, private filesystem integrations or GPU handoff policies.

Release checks cover resolved configurations for both BlinkOPD sizes and all five baselines, index/row-order verification, 1–8-rank loss-normalization math, DataProto padding transport, cache and tensor round trips, and corrected-answer parsing. A single-GPU test checks isolated packed Qwen3.5 forward/backward computation, and both released models are checked for image-conditioned forward/generation. These checks are not a rerun of every full training experiment or all benchmark evaluations.

## Acknowledgments and license

The runtime is derived from [OPD-Aha](https://github.com/Echochef/OPD-Aha), [Vision-OPD](https://github.com/VisionOPD/Vision-OPD) and [verl](https://github.com/volcengine/verl). The base models are [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Preserve the upstream attributions when redistributing this code. Apache-2.0 applies as described in `LICENSE`; third-party components retain their own license notices. Original license headers are retained even though implementation comments and docstrings are omitted from the release.
