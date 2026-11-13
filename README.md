<div align="center">

# 👁️ BlinkOPD

### Teach with Blinks: Breaking Linguistic Momentum in<br>On-Policy Distillation for Fine-Grained Multimodal Perception

<p>
  <a href="https://arxiv.org/pdf/2611.16402"><img src="https://img.shields.io/badge/arXiv-2611.16402-b31b1b?style=flat-square&amp;logo=arxiv" alt="arXiv: 2611.16402"></a>
  <a href="https://huggingface.co/Danny-Lian/BlinkOPD-4B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BlinkOPD--4B-FFD21E?style=flat-square" alt="BlinkOPD-4B weights"></a>
  <a href="https://huggingface.co/Danny-Lian/BlinkOPD-9B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BlinkOPD--9B-FFD21E?style=flat-square" alt="BlinkOPD-9B weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-4C78A8?style=flat-square" alt="Original contributions: CC BY 4.0"></a>
</p>

**[🤗 Models](#models) · [⚡ Quick Start](#quick-start) · [🚀 Training](#training) · [🧩 Baselines](#baselines) · [📊 Evaluation](#evaluation) · [🔬 Probes](#probes)**

</div>

---

<a id="paper-status"></a>
**📄 Paper:** [Teach with Blinks: Breaking Linguistic Momentum in On-Policy Distillation for Fine-Grained Multimodal Perception](https://arxiv.org/pdf/2611.16402)

<a id="models"></a>

## 🤗 Models & Data

| Artifact | Download |
| --- | --- |
| BlinkOPD-4B (0.5K training images) | [🤗 Download](https://huggingface.co/Danny-Lian/BlinkOPD-4B) |
| BlinkOPD-9B (0.5K training images) | [🤗 Download](https://huggingface.co/Danny-Lian/BlinkOPD-9B) |
| Training data source | [🤗 Official dataset](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) |

<a id="quick-start"></a>

## ⚡ Quick Start

### 1. Set up the environment

```bash
git clone https://github.com/Supltz/BlinkOPD.git
cd BlinkOPD
conda create -n blinkopd python=3.12 -y
conda activate blinkopd
pip install uv==0.13.7
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements.txt --override requirements.txt
pip install -e . --no-deps
pip install flash-attn==2.8.3.post1 --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

### 2. Download the official training data

Download [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) into `data/vision_opd_6k/`. Run all commands from the `BlinkOPD/` repository root.

```bash
hf download yuanqianhao/Vision-OPD-6K \
  --repo-type dataset \
  --revision eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4 \
  --local-dir data/vision_opd_6k
cat data/vision_opd_6k/images/images.tar.gz0* | tar -xz -C data/vision_opd_6k/images
tar -xzf data/vision_opd_6k/teacher_images/teacher_images.tar.gz -C data/vision_opd_6k/teacher_images
tar -xzf data/vision_opd_6k/original_images/original_images.tar.gz -C data/vision_opd_6k/original_images
python scripts/prepare_data.py --pool data/vision_opd_6k
```

The script excludes the 21 samples whose original images exactly match V*Bench images, giving a **decontaminated dataset of 6,220 examples**. BlinkOPD uses **500 randomly sampled images**, loaded from the published CSV index.

| CSV | Content |
| --- | --- |
| [Decontaminated dataset](indices/decontaminated_pool.csv) | 6,220 examples after decontamination |
| [Excluded images](indices/excluded_images.csv) | The 21 excluded images and matching V*Bench filenames |
| [Training selection](indices/k500.csv) | The exact 500 training examples |

<details>
<summary><b>Teacher views & random selection</b></summary>

4B uses the original tight teacher crops; 9B uses a crop with three times the bounding-box width and height. Both use the first-glance decisions in `indices/first_glance.csv`.

Verify the local dataset without creating training files:

```bash
python scripts/prepare_data.py --pool data/vision_opd_6k --check-only
```

Reproduce the original random selection with seed 42 and check it against the published 500-image index:

```bash
python scripts/select_k500.py --stages 3000 1000 --check
```

Training reads the fixed CSV index; rerunning random selection is optional.

</details>

<a id="training"></a>

## 🚀 Train BlinkOPD

One command per size trains and merges the final checkpoint:

```bash
bash scripts/train_blinkopd.sh 4b --gpus 0,1,2,3
bash scripts/train_blinkopd.sh 9b --gpus 0,1,2,3
```

<a id="baselines"></a>

## 🧩 Reproduce the Baselines

All five training baselines use the full **6,220-example decontaminated dataset**. Change `--size` to `9b` for the other model size; `--no-eval` stops after training and merging.

```bash
bash baselines/base_model.sh --size 4b --gpus 0,1,2,3
bash baselines/vision_opd.sh --size 4b --gpus 0,1,2,3
bash baselines/opd_v.sh --size 4b --gpus 0,1,2,3
bash baselines/cv_opd.sh --size 4b --gpus 0,1,2,3
bash baselines/vad.sh --size 4b --gpus 0,1,2,3
bash baselines/opd_aha.sh --size 4b --gpus 0,1,2,3
```

<details>
<summary><b>References & pinned official implementations</b></summary>

| Method / reference | Official implementation | Pinned revision |
| --- | --- | --- |
| Vision-OPD | [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD), [paper](https://arxiv.org/abs/2605.18740) | `c8a8fdd` |
| OPD-V | [aniri15/OPD-V](https://github.com/aniri15/OPD-V) | `a780c15` |
| Contrastive Vision-OPD | [HaoLin3419/Contrastive-Vision-OPD](https://github.com/HaoLin3419/Contrastive-Vision-OPD) | `8be9a55` |
| VAD | [DeepExperience/VAD_Multimodal_OPD](https://github.com/DeepExperience/VAD_Multimodal_OPD) | `4f2dd54` |
| OPD-Aha | [Echochef/OPD-Aha](https://github.com/Echochef/OPD-Aha) | `66982fb` |

</details>

<a id="evaluation"></a>

## 📊 Evaluation

### Fine-grained perception · 6 benchmarks

```bash
bash eval/run_eval.sh --model Danny-Lian/BlinkOPD-4B --gpus 0,1,2,3 --name blinkopd-4b
bash eval/run_eval.sh --model Danny-Lian/BlinkOPD-9B --gpus 0,1,2,3 --name blinkopd-9b
```

> **Corrected judge:** the original evaluator's first-letter shortcut can mistake an earlier uppercase letter for the selected option and mark a wrong answer as correct—for example, reading the “A” in `Answer: C` when the ground-truth option is A. Our corrected scorer rejects these matches when the parsed final choice disagrees with the ground truth. All reproduced six-benchmark results use this corrected scorer.

<details>
<summary><b>Official benchmark download links</b></summary>

| Benchmark | Download source |
| --- | --- |
| V*Bench | [lmms-lab-encoder/vstar-bench](https://huggingface.co/datasets/lmms-lab-encoder/vstar-bench) |
| ZoomBench | [inclusionAI/ZoomBench](https://huggingface.co/datasets/inclusionAI/ZoomBench) |
| HR-Bench 4K | [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench) |
| HR-Bench 8K | [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench) |
| MME-RealWorld EN | [yifanzhang114/MME-RealWorld-Lmms-eval](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval) |
| MME-RealWorld CN | [yifanzhang114/MME-RealWorld-CN-Lmms-eval](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-CN-Lmms-eval) |

</details>

### Generalization · 10 held-out benchmarks

One command evaluates all ten:

```bash
bash general_eval/run.sh --model Danny-Lian/BlinkOPD-9B --gpus 0,1,2,3 --name blinkopd-9b
```

<details>
<summary><b>Official benchmark download links</b></summary>

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

</details>

<a id="probes"></a>

## 🔬 Probe Experiments

After downloading and extracting the training data, run the readout experiments, statistics and plots with:

```bash
python probe/run.py --gpus 0,1,2,3
```

The default model is the frozen Qwen3.5-4B base model. `probe/items.csv` identifies the exact 768 questions in the decontaminated dataset and mismatched-image partners used for the prefix readouts.
