# UPAR 2027 Track 1 — Preserving Source-Specific Readouts (team `thanhnguyen`)

Training and evaluation code for our final submission, `best_submission.zip`
(SHA256 `3cb753cf4eaedc93885459e391cccd5d091b98ec7cbd31214cd95abf86d49d9d`),
and for the source-excluded results reported in the paper
*Preserving Source-Specific Readouts for Cross-Domain Pedestrian Attribute Recognition*.

| Result | Paper | How to reproduce |
|---|---:|---|
| Hidden test, final submission | 80.21 | `best_submission.zip` |
| In-domain audit, 6,552 images | 90.04 | Sections 3 and 4 |
| Source-excluded (LOSO) audit, 84,394 images | 86.05 | Section 5 |

The scores are the mean over sources of the harmonic mean of mA and instance F1, in %.

**Files hosted outside this repository (Google Drive).** Download them into the repository root.

| File | Size | SHA256 | Link |
|---|---:|---|---|
| `best_submission.zip` (the submitted v18 package) | 1.15 GB | `3cb753cf4eaedc93885459e391cccd5d091b98ec7cbd31214cd95abf86d49d9d` | BEST_SUBMISSION_LINK |
| `release_checkpoints.zip` (trained encoders and readouts, optional) | 2.5 GB | `6fe5f54003779c1ee0d44ddcd3b7fc580d82743b7486df57df13a599cda766ef` | https://drive.google.com/file/d/1loYjWglqsy7Qss319jgkU8i0mJiGYVKr/view?usp=sharing |

## 1. Method

```
image ──► PE-Core-L/14 (336×168) ─┐                                     ┌──► p_PE  ─┐
      └─► EVA02-L/14  (196×196) ─┤ global + 4 Gaussian body-region     ├──► p_EVA ─┤ mean ─► q = σ(logit p̄ − 0.4·logit π̄) > 0.5
                                  │ tokens ─► committee readout (below) ─┘           ─┘
```

1. **Features.** We take the class token and four patch-token averages, weighted by
   fixed Gaussian regions: head, upper body, lower body and whole body. All five are
   L2-normalised. Each attribute reads the global token plus its own region
   (`scripts/model.py`).
2. **Frozen head, then fine-tuning.** We first fit the regional head on frozen
   pretrained features. We then fine-tune all 24 blocks and the head for 3 epochs:
   - AdamW with learning rates 1e-5 (encoder) and 2e-4 (head), batch 64.
   - Warm-up followed by cosine decay, and an EMA of the weights.
   - Each row is weighted by `n_source^-0.5`, and BCE is polarity-weighted by
     `e^(1-π)` / `e^π`.
3. **WiSE.** The EMA weights are interpolated with the pretrained encoder and the
   frozen-feature head: `α = 0.80` for PE and `0.75` for EVA.
4. **Per-source committee.** We freeze the interpolated encoder and fit one regional
   head per training source (PETA, Market1501, PA100k):
   - Beta(0.4, 0.4) feature mixup, 12 epochs, AdamW learning rate 3e-3.
   - The heads are averaged uniformly in probability space.
   - The source of an input is never inferred.
5. **Distilled teacher readout.** A regional head learns Qwen3-VL-8B yes/no scores on
   24,000 training images (`teacher/targets.npz`) for 16 attributes. Its logit is
   mixed in with weight 0.2. The VLM never runs at inference.
6. **Fusion and decision rule.** The original WiSE head contributes 25% and the
   committee 75%, mixed in probability space. The PE and EVA probabilities are
   averaged. The final rule subtracts `τ·logit(π̄)` with `τ = 0.4`, where `π̄` is the
   source-averaged training prevalence. This gives one fixed offset per attribute,
   the same for every image.

EVA is fine-tuned at 224×224 but deployed at 196×196. Its committee is fitted on
196×196 FP16 features. The submission is an offline TorchScript package with an FP16
backbone, GPU microbatch 32, and no test-time augmentation or adaptation. It runs at
512.8 images/s on an RTX PRO 6000.

## 2. Setup

The code was tested on Linux with one NVIDIA RTX PRO 6000 (96 GB), Python 3.12 and
PyTorch 2.11. Fine-tuning at batch 64 needs about 40 GB of GPU memory. Plan for
about 150 GB of disk for the caches and checkpoints.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Official annotations come from the UPAR 2027 starter kit; images from the challenge page.
python tools/prepare_data.py \
  --official-train /path/to/starterkit/data/annotations/task1/train/gt.csv \
  --official-val   /path/to/starterkit/data/annotations/task1/val/gt.csv \
  --images-root    /path/to/images          # contains PETA/ Market1501/ PA100k/
python tools/download_pretrained.py         # pinned PE-Core-L and EVA02-L weights, hash-checked
```

**Docker.** The `Dockerfile` installs the same environment. It needs an NVIDIA driver and the
NVIDIA container toolkit on the host. Mount the repository, the official annotations and the
images, then run the commands of this README inside the container:

```bash
docker build -t upar-track1 .
docker run --gpus all --rm -it --shm-size=16g \
  -v "$PWD":/workspace/UPAR2027_Track1 \
  -v /path/to/starterkit:/starterkit:ro -v /path/to/images:/images:ro \
  upar-track1
# inside the container, e.g.:
python tools/prepare_data.py --official-train /starterkit/data/annotations/task1/train/gt.csv \
  --official-val /starterkit/data/annotations/task1/val/gt.csv --images-root /images
```

`prepare_data.py` rebuilds every split CSV from the official annotations and the path
lists in `splits/`. It checks each CSV against `reference/hashes.json` and links the
image folders under `data/`.

| Split | PETA | Market1501 | PA100k | Total |
|---|---:|---:|---:|---:|
| release/train (official train + 20,301 val) | 14,068 | 23,814 | 80,088 | 117,970 |
| release/tune (epoch selection) | 782 | 1,322 | 4,450 | 6,554 |
| release/audit (in-domain audit) | 781 | 1,322 | 4,449 | 6,552 |

Splits group Market1501 identities, PETA identity prefixes and exact-file duplicates.
Only organizer training and validation data are used. No hidden-test image is used
at any stage.

## 3. Evaluate the submitted package

```bash
python scripts/evaluate_archive.py --archive best_submission.zip --output reports/best_submission_audit.json
```

`best_submission.zip` is the file from the download table, placed in the repository root.
This loads the ZIP through the organizer API (`load_model` and `predict_batch`) and
checks batch, attribute-order and metadata invariance. It then scores the in-domain
audit. Expected result: **Challenge_Avg 0.900357** (mA 0.915111, instance F1 0.886081).

## 4. Train the submitted predictor from scratch

```bash
python tools/run_pipeline.py release          # print the stages
python tools/run_pipeline.py release --run    # run them (about 2 h on an idle RTX PRO 6000)
```

| Stage | Script | Output |
|---|---|---|
| `cache-{pe,eva}` | `cache_features.py` | frozen pretrained features |
| `frozen-*` | `fit_frozen_head.py` | regional head on frozen features (25 epochs, best tune) |
| `finetune-*` | `finetune.py` | fine-tuned encoder; best EMA epoch on tune |
| `wise-*` | `interpolate.py` | WiSE-interpolated encoder |
| `readout-cache-*` | `cache_features.py` | features of the interpolated encoder (EVA: 196×196, FP16) |
| `readouts-*` | `readouts.py fit` | per-source committee + teacher head |
| `export-*` | `readouts.py export` | TorchScript encoder + readouts |
| `package` | `package.py` | `work/release/submission.zip` |
| `evaluate` | `evaluate_archive.py` | `reports/release_audit.json` |

If other jobs share the GPU, add `--micro-batches 2`. It splits each batch of 64 into
two chunks for the forward and backward passes; the optimizer still steps on batches of 64.

Logs go to `logs/<stage>.log`. Finished stages are recorded in `work/done/`, so
running the same command again resumes after an interruption. A fine-tuning run
interrupted mid-epoch can also be resumed with
`scripts/finetune.py ... --resume <run>/resume.pt`.

GPU training is not bitwise deterministic across hardware and library versions. A
fresh run therefore gives a model close to, but not identical with, the submitted one.
Our own run of this README gave the results in Section 6.

**Exact reproduction from our trained checkpoints.** The optional
`release_checkpoints.zip` holds the interpolated encoders with their fitted
committees: `pe_release.pt` and `eva_release.pt`. Unzip it in the repository root (it creates `checkpoints/`), then
re-export them. This rebuilds predictions identical to `best_submission.zip`.

```bash
for e in pe eva; do
  python scripts/readouts.py export --checkpoint checkpoints/${e}_release.pt --output work/from_checkpoints/${e}_export.pt
done
python scripts/package.py --pe work/from_checkpoints/pe_export.pt --eva work/from_checkpoints/eva_export.pt \
  --output work/from_checkpoints/submission.zip
python scripts/evaluate_archive.py --archive work/from_checkpoints/submission.zip --output reports/from_checkpoints_audit.json
python tools/compare_predictions.py reports/from_checkpoints_audit.npz --require-exact
```

## 5. Source-excluded (LOSO) audit from the paper

Each source is held out in turn. A PE/EVA pair is trained on the other two sources
with the same recipe, and the pair is scored on the held-out source's audit images.
Those images are 4,970 PETA, 2,336 Market1501 and 77,088 PA100k, 84,394 in total.

```bash
python tools/run_pipeline.py loso --run       # about 5 h on an idle GPU; reuses the cache-* stages
```

The fold recipe follows the paper's saved runs:
- The frozen head is fit on the two training sources only (12 epochs, last epoch).
- The encoder uses the last EMA epoch.
- WiSE uses `α = 0.80` for PE and `0.70` for EVA.
- The committee has two source heads.
- Teacher rows from the held-out source are dropped.
- The decision prior is computed on the fold's training sources.

Each fold also reads 3,000 held-out-source tune images, used only for logging. The
audit images exclude those images' identity groups.

The paper's PETA-fold PE model was trained at batch 128; this script uses batch 64
for every fold. `scripts/evaluate_loso.py` writes per-source and mean scores to
`reports/loso_audit.json`.

## 6. Verification of this repository

We followed this README in a fresh checkout on 2026-10-22 and 2026-10-23, on one
RTX PRO 6000 with PyTorch 2.11.0+cu130:

| Check | Result |
|---|---|
| Split CSVs rebuilt from the official annotations | all hashes match |
| `best_submission.zip` on the in-domain audit | 0.900357 |
| Re-export from the trained checkpoints | identical predictions (max difference 0.0); 0.900357 |
| Committee and teacher refit from the original feature caches | identical tensors |
| WiSE interpolation from the original fine-tuned checkpoint | identical tensors |
| Fresh training, release (Section 4) | 0.899656 (mA 0.913562, F1 0.886170); best EMA epoch 2 for both encoders, as recorded; 0.6% of the 262,080 decisions differ from the submission |
| Fresh training, LOSO audit (Section 5) | 0.860893 (PETA 0.8359, Market1501 0.8826, PA100k 0.8641); paper: 86.05 |

## 7. Repository layout

```
scripts/   model.py data.py metrics.py losses.py        library code
           cache_features.py fit_frozen_head.py finetune.py interpolate.py readouts.py
           package.py evaluate_archive.py evaluate_loso.py teacher_scores.py
tools/     prepare_data.py download_pretrained.py run_pipeline.py compare_predictions.py
runtime/   run.py (organizer API, byte-identical to the submission), metadata.yaml, licences
splits/    ordered image-path lists (release, teacher, loso_<source>) and identity groups
teacher/   cached Qwen3-VL-8B scores for 24,000 training images (training only)
reference/ hashes, submission config and the submitted package's audit predictions
configs/   pinned Hugging Face revisions and SHA256 of the pretrained weights
```

`scripts/teacher_scores.py` regenerates `teacher/targets.npz` (optional). It needs
`transformers` and the teacher weights from `tools/download_pretrained.py --teacher`.

## Licences

PE-Core-L (Apache-2.0) and EVA02-L (MIT) licences are in `runtime/`. Qwen3-VL-8B
is used only offline, to produce training targets. The datasets keep their
original terms, and no images are redistributed. The code and trained models are
provided for verification of the challenge results.
