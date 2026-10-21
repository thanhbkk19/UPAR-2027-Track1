# UPAR 2027 Track 1 — PE/EVA Balanced

Training, export and evaluation code for team **thanhnguyen**'s selected
`UPAR_Track1_PE_EVA_Balanced_v18.zip` predictor. The repository implements the
fixed submitted recipe; it does not search over alternative systems or adapt
to hidden test images.

The predictor recognizes 40 pedestrian attributes using two pretrained visual
encoders, global and regional features, source-specific classifiers and a
training-only vision-language teacher. It exposes the organizer's
`load_model()`, `predict_image(sample)` and `predict_batch(samples)` API in a
self-contained, offline ZIP.

**Reference ZIP SHA256:**

```text
3cb753cf4eaedc93885459e391cccd5d091b98ec7cbd31214cd95abf86d49d9d
```

## Reproduction paths

| Goal | Required inputs | Procedure |
|---|---|---|
| Evaluate exactly the submitted artifact | Original ZIP and organizer images | Verify its hash, then run the evaluator below |
| Rebuild the predictor from its trained parameters | Original PE/EVA readout checkpoints in the organizer bundle | Restore the bundle; run `export-pe` through `evaluate` |
| Train the entire method again | Organizer data and public pretrained PE/EVA weights | Run the complete 17-stage pipeline |

The source repository includes the **original 24,000-image teacher targets**,
split assignments, recorded configurations and hashes. It excludes dataset
images and large model weights. The separate `UPAR_2027_Track1_Organizer.tar`
handoff contains this source tree, the original ZIP and the two trained
checkpoints. These checkpoints are optional for training from scratch.

Handoff verification reproduced the original probabilities **exactly on all
6,552 audit images**. WiSE interpolation and source/teacher-head retraining
from the original feature caches also reproduced their tensors exactly. Full
encoder fine-tuning was not rerun during handoff preparation.

Exact archive identity is established by SHA256. Re-exporting can change
TorchScript/ZIP serialization bytes while preserving model parameters and
predictions. Fresh GPU training is not guaranteed to be bitwise identical
across hardware, CUDA versions or library builds. The code preserves the
recorded data order, seeds, geometry, precision and selection rule; evaluate
each fresh run rather than assuming it has the reference score. See
[validation scope](docs/REPRODUCIBILITY.md).

## Method

1. Cache global and four fixed Gaussian-region features from pretrained PE and
   EVA encoders; fit supervised regional classifiers on the frozen features.
2. Fine-tune the transformer blocks and classifiers on the expanded public
   training split, selecting EMA checkpoints using the tuning split.
3. Interpolate each selected EMA with its pretrained encoder and supervised
   frozen-feature head (PE: 0.80; EVA: 0.75).
4. Freeze the encoders. Fit one classifier per source with feature mixup and
   source-local label weights; average their probabilities uniformly.
5. Fit a small regional classifier to Qwen3-VL-8B attribute scores on 24,000
   training images. Blend its logits with the source committee for 16 attributes.
6. Blend original and robust predictions, average the two encoders, then apply
   the same training-derived prior-odds offset to every image.

| Setting | PE | EVA |
|---|---|---|
| Backbone | PE-Core-L/14 | EVA02-L/14 CLIP |
| Fine-tuning resolution (height × width) | 336 × 168 | 224 × 224 |
| Inference resolution | 336 × 168 | 196 × 196 |
| Readout feature-cache backbone precision | BF16 autocast | Pure FP16 at 196 × 196 |
| Encoder probability weight | 0.5 | 0.5 |

The original-head probability weight is **0.25**; the robust-head weight is
**0.75**. Within the robust prediction, the teacher logit weight is **0.20**
on the selected attributes. Prior strength is **0.40**, global extra bias is
**0**, and the GPU microbatch is **32**. The teacher never runs at inference.
There is no flip augmentation, target adaptation, source routing or network
access in the submission runtime.

The independent source-head averaging, teacher logit mixing and original-head
probability mixing are different operations; their order is preserved in
[`scripts/robust_head_release.py`](scripts/robust_head_release.py).

## Environment

The recorded backbone training used **Python 3.12**, **PyTorch 2.11.0+cu128**,
and an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)**. Use a
Linux CUDA machine with sufficient memory for batch 64 fine-tuning. Reserve
approximately 100 GB of free working space in addition to images and optional
teacher weights; optimizer checkpoints and feature caches are large.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python tools/check_environment.py --strict --require-cuda
```

`--strict` checks Python and package versions; the report also records the
actual CUDA runtime. The historical training used CUDA 12.8; the packaging
workstation's current environment can differ. Preserve that distinction when
reporting reproducibility. The organizer inference environment is separate
from the training environment: the offline TorchScript models require CUDA
but do not import timm or Transformers. Historical PyTorch 2.4.1 CPU trace
compatibility checks are supplied; they do not certify organizer A10 execution.

## Data preparation

Obtain the official annotations and original images through the
[UPAR 2027 starter kit](https://github.com/speckean/UPAR-Challenge-2027) and
[competition page](https://www.codabench.org/competitions/18180/).
Keep the organizer image paths, including the PETA renaming/mapping procedure.
Your image root should contain `PETA/`, `Market1501/` and `PA100k/`.
Common nested PA100k release layouts are supported.

Run from this repository root:

```bash
python tools/prepare_data.py \
  --official-train /path/to/starterkit/data/annotations/task1/train/gt.csv \
  --official-val /path/to/starterkit/data/annotations/task1/val/gt.csv \
  --images-root /path/to/images
```

This reconstructs byte-identical CSVs using the committed ordered path lists,
checks their SHA256 hashes and recorded group separation, and links the image
directories under `data/`. It does not copy or modify original images.

| Source | Training | Tuning | In-domain audit |
|---|---:|---:|---:|
| PETA | 14,068 | 782 | 781 |
| Market1501 | 23,814 | 1,322 | 1,322 |
| PA100k | 80,088 | 4,450 | 4,449 |
| **Total** | **117,970** | **6,554** | **6,552** |

All 97,669 original training images remain in training, along with 20,301
public validation images. The remaining 13,106 images are partitioned into
tuning and audit sets. Known Market/PETA identities and exact-file duplicates
are grouped; incomplete identity metadata prevent a blanket identity-disjoint
claim. These public partitions were reused during development and are not a
historically untouched test set. Hidden images are never training inputs.

## Train from public pretrained weights

```bash
python tools/download_pretrained.py
python -m unittest discover -s tests -v
python tools/reproduce.py --plan
python tools/reproduce.py --run
```

The downloader pins Hugging Face revisions and verifies the encoder files
against the recorded SHA256 hashes. The pipeline checks split and teacher
hashes before execution. It trains PE and EVA sequentially and runs all steps
listed in [`configs/balanced.json`](configs/balanced.json):

```text
cache-pe → frozen-head-pe → finetune-pe → interpolate-pe
  → readout-cache-pe → readout-fit-pe
cache-eva → frozen-head-eva → finetune-eva → interpolate-eva
  → readout-cache-eva → readout-fit-eva
export-pe → export-eva → package → verify → evaluate
```

The frozen-head fit runs 25 epochs with seed 2092. Encoder fine-tuning runs
three epochs with seed 2093, batch 64, encoder/head learning rates
1e-5/2e-4, EMA and inverse-square-root source weights. The recorded best EMA
was epoch 2 for both encoders; a fresh run selects by the same tuning metric,
not by forcing that epoch. Source and teacher readouts run 12 epochs, batch
1,024, AdamW learning rate 0.003 and seed 0; source heads use Beta(0.4,0.4)
feature mixup.

Outputs:

```text
dist/UPAR_Track1_PE_EVA_Balanced_reproduced.zip
reports/local/reproduced_audit.json
reports/local/reproduced_audit.npz
logs/<stage>.log
artifacts/pipeline/<stage>.json
```

Each completed stage writes a receipt containing its command, code/input
hashes and output hashes. `--resume` skips only matching completed stages:

```bash
python tools/reproduce.py --run --resume
python tools/reproduce.py --from-stage readout-cache-pe --to-stage readout-fit-pe --run
```

An interrupted stage may leave partial outputs without a completion receipt.
The driver refuses to silently reuse or overwrite those. Move that stage's
partial output directory aside before rerunning it. For a long interrupted
encoder fit, the original training script supports `--resume <run>/resume.pt`;
its checkpoint restores optimizer and sampler position, but not all worker
augmentation RNG state. For the closest fresh-run comparison, restart that
stage. Changing batch size or precision constitutes a different training run.

## Restore and evaluate the original submission

The organizer handoff bundle is provided separately from Git because its model
artifacts total several GB. With the source repository already checked out:

```bash
python tools/restore_bundle.py --bundle /path/to/UPAR_2027_Track1_Organizer.tar
python tools/verify_release.py \
  --archive dist/UPAR_Track1_PE_EVA_Balanced_v18.zip --require-reference
python scripts/validate_submission.py \
  --archive dist/UPAR_Track1_PE_EVA_Balanced_v18.zip \
  --data data --annotations data/splits/train90_v1/audit.csv \
  --output reports/local/original_audit.json
```

To re-export the original trained checkpoints instead, use a fresh checkout
with prepared data, restore the bundle, and run:

```bash
python tools/reproduce.py --from-stage export-pe --run
python tools/compare_predictions.py \
  --candidate reports/local/reproduced_audit.npz --require-exact
```

The exporter checks FP16 probability agreement with the eager model. The
archive verifier checks every member hash and the complete inference recipe.
The evaluator checks empty batches, finite outputs, attribute reordering,
metadata independence and batches spanning internal boundaries, then scores
all requested images. Use `--limit 32` on the evaluator for an initial smoke
check; omit it for the full audit.
The committed reference predictions permit an independent per-image comparison.
For a fresh retraining run, omit `--require-exact` and inspect the differences
alongside the aggregate metrics.

## Reference results

The **original archive**, evaluated on the 6,552-image in-domain audit, has:

| Source-macro metric | Value (%) |
|---|---:|
| Balanced accuracy (mA) | 91.5111 |
| Instance F1 | 88.6081 |
| Mean per-source harmonic score | 90.0357 |
| Harmonic score with zero-filled unsupported polarities | 87.2846 |

Metrics use strict probability > 0.5 decisions and equal weighting of the
three sources. Instance F1 is the harmonic mean of aggregate instance
precision and recall. The default local mA averages supported polarities;
the alternative zero-filled result is reported separately because private
scorer parity for unsupported polarities is unverified.

These are local in-domain results, not a hidden leaderboard score. The paper's
84,394-image source-excluded audit uses separately trained fold checkpoints
and must not be reproduced by evaluating this all-source model on those
images. This repository focuses on reproducing the selected deployed predictor.

Historical matched local timing was 7.987 seconds for 4,096 images (512.8
images/s), excluding loading, on the RTX PRO 6000. Peak allocated GPU memory
was 1,372 MiB. These measurements do not establish A10 end-to-end timing.
See [`reference/`](reference/) for the original evidence and
[`reports/verification.json`](reports/verification.json) for handoff checks.

## Teacher regeneration (optional)

Use the included targets for the reference recipe. To investigate regenerated
targets, create a separate environment with `requirements-teacher.txt`, prepare
data as above, download the pinned teacher, and write a **new** output file:

```bash
python tools/download_pretrained.py --teacher
python scripts/vlm_attributes.py \
  --csv data/teacher.csv --attributes all --min-side 224 --images-per-batch 1 \
  --model artifacts/pretrained/qwen3_vl_8b \
  --output artifacts/teacher_regenerated.npz
```

Prompts and score extraction are in `scripts/vlm_attributes.py`. The target is
the first-token Yes-minus-No logit, converted to a sigmoid soft target by the
student fit. Regeneration may differ numerically across inference stacks;
the default pipeline deliberately verifies and uses the original cached
targets rather than silently replacing them.

## Repository map and attribution

- `configs/`: fixed training recipe and pinned foundation-model sources.
- `scripts/`: training, feature caching, readouts, export and evaluation.
- `tools/`: input checks, staged driver, interpolation and packaging.
- `runtime/`: exact submitted inference code and upstream model licenses.
- `splits/`: ordered path assignments and recorded grouping evidence.
- `teacher/`: original training-only cached targets.
- `reference/`: original configurations, checksums and local results.
- `tests/`: CPU input, mixture and configuration checks.
- `docs/REPRODUCIBILITY.md`: verification scope and practical limitations.

Foundation models: [PE](https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb),
[EVA](https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b), and
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
PE's Apache-2.0 and EVA's MIT license texts are retained in `runtime/`.
Dataset and foundation-model terms continue to apply; original dataset images
are not redistributed here. The project code and trained artifacts are
provided for organizer verification; no additional blanket license over
third-party material is implied. Contact the repository owner through GitHub
for the organizer artifact bundle or questions about the recorded recipe.
