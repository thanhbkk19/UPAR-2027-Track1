# Reproducibility scope

## What the handoff preserves

The target is the selected Balanced predictor, identified by the original ZIP
SHA256 in README.md. The shipped runtime is byte-identical to its `run.py`.
Ordered split assignments reconstruct the historical CSVs exactly. The public
foundation models are pinned by revision and file hash; the original Qwen
targets are included and checked against their recorded hash. Historical
configuration files are included without replacing them with inferred defaults.

The source-specific fitting, export, feature-pooling, metrics and runtime code
come from the saved release snapshot. The fine-tuning and frozen-head/cache
entry points are the available project implementation, with the Balanced
arguments explicitly recorded in `configs/balanced.json`. The sampler and
atomic checkpoint writer were extracted from the project's distillation
module so the backbone trainer does not import an unrelated compact model.
`reference/source_provenance.json` records source-file hashes and these changes.

## Numerical details that matter

- The initial WiSE head is supervised, not a zero-shot textual classifier.
- PE uses alpha 0.80; the deployed EVA uses alpha 0.75. The paper's fold EVA
  alpha 0.70 is not the release setting.
- PE committee features were cached with BF16 autocast at 336 × 168; EVA
  committee features use pure FP16 at 196 × 196 after positional resampling.
- Inference casts backbone weights and normalization buffers to FP16; regional
  feature normalization and classifiers remain FP32.
- The three source classifiers vote in probability space. Teacher supervision
  is mixed in logit space only for the stored 16 attribute columns. The original
  classifier is then mixed in probability space.
- Encoder probabilities are averaged before subtracting the prior offset.
  The released offset values retain historical float32 rounding from removing
  the prior extra bias; packaging verifies them against the training prior.
- JIT optimized execution is disabled in the runtime to avoid warmup-dependent
  fusion differences. CUDA inference is required by the shipped FP16 traces.

## Verification and its limits

The handoff checks reconstruct all split CSVs, validate all image paths,
check teacher rows are training-only, test the mixture against an explicit
calculation, reconstruct both interpolated encoders, refit every source and
teacher readout from the original feature caches, re-export both original
trained checkpoints, and evaluate the rebuilt archive on the full 6,552-image
in-domain audit. The reconstructed interpolation and all refitted readout
tensors match their originals exactly; all audit probabilities also match
exactly. The machine-readable
result is `reports/verification.json`.

A full fresh encoder-training run was **not** repeated during handoff
preparation. The existing GPU was running another experiment. A successful
checkpoint re-export and audit establish the inference reconstruction path;
they do not prove a new stochastic training run will reproduce every parameter.
The readout refit uses the saved feature caches; it does not establish that
newly fine-tuned encoders will generate identical features.
The current validation environment is recorded separately from the historical
training environment in `reports/verification.json`.

The driver supplies all fixed training commands and checks their inputs.
No test result here should be described as a successful full retraining unless
that complete run has actually been executed. Future results should be saved
under `reports/local/` and compared with the reference audit on identical rows.

Public tuning and audit data were reused during development. Grouping uses
available identities and duplicate hashes, with incomplete identity coverage
for PA100k and some PETA subsets. Public-validation use is part of the release
training recipe. The all-source release must not be presented as a
source-excluded model on images from those training sources.

## Checkpoint handoff

`UPAR_2027_Track1_Organizer.tar` contains `source/` plus these reference artifacts:

```text
dist/UPAR_Track1_PE_EVA_Balanced_v18.zip
artifacts/robust_heads_v15/pe_release.pt
artifacts/fast_robust_v17/eva_release.pt
```

Hashes are in `reference/checkpoint_hashes.json`; a SHA256 sidecar accompanies
the TAR. To obtain source without Git, extract only `source/`, enter that
directory, then use `tools/restore_bundle.py --bundle /absolute/path/to/bundle`.
The restoration tool accepts only these three regular-file paths and verifies
each file before installing it. Dataset images and generic pretrained weights
are intentionally supplied through their original sources.

The `.pt` training checkpoints contain trusted project metadata in addition to
tensors and are loaded with `weights_only=False`. Use only the supplied,
hash-verified checkpoints or checkpoints you produced yourself.
