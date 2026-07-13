# Response Patterns to Rotation Angle in Self-Supervised Learning

[![CI](https://github.com/paulyan678/rotation-angle-signatures/actions/workflows/ci.yml/badge.svg)](https://github.com/paulyan678/rotation-angle-signatures/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b)](https://openreview.net/forum?id=85rdlgTS50)

This repository contains the complete executable research code and experiment toolkit for
my University of Toronto thesis research project:

> **Response Patterns to Rotation Angle in a Rotation Pretext Task Vary Across Datasets
> and Architectures: An Observation and a Negative Result**

I conducted this work with Amy Saranchuk and Michael Guerzhoy. We presented it at the
NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations
([paper and reviews](https://openreview.net/forum?id=85rdlgTS50)).

## About my research

I study how the fixed rotation angle used during contrastive self-supervised pretraining
interacts with the image dataset and encoder architecture. Across 16 datasets, eight
encoders, and two contrastive methods (MoCo v2 and SimCLR), I observe distinct periodic
accuracy-versus-angle curves that act like dataset-and-architecture signatures.

I also investigate two follow-up questions:

- Can a classifier identify the dataset and encoder-method pair from its rotation-response
  curve?
- Can orientation-sensitive Histogram-of-Gradients features explain the oscillations seen
  in medical image segmentation?

I organize the project around four connected experiments:

1. A 16×16 fixed-angle contrastive pretraining and downstream classification grid.
2. Row and column prediction from the resulting signature curves.
3. A 3×3 MoCo-to-U-Net medical segmentation study.
4. A grouped HoG/RBF-SVM shortcut test.

I provide executable workflows for every experiment, with local and Slurm orchestration,
aggregation, provenance capture, and generation of Figures 1–4 from the corresponding
machine-readable results.

## Main findings

- Rotation-angle performance is periodic and non-monotonic rather than smoothly degrading
  with larger rotations.
- The response pattern changes systematically across datasets, encoder architectures, and
  contrastive methods.
- The published curves contain enough structure to identify their originating dataset and
  encoder-method setting, motivating the signature-prediction experiment.
- Medical segmentation performance also oscillates with the pretraining angle.
- My HoG/SVM experiment does not support the hypothesis that simple orientation-sensitive
  gradient shortcuts explain the full phenomenon.

## Quick start

```bash
git clone https://github.com/paulyan678/rotation-angle-signatures.git
cd rotation-angle-signatures

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,medical]'
```

Python 3.10+ is required. I strongly recommend a CUDA GPU for any non-smoke training.
For the exact environment I used to validate the project, install the lock before the
package:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## Explore my published Figure 1 and curve analysis

I archive the 256 original Figure 1 curves (921,600 measurements) at an immutable commit in
[`paulyan678/thesis`](https://github.com/paulyan678/thesis). This project's data command
fetches only those experimental CSVs, validates the full 16×16×3,600 tensor, and keeps the
published measurements separate from newly generated runs.

```bash
# Fetch and validate my published curve data.
rotation-patterns fetch-reference --config configs/paper.yaml

# Generate the 16×16 accuracy-versus-angle grid.
rotation-patterns figures --config configs/paper.yaml --only figure1

# Run the Appendix C signature-prediction analysis and render its figure.
rotation-patterns prediction --config configs/paper.yaml
rotation-patterns figures --config configs/paper.yaml --only figure2
```

I write the generated artifacts to `outputs/paper/figures/` and
`outputs/paper/prediction/`.

## Validate the full training pipeline

```bash
pytest
rotation-patterns smoke --config configs/smoke.yaml
```

I use the smoke profile to exercise MoCo v2, SimCLR, the frozen six-layer downstream
probe, MoCo-to-U-Net transfer, and the HoG/SVM baseline on small deterministic fixtures.
These smoke metrics validate the implementation; they are not substitutes for the full
paper-scale results.

## Run the full research grid

First read my [dataset setup guide](docs/DATASETS.md), place manually licensed datasets
under `data/`, and inspect the fully specified research configuration in
`configs/paper.yaml`.

```bash
# Report the exact number of jobs and model fits.
rotation-patterns estimate --config configs/paper.yaml

# Build resumable, configuration-fingerprinted job manifests.
rotation-patterns make-manifests --config configs/paper.yaml --shard-size 10000
```

Each manifest row is one independent, idempotent experiment. Run one row locally with:

```bash
rotation-patterns run-task \
  --config configs/paper.yaml \
  --manifest outputs/paper/manifests/classification-0000.csv \
  --index 0
```

For Slurm, submit each manifest as an array:

```bash
mkdir -p logs
MANIFEST=outputs/paper/manifests/classification-0000.csv
N=$(($(wc -l < "$MANIFEST") - 2))
sbatch --array=0-"$N" scripts/slurm_array.sh configs/paper.yaml "$MANIFEST"
```

I store each completed point as an atomic JSON result. Re-running a completed point skips
it unless `--force` is used. After the jobs finish, aggregate the results and generate the
paper figures:

```bash
rotation-patterns aggregate --config configs/paper.yaml --experiment classification
rotation-patterns aggregate --config configs/paper.yaml --experiment segmentation
rotation-patterns aggregate --config configs/paper.yaml --experiment hog
rotation-patterns figures --config configs/paper.yaml
```

## Experiment map

| Research result | Code path | Generated output |
|---|---|---|
| Figure 1: 16×16 accuracy curves | fixed-angle SimCLR/MoCo v2 + frozen six-layer probe | `aggregate/classification.csv`, `figures/figure1.*` |
| Figure 2: signature prediction | Appendix C held-out-axis weighted similarity classifier | `prediction/trials.csv`, `figures/figure2.*` |
| Figure 3: 3×3 Dice + HoG | MoCo → U-Net and grouped HoG/RBF-SVM | `aggregate/{segmentation,hog}.csv`, `figures/figure3.*` |
| Figure 4: qualitative BraTS panel | stored test prediction at the configured example angle | `figures/figure4.*` |

## Reproducibility design

I designed the project so large sweeps can be inspected and resumed safely:

- The unrotated and fixed-angle views are the only contrastive augmentation.
- Every paper-profile angle starts from a freshly constructed ImageNet-initialized encoder.
- Dataset splits, negative pairs, and job IDs are deterministic.
- I group medical and HoG splits by source item to prevent paired-view leakage.
- I keep model-selection validation data separate from the segmentation test set.
- I retain both `0°` and `360°` segmentation jobs because the paper reports both.
- I fingerprint every job with the full configuration, explicit data revision, and protocol
  version.
- I write results atomically and record software, hardware, source, timing, and split
  provenance.

Some implementation choices were not fully specified in the workshop paper. I make every
such choice explicit in [Methods and assumptions](docs/METHODS_AND_ASSUMPTIONS.md). For
example, I define the Appendix C L2 score as negative Euclidean distance because the
prediction rule takes an `argmax`. Because Appendix C also leaves the weight-fitting scope
and violin repetitions underspecified, I record both the published means and the output of
my explicit shared-weight interpretation in `summary.json`.

For job identity, deterministic execution, checkpoint policy, and validation tiers, see
[My reproducibility and execution model](docs/REPRODUCIBILITY.md).

## Repository layout

```text
configs/                 paper and smoke experiment profiles
docs/                    dataset, methods, and reproducibility documentation
scripts/                 local and Slurm launchers
src/rotation_patterns/   datasets, models, experiments, analysis, and figures
tests/                   deterministic unit and integration tests
outputs/                 generated artifacts (gitignored)
```

## License and citation

I release the code under the [MIT License](LICENSE). Dataset and pretrained-weight
licenses remain with their respective owners. I fetch my original curve measurements from
their pinned research archive rather than duplicating them here.

If you use this project, please cite the paper and software using [CITATION.cff](CITATION.cff).
