# Paper gaps and implementation assumptions

This document separates statements in the paper from choices required to make executable
code. These assumptions are centralized in `configs/paper.yaml`; changing them does not
require code edits.

## Main 16×16 experiment

| Item | Paper | This implementation |
|---|---|---|
| Angles | `0.0° … 359.9°`, step `0.1°` | Exact decimal grid; 3,600 values |
| Initialization | ImageNet-pretrained; reset every run | Fresh timm ImageNet weights for each job |
| Other augmentation | None | None |
| Resize | Not reported | 224×224, no crop, no expansion |
| Rotation | Not reported | centered bilinear interpolation, zero fill, fixed canvas |
| Normalization | Not reported | ImageNet RGB; mean/std 0.5 for one/four channels |
| SimCLR defaults | Not reported | 200 epochs, SGD 0.03, temperature 0.2, 128-D projection |
| MoCo v2 defaults | Not reported | same schedule, queue 65,536, EMA 0.999, temperature 0.2 |
| Projection head | Not reported | 2-layer MLP, hidden 2,048, output 128 |
| Probe architecture | “six-layer ReLU MLP,” no widths | five hidden layers `[1024,512,256,128,64]` + output |
| Probe training | SGD 0.015, batch 64, 20 epochs, 80/20 | As stated; momentum 0.9, no decay |
| Pair split | Not reported | group split by source image to avoid leakage |
| Negative image | random different image | deterministic derangement |
| Repeats/seeds | Not reported | seed 2025; configurable |

The paper says “default hyperparameters,” but neither SimCLR nor MoCo has one default that
applies to eight heterogeneous backbones and 16 datasets. The values above are explicit
reconstruction choices, not recovered facts.

## Prediction experiment

The implementation uses the exact hard splits from Appendix C: first ten curves on the
opposite axis for optimization and the last six for evaluation. Cosine is ordinary cosine
similarity. “L2 distance” is converted to negative Euclidean distance so higher logits mean
more similar; raw L2 with the paper's `argmax` would select the least similar class.

The paper does not explain what observations form each Figure 2 violin. This repository
runs 100 explicit evaluation repetitions, each containing the stated 100 held-out draws,
and plots those repetition accuracies. It also saves per-class and overall means. This
coherent shared-weight reconstruction does **not** reproduce the red means in the paper;
the machine-readable summary includes the published labels and the discrepancy.

## Segmentation experiment

| Item | Paper | This implementation |
|---|---|---|
| Angles | Integer `0° … 360°` | 361 independent values, including both endpoints |
| Datasets/encoders | 3 datasets × ConvNeXt-Tiny, ViT-B/16, ResNet-50 | Same order |
| Model | “conventional” U-Net, skips unspecified | timm four-stage features + explicit bilinear decoder |
| Objective | soft Dice | differentiable soft Dice, epsilon `1e-6` |
| Optimizer | Adam | Adam 1e-4, weight decay 1e-5 |
| Split | validation + held-out set, fractions absent | fixed 70/10/20 item- or case-level split |
| Early stopping | stated, details absent | validation Dice, patience 10 |
| BraTS input | table says four channels | FLAIR, T1, T1ce, T2 middle axial slice |
| BraTS target | not reported | labels `{0,1,2,4}` mapped to `{0,1,2,3}` |
| Dice reduction | “mean Dice,” details absent | mean of per-item macro class Dice, including background; binary threshold 0.5 |

There is an unresolved contradiction: Section 2.3 describes U-Net prediction Dice, while
the abstract, Figure 3 caption, and Appendix F describe Dice between saliency maps and
masks. This repository implements the more detailed Section 2.3 U-Net protocol and labels
the result accordingly.

## HoG/SVM experiment

The paper reports 64×64 HOG cells and a cross-validated Gaussian SVM, but no other HOG or
SVM settings. This implementation uses 9 orientations, 1×1 cells per block, L2-Hys block
normalization, RGB-to-grayscale conversion, and a documented C/gamma grid. It groups the
original and rotated view of an image into the same train/test fold and reserves 20% of
source-image groups for held-out testing.

## Published artifacts

- All 256 Figure 1 curve CSVs exist in `paulyan678/thesis` commit `bf9d887337`.
- The legacy prediction scripts in that repository use random placeholder tensors, not the
  curve CSVs; they are therefore not reused.
- No raw Figure 3 values, checkpoints, per-run logs, split manifests, or complete training
  driver were found in the authors' related public repositories.
- The earlier medical repositories are useful prototypes but contain architecture-transfer,
  input-size, loss, and evaluation issues; the new implementation does not copy those paths.

## Interpretation caution

The Figure 1 curves and Figure 3 curves are unusually smooth given independent stochastic
training at every adjacent angle. The repository does not invent undisclosed smoothing.
Newly computed raw points are plotted as produced, with provenance retained per job.
