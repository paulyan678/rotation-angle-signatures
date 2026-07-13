# Methods and declared assumptions

I use this document to separate settings stated directly in the workshop paper from the
implementation choices I formalized for this research repository. I centralize every
choice in `configs/paper.yaml` so the protocol can be inspected and changed without editing
the experiment code.

## Main 16×16 experiment

| Item | Workshop paper | Research configuration in this repository |
|---|---|---|
| Angles | `0.0° … 359.9°`, step `0.1°` | Exact decimal grid; 3,600 values |
| Initialization | ImageNet-pretrained; reset every run | Fresh timm ImageNet weights for each job |
| Other augmentation | None | None |
| Resize | Not specified | 224×224, no crop, no expansion |
| Rotation | Not specified | centered bilinear interpolation, zero fill, fixed canvas |
| Normalization | Not specified | ImageNet RGB; mean/std 0.5 for one/four channels |
| SimCLR defaults | Not enumerated | 200 epochs, SGD 0.03, temperature 0.2, 128-D projection |
| MoCo v2 defaults | Not enumerated | same schedule, queue 65,536, EMA 0.999, temperature 0.2 |
| Projection head | Not specified | 2-layer MLP, hidden 2,048, output 128 |
| Probe architecture | “six-layer ReLU MLP,” no widths | five hidden layers `[1024,512,256,128,64]` + output |
| Probe training | SGD 0.015, batch 64, 20 epochs, 80/20 | as stated; momentum 0.9, no decay |
| Pair split | Not specified | group split by source image to avoid leakage |
| Negative image | random different image | deterministic derangement |
| Repeats/seeds | Not specified | seed 2025; configurable |

The paper refers to “default hyperparameters,” but SimCLR and MoCo do not have one shared
default that applies to eight heterogeneous backbones and 16 datasets. I therefore expose
the exact values above as versioned research configuration rather than hiding them inside
training code.

## Prediction experiment

I use the hard splits from Appendix C: the first ten curves on the opposite axis for
optimization and the last six for evaluation. Cosine is ordinary cosine similarity. I
convert “L2 distance” to negative Euclidean distance so higher logits mean more similar;
using raw L2 with the paper's `argmax` would select the least similar class.

The paper does not state what observations form each Figure 2 violin. I run 100 explicit
evaluation repetitions, each containing the stated 100 held-out draws, and plot those
repetition accuracies. I save per-class and overall means as machine-readable data. My
configured shared-weight interpretation produces means different from the labels printed
in the workshop paper, so I retain both series in `summary.json` as an explicit result and
methodological limitation.

## Segmentation experiment

| Item | Workshop paper | Research configuration in this repository |
|---|---|---|
| Angles | Integer `0° … 360°` | 361 independent values, including both endpoints |
| Datasets/encoders | 3 datasets × ConvNeXt-Tiny, ViT-B/16, ResNet-50 | same order |
| Model | “conventional” U-Net, skips unspecified | timm four-stage features + explicit bilinear decoder |
| Objective | soft Dice | differentiable soft Dice, epsilon `1e-6` |
| Optimizer | Adam | Adam 1e-4, weight decay 1e-5 |
| Split | validation + held-out set, fractions absent | fixed 70/10/20 item- or case-level split |
| Early stopping | stated, details absent | validation Dice, patience 10 |
| BraTS input | table says four channels | FLAIR, T1, T1ce, T2 middle axial slice |
| BraTS target | not specified | labels `{0,1,2,4}` mapped to `{0,1,2,3}` |
| Dice reduction | “mean Dice,” details absent | mean of per-item macro class Dice, including background; binary threshold 0.5 |

Section 2.3 describes U-Net prediction Dice, while the abstract, Figure 3 caption, and
Appendix F refer to correspondence between saliency maps and masks. I implement the more
detailed Section 2.3 U-Net protocol and label the generated Figure 3 output as held-out
U-Net Dice.

## HoG/SVM experiment

The paper reports 64×64 HOG cells and a cross-validated Gaussian SVM but does not enumerate
the remaining settings. I use 9 orientations, 1×1 cells per block, L2-Hys block
normalization, RGB-to-grayscale conversion, and the C/gamma grid declared in the config. I
keep the original and rotated view of each image in the same train/test group and reserve
20% of source-image groups for held-out testing.

## Research artifacts

- I include all 256 original Figure 1 curves as the checksummed, lossless
  `figure1_curves.npz` artifact; `fetch-reference` validates and stages it locally.
- I run the prediction implementation on the actual 16×16×3,600 curve tensor and save all
  repetition-level results plus summary comparisons.
- I generate Figure 3 tables through the configured MoCo-to-U-Net and HoG pipelines.
- I record encoder transfer coverage, losses, held-out metrics, configuration fingerprints,
  split digests, software versions, and timings in each result.

## Plotting policy

I do not add undisclosed smoothing to newly generated curves. I plot raw computed points
and retain the configuration, split, timing, software, and source provenance for every
job.
