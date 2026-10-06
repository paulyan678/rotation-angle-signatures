# Published results and the current reconstruction

The 2025 workshop paper is the historical research result. The current executable
Appendix C interpretation produces different classification accuracies. Do not
present a current smoke test or regenerated Figure 2 as a replication of the
published numbers.

## One-command current analysis

```bash
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e .
python -m rotation_patterns.analysis_receipt --output outputs/current-analysis
```

This uses the paper configuration and checked Figure 1 tensor, without training
or downloading models/images. The output contains all trial rows, summary,
resolved config, and an immutable receipt with code hashes, Git revision/dirty
state, environment, reference checksums, axis splits and output hashes. Choose a
new output directory. `results/current-analysis/` is a committed execution
receipt, not an original-paper artifact; its README explains reproduction.

The current baseline macro accuracies are:

| Target / score | Current interpretation | Published labels recorded in code |
|---|---:|---:|
| Dataset / cosine | 0.13546875 | 0.56375 |
| Dataset / negative L2 | 0.11428750 | 0.609375 |
| Encoder-method / cosine | 0.05163750 | 0.64000 |
| Encoder-method / negative L2 | 0.29113750 | 0.60125 |

The published column is transcribed historical labels, not values recalculated
from original trials. Its uniform-guess reference is 1/16, or 0.0625. Tests
protect the explicit current interpretation, including an independently derived
small similarity example, shared-weight optimization, split isolation, and the
full bundled-data baseline. They do not assert the published values are reproduced.

## Why the mismatch remains unresolved

Targeted history inspection found a prediction scaffold (`2961fcf`), followed by
the explicit shared-weight reconstruction (`4d1f84f`); that first executable
version already discloses the mismatch. The later bundled-data commit (`e539667`)
contains the current reference curves, not original classifier trial outputs or
trained weights. No original trial-level classifier archive was recovered.

Negative Euclidean distance makes larger scores represent greater similarity;
a single shared class-weight vector avoids selecting weights using the unknown
query label. Evaluation uses labeled leave-one-curve-out reference pools inside
the held-out opposite-axis split. The original fitting scope and violin sampling
definitions are not recoverable from available artifacts. A correction needs the
original implementation and trials or a clearly labeled new experiment. Do not
tune hyperparameters to force agreement with printed labels.

## Distinguish the protocols

Classification SSL currently trains on all source images before partitioning
images for the downstream probe. This is **transductive pretraining**: the probe
does not fit on held-out source pairs, but the encoder has seen their unlabeled
images. Segmentation SSL trains only on its training subset. These are different
protocols; preserve and state them rather than quietly changing a scientific
method to improve a score.

The real-image grid and medical HoG/segmentation results have not been rerun in
this maintenance change. Before the million-job sweep, run a bounded pilot with
public datasets, sentinel angles, representative encoders, multiple seeds, split
manifests, transfer-coverage checks, and explicit runtime/storage estimates.

`requirements-ci.txt` pins the newly tested runtime. The existing
`requirements-lock.txt` remains the older validation environment. Explicit lint
rules prevent a new Ruff default from silently changing the quality gate.
