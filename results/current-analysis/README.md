# Current executable analysis — October 6, 2026

These are newly generated results from the explicit shared-weight classifier in
this repository, using the bundled 256 Figure 1 curves. They are **not recovered
original paper trials** and do not reproduce the published Figure 2 labels.

| Target / score | Current macro accuracy | Published label |
|---|---:|---:|
| Dataset / cosine | 0.13546875 | 0.56375 |
| Dataset / negative L2 | 0.11428750 | 0.609375 |
| Encoder-method / cosine | 0.05163750 | 0.64000 |
| Encoder-method / negative L2 | 0.29113750 | 0.60125 |

Execution used source commit `38aa536` with a clean working tree, Python 3.12,
and the package versions recorded in [receipt.json](receipt.json). The receipt
contains full source-file and artifact hashes, the reference tensor checksums,
resolved configuration identity, and the exact train/test axis indices.
The CSV retains all 6,400 trial/class records; the JSON contains per-class
summaries and comparisons. The printed historical labels remain unverified
against original trial data.

Reproduce from the repository root in a fresh directory:

```bash
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e .
python -m rotation_patterns.analysis_receipt --output outputs/current-analysis
```

The files here are byte-for-byte copies of the recorded output files; the
staged copy of the bundled reference tensor is omitted to avoid duplicating it.
Output paths and Git/environment fields can differ on a new machine; the
scientific macro accuracies above are protected by regression tests. No
source-image training, real medical-data experiment, or new Figure 3 result is
represented by this artifact. See [analysis decisions](../../docs/ANALYSIS_RECEIPTS.md)
for methodology, transductive pretraining, history inspection, and unresolved gaps.
