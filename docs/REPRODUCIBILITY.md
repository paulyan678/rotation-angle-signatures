# Reproducibility and execution model

## Job cardinality

The paper profile expands to:

- classification SSL + probe: `16 × 8 × 2 × 3,600 = 921,600` jobs;
- segmentation SSL + U-Net: `3 × 3 × 361 = 3,249` jobs;
- HoG/SVM: `3 × 361 = 1,083` jobs.

Because classification and segmentation jobs each contain two model fits, the full profile
performs 1,849,698 neural fits. The CLI estimator reports exact configured job and fit
counts rather than hiding the scale.

## Determinism

Each manifest row includes a stable SHA-256-derived ID over the task, parsed configuration,
explicit `data_revision`, and implementation protocol version. A row reconstructs the
model from its declared initialization, creates deterministic splits and negative pairs,
and writes one JSON file atomically. JSON provenance includes the Git commit, Python,
PyTorch, accelerator, metric name, and runtime. Bump the package/protocol version whenever
scientific behavior changes.

PyTorch deterministic algorithms are requested with warnings enabled because a small
number of accelerator kernels do not have deterministic implementations on every release.
For strict bitwise comparison, use the same hardware, driver, Python environment, and
worker count.

## Resume and failure behavior

An existing result file is treated as complete and skipped. Partial temporary files are
replaced atomically and never mistaken for completed rows. `--force` explicitly recomputes
a row. Cluster logs remain external to the metric store.

Checkpoints are transient by default because retaining the encoder, momentum encoder, and
optimizer for every classification point would require hundreds of terabytes. Result JSON
is the durable artifact. Modify the runners if checkpoint retention is scientifically
required for selected sentinel angles.

## Validation levels

1. `pytest` checks grids, manifests, transforms, losses, pair construction, result schemas,
   and the reference tensor.
2. `configs/smoke.yaml` executes all three training paths on deterministic fixtures.
3. A recommended pilot uses several real public datasets, 15° steps, two backbones, both
   methods, and at least three seeds.
4. The full profile should only be scheduled after pilot loss curves, storage, throughput,
   and cost are reviewed.

For a statistical replication, use at least three seeds and compare curves using RMSE,
rank correlation, and periodogram peaks with confidence intervals. Visual resemblance to
the published plots is not a sufficient acceptance criterion.
