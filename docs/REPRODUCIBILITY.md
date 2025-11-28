# My reproducibility and execution model

## Job cardinality

My paper profile expands to:

- classification SSL + probe: `16 × 8 × 2 × 3,600 = 921,600` jobs;
- segmentation SSL + U-Net: `3 × 3 × 361 = 3,249` jobs;
- HoG/SVM: `3 × 361 = 1,083` jobs.

Because classification and segmentation jobs each contain two model fits, my full profile
performs 1,849,698 neural fits. I provide a CLI estimator that reports exact configured job
and fit counts before a run is scheduled.

## Determinism

I derive each manifest row's stable SHA-256 ID from the task, parsed configuration,
explicit `data_revision`, and implementation protocol version. Each row reconstructs the
model from its declared initialization, creates deterministic splits and negative pairs,
and writes one JSON file atomically. I record the Git commit, Python, PyTorch, accelerator,
metric name, and runtime. Bump the package/protocol version whenever scientific behavior
changes.

I request PyTorch deterministic algorithms with warnings enabled because a small number of
accelerator kernels do not have deterministic implementations on every release. For strict
bitwise comparison, use the same hardware, driver, Python environment, and worker count.

## Resume and failure behavior

I treat a validated existing result file as complete and skip it. I replace partial
temporary files atomically so they are never mistaken for completed rows. `--force`
explicitly recomputes a row. I keep cluster logs external to the metric store.

I keep checkpoints transient by default because retaining the encoder, momentum encoder,
and optimizer for every classification point would require hundreds of terabytes. The
result JSON is the durable artifact. Modify the runners if checkpoint retention is needed
for selected sentinel angles.

## Validation levels

1. I use `pytest` to check grids, manifests, transforms, losses, pair construction, result
   schemas, and the reference tensor.
2. I use `configs/smoke.yaml` to execute all three training paths on deterministic fixtures.
3. I recommend a pilot with several real public datasets, 15° steps, two backbones, both
   methods, and at least three seeds.
4. Schedule the full profile only after reviewing pilot loss curves, storage, throughput,
   and cost.

For an extended statistical replication, I recommend at least three seeds and comparison
by RMSE, rank correlation, and periodogram peaks with confidence intervals. I do not treat
visual resemblance to the published plots as a sufficient acceptance criterion.
