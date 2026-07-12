#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/smoke.yaml}
WORKERS=${WORKERS:-1}

rotation-patterns make-manifests --config "$CONFIG"

find "$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$CONFIG")/manifests" \
  -name '*.csv' -print0 | while IFS= read -r -d '' manifest; do
    jobs=$(( $(wc -l < "$manifest") - 1 ))
    seq 0 $((jobs - 1)) | xargs -P "$WORKERS" -I{} \
      rotation-patterns run-task --config "$CONFIG" --manifest "$manifest" --index {}
  done

