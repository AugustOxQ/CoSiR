#! /bin/bash
set -uo pipefail
echo "=== python3 os.path.exists checks ==="
python3 - <<'PY'
import os
for p in (
    "/local/wding/pre_extract/artelingo/features",
    "/local/wding/pre_extract/artelingo/features/metadata.json",
    "/local/wding/Dataset/artelingo",
    "/local/wding/Dataset/artelingo/artelingo_train.json",
    "/local/wding/res/CoSiR_Experiment/artelingo",
):
    print(p, os.path.exists(p))
PY
echo "=== hostname ==="
hostname
echo "=== find (depth-limited) ==="
find /local/wding -maxdepth 2 2>&1
