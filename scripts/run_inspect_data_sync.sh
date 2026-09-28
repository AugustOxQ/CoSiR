#! /bin/bash
set -uo pipefail
echo "=== /local/wding/pre_extract ==="
find /local/wding/pre_extract -maxdepth 3 2>&1
echo "=== /local/wding/Dataset/artelingo ==="
find /local/wding/Dataset/artelingo -maxdepth 2 2>&1
echo "=== disk free ==="
df -h /local/wding 2>&1
