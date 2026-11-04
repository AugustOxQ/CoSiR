#!/usr/bin/env bash
# r18: the TPIPS stores on the local 3090. Gate (iv) failed on node402 (lowest row cosine 0.9987 with identical packages
# and image bytes: round log, build 4), so the fallback of ticket 04 applies. 6 shards of about 2.5 h: one res-queue job
# per shard, so other chats' GPU jobs run between shards. A rerun skips finished chunks.
#   bash src/test/20261103_outside_methods/om_tpips_local.sh <first shard> <last shard>        (0 to 5; run it detached)
# After all 6, in the tpips venv, then gate (iv) against the local rerun:
#   om_tpips.py merge --set artelingo --factor all --of 6 ; om_tpips.py merge --set mmart --factor all --of 6
#   om_tpips.py gate-iv --path local=$OUT/stage0/probe_b1_rerun
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=/root/venvs/tpips/bin/python
T=src/test/20261103_outside_methods/om_tpips.py
IN=/data/SSD2/cosir_r18/20261103_outside_methods/outputs/tpips_inputs
for i in $(seq "$1" "$2"); do
    echo "shard $i start $(TZ=Europe/Amsterdam date '+%F %H:%M')"
    res-queue gpu --est 2h30m --label "r18:tpips_local$i" -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        bash -c "for s in artelingo mmart; do $PY $T extract --set \$s --factor all --shard $i --of 6 --batch 1 --device cuda --manifest $IN/\$s.json; done"
    echo "shard $i end $(TZ=Europe/Amsterdam date '+%F %H:%M')"
done
