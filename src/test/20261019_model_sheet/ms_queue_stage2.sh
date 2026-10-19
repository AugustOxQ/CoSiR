#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/ms_queue_common.sh"
stage=2
status=queue_stage2_status.txt
printf 'MS_STAGE2_START %s\n' "$(date -Iseconds)" >> "$status"
# Regenerate only from complete canonical seed-0 heads; never trust an older pick list.
if ! step manifest cpu --pick-seeds; then
    finish
fi
# Validate before process substitution, whose exit status bash would otherwise hide.
if step validate_manifest env CUDA_VISIBLE_DEVICES='' "$python_bin" -c '
import json
from ms_rows import ROWS
r=json.load(open("outputs/seed_rows.json"))
assert r["seeds"] == [1,2] and r["rows"] and len(set(r["rows"])) == len(r["rows"])
assert all(n in ROWS for n in r["rows"])
'; then
    mapfile -t rows < <(env CUDA_VISIBLE_DEVICES='' "$python_bin" -c 'import json; print("\n".join(json.load(open("outputs/seed_rows.json"))["rows"]))')
    for row in "${rows[@]}"; do
        step "train_${row}" gpu --rows "$row" --seeds 1 2
    done
fi
step score012 cpu --score --sheet
if step d17 cpu --d17; then
    # One subprocess owns parsing and the argument vector: no shell eval of overrides.
    step rerun env CUDA_VISIBLE_DEVICES=0 "$python_bin" -c '
import json, subprocess, sys
r=json.load(open("outputs/d17.json"))
if r == "none":
    print("D17 none")
else:
    assert set(r) == {"name","base","set"} and r["name"] == "D17"
    sys.exit(subprocess.call(["flock","-o","-w","7200","/tmp/gpu0.lock",sys.executable,"-u","ms_train.py",
        "--row",r["base"],"--set",*r["set"],"--name",r["name"],"--seeds","0","--device","cuda"]))
'
fi
step score_final cpu --score --sheet
finish
