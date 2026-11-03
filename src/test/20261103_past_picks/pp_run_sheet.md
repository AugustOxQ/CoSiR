# r19 run sheet (ticket 12)

Run from the checkout holding all merged r19 code. The runner never commits.
Use eight threads. No worker count above eight. Each CPU command below queues each step on its own.
GPU work ends by 10:00 on a weekday. Heavy work that misses that window waits until 19:00.
CPU scoring may run until about 13:00 on 2026-11-04.

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/root/miniconda3/envs/CoSiR/bin/python
R=src/test/20261103_past_picks
OUT=/data/SSD2/cosir_r19/20261103_past_picks/outputs
unset PP_SMOKE PP_OUTPUTS
```

Affection extraction is ALREADY DONE. Skip it. Do not rerun `pp_extract_queue.sh affection` or `all`.
The sealed stores refuse rewrites. Check the three successful Affection lines and the three store-check JSONs.
Parity is next. MMArt and SemArt follow the node jobs below. The queue script reserves each GPU step separately.

```bash
sh "$R/pp_extract_queue.sh" parity                  # up to 12 min per view
rg 'extract-affection-(clip_b32|clip_b16|dinov2_large)' "$OUT/stage0/gate_i/extract_queue_status.txt"
$PY -c 'import json, pathlib; p=pathlib.Path("/data/SSD2/cosir_r19/20261103_past_picks/outputs/stage0/gate_i"); print({e:json.loads((p/f"store_affection_{e}.json").read_text())["ok"] for e in ("clip_b32","clip_b16","dinov2_large")})'
```

The three Affection lines must each end in exit code 0. The three checks must be true.
Use `pp_node_plan.sh` as the full input-check and node-smoke plan. It only prints commands.
The controller already applied the data map. Do not clean node jobs in this run.
Before the first pull, prepare the result symlink in the pulling checkout.

```bash
bash "$R/pp_node_plan.sh"
mkdir -p res
ln -sfn "$OUT" res/r19_pp
CLUSTER=/root/.claude/skills/cluster-run/cluster
RUN='bash scripts/run_pp_node.sh'
$CLUSTER sync --node node402
$CLUSTER check --node node402 --sync-data -- $RUN probe aux_art qwen_art aux_aff qwen_aff art_dinov2_large art_clip_b32 art_clip_b16 aff_dinov2_large aff_clip_b32 aff_clip_b16 art_clip_l14 aff_clip_l14 m5
$CLUSTER launch --queue --node node402 --est 10m --label r19:probe -- $RUN probe aux_art qwen_art aux_aff qwen_aff art_dinov2_large art_clip_b32 art_clip_b16 aff_dinov2_large aff_clip_b32 aff_clip_b16 art_clip_l14 aff_clip_l14 m5
$CLUSTER watch <probe-tag>
$CLUSTER pull --tag <probe-tag>
$CLUSTER check --node node402 --sync-data -- $RUN regress-ii
$CLUSTER launch --queue --node node402 --est 25m --label r19:ii -- $RUN regress-ii
$CLUSTER watch <ii-tag>
$CLUSTER pull --tag <ii-tag>
bash "$R/pp_run_cpu.sh" 0 --step gate_ii           # compare only, about 1 min
```

Stop affected training if gate (ii) fails. Debug it. X photo for an encoder starts only after its c* job exits 0.
A crash or nonfinite run retries once with the same seed. Two failures mark that path failed; other encoders go on.
The node runner implements this rule. Check `OUT/node/cstar_<enc>.json` and `photo_<enc>.json`, each with all five seeds.
Run these commands for each encoder, and watch and pull each tag. c* takes about 45 min per encoder; photo about 20 min.

```bash
for e in dinov2_large clip_b32 clip_b16; do
  $CLUSTER check --node node402 --sync-data -- $RUN part-a --encoder "$e"
  $CLUSTER launch --queue --node node402 --est 60m --label "r19:cstar_$e" -- $RUN part-a --encoder "$e"
done
$CLUSTER watch <cstar-dinov2-tag>
$CLUSTER pull --tag <cstar-dinov2-tag>
$CLUSTER watch <cstar-b32-tag>
$CLUSTER pull --tag <cstar-b32-tag>
$CLUSTER watch <cstar-b16-tag>
$CLUSTER pull --tag <cstar-b16-tag>
# Only for each encoder whose c* exited zero:
for e in dinov2_large clip_b32 clip_b16; do
  if "$PY" -c 'import json,sys; from pathlib import Path; d=json.loads((Path(sys.argv[1])/"node"/f"cstar_{sys.argv[2]}.json").read_text()); sys.exit(0 if d["status"]=="ok" else 1)' "$OUT" "$e"; then
    $CLUSTER launch --queue --node node402 --est 30m --label "r19:photo_$e" -- $RUN photo --encoder "$e"
  fi
done
$CLUSTER watch <photo-dinov2-tag>
$CLUSTER pull --tag <photo-dinov2-tag>
$CLUSTER watch <photo-b32-tag>
$CLUSTER pull --tag <photo-b32-tag>
$CLUSTER watch <photo-b16-tag>
$CLUSTER pull --tag <photo-b16-tag>
$CLUSTER check --node node402 --sync-data -- $RUN m5
$CLUSTER launch --queue --node node402 --est 15m --label r19:m5 -- $RUN m5
$CLUSTER watch <m5-tag>
$CLUSTER pull --tag <m5-tag>
```

M5 full takes about four minutes. Its sample gate rerun ran on the local 3090; the controller accepted the byte-equal
weights. The runner records this in the collection log. Gate (v) has one file per new encoder.
Now finish extraction. The probe and extraction use the local GPU through res-queue.

```bash
sh "$R/pp_extract_queue.sh" mmart                 # 6, 6, 8 min
sh "$R/pp_extract_queue.sh" semart                # 12, 12, 12, 15, 40 min
```

Start the probe only when `uptime`'s 1-minute load is under about 16. Start no CPU scoring of ours while it runs.
If gate (viii) fails with recorded load over 16, it is not a failed gate yet. Move `OUT/probe` aside without deleting
it, rerun on a quiet CPU, and note the first attempt in the run log. Only a failure on a quiet CPU counts for the
voids table. A quiet failure is a code bug to debug. Never widen the bar. Load is recorded; it never discards a repeat.
Raw files are never overwritten. `--resume` skips complete passes. Heads need the pulled seed-0 c* weights.

```bash
uptime
res-queue gpu --est 1h --label r19:probe-qwen -- $PY "$R/pp_probe.py" passes --enc qwen --resume
for e in siglip2 siglip1 clip_l14 dinov2_large clip_b32 clip_b16; do
  res-queue gpu --short --est 12m --label "r19:probe-$e" -- $PY "$R/pp_probe.py" passes --enc "$e" --resume
done
res-queue gpu --short --est 12m --label r19:probe-heads -- $PY "$R/pp_probe.py" heads
bash "$R/pp_run_cpu.sh" 3                        # cells_cost + cost_table + gate_viii, seconds
# Only after a loaded failure; choose a fresh suffix, keep every attempt:
mv "$OUT/probe" "$OUT/probe_load_attempt_$(date +%Y%m%d_%H%M%S)"
# Wait for load under about 16; rerun the probe commands and stage 3 above.
```

The move command is conditional on that loaded failure. Do not move a successful probe.
CPU scoring starts after the probe. Each step is separately queued. Gate (iv) r15 takes 30 min; gate (vii) r15 17 min;
gate (iv) SemArt 5 min. The r14 gate parts take about 5–15 min each. Exports take about 1–15 min each.
A gate command can exit 1 with fresh failure evidence; inspect gate.json and status.json, not just the queue exit code.
Stage 1 costs about 15–40 min per new encoder, plus about 4 min per SemArt validation entry.
The runner logs every attempt and continues other paths after a failure. Reduced gates remain failed even on exit 0.

```bash
bash "$R/pp_run_cpu.sh" 0
bash "$R/pp_run_cpu.sh" 1
# Execute the printed git add -f ... && git commit ... command. Inspect status first.
# Records: stage1_<new>.json, stage1_semart.json, semart_set.json, status.json.
bash "$R/pp_run_cpu.sh" 2                        # roughly 3–4 CPU h total scoring at 8 threads
```

Stage 2 refuses until the selected encoders' stage 1 records, the SemArt record, set and status equal HEAD.
SemArt validation entries are independent. Missing entries remain missing; their scorer fails while others proceed.
Stage 1 uses local merged validation blocks to build SemArt entries, then freezes them together in one commit.
No held data selects a pick. Stage 2 and stage 3 select nothing.
`pp_run.py <stage> --list` prints every step and estimate. `--step NAME` reruns one step. Recollect and commit status
before scoring if any gate or node status changes. Do not run the full stage again merely to collect status:

```bash
$PY "$R/pp_run.py" status
```

C13 is next, by Codex from the saved exports, committed records and probe raw batch timings. Never put the new three
through `mt_c13`. Re-derive the set, ranges, picks, SC1–SC15, intervals and verdicts within 1e-6 before writing outcomes.
The run chat writes and commits `records/c13.json`: `pass: true` and `sha256` of every `stage2/cells_*.json`,
`stage3/cells_cost.json`, every `probe/**/*.json`, and all `records/stage1_*.json`, `records/status.json`,
`records/semart_set.json`. Hash keys are relative to OUT, with records under `records/` even in a real run.
Stage 4 checks these exact bytes. It runs outcomes with the actual MMArt cut count and post-md5 SemArt painting and
artist counts. Diagnostics follow as `pp_diag.py run`; an absent module fails cleanly in its own log, after outcomes.
Allow about 30 min for diagnostics. r17 context rows are optional and never affect a verdict.

```bash
bash "$R/pp_run_cpu.sh" 4
```

Rerunnable CPU smoke, one stage at a time. No queue, GPU, extraction, training or real records writes in this smoke.
It uses small synthetic key subsets, stand-in exports and raw timing files, and the real MMArt/SemArt statistics.
The recorded reduced gates are failed, so its outcomes demonstrate inconclusive wiring, not a scientific gate pass.
It explicitly shows the uncommitted-record refusal and the strict wrong-status refusal.

```bash
export PP_SMOKE=1
export PP_OUTPUTS=/tmp/claude-0/-project-CoSiR/f230deb2-5b4b-4531-9369-0d454a9ed956/scratchpad/t12_smoke
$PY "$R/pp_run_smoke.py" 0
$PY "$R/pp_run_smoke.py" 1
$PY "$R/pp_run_smoke.py" 2
$PY "$R/pp_run_smoke.py" 3
$PY "$R/pp_run_smoke.py" 4
```
