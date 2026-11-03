# r19 run sheet (ticket 12)

Run from the checkout holding all merged r19 code. The runner never commits.
Order: 1. parity; 2. node gate (ii), c*, photo, M5; 3. rest of extraction; 4. probe; 5. stage 3; 6. stage 0;
7. stage 1; 8. commit; 9. stage 2; 10. C13; 11. stage 4.
Use eight threads. No worker count above eight. Each CPU command below queues each step on its own; `PP_CPU_JOBS` (default 3) sets how many reservations are open at once, and merge, collect, decide and diagnostics steps wait for all earlier steps.
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
Parity is next. `sh "$R/pp_extract_queue.sh" parity` runs in the background beside the node jobs (short local GPU steps), so it does not hold up the node's start. MMArt and SemArt follow the node jobs below. The queue script reserves each GPU step separately.

```bash
sh "$R/pp_extract_queue.sh" parity                  # up to 12 min per view
grep -E 'extract-affection-(clip_b32|clip_b16|dinov2_large)' "$OUT/stage0/gate_i/extract_queue_status.txt"
$PY -c 'import json, pathlib; p=pathlib.Path("/data/SSD2/cosir_r19/20261103_past_picks/outputs/stage0/gate_i"); print({e:json.loads((p/f"store_affection_{e}.json").read_text())["ok"] for e in ("clip_b32","clip_b16","dinov2_large")})'
```

The three Affection lines must each end in exit code 0. The three checks must be true.
Use `pp_node_plan.sh` as the full input-check and node-smoke plan. It only prints commands.
The controller already applied the data map. Do not clean node jobs in this run.
The node input probe (tag 20261103-232618-001d3d1-9a592cff, 59 inputs, 0 failures) and the node smoke (tag
20261103-232644-001d3d1-78737588, clip_b32 c* and photo seed 0 ok) passed in the build at 00:34 on commit 001d3d1.
Do not rerun them. The run chat runs `$CLUSTER sync --node node402` first, and reruns the probe only if
`pp_node_inputs.json` changed since 001d3d1. The checkout that pulls needs the symlink `res/r19_pp` (below).

```bash
bash "$R/pp_node_plan.sh"
mkdir -p res
ln -sfn "$OUT" res/r19_pp
CLUSTER=/root/.claude/skills/cluster-run/cluster
RUN='bash scripts/run_pp_node.sh'
$CLUSTER check --node node402 --sync-data -- $RUN regress-ii
$CLUSTER launch --queue --node node402 --est 25m --label r19:ii -- $RUN regress-ii
$CLUSTER watch --node node402 <ii-tag>
$CLUSTER pull --node node402 --tag <ii-tag>
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
$CLUSTER watch --node node402 <cstar-dinov2-tag>
$CLUSTER pull --node node402 --tag <cstar-dinov2-tag>
$CLUSTER watch --node node402 <cstar-b32-tag>
$CLUSTER pull --node node402 --tag <cstar-b32-tag>
$CLUSTER watch --node node402 <cstar-b16-tag>
$CLUSTER pull --node node402 --tag <cstar-b16-tag>
# Only for each encoder whose c* exited zero:
for e in dinov2_large clip_b32 clip_b16; do
  if "$PY" -c 'import json,sys; from pathlib import Path; d=json.loads((Path(sys.argv[1])/"node"/f"cstar_{sys.argv[2]}.json").read_text()); sys.exit(0 if d["status"]=="ok" else 1)' "$OUT" "$e"; then
    $CLUSTER launch --queue --node node402 --est 30m --label "r19:photo_$e" -- $RUN photo --encoder "$e"
  fi
done
$CLUSTER watch --node node402 <photo-dinov2-tag>
$CLUSTER pull --node node402 --tag <photo-dinov2-tag>
$CLUSTER watch --node node402 <photo-b32-tag>
$CLUSTER pull --node node402 --tag <photo-b32-tag>
$CLUSTER watch --node node402 <photo-b16-tag>
$CLUSTER pull --node node402 --tag <photo-b16-tag>
$CLUSTER check --node node402 --sync-data -- $RUN m5
$CLUSTER launch --queue --node node402 --est 15m --label r19:m5 -- $RUN m5
$CLUSTER watch --node node402 <m5-tag>
$CLUSTER pull --node node402 --tag <m5-tag>
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
# Execute the printed git add -f ... && git commit ... command (in OUT/stage1/logs/merge.log, written by the stage 1 `merge` step). Inspect status first.
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
Allow about 1 h for diagnostics (`r17_metrics` repeats about the k-means fits of `clustering_kclass`). r17 context rows never affect a verdict.

```bash
bash "$R/pp_run_cpu.sh" 4
```

Rerunnable CPU smoke, one stage at a time. No queue, GPU, extraction, training or real records writes in this smoke.
It uses small synthetic key subsets, stand-in exports and raw timing files, and the real MMArt/SemArt statistics.
The recorded reduced gates are failed, so its outcomes demonstrate inconclusive wiring, not a scientific gate pass.
It explicitly shows the uncommitted-record refusal and the strict wrong-status refusal.
Smoke diagnostics run gain_share, percept_strongest, cost_rows and node_time from saved synthetic artifacts.
The other eight diagnostics require full stores and are outside this small smoke. Production stage 4 runs all twelve.
Prior diagnostic attempts are moved aside without deletion before a smoke diagnostic rerun.

```bash
export PP_SMOKE=1
export PP_OUTPUTS=<scratch folder>/t12_smoke
$PY "$R/pp_run_smoke.py" 0
$PY "$R/pp_run_smoke.py" 1
$PY "$R/pp_run_smoke.py" 2
$PY "$R/pp_run_smoke.py" 3
$PY "$R/pp_run_smoke.py" 4
```
