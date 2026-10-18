# Stage 0 trace (r10 build 3, 2026-10-19)

## Problem

Stage 0 of the PercepT core spec (§3.5) re-runs the unchanged September scripts (commit `1b70d7f`) and must give the
recorded held-out AMIs to 4 decimals. On the local RTX 3090 both scripts give emotion 0.1238 (recorded 0.1225) and
genre 0.2617 (recorded 0.2274). The extended-seed Stage 2 script stops at its own re-fit check, so the seed 42 macro
AUC (recorded 0.8296) is never computed. Row 2 and the four-fix unit tests pass.

## Investigation

1. **Thread cap** (controller, GPU, 4 min): the Stage 1 snapshot script with default threads (32) instead of 8.
   Result: 0.1238 / 0.2617, the same as with 8 threads. Ruled out. (`threads_default_run.sh`, `threads_default_report.md`)
2. **The record's provenance** (Codex `gpt-6.1-sol` high, read-only; `trace0_findings.md`):
   - The recorded 0.1225 / 0.2274 come from runs on the DAS6 node node404 (launches in
     `/project/CoSiR-buddy_prototype_conditioning/cluster_sync.log` lines 157-178, 2026-09-23 08:57 and 2026-09-24
     01:38 to 01:59). Its GPUs were RTX A6000 in October; the September slot is not recorded.
   - Commit `3b3ce61` rebased the scripts' expected values to that node404 fit; it did not change the training code.
   - The September noise-floor run (`res/percept/percept_stage1_noise_floor_results.jsonl`) gives five identical
     node404 fits, 0.122469 / 0.227419: the fit is deterministic per machine.
   - The saved September node404 snapshot scores 0.122469 / 0.227419 with today's code (`artifact_comparison.txt`):
     scoring and labels are unchanged.
3. **The local September record** (controller check): the cluster-count sweep v2 report, generated 2026-09-23 02:19,
   before the first node launch (08:19), gives 0.1238 / 0.2617 at K 60 to 40 and seed 42. Its pretraining losses
   equal today's at all ten printed epochs (0.000077 ... 0.000028).
4. **Code drift:** the two scripts and every module they import are byte-identical between the commits that made the
   records and `1b70d7f` (`artifact_comparison.txt`). K-means on the saved pre-DEC latents reproduces September's
   initial labels exactly (`kmeans_probe.txt`).

## Root cause

The recorded values belong to another machine (node404). This machine gives 0.1238 / 0.2617, as it did in
September, with identical pretraining. The genre AMI rests on 159 held-out paintings, so one seed moves it by 0.034
between GPUs. No code, data or setting of ours changed.

## Fix

None possible tonight. No reservation exists on node404 (the user holds node406 and node408; agents do not pick nodes),
so September cannot run in its own environment. The spec's gate holds as written (§3.5; constitution C12): the run
stops the PercepT arms, the head rows (stage 2 of the spec) still run, and the user decides in the morning. Options
for the user are in the r10 run handoff.
