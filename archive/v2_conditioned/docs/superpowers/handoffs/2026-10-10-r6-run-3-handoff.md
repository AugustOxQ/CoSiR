# Handoff: launch the held read (`run_r6_held.py --mode held`), then the held GPU jobs, phase 2 on Codex, the verdict, the descriptive pass, reports and storage (run handoff §4 steps 9 to 15)

> Written 2026-10-10 03:37 by the `r6 run 2` chat (cut at 200k tokens). Loop step reached: step 8 (run), rule §6
> items 1 to 7 all passed, ledger row H5 committed, the read not started. Direction log entry: [2026-10-07 · Idea 3
> killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. This handoff whole.
2. [`2026-10-09-r6-run-handoff.md`](2026-10-09-r6-run-handoff.md) §2 (decided), §4 steps 9 to 15 (the plan) and §5
   (code and pitfalls); [`2026-10-10-r6-run-2-handoff.md`](2026-10-10-r6-run-2-handoff.md) §2 and §5. All still hold.
3. The rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md` (F) §8 to §10; the docstrings of `run_r6_held.py`
   (held mode), `r6_gpu_inputs.py`, `run_r6_dts.py` (held phrases), `run_r6_apply_rule.py`, `run_r6_descriptive.py`
   before calling them.
4. F's log, rows from 2026-10-10 01:40 on.

## 2. Decided, not to be reopened

As in the two handoffs above. New in this chat:

- **Codex routing (user, 2026-10-10 03:03: Claude usage is short; send remaining subagent work to Codex where
  possible, check `codex-usage`, make best use of it).** Re-derivation phase 2 → Codex (`gpt-6.1-sol` high), brief
  ready at `.scratch/r6-held-test/run/codex_phase2_brief.md`. Rule §8.4 asks only for an agent that has not read the
  implementation; the run handoff's "fresh Opus agent" was an agent default. Final review: the number re-derivation and
  a second-opinion review (`gpt-6-astra` high) on Codex, the claim review on a scoped Opus subagent; figures and the
  fidelity checks on Codex; the user-read cold read on haiku. `codex-usage` at 03:04: 5 h 100% left (resets 08:04),
  weekly 80%. Follow `~/.claude/references/codex-jobs.md` (the command, ledger row in
  `/project/claude-config/codex/jobs.md` before and after each job).
- **Agent defaults (report them, marked):** the listing splitter now places shard boundaries by uncached items
  (`prep_split_listing.py --cache NAME=FOLDER`, change log `.claude/20261010_log.md`), so each shard's batches equal
  the unsplit cached job's; the smoke listing ran unsplit.

## 3. Where things stand

- **DTS (item 6):** chosen W1 K8, built 2026-10-10 02:20:53 (budget deadline 12:29), stop: no stop
  (`results/dts_stop.json`, budget time from `dts_first_built.json`).
- **Smoke (item 7): PASSED** 03:31, no waiver (`results/smoke_record.json` 5b299985088a…; copy committed).
- **H5** committed (347c328) and checked with the runner's own `ledger_guard`: 7 cells, script SHA a82ca8bc128e…,
  report "(pending)". The read has **not** started; the time-box refuses a first read after 2026-10-15.
- Measured DAS6 speeds: verbaliser 0.93 s per call, reranker 3.67 s per episode, listing 0.24 to 0.29 s per prompt,
  FT features minutes.

## 4. Next steps

1. **The read:** `run_r6_held.py --mode held` in a background Bash (command in its docstring; log to
   `results/run_r6_held_held.log`). Once `held_episodes_seed5{2,3,4}.npz` exist and their hashes are in
   `held_started.json`, commit `held_started.json`'s copy if the runner did not, fill H5's episode cell with the nine
   64-hex hashes, commit and push.
2. **Held GPU jobs (rule §8.3), started as soon as the episode files exist** (they may run beside the read):
   `r6_gpu_inputs.py --episodes results/held_episodes_seed<s>.npz --out <folder>` for s = 52, 53, 54 (verbaliser);
   `--job rerank` on seed 52; `--job ft --episodes ...52 --also-episodes ...53 --also-episodes ...54`. Ship each folder
   with `--images` to **all three nodes** (and the FT checkpoints, as the smoke did: paths in
   `.scratch/r6-held-test/run/smoke_logs/gpu_inputs.log`). Commit a log row with a "cluster run" subject, `cluster
   sync` the three nodes, then run `.scratch/r6-held-test/run/gpu_queue.sh <queue file> <state dir>` in a background
   Bash. Queue lines `<label> <script and args under scripts/>`; suggested shards of about one hour: FT first, then
   the verbaliser `--wordings W1 --start a --stop b` in 2,048-episode shards (18), then the reranker in 1,024-episode
   shards (12); about 31 GPU-hours, 3.5 to 4 h on nine GPUs. Append the three held listing jobs (`run_r6_dts.py
   --stage list-input --for held --seed s ...`, split with `prep_split_listing.py ... --cache` holding the seed-42
   sanity, tuning and chosen listing folders) when a seed's verbaliser shards are pulled; write `END` to close the
   queue. The queue's untested parts: watch `events.txt` in its state dir after the first launches.
3. **Phase 2 on Codex** once `held_pass.json` exists (`codex e ... --json - < codex_phase2_brief.md` from a background
   Bash, `-C /project/CoSiR`). Check its rollout for forbidden reads (`r6_`, `run_r6_`, `design/`, held outputs
   before `phase2_results.json`), recount `rederive/phase2_compare.json`, then write `results/rederive_agreement.json`
   (contracts §8) and run `run_r6_apply_rule.py`; put `held_verdict.json`'s SHA-256 in H5's report cell.
4. External sources, descriptive pass, final review, reports, seed ledger note, disclosures, storage, close: run
   handoff §4 steps 10 to 15.

## 5. Code and pitfalls (new in this chat)

- The `cluster` on PATH is a graph tool; the cluster CLI is `~/.claude/skills/cluster-run/cluster`.
- **Never add a file named `r6_*.py` or `run_r6_*.py` in F, or `scripts/run_r6_*.sh`:** `r6_module_shas()` hashes
  them, and the held runner and the apply step refuse when the set differs from the smoke record's. Orchestration
  scripts stay in `.scratch/r6-held-test/run/` (`watch_list.sh`, `pull_list.sh`, `gpu_queue.sh`).
- Helpers: `watch_list.sh <list> <dir>` watches every `<tag> <node> ...` line in parallel; `pull_list.sh <list> <dir>`
  pulls them. Pulled outputs are under `res/cluster_jobs/<tag>/code/outputs/<family>/<job>/`.

## 6. State at handoff

- Running: nothing (no local process, no DAS6 job, no subagent, no Codex job).
- Uncommitted: nothing (this handoff and `chats.tsv` are committed with the cut). `.scratch/` and `F/results/` are
  local by design.
- On disk over 1 GB: nothing new. `F/results/smoke/` and the pulled job folders are small (MB); `rederive/out/`
  (75 MB) serves phase 2. `pending_deletions.md`: no new row yet.
