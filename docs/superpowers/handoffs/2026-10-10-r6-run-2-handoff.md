# Handoff: continue round 6's unattended run: watch and pull the nine chosen-W1 verbaliser shards, then DTS stages 5 to 7 (chosen listing, chosen, stop), then the smoke, the read and the rest of the run handoff's §4 (steps 8 to 15)

> Written 2026-10-10 01:36 by the `r6 run` chat (cut at 200k tokens). Loop step reached: step 8 (run), rule §6 items 1
> to 5 passed, item 6 (DTS) at stage 5. Direction log entry: [2026-10-07 · Idea 3 killed at development; next step
> open](../direction_log.md).

## 1. Read in this order

1. This handoff whole.
2. [`2026-10-09-r6-run-handoff.md`](2026-10-09-r6-run-handoff.md) §2 (decided), §4 steps 7 to 15 (the plan; steps 1
   to 6 are done) and §5 (code and pitfalls). Everything there still holds.
3. The rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md` (F) §6 to §10, and the docstrings of
   `run_r6_dts.py`, `run_r6_smoke.py`, `run_r6_held.py` before calling them.
4. F's log `20261125_artelingo_held_test_log.md`, rows from 2026-10-09 23:44 on (this chat's work).

## 2. Decided, not to be reopened

As in the run handoff §2. Agent defaults added by this chat (the run report shows them, marked as agent defaults):

- **Seed-42 episodes file:** made fresh with the merged code by `F/prep_episodes_seed42.py` (`load_rows` +
  `build_seed`, hashes checked by `load_episodes_checked`) → `results/dts_inputs/episodes_seed42.npz`
  (c8261dd8291e…). Every DTS stage takes it as `--episodes`.
- **Listing jobs split into shards** by `F/prep_split_listing.py <job> 9`: contiguous slices of `listing_input.jsonl`
  whose boundaries fall on multiples of the batch size 16, so each shard's batches equal the unsplit job's; each shard
  carries the earlier listing outputs under `cache/<name>/` (`listings.jsonl`, `provenance.json`). The stages merge
  listing folders by (phrase, K).
- **Chosen verbaliser reuses the tuning subset's W1 answers:** the chosen run covers only positions [1024,4096),
  [5120,8192), [9216,12288) of job `r6run_s42_verb_all`; the tuning job `r6run_s42_verb_tune` already holds W1 for
  the first 1,024 episodes per pair (global episode indices, same keys), so each key has one output.

## 3. Where things stand

- **Rule §6 items 1 to 5 passed** (log rows 23:47 to 00:03): refit (8 posteriors bit for bit), picks (B, B0, B1
  exact), regression 131 of 131, sensitivity written. **Re-derivation phase 1 AGREE**, 389 of 389 (`F/rederive/`,
  committed 140de8b). The phase-1 agent may be reused for phase 2 only if it can be resumed; otherwise a fresh Opus
  agent that has not read the implementation (its own code is in `F/rederive/rd_*.py`, which it may reuse).
- **DTS (item 6):** settings copy committed (fecaf3d) before any seed-42 GPU job. Sanity (DTS-N, K 8 and 16) passed
  (`results/dts_sanity.json`). Tune passed, **chose W1 K8** (`results/dts_tune.json`). Budget: built by
  **2026-10-10 12:29** (clock pinned 2026-10-09 12:29).
- **Running on DAS6 (launched 01:30 to 01:35, about 30 min each, 2,048 calls at 0.89 s):** nine chosen-W1 verbaliser
  shards of job `r6run_s42_verb_all`, one per GPU on node401, node402, node408. Tags, nodes and ranges:
  `.scratch/r6-held-test/run/ch_watch_list.txt` (`<tag> <node> <start> <stop>`, tags `20261009-2330xx` to
  `2334xx-334540a`). Watch each with `cluster watch <tag> --node <node>` (background), pull with
  `cluster pull --tag <tag> --node <node>`; outputs land in `res/cluster_jobs/<tag>/code/outputs/r6_verbalise/r6run_s42_verb_all/`.
- **Folders the next stages need** (all local, pulled and verified):
  - tuning verbaliser outputs: the nine `--verbalise-out` arguments in `.scratch/r6-held-test/run/tune_vo_args.txt`
    (`res/cluster_jobs/<tag>/code/outputs/r6_verbalise/r6run_s42_verb_tune`), job `results/dts_jobs/r6run_s42_verb_tune`;
  - tuning listing outputs: nine `--listing-out` arguments in `.scratch/r6-held-test/run/tune_lo_args.txt`;
  - sanity listing output: `res/cluster_jobs/20261009-214837-fecaf3d/code/outputs/r6_listing/r6run_s42_listing_sanity`.

## 4. Next steps

1. Watch, then pull the nine chosen shards (expect 2,048 lines of `phrases_W1.jsonl` each).
2. Stage 5: `run_r6_dts.py --stage list-input --for chosen --seed 42 --episodes results/dts_inputs/episodes_seed42.npz`
   with `--verbalise-out` for all 18 folders (nine tuning, nine chosen), `--verbalise-job` for both job folders, and
   `--job-out results/dts_jobs/r6run_s42_listing_chosen`. Split it (`prep_split_listing.py ... 9`), put the sanity and
   the nine tuning listing outputs into each shard's `cache/` (many W1 K8 phrases and the names are already listed),
   commit with a "cluster run" subject, `cluster sync` all three nodes, ship each shard with `das6_sync_r6.py
   --job-dir`, launch `bash scripts/run_r6_listing.sh <shard>`, watch, pull.
3. Stage 6, chosen: `--verbalise-out` (18), `--verbalise-job` (2), `--listing-out` the chosen listing shards (their
   outputs include the copied cache entries; add the earlier listing folders only if a key is reported missing).
   Writes `results/dts_seed42.json` and `dts_first_built.json`.
4. Stage 7: `--stage stop --clock-start "2026-10-09 12:29"`. Exit 3 or a budget miss: stop before the read (rule §9),
   write up, report, open `r6 read`.
5. Then the run handoff's §4 steps 8 to 15 (smoke, H5 and the read, held GPU jobs, phase 2, verdict, descriptive
   pass, reports, storage, close with `loop-next ... --label "r6 read" --notify`).

## 5. Code and pitfalls (new in this chat)

- **The shell is zsh:** arrays are 1-indexed (a 0-based loop launched a wrong shard set once here), and a variable
  holding several arguments needs `${=VAR}` to split. Prefer a small bash script file in the scratchpad for launch loops
  (`.scratch/r6-held-test/run/launch_chosen.sh` is a model).
- **A long one-line command with `python3 -c` and arrays was refused** by a built-in removal check (no `rm` in it):
  keep commands simple, put loops in a script file.
- **Before every launch after a commit:** HEAD's subject must contain "cluster run" and `cluster sync --node` must
  have run on all three nodes, or the launch's code check fails. Log rows committed with a "cluster run" subject
  are fine (they carry the run's record).
- `cluster pull` puts outputs under `res/cluster_jobs/<tag>/code/outputs/...` (extras count shows 0; the files are
  there).

## 6. State at handoff

- Running: the nine chosen-W1 verbaliser shards on DAS6 (above). No local process, no subagent.
- Uncommitted: nothing (this handoff and `chats.tsv` are committed with the cut). `.scratch/` and `F/results/` are local
  by design.
- On disk over 1 GB: `F/results/gpu_images/` holds symlinks only; nothing new over 1 GB locally. Pulled job folders
  under `res/cluster_jobs/2026100{9}-2*` are small (MB). `pending_deletions.md`: no new row yet.
