# Handoff: regenerate the held listings of seeds 53 and 54 with the earlier held listings as caches, rerun the descriptive pass, then the final review, reports, storage and close (run handoff §4 steps 12 to 15)

> Written 2026-10-10 07:55 by the `r6 run 3` chat (cut at 200k tokens). Loop step reached: step 8 done up to the
> verdict (**GO**); the descriptive pass stopped on a listing conflict, cause traced. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

**At the top for the morning (unattended notes):**
- **Codex is refused by the auto-mode permission check** (`codex e --dangerously-bypass-approvals-and-sandbox`,
  03:45). Phase 2 went to an Opus subagent instead; every remaining Codex job of this run (final-review number
  re-derivation and second opinion, figures, fidelity checks) goes to Claude subagents. The user asked (03:03) to save
  Claude usage: use `sonnet` where the job allows (fidelity, figures, the number re-derivation), `opus` only for the
  final claim review; `haiku` for the user-read cold read. Ledger row in `/project/claude-config/codex/jobs.md` says
  "taken back".
- **The descriptive pass stopped (exit 5)** on 416 listing keys with different answers across held seeds; the fix
  below is an agent default (reported as such). The verdict is untouched.

## 1. Read in this order

1. This handoff whole.
2. [`2026-10-09-r6-run-handoff.md`](2026-10-09-r6-run-handoff.md) §2, §4 steps 12 to 15, §5; the §2 and §5 of
   [`2026-10-10-r6-run-2-handoff.md`](2026-10-10-r6-run-2-handoff.md) and
   [`2026-10-10-r6-run-3-handoff.md`](2026-10-10-r6-run-3-handoff.md). All still hold.
3. F's log (`src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md`), rows from 2026-10-10 03:33.
4. The docstrings of `run_r6_dts.py` (held phrases), `prep_split_listing.py`, `run_r6_descriptive.py`; the comment at
   the top of `r6_external.py` (the `external_sources.json` contract).

## 2. Decided, not to be reopened

As in the three handoffs above. Agent defaults added by this chat (the run report shows each, marked):

- Phase 2 on a fresh Opus subagent (Codex refused), same brief and independence list; the controller checked its 44
  tool calls (no implementation file, no held output before its SHA print) and recounted its comparison.
  `rederive_agreement.json` carries extra provenance fields (`compare_file`, `compare_sha256`, `phase2_results_sha256`,
  `agent`) beside contracts §8's.
- Held GPU shards: verbaliser 2,048 episodes, reranker 1,024, listing 3 per seed; FT shipped with the checkpoints to
  all three nodes.
- The listing fix below; `external_sources.json` moved aside (never deleted) and written again.

## 3. Where things stand

- **The read** (03:33 to 03:44, attempt 1, exit 0): `held_pass.json` 93e001572172…; H5's episode cell holds the nine
  hashes (c93c6f9). **Phase 2 AGREE**, 249 of 249 (`rederive/phase2*.{json,md}`, committed 1d73516).
  **Verdict GO** (`held_verdict.json` 55d88be907c4…, in H5's report cell): P1 to P7 pass, each with n_j = 0 (points
  in R@1 points: P1 +6.19, P2 +5.89, P3 +0.63, P4 +0.60, P5 +0.63, P6 +3.06, P7 +2.92); after the GO, S1 (against
  B′(A1)) +0.136 [−0.028, +0.303], n 258 against n* 61, and S2 (against R1) +0.022 [−0.063, +0.112], n 1,611 against
  124: both fail, read inconclusive. No boundary case. Claim text: `held_verdict.json` "claim".
- **Seed ledger** updated (seeds 52 to 54 held seeds of H5; free from 55).
- **Held GPU jobs: 40 of 40 done**, pulled and count-checked (03:45 to 07:44). Tags and labels:
  `.scratch/r6-held-test/run/held_queue/state/pulled.txt` (`<tag> <node> <label>`; labels `ft`, `v<seed>_<start>`,
  `r52_<start>`, `lr6run_s<seed>_listing_part<i>`); outputs under `res/cluster_jobs/<tag>/code/outputs/<family>/<job>/`.
- **`results/external_sources.json`** (935562b67783…) written, then the descriptive pass stopped: see F's log row
  07:52 for the cause and counts.

## 4. Next steps

1. **Seed 53's listing again** (no code change; `held_after.sh`'s `listing_for` is the model):
   `run_r6_dts.py --stage list-input --for held --seed 53 --episodes results/held_episodes_seed53.npz`, the six
   `--verbalise-out` folders of labels `v53_*` and `--verbalise-job results/held_jobs/r6run_s53_verb`,
   `--job-out results/held_jobs/r6run_s53_listing_c`. Split: `prep_split_listing.py <job> 3` with
   `$(cat .scratch/r6-held-test/run/held_cache_args.txt)` (the 19 seed-42 folders) **plus** `--cache
   s52_held_part<i>=<pulled folder of lr6run_s52_listing_part<i>>` for i = 0, 1, 2. Ship each shard with
   `das6_sync_r6.py --node <n> --job-dir <shard> --run` to the three nodes, commit (subject with "cluster run"),
   `cluster sync` the three nodes, launch `bash scripts/run_r6_listing.sh <shard>` (one per node), watch, pull, check
   `listings.jsonl` lines equal `listing_input.jsonl` lines.
2. **Seed 54 the same**, `r6run_s54_listing_c`, caches: the 19 + seed 52's three + the three new seed-53 outputs.
3. **Check before the pass:** over the 19 seed-42 folders, 52's three and the six new ones, every (phrase, K) has
   one answer (the check in F's log row 07:52 is a model).
4. `mv results/external_sources.json results/external_sources.json.failed1`; copy
   `.scratch/r6-held-test/run/write_external_sources.py` to a new name, make its listing list 52's old shards plus
   the new 53 and 54 shards (labels or tags), run it, then the descriptive pass (command in its docstring;
   `descriptive.json` was not written, so it runs again). A second stop goes at the top of the next handoff.
5. Then run handoff §4 steps 13 to 15: final review of the run's results (routing above), the auto report
   `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` with its index row, the user-read report, the
   disclosures of rule §10.4, H5's report cell gets the report link beside the verdict SHA; storage (step 14:
   `cluster du` on node401, node402, node408, every item on `.scratch/pending_deletions.md`, nothing deleted, the
   first seed-53/54 listing outputs included); close with `loop-next <handoff> --label "r6 read" --notify`.

## 5. Code and pitfalls (new in this chat)

- **Listing caches must hold every earlier listing of the same phrases**, held seeds included: the DTS merge refuses
  two answers for one key, and greedy answers change with the batch.
- While any DAS6 launch is pending, every commit needs "cluster run" in its subject and a `cluster sync` of all three
  nodes after it.
- `gpu_queue.sh` is not resumable (it relaunches from the first line); for a few jobs launch by hand.
- Scripts of this chat in `.scratch/r6-held-test/run/`: `held_gpu_prep.sh`, `held_after.sh` (pull, count checks,
  listing per seed), `write_external_sources.py`, `held_cache_args.txt`, `held_queue/` (queue, state, logs).

## 6. State at handoff

- Running: nothing (no local process, no DAS6 job, no subagent).
- Uncommitted: nothing (this handoff and `chats.tsv` are committed with the cut). `.scratch/` and `F/results/` are
  local by design.
- On disk: pulled held outputs 371 MB in `res/cluster_jobs/2026101[0]-0[1-4]*` (FT features 348 MB);
  `F/results/held_jobs/` about 40 MB; `F/rederive/out/` 102 MB (phase 1 and 2; both marked for the storage summary).
  On the nodes: held images and job folders under `/local/wding/r6_jobs/` (for `cluster du`). `pending_deletions.md`:
  no new row yet.
