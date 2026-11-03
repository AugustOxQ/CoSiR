# Handoff: build r17 (new scores from saved outputs) from its approved spec, then open the run chat

## Brief
- Done: r17 spec approved 20:45 from its brief; the Opus `xhigh` rule review passed after fixes, all applied.
- State: CPU only, no training: 412 metric cells in eight families (SC1 to SC8); every input is saved.
- Next: tickets and build per `matt-chain.md`, the final review, merge into main, then open the run chat.
- Blocked: nothing. No container restart (user).

> Written 2026-11-03 20:50 by tab "r17 decide 2". Loop step reached: after step 6 (spec approved), before step 7.
> Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "r17 grill (B)" and "r17 spec
> approved".

## 1. Read in this order

1. The spec, from §1 on: [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md). Its decision
   rule is §6 with §3.4 and §3.5; the stage 0 gates and their gate-scope table say which cells each check serves;
   §3.7 is the shared scorer's interface, frozen at merge, because r18 and r19 import it read-only.
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any
   Codex job.
3. The saved-outputs inventory, [`ns_saved_outputs.md`](../../../src/test/20261103_new_scores/ns_saved_outputs.md):
   paths, shapes, class counts, the scoring functions to import and where each seed range comes from.
4. The rule review, [`ns_rule_review.md`](../../../src/test/20261103_new_scores/ns_rule_review.md): the evidence
   behind each fix. Its two check scripts are now in the round folder (`ns_review_check_classes.py`,
   `ns_review_check_subset.py`).
5. `~/.claude/rules/shared-resources.md`, "The queue for the local GPU and CPU" and "CPU and RAM" (rule update of
   2026-11-03 20:45).

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`; slash-only, so read their SKILL.md).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D20) and the direction log, 2026-11-03. The spec was approved from its brief at 20:45,
with no container restart. The shared-queue rule (20:45) is applied in the spec's §8 and §9: every CPU-heavy step
goes through `res-queue cpu`.

## 3. Where things stand

- **Inputs:** all saved (inventory). CRL vectors are not saved; they are built in seconds from r11's form
  (`ac_crl.vectors`). PercepT validation probabilities exist only for Qwen on ArtELingo; the others come from the
  saved mappers on CPU (stage 0 gate (iv) checks the replay first).
- **Machine at 20:45:** run mode night. `res-queue status`: GPU idle, CPU 4 slots free. DAS6: no waiting launches.
  Load 0.3, 54 GB RAM available, SSD2 1.8 TB free. GPU check passes (driver 595.99.02), though r17 needs no GPU.
- **Codex at 20:45:** `codex-usage --room implementer review` says no room (five-hour window 42% left, resets
  23:34). The weekly limit is at 91% left: the earlier "5% until 11-04 08:27" note is out of date. Claude implements
  until 23:34; check again after.
- **Timing:** build about 3 h from about 21:00, so it ends about 00:00. The run (about 13 to 20 CPU h, 5 to 8 h of
  wall time in three `res-queue cpu` jobs) then ends about 05:30 to 08:30, before 11:00. r18 and r19 are in their
  spec chats; their runs are mainly GPU work (DAS6 and the local GPU), which takes no CPU slot, and their scoring
  imports r17's scorer after r17 merges. Report due 2026-11-04 18:00.

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to
the run chat.

## 5. Code and pitfalls

- **Folder and outputs:** round folder `src/test/20261103_new_scores/`, module prefix `ns_`. Outputs under
  `/data/SSD2/cosir_r17/20261103_new_scores/outputs/` (not created yet).
- **Branch:** integration branch `r17-new-scores` from `main` (it holds this handoff). r18 and r19 build later from a
  main that holds r17's merge: merge first, and keep §3.7's interface as specced.
- **Verdict code** gets unit tests and an Opus `high` ticket review (spec §9): the metrics (AMI, NMI with arithmetic
  normalization, ARI, ACC by Hungarian matching), the partitions at K, the bootstraps (SC8 draws PercepT's seeds
  apart from c*'s; told against untold keeps one shared draw, D19), the read-out picks, the untold join, the CPU
  mappers, the CRL vectors and the outcome code with its inconclusive paths. The rest: the controller's diff read
  and one end-to-end smoke run.
- **Frozen round folders** are imported, never edited (r11 `rt_*`, r12/r13 `ac_*`, r14 `bs_*`, r15 `mt_*`); copy and
  extend in `ns_`. `ac_cluster.partitions` asserts k = 20; `bs_emoset_score.partitions_k(x, k)` takes k.
  `rt_stats.paired` needs float64 inputs. EmoSet exports are uncut: use `bs_emoset.keys(split, cut=True)`. MMArt test
  is the 16,384 subset in r15's `records/cutsets.json`. Order follows r15 D9 (sha256 of the key). c*'s three heads
  of one seed are three separate runs: check key equality before the untold join.
- **k-means threads (D20):** `res-queue cpu` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 8 per slot, but
  Affection and EmoSet need 1 thread. Set the count in code per set, as the earlier rounds did:
  `threadpool_limits(limits=8)` in `ct_score.output_partitions` for ArtELingo and MMArt; 1 thread in
  `rs_vh.r20_partition` and `bs_emoset_score.partitions_k` for Affection and EmoSet (rule review M3). Stage 0 gate
  (i) c replays those cells and catches a mismatch.
- **Queue:** each CPU-heavy step is its own `res-queue cpu --label r17:<step> --est <duration> -- <cmd>` job (checks
  under 15 minutes: `--short`), never a whole script. Full test suites go through `res-queue cpu` too. Subagents pass
  `--no-wait`; exit 75 means busy: they report it and do not retry. The run handoff gives the run plan as these
  commands, with the end time estimated again from `res-queue status`.
- **Frozen inputs live only in `/project/CoSiR`:** gitignored outputs of earlier rounds (records, held lists) are not
  in worktrees; use absolute paths.
- **Git:** `*.json` and `*.log` are gitignored (`git add -f` for check records); `git fetch .` fails here, so move
  main with `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD`. Commit only your own
  paths (`git commit -- <paths>`): the paper chat and the r18 and r19 chats commit on this repo too. zsh form:
  `git commit -F - <<'EOF'`.
- **Unattended:** delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing of this chat (no subagents, no queue job, no Codex job).
- Uncommitted: nothing of this chat's. Other chats' changes (`GLOSSARY.md`, r18's files) and older untracked files
  in `archive/` and `src/test/` are not ours.
- On disk over 1 GB: nothing new.
