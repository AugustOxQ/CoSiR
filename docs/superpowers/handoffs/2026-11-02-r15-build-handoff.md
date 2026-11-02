# Handoff: build r15 (c* on MMArt's unseen paintings, four encoders) from its approved spec, then open the run chat

## Brief
- Done: r15 specced, rule-reviewed and approved (2026-11-02 18:37); MMArt's seven views extracted, all checks pass.
- State: test cut moved to 0.95 by the user (spec D19): 16,384 test paintings; the 0.85 set (10,403) is a diagnostic.
- Next: tickets and build on CPU (no GPU expected); final review; merge into main; then open the run chat.
- Blocked: nothing.

> Written 2026-11-02 21:05 by tab "r15 decide". Loop step reached: after step 6 (spec approved), before step 7.
> Direction log entry: [2026-11-02 · Backbones sweep read ... next, the MMArt (PolyArt) test (r15)](../direction_log.md).

## 1. Read in this order

1. The spec, from §1 on: [2026-11-02-mmart-test-design.md](../specs/2026-11-02-mmart-test-design.md). Its decision
   rule is §6 with §3.4 and §3.5; it inherits r14's part a (its header links the r14 spec). D19 (the 0.95 cut) is
   the newest decision; §3.3 and §3.6 already carry it.
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any
   Codex job.
3. The facts brief, `src/test/20261102_mmart_test/mt_facts_brief.md` (labels, stores, checkpoints, scoring code,
   timings).
4. The rule review, `src/test/20261102_mmart_test/mt_rule_review.md`: the stored paths it checked (the validation
   teacher views in the ArtELingo `<view>_s2img` stores, the 20 PercepT mapper files) and the evidence behind each fix.
5. The MMArt prep log, `src/test/20261102_mmart_prep/20261102_mmart_prep_log.md`, section "Run": the stores and the
   near-copy counts.

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`; slash-only, so read their SKILL.md).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D19) and the direction log, 2026-11-02. The spec was approved from its brief at 18:37,
with no container restart. D19 (the test cut at 0.95, the 0.85 set as the far-subset diagnostic in place of the
uncut set) was decided at 21:00, after the near-copy counts and before any test score was read.

## 3. Where things stand

- **Stores ready:** seven views for the 16,782 new paintings under `/data/SSD2/pre_extract/mmart/` (Qwen default,
  emotion_focus, style_word, genre_word, SigLIP 2, SigLIP 1, CLIP L/14), plus 200-painting parity stores. Parity
  min cosine 0.99935 to 1.0 (bar 0.999); store check 14 of 14 (`src/test/20261102_mmart_prep/mp_*.json`).
- **Near-copy file:** `/data/SSD/mmart/mmart_nearcopy.parquet` (columns `key`, `max_cos`, `nearest_painting`; one
  row per test key). Counts: ≥ 0.85, 6,375; ≥ 0.90, 2,249; ≥ 0.95, 390; ≥ 0.99, 49. The 34 exact copies all have
  cosine ≥ 0.9995.
- **Cut sets** (computed in r15 decide with the spec's order: exact copies, then the cosine cut, then the md5 groups
  in sha256 key order): 0.95, 16,384 paintings (the main test); 0.85, 10,403 (the far subset). The build recomputes
  both; stage 0's gate (i) checks them.
- **Index:** `/data/SSD/mmart/mmart_index.parquet` (column `split`: new 16,782, val 5,769, held 11,528, train
  40,155; `exact_dup_trainval`, `md5`, `emotion`, `style`, `genre`).
- **Run mode:** night. GPU check at 21:00: driver 595.99.02 answers, the GPU is idle, no lock held. CPU idle (load
  0.03), 55 GB RAM free.
- **Codex:** at 21:00 `codex-usage --room implementer review` said no room (a job needs 110% of the five-hour window);
  the window resets 2026-11-03 02:00. Claude implements until then; check again after 02:00.
- **Timing:** build about 3 to 4 h from about 21:15, so it ends about 01:00; the run (CPU 3½ to 4½ h) then ends about
  05:30, well before 11:00. The report is due 11-03, at the latest 11-04 12:00.

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to
the run chat.

## 5. Code and pitfalls

- Round folder `src/test/20261102_mmart_test/` (it holds the facts brief and the review), module prefix `mt_`.
  Outputs under `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/`.
- **Branch:** start the integration branch from `main` (now 6e7c1a9). The r14 branch `r14-backbones-sweep` is on
  `.scratch/pending_deletions.md`, kept "until the r15 build branch starts"; leave it, and leave the row for the
  storage summary.
- **Verdict code** gets unit tests and an Opus `high` ticket review: the test cut (both thresholds), the export and
  matched-arm paths, the picks, the scorer and the outcome code with the inconclusive paths (spec §9). The rest gets
  the controller's diff read and one end-to-end smoke run.
- **Keys:** joins by key, never by store position (C6). MMArt keys start with the style folder, so evaluation order
  is sha256 of the key string (D9); no code reads a label from a key.
- **Frozen inputs live only in `/project/CoSiR`:** `src/test/20261020_claim_test/outputs/held/held_paintings.json`
  and other gitignored outputs are not in worktrees; use absolute paths.
- **Git:** `*.json` is gitignored (`git add -f` for check files); `git fetch .` fails here, so move main with
  `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD`; commit only your own paths
  (`git commit -- <paths>`), since other chats (the paper chat) also commit on this repo.
- **Unattended:** delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (the extraction ended 19:55; no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's (older untracked files in `archive/` and `src/test/` predate it).
- On disk over 1 GB: nothing new. The MMArt stores (SSD2, shared cache, about 0.4 GB) are kept.
