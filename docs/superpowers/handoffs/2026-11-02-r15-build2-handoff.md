# Handoff: finish the r15 build (review and merge ticket 05; build 02, 07, 08; final review; merge into main), then open the run chat

## Brief
- Done: tickets 01 (MMArt data, cut sets), 03 (matched arms), 04 (clustering scorer) and 06 (outcome code) built, reviewed and merged; 05 (retrieval and picks) built, not reviewed.
- State: cut 16,384 / 10,403 as the spec; gates (i), (iv), (v) pass on real data; replay 0(iii) clustering equals r14 exactly (difference 0).
- Next: Opus review of 05 and its full replays; then tickets 02, 07, 08; the final review; merge; open the run chat.
- Blocked: nothing.

> Written 2026-11-02 21:46 by tab "r15 build" (cut at the context meter, 200k). Loop step reached: step 7 (build),
> 4 of 8 tickets merged. Direction log entry: [2026-11-02 · Backbones sweep read ... next, the MMArt (PolyArt) test (r15)](../direction_log.md).

## 1. Read in this order

1. The previous build handoff, sections 3 and 5: [2026-11-02-r15-build-handoff.md](2026-11-02-r15-build-handoff.md)
   (stores, timings, git pitfalls). The spec: [2026-11-02-mmart-test-design.md](../specs/2026-11-02-mmart-test-design.md),
   from §1 on.
2. The ticket folder `/project/CoSiR/.scratch/r15-mmart-test/` (outside git): `issues/00-index.md` (tickets, shared
   interfaces, coverage line, and "Controller decisions", binding), the tickets, and `notes/`:
   `controller_notes.md` (the timeline and every note for later tickets), `implementer_brief.md`,
   `reviewer_brief.md`, `code_map.md` (sections A to G checked; H, the r14 runner and diagnostics, not checked).
3. `~/.claude/references/matt-chain.md` (the final review), `~/.claude/references/codex-jobs.md` before any Codex job.

Skills: none (to-tickets and implement-spec are read from their SKILL.md; the tickets exist).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D19) and the direction log, 2026-11-02. Nothing new in this chat.

## 3. Where things stand

Integration branch `r15-mmart-test` (from main 88cb70f), pushed. Round folder `src/test/20261102_mmart_test/`.

| # | Ticket | State | Branch, worktree |
|---|---|---|---|
| 01 | `mt_data.py` | merged (Opus review, fix e7e9e4e) | `r15-t01-mmart-data`, `.claude/worktrees/agent-a2c84c0767bebbac6` |
| 02 | `mt_export.py` | not started | |
| 03 | `mt_matched.py` | merged (Opus review, fix 5c6c845) | `r15-t03-matched`, `.claude/worktrees/agent-a8de6ae0cfc1fbabf` |
| 04 | `mt_clu.py` | merged (Opus review, fix 845a467) | `r15-t04-clustering`, `.claude/worktrees/agent-a16958baa96d65e17` |
| 05 | `mt_ret.py` | built (59534e1, dff9fa5, merge 5281eb7; 21 tests), not reviewed | `r15-t05-retrieval`, `.claude/worktrees/agent-ae5b3b091c14d16f9` |
| 06 | `mt_outcome.py` | merged (Opus review, fix 7b179e2) | `r15-t06-outcomes`, `.claude/worktrees/agent-a42649942dc784447` |
| 07 | runner | not started | |
| 08 | diagnostics | not started | |

Real-data checks so far (build checks in scratch outputs; the run chat runs the gates for the record):
- Cut: 16,384 at 0.95 (34 exact, 356 near, 8 md5), 10,403 at 0.85 (34, 6,341, 4); an independent rebuild by the 01
  reviewer gives the same lists. `gate_i()` and `gate_v()` pass, with `parts` per D17 part.
- Gate (iv): all 20 mappers on CPU within 8.3e-7 of the stored held probabilities; stored partitions and
  `reported_stage4` readings exact; no retrain needed.
- Replay 0(iii), clustering part, 2,000 draws: SigLIP 2 SC1, SC3, SC6 and Qwen 12-28 values and intervals, max
  difference 0 (897 s; `/tmp/claude-0/-project-CoSiR/4a2a241b-447c-4748-91bb-93507e8ef7e4/scratchpad/build_checks/t04_replay/stage0/gate_iii_clu.json`).
- Tests on the integration branch: 198 pass (`pytest -q src/test/20261102_mmart_test/`, 45 s).

## 4. Open points for the next step

None for the user. One agent default to carry into ticket 08 and the report (D14, descriptive): the unseen-artist
subset holds 15 paintings by 12 artists (ArtELingo train list 42,969 paintings, `ct2_data.split2('full')` fit), fewer
than the 20 clusters, so the diagnostic reports the count and the artists only, with no cells; the report says why.

## 5. Code and pitfalls

- **Next steps, in order:** (1) Opus `high` review of 05 with `notes/reviewer_brief.md`; run its full replays
  0(iii) retrieval and 0(vi) in the background in scratch outputs (`MT_OUTPUTS`), as for 04; fix round; merge.
  (2) Ticket 02 (Sonnet `high`, verdict code), then its review. (3) Tickets 07 (Sonnet `medium`) and 08 (Sonnet
  `high`, descriptive), at most three implementers at once. (4) The end-to-end smoke (ticket 07). (5) The final
  whole-branch review, Opus `xhigh` (`matt-chain.md`), one fix wave, a Sonnet `high` re-review. (6) Merge into main
  (`git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD` after merging; or merge main in
  first if it moved), push, write the run handoff with the ticket list (`matt-chain.md`, "The record"), open the run
  chat with `loop-next ... --unattended`.
- **Codex** had no room until 2026-11-03 02:02. Check `codex-usage --room` for any job after that; C13 (the
  re-derivation) is the run chat's Codex job.
- **Ticket 05 as handed back:** replay 0(iii) retrieval passed with max difference 0 on full draws, but run as two
  parts (`--parts r14`: SigLIP 2, 57 values; `--parts r11`: Qwen, 51 values), so each file says `passed false,
  complete false, values_ok true`; run `mt_ret.py replay-iii` whole (6 to 10 min, expect PASS, 108 values). 0(vi)
  checked only for the `instructed` and `percept` arms (difference 0); run `mt_ret.py replay-vi` whole (15 to 25 min).
  Both in the background with `MT_OUTPUTS` set to a scratch folder. Real-data smoke on validation: SigLIP 2 frozen and
  teacher picks `center+k10` (stand-in ranges); `score_encoder` on the 16,384-row cut ran with a stub export.
  Schemas for 07: `ret_ranges_<enc>.json` `{encoder, kind, ranges, per_seed, sources, n_val, labelled}`;
  `picks_<enc>.json` `{encoder, arms: {cstar, frozen}}`; `picks_teacher.json` `{ranges_encoder: 'qwen', arms:
  {teacher}}`, each arm `{pick, table, ranges, dropped}`; the scorer reads `stage1_<enc>.json` (`ret_ranges`,
  `picks`) and `stage1_teacher.json` through `read_committed_record`. Order: `ranges` for all four, `pick` each,
  then `pick-teacher` (needs Qwen's ranges). The review should check: picks read validation only; the teacher pick
  uses Qwen's mAP ranges; each c* seed's map is fitted on its own validation export; the comparator readout.
- **Notes for ticket 02** (from the 04 review): sidecar field `sha256` (04's ranges file copies it);
  `mt_export.load` must check the sidecar sha256 against the file and refuse a stale export (checkpoint sha256),
  as 03's `load_probs` does.
- **Notes for ticket 07:** status schema in the index (`gates.i_default`, `i_teacher`, `v` per part from
  `gate_i()['parts']`, `gate_v()['parts']`); 04's stage 1 file shape `{encoder, ranges: {aspect: r}, per_seed,
  sources, n_val, labelled}` merges unchanged as `clu_ranges`; SC4 cells of a voided encoder carry their `r` or are
  left out (06's `check_cell` refuses `r` null); scoring refuses without committed stage 1 records
  (`read_committed_record`), so the smoke runs with `MT_SMOKE=1`; `mt_matched.py export --enc all` takes 15 s and
  `gate-iv --enc all` 70 s; `mt_clu.py replay-iii` 15 min.
- **Notes for ticket 08:** the unseen-artist default above; `mt_clu.score_encoder(enc, keyset, scs)` writes
  `OUT/diag/cells_clu_<keyset>.json` for non-test keysets; `shared_held` has no SC3.
- **Worktrees** (the Agent tool's `isolation: worktree`) sit under `/project/CoSiR/.claude/worktrees/`; after the
  merge into main, list them and their branches on `.scratch/pending_deletions.md` (unattended: delete nothing).
- Git: commit only your own paths; `*.json` is gitignored (`git add -f` for records).

## 6. State at handoff

- Running: nothing (subagents, background jobs and the watch all ended).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: nothing new (scratch outputs under the session scratchpad, a few hundred MB at most).
