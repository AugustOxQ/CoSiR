# Handoff: r29 build 3: fix the import clash, fix and merge 02, build 05, 07, 08, then stage 2 (10 to 14) and 15, final review, merge, open the run chat

## Brief
- Done: tickets 01, 03, 04, 06, 09 reviewed, fixed and merged; 02 reviewed (pass after fixes). Gate (i)(c) passes on the local GPU.
- State: 107 of 113 round tests pass; 6 fail only when the text scorer's and the trainer's tests share a process (an import-path clash). Fix it first.
- Next: fix the import clash, Codex fix of 02, then tickets 05 and 07, then 08, then stage 2 (10 to 15).
- Blocked: nothing.

> Written 2026-11-12 23:36 by the "r29 build 2" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build) in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5: [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md), and
   build 2's section 5: [2026-11-12-r29-build-2-handoff.md](2026-11-12-r29-build-2-handoff.md). Both still hold, except
   build 2's gate (i)(c) flag, which is settled (below).
2. The tickets and index: `/project/CoSiR/.scratch/r29-siglip-only/issues/00-index.md` (Shared interfaces now has
   the gate part-file rule: `iv_store.json` from 02, `iv_hash.json` from 04, merged by 05) and the ticket files.
3. The notes (`/project/CoSiR/.scratch/r29-siglip-only/notes/`): reviews `t0N_review.md` and fix hand-backs
   `t0N_fix_handback.md` for 01, 02, 03, 04, 06; `brief_common.md` (implementer briefs), `review_common.md` (reviewer
   briefs); briefs ready to launch: `t05_brief.md`, `t07_brief.md`, `t02_fix_codex.txt` (Codex resume).
4. `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`, `~/.claude/references/loop-chats.md`.

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As in the first build handoff, section 2 (spec §4, D1 to D9 and D25). Run mode is night.

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, pushed: tickets 01, 03, 04 (each with review
fixes), 06 (with review fixes), 09, `r29_paths.ENCODER_CAPTION_SOURCES` (new), decisions.md (40 lines). Round tests: each file passes alone (image cells 40, text 34, trainer 40); together 6 trainer tests fail (step 1).

| Ticket | Branch, worktree | State |
|---|---|---|
| 01 | r29/t01 | merged (fix 6b5291a9, dd00044c; painting means bit-equal before and after) |
| 02 | r29/t02, /project/CoSiR-r29-t02, d3edba90 | Codex built (22 tests; GPU smoke pass); review pass after fixes; fix not started: `notes/t02_fix_codex.txt`, Codex session `01a1d1d2-9e9b-7070-b35f-ea7a45f963e0` (resume form in `codex-jobs.md`) |
| 03 | r29/t03 | merged; Codex's fix (3dfa96bf) needs a scoped re-review (Sonnet `high`, `review_common.md`, diff 109ebccf..3dfa96bf) |
| 04 | r29/t04 | merged (fix 7134fa21: NaN fails gate (ii), comparison tests, per-unit score provenance, `iv_hash.json`, arm `gate_i_a_new`, `R29_TEXT_SMOKE=1` for smoke) |
| 06 | r29/t06 | merged (fix faf26f94: validation-file hashes, write-once picks, r from the picks body, r11's loader checks, gate (v) validation part compares full tables at 1e-9); the full gate (v) image run is open |
| 09 | r29/t09 | merged; smoke: SigLIP 1 text pass 1.16 ms per caption, three towers 0.024 ms |
| 05, 07 | none | briefs ready (`t05_brief.md`, `t07_brief.md`); 05 needs 03 and 04 merged, 07 needs 04 |
| 08, 10 to 15 | none | not started; the stage 2 notes: ticket 12 must add an Affection key function to `check_store` (02 review); 10 maps `clip_l14` (trainer) to `clipl14` (sources) through `ENCODER_CAPTION_SOURCES` |

Gate (i)(c) on the local GPU (`--no-record`, 23:12): pass, 9 of 9 arrays bit-equal to r14's node402 files. The
recorded gate runs with the default devices (val cuda, held cpu) into a fresh `try<N>` folder.

Next steps, in order:
1. **The import clash (first).** `pytest -q src/test/20261112_siglip_only/` fails 6 trainer tests (for example
   `test_non_finite_outcome_raises`): `rs_paths.py:75` asserts `rs_vh` is imported from the worktree
   (`/project/CoSiR-r29/src/test/20261027_recipe_sweep/`), but the text scorer's chain (r28 -> r14's `bs_paths`, which
   points older rounds at the main checkout) imported it from `/project/CoSiR/...` first. Each file passes alone
   (trainer 40 of 40). Ticket 05 imports the trainer's and the text scorer's modules in one process, so fix it before
   05 (Sonnet `high` fixer on the integration branch: one place that sets the older rounds' paths in one order for every
   r29 module; the full round suite must pass in one process).
2. Codex: resume session `01a1d1d2-…` with `notes/t02_fix_codex.txt` when `codex-usage --room implementer` says room
   (one Codex implementer job at a time: two need 160%). Then a Claude scoped re-review of the fix, merge.
3. A scoped re-review of ticket 03's Codex fix (Sonnet `high`).
4. Tickets 05 and 07 (Sonnet `high` or Codex), then 08; then 10 to 14 (stage 2), then 15. At most three implementers.
5. Stage 0 gate runs, from the integration worktree after their tickets merge: gate (i)(c) recorded (GPU 1.5 min);
   gate (ii) (`r29_text.py score --arm qwen --set art_val --device cuda`, then `gate-ii`; GPU about 4 min); gate (v)
   image full (CPU 40 to 60 min, `t06_handback.md`); the prefixed store extraction and gate (iv) store part (GPU 6 to
   10 min, after 02's fix); gate (i)(a) (three GPU trainings of about 6 min). Run them in the build when the queue
   allows, else list them for the run chat.
6. Final whole-branch review (Opus `xhigh`), one fix wave, scoped re-review; merge into main, push; run handoff; GPU
   check; `loop-next` (first build handoff, section 5).

## 4. Open points for the next step

None for the user (night mode).

## 5. Code and pitfalls

- `r29_model.load_view(weights, device, *, expect_store_sha256)` now needs the store hash; `r29_captions._read`
  asserts the store's `smoke`, `backbone`, `progress.json` (`txt_done`, `gate.ok`, `index_sha256`). Synthetic test
  stores need those fields (`test_r29_captions.py` shows how).
- `decisions.md` is edited by fixers in ticket worktrees too: merges conflict there; keep both sides, drop duplicates.
- Codex's test runs left pytest basetemp folders `t03_review_tests_*` under `/data/SSD2/cosir_r29/20261112_siglip_only/`;
  Codex left `/project/CoSiR-r29-t02/.scratch/r29-t02-tests/`. List on `.scratch/pending_deletions.md` at the end.
- Codex ledger rows written: t02 (23:30), t03 fix (23:29). Every Codex job adds a row.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r29 after this handoff's commit.
- On disk over 1 GB: nothing. Small outputs under `/data/SSD2/cosir_r29/20261112_siglip_only/` (`smoke/`,
  `scratch_t03_ic/`, `scratch_gate_ic_gpu/`, `t03_review_tests_*`).
