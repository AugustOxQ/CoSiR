# Handoff: continue the r12 build (08 review fixes, 13 smoke mode and 09 runner, then 10, 12, the final review), then open the run chat

## Brief
- Done: 9 of 13 r12 build tickets merged into `r12-affection` (01, 03, 04, 05, 08 in this chat); 433 fast tests pass.
- State: early gate runs pass: 0a(iv) 6.1e-5 (bar 1e-3), 0a(v) all five PercepT hashes equal, 0a(ii)(a) cosine 0.99982, 0a(vi) exact, 0a(vii) hashes equal, r11 stage 5 replay within 1e-9.
- Next: the 08 review fixes, then tickets 13 (smoke mode, new) and 09 in parallel, then 10, 12, the final review, merge to main.
- Blocked: nothing.

> Written 2026-10-30 22:33 by the r12 build chat 2 (cut at the context meter, 200k). Loop step reached: step 7 (build),
> about two thirds done. Direction log entry: [2026-10-30](../direction_log.md).

## 1. Read in this order

1. The ticket index `.scratch/r12-affection/issues/00-index.md` and the open tickets 09, 10, 12 and the new 13
   (`Status:` lines say which are done). Then `.scratch/r12-affection/notes/08-review-fixes.md` (the 08 fix list).
2. The spec, from §1 on: [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md).
3. `~/.claude/references/matt-chain.md` (the build chain), and `~/.claude/references/codex-jobs.md` before a Codex job.
4. The round log on the integration branch: `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md`
   (build decisions, readings and every check so far). The first build handoff,
   [2026-10-30-r12-build-handoff.md](2026-10-30-r12-build-handoff.md), §3 and §5, and the second,
   [2026-10-30-r12-build-2-handoff.md](2026-10-30-r12-build-2-handoff.md), §5: their pitfalls still hold.

Skills: `implement-spec` (per `matt-chain.md`; read its SKILL.md, it is slash-only).

## 2. Decided by the user, not to be reopened

All in the spec's §4 and the direction log, 2026-10-30; the spec was approved from its brief at 21:17.

## 3. Where things stand

| # | Ticket | State | Who built it |
|---|---|---|---|
| 01 | FI in the store: adapter, FI-only spec, `ac_store` record and merge | merged; the old store's hashes recorded and committed (`cc2a102`) | Sonnet `medium` |
| 03 | Key-list extractor `ac_extract.py` | merged | Sonnet `medium` |
| 04 | Frozen exports `ac_export.py` | merged | Sonnet `medium` |
| 05 | CRL texts and vectors `ac_crl.py` | merged | Sonnet `medium` |
| 08 | Seed ranges, SC table, verdict `ac_stats.py` (verdict code) | merged after an Opus `high` review: pass, no must-fix; 4 should-fix test gaps open | Sonnet `high` |
| 09 | Runner, stages 0a to 3 | open, unblocked | Sonnet `medium` planned |
| 10 | Diagnostics, Impressions figure | open (after 09) | Sonnet `medium` |
| 12 | Queue script and smoke run | open (after 09, 10, 13) | Sonnet `medium` or controller |
| 13 | Smoke mode for the loaders (new, verdict code) | open, unblocked | Sonnet `high`, Opus `high` review |

02, 06, 07 and 11 were merged by build chat 1.

- Early gate runs in this chat (scratch outputs; the run chat's stage 0a reruns all of them into the real outputs):
  0a(iv) max difference 6.1e-5 over 15 heads (bar 1e-3); 0a(v) PercepT hashes equal for seeds 0 to 4 and AMI
  0.08742922158584242 (equal); 0a(ii)(a) min cosine 0.999817 (bar 0.999); 0a(vi) cosine 1.0, difference 0; 0a(vii)
  CLIP-CRL vectors hash-equal on three aspects, Qwen texts min cosine 0.99952 (bar 0.999); 0a(iii) r11 stage 5 part
  within 1e-9 (250 s). The three views sealed on 64-image smokes (instructed Qwen about 13 images/s).
- Codex: no room for an implementer at 22:31 (5-hour window 58% left, needs 80%); it resets at 23:52 (weekly 65%).
  After the reset, Codex can take 09 or 10 (`codex-usage --room implementer`).

## 4. Open points for the next step

None for the user. One plumbing decision made here (agent default, changes no number): ticket 12's planned smoke
cannot run as written, since `ac_data` and `ac_vh` refuse smoke-flagged views and exports and the full views need the
run chat's heavy GPU steps. Ticket 13 adds an explicit smoke mode (`AC_SMOKE=N`, first N keys per split, smoke views
under a non-default `AC_OUTPUTS`); it touches keys, so it is verdict code with an Opus `high` review. Give 09's
implementer ticket 13's interface so `--smoke N` uses it.

## 5. Code and pitfalls

- Integration worktree: `/project/CoSiR/.claude/worktrees/r12-int` on `r12-affection` (pushed). Main is untouched.
  Agent-tool worktrees start on main: each brief tells the implementer to `git checkout -B r12-tNN r12-affection`.
- **Committed records need `git add -f`**: the repo's `.gitignore` ignores `*.json`, so `stage0a_store_hashes.json`
  (done), `stage0b_ranges.json` and `stage1_picks.json` are force-added; 09's guard reads them from `git HEAD` and 12's
  commit steps must use `-f`.
- Stage 0a's GPU items and the JSON names their code writes under `OUTPUTS/stage0a/`: `store.json` (01, merge),
  `extractor_artelingo.json`, `extractor_default_affection.json`, `clip_l14_artelingo.json` (03, parity a, b, c),
  `cstar_reinfer.json`, `percept_loader.json` (04), `crl.json` (05). Their commands are in each module's docstring.
  `ac_crl.py check` needs `ac_crl.py texts` first.
- 0a(ii)(b) and the merge need FI in the store; the run chat runs them after the FI-only extraction. `ac_store merge`
  refuses a smoke FI store and an existing `--out`.
- From the 08 review, for 09: assert `dropped == 0` in full (non-smoke) runs of `ac_stats.arm_retrieval`, as r11's
  `score_arm`; stage 3's verdict file (or the report) carries spec §6's note that SC5 to SC7 were set after r11's
  descriptive held CRL numbers.
- `ac_vh.py` smoke takes the first N sorted keys (default 2,100) and `--draws-clu`, `--draws-ret`; the runner's smoke
  must use the same draws when vh checks it.
- Slow tests (run in the background, not in a subagent): `test_ac_select.py -m slow` about 23 min,
  `test_ac_stats.py -m slow` about 4 min, `test_ac_cluster.py -m slow` about 3 min.
- Workers must not run CPU jobs over about 4 minutes (the 05 worker ran a 15-minute CPU Qwen encoding and stopped it):
  say so in each brief.
- Unattended: delete nothing; this chat's worktrees and branches are on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no GPU lock, no background command).
- Uncommitted: nothing of this chat's in CoSiR (`.scratch/` is outside git). `/project/claude-config/codex/jobs.md`
  still has build chat 1's two rows (sync at the wrap-up); this chat ran no Codex job.
- On disk over 1 GB: nothing new. GPU check at 22:31: driver 595.99.02 answers, module and library agree, GPU free,
  no lock. Run mode: night.
