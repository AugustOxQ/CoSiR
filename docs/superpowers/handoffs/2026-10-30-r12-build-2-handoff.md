# Handoff: continue the r12 build (tickets 01, 03, 04, 05, 08, then 09, 10, 12 and the final review), then open the run chat

## Brief
- Done: 4 of 12 r12 build tickets merged into `r12-affection` (data and labels, clustering, stage 1 selection, the C13 re-derivation), each reviewed by Opus `high`; 292 fast tests pass.
- State: labels equal the spec's gate exactly; r10's held emotion block and r11's stage 4 scores and picks replay within 1e-9 through the new code.
- Next: tickets 01, 03, 04, 05, 08 in parallel (at most three at once), then 09, 10, 12, the final review, merge to main; then open the run chat.
- Blocked: nothing.

> Written 2026-10-30 22:08 by the r12 build chat (cut at the context meter, 200k). Loop step reached: step 7 (build),
> about one third done. Direction log entry: [2026-10-30](../direction_log.md).

## 1. Read in this order

1. The ticket index `.scratch/r12-affection/issues/00-index.md` (shared interfaces, updated after the reviews; rules
   for every implementer) and the open tickets 01, 03, 04, 05, 08, 09, 10, 12 beside it (`Status:` lines say which
   are done).
2. The spec, from §1 on: [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md).
3. `~/.claude/references/matt-chain.md` (the build chain), and `~/.claude/references/codex-jobs.md` before a Codex job.
4. The round log on the integration branch: `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md`
   (build decisions and readings so far). The first build handoff, [2026-10-30-r12-build-handoff.md](2026-10-30-r12-build-handoff.md),
   §3 and §5, for the data facts and pitfalls; they still hold.

Skills: `implement-spec` (per `matt-chain.md`; read its SKILL.md, it is slash-only).

## 2. Decided by the user, not to be reopened

All in the spec's §4 and the direction log, 2026-10-30; the spec was approved from its brief at 21:17.

## 3. Where things stand

| # | Ticket | State | Who built it |
|---|---|---|---|
| 01 | FI in the store: adapter, FI-only extraction, merge, hash gate | open | Claude Sonnet `medium` planned |
| 02 | Round data: keys, labels, keyed arm loaders | merged (127 tests + 1 skip until 05) | Codex, fix round by resume |
| 03 | Key-list extractor: instructed wordings, CLIP L/14, parity checks | open | Claude Sonnet `medium` planned |
| 04 | Frozen exports: c* (CPU), PercepT core (light GPU), Impressions c* | open | Claude Sonnet `medium` planned |
| 05 | CRL texts and vectors | open | Claude Sonnet `medium` planned |
| 06 | Clustering (verdict code) | merged | Claude Sonnet `high`, fixer Sonnet `high` |
| 07 | Stage 1 selection (verdict code) | merged | Claude Sonnet `high`, fixer Sonnet `high` |
| 08 | Seed ranges, SC table, verdict (verdict code) | open; needs an Opus `high` review | Codex or Sonnet `high` |
| 09 | Runner, stages 0a to 3 | open (after 01 to 08) | Sonnet `medium` |
| 10 | Diagnostics, Impressions figure | open (after 09) | Sonnet `medium` |
| 11 | C13 re-derivation `ac_vh.py` (verdict code) | merged (89 tests) | Codex, fix round by resume |
| 12 | Queue script and smoke run | open (after 09, 10) | Sonnet `medium` or controller |

- Every merged ticket had an Opus `high` review: no must-fix; the should-fix items were applied (details in the
  round log and the commit messages).
- Checks already run on real data (CPU): the label counts (spec §3.4 gate), 0a(iii)'s r10 part (within 1e-9) and
  r11 stage 4 part (seven arms, within 1e-9, same picks).
- Codex: the five-hour window resets at 23:52 (58% left at 22:04; weekly 65%). Today's jobs cost far less than the
  room estimate (an implementer about 10 to 13% of the window). Ledger rows are in
  `/project/claude-config/codex/jobs.md` (uncommitted there, beside another session's row; `sync.sh` at the wrap-up).

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- Integration worktree: `/project/CoSiR/.claude/worktrees/r12-int` on `r12-affection` (pushed). Main is untouched.
  Agent-tool worktrees start on main: each brief tells the implementer to `git checkout -B r12-tNN r12-affection`.
- Names fixed by the merged code (in the index): Affection arms in `ac_data.embed`; `ac_data.percept_prob` for
  PercepT's own reading; ArtELingo read-out arms `cstar`, `crl_clip_painting` (r11's `crl`), `crl_qwen_painting`;
  sidecar fields for views, exports and CRL texts. Ticket 05 must write the CRL `.json` fields 02 asserts
  (`texts`, `encoder`, `model_id`, `model_revision`, `instruction`, `template`); ticket 04's export sidecars must carry
  `weights_sha256` equal to r10's held c* and r11's held PercepT sidecars.
- `ac_cluster.paired(..., shared_seeds=False)` is the default (a seeded comparator draws its own seeds, r10's d_R);
  `ami` takes PercepT's own reading ([1, n]); `paired` does not.
- `ac_select.build_stage1` re-derives each table's scores and picks and refuses a mismatch; pass it the right r.
- `ac_vh.py` takes `--draws-clu` and `--draws-ret` (defaults 2,000 and 10,000); the runner's `--smoke` must use the
  same draws when vh checks a smoke run. It is single-process and recomputes stage 0b and stage 1 inside stages 2 and
  3: tens of minutes per stage, so the queue runs it as a CPU job.
- Slow tests: `test_ac_cluster.py -m slow` about 3 minutes; `test_ac_select.py -m slow` about 23 minutes (r11 stage 4
  replay). Run them in the background, not in a subagent. Stage 0a in the run chat reruns them as gate items.
- Spec §3.5 0a(i): `ac_store.py record` (ticket 01) must run and its hashes file be committed before FI is added; it
  is CPU-only, so the build may run and commit it.
- Unattended: delete nothing; the four ticket worktrees and six branches are listed on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's in CoSiR (`.scratch/` is outside git); `/project/claude-config/codex/jobs.md`
  has two new rows (sync at the wrap-up).
- On disk over 1 GB: nothing new. GPU check at 22:06: driver 595.99.02 answers, module and library agree, GPU free,
  no lock. Run mode: night.
