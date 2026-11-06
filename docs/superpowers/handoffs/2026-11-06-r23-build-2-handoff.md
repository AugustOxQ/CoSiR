# Handoff: r23 build 2: fix ticket 2, review tickets 3, 4 and 7, build tickets 5, 6, 8, 9, 10, final review, merge

## Brief
- Done: tickets 1, 2, 3, 4 and 7 are built; ticket 1 is reviewed and merged.
- State: no held score exists. Forward gate (ii) passes on all 15 weight files: largest difference 6.1e-5 against a bar of 1e-3.
- Next: fix ticket 2 from its review, review tickets 4 and 7, merge, then build tickets 5, 6, 8, 9 and 10.
- Blocked: nothing. The cut came from the context meter (206k tokens), not from a problem.

> Written 2026-11-06 05:40 by tab "r23 build". Loop step reached: inside step 7. Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. `/project/CoSiR/.scratch/r23-text-claim/issues/00-index.md`: ticket list, shared interfaces, coverage line. Tickets sit beside it, outside git. Tickets 05 and 06 end with follow-ups from ticket 1's review.
2. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule. Skip the brief.
3. `/project/CoSiR/.scratch/r23-text-claim/notes/t02_review.md`: the fix list for ticket 2.
4. `~/.claude/references/matt-chain.md` and `codex-jobs.md`: the chain, and Codex's launch form.
5. The [first build handoff](2026-11-06-r23-build-handoff.md) §5: code pitfalls, all still true.

Skills: tdd (verdict code).

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. A change to the rule is an amendment (hard rule 5).

## 3. Where things stand

| Ticket | Branch, commit | State | Next |
|---|---|---|---|
| Setup | `r23-text-claim` | `tcl_paths.py`, `conftest.py`, `pytest.ini` merged | none |
| 01 metrics | merged, 50270ae9 | Opus review passed: own plain-loop interval agrees to 2.8e-17; 31 of 32 mutations caught | done |
| 02 data door | `r23-t02`, 6ca11f27 | 22 tests pass; review passed the numbers, asks for fixes | fix round, then merge |
| 03 extraction | `r23-t03`, 59064446, on t02 | smoke passes on every job; not merged | controller reads the diff, merge after t02 |
| 04 forward | `r23-t04`, 63501485, on t02 | 5 tests, gate (ii) passes | Opus `high` review, merge after t02 |
| 07 stage 1 | `r23-t07`, 29b49859, on t01 | 18 tests by Codex | Opus `high` review, merge |
| 05, 06, 08, 09, 10 | none | not started | frontier opens as above: 05 and 08 next |

Integration worktree: `/project/CoSiR/.claude/worktrees/r23-int`, branch `r23-text-claim`, pushed. Ticket worktrees `.claude/worktrees/r23-tNN`; their branches are local only.

## 4. Open points for the next step

None for the user. Notes for later, no action in this chat:
- Spec D9's ArtELingo genre share, 2.8%, is the share over all rows; on labelled rows it is 3.5%. The other D9 values are labelled-row shares, and the licensed claim quotes only MMArt's values, which are right. The bar is untouched, so no amendment. The report gives 3.5% with a one-line note; stage 3 prints it again.
- Ticket 04's acceptance line "a held label is never read" holds for `tcl_forward.py`, not for `tcl_data.frame`, which loads held labels when a held row is built. No effect: held scoring reads those labels anyway, and no pick reads held rows.

## 5. Code and pitfalls

- Create worktrees by hand, based on the integration branch (`git worktree add .claude/worktrees/r23-tNN -b r23-tNN r23-text-claim`), and name the absolute path in the brief. Agent `isolation` would base them on main. Merges: the controller, in `r23-int`.
- Codex: 91% left of five hours at 05:40, reset 09:53; weekly 48%. `--room implementer` says room while over 80% is left; two at once say no room. Real cost was 4 to 5% per ticket against the estimate of 40%. So one Codex job at a time, Claude workers beside it. Its log's last `agent_message` holds the note.
- Reviews: Opus `high`, 11 to 13 minutes each. Ask for mutation runs: they found ticket 2's weak tests, while its numbers were right.
- `fit_means(t, i)` returns (text, image); `evaluate` takes (query, gallery). In i2t the pair is flipped by hand; nothing asserts it (ticket 05's note).
- Text rows are in sample-id order, so rows of one painting are not adjacent. Everything joins through `tp`.
- `tcl_data` extras: `store_base`, `art_teacher_path`, `mm_store_path`, `read_store`, `image`, `label_share`. Ticket 03 writes through them.
- `gate_i.json` keys are `<set>/<kind>/<instruction>/<row>`, 39 stores (index). Ticket 08 parses them to void per store. A store not yet written fails.
- Gate (i) parity bar is min cosine >= 0.999; smoke's lowest was 0.99936, on `art_held` genre_word. Thin, so a miss needs a trace before any verdict is voided.
- Gate (iii)'s Table 2 clause has a thin margin: 2.65e-4 against 3e-4. Fixed seeds keep it stable.
- `n_truncated` counts prompts over 512 tokens; a text of exactly 512 is not cut.
- C13 is the run chat's Codex job, as r15: `tcl_run.py verdicts` needs `records/c13.json` with hashes. Do not build it here.
- After the final review (`opus` `xhigh`) and the scoped re-review (`sonnet` `high`): merge into main, push main, list worktrees and ticket branches on `.scratch/pending_deletions.md`, then cut to the run chat with the ticket list in its handoff.
- Run timing (D13): heavy jobs that would fall between 10:00 and 19:00 on a weekday wait for 19:00. Report by 2026-11-07 12:00, at the latest 2026-11-08. Extraction is shorter than the spec's estimate, by smoke rates: about 1 h.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat, apart from the Codex ledger rows in `/project/claude-config/codex/jobs.md`, which the next `sync.sh` commits.
- On disk over 1 GB: nothing new. Smoke outputs sit in this chat's scratchpad, `/tmp/claude-0/-project-CoSiR/01e916db-e2c8-4a25-9b9a-4b04b71498d3/scratchpad/`, under 100 MB by the smoke sizes (not measured).
- GPU check before this unattended chat: passed. Driver 595.99.02 in the module and the library.
