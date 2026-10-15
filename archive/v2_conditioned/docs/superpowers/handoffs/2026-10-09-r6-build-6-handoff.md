# Handoff: run the final review's fix wave on round 6, its scoped re-review, the merge into main and the run handoff (unattended)

> Written 2026-10-09 20:30 by the `r6 build 5` chat, cut at a clean point because its context passed 200k tokens.
> Loop step reached: step 7, after all fifteen tickets and the final whole-branch review.
> Direction log entry: [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. `/project/CoSiR/.scratch/r6-held-test/notes/final_findings.md` (local build tracker, outside git): every finding
   of the final review and the ticket-15 review, and the **fix-wave list, items 1 to 5** (item 5 sits at the end of
   the file, after reviewer C's section).
2. The final review's area reports on `r6-held-test`, `src/test/20261125_artelingo_held_test/final_review/`
   (`fr_a.md` verdict path, `fr_b.md` scoring path, `fr_c.md` comparator, GPU jobs, descriptive pass, smoke): only the
   findings the fix-wave list cites. The review brief is `.scratch/r6-held-test/notes/final_review_brief.md`.
3. `.scratch/r6-held-test/notes/run_handoff_items.md`: this chat added the DTS budget reading, DTS rerun mechanics,
   smoke reruns, GPU waivers and reviewer C's run items.
4. Still binding: the build-5 handoff [2026-10-09-r6-build-5-handoff.md](2026-10-09-r6-build-5-handoff.md) §2, §4
   steps 3 to 6 and §5; the build-4 handoff §2 and §5; the build-3 handoff §5; the build-2 handoff §5 (what the run
   handoff must carry); the first build handoff §2 to §5.
5. `~/.claude/references/matt-chain.md` ("implement-spec, with our changes"), `~/.claude/references/loop-chats.md`
   ("Unattended chats").
6. The run log `src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md` on `r6-held-test`, rows
   from 17:50 on.

## 2. Decided, not to be reopened

- By the user: as the build-5 handoff §2 (spec §4 and rule approved 2026-10-09 04:53; build and run unattended; the
  run chat closes by opening `r6 read` with `--notify`; delete nothing in the unattended build).
- Agent defaults made in this chat (the run report must show them to the user):
  - **DTS "built" is a one-time event** (final review C1, ticket-15 review note 6). The stop judges the 24-hour
    budget on `results/dts_first_built.json` (written once by the first in-budget seed-42 chosen stage) while the
    rerun's setting, settings SHA and GPU output fingerprints are unchanged; otherwise on the rerun's time.
    `DTS_CLOCK_START` is pinned to 2026-10-09 12:29.
  - **Smoke GPU coverage**: a GPU family not given fails the smoke unless waived (`--waive <family>=<reason>`,
    recorded); a smoke-scale DTS sanity or stop failure retries seeds 9002 and 9003, then marks DTS missing.

## 3. Where things stand

- Integration branch `r6-held-test`, worktree `/project/CoSiR-r6`, head f147d70, pushed. **All 15 tickets merged**
  (ticket 15 at 96edb20 after one review and one fix round, checked by the controller).
- Final review (Opus, three areas in parallel on a fixed snapshot): **A, B and C each confirmed with fixes, no
  blocking finding.** Every load-bearing number was re-derived by the reviewers' own code and matched exactly (log
  rows 18:26, 18:31, 19:52). Should-fix A1 (apply step's smoke-record check), A2 (held runner's DTS stop guard), C1
  (budget on rerun time), C2 (clock start) and C3 (smoke without GPU jobs) are fixed in ticket 15's rounds (merged).
- **Left for the fix wave** (`final_findings.md`): 1 `results/value_sets.json` (rule §5.2; B1, should-fix); 2 test
  gaps (Holm quantities table, apply edge cases, the nine "did not beat" names); 3 the read's time-box (no first read
  after 2026-10-15; agent default); 4 provenance (sensitivity input SHA, git HEAD in `held_started.json`); 5 the
  `results/dts_settings.json` check (C4).
- CoSiR's `CLAUDE.md` test command now sets `MKL_NUM_THREADS=8` (build-5 handoff step 5, done). `.claude/` is
  gitignored in CoSiR, so the edit is on disk only; `sync.sh` backs it up at the wrap-up.
- **Comparator clock (rule §7.7): started 2026-10-09 12:29; DTS must be built by 2026-10-10 12:29.** Open the run chat
  by about 02:00 at the latest. If the fix wave or re-review would run past that, log it and tell the run chat to start
  rule §6 items 1 to 6 on the reviewed code.

## 4. Next steps

1. **Fix wave**: one Opus implementer (`isolation: "worktree"`, reset onto `r6-held-test`), the five items of
   `final_findings.md` with tests and guards mutated on copies. The controller reads the diff and reruns the touched
   suites, then merges.
2. **Scoped re-review** (Opus): the fix wave's diff plus ticket 15's fix-round commits 59dd4d9 and a186c2c (checked
   by the controller only: GPU waivers, DTS retries, record-guard tests, stop binding, `dts_first_built.json`, pinned
   clock start). One fix round if needed.
3. Merge `r6-held-test` into main and push (build-5 handoff §4 step 3). Add the fix wave's worktree to
   `pending_deletions.md`; remove nothing.
4. Write the run handoff (build-2 handoff §5 list, build-4 handoff §5, `run_handoff_items.md`, and the final review's
   verdict with its reports' paths), then `loop-next <handoff> --label "r6 run" --unattended` (no `--notify`).

## 5. Code and pitfalls (new in this chat)

- **Interim notifications**: a finished reviewer with a leftover background job resent its report six times. Once its
  report is recorded, stop it with `TaskStop` (deferred tool; load it with ToolSearch).
- **Parallel reviews on a snapshot**: a detached worktree at a fixed commit (`/project/CoSiR-r6-fr`) let the final
  review run while ticket 15 was built and merged elsewhere. Reviewers write `final_review/fr_<area>.md` there; the
  controller copies them into the integration branch.
- A finished implementer resumes with `SendMessage` to its agent ID (fix rounds). The IDs do not carry over to a new
  chat; the fix wave needs a fresh implementer.
- `test_r6_common.py`'s worktree tests run `git worktree remove` inside their fixture (on their own temporary
  worktree); reviewer B deselected them. They are not a deletion by the chat.

## 6. State at handoff

- Running: nothing. No subagent, no GPU job, no DAS6 job.
- Uncommitted: nothing; `.scratch/` is local by design.
- Storage (nothing deleted; for the next storage summary): `pending_deletions.md` gained three rows: ticket 15's
  worktree and branch (70 MB), the review snapshot worktree `/project/CoSiR-r6-fr` (71 MB, detached at bceff9c), and
  this chat's session scratchpad (243 MB).
