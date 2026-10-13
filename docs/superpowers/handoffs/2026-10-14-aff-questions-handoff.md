# Handoff: finish the k-shot support-size sweep (Codex is running it), then the Leiden vs k-means check

## Brief
- Done: answered the user's seven questions about AFF; wrote the AFF model card with a training figure (f96acb2).
- State: the k-shot sweep (k = 2, 3, 4 example pairs, frozen AFF, seed 42) runs in Codex; the CLIP-text check is on hold.
- Next: check the sweep's three checks and numbers, write a short report, then check why Leiden for GoEmotions but k-means for CLIP.
- Blocked: nothing.

> Written 2026-10-14 00:06 by the "AFF questions" chat (a side chat, outside the r9 loop). Cut at about 210k tokens on
> the user's word while the Codex job was still running (an exception to loop-chats.md's "never cut during a
> background run"; the user asked).

## 1. Read in this order

1. This handoff.
2. The Codex brief of the sweep:
   `/tmp/claude-0/-project-CoSiR/c364095d-9895-4914-914d-0550c5482f65/scratchpad/codex_kshot.txt` (design, checks C1
   to C3, outputs). Its log: `codex_kshot.log` in the same folder; Codex thread `01a1374c-e107-7870-a745-d0f1903b866c`.
3. The AFF model card, `docs/reports/stage/2026-10-13_aff_model_card.md`, §3 and §4 (parts, fused score, frozen
   settings), for terms and the numbers the sweep is read against.

## 2. Decided by the user, not to be reopened

- **k-shot sweep** (2026-10-14): a small, cheap, quick sweep of the example-set size, in the spirit of k-shot
  learning (user). The 13 candidates stay as they are (user, after seeing the recommendation). AFF and every
  comparator stay frozen at their seed-42 settings, nothing re-fitted (user, after seeing the recommendation).
  k = 2, 3, 4 on the 12,288 seed-42 episodes, nested and paired (agent default).
- **AFF without CLIP-text** (question 1): on hold, since it changes the base of the design (user, 2026-10-14).
- **Questions 3 and 6** (why Leiden for the GoEmotions grouping but k-means for the CLIP groupings): a small check
  outside the research loop (user, 2026-10-14).
- **Model card:** written without the outside-view process, as a Markdown file only (user, 2026-10-13).

## 3. Where things stand

- **The model card** is committed (`f96acb2`, branch r9-scope) and indexed; its training figure is
  `docs/reports/assets/2026-10-13_aff_model_card/aff_training.png`, its fact sheet `training_facts.md` beside it.
- **The sweep:** Codex (gpt-6.1-sol, high) builds and runs it in `src/test/20261203_kshot_supports/` (module prefix
  `ks_`). Codex usage before the job: 5h 88% left, week 98% left. Expected outputs: `results/kshot_seed42.json` and
  `20261203_kshot_supports_log.md`. C1 is the key check: at k = 4 every scorer must equal the recorded seed-42
  values (AFF 19.137, B 18.341, B′(A0) 18.437).
- **Episode facts the user corrected on:** an episode is 4 + 4 + 13 (target, other-aspect candidate, 11 negatives),
  not 4 + 4 + 5; k = 1 is left out (the reader's spread features need 2 pairs); k > 4 needs new draws, and emotion has
  only 8 values, so 8 supports would repeat values.

## 4. Open points for the next step

1. **When the sweep ends:** check C1 to C3, read the numbers, then write a short auto report
   (`docs/reports/auto/v2/2026-12-03_kshot_supports.md`, CoSiR sequence date, index row) and a half-page user-read
   report (`docs/user_read/`), as a small loop's step 9 asks. Keep it cheap (the user's wish). Tell the user the
   result in plain words: does AFF's margin over B′(A0) or its condition gain change with k?
2. **Minor, ask the user:** if there is a trend, whether to add k = 6 and 8 (new draws, emotion repeats values).
3. **Questions 3 and 6:** first check the records (`docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`,
   `src/test/20261112_community_sweep/`, `src/test/20261116_grouping_step1_style/`) for any Leiden run on CLIP image
   or caption features. If none, propose a small check to the user: Leiden on k-NN graphs of CLIP image and caption
   features against k-means 64, measured as the 2026-11-11 told-oracle round did (told margin and reader margin over
   the matched control, seed 42). It is a question, not AFF tuning (the user stopped AFF tuning).

## 5. Code and pitfalls

- **Waiting for the Codex job:** this chat cannot get its notification. Its output file
  `/tmp/claude-0/-project-CoSiR/c364095d-9895-4914-914d-0550c5482f65/tasks/b2eigj27t.output` gets the line
  `codex exit N` when it ends; wait with a Monitor until-loop on that line. Do not use `pgrep codex`: the r9 build
  chats run Codex jobs too.
- **After the job:** check for new `.ccg` records, run `codex-usage`, add a row to `/project/claude-config/codex/jobs.md`.
  One fix round goes by `resume` of the thread above (`codex-jobs.md`); after that, Claude takes the task back.
- **Codex worked in the main tree**, not a worktree: the cached artefacts (A3 checkpoint, posteriors, picks) are not
  in git. It may write only new files under its folder; check that nothing else changed (`git status`).
- **Branch r9-scope is shared** with the r9 build chats: commit only your own files, by path.

## 6. State at handoff

- Running: the Codex k-shot job, started 2026-10-14 00:01 from the old "AFF questions" tab. Closing that tab may stop
  the job; the user is told to leave it open until the sweep ends.
- Uncommitted: nothing of this chat's (the two `src/test/20261129_r8_grouping/results*` folders are not ours).
- On disk over 1 GB: nothing.
