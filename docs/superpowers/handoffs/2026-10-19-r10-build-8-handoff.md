# Handoff: finish the stage 2 build (tickets 03 and 04, GPU smoke, final review, merge, stage 0), then open the run chat

## Brief
- Done: tickets 01 (scorer with AMI, V, ARI, FMI) and 02 (trainer core) built by Codex and merged into the integration branch.
- State: stage 1's genre head is reproduced bitwise on CPU (2 epochs; best epoch 1); PercepT core AMI 45/45 equal.
- Next: tickets 03 and 04 to Codex in parallel, a GPU smoke, the final review, merge to main, stage 0.
- Blocked: nothing.

> Written 2026-10-19 20:18 by "r10 build 7" (cut at the context meter, 200k). Loop step reached: step 7 (build),
> two of four tickets merged. Direction log entry:
> [2026-10-19 · PercepT core with K = 40 read](../direction_log.md).

## 1. Read in this order

1. [build 7 handoff](2026-10-19-r10-build-7-handoff.md) §2, §4 and §5 (decisions, open points, pitfalls); skip the
   rest, this file replaces it.
2. The ticket index `.scratch/r10-model-sheet/issues/00-index.md` (shared interfaces, coverage line), then tickets
   `03-trainer-extensions.md` and `04-stage0-sheet-queues.md`. Tickets 01 and 02 are done (set their `Status:` line
   to `done`).
3. The round log `src/test/20261019_model_sheet/20261019_model_sheet_log.md` on branch `r10-model-sheet`: the
   build's agent defaults 1 to 11 (D12 epochs, form b, form c at the CSN code's defaults, distillation, D1 batches,
   seed picks, queues) and the timeline.
4. Code to brief against (integration worktree `/project/CoSiR-wt-msint`): `ms_train.py` (seams `build_model`,
   `head_loss`; config keys), `ms_rows.py` (`validate` still rejects all but form a), `ms_score.py`
   (`measures`, `bar_vectors`, `network_vectors` reads `config['graph_row']`, `sample_sha256`, `input_img`).
5. The Codex briefs of this build: `.scratch/r10-model-sheet/notes/codex_common.txt` (the rules block every brief
   ends with) and `codex_brief_02.txt` (an example head). The spec and `matt-chain.md`, `codex-jobs.md` as in build 7.

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec, approved as written (2026-10-19 19:58); no container restart before the build (user, 19:58).

## 3. Where things stand

- Branch `r10-model-sheet` (pushed, `11fded3`), worktree `/project/CoSiR-wt-msint`: ticket 01 `ms_score.py`, ticket
  02 `ms_rows.py` and `ms_train.py`, their build logs `ms_01_build.md`, `ms_02_build.md`, the round log.
- Ticket 01 smoke: PercepT core P1 to P5, 3 seeds, 3 labels: 45/45 AMI equal at 6 decimals; back-fill 22 of 24 cells
  equal to the K = 40 table at 3 decimals; emotion and style network cells wait for stage 0's arrays.
- Ticket 02 smoke: separate genre R0o seed 0, all 62,906 fit pairs, 2 epochs on CPU: train and dev losses `==` and both
  float16 arrays bitwise equal to `outputs_fu/genre__R0o__s0.npz`. T1 `--smoke`: three heads, valid outputs.
- Not checked: the CUDA path of `ms_train.py` (no GPU smoke yet).
- Codex at 20:16: five-hour window 58% left (resets 20:48), weekly 70%. Ledger rows for 01 and 02 are in.

## 4. Open points for the next step

No user question is open. Steps for this chat, in order:
1. Worktrees `/project/CoSiR-wt-ms03` and `-ms04` from `r10-model-sheet` (`git worktree add -b r10-ms-03-ext` …),
   each with the sample symlink (below); Codex 03 at `gpt-6.1-sol` high, 04 at medium, in parallel
   (`codex-usage --room implementer implementer` first).
2. Meanwhile, the GPU smoke of ticket 02 under the lock: `ms_train.py --rows T1 --smoke --device cuda` (and A1).
3. Diff read and smoke per ticket, merge (03 before 04 if both touch nothing in common, any order); GPU smoke of
   the 03 rows (`--smoke --device cuda`: T4, T5, D1, D2, D5, I1).
4. Final whole-branch review (`worker`, opus, xhigh; `matt-chain.md`), one fix wave, scoped re-review (sonnet
   high).
5. Merge into main from the main checkout (it stays on `main`): `git merge --no-ff r10-model-sheet`, push.
6. Stage 0 from the main checkout (`ms_stage0.py`, CPU, 8 threads; three `--aspects` processes can run at once):
   about 7 runs of about 5 minutes. Then `ms_score.py --backfill --stage0-dir <round>/outputs/stage0`.
7. The run handoff (ticket list for the record, agent defaults, queue commands, timing, the back-fill to fold into
   the K = 40 report), worktrees and ticket branches on `.scratch/pending_deletions.md`, then `loop-next
   --unattended --label "r10 run 4"`.

## 5. Code and pitfalls

- **Worktrees lack the data.** Earlier rounds' outputs, graphs, `*_results.json` and `sample.npz` are gitignored or
  in `.git/info/exclude`. Briefs say: read them by absolute path under `/project/CoSiR/src/test/`. Each new worktree
  needs `ln -s /project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz
  <wt>/src/test/20261017_prompted_aspect_views/sample.npz` (excluded, never committed).
- **Keep the main checkout on `main`**: another tab may work there. Merges into the integration branch run in
  `/project/CoSiR-wt-msint`.
- **Ticket 03 brief must add:** keep the AdamW construction unchanged for separate and trunk rows (parameter groups
  only for `share='mask'`), so the bitwise path stays; set `input_img` and `input_cap` in the config to the aspect
  key for instructed rows (the scorer reads them); relax `ms_rows.validate`; rerun 02's bitwise smoke.
- **Ticket 04 brief must add:** do not edit `ms_train.py` (ticket 03 owns it); stage 0 runs
  `--row A1 --set graphs.emotion=Fres1 --name stage0 --device cpu` style calls, which write
  `outputs/stage0/<aspect>__<graph row>__s<seed>.npz`, the folder `ms_score.py --stage0-dir` reads; gate (b) reads
  `/project/CoSiR/src/test/20261018_aspect_graph_sheet/ags_head_results.json`.
- If stage 0 fails: trace the cause, put it at the top of the run handoff, send it to the phone, and do not open
  the run chat (spec §3.5).
- Delete nothing (unattended); list items in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing of this chat (no Codex job, no subagent, no GPU lock).
- Uncommitted: nothing in git; `.scratch/r10-model-sheet/` (tickets, notes) is outside git by design.
- Worktrees: `/project/CoSiR-wt-ms01`, `-ms02` (merged ticket branches), `-msint` (integration). Delete none.
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB; deferred until the back-fill).
