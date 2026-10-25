# Handoff: finish the encoders-and-genre build (fix 09 by Codex resume, review 13, Codex 14, the real X1/X2 scorer checks, tickets 06, 10, 16; gates; final review; merge into main), then open "r10 run 21"

## Brief
- Done: tickets 05, 07, 11 and 15 merged (10 of 16). VG1, the fresh-seed parity, VL1 (a) and the VG1 scorer half pass on the real outputs; 07 to 08 runs end to end on the full P.
- State: 09 (Codex) review FIX, four file-format mismatches with 08; 13 built, not reviewed. 06, 10, 14 and 16 not started.
- Next: Codex fix of 09 by resume, Opus review of 13, Codex 14, the real X1/X2 scorer checks; then 06.
- Blocked: nothing.

> Written 2026-10-25 20:05 by "r10 build 21c" (cut at 200k tokens of context, at a clean point: no subagent, Codex
> job or GPU job running). Loop step reached: step 7 (build), 10 of 16 tickets merged. Direction log entry:
> [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the progress notes `.scratch/r10-build-21/notes/build_progress.md`, section "r10 build 21c
   progress" (per ticket: result, gates, open review findings, interface notes). They are the record for the run
   handoff.
2. [2026-10-25-r10-build-21b-handoff.md](2026-10-25-r10-build-21b-handoff.md) §2 to §5 (decided items, waves, GPU work
   in the build, final review, the run chat's handoff); they still hold.
3. The ticket index `.scratch/r10-build-21/issues/00-index.md` (VG2 now writes one file per phase) and the briefs
   `notes/implementer_brief.md`, `notes/reviewer_brief.md`. Tickets 14, 15 and 16 carry controller notes at their end.
4. `~/.claude/references/matt-chain.md`, `~/.claude/rules/agent-routing.md`, `~/.claude/references/codex-jobs.md`.

Skills: `tdd` (named in verdict-code briefs), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As the 21b handoff §2: the spec's §4 rows D1 to D17 marked "user" or "user, after seeing the recommendation", the
brief's approval at 18:35, the container restart at 18:44. This cut stays inside the build phase, so no restart
question was asked (GPU check below); the cut to the run chat asks again.

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at 443c735, pushed (worktree `/project/CoSiR-r10-int`). Merged in this
  chat: 07 (c7d39aa), 11 (39f20da), 15 (ed20aa2), 05 (443c735). Done: 01, 02, 03, 04, 05, 07, 08, 11, 12, 15.
- **Gate files on the real outputs** (`/data/SSD2/cosir_r10/20261025_encoders_genre/outputs`, 2.8 GB): from 21b
  `stores/x0.json`, `part2/regression/x{1,2}_{student,matched}.json`; from this chat `stage_iii/vg1.json` (53 s GPU),
  `stage_iii/fresh_parity.json` (seeds 1 to 3 equal K4's, 123 s GPU), `stage_iii/vg1_scorer.json`,
  `stage_ii/vl1_scorer.json` (18 values, diff 0). All pass. Not yet written: `x{1,2}_scorer.json` (next step),
  `stores/v5.json` and `stage_ii/vl1.json` (the run chat's).
- **07 checked on the GPU** into scratch `/tmp/claude-0/eg07e2e/out` (full P, old arms; new arms listed missing):
  VL1 (b), (c), VL3 pass over 60 files. 08's `score --smoke` (20 draws) ran on those exports in 7 min.
- **Built, not merged:**

  | Ticket | Branch, last commit | What is left |
  |---|---|---|
  | 09 VL2 (Codex) | `r10-t09-vl2` 39517f3, worktree `/project/CoSiR/.claude/worktrees/codex-t09` | fix round by Codex resume, thread `01a17461-28f4-7783-bf85-e00ed656fba3`: findings 1 to 7 in the progress notes; then the scratch check, expect `agree` True; merge |
  | 13 fresh seeds, report | `worktree-agent-a74de5ab3211c5008` 8ffbbc2 | Opus `high` review (departures and two open points in the progress notes), then merge |

- **Not started:** 14 (ready: formats in its ticket note; Codex, kind `rederivation`), 06 (ready: 05 merged), 10
  (after 06), 16 (last).
- **Codex:** five-hour window 68% left at 19:59 (reset 21:26); a rederivation job needs 20%, an implementer 80%.
  Ledger row for 09 written (fix round).

## 4. Open points for the build (agent work; nothing for the user)

a. **Next, in parallel:** the Codex resume fix of 09; an Opus `high` reviewer of 13; Codex on 14 (new worktree from
   the integration tip, brief from the ticket and its note, independence as for 09); in the background on the CPU,
   `eg_score.py --check x1`, then `--check x2` into the real outputs (about 17 and 8 min; from the integration
   worktree; write a bash script, the shell does not split a command held in a variable); a Sonnet `high`
   implementer for 06.
b. **09's scratch check after the fix:** `eg_vl2.py --smoke --outputs /tmp/claude-0/eg07e2e/out` (20 draws after
   the fix; else `eg_vl2.run(Path(...), workers=6, smoke=True, n_draws=20)` from the codex worktree), about 7 min CPU.
   It compares with 08's smoke `lift.json` there.
c. **After `x1_scorer.json` passes:** B1 and B6 on ArtELingo-28 at the sample scale (`eg_matched.py --scale sample
   --arm B1`, then `B6`, GPU, under the lock).
d. **Then** 10 (after 06), 16 (its notes: `gate v2 --stage 3` before stage 4; `--rerun-failed` once; `replace` must
   move stale partition caches aside first; run `replace` once in its smoke), the final whole-branch review, one fix
   wave, a Sonnet `high` re-review, the merge into main, the `EG/outputs` link in the main checkout, the worktrees
   and branches on `.scratch/pending_deletions.md`, and the run chat's handoff (21b handoff §4 d and e).
e. **Gate files after the fix wave:** each records its code commit. If the fix wave changes a gate's producer
   (`eg_train3.py`, `eg_score3.py`, `eg_lift.py`, `eg_student.py`, `eg_matched.py`, `eg_score.py`, `eg_pinfer.py`),
   rerun that gate before the run handoff.

## 5. Code and pitfalls

- The 21b and 21c pitfalls hold. New in 21c: the Bash tool runs zsh, which does not word-split `$VAR` commands; put
  multi-step jobs in a bash script under `/tmp/claude-0/`.
- `eg_pinfer.py` without `--smoke` locks the stage ii plan on first write (VL4): never run it on the real outputs
  before the stage i weights of K5 to K8 and M5 to M8 exist (the run chat's step).
- `eg_score.py --replace` needs `--candidates`; after a rewritten held output, the stale
  `part2/partitions/<scale>/<arm>/<aspect>__s<seed>__*` files must be moved aside before the recompute.
- Subagents leave scratch under `/tmp/claude-0/`; list it on `.scratch/pending_deletions.md` (a hook refuses `rm`).

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat). GPU check at 20:03: driver
  595.99.02 answers, kernel module and library 595.99.02, GPU idle; container started 18:44.
- Uncommitted: nothing in git; tickets and notes are in `.scratch/` (outside git).
- On disk over 1 GB: `/tmp/claude-0/eg07e2e` (5.2 GB, needed for 09's check), 13's fixture
  `/tmp/claude-0/-project-CoSiR/46fd8f3c-3629-40f8-98bc-63333032b546/scratchpad/fx` (2.9 GB), `/tmp/claude-0/r12_smoke`
  (2.1 GB); all on the system disk and on `.scratch/pending_deletions.md`. The round's outputs are 2.8 GB on
  `/data/SSD2`.
