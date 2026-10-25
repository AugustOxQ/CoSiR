# Handoff: build the encoders-and-genre round from its approved spec (tickets, build, ticket reviews, final whole-branch review, merge into main), then open "r10 run 21"

## Brief
- Done: spec and decision rule for the next loop written, reviewed (Opus `xhigh` rule review, cold Sonnet check) and approved by the user at 18:35; five glossary terms added.
- State: three stages: four more encoders on held paintings; each student's lift over its own PercepT arm on development paintings; five genre variants toward 0.4989.
- Next: tickets from the spec, the build per `matt-chain.md`, the final review, merge into main, the run handoff.
- Blocked: nothing. The user restarts the container before this chat opens.

> Written 2026-10-25 18:40 by "r10 decide 21". Loop step reached: step 6 done (spec approved); step 7 (build) next.
> Direction log entry: [2026-10-25 · Recipe and backbone round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The spec, [2026-10-25-encoders-and-genre-design.md](../specs/2026-10-25-encoders-and-genre-design.md), from §1 on
   (skip the brief), and its rule, `src/test/20261025_encoders_genre/DECISION_RULE.md` @ f0f7c91 (it governs where they
   differ). The rule inherits 12-24's rule (`src/test/20261025_recipe_backbones/DECISION_RULE.md`, part 2 and stage 5)
   and its part 1 statistics; read the sections it cites.
3. 12-24's code, `src/test/20261025_recipe_backbones/` (prefix `rb_`, code commit 08cab72): `rb_queue.md`, the run log
   `20261025_recipe_backbones_run_log.md`, and the modules stage i reuses (`rb_student`, `rb_matched`, `rb_score`,
   `rb_v2`, `rb_a28`, `rb_part2_report`, `rb_run`, `rb_queue_*.sh`), stage ii reuses (`rb_part1`, `rb_pick`,
   `rb_sheet`) and stage iii reuses (`rb_part1`, `rb_probe`, `src/test/20261021_d2_ablation/ab_train.py`,
   `src/test/20261019_model_sheet/ms_train.py`).
4. The decide 20 facts brief, `.scratch/r10-decide-20/facts_decide20.md` (gitignored): §2 lists every code site that
   names the old encoder list; §5 the sheet's portability; §7 where per-aspect genre terms plug in.
5. `~/.claude/references/matt-chain.md` and `~/.claude/rules/agent-routing.md` (Codex first when `codex-usage --room`
   allows; ticket reviews and unit tests for verdict code only).

Skills: none beyond those `matt-chain.md` names.

## 2. Decided by the user, not to be reopened

The spec's §4 rows marked "user" or "user, after seeing the recommendation" (D1 to D17), with the approval of the
brief at 18:35 and the container restart before this chat (D16). Run mode is "day" at the time of writing; the user or
the main chat may switch it to night.

## 3. Where things stand

- Rule committed at f0f7c91 before any code (C7). No code of this round exists yet; the round folder holds only the
  rule. Module prefix `eg_` (spec D32); outputs go on `/data/SSD2` and are linked from the round folder (D31).
- Inputs on disk (checked 2026-10-25):
  - 12-24's sample-scale weights of K2, K3, K4 (`outputs/part2/sample/weights/`), M2, M3 (`outputs/part2/sample/matched/`),
    their held and ArtELingo-28 outputs at both scales, `verdict_stage{3,4,5}.json`; part 1's R6-5 exports on P
    (`outputs/part1/train/R6-5/*__s0__P.npz`), `pick.json`, `reported.json`. All under
    `src/test/20261025_recipe_backbones/`.
  - The claim test's sample-scale PercepT core weights, P1 to P4 seeds 0 to 4
    (`src/test/20261020_claim_test/outputs/percept/core/stage2/k40/`), and its Stage 1 artifacts.
  - Stores in `/data/SSD2/pre_extract/{artelingo_full,artelingo28}/`: `clip_b32`, `siglip2_so400m`, `qwen3vl_emb_2b`. The
    archived DINOv2-large store for V5's check: `/data/SSD2/pre_extract/artelingo/r8_enc_dinov2l`.
  - HF cache (`/data/SSD2/HF_home`): `facebook/dinov2-large` cached; `openai/clip-vit-base-patch16`,
    `openai/clip-vit-large-patch14` and `google/siglip-so400m-patch14-384` must be downloaded (about 6 GB).
- The rule review found that 12-24's K4 seed 0 weights equal part 1's R6-5 seed 0 weights (`torch.equal`); VL1 (b)
  and the fresh-seed check rest on that.
- Disk: system disk 219 GB free; `/data/SSD2` 1.9 TB free. Codex at 18:35: five-hour window 90% left (reset 21:26),
  weekly 93%.

## 4. Open points for the build (agent work; nothing for the user)

a. Tickets from the spec and rule with `to-tickets`; the coverage line names every rule ID: X0, V5, X1, X2, V1 to V4
   and the S1, S2, A outcomes for K5 to K8; VL1 to VL4 and the L outcomes; VG1 to VG4, failed runs, the guard, the
   extras, the winner, the III and C outcomes, kept weights.
b. Verdict code (ticket reviews and unit tests): stage i's reuse of 12-24's scorer, bootstrap and re-derivation for the
   new candidates; stage ii's lift and its bootstrap, VL2; stage iii's effects, guard, extras rule, winner, fresh-seed
   statistics, VG2.
c. Plumbing (controller diff read and one end-to-end smoke): extractor entries for the three new checkpoints (images
   and texts; DINOv2 images only) and X0 against the extractor at the commit before the change; V5; the student and
   matched-arm width map with 768 for CLIP L/14; K5 under R6-5 without the R7 guard (`rb_student.py:99–104`,
   `rb_run.py:189–203`), changed in the new round's code, never in `rb_` code in place; P inference of saved weights;
   a default-only path for the sheet statistics (the sheet's loader asserts four Qwen views, `rb_sheet.py:135,196–228`);
   the stage iii trainer switches (per-aspect teacher weight; the relational term beside the direct one, so
   `ab_rows.py:43–45` must not block it; the GA adaptor with its own generator); queues; B1 and B6 ArtELingo-28
   inference at the sample scale for a stage 5 at that scale.
d. Short GPU work during the build (smokes, X0, the VG1 and fresh-seed regression checks) is allowed; long heavy runs
   (over about 1 hour) wait for the run chat, and on weekdays not between 10:00 and 19:00 (`shared-resources.md`).
e. The run chat's handoff: ticket list, queue order (spec §3.5; stage iii can go first, it needs no new store), the
   commands, the end estimate against 11:00 (spec D15: start when the build is done if the run can end by 11:00 on
   Monday 2026-10-26, else Monday 19:00), deferral order K5, K8, K7, K6. Then the GPU check and
   `loop-next <handoff> --label "r10 run 21" --unattended`.

## 5. Code and pitfalls

- Round folders go on `sys.path`: the new round's modules use the prefix `eg_`. Change no `rb_` module in place.
- `scripts/extract_features.py` knows only `clip_b32`, `qwen3vl_emb_2b`, `siglip2_so400m`, `dinov2_large` and forces
  texts for every encoder except DINOv2; its text truncation dispatch (`:343–352`) needs the new entries too.
- The SigLIP 2 store was written on 2026-10-13 and the extractor changed after it (6033362, 647867f), so X0 compares
  the changed extractor with the old one, not with the stored rows.
- Joins are positional by extraction IDs (C6): new stores must keep the existing stores' name order (V5).
- Unattended: delete nothing; list items in `.scratch/pending_deletions.md`.
- Codex: a resumed session re-reads the whole session; a fresh session is far cheaper (12-24 run).

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: the 12-24 outputs, 24 GB on the system disk (kept; ledger rows of 2026-10-25).
