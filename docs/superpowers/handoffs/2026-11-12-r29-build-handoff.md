# Handoff: r29 build: tickets and build of the SigLIP-only aspect adapter, from the approved spec, then open the run chat

## Brief
- Done: r29 decided and specced: the aspect adapter reads SigLIP 1 only; Qwen3 keeps the graphs, correction and targets. Spec approved 22:42.
- State: no code yet. The pass bounds come from r28's records: image group 0.3863 (seed range 0.0030), text group 0.2594 (range 0.0020).
- Next: to-tickets from the spec, build, final review; merge to main, push, then open the r29 run chat.
- Blocked: nothing. r28's run chat holds the local GPU tonight; smoke runs queue with --short.

> Written 2026-11-12 22:45 by the "r29 decide 2" tab. Loop step reached: step 6 done (spec approved), step 7
> next. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

## 1. Read in this order

1. The spec, from §1 onward: [2026-11-12-siglip-only-adapter-design.md](../specs/2026-11-12-siglip-only-adapter-design.md).
   §6 with §3.4 and §3.5 is the decision rule; it governs, and this handoff only adds code facts.
2. `~/.claude/references/matt-chain.md` (to-tickets, implement-spec, the final whole-branch review) and
   `~/.claude/references/loop-chats.md` ("Unattended chats": decision lines, nothing deleted).
3. The Codex facts brief, with code paths and widths: `src/test/20261112_siglip_only/r29_facts.md`.
4. The rule review and the re-review, for why each gate exists: `src/test/20261112_siglip_only/r29_rule_review.md`,
   `r29_rule_rereview.md`.
5. The constitution's "Decisions agents never make alone": `docs/superpowers/constitution.md`.

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

All on 2026-11-12, in the spec's §4 (D1 to D9) and the direction log:
- A SigLIP-only model; medium loop (user). Qwen3 keeps the instructions, graphs, content correction and alignment
  targets; the image tower reads SigLIP 1 and the caption tower the captions' SigLIP 1 text embedding (user).
- Decided on validation: a pass makes it the main model and starts stage 2 (more encoders, all tables); a fail gives an
  extra row (user).
- The pass rule: one mean per group, each within the current model's seed range (user, after seeing the
  recommendation). Plain caption and the prefix both run on seed 0; c*'s settings; local GPU, stage 2 after 19:00 or
  on DAS6; Table 8 as D8; on a fail the cells go to the numbers sheet; the new model trains locally against the
  current model's DAS6 seeds, the machine difference shown beside the pass rule (D25) (user, after seeing the
  recommendation).

## 3. Where things stand

- Nothing exists for r29 except the round folder `src/test/20261112_siglip_only/` (facts brief, its Codex task, the
  review, the re-review). Module prefix `r29_`; outputs in `/data/SSD2/cosir_r29/20261112_siglip_only/`.
- K0 is r14's c* on SigLIP 1: `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/` (`train/k11=40/`,
  seeds 0 to 4, trained on node402). r14's trainer is `src/test/20261102_backbones_sweep/bs_cstar.py`; r28's wrappers
  and tests are in `src/test/20261112_siglip_main/` (`r28_trainer.py`, `r28_text.py`, `r28_mmart.py`,
  `r28_ablation.py`, `r28_time.py`, `test_r28_trainer.py`).
- K0's records for the pass rule and stage 0: `src/test/20261112_siglip_main/records/valpick.json`, `text.json`,
  `photos.json`, `mmart.json`, `ablation_ref_replay.json`.
- Gate (i)(a) evidence: r10's V4a retrained c* on this GPU bit for bit (`src/test/20261027_recipe_sweep/rs_full.py`,
  `regress v4a`); r21's spec says CUDA reruns are not bit-exact. The spec's fallback for (i)(a) covers this.

## 4. Open points for the next step

None for the user. The build chooses the code layout (a caption-width parameter beside the image width, the SigLIP
caption loader, the prefixed extraction). Each result-shaping choice gets a line in
`src/test/20261112_siglip_only/decisions.md`.

## 5. Code and pitfalls

- The 2048 width is asserted in many places (facts brief §3). Change only the caption input width; the heads, the
  teacher and graph embeddings and the loss targets stay 2048. The checkpoint metadata records the caption input width.
- Joins follow extraction IDs and paintings (C6; the project CLAUDE.md), never `range(n)`.
- `src/test/20261112_siglip_main/20261112_siglip_main_log.md` and r28's other files belong to r28's run chat: import,
  never edit or commit them. Earlier round folders are frozen.
- CoSiR main has other chats' uncommitted files; commit only r29's. Build on branch `r29-siglip-only`; merge into main
  locally and push.
- GPU through `res-queue gpu --short --label r29:<step>` (subagents add `--no-wait`). New stores only in the folders of
  D15. Python: `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`; tests name their folder.
- **r29's main chat** opens only after r28's run chat ends: chats.tsv then has a row opened from r28 run's session
  (`dae700bf-e7a2-4618-ae57-fa086b1e35b9`), or r28's run reports are committed. Check before the cut to the run chat;
  then run `loop-next ~/.claude/templates/main_chat_start.md --label "r29 main" --main --new-workspace "CoSiR r29"
  --model sonnet --effort high --notify`. If r28's run chat is still open at the cut, the run handoff carries this.
- At the end: write the run handoff (`docs/superpowers/handoffs/2026-11-1x-r29-run-handoff.md`), run the GPU check
  (`loop-chats.md`, "How to cut"), and open `loop-next <run handoff> --label "r29 run" --unattended`.
- The paper plan's SigLIP-only row (`docs/paper/.../notes/paper_plan.md`) still describes the wider version; the paper
  chat owns that file. Do not edit it.
- The repo is public: no keys or personal data in files.

## 6. State at handoff

- Running: nothing of this chat. r28's run chat holds the GPU (workspace "CoSiR r28").
- Uncommitted: nothing of r29 after this handoff's commit. `/project/claude-config/codex/jobs.md` holds r29's Codex
  row beside other sessions' rows; the decide chat's end ran `sync.sh`.
- On disk over 1 GB: nothing.
