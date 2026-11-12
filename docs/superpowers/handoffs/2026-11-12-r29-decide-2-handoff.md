# Handoff: r29 decide 2, apply the rule review's fixes to the r29 spec, then get the user's approval of its brief

## Brief
- Done: the grill is closed, the r29 spec (the SigLIP-only aspect adapter) is drafted, and the rule review is done: pass after fixes (3 must-fix, 17 should-fix, 9 nits).
- State: the pass bounds on validation come from r28's records: image group 0.3863 (seed range 0.0030), text group 0.2594 (range 0.0020).
- Next: apply the review's fixes, paste the brief for the user's approval, then write the direction log entry and cut to the build.
- Blocked: nothing. r29's main chat waits until r28's run chat ends.

> Written 2026-11-12 21:55 by the "r29 decide" tab, cut at 201k tokens of context. Loop step reached: step 6 (spec),
> after the grill (decisions A and B made), before the brief's approval. Direction log entry: none yet; the draft is in
> section 4.

## 1. Read in this order

1. This handoff, all of it.
2. The spec, all of it: `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md` (status draft).
3. The rule review, all of it: `src/test/20261112_siglip_only/r29_rule_review.md` (M1 to M3 first; it gives the
   replacement text for most fixes).
4. The Codex facts brief: `src/test/20261112_siglip_only/r29_facts.md` (code paths, the caption join, the 2048 width
   asserts, the text scoring, the photo model's caption input, run times).
5. Before editing the spec: `~/.claude/rules/spec-templates.md`, `~/.claude/templates/spec_core.md` (hard rules) and
   `spec_experiment.md`. Before the cut to the build: `~/.claude/references/loop-chats.md` ("How to cut") and
   `~/.claude/references/matt-chain.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- **Direction (A):** a SigLIP-only model, medium loop, step 3 skipped (user, 2026-11-12 21:10, paper chat).
- **Idea (B):** Qwen3 keeps the instructions, graphs, content correction and alignment targets; the aspect adapter
  reads SigLIP 1 only (image tower: the SigLIP 1 image embedding; caption tower: the captions' SigLIP 1 text
  embedding). User, 21:10: "the instruction part and buddy part still from qwen, just the later student part becomes
  all siglip". The paper plan's row (SigLIP in place of the graphs too) is the older, wider version.
- **Place in the paper (D3):** decided on validation (option c), with the user's extension, own pick before our
  recommendation (21:29): "if it works we can possibly try more models (Siglip2 own, CLIP L own, and more), and then
  this becomes the new model to replace qwen3+siglip. But only if it work, if not possibly as another abolation or
  something".
- **Pass rule form (D4):** one mean per group (image six cells, text nine cells), each within the current model's
  seed range; a text emotion loss can be offset by style and genre gains (user, after seeing the recommendation, the
  scenario check, 21:34).
- **Minor points (D5 to D9), all as recommended (user, after seeing the recommendation, 21:34):** run the plain
  caption and the "Represent the caption: " prefix on seed 0, the higher text group wins; c*'s settings; local GPU,
  stage 2 after 19:00 or on DAS6; Table 8 measures the SigLIP 1 text pass and the new caption tower; on a fail, the
  cells go in the numbers sheet and the paper chat places them.
- Closing summary confirmed (21:35); no HTML plan page ("plain"). The user's "yes" at 21:42 agreed to open r29's main
  chat only after r28's run chat ends. It did not approve the spec, which the user has not seen yet.

## 3. Where things stand

- **The spec draft** has the brief and §1 to §9; its §6 with §3.4 and §3.5 is the decision rule (C7).
  - Its §4 "When" times are wrong; the draft guessed 21:50 to 22:10. Use: D1, D2 21:10; D3 21:30; D4 to D9 21:34;
    D10 to D16 21:38. The header date is 21:40, when the draft was written.
- **The review** (fresh Opus `xhigh` worker, 21:51) gave "pass after fixes":
  - M1: gate (i) asks for a bit-for-bit retrain on the local GPU, but r14's SigLIP 1 weights were trained on node402.
    Replace it with the review's three-part gate.
  - M2: the LEVEL unit, the read-out pick and the comparator source are not defined for each cell type. The review
    gives a per-cell table; its photo and MMArt mAP comparator files are marked "name" and need a check.
  - M3: stage 0 misses the test scorers, MMArt, the photo path, the ablation path and the new stores. The review
    proposes gates (v) to (viii).
  - Should-fix S0 to S16, among them unrounded bounds in the rule (S1), a validation step for E2, E3 and F1, and
    per-gate blocking.
- **Numbers, recomputed by the reviewer, equal to the records:**
  - Image group 0.3862917081 (per seed 0.3848908 to 0.3878697, range 0.0029789; `records/valpick.json`).
  - Text group 0.2593512884 (per seed 0.2586239 to 0.2606347, range 0.0020107; `records/text.json`, read-out r0).
  - Both records are in `src/test/20261112_siglip_main/`.
- **r28** holds the local GPU tonight: its ablation, then a settings rerun. r28's settings check found the learning
  rate 0.0005 ahead of c*'s settings by +0.0007, inside the 0.0030 seed range; r28 flagged it for the user.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

1. Apply M1 to M3 and S0 to S16; take the nits where they help.
   - A fix that changes a user decision (D3 to D9) is asked first. The review's items look agent-level.
   - Then self-check the hard rules. A Sonnet `high` re-review of the must-fix changes, before the brief goes to the
     user, is the routing default for a re-review.
2. Paste the brief into chat with the file link, and get the user's approval. Then set the status to "approved
   <time>", fill "Reviewer" in the header, and commit and push.
3. Add the direction log entry above the 2026-11-11 r26 entry (draft below; set the spec status and the links).
4. Cut to "r29 build" (unattended).
   - Write the build handoff, run the GPU check (`container-ops.md`), then
     `loop-next <handoff> --label "r29 build" --unattended`.
   - The build handoff says that r29's main chat opens only after r28's run chat ends (`chats.tsv`: r28 run, opened
     19:30; r28 main, 19:28). The command is
     `loop-next ~/.claude/templates/main_chat_start.md --label "r29 main" --main --new-workspace "CoSiR r29" --model sonnet --effort high --notify`.

Draft direction log entry:

```markdown
## 2026-11-12 · r29 decide: a SigLIP-only aspect adapter, the main model only if it holds on validation; spec approved, build next

- **Read:** no new result. The paper chat opened r29 ([decide handoff](handoffs/2026-11-12-r29-decide-handoff.md)).
  CoSiR on SigLIP 1 reads texts through Qwen3 (3.6 ms per caption); on test its text-to-image emotion mAP is 0.264
  against the instructed Qwen3's 0.268 and plain SigLIP 1's 0.211 (r28's numbers sheet).
- **Meaning (C):** none; no results read.
- **Decided (A):** a SigLIP-only model; medium loop, step 3 skipped (user, 2026-11-12, paper chat). Stage types: a
  sweep (caption form), incremental on validation, claim tests on test, an ablation (agent default).
- **Idea (B):** Qwen3 keeps the instructions, graphs, content correction and alignment targets; the caption tower
  reads the captions' SigLIP 1 text embedding (user). If the image group and the text group each hold within the
  current model's seed range on validation (one mean per group: user, after seeing the recommendation), it becomes
  the main model, and SigLIP 2 and CLIP L/14 get their own text encoders; otherwise an extra row or ablation (user).
- **Why:** while texts need Qwen3 at test time, the paper's PercepT sentence and its light-model claim are weak.
- **Links:** [spec](specs/2026-11-12-siglip-only-adapter-design.md) · [decide handoff](handoffs/2026-11-12-r29-decide-handoff.md) · [decide 2 handoff](handoffs/2026-11-12-r29-decide-2-handoff.md)
```

## 5. Code and pitfalls

- The round folder `src/test/20261112_siglip_only/` exists: the review, the facts brief and its Codex task. Module
  prefix `r29_`; output root `/data/SSD2/cosir_r29/20261112_siglip_only/`.
- `src/test/20261112_siglip_main/20261112_siglip_main_log.md` is r28's run chat's file: never commit it from r29.
- `/project/claude-config/codex/jobs.md` holds this chat's Codex row (21:35), beside other sessions' rows, all
  uncommitted. The chat end runs `/project/claude-config/sync.sh` and commits that repo.
- The paper plan's SigLIP-only row (`docs/paper/.../notes/paper_plan.md`) still describes the wider version. The paper
  chat owns that file; tell the user, or the paper chat, when it comes up.

## 6. State at handoff

- Running: nothing of this chat. r28's run chat holds the GPU (workspace "CoSiR r28").
- Uncommitted: nothing of ours after this commit, except `codex/jobs.md` in claude-config (above).
- On disk over 1 GB: nothing.
