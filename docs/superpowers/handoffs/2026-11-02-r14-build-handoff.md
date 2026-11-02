# Handoff: build r14 (c* on three encoders, the photo model on EmoSet, the analysis sweep) from its approved spec, start the extraction early, then open the run chat

## Brief
- Done: r14 specced, rule-reviewed (Opus xhigh, pass after fixes, all applied) and approved from the brief 2026-11-02 04:13; no container restart.
- State: c* above its teacher on ArtELingo, the photo model above its teacher on Affection (both with Qwen only); r14 tests three more encoders and EmoSet.
- Next: tickets and build; the extraction ticket first, then the extraction runs in the background; final review; then open the run chat.
- Blocked: nothing.

> Written 2026-11-02 04:15 by "r14 decide 3". Loop step reached: step 6 done (spec approved 04:13), step 7 next.
> Direction log entry: [2026-11-02](../direction_log.md) (Read, Meaning (C), Decided (A), Idea (B), approval).

## 1. Read in this order

1. The spec, from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md). Its
   decision rule is §6 with §3.4 and §3.5. It inherits from the r10, r11, r12, r13 and 10-25 specs (its header links
   them); the spec governs where they differ.
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any
   Codex job.
3. The facts brief, `src/test/20261102_backbones_sweep/bs_facts_brief.md` (code pointers, sweep grids, the EmoSet
   protocol, timings).
4. The rule review, `src/test/20261102_backbones_sweep/bs_rule_review.md`: the stored paths it checked (matched
   arms, r10's ablation exports and sidecars, r11's per-query outputs, the EmoSet split format) and the evidence
   behind each fix.

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`; slash-only, so read their SKILL.md), `cluster-run`.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D10) and the direction log, 2026-11-02. The spec was approved from its brief on
2026-11-02 at 04:13, with no container restart. The rule review's fixes (D17 to D21) were in the brief's "Check
these" and are approved.

## 3. Where things stand

- **The extraction comes first (D8):** the first ticket is the extraction code (new encoder stores for Affection
  and EmoSet, the three Qwen views on EmoSet). As soon as it passes, start the extraction in the background under the
  GPU lock (`flock -o -w <seconds> /tmp/gpu0.lock ...`, a background Bash) and go on with the build. Daytime use of
  the local GPU is allowed for it, weekdays 10:00 to 19:00 included (user, D8). Budget about 2¼ h (spec §9).
- **EmoSet:** `/data/SSD/emoset/EmoSet-118K.zip` (11.3 GB). Unpack only the validation and test images and the
  annotation files (5,905 and 17,716 entries, 8 classes; the review counted them). Each split entry is
  `[label, "image/<class>/<class>_<n>.jpg", "annotation/..."]`, so the key carries the class: use the hashed order
  of §3.3 (D17).
- **Stored inputs:** r10's ablation runs under
  `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/{weights,held,train}/k11=40-no_*` and
  `outputs/ablation/held.json`; matched arms M2 in `src/test/20261025_recipe_backbones/outputs/part2/full/held/M2/`,
  M7 and M8 in `src/test/20261025_encoders_genre/outputs/part2/full/held/`, partitions in each `partitions/full/`;
  r13's final photo runs `runs/final/s*/emotion.pt` (paths in the facts brief).
- **node402** is reserved until 2026-11-04 18:06 (the user re-reserves on 11-04). The node work (part a, then the
  photo sweep, then the knob sweep) belongs to the run chat. The new stores need `DATA_MAP` entries, each verified
  with a real job (project memory, cluster CLI).
- **Run mode:** night. GPU check at 2026-11-02 04:09: driver 595.99.02 answers, the kernel module and the library
  match, the GPU is idle and no lock is held.
- **Timing:** the build is expected to take about 5 to 7 h. The spec's report deadline is 2026-11-04 23:00.

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to
the run chat, not to the user.

## 5. Code and pitfalls

- Round folder `src/test/20261102_backbones_sweep/` (it holds the facts brief and the review), module prefix `bs_`.
- **Verdict code** gets unit tests and an Opus `high` ticket review: the generalized trainers (c* with another
  image width; the photo fine-tune), the near-copy cut, the EmoSet order, the scoring paths (SC3's `b_matched`
  form, §3.4), the read-out and wording picks with r_E, and the outcome code with the inconclusive paths (§6). The
  rest gets the controller's diff read and one end-to-end smoke run (`agent-routing.md`, "Builds").
- **Stage 0's checks (i) to (xii)** are in §3.5; (iv) and (v) train on the local GPU (about 8 min). Light checks may
  run in the build under the GPU lock.
- **The `cluster` on PATH** (`/root/miniconda3/bin/cluster`) is another tool. Use the skill's CLI,
  `CLUSTER=~/.claude/skills/cluster-run/cluster`, as its SKILL.md says.
- **Joins by key**, never by store position (C6). EmoSet keys are the path strings; no code reads the class from one.
- **Unattended:** delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's (older untracked files in `archive/` and `src/test/` predate it).
- On disk over 1 GB: `/data/SSD/emoset/EmoSet-118K.zip` (11.3 GB; keep: r14 needs it; it can go at r14's storage
  summary once the validation and test images are unpacked).
