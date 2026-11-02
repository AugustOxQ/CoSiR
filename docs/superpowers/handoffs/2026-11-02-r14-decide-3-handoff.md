# Handoff: r14 decide 3: have the r14 rule reviewed by a fresh Opus `xhigh` worker, apply the fixes, get the user's approval of the brief, then cut to the build

## Brief
- Done: the r14 grill (two rounds; every key answer and the minor batch confirmed); the spec draft with its inline rule; three glossary terms; the direction log's Idea (B).
- State: spec draft written, not yet reviewed or approved.
- Next: the rule review (C7), fixes, the user's approval from the brief, the approving commit, then the cut to r14 build.
- Blocked: the user's approval.

> Written 2026-11-02 03:45 by "r14 decide 2". Loop step reached: step 6 (spec drafted; review and approval next).
> Direction log entry: [2026-11-02](../direction_log.md). Cut because the chat passed 200k tokens of context.

## 1. Read in this order

1. The spec draft, [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md), all of
   it: the brief, §1 to §9, and §4 with the grill's answers and who decided each.
2. The direction log's 2026-11-02 entry (its Idea (B) line holds the grill's outcome).
3. `~/.claude/rules/spec-templates.md` (the seven hard rules, the self-review) and, before the cut,
   `~/.claude/references/loop-chats.md` ("How to cut").

Skills: none.

## 2. Decided by the user, not to be reopened

- Everything in the spec's §4 marked "user" or "user, after seeing the recommendation" (D1 to D8). The user also
  accepted the minor batch (D9, D10).
- Daytime use of the local GPU for the extraction, weekdays 10:00 to 19:00 included (user, 2026-11-02, D8).
- The build is expected to take about 5 to 7 h (told to the user; they expected a middle-sized build).

## 3. Where things stand

- **r10's full-scale ablation already exists:** all eight removals, seeds 0 to 4, under
  `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/{weights,held,train}/k11=40-no_*`. Each seed has about
  33 MB of weights per head, plus held and validation exports. So part b's ablation table (B2) needs no training,
  only retrieval scoring (D13).
- **The matched arms are stored:** SigLIP 1 and CLIP L/14 in
  `src/test/20261025_encoders_genre/outputs/part2/` (`full/matched`, readings own and 20 clusters, all three aspects,
  margins in `reported_stage3.md`); SigLIP 2's are in the 10-25 round (not opened).
- **Extraction estimate:** about 2¼ h of local GPU, from the ArtELingo stores' throughputs (SigLIP 2 75 images/s,
  SigLIP 1 51, CLIP L/14 114) and r13's Qwen extraction (20 images/s).
- **Spec size:** the body is about 230 lines against a claim-test target of about 150 plus the rule, with the reason
  stated in the spec.

## 4. Open points for the next step

- **The rule review (C7):** a fresh Opus `xhigh` `worker`. Give it the spec and the inherited specs it cites (r10,
  r11, r12, r13, 10-25), never our conclusions. It checks the seven hard rules, the parts table, `held-data.md`, and
  whether each SC can be computed from what exists. Look closely at:
  - SC3's matched-arm reading;
  - the part a verdict pattern;
  - the EmoSet wording rule with r_E (r12 used c*'s ranges);
  - B2's same-machine reference;
  - the near-copy cut;
  - the budget.

  Apply its must-fix and should-fix items and record the review in the spec header, as the r13 spec did.
- **A fix that changes a user decision** (D1 to D8) goes back to the user as a key question, answer first. All other
  fixes are agent defaults; the most important go under "Check these".
- **Approval:** paste the brief into chat with the file link. The user approves from the brief. Commit "approved"
  with the time.

## 5. Code and pitfalls

- **Times:** get every time with `TZ=Europe/Amsterdam date '+%F %H:%M'`. This chat once wrote made-up later times
  into the spec; they were corrected.
- **Before the build cut:** ask the user whether to restart the container first, with the GPU check
  (`container.md`), because the build runs unattended and starts the extraction on the local GPU. A decide chat
  always asks.
- **The build's first ticket is the extraction code.** The build chat starts the extraction in the background,
  under the GPU lock, as soon as that ticket passes, and goes on with the build (D8).
- **node402** is reserved until 2026-11-04 18:06; the user re-reserves on 11-04.
- **Module prefix:** `bs_` in `src/test/20261102_backbones_sweep/`.

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing after this cut's commit.
- On disk over 1 GB: `/data/SSD/emoset/EmoSet-118K.zip` (11.3 GB; keep: r14 needs it; it can go at r14's storage
  summary once the validation and test images are unpacked).
