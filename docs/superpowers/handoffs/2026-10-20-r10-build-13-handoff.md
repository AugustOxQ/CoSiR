# Handoff: build stage 2 of the claim test (facts brief, tickets, build, reviews, merge), unattended; then open the run chat

## Brief
- Done: claim test stage 1 is GO (held emotion AMI, D2 0.1001 against PercepT core 0.0855); both reports committed.
- State: by the rule, stage 2 runs: every arm retrained on the 42,969 train paintings, same recipes, same held set.
- Next: facts brief, tickets and build of the stage 2a regression check and stage 2; smoke runs only, no full run.
- Blocked: nothing; "r10 decide 15" meets the user at the same time.

> Written 2026-10-20 06:56 by "r10 run 7". Loop step reached: track 1 (claim test), stage 1 step 9 done; stage 2
> step 7 (build) next, per the spec's plan (D28). Track 2 waits for decision B in "r10 decide 15". Direction log
> entry: [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. This file.
2. The spec, [2026-10-20-claim-test-design.md](../specs/2026-10-20-claim-test-design.md): §3.2 (arms), §3.3 (data
   roles), §3.5 (stage 2a and stage 2), §3.7 (frozen and rerun), §4 rows D2, D9, D23, D25 to D27, D35, and §9 (stage 2
   budget). Check its git log first: "r10 decide 15" may commit an amendment after results (its point b).
3. The binding rule, `src/test/20261020_claim_test/DECISION_RULE.md`: V1 (stage 2 needs a passed stage 2a on every
   training device), R-COL (stage 2 fit pairs: all train pairs of the 42,969 paintings), and the stage 2 outcomes
   S2-GO to S2-NO with what follows. It governs where it and the spec differ.
4. `src/test/20261020_claim_test/ct_queue.md`: how stage 0, stage 1 and the CPU queue run, the V1 gate, the V4 rerun
   path. Stage 2 reuses this pattern.
5. [The r10 build 10 handoff](2026-10-20-r10-build-10-handoff.md) §5 and [the r10 run 5
   handoff](2026-10-20-r10-run-5-handoff.md) §5: the pitfalls of the stage 1 build and run, still true.
6. `~/.claude/references/matt-chain.md` (the build chain) and `~/.claude/references/codex-jobs.md` (Codex room: the 5
   h window resets 07:01; weekly 45% left at 07:00).

Skills: none for the local build; `cluster-run` for the stage 2a check and any smoke run on node408.

## 2. Decided by the user, not to be reopened

- Every grill answer of the claim test (spec §4, D1 to D15); night mode and the overnight go-ahead (2026-10-20 01:27,
  D29); node408 allowed (D23).
- After S1-GO, stage 2 runs (rule §4, user 8b and S8). Stage 2's result counts over stage 1's (D9, user S1).
- Stage 2 data (D25, user m8): train = all 42,969 train paintings; D2 stops early on the validation split (6,152
  paintings); the held set is unchanged.
- Agent defaults, from the spec: frozen recipes, only the data grows (D27); the missing features go to the shared
  cache (D26); node408 for stage 2 training (D23); the stage 2a check on every training device (§3.5).

## 3. Where things stand

- **Stage 1 outcome S1-GO** (`verdict_stage1.json`, b77d3ec; code 81113a1): d +0.0146 [+0.0093, +0.0157]. All 10
  strict paper settings runs collapsed (kept 1 to 2 or 98 to 99 of 100 topics); the K = 40 versions entered and
  scored below core. Auto report: [2026-12-18_claim_test_stage1.md](../../reports/auto/percept/2026-12-18_claim_test_stage1.md).
- **Stage 1 GPU times** on 13,400 fit paintings (run log): D2 seeds 0 to 4, 174 s; core seeds 0 to 4, 761 s; paper
  settings Stage 1, 10 runs, 1,583 s; paper Stage 2, 476 s; held inference 65 s.
- **Instructed images:** 10.2 to 10.7 images/s (sample: 14,889 images in 1,460 s; held: 36,843 in 3,449 s). The
  shared cache (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/`) holds the sample stores
  (`emotion_focus`, `style_word`, `genre_word`; 19,889 paintings, not checked), `*_trainimg` (14,889) and `*_held`.
- **Stage 2 estimate, not checked:** missing instructed images for 23,080 train and 6,152 validation paintings, three
  instructions, about 87,700 images, about 2.3 GPU h (the spec said 1 h); emotion-instructed captions and RoBERTa for
  their captions on top. Training about 2.6 GPU h if time scales with the data (about 3.2 times stage 1); DEC epochs
  and graphs on 42,969 nodes may not scale linearly. So about 5 GPU h, plus CPU for graphs (spec: 0.5 h), scoring and
  diagnostics (stage 1: 118 s and 967 s).

## 4. Open points for the next step

None for the user. Order:
1. **Facts brief** (fact-finding read; Codex `gpt-6.1-sol` medium when `codex-usage --room other` says room, else a
   Sonnet `medium` worker). It finds: which features exist for the 42,969 train and 6,152 validation paintings
   (default Qwen image and captions, three instructed images, emotion-instructed captions for P4, RoBERTa for P1, CLIP
   patch features for the paper Stage 2); the sheet's and stage 1's hard-coded row counts (5,000, 19,889, 13,400,
   67,327); how D2's graphs (Gres, Fres05, R0o) are built and what 42,969 nodes cost in memory and time; where stage
   1's code assumes the sample; the validation split's loader and its painting grouping; what node408 has (data,
   env, the stage 1 code identity).
2. **Tickets and build** per `matt-chain.md` (`to-tickets`, `implement-spec`, the final whole-branch review). Verdict
   code: the stage 2 data roles, the stage 2a check, R-COL on all train pairs, the stage 2 verdict
   (`verdict_stage2.json`, S2-GO to S2-NO), and the stage 2 PercepT and D2 training paths. Plumbing: feature
   extraction, graphs, queues, node408 sync.
3. **The stage 2a regression check** (spec §3.5): the stage 2 code set to the sample sizes, seed 0, reproduces stage
   1's seed 0 outputs (hash, or AMI within 1e-4 with the cause named), on every device that trains a stage 2 arm,
   node408 included. Without it V1 blocks every held read of stage 2.
4. **Smoke runs:** local GPU in the daytime only for short smoke runs under the lock (`flock -n -o -E 75
   /tmp/gpu0.lock ...`); node408 smoke runs through the cluster CLI only. Feature extraction is a full GPU job: it
   waits for the evening or runs on node408 (decide 15's point d may settle the device; read the direction log
   before the run chat's handoff).
5. **Handoff for the run chat**, "r10 run 8" (`--unattended`): the queues, the device, an end estimate against 11:00
   for a local run (`shared-resources.md`), V2 for stage 2 (an independent re-derivation from the rule and the data,
   never our numbers), the auto report and the user-read report, and what the rule says follows each stage 2 outcome.
   GPU check before opening (`loop-chats.md`, "How to cut"; in night mode the check only).

## 5. Code and pitfalls

- **Stage 1's covered files stay unchanged:** every `ct_*.py`, the `ct_queue*.sh` scripts, `DECISION_RULE.md` and the
  six reference round folders. Any edit breaks the stage 0 pass that V1 checks for stage 1, and stage 0 cannot retrain
  any more (held NPZs exist). So stage 2 code goes in new modules with a new prefix (for example `ct2_`) that import
  stage 1's, or into a copy that the stage 2a check covers. Say in the run chat's handoff which way the build went.
- **Rule edits:** stage 2 needs no new rule (rule §4 already has S2-GO to S2-NO). Any amendment is written as one,
  dated, and committed before any stage 2 held output exists.
- **Round folders go on `sys.path`:** a new module prefix avoids clashes.
- **The pass is tied to the host name** (the container id); a recreated container fails closed.
- **Never run `ct_percept.py --smoke` in the main checkout:** its smoke training writes the full-run paths. Use
  `ct_smoke.py` with a scratch destination, or the stage 2 equivalent.
- **JSON outputs are gitignored:** `git add -f` for verdict and diagnostics JSON.
- **The same repo is in use by "r10 decide 15"** (docs only): commit only your files, pull before you push.
- The auto-mode check refused a few plain Bash calls for some minutes after a Codex launch; the Read tool works
  meanwhile. A refused Codex job goes to Claude (`agent-routing.md`).
- Unattended: delete nothing; list items in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: "r10 decide 15" (with the user). No background job, GPU lock or Codex job of "r10 run 7".
- Uncommitted: nothing in git after this cut's commit.
- On disk over 1 GB: `src/test/20261020_claim_test/outputs/`, 3.1 GB (keep; disk ledger);
  `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger); `/project/CoSiR-wt/ct-09/.scratch/`, 1.6 GB,
  on `.scratch/pending_deletions.md`. System disk: check `df -h /` before writing features (they go to `/data/SSD2`).
