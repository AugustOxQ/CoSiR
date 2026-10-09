# Handoff: build round 6 (the ArtELingo held-split paper test of AFF) from the approved spec and rule, unattended

> Written 2026-10-09 04:56 by the `r6 decide 2` chat. Loop step reached: step 6 done (spec approved by the user at
> 04:53, rule committed); step 7 next. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The decision rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md` @ c394b60 (SHA-256
   7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724). It governs; tickets cite its IDs (§2 to §10:
   P1 to P7, S1, S2, §5 items, §6 items, §7, §8 items).
2. The spec `docs/superpowers/specs/2026-10-09-r6-held-test-design.md` from §1 (skip the brief).
3. `~/.claude/references/matt-chain.md` (to-tickets, implement-spec, ticket reviews only for verdict code, the final
   whole-branch review) and `~/.claude/references/loop-chats.md`, "Unattended chats".
4. Background when a ticket needs it: `src/test/20261125_artelingo_held_test/design/facts.md` (what AFF loads, reuse
   hazards) and `rule_check/opus_rule_check.md` (why the rule says what it says).

## 2. Decided by the user, not to be reopened

- Everything in the spec's §4 and the rule (approved 2026-10-09 04:53 from the brief). The secondary checks are
  tested only after a GO, with Holm across the two (user, after seeing the recommendation).
- "r7 decide can be later, first go with the build chat and run overnight" (user, 04:55): build and run now,
  unattended; design L's decide chat is opened later, as the run chat's usual next decide chat.

## 3. Where things stand

- No code of this round exists. The round folder holds `design/`, `DECISION_RULE.md` and `rule_check/`.
- AFF and every number it rests on: the rule's §6 items 3 and 4 (seed 42) and round 3's report.
- DAS6: node401, node402, node408, three GPUs each, about 120 hours of reservation left when checked on the night of
  2026-10-09; the cluster selftest has not run on them yet.

## 4. Where the build ends and the run begins

- **Build (this chat):** tickets from the rule, the code with its unit tests (synthetic data of the real shapes, and
  selection rows), the describe-then-score code made runnable on DAS6, ticket reviews for verdict code, the final
  whole-branch review on Opus, one fix wave, a scoped re-review; merged into `main` and pushed. No held row is loaded
  by any build step or test.
- **Run (next chat, `r6 run`, opened with `loop-next` at the end of the build, no `--notify`):** the rule's §6 in
  order (inputs, refits, picks, regression, sensitivity input, the describe-then-score seed-42 tuning and its stop,
  the smoke last), then §8 (the read, phase-2 re-derivation, verdict), then §10.5, the auto report and the user-read
  report, then it opens the next decide chat with `--notify`.
- The describe-then-score budget clock (rule §7) starts at the first commit of its code.
- Pre-read stops (rule §9) end the run's read part only: the run chat writes everything up, leaves the read for the
  user, and still reports what it has.

## 5. Code and pitfalls

- Folder `src/test/20261125_artelingo_held_test/`; modules `r6_*`, `run_r6_*`, `test_r6_*` (round folders go on
  `sys.path`). Earlier round folders are read-only; round 4's code is reused only through fixed `r6_` copies (rule
  §6 item 1).
- Every Python call with `PYTHONDONTWRITEBYTECODE=1`; CPU work at 8 threads, at most three processes; mutation tests
  on copies; tests on the real data shape; times from `TZ=Europe/Amsterdam date`. Round 5's lapses were exactly these.
- The refuse-twice runner and the ledger row follow rule §8.1 (pattern: H3's
  `src/test/20261019_affect_factor_learning_held/run_held.py`).
- Cluster: the first cluster run needs a commit whose subject says "cluster run", then `cluster sync`; run
  `cluster launch -- cluster-selftest gpu` on each node first; `/local/wding` is per node, so data is copied to each
  node used. Never hand-rolled ssh or rsync (memory `project_cluster-cli-v2`).
- The fine-tune checkpoints: `res/cluster_jobs/2026100707*/code/outputs/clipft/`. The 8B probe:
  `src/test/20261106_mllm_probe_8b/`.
- Commit and push without asking (`git.md`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (this handoff and `chats.tsv` are committed with the cut).
- On disk over 1 GB: nothing. The decide chats' scratchpads hold small files only.
