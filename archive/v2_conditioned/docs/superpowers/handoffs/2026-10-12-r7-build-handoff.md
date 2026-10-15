# Handoff: r7 build: turn the approved design L spec into tickets and build them (Matt Pocock's chain), final review, then cut to r7 run

## Brief
- Done: the user approved round 7's spec at 02:34. The decision rule was reviewed on Opus, re-reviewed on Sonnet, and committed.
- State: design L is a ceiling check, six label-free style groupings with an NrKmeans reference, a seed-42 gate, and fresh seeds 55 to 57. Nothing is built.
- Next: tickets from the spec, the build, the final whole-branch review, then `r7 run`.
- Blocked: nothing. Night mode; the report is due Thursday 2026-10-15.

> Written 2026-10-12 02:36 by the `r7 decide 2` chat. Loop step reached: after step 6 (spec approved), before step 7.
> Direction log: this round's entry is written at its wrap-up.

## 1. Read in this order

1. The rule, binding: `src/test/20261126_design_l_style_source/DECISION_RULE.md` (L1 to L16, G1 to G4, K1, S1 to S3,
   P1, F1 to F3, §9 process, §10 inputs with SHA-256s).
2. The spec from §1 onward: `docs/superpowers/specs/2026-10-12-r7-design-l-design.md` (decisions D1 to D27).
3. `~/.claude/references/matt-chain.md` (verdict code, ticket size, the lean research build) and
   `~/.claude/references/codex-jobs.md` (Codex first for implementer tickets when `codex-usage` shows room).
4. The review trail, for context only: `src/test/20261126_design_l_style_source/rule_check/`.

## 2. Decided by the user, not to be reopened

- Everything in spec §4 (D1 to D27) and the rule, approved 2026-10-12 02:34 ("Yeah thats good").
- NrKmeans on CSD added as a descriptive grouping-step comparator ("That could be added", 2026-10-12); details in rule L7.
- The build runs overnight, unattended (run mode night, set 02:35).

## 3. Where things stand

- No round-7 code exists. The step-3 helper `r7_label_stats.py` is in the round folder.
- Stage 0 targets, all in the rule: AFF seed 42 pooled 19.137 per anchor; told CSD style × genre 21.82 per anchor; the A1
  reader retrained equal to round 1's; B 18.341064453125, B′(A0) 18.436686197916664, AFF − B′(A1) equal to round 6's `diff__S1`.
- Time-box: the run must finish by Tuesday; the report is due by Thursday 2026-10-15 23:59. Otherwise Friday is a
  NO-GO (rule §8).

## 4. Open points for the build (agent defaults; nothing for the user)

- Verdict code gets ticket reviews and unit tests: K1, S1 to S3 and P1, the chain (F1 to F3 with the Holm integer form),
  the bootstrap counts, the X-AFF gate, and the Phase K, A and B re-derivations. Plumbing (token extraction, groupings,
  heads, reader runs, seed builds) is proven by one end-to-end smoke (rule §9).
- The build computes no stage 1, 2 or 3 number. G1 to G4 may run in the build, since they reproduce known numbers.
- The full token extraction (GPU, about 15 min) and every long CPU job are launched by the controller, not by
  subagents.

## 5. Code and pitfalls

- The module prefix is `r7_`. Develop on seed 42 only, and read no held row (C5). Join rows through `row_to_image` (C6). Run every
  Python call with `PYTHONDONTWRITEBYTECODE=1` and 8 threads.
- Round 3's and round 6's modules guard their seeds: `r3_bundle._check_seed` and `r6_context.ADMITTED`. Use `r7_`
  copies that change only the guard and add the csd and X posteriors (rule §7.3). `r6_heads` has no guard.
- Round 1's bank builder sorts the keys, so the new grouping must use the key `csd` (rule L11, I2).
- Reused folders are read only. Never call `sp_run.run_one`, `run_r2_fusion.run_candidate` or the `rb_build` stages;
  they would overwrite G2's and G3's references (rule §9).
- Install ClustPy with `pip install --no-deps` so torch is untouched (rule L7).
- Run GPU work under `flock -n -o -E 75 /tmp/gpu0.lock`. New features go to `/data/SSD2/pre_extract/artelingo/clip_b32_tokstats/`.
- Codex's allow rule matches only commands that start with `codex e --dangerously-bypass-approvals-and-sandbox`.
- The DAS6 reservation ends about 2026-10-14 02:40; this round needs no cluster.

## 6. State at handoff

- Running: nothing. The CoSiR main chat opens with this cut, in its own Herdr workspace.
- Uncommitted: nothing after this cut's commit.
- On disk over 1 GB: `res/cluster_jobs/` 3.1 GB (round 6's pulled outputs; keep, ledgered).
