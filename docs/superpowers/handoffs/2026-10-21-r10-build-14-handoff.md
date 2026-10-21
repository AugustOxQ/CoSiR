# Handoff: build the D2 ablation and the four-view analysis (tickets, build, final review, merge), then open the run chat

## Brief
- Done: the claim test's wrap-up; the grill and spec of a D2 building-part ablation with a four-view analysis.
- State: spec approved (user confirmed the closing summary, 03:30); 13 rows, development data, local 3090, no bar.
- Next: tickets from the spec, build, final review, merge to main; then open "r10 run 12" unattended.
- Blocked: nothing. Night mode. Item 5 (backbones, ArtELingo-28) waits for the morning decide chat.

> Written 2026-10-21 03:40 by "r10 decide 16". Loop step reached: step 6 done (spec), step 7 (build) next.
> Direction log entry: [2026-10-21 · Claim test read](../direction_log.md).

## 1. Read in this order

1. The spec, [2026-10-21-d2-ablation-design.md](../specs/2026-10-21-d2-ablation-design.md), from §1 (the brief is the
   user's page).
2. `~/.claude/references/matt-chain.md` (the build chain), `~/.claude/references/codex-jobs.md` (Codex implementers).
3. The facts brief `.scratch/r10-decide-16/f1_facts.md`: D2's parts and the switch for each (§1), existing rows (§2),
   costs (§4), the four Qwen view stores (§5), pitfalls (§7). It is gitignored but on disk.

Skills: `to-tickets` and `implement-spec` per `matt-chain.md`; `tdd` for the new switches and graphs.

## 2. Decided by the user, not to be reopened

- The spec's §4 rows marked "user" or "user, after seeing ...": direction, device (local 3090), rows R0 to R12, data
  (development only), seeds, scores, the analysis lists, no bar, no plan page, item 5 in the morning, no restart.
- Night mode (user going to sleep, 03:14): no questions; agent defaults, recorded in the handoff.

## 3. Where things stand

- Code to reuse: the model sheet round `src/test/20261019_model_sheet/` (`ms_train.py`, `ms_rows.py`, `ms_score.py`;
  rows set with `--row D2 --name NAME --set KEY=VALUE`) and the aspect graph round
  `src/test/20261018_aspect_graph_sheet/` (`ags_graphs.py`, `ags_head.py`, graphs in `graphs/`).
- Switches that need new code (spec §3.2): R3 teacher source, R5 buddy coefficient, R6 own-caption coefficient, R7
  image endpoints only, R9 default-view graph, R10 degree-preserving random graph. Each is default-off; with all off,
  the path must stay bit for bit (stage 0, spec §3.5).
- Existing flags: R1 `share=separate`, R2 `distill=none`, R4 `graph_terms=false` (spec D13 sets its batch to 1,024
  paintings and weight 0.5), R8 `graphs.emotion=R0, graphs.style=R0o` (check both rows exist), R12 `inputs=instructed`.
- Cost on the 3090 (f1 §4): about 35 s per D2 seed for all three aspects; scoring about 20 s per row and seed.
- Node408 is no longer prepared for CoSiR (its inputs were deleted at the user's yes); the run is local.

## 4. Open points for the next step

a. Tickets (`to-tickets`) citing the spec's D and SC IDs, ending with the coverage line (spec-templates.md hard rule 6).
b. Build per `matt-chain.md`: no verdict code, so the controller's diff read, unit tests for the new switches and
   graph builders, one end-to-end smoke (a CPU or short GPU smoke under the GPU lock), the final whole-branch review.
   Codex implementers when `codex-usage --room implementer` says room (the estimate is now 40%).
c. The run queue: stage 0 (R0, R2 regression against the model sheet's saved D2 and T1 outputs), stage 1 (the other
   rows), stage 2 (CPU analysis). One queue script per GPU stage, ending with one summary line.
d. At the end: merge into main, push, write the run handoff (it lists the tickets and the run commands), and open
   `loop-next <handoff> --label "r10 run 12" --unattended`. Night mode: no restart question; run the GPU check only.
e. The run chat's side work (spec D20): the literature check (deep-research, Opus, deep), the framing note (Opus
   `xhigh` worker), the ArtELingo-28 overlap fact. Then the auto and user-read reports, and it opens
   `--label "r10 decide 17" --notify` for the morning: decision C on the ablation and item 5's grill.

## 5. Code and pitfalls

- New round folder `src/test/20261021_d2_ablation/` with a new module prefix (`ab_`); round folders go on
  `sys.path`, so check resolved imports (f1 §7).
- The claim test round `src/test/20261020_claim_test/` is closed: do not edit `ct_*.py`, `ct2_*.py`, its queue scripts
  or `DECISION_RULE.md`.
- A model sheet rerun with any changed config replaces a verified cohort: write to the new round's own output root,
  never into `src/test/20261019_model_sheet/outputs/`.
- The cluster CLI is `~/.claude/skills/cluster-run/cluster`; the `cluster` on PATH is an unrelated graph tool.
- GPU work under `flock -n -o -E 75 /tmp/gpu0.lock`; unattended chats wait with `flock -o -w`.
- Delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no Codex job, subagent, GPU lock or node job).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: `src/test/20261020_claim_test/outputs/stage2/` (4.6 GB, keep, ledger row).
