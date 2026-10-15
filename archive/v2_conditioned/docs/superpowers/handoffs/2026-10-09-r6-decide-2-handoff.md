# Handoff: write round 6's spec and decision rule (the ArtELingo held-split paper test of AFF) from the finished grill, and get the brief approved

> Written 2026-10-09 03:57 by the `r5 read` chat (it also ran r6's steps 2 to 5). Loop step reached: after step 5 (the
> grill closed; the user confirmed its summary), before step 6. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md) (decisions C, A and B recorded).

## 1. Read in this order

1. §2 below: every decision of the grill. The spec writes them into §4 without asking again (no second interview).
2. `src/test/20261125_artelingo_held_test/design/facts.md`: the facts the grill rested on, sections A to H, each number
   with its source (what AFF loads and where it was fitted, the held split, round 3's rule, K2 and §10, comparators,
   reuse hazards, seeds).
3. `docs/reports/literature/2026-11-25_held_claim_check.md`: the literature check (step 3); §1 specifies the new
   describe-then-score comparator and the protocol wording a CVPR reviewer will hold us to.
4. The templates: `~/.claude/rules/spec-templates.md`, `~/.claude/templates/spec_core.md`,
   `~/.claude/templates/spec_experiment.md` (claim test: body about 150 lines plus the rule; a baseline stage on seed 42
   precedes it) and `~/.claude/templates/decision_rule_template.md`.
5. `docs/superpowers/constitution.md` (version 2: C5 amended today), the plan
   `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §8, §10 and §16 (revision 4, today).
6. Round 3's rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md` and report
   `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` (Table 4, Table 11, §8): the model for the checks, the
   bootstrap and the frozen cells.
7. `~/.claude/references/loop-chats.md`, "Unattended chats": build and run never stop for the user.

## 2. Decided by the user, not to be reopened (2026-10-09)

- **Paper target** (plan §16): go/no-go moved to Fri 2026-10-16; an ArtELingo-centred paper first, the full paper when
  time and results allow (user's own).
- **Held reads** (constitution C5, amendment 2; plan §16; held ledger header): one read per pre-registered final method
  and backbone; nothing ever tuned or picked on held data; every read ledgered, reported and disclosed; the reserve
  stays a fix after a final-review finding. Design L and the second backbone get their own later reads (user, after
  seeing ours: split the held paintings in two).
- **What is tested:** AFF frozen exactly as in round 3, on the **full** ArtELingo held split; every fusion cell and
  weight frozen from seed 42 (plan §8; with seed 42's cells round 3 gave +0.585 against +0.591 cross-fitted).
- **Pass checks:** round 3's seven (AFF's fused reader beats cosine, RCA, B, B′(A0) and the matched counterpart on R@1;
  the gain statistic; RCA's condition gain). They contain K2's six (agent default, confirmed in the summary).
- **Bar (Q4):** every pass check's interval above 0 under Holm across the seven (plan §10); no minimum size; GO needs all
  seven (user's own, after an explanation of Holm; ours the same).
- **Episodes (Q5):** round 3's count, 4,096 per aspect pair per seed, seeds 52, 53, 54 (user; ours the same).
- **B′(A1), the style-aware scorer (Q2):** a pre-registered secondary check: computed and reported whatever happens,
  never decides GO; if its lower bound is above 0 the paper may say "also beats it" (user chose "reported beside", then
  agreed to the secondary-check form; the user hopes it passes).
- **Describe-then-score comparator (Q3):** built in this round. Qwen3-VL-8B states what the 4 support pairs share and
  the contrasts lack; CRL (NeurIPS 2025) turns the phrase into a text basis on frozen CLIP B/32; fused and controlled
  like every baseline; prompt and settings tuned on seed 42 with the same budget, then frozen; reported beside, with a
  true-aspect-name ceiling (privileged, reported only). Not a pass check (user switched from "pass check" after the
  scenario). **Stop rule:** if it beats AFF's R@1 on seed 42, the build stops before the held read and the user decides.
- **Claim (Q6):** pooled, "on new paintings"; per-pair results reported; the style × genre loss disclosed with design L
  as the route (user; ours the same).
- **Minor items, confirmed as listed in the chat:**
  1. R1 (the round 1 reader without AFF's gate): secondary check, as in round 3.
  2. The three CLIP fine-tune baselines (LP, LB, LoRA): reported; LB and LoRA need held image features (DAS6).
  3. Style set: the 23 development styles (held has one more, New_Realism, excluded).
  4. Seeds 52 to 54, hashes in the held ledger.
  5. The new comparator: as above.
  6. B′(A1): CSD classifier refit on scorer-train; round 4's code reviewed before reuse (round 5's final review §4).
  7. B's and B′'s seed-42 picks recomputed and stored before the read (never saved).
  8. Refit heads must reproduce the stored selection posteriors; if not, the build stops before the read.
  9. Safeguards: held-ledger row and a refuse-twice runner; a selection-row smoke of the same script bytes; the C12
     regression (round 3's seed-42 numbers); the C13 re-derivation; the final whole-branch review.
  10. Descriptive extras: two-way bootstrap and swap success (§10).
  11. Disclosures: H1 to H3 (value episodes on held rows), about 50 variants looked at on seed 42, near-duplicate held
      images, later designs made after this read.
  12. The in-context MLLM reranker (plan tier 2): reported, on seed 52 only.
  13. The second backbone: not in this read.
  14. Compute on DAS6 (three nodes); the local GPU only for small steps.

## 3. Where things stand

- AFF: GO on fresh seeds 49 to 51 in round 3, bar margin +0.591 [+0.462, +0.729] R@1 over B′(A0); style × genre
  −0.580. Seed 42: AFF 19.137, B′(A1) 18.805 (AFF − B′(A1) +0.332 [+0.048, +0.625]). Everything else is in facts.md.
- The held split: 12,281 paintings, about 9,900 anchor paintings per aspect pair (1.9 times the selection pool); never
  read with aspect episodes; CLIP B/32, Qwen3-VL-Embedding-2B and CSD features exist for it.
- DAS6: node401 (default), node402, node408, three GPUs each, with about 120 hours of reservation left when checked in
  the night of 2026-10-09; `node403` is gone. The cluster tool's selftest has not yet run on them.

## 4. Open points for step 6

- **Write the spec** (`docs/superpowers/specs/2026-10-09-r6-held-test-design.md`, a name to taste) from the core spec and
  the experiment module: a baseline stage on seed 42 (the new comparator, B′(A1) made held-ready, B and B′ picks, head
  refit checks, the stop rule), then the claim test on held. Its type is the agent default, listed under "Check these".
  No stop points between approval and the reports except the two pre-registered stops (the comparator beats AFF on seed
  42; the refit heads do not reproduce).
- **Ask only what the grill left open**, at most three questions. Candidates: the exact Holm family (the seven), the
  level of the secondary checks (95% unadjusted, as round 3), and how the stop rule compares R@1 (pooled point estimate
  over the three pairs, as written above).
- **The decision rule** from `decision_rule_template.md`, citing the constitution; reviewed by a fresh independent agent
  on the most capable model (a Codex second opinion is allowed, `agent-routing.md` job 6), then committed before any
  code (C7). Paste the brief to the user with the file link; the user approves from it.
- **`GLOSSARY.md`** at the repo root does not exist yet; create it with this round's new terms (at most two, e.g. the
  describe-then-score comparator).
- **Then cut:** open `r6 build` (unattended) with `loop-next`, and open design L's decide chat (`r7 decide`, with
  `--notify`; the user chose to do design L next in its own chat; start from the direction log's parked ideas, the first
  being design L and its 30-minute label-free check).

## 5. Code and pitfalls

- Round folder `src/test/20261125_artelingo_held_test/`; module names need a new prefix (suggest `r6_`), since round
  folders go on `sys.path`.
- Reuse (facts.md §A, §F): `r3_fusion` and `r3_stats` are pure; `r3_bundle` uses `run_gonogo.EvalContext`, which masks
  to selection rows and asserts it; its seed guard admits only 42, 49 to 51 and smoke seeds; `run_baselines.py` is
  selection-only. A held runner needs a held context, held episodes and held baselines.
- The image, caption, affect-km and CSD heads exist only as stored selection posteriors: refit on scorer-train (same
  draw) and check the refit reproduces them.
- The refuse-twice pattern: H3's `src/test/20261019_affect_factor_learning_held/run_held.py` (results or started file
  blocks a rerun, `--after-crash` recorded, a passing selection smoke of the same script bytes required).
- Round 5's lapses: every Python call with `PYTHONDONTWRITEBYTECODE=1`; mutation tests on copies; tests on the real data
  shape; times from `TZ=Europe/Amsterdam date`.
- The fine-tune checkpoints are under `res/cluster_jobs/2026100707*/code/outputs/clipft/`.
- Cluster: the first cluster run needs a commit whose subject says "cluster run", then `cluster sync`; run
  `cluster launch -- cluster-selftest gpu` on each new node first. `/local/wding` is per node, so data is copied to
  each node used.
- Commit and push without asking (`~/.claude/rules/git.md`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (this handoff, the direction log and chats.tsv are committed with the cut).
- On disk over 1 GB: nothing. The session scratchpad holds small files only (`count_held.py`, `counts.json`, a
  `cluster.conf` backup).
