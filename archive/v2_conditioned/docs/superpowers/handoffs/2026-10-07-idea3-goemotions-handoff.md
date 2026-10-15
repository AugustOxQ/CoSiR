# Handoff: idea 3, GoEmotions placement of captions on AFF (measured step first), after round 4's kill

Written 2026-10-07 by the method-improvements tab of the CoSiR Herdr workspace, for a fresh agent in a new tab
(`idea3-goemotions`). The user drives the decisions; the ones already taken are in §2 and are not to be reopened.

## 1. Read in this order

1. **Round 4's user-read briefing** `docs/user_read/2026-10-07_round4_vetoes.md` (what round 4 tried, why no veto
   helped, the open decisions). For the story up to round 3: `docs/user_read/2026-10-06_reader_fix_affect_steering.md`.
2. **Round 4's full report** `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md` (final-reviewed): Summary, §1
   (terms), §3 (method), §6 (why no veto helped), §7 (disclosures, including the process lapses), §8.4 (rulings and
   what they cost), §9 (what follows). Round 4's rule `src/test/20261122_round4_aff_vetoes/DECISION_RULE.md` and round
   3's rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md` are the templates for the next rule. Round 4's run
   log `src/test/20261122_round4_aff_vetoes/20261122_round4_aff_vetoes_log.md` and SDD ledger
   `.superpowers/sdd/2026-10-07-round4-aff-vetoes/progress.md` show the process; its final review
   `src/test/20261122_round4_aff_vetoes/final_review/final_review.md` lists what to fix before reusing the code.
3. **The brainstorm's idea 3** `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md` §3.3 (mechanism, why it
   should work, label-free reasoning and the counterpart, cheapest test, risk, cost), with §2.2 (where the either-rate
   cost sits) and §2.3 (the perfect-emotion-gate ceiling +1.805 is bounded by the head-based affect term).
4. **The affect grouping and its source:** `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`
   §3 (pair lift 2.71 on the groups, 1.145 through the heads), §4, §5 (the Leiden graph's k and resolution), §8; the
   GoEmotions code `src/data/affect.py` (`load_goemotions`, `goemotions_probabilities`; model
   `SamLowe/roberta-base-go_emotions`, 28 sigmoid labels, batch 256, max_length 64); the scorer-train probabilities
   `src/test/20261018_affect_factor_learning/cache/affect_prepare.npz` (`affect_probs`, 183,694 × 28, scorer-train rows
   only: `run_affect.py` joins only scorer-train captions, its row-scope rule); the 41 communities `partition_L` in
   `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz` and how they were made
   (`src/model/communities.py::detect_communities`, the kNN union graph; `run_told_oracle.py::merge_small` merges
   communities under 200 rows into the nearest centroid by Euclidean distance in the 28-d space; `fit_one_head`,
   `global_labels`).
5. Project memory (`project_v2-publication-plan-pending.md` top entries, `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`, `feedback_final-review-catches-real-issues.md`) and the global rules in
   `~/.claude/rules/` (`agent-routing.md`, `final-review.md`, `shared-resources.md`, `storage.md`, `timestamps.md`,
   `debug-and-changelog.md`, `reports-layout.md`, `report-writing.md`, `user-read-reports.md`).

## 2. Decided by the user (2026-10-07), not to be reopened

- **Round 4's kill stands.** No veto on AFF's gate beat AFF on seed 42; no test seed was built; **AFF, frozen as tested
  in round 3, stays the current best**. The veto direction on A0 is considered spent.
- **Idea 3 first, before the held-split paper test.** Reason: the held-read budget for ArtELingo aspect episodes is 1
  main + 1 reserve read (`docs/superpowers/held_ledger.md`, 0 of 2 used). Running idea 3's cheap measured step first
  keeps both reads for whichever method wins; testing AFF on held rows first and then a later winner would spend the
  reserve.
- **Measured step first, a round only if it looks good** (the brainstorm's plan): GoEmotions on the selection
  captions, a placement of each caption into the 41 affect communities, the detection quality and the sharper term
  measured before any round is written.
- **Process as in rounds 3 and 4:** a short spec the user approves (brainstorming flow: propose, the user approves, one
  question at a time); the rule written from it; a fresh Opus check of the rule; fixes; commit before any code; the
  rule sent to the user with SendUserFile (`proactive`); a short plan and one question for the execution method
  (subagent-driven is the norm); subagents implement; the main session launches every real run; an independent agent
  re-derives every decision number; the rule applied; the whole-branch final review on the most capable model (one fix
  wave, a scoped re-review) **before** the full report is committed; a user-read report when the user asks.
- **Commits:** allowed on `main` without asking, scoped by explicit path, never `git add -A`, never push; `bin/`,
  `docs/paper/` and `.DS_Store` stay untracked; end commit messages with your session's attribution lines.

## 3. Where things stand (facts; R@1 in percentage points)

| Item | Value | Source |
|---|---|---|
| AFF on the fresh seeds 49 to 51 (round 3) | bar margin +0.591 [+0.462, +0.729] against B′(A0), pooled over the three aspect pairs; style × genre −0.580 | round 3 report |
| AFF on seed 42 (development) | fused R@1 19.137, counterpart 18.396, bar margin +0.700 against B′(A0) | round 3 rule §5 item 3 |
| B′(A0) / B′(A1) on seed 42 | 18.437 / 18.805; AFF minus B′(A1) +0.332 [+0.048, +0.625], below it on style × genre | round 4 report §5, §9 |
| Round 4 (vetoes V4, V2, V24) | kill: Δ against AFF −18, −18, −34 net rankings of 49,152; V2 and V24 failed the bar against B′(A1) | round 4 report |
| Caption head of the affect grouping | 35.7% held-out accuracy; AFF's margin over its counterpart larger with a caption query (+0.671) than an image query (+0.216), on R1, seed 42 | brainstorm §3.3 |
| Episode seeds | 42 development (read very many times); 43, 45 to 51 spent; **52 and later free**; 9001 to 9003 smoke | `docs/superpowers/episode_seed_ledger.md` |
| Row counts | selection 32,413 rows (6,451 paintings; all episode seeds draw from them); held 61,744 rows (12,281 paintings; reserved) | `src/data/artelingo_splits.py`, reports |
| GoEmotions cost | 183,694 scorer-train captions took 85 s on the GPU; the 32,413 selection captions: under a minute on the GPU, roughly 10 to 30 minutes on CPU (estimate) | `affect_prepare.json` |

**The user's other open decision** (from round 4's briefing, not needed for idea 3's measured step): how B′(A1) enters
the eventual paper test (GO check, reported comparator, or not at all).

## 4. Idea 3 in short (the brainstorm's §3.3; our summary)

Place every caption in the Leiden affect grouping by its own GoEmotions probabilities (nearest communities of the
scorer-train rows in the 28-d GoEmotions space) instead of by the CLIP caption head; keep the image head. This changes
the caption-side posterior p_affect(t), hence the support and contrast agreements S_affect, C_affect that the reader
reads, and the steering term s_affect(q, k) wherever the query or the candidate is a caption. The groupings themselves
stay as they are. Why: the communities carry emotion strongly (pair lift 2.71) and the heads weakly (1.145); a sharper
affect term needs less weight for the same condition gain, which is where AFF's either-rate cost sits (about 0.52 of
either rate per point of gain). It is the only remaining idea that raises the ceiling rather than redistributing
steering. Label-free: GoEmotions is the affect grouping's own external source (disclosed); no evaluation label is used.
**B′ and the matched counterpart must be rebuilt with the same caption placement**, so that any condition-free value of
the sharper term is credited to the comparators.

## 5. Open points for the spec (ask the user, one at a time; our proposals)

1. **Does it count as the parked grouping redesign (design L)?** The round-4 spec deferred this question to now.
   Proposal: no; it changes how captions are placed into the existing groups, not the groups.
2. **Placement method.** Options: (a) soft kNN posterior in GoEmotions space (share of each community among the k
   nearest scorer-train rows; consistent with the kNN graph the communities came from); (b) a multinomial logistic
   regression from the 28 GoEmotions probabilities to the community, fitted on scorer-train rows (like the heads, but
   on GoEmotions instead of CLIP); (c) nearest centroid (hard). Proposal: (b) or (a), chosen and frozen in the rule;
   measure its held-out placement accuracy on scorer-train rows against the caption head's 35.7%.
3. **What changes.** (a) The steering term only (AFF's gate and reader stay on the standard heads, so AFF's side
   mechanism is untouched); (b) also the reader's input features S_affect, C_affect (the frozen half-readers were
   trained on head-based features: a distribution shift); (c) both, measured separately. Proposal: the measured step
   reports both, and the rule pre-registers which one may go to a round.
4. **The measured step's own bar** (pre-registered, seed 42 only, decides whether a round is written): for example the
   detection AUC of the new affect evidence for emotion conditions against R1's 0.787 (the brainstorm suggested above
   0.83), and AFF-with-the-new-term's paired gain over AFF in fused R@1 with B′ and the counterpart rebuilt. Then a
   round as in round 4: development already on seed 42, a fresh test on seeds 52 to 54 with the paired AFF check in
   GO.
5. **Comparators and floor:** B′ rebuilt with the same caption placement; whether B′(A1) also enters (it uses the CSD
   heads, which idea 3 does not read).
6. **GPU:** GoEmotions on the 32,413 selection captions on the shared local GPU (check
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and use `flock -n -o -E 75 /tmp/gpu0.lock`)
   or on CPU. Held captions only when a method reaches the paper test. DAS6 node 404 is available if needed (user,
   2026-10-07), but not expected to be.

## 6. Code and pitfalls

| What | Where |
|---|---|
| Seed-parameterised pipeline with the A1 extension (round 3's A0 bundle plus CSD posteriors, A1 features, A1 readers, B′(A1), v) | `src/test/20261122_round4_aff_vetoes/r4_bundle.py` (and round 3's `r3_bundle.py` underneath) |
| Candidate gates, `run_candidate`, development record, carry, GO checks, sensitivity | `r4_fusion.py`, `r4_stats.py` |
| Seed-42 runner: regression items in order behind a guard, development step, carry, boundary stop | `run_r4_seed42.py` |
| Constants and input asserts; **sets `r3_common.TEST_SEEDS = (52, 53, 54)` on import** | `r4_common.py` (the next round needs its own copy, its own rule SHA and its own seeds) |
| Independent re-derivation (phase 1) | `src/test/20261122_round4_aff_vetoes/rederive/rd4_*.py` |

- **Not built in round 4** (the kill made them unnecessary): the test-seed build runner, the GO/descriptive phases, the
  rule application and the end-to-end wiring smoke test, and the rule §10 test-seed tests (round 4 final review S3).
  Round 3's versions (`run_r3_build.py`, `run_r3_test.py`, `r3_apply_rule.py`, `test_r3_wiring.py`) are the pattern.
  They and the final review's before-reuse items must exist before any test-seed run.
- **Round 4's process lapses, not to repeat** (its report §7): the re-derivation must write its own code (do not let
  it import round 3's re-derivation code unless the rule lists it); keep the implementation blind to the
  re-derivation's results (do not write them into the ledger or log before the implementation's runner is written);
  do not skip the rule's listed tests without asking the user; run **every** Python call, `--help` included, with
  `PYTHONDONTWRITEBYTECODE=1`.
- No module may share a name with a module of rounds 1 to 4 (`common`, `rc_core`, `rb_*`, `r2_*`, `r3_*`, `r4_*`,
  `run_*`, `test_*` of those rounds): their folders end up on `sys.path`. Use the prefix `r5_` (or a topic prefix).
- z-score per ranking row first, then gate; the matched counterpart is G_cf under the candidate's own gates; cross-fit
  ties and the carry compare integer sums.
- Smoke and dry runs never print or log a metric value; read their logs with grep for markers only. Round 1's
  `load_bundle` prints seed-42 values: wrap it.
- `run_baselines.py` rewrites `codes_provenance.json` on every build: build fresh seeds in one invocation; never open
  the build logs.
- Re-derivation agreement: τ within 1e-15 absolute or 1e-9 relative; points and bounds within 1e-9 pp; a lower bound
  within 1e-12 of 0 goes to the user before the verdict.
- The GoEmotions row-scope rule of the affect factor-learning line: the affect model was applied only to scorer-train
  captions there. Applying it to selection captions is new; selection rows are development rows (free to reuse), the
  held rows are not touched until the paper test.
- Times with `TZ=Europe/Amsterdam date '+%F %H:%M'`; never estimate them.
- CPU work: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, at most 3
  processes, `uptime` and `free -g` first, `run_in_background` for real runs.
- Commit the full report only after the whole-branch final review and its fix wave.

## 7. State at handoff

- Round 4's commits on `main`: 7f0a00c (spec), 09fd742 (rule and plan), aa70988, 15b2eae, c8a7169, 89914bd, 17e7976
  (code), 06d7ae1 and 25ba23d (re-derivation), 149112e (kill recorded), 7f7553b (final-review code fixes), 0bba3fa
  (report), 9667373 (user-read briefing), and this handoff. Nothing pushed.
- No process of this line is running. Storage left behind, gitignored, nothing over 1 GB:
  `src/test/20261122_round4_aff_vetoes/rederive/out/` about 148 MB, `final_review/out/` about 12 MB, `results/` about
  2 MB.
- Suggested names: folder `src/test/20261123_<topic>/` (folder dates are sequence numbers, next free after 20261122),
  report `docs/reports/auto/v2/2026-11-23_<topic>.md`, spec `docs/superpowers/specs/2026-10-07-<topic>-design.md` (use
  the calendar date of the day you write it).
- The new tab `idea3-goemotions` and its Claude agent `idea3` were opened by the method-improvements tab with Herdr;
  the start prompt is `docs/superpowers/handoffs/2026-10-07-idea3-goemotions-start-prompt.md`.
