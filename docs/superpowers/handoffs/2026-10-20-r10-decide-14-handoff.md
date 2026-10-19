# Handoff: write the claim test spec and its decision rule from the confirmed grill; then decision B for the cluster-loss track

## Brief
- Done: the claim test grill (8 key questions, closing summary confirmed); a development Leiden reading.
- State: D2 emotion AMI leads PercepT core by +0.019 under k-means and +0.009 under Leiden (development data).
- Next: write the two-stage claim test spec and its decision rule; reviewer, user approval; then decision B, track 2.
- Blocked: nothing.

> Written 2026-10-20 01:30 by "r10 decide 13" (cut at the context meter, 200k). Loop step reached: track 1 (claim
> test) after step 5, the grill done; step 6 (spec) next. Track 2 (cluster-level losses) after step 3, before
> decision B. Direction log entry: [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim
> test".

## 1. Read in this order

1. The direction log entry of 2026-10-20, bullet "Grill, claim test": every grill answer, with who decided. The spec
   carries these into §4 and does not ask them again.
2. §2 and §4 below: the details that the log bullet compresses, and what the spec must still settle.
3. The facts brief `.scratch/r10-claim-test/facts.md` (not in git): held split (§1), feature coverage (§2), D2 (§3),
   full-train cost (§4), PercepT core (§5), the paper settings edit sites (§6), Leiden code (§7), the scorer's
   5,000-row assumptions (§8), timings (§9).
4. The [development Leiden reading](../../../src/test/20261020_leiden_dev_reading/20261020_leiden_dev_reading_log.md).
5. Templates: `~/.claude/templates/spec_core.md`, `spec_experiment.md` (type "claim test", about 150 lines plus the
   rule), `decision_rule_template.md`; `~/.claude/rules/spec-templates.md`; `~/.claude/references/ste-writing.md`
   for the brief. Earlier specs to cite: [stage 2 sheet](../specs/2026-10-19-stage2-model-sheet-design.md) (D2),
   [PercepT core](../specs/2026-10-18-percept-core-design.md) (arms, A1 to A3).

Skills: none for the spec. Decision B (track 2) is a plain choice from the literature note's ranked list.

## 2. Decided by the user, not to be reopened

The log bullet holds each answer. The grill's labels, for the spec's §4 (D rows):

| Grill item | Decision | Decided by |
|---|---|---|
| Q1 | Two stages: the developed sample for the 10-29 gate (careful claim wording), then full data | user |
| Q2 | Paper settings PercepT enters the bar only when not collapsed; it also runs at full scale (its collapse may come from the sample size) | user |
| Q3 | Leiden 3a: union 20-NN graph of the held paintings' own output vectors, resolution 1.0, Leiden seeds 0 to 2; then 3b (Leiden on training outputs, mutual 20-NN with repair, held paintings join by a 20-NN vote) as a follow-up | user |
| Q4 | 4b: this spec = stage 1 (sample) + stage 2 (full ArtELingo train side, English captions); ArtELingo-28 gets its own facts brief and spec after the gate | user |
| Q5 | 5c: both paper settings versions, strict (Algorithm 1, γ = 2) and K = 40; arms P1 and P4; collapse thresholds as in `GLOSSARY.md` | user (5c); arms and thresholds: agent proposal shown in the option |
| Q6 | 6b: GO needs d ≥ 0.01 and the paired 95% interval of d above 0 | user |
| Q7 | 7a: Leiden is reported beside the verdict and decides nothing | user, after the development Leiden reading |
| Q8 | 8b: stage 2 is a claim test with its own rule, run only after a GO or inconclusive stage 1; after a NO-GO, back to step 2 | user |
| S1 | Stage 2's result counts over stage 1's; an uncollapsed paper settings PercepT that beats D2 there becomes the comparator to improve against | user |
| S6 | "Inconclusive, margin met": d ≥ 0.01 but the interval reaches 0; we improve the margin, choosing the change on development data, and test again | user (the user called it a "half-go"; no new name) |
| S2, S4, S5, S8 | Yes: the bar can differ by stage; a long ArtELingo-28 translation moves past 11-10; a K = 40 PercepT can raise the 10-29 bar above D2's development score; after an inconclusive stage 1, stage 2 runs and its GO is the paper's sentence | user |
| S3 | k-means at Leiden's own count: the user deferred it; agent default, a diagnostic that decides nothing | agent default |
| m1 to m10 | Accepted as recommended (below) | user, after seeing the recommendation |
| Plan page | No html-plan page (read from "yes to all"; default) | user |

d = D2's held emotion AMI minus the strongest uncollapsed PercepT version at its stronger reading (k-means k = 20 or
its own assignment, argmax of 40), as means over 5 model seeds; k-means k = 20 as the sheet (3 seeds, n_init 3).

Minors as accepted:
- m1: 5 model seeds (0 to 4) per arm; 3 k-means and 3 Leiden seeds per vector set.
- m2: held labels by the sheet's rule (emotion = mode of caption emotions, "something else" included, ties to the
  smallest code; style and genre from the first caption; genre missing on 2,332 held paintings, masked); facts §1.
- m3: stage 0 regression: D2 and PercepT core (P2, P3) seed 0, retrained on the sample with weights saved,
  reproduce the saved development outputs: output hash, or AMI within 1e-4 with the cause named if CUDA does not
  repeat bitwise; otherwise stop. The sheet's own stage 0 ran on CPU; D2's sheet outputs were trained on CUDA, and
  whether a CUDA rerun repeats them bitwise is not checked.
- m4: extract instructed Qwen images for the 12,281 held paintings, three instructions (about 150 MB), for the
  target row; no RoBERTa or instructed captions on held (held inference is image-only).
- m5: arm per aspect fixed from development data (P3 emotion, P2 style, P3 genre, for both readings: P3 0.082 /
  0.191 / 0.400 and P2 0.066 / 0.220 / 0.383 by own assignment, K40 report); the stronger reading and the strongest
  uncollapsed version are taken on held data, which only raises the bar; the report says so.
- m6: diagnostics after the verdict: V, ARI, FMI with a random baseline at the same count; k-means at Leiden's count;
  Leiden 3b; an in-sample reading on the fit paintings (PercepT's paper scores on its training set); style and genre
  margins; the instructed-image target row.
- m7: not here: TURTLE, TEMI (track 2's idea 3, or 11-10); retrieval and the instructed bar (11-10).
- m8: stage 2's early stopping uses the validation split (6,152 paintings); the 5,000 development evaluation
  paintings become training data. Assumption: this line never used the validation split (not checked).
- m9: names approved and added to `GLOSSARY.md`: "PercepT, paper settings, strict" and "PercepT, paper settings,
  K = 40".
- m10: local GPU under the lock; stage 1 at most about 2 to 3 h GPU (an estimate, paper settings arms included).

## 3. Where things stand

Development data, image-side emotion AMI (5,000 evaluation paintings, 3 seeds): D2 0.0962; PercepT core 0.0772
(k = 20) and 0.082 (own assignment); so the stage 1 bar today would be about 0.092, before any paper settings
version runs. The instructed-image target is 0.0929. Leiden lowers emotion AMI for D2 (0.0815), PercepT core
(0.0725) and the target (0.0811); counts 10 to 18 communities.

## 4. Open points for the next step

1. Write the spec, `docs/superpowers/specs/2026-10-20-claim-test-design.md`, from §2 above, with no question rounds of
   its own (`spec-templates.md`). Both stages are claim tests; parts per the parts table. Time-box for the spec:
   2026-10-23.
2. Write the decision rule (constitution C7, C8) from the spec, in the round folder (the round 4 example kept it as
   `DECISION_RULE.md` in its round folder). The trimmed constitution has no C9 to C11: the rule states its own
   statistics (paired bootstrap over held paintings, resampling the same paintings for every arm; partitions fixed;
   seed means; 95% percentile interval; the number of resamples is an agent default). One rule file can cover both
   stages, with stage 2's comparator named as "the strongest uncollapsed PercepT at full scale".
3. Rule review by a fresh reviewer on the most capable model (opus, `xhigh`), then commit, before any code.
4. Paste the spec's brief to the user for approval. Then cut to the build chat (ask about a container restart first,
   `loop-chats.md`).
5. Track 2, decision B, in the same chat if context allows: the literature note's ranked ideas (1 teacher-partition
   anchor, 2 TEMI-style head, 3 comparator probe) against the user's three forms; the note does not support plain DEC
   from a good partition. Key, answer first.

Points the spec must settle as agent defaults (list the main ones under "Check these"):
- Stage 2 details: graphs at 42,969 nodes (facts §4: about 10× the work), instructed Qwen extraction for the 23,080
  train paintings and 116,108 emotion captions not yet covered (about 1.7 GB, facts §4), RoBERTa for the P1 arm's
  missing train captions, early stopping on the validation split, and whether the training settings stay frozen (they
  should: the recipe is frozen, only the data grows).
- The held scorer: N from the vectors, labels from the held paintings' own captions, one painting order for every
  arm (C6), and the normalization (the probe's float32 re-normalization moved one k = 20 value by 0.0037; match the
  sheet: cast, check unit norm).
- The PercepT own-assignment reading on held: argmax of the 40 Stage 2 probabilities, as `native_img`.
- What each stage 2 outcome leads to, beside S1.
- Held ledger: a plain record of the reads (`held-data.md`); no read counts or warnings anywhere.

## 5. Code and pitfalls

- No weights exist for D2 or PercepT core (facts §3, §5): every arm retrains, with weights saved this time.
- The sheet's code asserts 5,000 evaluation rows and 19,889-row inputs (facts §4, §8); stage 1 needs only a held
  inference path and a held scorer, stage 2 a parameterized sample.
- New round folder, new module prefix (`sys.path` clash): for example `src/test/20261020_claim_test/`, prefix `ct_`.
- The paper settings edit sites are in facts §6; DEC target frequency per minibatch and the paper's convergence rule
  are not stated in the paper: choose and record them as agent defaults.
- Leiden: `leidenalg` 0.12.0, `igraph` 1.0.0 in the CoSiR env; the probe's `ld_probe.py` holds the 3a graph code.
- Codex room at 00:30: 5 h window 68% left, weekly 62%.

## 6. State at handoff

- Running: nothing (no GPU lock, background job, Codex job or subagent).
- Uncommitted: nothing after this cut's commit.
- On disk over 1 GB: `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger). The Leiden probe left
  92 KB.
