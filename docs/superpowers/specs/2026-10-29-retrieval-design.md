# Retrieval by a named aspect on held ArtELingo: label retrieval, the aspect-switch table, graded emotion and a read-out sweep: design

> Date: 2026-10-29 18:05 · Status: approved 2026-10-29 18:30 (from the brief, r11 decide 2; no container restart) · Kind: experiment
> Builds on: [report 12-28](../../reports/auto/percept/2026-12-28_recipe_sweep.md) (c* frozen); the grill in the r11
> decide 2 chat (2026-10-29, 16:20 to 17:30; its answers are §4) · Constitution: [constitution.md](../constitution.md)
> Decision rule: inline in §6, with §3.4 and §3.5 · Reviewer: a fresh Opus `xhigh` worker, 2026-10-29 17:55 (pass
> after fixes: three must-fix and fifteen should-fix items, all applied)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The paper needs a second task beside clustering. This loop tests whether c* (the frozen recipe: model
G with 40 graph neighbours) retrieves paintings by a named aspect. A query is a held painting plus an aspect name
(emotion, style or genre); a hit counts when it shares the query's label for that aspect. Nothing is retrained.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks, CPU): the seed ranges for "level" are measured on validation paintings and committed; then the
   new code's inputs reproduce r10's held clustering numbers exactly.
2. Stage 1 (claim test, held): label retrieval, mAP and P@10, for c* and six baselines, plain cosine.
3. Stage 2 (claim test, held): the aspect-switch table of each model; c*'s test: in each label column, the head of
   that aspect beats c*'s other two heads.
4. Stage 3 (confirmation, held): emotion scored by how much two paintings' caption emotions overlap (nDCG).
5. Stage 4 (sweep, validation): up to 25 ways to compare saved embeddings (centering, whitening, query expansion);
   each model picks its own from the same menu. Stage 5 (claim test, held): stage 1 again with the picks.

**Compared against.** The bar: the plain Qwen image (it ignores the aspect name) and the Qwen image instructed for that
aspect (c*'s teacher). Reported without a bar: CLIP L/14, SigLIP 2, PercepT core's topic vector and our
re-implementation of CRL (NeurIPS 2025: frozen CLIP compared through an LLM-written list of the aspect's values).

**Outcomes.** Per aspect, as in r10: WIN (interval above zero), LEVEL (no win, lower bound above minus one seed range),
BELOW. GO when c* WINs over the plain Qwen image on all three aspects and is LEVEL or WIN against the instructed image
on each. The paper's claim rests on stage 5; stage 1 is the untuned row. NO-GO: the next step aims at the failing
aspect. Emotion rides on mAP; nDCG agrees ("holds") or is reported beside it. The switch test is reported either way;
you decide later whether the paper claims it. We expect GO on emotion and style; genre is at risk against the
instructed image (clustering: −0.0060, level).

**Cost.** CPU about 1 hour at 8 threads; GPU about 10 minutes (CRL's value lists), light. Build about half a day and
run under 2 hours, both unattended; the report by 2026-10-31.

**Check these.**
1. The switch test is written now and its claim decided later: your pick after seeing our recommendation.
2. The read-out sweep (stages 4 and 5) is in this spec and runs right after stage 3: our reading of "i first".
3. "Level" uses seed ranges measured on validation paintings before any held retrieval number.
4. CRL: 50 values per aspect listed by an 8B Qwen model from the aspect name, CLIP L/14; the lists may contain the
   label names, which helps CRL.
5. Stage types: claim test for stages 1, 2 and 5; confirmation for 3; sweep for 4.

## 1. Context and goal

Asked: "I want to do the mix of fall, follow the order of 1, then aspect-switch-table then graded emotion relevance.
You can take into account the cluster node for parallel experiments." On the optimization steps: "I think we do i
first (it's cheaper) then decide to go deeper into i or continue to ii." (user, 2026-10-29)

r10 froze c* (report 12-28): held clustering AMI above the instructed Qwen image on emotion (+0.0145) and style
(+0.0041), level on genre (−0.0060), 5 seeds. The plan to the deadlines (direction log, 2026-10-29) puts retrieval in
r11: a fast build, first results, a few small optimization steps, a good result before 11-10. The query form is fixed:
a held painting plus an aspect name, no text queries (user, 2026-10-25).

Goal: measure c*'s retrieval by a named aspect on held ArtELingo paintings against the plain and instructed Qwen
images, with no retraining.
Question this step answers: does the one condition-free model trained for clustering also retrieve by the named
aspect, at or above the instructed Qwen image?

## 2. Scope

In: stages 0 to 5 on ArtELingo with Qwen3-VL-Embedding-2B features; c*'s saved held and validation embeddings; the six
baselines; the auto report and the user-read report.
Out: retraining c* (the grill's option ii; the user's next decision after this report); other datasets (r12); text
queries (decided out 2026-10-25); CLAY (CVPR 2026, keyword-conditioned retrieval; r13 if needed); a bar against CRL
(decided after its number); c* on other encoders.

## 3. Design

### 3.1 Hypothesis and prior

Claim: c*'s head for aspect a ranks held paintings by a's labels better than the plain Qwen image, and at least level
with the Qwen image instructed for a. Mechanism: each head learns from buddy graphs on its aspect's instructed view and
is distilled toward the instructed image, so its neighbours should share a's label more often than the plain image's.

Prior, before any number: retrieval follows clustering. Emotion: WIN against both. Style: WIN against plain, LEVEL or a
narrow WIN against instructed. Genre: WIN against plain, LEVEL against instructed with a real risk of BELOW. Emotion mAP
will sit near its chance level (majority labels from about 5 captions, 9 classes), so its absolute gains will look
small. Read-outs: whitening may help the plain and instructed images more than c*, whose outputs span at most 1,025
dimensions. Switch test: the emotion and style columns pass; the genre column is at risk, since styles and genres
co-occur and the style head may rank genre well (not checked). A surprise: c* below the plain Qwen image on any aspect.

### 3.2 Arms and controls

| Arm | What it is | Role |
|---|---|---|
| c* | r10's arm k11=40 (model G with 40 graph neighbours), full scale, seeds 0 to 4; the head of the query's aspect; saved held and validation embeddings | candidate |
| plain Qwen | Qwen3-VL-Embedding-2B default image; the same ranking for every aspect | comparator (bar) |
| instructed Qwen | the Qwen image under r10's aspect instruction (emotion focus, style word, genre word); c*'s teacher | comparator (bar) |
| CLIP L/14 | image embedding | descriptive |
| SigLIP 2 | so400m image embedding | descriptive |
| PercepT core | r10's PercepT entry, own reading: arm P3, 40-topic probabilities, seeds 0 to 4 | descriptive |
| CRL, our re-implementation | CLIP L/14 image features mapped to their cosine with the CLIP L/14 text embeddings of the aspect's value texts (D12); these vectors are compared | descriptive |

No arm is trained. Every arm gets the same read-out menu in stage 4 and picks on the same validation paintings. The
narrow arms (PercepT core, 40 wide; CRL, 50 wide) have one whitening size, their full width, so their menus hold 15
read-outs instead of 25.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| 12,281 held paintings (genre labels on 9,949) | held: scored only; no fit, no selection | 0 (AMI replay), 1, 2, 3, 5 |
| 6,152 validation paintings (genre labels on 4,960) | development: seed ranges, the read-out fit (no labels) and its selection | 0, 4 |
| 42,969 training paintings | not touched | none |

The validation paintings already chose c*'s epochs and are inside r10's development set P, on which c* was picked among
about 60 configurations (§8).

### 3.4 Measurement

- **Retrieval.** For aspect a: the queries are the paintings with an a-label, in the held list's order (validation:
  sorted names); the gallery is the same set without the query. Similarity is cosine (stored norms are 1 within 3e-4;
  inputs are L2-normalized anyway). Ties keep gallery row order (stable sort). A gallery painting is relevant when its
  a-label equals the query's. Every query has at least one relevant painting (the smallest class: 15 held, 5
  validation paintings); a query without one would be dropped and counted.
- **Metrics.** AP per query over the full ranking, mAP its mean (the form CLAY reports); P@10, descriptive. Graded
  emotion: the gain of gallery painting x for query q is Σ_c min(p_q,c, p_x,c), with p a painting's caption-emotion
  distribution over all its captions in the ArtELingo training annotations (held paintings have 1 to 12, median 5).
  DCG = Σ_rank gain / log2(rank + 1) over the full ranking; nDCG = DCG / the DCG of the gallery sorted by gain.
- **Seeded arms** (c*, PercepT core): per-query values averaged over seeds 0 to 4.
- **Statistics** (r10's joint bootstrap, adapted to queries). Each comparison (arm pair × aspect × stage) has its own
  `rng = numpy.random.default_rng(0)` and 10,000 draws. Per draw, in this order: I_b = `rng.integers(0, n_a,
  size=n_a)` over the aspect's n_a labelled queries in the order above; then, when either arm is seeded, S_b =
  `rng.choice([0, 1, 2, 3, 4], size=5, replace=True)`, one draw shared by both arms. The difference d_b is the mean over
  I_b (and S_b) of candidate minus comparator. The 95% interval is `numpy.percentile(d, [2.5, 97.5])`. The point value
  uses all queries and all five seeds. The gallery stays fixed.
- **Outcomes** (r10's T-WIN, T-LEVEL, T-BELOW): WIN when the lower bound > 0; LEVEL when not WIN and the lower bound >
  −r_a; BELOW otherwise. r_a is the seed range: max minus min over seeds 0 to 4 of c*'s validation mAP (nDCG for stage
  3) under read-out R0, per aspect. SC10 to SC15 use the same r_a.
- **Switch tables.** Per arm, cell (h, a) is the mAP of the ranking by head h (the query's aspect name h) scored with
  a-labels, over the a-labelled paintings as queries and gallery; each column shows its chance level Σ_c p_c² over
  a's class shares. c*'s test: in column a, head a against each other head, with the bootstrap above.
- **C13.** Separate code re-derives, from this rule and the data and never from our numbers: r_a (mAP and nDCG), the
  SC point values, intervals and outcomes, stage 3's outcomes, and stage 4's scores and picks. It must match within
  1e-6, with the same outcomes and the same picks, before the verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0a | measurement | validation, read-out R0 | r_a (mAP and nDCG) per aspect written to a file and committed before any held retrieval number | stop the chain, note it for the user |
| 0b | regression check | the new code's held matrices and label vectors, passed to r10's frozen partition and AMI functions; c*'s weights | (i) AMI within 1e-9 of r10's stage 4 values: c* 0.10658729378204021 / 0.40664322288431654 / 0.5001038781216712; instructed Qwen 0.09207439052277983 / 0.40256525412113753 / 0.5061513009972589; PercepT core P3 (own reading, emotion) 0.08742922158584242. (ii) The plain Qwen held matrix (float32, normalized, held order) has the sha256 recorded as `image_sha256` in c*'s held sidecars. (iii) Genre: 9,949 labelled, 2,332 masked. (iv) Re-inference from the weights, all 15 heads, validation: largest absolute element difference to the saved outputs at most 1e-3; the held sidecars' weight hashes equal the weights' | stop the chain, trace the cause, note it for the user |
| 1 | claim test | held, R0 | SC1 to SC6 | report; the next decision targets the failing aspect |
| 2 | claim test | held, R0 | SC7 to SC9 | report |
| 3 | confirmation | held, R0 | holds or qualified (§6) | report |
| 4 | sweep | validation | one read-out per arm (§6, selection rule) | report |
| 5 | claim test | held, stage 4's picks frozen | SC10 to SC15 | report |

R0 is plain cosine. Stages 1 to 3 run before stage 4; stage 5 starts only after stage 4's picks are written.

### 3.6 Diagnostics

Measured after the verdicts; they decide nothing.
- Per-class AP for c* and the instructed Qwen image per aspect: which classes carry a gain.
- c*'s held per-seed mAP spread beside the validation seed range r_a.
- Chance mAP per aspect; P@10 for every arm.
- CRL's value lists, and how many items equal a label name.
- The switch tables of every arm beside c*'s.

### 3.7 Frozen vs rerun

Frozen: c*'s weights and its saved held and validation embeddings (r10, arm k11=40); every baseline store; PercepT
core's P3 outputs; the labels (r10's painting labels); r_a after stage 0a; stage 4's picks before stage 5. Rerun:
scoring only.

## 4. Decisions

Grill rows: the r11 decide 2 grill, 2026-10-29 16:20 to 17:30.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Query form | a held painting plus an aspect name; no text queries | c* has no text side | user, after seeing the recommendation | 2026-10-25 |
| D2 | Protocol | label retrieval, then the aspect-switch table, then graded emotion, in that order | (one or two of them) | user | 2026-10-29 grill |
| D3 | Which emotion metric carries the claim | mAP; nDCG after it, reported beside it if it differs | (nDCG alone; both required) | user | 2026-10-29 grill |
| D4 | The bar | c* WIN over the plain Qwen image on all three aspects, LEVEL or WIN against the instructed image on each | (above both images; plain only) | user | 2026-10-29 grill |
| D5 | Baselines in the main table | plain and instructed Qwen, CLIP L/14, SigLIP 2, PercepT core, CRL | (no CRL; CRL and CLAY) | user | 2026-10-29 grill |
| D6 | CRL in the bar | no: descriptive; a bar against it decided after its number | (in the bar now) | user | 2026-10-29 grill |
| D7 | The switch table's claim | tables reported; c*'s column test written now and reported either way; its claim decided after the numbers | (descriptive only, test written later) | user, after seeing the recommendation | 2026-10-29 grill |
| D8 | What optimization may change | the read-out only, no retraining; then the user chooses more read-out work or retraining | (retraining now) | user | 2026-10-29 grill |
| D9 | Minor protocol choices | cosine; query out of its gallery; genre on labelled paintings only; mAP and P@10; nDCG with the overlap gain; joint bootstrap, 10,000 draws | the grill's minor batch, not objected to | agent default | 2026-10-29 grill |
| D10 | Seed range r_a for LEVEL | c*'s per-seed max minus min on validation paintings, committed before any held retrieval number | (the held run's own spread, known only with the result) | agent default | 2026-10-29 17:50 |
| D11 | PercepT core's row | r10's PercepT entry, own reading: arm P3 topic probabilities, seeds 0 to 4, cosine | r10's PercepT comparator; P1 is not it | agent default | 2026-10-29 17:50 |
| D12 | CRL settings | Qwen3-VL-8B-Instruct (cached snapshot 0c351dd), greedy, prompt "List 50 distinct values of the following criterion for describing paintings: {aspect}. One value per line, no numbering." with aspect = emotion, style or genre; parse: strip numbering and bullets, lowercase, drop empty lines and repeats, keep the first 50 (fewer than 40 left: note it, use them); value text "a painting with the {aspect} of {value}"; text and image encoder openai/clip-vit-large-patch14 (the store's encoder) | the public code's form (GPT-written lists of 22 to 318 values, the template "objects with the {criterion} of ", CLIP B/32); differences: a local 8B model for GPT, 50 values, CLIP L/14 to match our CLIP row; hence "our re-implementation" | agent default | 2026-10-29 18:05 |
| D13 | Read-out menu and selection rule | §6 | read-outs that are standard in image retrieval; one pick per arm keeps one model per arm; cross-fitting avoids an in-sample whitening fit | agent default | 2026-10-29 18:05 |
| D14 | Stage types | §3.5 | stages 1, 2 and 5 feed paper claims; 3 checks stage 1; 4 picks among settings | agent default | 2026-10-29 17:50 |
| D15 | Which held table carries the paper's claim | stage 5 (each arm with its validation-picked read-out); stage 1 reported as the untuned row | (stage 1 alone; both required) | user | 2026-10-29 18:00 |

**Amendments**:
- A1 (build, 2026-10-29, agent default, before any result): the saved validation P3 outputs come from PercepT's full-scale mapper (fit on 42,969 paintings), the held ones from the sample-scale mapper (13,400; r10's entry, D11). Validation P3 is re-inferred from the held mapper's weights (`ct_percept.infer_stage2`, same seeds), so stage 4 fits and picks on the model it is applied to. Held, D11 and stage 0b are unchanged. Places: §4, §5; ticket 01 and the round log.

## 5. Assumptions and open questions

- Assumption: the validation files under c*'s run folder (`R6-5`: emotion Gres, style Fres05, genre R0o) are c*'s
  heads. Checked: the reviewer's CPU re-inference of seed 0 matched within 6.1e-5; stage 0b checks all 15.
- Assumption: the validation and held instructed images share prompts and model. Checked: the same instruction text
  and model revision (9f2f7e7) in both stores' metadata.
- Assumption: PercepT core's held and validation P3 outputs come from the same mapper. Found wrong in the build
  (different mappers); fixed by amendment A1 (§4).
- Assumption: Qwen3-VL-8B-Instruct is in the HF cache. Checked: snapshot 0c351dd is there.
- Open questions: none (D15 answered).

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1, SC2, SC3 | c*'s mAP, emotion / style / genre, read-out R0 | plain Qwen | WIN | held |
| SC4, SC5, SC6 | the same | instructed Qwen | LEVEL or WIN | held |
| SC7, SC8, SC9 | switch column emotion / style / genre: c*'s head for that aspect | each of c*'s other two heads | WIN against both | held |
| SC10 to SC15 | as SC1 to SC6, each arm with its stage 4 pick | plain and instructed Qwen with their own picks | as SC1 to SC6 | held |

- **GO** for a table: all six of its criteria pass. **The paper's claim rests on stage 5 (SC10 to SC15)**; stage 1 is
  reported as the untuned row (D15). **NO-GO** otherwise; per-aspect outcomes are recorded and the next decision
  targets the failing aspect. No inconclusive outcome: the three outcomes are exhaustive. A failed stage 0 leaves no
  verdict.
- **Switch GO:** SC7 to SC9 all pass. Reported either way; whether the paper claims it is the user's call (D7).
- **Stage 3 (confirmation):** emotion nDCG, c* against the plain and instructed Qwen images, with r_a from validation
  nDCG. It "holds" when both outcomes equal SC1's and SC4's; otherwise "qualified", reported beside them. It changes no
  outcome (D3).
- **Selection rule (stage 4).**
  - Menu: W ∈ {none, centering, PCA whitening to 64, 256 or 1024 dimensions}, a size at or above the arm's width
    becoming whitening at full width (repeats dropped), each W followed by L2 normalization; crossed with Q ∈ {none,
    average query expansion with the top k = 1, 3, 5 or 10 gallery neighbours: the query replaced by the normalized
    mean of itself and them, then the gallery ranked again}. At most 25 read-outs per arm.
  - Whitening: x ↦ diag(λ_i + ε)^(−1/2) U_dᵀ (x − μ), with μ, U and λ from the fit set's covariance and ε = 1e-6 · λ_1;
    centering subtracts μ. Both are fit without labels on every painting of the fit set, per arm, aspect and seed.
  - Cross-fit: validation is split into two fixed halves by the parity of each painting's position in the sorted name
    list. Each read-out, R0 included, is fit on one half and scored on the other (queries and gallery inside that
    half), and the score is the mean over the two halves.
  - Score: r10's mean gain, (1/3) Σ_a (mAP_a − mAP_a(R0)) / r_a, with seeded arms averaged over seeds. One pick per arm,
    for all three aspects. Guard: no aspect more than r_a below its R0 value; c*'s r_a for every arm.
  - Ties: within 0.001 (absolute, in the score's units) of the best, the order is lexicographic: R0 first, then no
    expansion before expansion, none < centering < whitening, smaller k, larger whitening size.
  - Stage 5 fits each picked W on all 6,152 validation paintings and applies it to held; query expansion uses the held
    gallery (embeddings only, no labels) with k fixed by the pick.
- **Licensed claim (GO, stage 5):** "On 12,281 held ArtELingo paintings with Qwen3-VL-Embedding-2B features, c*, given
  a painting and an aspect name, retrieves paintings with the same emotion, style or genre label better than the plain
  Qwen image, and level with or better than the Qwen image instructed for that aspect (mAP, 5 seeds, each model with a
  read-out picked on validation paintings)." Stage 1's GO licenses the same sentence with "with plain cosine". Not
  licensed: other datasets or encoders, text queries, any claim against CRL, CLIP, SigLIP or PercepT core, per-class
  margins.
- **Licensed claim (switch GO):** "c*'s head for an aspect retrieves that aspect's labels better than its heads for
  the other aspects", on the same data.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the plain image (strongest condition-free Qwen input) and the instructed image (c*'s teacher) carry the bar; PercepT core and CRL beside them |
| C6 sample IDs | pass | c*'s and PercepT's npz rows join by position to the held list; name-keyed stores join by painting name; stage 0b checks the new code's own joins through r10's numbers and hashes |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker, committed before any code |
| C8 prior and disclosure | pass | prior in §3.1; disclosure in §8 |
| C12 regression check | pass | stage 0b |
| C13 re-derivation | pass | Codex re-derives r_a, the SC numbers and stage 4's picks from the rule and the data |
| `held-data.md` | pass | stage 4 fits and selects on validation paintings; held is scored only; stage 5's query expansion reads held embeddings at test time, no labels, k fixed on validation; a later read-out step again selects on validation only |
| Stage types | pass | stage 2 stays a claim test: SC7 to SC9 are reported whatever they show, and the claim decision (D7) comes after |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Emotion majority labels are noisy, so emotion mAP sits near chance with wide intervals | chance mAP; interval width | stage 3's nDCG beside it; no re-scoring after results |
| Styles and genres co-occur, so off-diagonal switch cells are high for every arm | the plain Qwen table (three equal rows); chance levels | the test compares c*'s heads with each other, never with zero |
| The validation seed range differs from the held spread, so LEVEL is too strict or too loose | the diagnostic held spread beside r_a | reported; r_a is not changed after results |
| CRL's value lists contain the label names, a privileged hint | the lists | reported as such; CRL stays descriptive |
| A wrong join scrambles an arm | stage 0b's numbers and hashes | stop, trace |
| Selection disclosure: the validation paintings chose c*'s epochs and sit inside P, where c* was picked among about 60 configurations; stage 4 adds up to 25 read-outs per arm | validation numbers are inflated | only held numbers are results; stage 4's picks are frozen before held is scored |

## 9. Deliverables and review

- At the end: a new round folder (module prefix `rt_`) with code, tests, the log and per-query outputs; the auto
  report `docs/reports/auto/percept/2026-12-29_retrieval.md` with its results brief and index row; a user-read report.
- Reviews: this rule, done before the commit; tickets per `matt-chain.md`, with the metric, bootstrap, outcome and
  selection code as verdict code (unit tests against scikit-learn's AP and nDCG on tie-free inputs, one test for the
  tie rule; an Opus `high` ticket review); the C13 re-derivation by Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-10-31 23:00.
- Budget: CPU about 1 hour at 8 threads (about 30 similarity matrices of 12,281² on held, and about 1,400 of 3,076² in
  stage 4's halves; an estimate, not measured). GPU: the CRL lists, about 10 minutes, light, under the lock, outside
  19:00 to about 20:15 tonight (the Affection extraction). No DAS6 needed; the node matters for retraining (option ii).
  Wall time: build about half a day, run under 2 hours.
