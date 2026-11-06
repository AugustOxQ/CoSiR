# c*'s caption side on held rows: design

> Date: 2026-11-06 05:00 · Status: approved 2026-11-06 05:09 (from the brief, r23 decide 2; no container restart) · Kind: experiment
> Builds on: [report 2027-01-09](../../reports/auto/percept/2027-01-09_text_retrieval.md) (r22, exploration) and
> [report 2027-01-10](../../reports/auto/percept/2027-01-10_pair_hit_checks.md) (short checks); decisions C and A of
> 2026-11-05 and 2026-11-06; the grill in the r23 decide 2 chat (2026-11-06, two rounds and a closing summary; its
> answers are §4) · Constitution: [constitution.md](../constitution.md) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: the r15 spec ([2026-11-02-mmart-test-design.md](2026-11-02-mmart-test-design.md)): its test cut, evaluation
> order, MMArt labels and C13 form, and the WIN, LEVEL, BELOW outcomes of r14 §3.4; the r22 round log: rows, hit rules
> and leak checks · Reviewer: a fresh Opus `xhigh` worker, 2026-11-06 04:40 (pass after fixes: seven must-fix, twelve should-fix and seven nit items, all applied; [review](../../../src/test/20261106_text_claim/tcl_rule_review.md))

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** r22 found, on validation with one seed, that c*'s caption side beats plain Qwen and the teacher's text
on label hits in 6 of 9 cells. A cell is one aspect in one direction. The short checks found that late fusion (a weighted
sum of c* and plain Qwen cosines, our earlier "joined scores") also keeps pair hits near plain Qwen. This loop tests both
on data that shaped nothing, for the paper.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): the new text stores pass their checks, and the new code reproduces r22's numbers.
2. Stage 1 (sweep, validation only): five seeds; seed ranges and read-outs are picked and committed before any held score.
3. Stage 2 (claim test): held ArtELingo (12,281 paintings, 61,744 captions) and MMArt's new rows (16,384 paintings outside
   ArtELingo, narrative text). c* stays frozen, and a stays 0.7.
4. Stage 3 (diagnostics, after the verdicts): why a = 0.7 helps, the side rows, the other MMArt texts.

**Compared against.** Plain Qwen and the teacher's text, the Qwen text embedded under each aspect's instruction.

**Outcomes.** Per data set, c* and late fusion each get a verdict; the data sets do not affect each other.
- c* holds: a WIN over plain Qwen in 9 of 9 cells and a WIN over the teacher's text in 5 or more. Qualified: 9 of 9 and 3
  or 4. Fails: anything else. Each teacher cell is also its own claim: WIN "better than", LEVEL "level with or better
  than", BELOW nothing.
- Late fusion: the same label bar, and pair R@10 no lower than 0.02 below plain Qwen in text to image and image to text,
  at the lower end of the interval.
  A miss on pairs means fails.
- A failed verdict stands. We expect c* to hold on ArtELingo (0.6), on MMArt (0.3). Late fusion most likely ends
  qualified or fails: "holds" needs a validation cell with a margin of 0.0018. A failed check that cannot be fixed gives
  inconclusive for that set, never fails.

**Cost.** GPU about 2.5 to 3 h, CPU about 1 h (our estimates). Local, jobs under 1 hour each. Report by Saturday 11-07,
ahead of the 11-10 target.

**Check these.**
1. Your picks after seeing our recommendation: 9 of 9 over plain Qwen, so 8 of 9 means fails (D4); the verdict form of late fusion (D5); each set's verdict on its own (D7); the minor batch (D8, D10 to D12).
2. MMArt's bar text is the narrative text (D9): you were unsure, and confirmed after our recommendation. Its emotion margin on validation is thin (+0.0082). If c* misses a WIN over plain Qwen in one cell, MMArt's verdict is fails, and the unified text cannot replace it.
3. "MMArt test rows" means r15's new rows, not the 11,528 rows in the handoff (D6, your pick).
4. Stage 2 is a claim test, stricter than the confirmation you chose, with the outcome words kept (D2).
5. Two agent defaults: the pair cells read one head and read-out picked on validation (D16), and the interval uses paintings as its unit (D15).

## 1. Context and goal

Asked: "I will go with 3, that's a less tight bar and we won't miss anything / Then a. / Let's go with i." (the grill),
on the claim loop decided in the r23 decide chat.

r22 and the short checks are exploration numbers: validation, seed 0. c* is above plain Qwen in 9 of 9 label cells and
above both comparators in 6 of 9 (5 of 9 with one fixed read-out). Late fusion at a = 0.7
gives pair R@10 of 0.3215, 0.4954, 0.1952 against plain Qwen's 0.3102, 0.5068, 0.1823. No text row has been scored on held
paintings, and MMArt has no teacher text row.

Goal: test whether frozen c*'s label gains in text and cross-modal retrieval hold on data that shaped nothing.
Question this step answers: on held ArtELingo and MMArt's new rows, where does c* beat plain Qwen and the teacher's text,
and does late fusion keep those gains with pair hits near plain Qwen?

## 2. Scope

In: frozen c* (seeds 0 to 4, three heads) and late fusion on ArtELingo validation and held, and on MMArt validation and
test; extraction of the missing text; the diagnostics of §3.6.
Out: any training (the own-caption retrain stays declined, user); other encoders' c*; the photo model; SemArt and word
queries (exploration numbers stay in report 2027-01-09); the shared 11,528 MMArt rows (they can follow if MMArt fails);
clustering; the wording of the pair-hit limit, which waits for the user's talk with the other authors.

## 3. Design

### 3.1 Hypothesis and prior

Claim tested: c*'s caption side orders text and images by aspect better than plain Qwen on unseen paintings, and better
than the teacher's text in most cells. Mechanism: its heads learn aspect maps from instructed graphs, and its caption side
reads the plain Qwen text at low cost. Late fusion adds plain Qwen's cosine to c*'s; why a = 0.7 works is not known.
Facts (validation, seed 0): c* minus plain Qwen is +0.0132 to +0.0914 with intervals above 0 in all 9 cells; c* minus
teacher is +0.0057 to +0.0407 in the 6 style and genre cells and -0.0064 to -0.0335 in emotion's 3. Style's 3 cells and
genre text to image hold in every check; genre image to text and text to text depend on the read-out. On MMArt validation, unified text, text to image: c* 0.2180, 0.3085, 0.5080 against plain Qwen 0.1991, 0.2166, 0.4726; no other direction and no teacher row exists. Narrative text: c* 0.2040, 0.2232, 0.4941 against plain Qwen 0.1958, 0.1549, 0.4456. Late fusion minus plain Qwen pair R@10 on validation, five seeds, 2,000 draws, as an estimate only: ArtELingo text to image +0.0091 [+0.0060, +0.0124], image to text -0.0086 [-0.0155, -0.0016]; MMArt unified text +0.0009 [-0.0026, +0.0044], -0.0028 [-0.0064, +0.0007]; the narrative text is not measured.

Priors, written before any held number:

| Verdict | Holds | Qualified | Fails |
|---|---|---|---|
| c*, ArtELingo | 0.60 | 0.32 | 0.08 |
| c*, MMArt | 0.30 | 0.25 | 0.45 |
| Late fusion, ArtELingo | 0.20 | 0.40 | 0.40 |
| Late fusion, MMArt | 0.15 | 0.25 | 0.60 |

Surprise: c* not WIN over plain Qwen in any ArtELingo cell. MMArt priors are lower since its text is generated, long and new to
the caption side, which trained on caption means.

### 3.2 Arms and controls

| Arm | Text | Image | Role |
|---|---|---|---|
| c* | caption side of the aspect's head, on single texts | image side of that head | candidate, seeds 0 to 4 |
| late fusion | vector (sqrt(1 - a) c*, sqrt(a) plain Qwen), a = 0.7; same head and seed as c* | same | candidate, seeds 0 to 4 |
| plain Qwen | plain | default image | comparator, the best plain encoder in 6 of 9 validation cells |
| teacher's text | under the aspect's instruction | image under the same instruction | comparator; on MMArt, the ArtELingo instructions as r15 |
| side rows, ArtELingo held | free variant (captions through the image side); CLIP L/14, SigLIP 2, SigLIP plain; RoBERTa, emotion text to text | | reported, no bar |

At raw cosine, the cosine of two late fusion vectors is (1 - a) c* cosine + a plain cosine. Arms share queries, galleries and labels.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo validation, 6,152 paintings, 30,872 captions | development: gates, seed ranges, picks | 0, 1, 3 |
| MMArt validation, 5,769 paintings, MMArt labels | development: gates, seed ranges, picks | 0, 1, 3 |
| ArtELingo held, 12,281 paintings (genre: 9,949 labelled), 61,744 captions | held: scored only | 2, 3 |
| MMArt test: r15's cut set of the new rows, 16,384 paintings, its keys and evaluation order | held: scored only | 2, 3 |
| Training paintings and MMArt's shared and train rows | not touched | none |

Labels follow r22 (ArtELingo: a caption's own emotion, a painting's plurality emotion, style and genre of the painting;
unlabelled rows leave query and gallery) and r15 (MMArt: 9 emotion classes, 20 style, 10 genre; a text takes its painting's
labels). In the bar, MMArt uses the narrative text (D9). The other four texts, unified, formal, emotional and historical, are read in stage 3.

### 3.4 Measurement

- **Cells.** Three aspects by text to image, image to text and text to text: 9 label cells per set. A row reads its
  head for its aspect (plain Qwen has one vector). Label mAP is the r22 metric. A query's own row leaves the gallery only.
- **Pair cells.** R@10 with the r22 hit rule: the own painting, or any own caption. Only text to image and image to text
  enter the bar. Pair cells cover every painting and every text of the set, labelled or not; a query is 1 when its first hit rank is at most 10, else 0. Text to text has no bar, and MMArt has no text to text pair since each painting has one text per kind.
- **Read-out menu.** r22's `evaluate`, unchanged: raw cosine; centering; centering plus expansion k = 10 (the query plus its top 10 gallery rows, normalised, then ranked again; a query's own row never counts). Centering subtracts the mean of the row's text and the mean of its image rows, one mean for text to text, then L2-normalises. Means use every validation row of the set, labelled or not, with the same row, head, seed and text kind. For late fusion, the row is the concatenated vector: the mean and the normalisation act on it whole. Ranking and expansion use every gallery row of the set, labelled or not; unlabelled rows are masked only at AP, as r22. Expansion reads no label, and k is fixed.
- **Picks (stage 1, D11).** Per row, text kind, aspect and direction, the read-out with the best label mAP on the set's
  validation rows, mean over seeds for seeded rows. §3.6's text kinds are picked in stage 1 too, before any held number.
  Ties take the earlier read-out in the menu's order.
- **Pair picks (stage 1, D16).** On the set's validation rows, per direction. For each late fusion head: the read-out with the best MRR, mean over seeds. Then the head with the best R@10, mean over seeds, at its picked read-out. Plain Qwen: the read-out with the best MRR. Ties: earlier read-out in the menu; heads in the order emotion, style, genre. One head and read-out per direction and set, frozen before stage 2.
- **Seed ranges (stage 1).** For each candidate and cell, maximum minus minimum over seeds 0 to 4 of its label mAP on
  the set's validation rows at its picked read-out. Unrounded. It is the LEVEL unit of that cell.
- **Interval (D15).** Per comparison, one `default_rng(0)`, 10,000 draws. The units are the paintings with at least one scored query in that cell, in evaluation order (ArtELingo: sorted names; MMArt: r15's sha256 order); call their count m. In each draw, in this order: I = `rng.integers(0, m, size=m)`; S = `rng.choice([0,1,2,3,4], size=5, replace=True)`, shared by both sides. With w_p the times painting p is in I, a seed's value is the sum over p of w_p times the sum of its query values, divided by the sum over p of w_p times its query count, so repeats count in both. The draw's value is the candidate's mean over the five entries of S minus the comparator's value (unseeded, no S). A scored query is one with a finite AP (label cells) or any query (pair cells). Point value: the mean over all scored queries and five seeds. Interval: `np.percentile` at 2.5 and 97.5, default linear.
- **Outcomes per teacher cell.** WIN when the lower bound is above 0; LEVEL when not WIN and the lower bound is above minus
  the seed range; BELOW otherwise. Over plain Qwen only WIN counts. No multiplicity correction (as r14, r15).
- **C13.** Codex re-derives, in separate code and without our scored outputs, from this rule, the stores, the weights, the keys and the labels: the forward passes, late fusion vectors, means, per-query AP and first hit rank of every bar row; the seed ranges and picks from validation; every bar cell, its interval and outcome; Q; and all four verdicts. Within 1e-6, before any verdict is written. It may run on the GPU under the queue.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new stores; forward paths; scorer | (i) to (iv) below | stop that set's path, trace, fix and rerun; the report tells the user the difference and its cause |
| 1 | sweep (development only) | ArtELingo and MMArt validation | ranges and picks written and committed before any held number | that set's verdicts inconclusive |
| 2 | claim test | ArtELingo held, MMArt test | SC1 to SC4 | report |
| 3 | descriptive | all | §3.6, after the verdicts | report |

- (i) Stores: count, finite values, unit norm within 1e-3, keys join (C6). Each store's manifest gives the exact instruction string. Held teacher text is compared on 200 validation rows with `emotion_focus_s2txt` and r22's `style_word` and `genre_word`; plain MMArt test text with r22's `mmart_val`. MMArt teacher text has no reference, so its 200 rows only test a rerun.
- (ii) Forward, 15 weight files: the image side on ArtELingo validation, held and MMArt test reproduces r10's validation and held exports and r15's `new` exports within 1e-3; the caption side on normalised caption means reproduces the stored validation `cap` within 1e-3.
- (iii) Scoring replay within 1e-5: fed r22's validation rows with means fitted on those rows, the new code gives r22's label mAP for c*, plain Qwen and the teacher's text, seed 0, at every read-out; and the short checks' late fusion at a = 0, 0.7 and 1, label and pair; and r22's MMArt validation label mAP and pair R@10 for c* and plain Qwen, seed 0, unified text, text to image, raw and centering, through r15's labels and order. On a synthetic case of 6 paintings, 2 seeds, and a hand-computed I and S, the interval code gives the hand value within 1e-12; on r22's per-query differences, one seed, its bounds lie within 3e-4 of r22's Table 2.
- (iv) Sets: held 12,281 paintings and 61,744 captions; MMArt test 16,384 keys, in r15's order.

### 3.6 Diagnostics

After the verdicts, deciding nothing. **Why a = 0.7 helps** (D10): on ArtELingo validation and held rows, and MMArt test as well, at the same cost, split queries into both
right, only c*, only plain Qwen and neither, at pair R@10 and per direction, with late fusion's hit rate in each group; the
same split by AP above or below the median for label cells; the spread of c* and plain cosines, and a times one spread over
(1 - a) times the other, since a acts on raw cosines; the rank correlation of the two scores on a sample of queries.
Also: the validation curve of a on seeds 1 to 4 (changes nothing); pair R@1, R@5, R@10 and median rank for every row and
head, c*'s included; macro mAP; every cell at each fixed read-out; rows without label-naming text and without the query's
own painting (r22's leak checks); per-class AP for emotion text to image; side rows; MMArt's other four texts for c*, plain
Qwen, the teacher's text and late fusion, rows without label-naming text per aspect for narrative and unified text, and seed spreads on held beside
the validation ranges.



### 3.7 Frozen vs rerun

Frozen: c*'s 15 weight files; a = 0.7; r15's test cut; r22's rows and hit rules; stage 1's ranges and picks before stage 2.
Rerun: text extraction, c* forward passes, all scoring.

## 4. Decisions

Grill rows: the r23 decide 2 chat, 2026-11-06, about 03:50 to 04:30.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Rows, and the pair-hit limit | c* main row, frozen; late fusion one extra row; the limit is not worded by any agent; c*'s pair R@10 reported only | a retrain risks every earlier result; the limit needs the other authors | user, after seeing the recommendation (rows); user (limit) | 2026-11-05, 11-06 |
| D2 | Stage types | medium loop; stage 2 a claim test with the outcome words holds, qualified, fails; stage 1 a sweep; stage 3 descriptive | the result feeds a paper claim, as r15's D6; a type may only get stricter (confirmation) | user, after seeing the recommendation (minor e) | 2026-11-06 |
| D3 | The bar for c* | two levels, per set: holds 9 of 9 over plain Qwen and 5 or more over the teacher's text; qualified 9 of 9 and 3 or 4; fails otherwise; teacher cells also own claims | "a less tight bar and we won't miss anything" (one count, r22's form; per cell with no count, r15's form) | user | 2026-11-06 |
| D4 | 8 of 9 over plain Qwen | stays fails; both levels need 9 of 9 | the claim can say every cell; smallest validation gap +0.0132 (qualified accepts 8 of 9) | user, after seeing the recommendation | 2026-11-06 |
| D5 | Late fusion's role | own verdict: D3's label bar plus the lower bound of late fusion minus plain Qwen pair R@10 above -0.02 in text to image and image to text; holds needs label holds, qualified needs label qualified, a pair miss fails; a is not repicked | gives a second claim and a third choice on the pair-hit limit (labels only; no bar). The bar is the user's; the form was shown in scenario 2 | user (bar); user, after seeing the recommendation (form) | 2026-11-06 |
| D6 | Which MMArt rows | new rows, r15's cut set, 16,384 | a second data set (shared rows only; both) | user | 2026-11-06 |
| D7 | Verdicts across sets | each set has its own; neither changes the other | scenario 3 | user, after seeing the recommendation | 2026-11-06 |
| D8 | a | 0.7 for all seeds and both sets | one number for the paper (a pick per seed) | user, after seeing the recommendation (minor a) | 2026-11-06 |
| D9 | MMArt text in the bar | the narrative text. Own-label word share (emotion, style, genre): narrative 2.3%, 0.0%, 4.7%; unified 15.7%, 83.2%, 37.6%; ArtELingo captions 11.5%, 0.2%, 2.8%; labelled validation rows; so it tests what ArtELingo tests, and 9 cells, not 45 (unified, the first pick, was shown the emotion share only) | 9 cells, not 45 | user, after seeing the recommendation (minor b) | 2026-11-06 |
| D10 | Why a = 0.7 helps | §3.6, after the verdicts, on ArtELingo validation and held; MMArt test as well, at the same cost | held rows cost seconds, and a paper sentence needs them (validation only) | user, after seeing the recommendation (minor c) | 2026-11-06 |
| D11 | Read-outs | per row, aspect and direction, on validation over five seeds; MMArt on its own validation; fixed read-outs reported after the verdict | r22's rule (one read-out per row) | user, after seeing the recommendation (minor d) | 2026-11-06 |
| D12 | Side rows | §3.2; SemArt and word queries out | a table for the supplementary | user, after seeing the recommendation (minor f) | 2026-11-06 |
| D13 | Device | local GPU, jobs under 1 hour each; heavy jobs that fall between 10:00 and 19:00 on a weekday wait for 19:00, lifted for this run on 2026-11-06 (Amendments) | a DAS6 move needs a data sync for about 2 h of work | agent default, not objected to; the lift: user | 2026-11-06 |
| D14 | Term | "late fusion", with "joined scores" as its old name; WIN, LEVEL, BELOW enter the glossary | the field's term | user, after seeing the recommendation | 2026-11-06 |
| D15 | Interval | §3.4: r11's form with paintings as the unit | caption queries of one painting are not independent (r22 used paintings, one seed) | agent default | 2026-11-06 |
| D16 | Pair cells | one head and read-out per direction from validation | the bar needs one row; a best head per seed adds a pick (best head per draw) | agent default | 2026-11-06 |
| D17 | Stores | new text stores in the shared cache, `pre_extract/artelingo_full/` and `pre_extract/mmart/`; run outputs in `/data/SSD2/cosir_r23/` | `storage.md` | agent default | 2026-11-06 |
| D18 | Round | folder `src/test/20261106_text_claim/`, prefix `tcl_`; report sequence date 2027-01-11 | a new prefix per round | agent default | 2026-11-06 |

**Amendments**
- 2026-11-06 12:49, before any held score, user: "No one will be in the office, so start running freely." The 19:00 wait of D13 is lifted for this run. It changes when the jobs run. It changes no rule, row, pick or bar. Places: this section's D13 row; `tcl_queue.sh` (`R23_D13_LIFTED=1`).

## 5. Assumptions and open questions

- Assumption: c*'s caption side has weights for all 15 files. Checked: yes, 2026-11-06.
- Assumption: plain Qwen, CLIP L/14, SigLIP 2 and SigLIP text cover held captions. Checked: the stores hold all 308,723 rows;
  held rows are confirmed by gate (iv).
- Assumption: extraction rates hold on test rows: 60 to 122 texts per second by text kind (r22, batch 64), 295 for
  captions. Not checked on test rows.
- Assumption: scoring costs about 1 h. Scaled from r22's 3.26 minutes; not measured. RoBERTa has no held captions; extraction takes about a minute, GPU, same settings (max length 64).
- Open questions: none.

## 6. Success criteria

**Decision mode.** The rule is this section with §3.4 and §3.5.

**Selection rule.** Stage 1 only: §3.4's picks and pair picks, exact ties as written, no tie band. Stage 2 selects nothing. **Outcome words.** Holds is GO; qualified is GO for the qualified claim; fails is NO-GO (as r15, D2).

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | c*, 9 label cells, at picked read-outs | plain Qwen; teacher's text | P = cells WIN over plain Qwen; T = cells WIN over the teacher's text | ArtELingo held |
| SC2 | as SC1, narrative text | as SC1 | as SC1 | MMArt test |
| SC3 | late fusion, 9 label cells; pair R@10, text to image and image to text | as SC1; plain Qwen for pairs | P, T as SC1; Q = both pair lower bounds above -0.02 | ArtELingo held |
| SC4 | as SC3, narrative text | as SC3 | as SC3 | MMArt test |

**c* (SC1, SC2), per set.** Holds: P = 9 and T >= 5. Qualified: P = 9 and T is 3 or 4. Fails: anything else.
**Late fusion (SC3, SC4), per set.** Holds: P = 9, T >= 5 and Q. Qualified: P = 9, T is 3 or 4, and Q. Fails: anything else.
**Inconclusive.** A failed gate voids only what it serves, after the fix and rerun that this spec allows: (i) on a store a bar cell reads, or (ii), or (iv): that set's two verdicts and its cell claims; (i) on a store only §3.6 reads: that diagnostic; (iii): all four verdicts and all cell claims; a missing stage 1 record: that set's. A bar cell that cannot be computed, or a C13 difference over 1e-6 that stays after its cause is traced, voids the verdict that reads it. An inconclusive verdict is never counted as fails or qualified.

**Teacher cells.** Every cell of SC1 to SC4 is its own claim: WIN "better than the teacher's text", LEVEL "level with or better than", BELOW nothing. These stand whatever the verdict of a set that is not voided.

**Licensed claim.** c* holds or is qualified: "On 12,281 held ArtELingo paintings, c* (five seeds, read-outs picked on validation) beats plain Qwen in label retrieval in all nine aspect and direction cells, and beats the teacher's text in T of them, with one encoder pass and a small head per aspect, where the teacher's text needs an encoder pass per aspect." MMArt: the same sentence with "on 16,384 WikiArt paintings outside ArtELingo, with generated narrative text and MMArt's labels" and "its emotion in 2.3% of rows, its style in 0.0%, its genre in 4.7%". Late fusion holds or is qualified: "A weighted sum of c* and plain Qwen cosines (a = 0.7) beats plain Qwen in label retrieval in all nine cells, and beats the teacher's text in T of them. Its pair R@10 is not more than 0.02 below plain Qwen's in text to image and image to text, at the interval's lower bound." Pair R@10 on MMArt is near its ceiling, so the pair clause says less there than on ArtELingo; R@1 and median rank stand beside it. Not licensed: any sentence on what c*'s pair numbers mean. The report lists them in a table, with no wording, until the user has talked to the other authors. Neither covers other encoders or human text on MMArt. The word "better" applies per named cell only.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 comparator | pass | plain Qwen and the teacher's text; other plain encoders stand beside them on ArtELingo held only; on MMArt the strongest plain row is not measured (D12) |
| C6 sample IDs | pass | keys and caption ids from the stores; gate (i) |
| C7 rule before code | pass | §3.4, §3.5, §6, reviewed by a fresh Opus `xhigh` worker, committed before any code |
| C8 prior and disclosure | pass | §3.1, §8 |
| C12 regression first | pass | gates (ii), (iii) |
| C13 re-derivation | pass | §3.4 |
| `held-data.md` | pass | no pick reads held rows; expansion reads held embeddings without labels, with k fixed; a, read-outs and ranges come from validation |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| Means fitted on validation fit held rows badly | raw cosine and centering rows disagree in stage 3 | report; verdict stands |
| Per-cell picks favour c* (6 of 9 against 5 of 9 on validation) | fixed read-out rows in stage 3 | report beside the verdict |
| Text to text hits come from a query's own painting | r22's row without the own painting | report; the claim names it |
| MMArt's narrative text was generated by Qwen3-VL-8B-Instruct, so it may favour Qwen rows; it names its emotion in 2.3% of rows, its style in 0.0%, its genre in 4.7% | the rows without label-naming text | report; the MMArt sentence names it |
| MMArt's emotion labels differ from ArtELingo's (43% agreement on shared paintings) | emotion cells | report beside the verdict |
| Long text exceeds the encoder's length limit | gate (i), and the extraction log's truncation count | report the count; the build adds it; longest MMArt unified text is 1,310 characters; limit 512 tokens |
| Late fusion's weight fits seed 0 | the curve on seeds 1 to 4 | report; a stays 0.7 |

**Selection disclosure.** On ArtELingo validation: three read-outs per cell, 11 weights of a, one seed for a, two caption
routes (c* and the free variant, with the rule fixed on c*). c* itself was picked on validation clustering in r10. So
validation numbers are slightly high; held rows and four more seeds protect the verdicts. No text row of held ArtELingo or MMArt test has shaped any choice.

## 9. Deliverables and review

- At the end: round folder and log `src/test/20261106_text_claim/`, outputs `/data/SSD2/cosir_r23/20261106_text_claim/`,
  report `docs/reports/auto/percept/2027-01-11_text_claim.md`, user-read report `docs/user_read/2026-11-07_text_claim.md`.
- Reviews: rule review (C7); ticket reviews and unit tests for verdict code; the final whole-branch review; C13; a claim
  review of the report.
- Stop points: none between approval and the reports. Time-box: report by 2026-11-07 12:00, at the latest 2026-11-08.
- Budget: GPU about 2.5 to 3 h on the local GPU under the queue: extraction about 1.5 h, scoring about 1 h, C13 about 15 minutes. CPU about 1 h for intervals. New stores about 2.5 GB, run outputs about 2 GB, both on `/data/SSD2`.
Body length: about 217 lines against a target of about 150 plus the rule; the rule is inline, and §4 carries 18 rows.
