# Rule review: text claim spec (docs/superpowers/specs/2026-11-06-text-claim-design.md), 2026-11-06

**Verdict: pass after fixes** (7 must-fix, 12 should-fix, 7 nit). The design is sound and its counts check out. Two
independent coders would still compute different intervals, late fusion read-outs and pair cells, and C13 as written does
not re-derive the scores. One number behind D9 reads wrong for two of MMArt's three aspects. None of the fixes reopens a
user decision; finding 5 asks the user to look at a fact again. The pair bar's validation numbers came in after the hand-back
and are in the last section.

Checked and correct (no action):
- Counts: 12,281 held and 6,152 validation paintings, no overlap; 61,744 held and 30,872 validation captions, all
  distinct ids; 9,949 held and 4,960 validation paintings with genre; 27 style and 9 emotion classes present on held. Plain Qwen,
  CLIP L/14, SigLIP 2 and SigLIP text stores hold 308,723 rows, and each holds exactly the 61,744 held caption ids. Images
  cover all held and validation paintings. §5's assumption can read "Checked by id".
- `emotion_focus_held`, `style_word_held` and `genre_word_held` hold 12,281 images. The three `_s2img` stores hold 29,232.
  `emotion_focus_s2txt` holds all 30,872 validation captions and no held caption, so held teacher text is new extraction, as §3.7 says.
- MMArt test: `mt_data.keys('test')` gives 16,384 keys in sha256 order. All 16,384 are in the raw text file, with no null
  or empty text in any of the five kinds. Default and all three teacher image stores cover them (16,782 rows, keyed).
  Labels on test: emotion 16,218, style 16,384, genre 13,420. Validation: 5,720, 5,769, 4,679. No class is empty and
  no class has one row, so no MMArt query loses its cell for lack of a relevant row.
- Weights: all 15 files load with both sides. r15 exports exist in `exports/{val,new}/qwen/cstar` for 15 files, keyed, 16,782 rows on `new`.
- Gate (ii) can pass: on 400 random rows and all 15 files, the image side gives max absolute error 5.9e-5 on r15's `new`
  exports, 6.0e-5 on held exports and 6.1e-5 on validation exports. The bar is 1e-3.
- Gate (i) unit norm can pass: float16 stores deviate by at most 2.3e-4 on 2,000 rows.
- Gate (iii) tolerance can pass: r22's `evaluate` on CPU float32 differs from its stored GPU value by at most 8.9e-8 on
  plain text to image, and a float64 stable-sort run by 2.0e-8. So 1e-5, and C13's 1e-6, are not numerically out of reach.
- Ties: only 46 of 61,744 held captions and 12 of 30,872 validation captions are exact repeats, so ranking ties do not matter.
- Facts in §1, §3.1 and the brief: the r22 margins, 6 of 9, 5 of 9, late fusion's 0.3215, 0.4954, 0.1952 against 0.3102,
  0.5068, 0.1823, the 0.0018 margin, MMArt's 0.2180, 0.3085, 0.5080 against 0.1991, 0.2166, 0.4726 (r22 picked these at
  centering from raw and centering only, seed 0), and text to image as the only MMArt direction in `mmart.json`.
- Extraction: rates 59.9 to 122.2 per second match §5. My count of 414,215 MMArt rows plus 185,232 held rows gives about
  1.5 h, as §9 says.
- Verdicts: the c* and late fusion rules are exhaustive and exclusive, and §3.4, §6, D3, D4, D5 and the brief agree on them. All four prior rows sum to 1.
- Date: 2026-11-07 is a Saturday. Sequence date 2027-01-11 is free.
- Held data: no pick, range or weight reads held rows. Stage 2 selects nothing. Gate (ii) reads held image rows but
  scores nothing, so no stage 2 number can reach stage 1 if stage 1 is committed first.
- Hard rules 1, 2, 3 and 7 hold. Parts table: every part is present; findings 6 and 13 are about content.

## Must-fix

1. **§3.4 Interval (:139-142) and D15: two coders get different intervals, so C13 fails at 1e-6.**
   - Stream: r11 (`rt_stats.paired`) calls `rng.integers(0, n, size=n)` then `rng.choice`. r22 (`tx_metrics.painting_bootstrap`)
     calls `rng.multinomial` in chunks of 100. "The form of r11" with "paintings" fits a hybrid the spec does not write out.
   - Ratio: "divided by the query count" can mean the drawn count, with repeats (r22's `w @ counts`), or the fixed total.
   - Unit: r22 draws over all paintings, so a genre cell draws 12,281, of which 2,332 add nothing. r11 draws over the
     labelled queries, so 9,949.
   - Order: `integers` maps to paintings by position. Held is sorted by name; MMArt by sha256. The spec names neither for the draw.
   - Undefined rows: a query with no relevant gallery row has a NaN AP, and r22 drops it. The spec does not say so.
   Fix, replacing the bullet: "**Interval (D15).** Per comparison, one `default_rng(0)`, 10,000 draws. The units are the
   paintings with at least one scored query in that cell, in evaluation order (ArtELingo: sorted names; MMArt: r15's sha256
   order); call their count m. In each draw, in this order: I = `rng.integers(0, m, size=m)`; S = `rng.choice([0,1,2,3,4],
   size=5, replace=True)`, shared by both sides. With w_p the times painting p is in I, a seed's value is the sum over p of
   w_p times the sum of its query values, divided by the sum over p of w_p times its query count, so repeats count in
   both. The draw's value is the candidate's mean over the five entries of S minus the comparator's value (unseeded, no
   S). A scored query is one with a finite AP (label cells) or any query (pair cells). Point value: the mean over all scored
   queries and five seeds. Interval: `np.percentile` at 2.5 and 97.5, default linear."

2. **§3.2 (:102, :107) and §3.4 read-out menu (:129-131): late fusion read-outs and the gallery of expansion are open.**
   - "The cosine of two late fusion vectors is (1 - a) c* cosine + a plain cosine" holds at raw cosine only. r22's
     `check_joined` centers and normalises the concatenated 4,096 values as one vector. A coder who centers each part and then fuses gets other numbers, and gate (iii) would catch it only on validation seed 0.
   - "Expansion reads the unlabelled gallery ... (as r11, r15)" is false. r11's `rt_metrics.retrieval` and r15's
     `ac_stats.arm_retrieval` expand on the labelled gallery (r15 §3.3 says so). r22's `evaluate` keeps unlabelled rows in the
     ranking and masks them at AP. Genre and emotion cells differ. On held, 2,332 paintings and their captions are at stake.
   - The means' rows are open: all rows, or the aspect's labelled rows.
   Fix, replacing the read-out bullet: "**Read-out menu.** r22's `evaluate`, unchanged: raw cosine; centering; centering
   plus expansion k = 10 (the query plus its top 10 gallery rows, normalised, then ranked again; a query's own row never counts).
   Centering subtracts the mean of the row's text and the mean of its image rows, one mean for text to text, then L2-normalises.
   Means use every validation row of the set, labelled or not, with the same row, head, seed and text kind. For late
   fusion, the row is the concatenated vector: the mean and the normalisation act on it whole. Ranking and expansion use
   every gallery row of the set, labelled or not; unlabelled rows are masked only at AP, as r22 (this differs from r11 and
   r15, which drop them first). Expansion reads no label, and k is fixed." In §3.2: "At raw cosine, the cosine of two late
   fusion vectors is ...".

3. **§3.4 Pair cells and pair picks (:127, :134-136): the population and the pick order are open, and Q decides a verdict.**
   - §3.3's "unlabelled rows leave query and gallery" sits under labels, with no word that pair cells keep every row. A coder who applies it drops 2,964 MMArt test paintings.
   - "the head with the best R@10" does not say at which read-out or how to average. No tie rule exists.
   - "Table 2 of r22" is the label rule table. The pair table is Table 3 of `results/tables.md`.
   Fix: "**Pair cells.** R@10 with r22's hit rule, on every painting and every text of the set, labelled or not. The value
   of a query is 1 when its first hit rank is at most 10, else 0." And: "**Pair picks.** On the set's validation rows, per
   direction. For each late fusion head: the read-out with the best MRR, mean over seeds. Then the head with the best R@10,
   mean over seeds, at its picked read-out. Plain Qwen: the read-out with the best MRR. Ties: earlier read-out in the
   menu; heads in the order emotion, style, genre." Check against `best_head` in `tx_followup.py`, whose `max` over heads uses the same order.

4. **§3.4 C13 (:145-146) does not re-derive the scores, and it contradicts the header.**
   - The header (:8) inherits r15's C13 form, which read keyed stores and exports (`mt_c13.py`). §3.4 reads "saved per-query
     outputs", which are our outputs, "without our outputs". Ranges and picks "from ... the stage 1 record" are circular.
   - A bug in fusion, centering, expansion or AP would pass, since none of these is recomputed. Constitution C13 says every
     number that decides a verdict. AP and first rank decide the verdicts.
   - It costs little. From r22's per-ranking times, the bar cells at picked read-outs on both held sets, plus the menu on
     validation, need about 15 GPU minutes (my estimate). 1e-6 is reachable: see the 8.9e-8 above.
   Fix: "**C13.** Codex re-derives, in separate code and without our scored outputs, from this rule, the stores, the
   weights, the keys and the labels: the forward passes, late fusion vectors, means, per-query AP and first hit rank of
   every bar row; the seed ranges and picks from validation; every bar cell, its interval and outcome; Q; and all four
   verdicts. Within 1e-6, before any verdict is written. It may run on the GPU under the queue." Add about 15 minutes to §9.

5. **D9 (:198), §8 row 4 (:268) and §3.1: 15.7% is the emotion share only; style is far higher.** Measured on MMArt validation
   with r22's own word patterns, unified text:

   | Own label named | MMArt unified | ArtELingo captions |
   |---|---|---|
   | emotion | 15.7% (labelled rows) | 11.5% |
   | style | 83.2% | 0.2% |
   | genre | 37.6% (labelled rows) | 2.8% |

   So §8's "unified text names its label in 15.7% of rows" is wrong as it reads. D9's reason compares one aspect only. The
   user chose D9 on that comparison. MMArt's style cells mostly test whether a row reads the style word, which ArtELingo's
   do not. It matters for SC2 and SC4, and for the MMArt sentence. D9 is the user's and stays, but the user should see this.
   Fix:
   - D9 reason: "own-label word share in the unified text: emotion 15.7%, style 83.2%, genre 37.6%; ArtELingo captions 11.5%, 0.2%, 2.8%".
   - §8 row 4: "unified text names its emotion in 15.7% of rows, its style in 83.2%, its genre in 37.6%".
   - Brief "Check these" item, new: "MMArt's unified text names its own style in 83% of validation rows, against 0.2% for
     ArtELingo captions. Your pick of the unified text (D9) was shown the emotion share only, 15.7%."
   - §3.6: name the style and genre rows without label-naming text, with r22's patterns lower-cased to MMArt's 20 style
     names, and report their counts.
   - §6: "The MMArt sentence names the share of label-naming text in each aspect."

6. **§6 Licensed claim (:241-245): two sentences are missing and one overclaims.**
   - "c* holds:" leaves qualified without a sentence, though the same sentence with T = 3 or 4 fits.
   - "MMArt has its own sentence" is not written, and the template asks for the sentence.
   - "keeps those label gains" overclaims. "Those" points at c*'s gains, and on validation late fusion is below c* in all 9
     cells (0.2435 against 0.2650, for example). The bar only says it beats plain Qwen.
   - "stays within 0.02" reads as two-sided, and as a point value. The bar is a one-sided lower bound.
   - No sentence says read-outs were picked on validation or that five seeds were used, as r15's did.
   Fix: "**c* holds or is qualified:** 'On <n> held ArtELingo paintings, c* (five seeds, read-outs picked on validation)
   beats plain Qwen in label retrieval in all nine aspect and direction cells, and beats the teacher's text in T of them,
   with one encoder pass and a small head per aspect, where the teacher's text needs an encoder pass per aspect.' MMArt: the
   same sentence with 'on 16,384 WikiArt paintings outside ArtELingo, with generated text and MMArt's labels' and the
   label-naming shares of finding 5. **Late fusion holds or is qualified:** 'A weighted sum of c* and plain Qwen cosines
   (a = 0.7) beats plain Qwen in label retrieval in all nine cells, and beats the teacher's text in T of them. Its pair R@10 is
   not more than 0.02 below plain Qwen's in text to image and image to text, at the interval's lower bound.'"

7. **§6 Inconclusive (:234-236), §3.5 stage 0 row and Teacher cells (:238-239): the outcomes leave cases open.**
   - Gate (iv) has no entry, though §3.5 says "§6 says what it voids".
   - A failed gate (i) on a stage 3 store (MMArt's narrative text, a side row) voids a verdict it does not touch.
   - A bar cell that cannot be computed falls under "Fails: anything else", so a crash reads as a failed claim.
   - A C13 mismatch has no outcome.
   - "These stand whatever the verdict" contradicts a voided set, whose cells have no valid numbers.
   Fix, replacing the paragraph: "**Inconclusive.** A failed gate voids only what it serves, after the fix and rerun that
   this spec allows: (i) on a store a bar cell reads, or (ii), or (iv): that set's two verdicts and its cell claims; (i) on a store only §3.6 reads: that diagnostic; (iii): all four verdicts and all cell claims; a missing stage 1
   record: that set's. A bar cell that cannot be computed, or a C13 difference over 1e-6 that stays after its cause is traced,
   voids the verdict that reads it. An inconclusive verdict is never counted as fails or qualified." In Teacher cells: "These stand whatever
   the verdict of a set that is not voided."

## Should-fix

8. **§3.5 (ii) (:159-160): the caption side, the thing under test, has no forward gate.** A wrong caption-side load on seeds
   1 to 4 passes (ii), and r22 checked seed 0 only. The gate is cheap: on 500 validation paintings, the caption side on
   caption means gives the stored `cap` within 6.0e-5 on all 15 files, and the image side gives validation exports within 6.1e-5.
   Also "r11's held exports" do not exist: r11 read r10's (`cosir_r10/.../full/held/k11=40/`). Fix: "(ii) Forward, 15 weight
   files: the image side on ArtELingo validation, held and MMArt test reproduces r10's validation and held exports and r15's
   `new` exports within 1e-3; the caption side on normalised caption means reproduces the stored validation `cap` within 1e-3."

9. **§3.5 (iii) (:161-164): no replay covers the MMArt path, and the interval clause cannot be checked.** r22's replay is
   ArtELingo only, though SC2 and SC4 use other keys, labels and order. `mmart.json` has r22's numbers. "Given one seed and
   r22's painting draws, matches its form" names no number, and r22's draws are multinomial. Fix, adding: "and r22's MMArt validation
   label mAP and pair R@10 for c* and plain Qwen, seed 0, unified text, text to image, raw and centering, through r15's labels and
   order." Replace the last sentence: "On a synthetic case of 6 paintings, 2 seeds, and a hand-computed I and S, the interval code gives the
   hand value within 1e-12; on r22's per-query differences, one seed, its bounds lie within 3e-4 of r22's Table 2."

10. **§3.5 (i) (:157-158): the 200-row check has no reference for new stores.** A teacher instruction on MMArt text has no
    earlier row, so it compares a new store with itself and misses a wrong instruction. Fix: "Each store's manifest gives
    the exact instruction string. Held teacher text is compared on 200 validation rows with `emotion_focus_s2txt` and r22's `style_word`
    and `genre_word`; plain MMArt test text with r22's `mmart_val`. MMArt teacher text has no reference, so its 200 rows only test a rerun."

11. **§3.2 (:103) and §7 C1 (:251): "the strongest plain row" is false in 3 of 9 validation cells.** Style text to image: Qwen
    0.1342 against SigLIP 0.1529, CLIP L/14 0.1516, SigLIP 2 0.1495. Genre text to image: 0.4179 against SigLIP 2 0.4293.
    Genre image to text: 0.3390 against CLIP L/14 0.3569. c* is above all of them, but on MMArt no such row is planned. Fix §3.2:
    "comparator, the user's plain row; the best plain encoder in 6 of 9 validation cells". Fix §7: "plain Qwen and the
    teacher's text; other plain encoders stand beside them on ArtELingo held only; on MMArt the strongest plain row is not measured (D12)".

12. **§5 (:214-215) and §3.2: the RoBERTa side row has no held text.** `roberta_go_emotions_s2` holds 0 of the 61,744 held
    captions (30,872 of 30,872 validation). Fix §5: "RoBERTa has no held captions; extraction takes about a minute, GPU, same settings (max length 64)."

13. **§6 and parts table: stage 1 is a sweep, which requires a §6 selection rule; the claim test's outcome mapping is absent.**
    Fix, adding to §6: "**Selection rule.** Stage 1 only: §3.4's picks and pair picks, exact ties as written, no tie band.
    Stage 2 selects nothing. **Outcome words.** Holds is GO; qualified is GO for the qualified claim; fails is NO-GO (as r15, D2)."

14. **§3.1 priors (:89, :94): c*, ArtELingo fails 0.18 does not fit the rest.** Fails needs P below 9 or T at most 2. Yet
    the spec calls a miss over plain Qwen a surprise, the smallest gap is +0.0132 with an interval of about 0.0006, and style's
    3 cells hold in every check. BELOW is also undefined over plain Qwen, where only WIN counts. Fix: lower fails to about
    0.08 and raise qualified to 0.32 (the agent's numbers, before any held number), or state the route to 0.18. Write "c* not
    WIN over plain Qwen in any ArtELingo cell".

15. **§3.5 stage 0 row (:152) and §6 "fixed and rerun": C12 says a difference goes to the user with its cause traced.** "Note
    it" does not. With no stop points, the fix is: "stop that set's path, trace, fix and rerun; the report tells the user the difference and its cause".

16. **Brief "Outcomes" (:28-35): inconclusive is missing, and the pair line reads as a point value.** The template asks for
    inconclusive and what follows each outcome. Fix: "pair R@10 no lower than 0.02 below plain Qwen at the lower end of the
    interval"; add "A failed check that cannot be fixed gives inconclusive for that set, never fails."

17. **§1 (:56), §3.1 (:78) and brief (:16): sentences that word c*'s pair-hit limit, which D1 forbids.** "Pair hits stay below
    plain Qwen" states it. "Late fusion adds plain Qwen's pair signal back" implies c* lost it. Nothing tells the report how to
    handle it. Fix §1: delete the sentence, and keep late fusion's numbers. Fix §3.1: "Late fusion adds plain Qwen's cosine to
    c*'s". Fix §6, adding: "Not licensed: any sentence on what c*'s pair numbers mean. The report lists them in a table, with no
    wording, until the user has talked to the other authors." §6's "or c* alone on pair hits" stays, as a limit on what is licensed.

18. **§3.3 (:114), §3.4 picks and D10: stage 3's inputs are open.** §3.6's other MMArt texts need validation picks, but the table
    gives MMArt validation stages 0 and 1, and picks have no text kind. D10's "validation and held rows" does not name the set.
    Fix: stages "0, 1, 3"; picks "per row, text kind, aspect and direction; §3.6's text kinds are picked in stage 1 too"; D10
    "ArtELingo validation and held; MMArt test as well, at the same cost".

19. **§3.7 (:181-182): which image rows are scored is open.** Exports are float16 and a live pass float32; they differ by up
    to 6e-5, enough to move a cell past 1e-6. Fix: "Scored c* rows are live forward passes in float32 from unit float32 inputs, as r22;
    exports serve gate (ii) only. Text and image stores are unit-normalised once after the cast."

## Nit

20. §8 disclosure (:275-276): "MMArt test image outputs were read in r15 and r17" is incomplete, since the reports 2027-01-05,
    2027-01-06 and 2027-01-07 also list MMArt test. It is also a held-read record, which `held-data.md` says to keep out. Replace both sentences:
    "No text row of held ArtELingo or MMArt test has shaped any choice."
21. D9: 15.7% is on labelled rows and 73.8% on all rows. On one base: 15.6% and 73.8% (all), or 15.7% and 74.5% (labelled).
22. §9 (:286): "New stores about 1.6 GB". 599,447 rows at 4,096 bytes give about 2.5 GB. §5's "about 40 GPU minutes": r22's
    per-ranking times scale to about 1 h when every read-out, head and §3.6 text is scored (my estimate). Neither affects a verdict.
23. §7 `held-data.md` row (:257): add, as r15 did, that expansion reads held embeddings without labels, with k fixed.
24. §8 row 6 (:270): "the extraction log's truncation count": r22's `encode_store` records none, so the build must add it.
    The longest MMArt unified text is 1,310 characters; the limit is 512 tokens.
25. Hard rule 4: D7 was taken after the recommendation and is not in "Check these". D15 is a consequential agent default, also absent.
26. Line 48: the body-length note sits in an HTML comment inside the brief; r15 put it after §9.

## Added after the hand-back: the pair bar on validation

The CPU job finished after the hand-back, so the controller's report does not hold these numbers. The counts above are unchanged:
this section adds no numbered finding. Validation only, a = 0.7, five seeds, 10,000 draws in the form of finding 1. Heads and
read-outs: r22's seed 0 pick on ArtELingo (genre, centering); on MMArt, picked on seed 0 here (the rule uses the mean over seeds,
so stage 1 may pick otherwise). These are development numbers and license nothing.

| Set | Direction | Late fusion, seeds 0 to 4 | Plain Qwen | Difference [95% interval] |
|---|---|---|---|---|
| ArtELingo | text to image | 0.3163 to 0.3215 | 0.3102 | +0.0091 [+0.0060, +0.0123] |
| ArtELingo | image to text | 0.4954 to 0.5010 | 0.5068 | -0.0086 [-0.0154, -0.0016] |
| MMArt | text to image | 0.9393 to 0.9412 | 0.9393 | +0.0009 [-0.0025, +0.0044] |
| MMArt | image to text | 0.9454 to 0.9482 | 0.9497 | -0.0028 [-0.0066, +0.0008] |

- **Q can pass on numbers alone.** Every lower bound clears -0.02 on validation, and held rows narrow the painting part of the interval.
  ArtELingo image to text is the tight one, with 0.0046 to spare. Seed 0 (-0.0114) was the worst of the five seeds; the mean gap is -0.0086.
- **On MMArt, Q is weak.** Pair R@10 is about 0.94 on a gallery of 5,769, near the ceiling. A fall of 0.02 there raises misses by about a
  third, where on ArtELingo image to text it raises them by about 4%. The test gallery of 16,384 will lower R@10, by an amount not measured.
  D5's bar is the user's and stays. Proposed text for §6's MMArt sentence: "Pair R@10 on MMArt is near its ceiling, so the pair clause says
  less there than on ArtELingo; R@1 and median rank stand beside it." §3.6 already lists those numbers.
- **Finding 1, correction of one sub-point.** On these four cells the drawn and the fixed denominator gave the same lower bound to six
  decimals. So the ratio matters less than finding 1 implies, at least here. The stream and the unit stay open, and the
  spec must still name one form. Label cells were not tested.

Not done: §8's "MMArt's text was made with Qwen-family models" was not verified. No code ran beyond read-only CPU checks. No held score
was computed. The tickets (hard rule 6) do not exist yet. The job's log and arrays are in
`/tmp/claude-0/-project-CoSiR/88455088-ddbb-4074-8cbe-a09cc630ef5c/scratchpad/` (`c10.log`, `pairs.pkl`).
