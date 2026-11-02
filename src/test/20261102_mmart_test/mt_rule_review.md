# Rule review: MMArt test spec (docs/superpowers/specs/2026-11-02-mmart-test-design.md), 2026-11-02

**Verdict: pass after fixes** (4 must-fix, 8 should-fix, 9 nit). The design is sound and its facts mostly check out.
Two independent coders would still compute different read-out picks and clustering scores, and one stage 0 gate
can fail for numerical reasons alone. The four must-fix items are text changes; none reopens D1 to D5 or D15.

Checked and correct (no action):
- D7 cells: r14 `records/verdicts.json` cells 9-14, 28-33, 47-52 give WIN for emotion AMI, style AMI, style mAP and
  genre mAP on all three encoders. The Qwen cells (emotion AMI, style AMI, genre mAP) match
  direction_log.md:224-226 and :182-184. The numbers in §3.1 match.
- Matched-arm IDs: P3 (= B1, the claim test's core P3), M2, M8, M7 (`bs_artclu.py:59-69`, `rb_matched.py:33-35`).
  All 20 mapper files (`.pt`) exist: CT `stage2/full/percept/core/stage2/k40/P3__s{0-4}.pt`, 10-25
  `part2/full/matched/M2/`, 10-26 `part2/full/matched/{M7,M8}/`. Every sidecar has `device: cuda`.
- Counts in §3.3 and D8 (5,769 validation paintings; 5,720 / 5,769 / 4,679 labelled; 190 and 3,043 test paintings
  without labels; 34 exact copies; 49,121 reference paintings; 17,307 = 3 × 5,769).
- The 57 validation paintings whose file sits in another style folder have byte-identical files (md5 checked).
  "Same images" in D8 holds.
- Nested WIN/LEVEL in a D7 cell is sound. The cell has one one-sided lower bound, compared with two nested
  thresholds (0 and −r). "d > 0" implies "d > −r", so the two claims add no error beyond that bound. Both thresholds
  are fixed in advance.
- D12 (one teacher pick, Qwen's ranges) matches r11. r11 used c*'s r_a for every arm (retrieval spec :238).

## Must-fix

1. **§3.4 "Read-out picks" (spec:136-139) and §8 disclosure (:273): "per arm and aspect" is not r11's rule.**
   r11 makes one pick per arm for all three aspects. Its score is the mean over aspects of (mAP_a − mAP_a(R0)) / r_a,
   with a guard on each aspect (2026-10-29-retrieval-design.md:237-238; `rt_readout.py:209-233`, whose `_aspects`
   asserts all three aspects). r14 applied it this way: `records/stage1_siglip2.json` artret_picks, one pick
   "center+k10" for three aspects. Under the spec's wording, one coder picks per aspect and another per arm. Stage
   0(vi) can only replay r11 with the per-arm rule.
   Fix: "r11's read-out rule, menu, guard and ties on MMArt validation: one pick per arm for its three aspects, scored
   as (1/3) Σ_a (mAP_a − mAP_a(R0)) / r_X,a,mAP; guard: no aspect more than r_X,a,mAP below its R0 value." In §8:
   "up to 25 read-outs per arm". If per-aspect picks are wanted instead, write it as a stated change from r11, with
   its own tie and guard rule and a unit test in place of the 0(vi) replay. That needs the user, since D4 says "one
   rule for every new set".

2. **§3.3 (:119-120) and §3.4 (:128-135): the population behind "each aspect then drops its unlabelled paintings"
   is undefined for clustering.** The inherited protocol fits k-means on every row and masks unlabelled rows only
   when it computes AMI. The bootstrap draws rows over all n rows, then masks (`rb_score.py:52-56`,
   `draw_bootstrap(n_rows=len(labels))`; `bs_artclu.py:117-121`). Retrieval uses only labelled queries and gallery
   (`rt_metrics.py:75-76`). Read literally, the spec makes coders drop the rows before k-means, which gives
   different partitions, AMI and intervals, and breaks the "inherited, unchanged" claim.
   Fix, appended to §3.3 after the cut sentence: "Clustering (every arm, validation ranges included) fits k-means on
   every painting of the set (test: the cut set; validation: all 5,769). AMI and its bootstrap rows run over all
   rows, and rows without the aspect's label are masked after indexing (r10/r14). Retrieval queries and gallery are
   the aspect's labelled paintings (r11). Centering and whitening are fitted on all 5,769 validation paintings. Query
   expansion reads the labelled test gallery."

3. **§6 "Inconclusive" (:236-237) and §8 row 4 (:271) contradict each other, and no gate is mapped to the cells it
   voids.** §6: a failed gate on an encoder's path makes that encoder's verdict and SC4 cells inconclusive. §8: a
   mapper failure makes only that encoder's SC3 inconclusive. Read with §8, an encoder can still come out
   "qualified", since "one or more SC3 cells do not WIN" counts an inconclusive SC3 as a miss. A failed teacher view
   or 0(vi) would also void SC1 to SC3, which do not use them.
   Fix: add a scope table under §3.5 and change §8 row 4 to "else that encoder's verdict inconclusive":
   - (i) Qwen default store or near-copy file: all encoders (the cut set is undefined).
   - (i) X's default store, or (ii) X's export: X (verdict and SC4).
   - (i) a teacher store, or (v): that aspect's SC4 cells for all encoders; verdicts unaffected.
   - (iii): all encoders.
   - (iv) X's mapper after the fallback: X's verdict inconclusive (never "qualified" from a missing SC3); SC1, SC2
     and SC4 rows reported; SC4 claims stand.
   - (vi): SC2 and the SC4 mAP cells for all encoders, so every verdict is inconclusive; SC4 AMI claims stand.
   - Stage 1 gate: X's ranges or picks missing: X. The teacher's pick missing: all SC4 mAP cells.

4. **§3.5 (iv) (:160-163) can fail for numerical reasons alone, and its fallback bar is undefined.**
   - Probabilities: the mappers were trained and stored on CUDA. Comparing CPU against CUDA within 1e-5 is reasonable.
   - Readings: rebuilding the `reported_stage4` readings within 1e-9 from the CPU probabilities is not. The own
     reading is an argmax over 40 topics, and the 20-cluster reading runs k-means on L2-normalised probabilities
     (`ac_cluster.py:51-54`). A difference near 1e-6 can flip a near-tied row, which moves AMI by about 1e-5. Over 20
     runs × 4 partitions, a spurious failure is plausible, and it would void every verdict of that encoder.
   - Fallback: "within one of the matched arm's seed ranges" names no range (held or validation? AMI per aspect or
     reading?).
   - Fix, "(iv) (a)": "each mapper on CPU gives its stored held probabilities within 1e-5 (max absolute)".
   - Fix, "(iv) (b)": "the new reading code, fed the stored held probabilities, gives the stored partitions exactly
     and the `reported_stage4` readings within 1e-9".
   - Fix, "(iv) (c)": "readings from the CPU probabilities are recorded beside them, no gate".
   - Fix, fallback: "a retrained mapper's held readings (own and 20-cluster, each aspect) lie within the max−min
     over seeds 0 to 4 of the stored arm's held reading of that aspect".
   - All 20 mapper files exist (see above), so the fallback is unlikely to run.

## Should-fix

5. **§3.3 (:122-124) and D9 (:198): the order key is underspecified.** The validation stores key on painting stems.
   MMArt keys are `Style/stem.jpg`. A coder hashing stems gets another order, so different k-means runs, bootstrap
   rows and cross-fit halves, and C13 fails at 1e-6.
   Fix: "the MMArt `key` (image_id, with its style folder), sorted ascending by `hashlib.sha256(key.encode('utf-8'))
   .hexdigest()`; the same order for D10".
   D9's reason should also say that MMArt keys begin with the style folder, which is the style label. Sorted keys
   would group the label itself.

6. **§5 (:210-211), D8 (:197) and 0(v) (:164-165): the validation teacher-view facts are wrong.** The 99,999 rows
   are caption `sample_ids`. The `emotion_focus`, `style_word` and `genre_word` stores hold 5,000 images each, and
   `*_trainimg` covers 0 of the 5,769. The covering stores are `<view>_s2img` (29,232 paintings each), the ones r11
   read for validation (`rt_data.py:178`). They cover 5,769/5,769 in all three views (checked by key). The prep
   loader refuses validation teacher views (`mp_stores.py:91`). The prep extractor has no validation set
   (`mp_extract.py:60-67`), so the 0(v) fallback cannot run as written.
   Fix in §5: "Checked 2026-11-02: the ArtELingo `<view>_s2img` stores (r11's validation teacher views) hold all
   5,769 validation paintings for each view." Fix in 0(v): "validation teacher views are read from `<view>_s2img` by
   painting name; gate: 5,769 keys found, finite, unit norm within 1e-3". Drop the extraction fallback or name the
   new extractor path.

7. **§3.1 prior (:89-92) is not consistent with itself.** With P(SC1 and SC2 all WIN) = 0.85 and holds = 0.65,
   qualified = 0.85 − 0.65 = 0.20 and fails = 0.15. The spec has 0.15 and 0.2.
   Fix: "holds 0.65, qualified 0.20, fails 0.15", or set SC1/SC2 to 0.80. The brief's "We expect holds on all four
   encoders" (:37) also sits uneasily with 0.65 per encoder. Say "holds on most encoders", or give a joint prior.

8. **Brief Plan item 3 (:25-26) and §7 held-data row (:260): "scored once"** is the single-look wording that
   `held-data.md:8` bans from specs and briefs. Fix: "scored; it selects nothing".

9. **0(iii) (:157-159): the replay inputs are incomplete.** Replaying r14's SC2 and SC6 mAP needs the picked maps
   fitted on ArtELingo validation, so validation features and exports are needed too. The Qwen replays should also
   pin "values and intervals", as r14 0(vi) did.
   Fix: "fed ArtELingo validation and held features, exports and labels in r14's (r10's, r11's) order and with their
   recorded picks, … values and intervals".

10. **§3.2 matched arm (:102) and §3.4: name how the MMArt readings are made.** r14's D16 text says "from its stored
    held outputs and partitions". On MMArt they are new.
    Fix: "own = argmax of the 40 probabilities; 20 clusters = k-means (states 0 to 2) on L2-normalised probabilities,
    as r12's PercepT row (`ac_cluster.partitions`); larger reading as r14 D16, taken again in each draw."

11. **Cost (brief :40-42, §9 :284-286) is underestimated.**
    - The facts brief's 50 to 62 min covers SC scoring only and leaves out validation picks, exports, stage 0
      replays, diagnostics and C13 (mt_facts_brief.md:78-79).
    - §3.6 adds the shared held set (four encoders), the uncut test set (a near-full rescoring), the unseen-artist
      subset, class-count K, and C13's independent rescoring.
    - Realistic total: CPU about 3½ to 4½ h. It still fits the 11-04 12:00 time-box.
    - Fix: state it, or cut the uncut-set and shared-held diagnostics to point values without intervals.

12. **§3.3 cut and D10 (:116-119, :199): the order and the counts are left open.**
    - Within the test set, 7 md5 groups (15 rows) remain, so D10 drops 8. The "97" counts all of MMArt.
    - 10 further groups pair a test painting with an ArtELingo held painting. They are not cut and do not leak, but
      they overlap the §3.6 shared-held diagnostic.
    - "Near-copy file pass" (0(i)) has no definition.
    - Fix: "cut order: exact copies, then near-copies (cosine ≥ 0.85, from `mmart_nearcopy.parquet`), then D10 on the
      rest"; "the near-copy file passes when it has one row per test key, finite cosines, and a reference of 49,121
      paintings"; the D10 count is 8 paintings dropped.

## Nit

13. D1 (:190) says "confirmation loop" but D6 makes stage 2 a claim test. Write "type: D6 (stricter than the user's
    confirmation, flagged)".
14. Hard rule 4: D5 (the cut at 0.85, matched arms frozen with ArtELingo topics) was taken after the recommendation
    and is not in "Check these". Merge it into item 1, or say why it is minor.
15. §5 (:212-213): the mapper assumption can now read "Checked: all 20 `.pt` files present" (paths above).
16. §3.2 (:104) and §3.6: define "strongest PercepT entry of any encoder" on MMArt, for example "the largest of
    the four matched arms' larger readings per aspect".
17. §7 held-data row (:260): add, as in r14's §7, that the matched arm's larger reading is taken on test (only
    stricter) and that query expansion reads unlabelled test embeddings.
18. §3.3 table (:113): "ArtELingo train and validation images: near-copy reference only" conflicts with the
    validation images' role as MMArt validation. Split the row.
19. §3.6: add "removed paintings per class (cut)" (as r14 for EmoSet) and "test pairs with default-Qwen cosine
    ≥ 0.95 inside the cut set". Near-duplicates whose bytes differ stay in the gallery for every arm.
20. Brief Plan: stage 1's type is "development only"; the template asks for the type (sweep).
21. §3.1 "Similar for the other encoders" leaves the prior of their extra D7 cell (style mAP) unstated. One number
    is enough.

Not done: no code was run beyond read-only CPU checks of stores, the index and mapper files. The extraction store
under /data/SSD2/pre_extract/mmart/ was not read, so the actual near-copy and parity results were not checked. The
tickets (hard rule 6) do not exist yet.
