# Review of the round 9 decision rule (fresh reviewer, C7)

> Rule: `src/test/20261202_r9_scope/DECISION_RULE.md`, uncommitted, branch `r9-scope`, SHA-256 52629bd2a7574a9f28cfe61e4b4d3abd4ee76887b04a30ff40541ab028b1bdb4
> Reviewed 2026-10-13 15:49 (Amsterdam) by a fresh Opus subagent that did not write it. Read-only; nothing in the rule changed.
> Scripts and outputs: `/tmp/claude-0/-project-CoSiR/9be763ef-3a11-43a1-9938-43dfc67d003a/scratchpad/review/` (`rv_stats.py`, `rv_a28.py`, `rv_optB.py`, each with its `.out`)

**Verdict.** The rule needs two blocker fixes before it is committed. Six major and twelve minor fixes should also go in.
- **Correct as written:** the statistics, the Holm bounds, the shares, the seeds and the value sets. I re-derived each
  one, and each matches.
- **The two blockers:** the inputs of the ArtELingo-28 reads have no pins, and nothing shows that a new backbone's AFF
  was really re-fitted from that backbone.
- **The six majors:** stage 0b's case logic, a missing C12 exception, re-derivation of the new picks, head fits on DAS6,
  disclosure of the earlier label reads, and the set-aside procedure.

## Blockers (fix before commit)

**1. The ArtELingo-28 reads and CLIP's read have no pins file, so the H8 ledger row and refusal are undefined (§3.4, §6.1, §7.1, §7.3).**
- What is wrong:
  - `pins_<backbone>.json` exists only for the new backbones (line 125). CLIP has only "round 6's pinned files" (line 41).
    Yet the H8 ledger row must record "the pins file's SHA-256" (line 283), and the runner refuses on "another ... pins
    SHA-256" (line 291).
  - No file pins the ArtELingo-28 inputs: the translation output (`translations.parquet`, `coverage.json`), each
    backbone's ArtELingo-28 image and caption stores, the read-set list with its mapped labels, `value_sets_a28.json`
    (line 181, written to the gitignored `results/`) and the language flags (line 118, "written before any read", no
    file named).
  - H9 and H10 re-assert H8's episode hashes. That guards rows and labels, but not features or translations.
- Evidence: rule lines 41, 125-129, 118, 181, 253, 283, 291.
- Fix:
  - Add to §3.3: "Stage 1 ends by writing `pins_a28.json`. It holds the SHA-256 of `translations.parquet` and
    `coverage.json`, of each backbone's ArtELingo-28 image and caption stores, of `value_sets_a28.json`, of
    `language_flags_a28.json`, and of the read-set file (746 painting names, row IDs, mapped labels). It is committed
    before H8's ledger row."
  - Add `pins_clip.json`, which lists round 6's pinned inputs (R6 rule §6 item 1) with their SHA-256s.
  - In §6 item 1, §7.1 and §7.3, H8 to H10 assert and record both their backbone's pins file and `pins_a28.json`.

**2. No gate shows that a new backbone's AFF was re-fitted from that backbone (§3.2, §3.4, §6 items 2 and 3).**
- What is wrong:
  - Every gate is a self-consistency check. On CLIP, stage 0b cannot tell a part re-fitted from CLIP features from the
    same part loaded from CLIP's pinned file, because both give identical arrays.
  - On Qwen or SigLIP 2, a part that the generic chain still loads from a CLIP path would pass stage 2, §6 item 2
    (refit against its own pin) and §6 item 3 (seed-42 numbers against its own pin).
  - Label-shaped parts do not depend on feature width, so no shape error would catch them: the image and caption
    k-means labels, the content graph, bank AIC and the practice banks.
  - The chain has such fixed CLIP paths:
    - `rb_build.py:65-78` pins CLIP's `partitions.npz` image and caption keys by SHA;
    - `run_n6.py:31` has a PARTITIONS constant;
    - `run_checks.model_inputs` uses a fixed A3 folder and `A3_SHA`.
- Evidence: `.scratch/r9-scope/notes/aff_chain_map.md` lines 126, 216, 220. The new-backbone claim (§4.3, "re-fitted by
  its fixed recipe on frozen Qwen features") rests on this.
- Fix: add to §3.4: "`pins_<backbone>.json` records the following for every file of §11's re-fitted list: its SHA-256,
  the SHA-256 of the feature store it was fitted from, and the SHA-256s of the recipe scripts that made it. Three
  assertions follow:
  - every re-fitted array differs from CLIP's pinned counterpart (compared as arrays);
  - every part of §11's frozen list equals CLIP's;
  - the recipe script SHA-256s equal those of the run that passed 0b. A changed recipe script reruns 0b."

## Major (should fix)

**3. Stage 0b's case logic is not decidable (§3.2 item 2).** Four places can be read in two ways.
- (a) "L1 is bit-equal to the pinned checkpoint" cannot be a file-hash match. The replay's config gains
  `aspect_tau_fixed: False` (line 81), so the stored and replayed files always differ.
- (b) Case 2 says "the tolerance of item 3 applies" but not whether to L1, to L2 or to both.
- (c) "C1 passes the same gates" has no mapping. C1 runs on another GPU, so it will not be bit-equal, and case 2 needs
  two runs.
- (d) Line 87 says A3 feeds the R1 cells and B′(A1). Yet R1's cells (116/119 and 58/123) and B1's picks (16/16 and
  16/8, `picks_seed42.json`) are neither in the "exact" list nor under the tolerance.

Fix:
- "Bit-equal: every state-dict tensor is equal (`torch.equal`, same dtype and shape), and the config is compared as
  item 2 says."
- "Case 2: L1 and L2 each pass item 3."
- "C1 passes the identity gates and item 3's tolerance against the pinned files."
- "R1 cells and B1 picks: exact." If they are meant to be reported only, say "reported, not gated" instead.

**4. Case 2's tolerance is a C12 exception that §12 does not list.**
- C12 says that a reused pipeline reproduces its recorded numbers exactly. In case 2 the replayed training pipeline
  reproduces them only within 0.05 R@1 points. D32 authorises this, but the constitution asks for the exception, its
  reason and the rejected option.
- Fix, a new §12 row: "C12 | stage 0b case 2: picks exact, seed-42 points within 0.05 | A3's GPU non-determinism, shown
  by L1 ≠ L2 (D32) | an exact replay, which would stop on GPU noise".
- Also correct the reason on line 91. The narrowest seed-42 half-width is P3's, 0.212, not P4's 0.239 (P5 0.220). The
  0.05 bound is about one quarter of it, not one fifth (`rv_stats.out`).

**5. The re-derivation (C13) leaves out the new backbones' picks and the switched-off decision (§7.5).**
- Round 6's phase 1 re-derived only the B, B0 and B1 picks, because AFF's and CF's cells were round 3's constants and
  round 3 had already re-derived them.
- On Qwen and SigLIP 2 these are new numbers: τ, σ*, the AFF and CF cells per half, the B and B0 picks and RCA's λ.
  Each feeds every P1 to P7 check and the switched-off stop. §7.5 does not list them.
- Fix: "Phase 1 of H6, H7, H9 and H10 re-derives from the pinned files (reader margins, posteriors, A3 codes):
  - τ, σ* and the B and B0 picks;
  - AFF's min-margin cells and CF's max-R@1 cells per tune half;
  - RCA's λ;
  - the switched-off outcome.

  Agreement is as R3 rule §8."

**6. Head fits made on DAS6 would stop the local reads (§3.4, §6 item 2, §7.7).**
- The read refits the heads locally and asserts that they equal the pinned posteriors bit for bit (R6 rule §5 item 5,
  through this rule's §5 item 4 and §6 item 2). Line 307 lets re-fits run on DAS6.
- Round 6 needed `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8` for the coefficient hashes to match
  (`aff_chain_map.md` line 90). A DAS6 CPU and its BLAS would change the LBFGS bits, and the read would stop.
- Fix: "Every head that a read refits is pinned from a local fit at 8 threads. A head fitted on DAS6 is refitted
  locally before it is pinned."

**7. The rule says ArtELingo-28 labels are not read, but its own design read them (Status line 24, §1, §3.3 line 99).**
- Line 99 says "No label is read". Yet stage 1 maps labels (lines 110-112) and checks the read set (lines 107-109), and
  §5 item 2 computes V from labels before the read.
- Before this rule, its writers built episodes from ArtELingo-28 labels (`scratchpad/rule/a28_*.py`) to choose option
  B; §5.1 items 7 and 8 report those builds. D31 allowed work before the rule only because it "reads no label".
- These reads touched labels only, with no feature and no score, so they are not tuning. C5 and C8 still require them
  to be disclosed.
- Fix:
  - Replace line 99 with: "Extraction and translation (D31) read no label. The label mapping, the read-set checks and
    the value sets read ArtELingo-28 labels only, never a feature or a score."
  - Add to §1: "Before this rule, its writers read ArtELingo-28's labels (no feature, no score) to set the value sets
    and to choose the episode design (§5.1 items 7 and 8). The round 9 recipe and episode design were made after H5."

**8. The set-aside procedure can give two different episode sets (§5.1 item 6; §10 line 337).**
- (a) The rule does not say whether to set aside the first failure or every failure of a build. After each re-split the
  shares change, so the result depends on that order.
- (b) The values-subset guard raises `AssertionError` (`r6_episodes.py:83-86`, `_require`'s default). §10 says every
  failed assertion stops the step, but item 6 says a fired guard sets the language aside.
- The runner and the C13 re-derivation (§7.5) must agree, so this can change a verdict.
- Fix: "A pass builds every remaining language of the pair and seed in ℓ order.
  - A language is set aside if its call raises the values-subset guard or the builder's failed-draw `RuntimeError`.
    These two are caught only here; any other exception, and any validation failure, stops the step.
  - After a pass with set-asides, re-split by item 2 over the remaining languages and start a new pass. Stop when a
    pass has no set-aside."

## Minor

9. **The language flags (§3.3, lines 115-118) cannot be read "from coverage.json on read rows".**
   - `coverage.json` covers all 167,230 non-English rows (round log), not only the read rows.
   - Its lengths are in characters (`translate_artelingo28.py:224`), and it has no word counts and no English median.
   - It does not say whether "failed" is the union or the sum of the three counts.
   - Fix: name the source (`coverage.json` counts over all of a language's rows, or a recount on read rows from
     `translations.parquet`), define the word count as `len(str.split())` on outputs and on the English CSV rows of
     the 746 paintings, and define failed as the sum of the three counts over rows. Keep coverage's ">50% non-Latin
     letters" measure (`letters()`, lines 45-48) for the languages whose source is non-Latin, and write
     `language_flags_a28.json`.
10. **The episode index is built after the empty-output drops (§5.1 item 3, with §5 item 1).** A dropped row's emotion
    then no longer counts for `lacks`, so its painting can become a wrong negative. Round 6 built the index over all
    rows. Fix: build `PaintingValueIndex` over language ℓ's read rows before the drops, and form the pool after them.
11. **The claim wording breaks after a set-aside (§4.3; line 118 "Every language stays").** With a language set aside,
    "27 other languages" is false. Fix: "captions from the N languages that entered the read", with the set-aside and
    flagged languages named beside each claim. Line 118 gets "except a language set aside by §5.1 item 6".
12. **The 498 new paintings have no near-duplicate check (§3.3).** They are matched by name only. Round 6 disclosed 24
    held rows with a training image at CLIP cosine ≥ 0.99 (R6 rule §10.4). Fix: count the read paintings whose zip
    image has a train-side image at CLIP cosine ≥ 0.99 (and ≥ 0.98), and disclose the count. Exclude none (D7).
13. **The smoke builder seeds are not in the ledger (§5.1 item 4).** The seeds 9001001 to 9003027 are not entered,
    and C4 names smoke seeds 9001 to 9003. Fix: enter them in the smoke row of `episode_seed_ledger.md`, and name them
    in §12's C4 row.
14. **It is unclear whether H8 needs stage 0b (§6 item 1 against §3.2 item 4).** §6 item 1 requires 0b to pass before
    every read, while §3.2 item 4 says CLIP's reads never use the replay. Fix: say whether a failed 0b blocks H8.
15. **The array checks are not fully defined (§3.4, §6 item 3, §3.1).**
    - "The SHA-256 of the per-anchor arrays" names neither the keys nor the bytes.
    - Fix: the same keys as stage 0a's 96 array items where they exist, hashed as C-contiguous bytes with dtype and
      shape recorded.
    - Stage 0a: say "shape, dtype and `np.array_equal`, as round 6's `array_item`". Add that every field of
      `seed42_pass_counts.json` must be equal.
16. **"As R6 rule §6" brings in steps that do not apply.**
    - Round 6's arm-L head identity (§6 item 2) does not exist on a new backbone.
    - Round 6's describe-then-score stop (§6 item 6, §7) and its runner's stop-file refusal do not apply.
    - Fix: one line that excludes them.
17. **The ArtELingo-28 smoke never runs the ArtELingo-28 loader (§6 item 5).** The smoke uses selection rows with a
    synthetic language column, so the ArtELingo-28 loader and bundle first run in H8.
    - Fix: the smoke also builds the ArtELingo-28 bundle inputs on read rows: features, A3 codes, posteriors and NaN
      masks, with no label, no episode and no score.
    - It also names the value set the synthetic smoke uses.
18. **The time-box can be read two ways (§10 line 348).** "No read started by 2026-10-18 → no read starts" can mean
    "if none has started". Fix: "A read not started by 2026-10-18 (Amsterdam) does not start; the user decides" (the
    spec's §9 says "all reads by 2026-10-18"; D14 forbids a cap that drops anything).
19. **Stage 1 leaves out SemArt (§3.3).** The spec's §2 and §3.5 include SemArt's SigLIP 2 features in stage 1. Also,
    C5 lists "SemArt" among the held splits. Fix: add the features with a counts and finite-values gate, and state
    that SemArt's official val split is development data (D23).
20. **The text of §5.1 hides how thin some languages are.**
    - The intro says "median 4 per painting". Per language, the median is 1 row per painting for Emakhuwa and
      IsiNdebele and 2 for IsiXhosa, Kazakh and Urdu (`rv_a28.out`).
    - Some languages hold a V value on very few paintings, for example Kyrgyz excitement on 2 and Kazakh still_life on
      3. The spec's "≥ 30 paintings" (§3.4) then holds only for the pool of all languages together.
    - Item 7's header says seed 9001, while its body uses 9001 to 9003.
    - Fix: state these ranges and the pooled-only eligibility in §5.1 and in the disclosure.

## If the user picks option A

Under option A, eight parts of the rule outside §5.1 would also change:

| Rule part | Change under option A |
|---|---|
| §6 item 5 (lines 264-265) | the synthetic language column goes; the smoke runs the plain builder |
| §7.5 (line 303) | "its own §5.1 split, seeds and assembly" becomes its own builder call (`values=V`, `min_paintings` 30) |
| §8 item 5 (lines 316-318) | "per language, each from its own episodes" and "the pooled read without the flagged languages" are undefined for mixed-language episodes, so they need a definition (for example, episodes whose members all come from non-flagged languages); the set-aside item goes |
| §10 (line 345) | the "more than two languages set aside" row goes |
| §12 (line 368) | the second C4 row goes; the seeds are plain 55 to 57 and need no exception |
| §13 (line 382) | `Pool_ℓ`, `n_ℓ` and `R_ℓ` go |
| §1 and §4.3 | §1 discloses the skew: in the emotion pairs, every anchor's emotion is anger, disgust, fear or sadness. Every ArtELingo-28 claim then carries that skew |
| Findings 8, 10 and 13 | lapse; findings 9 and 11 still apply |

## Checked and found right

- **Holm bounds.** For seven checks, n* = 16, 19, 24, 30, 40, 61, 124. For level 1, 61 and 124; for level 2, 24, 30,
  40, 61, 124. Each was verified exactly against p = (n + 1)/5,001 ≤ 0.025/(m + 1 − k) (`rv_stats.out`).
- **Detectable-margin multipliers.** 3.5317, 3.0830 and 3.4175, and 2.8016 for x₉₅.
- **Serial gatekeeping.** Level 2 runs only after both level-1 checks pass, so family-wise error stays at 0.025 per
  backbone in the strong sense. Both levels share one bootstrap, clusters are anchor paintings, and P1 to P7 are round
  6's.
- **0b tolerance arithmetic.** 0.05 R@1 points = 24.6 rankings of 49,152.
- **Cited files.**
  - Every cited SHA-256 matches: both earlier rules, `regression_seed42.json`, `seed42_pass_counts.json`,
    `picks_seed42.json`, `graph.npz` and `bank_AIC.npz`.
  - The regression has 131 items: 39 + 57 + 35.
  - CLIP's picks match: AFF 39/119, CF 149/10, B and B0 as `picks_seed42.json`, σ* = 0 and 0 (R3 rule line 327).
- **Read set** (my own recount from the CSV, `rv_a28.out`):
  - 746 paintings = 248 held + 498 new, and the 912 train-side paintings include our val split;
  - 74,569 rows in 27 languages;
  - wikiart genre equals the CSV genre on every row;
  - V is emotion 8, style 7 and genre 8 with the stated codes, and every eligible value is a development value.
- **Shares and language order.** Exact-integer largest remainder gives the rule's 27 shares (35 to 215, sum 4,096,
  13 left over). The ℓ order is Python's sort, capitals first.
- **Seeds.** The 81 read seeds and 81 smoke seeds are distinct, and none is in `episode_seed_ledger.md`.
  `default_rng([55, 0])` equals `default_rng(55)`, so the rule rightly avoids tuple seeds.
- **Option B feasibility** (round 6's builder, smoke seeds only, labels only; `rv_optB.out`). All 27 languages reach
  4,096 per pair on 9001 to 9003, and every episode stays within one language. The anchor paintings on 9001 are 498,
  711 and 489, as the rule states. I did not build seeds 55 to 57.
- **Switched-off stop.** It is decidable from the cell numbers. A gate that is closed on seed 42 lands in it: a cell
  needs γ > 0 to beat cell 0, and ties go to the lowest cell.
- **The spec's bars and the user's decisions.**
  - The bars are carried: SC1 to SC3, D5, D8, D16, D17, D25, D27 and D30.
  - Order D9 is enforced by the refusal on H6's and H7's verdicts; H9 and H10 wait for H8's episodes.
  - The ledger row comes first, and the pins come before the ledger row.
  - Nothing is scored on held rows or on ArtELingo-28 before its read.

## Fixes applied

Applied 2026-10-13 by a separate Opus subagent; the rule's `Reviewer:` field records it. Option B stays in §5.1; the
HTML comment there now lists the places outside §5.1 that change under option A.

1. Fixed. §2 (`pins_clip.json`), §3.3 (`pins_a28.json`, `read_set_a28.parquet`, `language_flags_a28.json`), §5 item 2, §6 item 1, §7.1, §7.3.
2. Fixed. §3.4 "Provenance": feature-store and recipe-script SHA-256s, the three assertions (the "differs" test is per array the fit makes, so copied label-only arrays such as row draws do not fail it); §11 adds the CSD style grouping to the frozen list.
3. Fixed. §3.2 item 2 (bit-equal by `torch.equal`; case 2: L1 and L2 each pass item 3; C1: identity gates and item 3); §3.2 item 3 (R1 cells and B1 picks exact, not "reported only", to keep the C12 exception narrow).
4. Fixed. §12 new C12 row; §3.2 item 3 reason corrected to P3, 0.212, about 1/4.
5. Fixed. §7.5 phase 1 for H6, H7, H9, H10.
6. Fixed. §3.4 "Heads"; §7.7.
7. Fixed. §3.3 opening; §1 disclosure.
8. Fixed. §5.1 item 6 (pass procedure, the two caught errors); §10 assertion row names the exception.
9. Fixed. §3.3 "Language flags" (sources, sum of three counts, `len(str.split())`, `letters()`, `language_flags_a28.json`).
10. Fixed. §5.1 item 3 (index over R_ℓ before the drops, rows after), §5.1 item 2, §5 item 1.
11. Fixed. §4.3 (N languages; set-aside languages named), §3.3 flags line.
12. Fixed. §3.3 "Near-duplicates" (counted at 0.99 and 0.98, disclosed, none excluded).
13. Fixed. §5.1 item 4; §12 C4 row.
14. Fixed. §3.2 item 4, §6 item 1, §10: a failed 0b does not block H8.
15. Fixed. §3.1 gate (`array_item`, every field of `seed42_pass_counts.json`); §3.4 array hashes; §6 item 3.
16. Fixed. §6 opening (arm-L identity CLIP only; describe-then-score stop and its refusals excluded).
17. Fixed. §6 item 5 (bundle inputs on read rows via the ArtELingo-28 loader; values V for the synthetic smoke).
18. Fixed. §10 time-box row.
19. Fixed. §3.3 "Stores" (SemArt SigLIP 2 features, counts and finite values; val split is development data).
20. Fixed. §5.1 intro (median ranges, thin values, pooled-only eligibility), item 7 header; §8 item 5; §1 disclosure.

## Re-review of the fixes

Re-reviewed 2026-10-13 16:02 (Amsterdam), scoped, read-only. The rule is unchanged by this pass. I checked each fix against the rule
text (449 lines), the code map, `r6_episodes.py`, `run_step1.py`, `r6_heads.py`, `translate_artelingo28.py` and
the stored `seed42_pass_counts.json`.

**Verdict.** 19 of 20 fixes are in and do what the finding asked. One fix (15) added a gate that cannot pass as written
(N1, major). There is no new blocker. Section numbers all resolve.

### The 20 findings

1. ok. §2, §3.3 "Pins", §6 item 1, §7.1 and §7.3 cover `pins_clip.json` and `pins_a28.json` (the stores of all three backbones, flags, read set, V).
2. ok. §3.4 "Provenance" has the three assertions. Two gaps are minor (N4).
3. ok. Bit-equal is `torch.equal`; case 2 is L1 and L2 each against item 3; C1 has its gates; R1 cells and B1 picks are exact. The case split is complete (L1 equal, L1 ≠ L2, L1 = L2 ≠ pinned).
4. ok for case 2. The P3 figure (0.2124) and the 1/4 reading are right. The C1 tolerance is not in the C12 row (N2).
5. ok. §7.5 lists τ, σ*, B and B0 picks, AFF and CF cells per half, RCA's λ and the switched-off outcome.
6. ok. §3.4 "Heads" and §7.7 pin the heads from a local 8-thread fit. Two follow-ups are minor (N5).
7. ok. §3.3 opening and §1 disclose the label reads. Two wordings still clash with it (N3, minor).
8. ok. §5.1 item 6 gives one procedure: pass in ℓ order, two named errors caught, re-split, repeat. I checked the code: the subset guard is an `AssertionError` tagged `guard:values_subset`, and the failed-draw `RuntimeError` has its own message. The other `RuntimeError` ("need more than NUM_PAIRS values") cannot fire after the subset guard, so it correctly stops the step. §10 names the exception.
9. ok. Sources, the sum of the three counts, `len(str.split())`, `letters()` and the flags file are named. Two small gaps (N6).
10. ok. §5.1 item 3 builds the index over R_ℓ before the drops and the pool after them. §5 item 1 and item 2 agree.
11. ok. §4.3 and §3.3 use "N languages"; set-aside languages are named. N is not defined per pair and seed (N7, minor).
12. ok. §3.3 "Near-duplicates" counts at 0.99 and 0.98 and excludes none. "Train-side" is not defined (N7, minor).
13. ok. §5.1 item 4 and the §12 C4 row list the smoke seeds 9001001 to 9003027; the 81 read seeds are 55001 to 57027.
14. ok. §3.2 item 4, §6 item 1 and §10 agree: a failed 0b does not block H8.
15. NOT OK. See N1. The array-hash part (§3.4, §6 item 3) is fine.
16. ok. §6 opening excludes arm-L identity (CLIP only) and the describe-then-score stop.
17. ok. §6 item 5 adds the bundle inputs on read rows with the ArtELingo-28 loader, with no label, episode or score.
18. ok. §10 time-box row reads one way and keeps D14 (the user decides).
19. ok. §3.3 "Stores" adds SemArt with counts and finite values; its val split is development data.
20. ok. §5.1 intro gives the median range, the thin values and pooled-only eligibility; item 7's header is fixed; §8 item 5 reports it.

### The three confirmations

a. **CSD grouping: confirmed frozen, heads re-fitted.** `run_step1.py` builds `style_csd` by Leiden on CSD
   style embeddings (768 d, `STYLE_DIR`), merges small communities in CSD space, and `style_rand` is a permutation of it.
   No CLIP, Qwen or SigLIP feature enters the labels. Its heads (`style_csd__img/txt`) are fitted on the backbone's features
   (R6 rule §6 item 2: "on CLIP B/32 features ... not the CSD-feature head"). The rule says so only through §3.2
   item 1 ("every head ... `step1_heads_style`") and §11 "all heads". That is enough but implicit. Optional: add
   "(the style heads included: CSD labels, the backbone's features)" to §11.
b. **Local heads versus D12 and D33: a narrowing, not a contradiction.** D12 says reads run locally and the cluster is
   "for re-fits"; it does not say every part of a re-fit runs there. D12's own reason is "bit-exactness is local". D33
   plans "the two re-fits" on the nodes; the rule moves the head fits of those re-fits to the local machine. No user
   decision is contradicted. The controller should tell the user, because it shrinks the node plan.
c. **Language flags: consistent and decidable, with two small gaps (N6).** Failed and script measures use
   `coverage.json` over all of a language's translated rows (fields `empty_outputs`, `outputs_equal_to_source`,
   `hit_max_new_tokens`, `outputs_still_in_source_script`, `rows` all exist in `translate_artelingo28.py`). Only the
   word-count flag is recounted on read rows. The rule says both in one bullet and does not contradict itself.

### New findings

**N1 (major). Stage 0a's "every field of `seed42_pass_counts.json` equal" cannot hold (§3.1 Gate).** The file also
holds `rule_sha256`, `runner_sha256`, `module_sha256`, `time` and `input_sha256`. A new chain has another runner,
other modules, another rule and another time, so 0a would always fail, or each implementer would drop different fields.
Fix: "Every field of `seed42_pass_counts.json` equal except `rule_sha256`, `runner_sha256`, `module_sha256`, `time`
and `input_sha256` (they name the code and the clock). `coef_sha256`, `episodes_sha256`, `checks`, `secondary`,
`holm_order`, `mode`, `seeds`, `n_episodes` and `n_clusters` must be equal."

**N2 (minor). The C1 tolerance is a C12 exception that §12 does not list (§3.2 item 2, last bullet; §12).** C1 is
held to item 3's 0.05 tolerance, but for another reason (another GPU), and the C12 row names only case 2. Fix: in the
C12 row write "stage 0b case 2 and the cluster run C1" and in the reason add "or another GPU (C1)".

**N3 (minor). Two wordings clash with the label-read disclosure (§1 first line, Status).** §1 says "Written before any
number of this round", and Status says nothing is "picked" on ArtELingo-28. Yet §5.1 items 7 and 8 hold numbers, and
the episode design was chosen after the label reads. Fix: "Written before any score of this round" and, in Status,
"Nothing is fitted, picked or tuned on held rows or on ArtELingo-28, except the label-only reads disclosed in §1."

**N4 (minor). Provenance (§3.4) has three loose ends.**
- "Every part of §11's frozen list equals CLIP's" includes items that are not files ("AFF's method", "CLIP's whole AFF
  (H8)"). Fix: "every frozen file of §11 (split and leakage groups, halves, row draws, affect partition and grouping, CSD
  grouping)".
- "Each array its fit makes" is not listed. Fix: the pins file lists, per file, the keys tested as made and as copied.
- Stage 0b must record the recipe-script SHA-256s that item 3 of the assertions compares. Fix: add to §3.2 item 4's
  preceding text, "0b records the SHA-256s of the recipe scripts it ran".

**N5 (minor). Local heads (§3.4 "Heads", §7.7) leave two gaps.**
- Stage 0b's head replay (§3.2 item 1) needs the same local 8-thread run, or the bit-equal test fails on thread noise.
  Fix: "the head replay of item 1 runs locally at `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8`".
- Parts made from head posteriors (reader and heads `rb_*`, practice banks, τ, the seed-42 picks) must come from the
  locally pinned heads. Else a last-bit difference stops the read at §6 item 2 or 3. Fix: add to "Heads": "Everything
  that consumes a head's posteriors is made after that head is pinned."

**N6 (minor). Language flags (§3.3).**
- "Its NLLB source script is not Latin" needs a test. Fix: "the script suffix of the `code` field in `coverage.json`
  is not `Latn`".
- Say whether the word-count median is taken before or after the empty-output drops. Fix: "after the drops".
- "Except one set aside" should read "except a language set aside" (up to two can be).

**N7 (minor). Two terms are undefined.**
- N in §4.3 differs by pair and seed when set-asides differ (the subset guard depends on the pair, the failed-draw
  error on the seed). Fix: "N = the languages that entered all nine builds; set-aside languages are named with their
  pair and seed".
- §3.3 "Near-duplicates": "train-side" is undefined. Fix: "train-side = our scorer-train and selection paintings".

## Fixes after the re-review

Applied 2026-10-13 16:03 (Amsterdam) to `DECISION_RULE.md`. Nothing deleted.

- N1: fixed. §3.1 Gate. The five run-specific fields are excluded; the nine others must be equal (field names read from the stored file).
- N2: fixed. §12 C12 row (now "stage 0b case 2 and the cluster run C1", reason adds "or another GPU (C1)") and §3.2 item 3 (C1 held to the tolerance).
- N3: fixed. §1 first line ("any score") and the Status paragraph (label-only reads excepted).
- N4: fixed. §3.4 Provenance (frozen-file wording, per-file keys listed) and §3.2 item 3 (0b records recipe-script SHA-256s).
- N5: fixed. §3.2 item 1 (head replay runs locally at 8 threads) and §3.4 Heads (consumers made after the head is pinned).
- N6: fixed. §3.3 Language flags: `Latn` script test, median after the drops, "a language set aside (up to two can be)".
- N7: fixed. §4.3 defines N (languages that entered all nine builds); §3.3 Near-duplicates defines train-side.
- 15: fixed through N1. §3.1 Gate; the array-hash part (§3.4, §6 item 3) was already ok.
- Optional (confirmation a): done. §11 re-fitted list now says "the style heads included: CSD labels, the backbone's features".
- Also: §5.1 comment rewritten (option B taken as the agent default at 15:59; the user may switch to A by amendment before H8's ledger row); header Reviewer field extended.
