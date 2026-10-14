# Round 9 final review, area B1 (ArtELingo-28 data path): tickets 14, 15, 16, 25c

Reviewer B1, 2026-10-14, 01:04 to 01:20. Snapshot `/project/CoSiR-wt/r9-fr-b` at 4458634 (r9-scope). Records read
from the main checkout's `src/test/20261202_r9_scope/results/` (read only). My scripts sit beside this file, each with a
JSON output: `fr_b1_counts.py`, `fr_b1_stores.py`, `fr_b1_spot.py`, `fr_b1_neardup_wide.py`, `fr_b1_mutations.py`;
`fr_b1_loader.json` is the real loader run. The scripts do not import any `r9_*` module, except the loader run and the
mutation runner, which test the project's code. C5: I counted rows, paintings, languages and splits, and recomputed
only what the preparation step computes (labels, flags, V, joins, image-to-image cosines). I computed no score, no AFF
or baseline number and no label-by-model statistic.

## 1. Verdict

**CONFIRMED WITH FIXES.** Every number of the ArtELingo-28 data path is true. I re-derived each one from the CSV, the
translations and the stores with my own code, and each equals its record and the rule's text. The three user decisions
are applied wherever they apply. All 22 pinned files hash to their pins. The real loader and `check_pins` pass on all
three backbones.

No fix is needed before the reads, and no fix touches the 17 stage 2 closure modules. The fixes are:

- one MEDIUM: guards that the rule names but no test covers (test files only);
- five LOW findings and two nits.

The one item for the user is LOW 1. At least two of the "new" read paintings are the same artwork as a train-side
ArtELingo painting under another file name. The near-duplicate counts cannot see them. The claim's counts stay true.

## 2. What I re-derived

| Check | How | Mine | Record | Result |
|---|---|---|---|---|
| ArtELingo-28 paintings, train side, read, held, new | CSV painting names against ArtELingo's paintings (`load_artelingo`) and round 6's frozen `held_codes.npz`; train side = every other ArtELingo painting | 1,658; 912 (790 in split train + 122 val); 746; 248; 498 | `a28_counts.json`: 912, 746, 248, 498 | equal |
| "New" paintings are absent from ArtELingo | read minus held, intersected with ArtELingo's 61,402 painting names | 0 | rule: "498 new" | holds (by name; see LOW 1) |
| Read rows, languages, English rows dropped | non-English CSV rows on the 746 paintings | 74,569; 27; 2,000 | 74,569; 27; 2,000 | equal |
| Translation join | `translations.parquet` indexed by `csv_row`: one entry per read row, same painting and language | 0 missing; 0 painting and 0 language mismatches; it covers exactly the 167,230 non-English CSV rows | (asserts) | holds |
| Empty-output drops | `translation.strip() == ""` on read rows | 0 in all 27 languages | 0 | equal |
| Labels | own name lists (sorted ArtELingo emotions without `something else`, sorted styles, `GENRE_NAMES`); `other`, `something else` to -1 | all 8 columns of `read_set_a28.parquet` equal row by row; no unmapped name | parquet | equal |
| Genre, `load_wikiart_genre` vs CSV | per painting | 746 of 746 equal, 0 unknown | (assert) | holds |
| Held rows | rows on held paintings | 25,055 | 25,055 | equal |
| Style differs | CSV style vs our ArtELingo style, held paintings (one style per painting on both sides) | 6, the same paintings and styles | `a28_style_differs.json` | equal |
| V | own count of distinct paintings per value, kept rows with all three labels known, cut 30, intersected with round 6's `value_sets.json` | emotion 0 to 7; style 9, 12, 17, 20, 21, 23, 24; genre 0, 1, 2, 4, 6, 7, 8, 9; no eligible value outside the development set | `value_sets_a28.json` | equal |
| V margin | paintings per value near the cut | Baroque (style 4) 29, Northern_Renaissance 32, still_life 34 | | Baroque misses by 1 painting (it would enter if a held painting with our style "Baroque" kept our style, as the rule forbids) |
| Language flags | own four criteria from `coverage.json`, the translations and the CSV English rows (median 13 words) | flagged Emakhuwa, IsiNdebele (proxy only); every value, fired bit and median equal | `language_flags_a28.json` | equal |
| `coverage.json` | recount from `translations.parquet`: rows, empty outputs, outputs equal to source, outputs over 50% non-Latin letters (own letter test) | equal in all 27 languages (`hit_max_new_tokens` cannot be recounted from text) | coverage.json | equal |
| Flag margins | | failed share ≤ 1.55% (Emakhuwa); non-Latin script share ≤ 0.016% (Kyrgyz); length ratio 0.77 to 1.31 | | far from every threshold |
| Store joins, three backbones | `index.json` header; `texts` row_id and painting list = the kept read rows in order; each `img` points at its own painting; 746 images, one per painting, sorted names; path = `images/` + CSV `image_name`, each zip member once; float16, finite, norms 0.9997 to 1.0003; `progress.json` index SHA | all true for CLIP, Qwen, SigLIP 2 | `stage1_checks_<B>.json` | holds |
| Spot-check records | rows vs `default_rng(0)` draws; painting and row_id vs `index.json`; minimum cosine (no re-encoding) | rows equal; min image / text cosine: CLIP 1.0000 / 1.0000, Qwen 0.9989 / 0.9997, SigLIP 2 0.9998 / 0.9999 | stage1 records | holds |
| Real loader and pins | `r9_a28_data.check_pins()` and `load_a28(B)` on the real files | 22 pins pass; n 74,569, 27 languages, 746 paintings, 25,055 held rows, all kept rows finite, for each backbone | | passes |
| Pins re-hash | SHA-256 of all 22 files in `pins_a28.json` | 22 of 22 equal; all 19 items of rule §3.3 present, plus 3 `index.json` | pins_a28.json | equal |
| Shards vs merged translations | the 4 shard tables concatenated, sorted by `csv_row` | equal to the merged table | meta.json `merged_from` | holds |
| Painting join (248 held) | zip image (A28 CLIP store) vs the first-row CLIP image of each of 61,402 ArtELingo paintings | own painting nearest for 248 of 248; cosine 0.924 to 1.000, 196 below 0.99; minimum margin 0.0179; each cosine and margin equal to the record (max abs difference 0.0) | `stage1_checks_clip_b32.json` | equal |
| Near-duplicates (912) | max cosine of the 746 zip images over the 912 train-side images | ≥ 0.99: 0; ≥ 0.98: 0; ≥ 0.92: 3 (2 held, 1 new); none within 0.001 of 0.92 | `near_duplicates_a28.json`: 0, 0, 3 | equal |
| Near-duplicates (49,121) | same, over every non-held ArtELingo painting | 0; 0; 72 | (not in any record) | see LOW 1 |
| rd9_compare claim fields | code reading plus 10 mutants (§5) | `set_aside` (language, pair, seed), `flagged_languages` (pinned flags file), `near_duplicates` (cos_099, cos_098); missing on either side counts as a difference; H8, H9 and H10 all compared | | covered |
| User decisions | grep and code reading | train side 912 (`run_r9_a28_prepare.py:41`, `run_r9_stage1_checks.py:48`, `rd9_compare.py` asserts 912); painting join option A (`run_r9_stage1_checks.py:64-78`); ≥ 0.92 count (`run_r9_stage1_checks.py:83`; `run_r9_descriptive.py:389-391` states it with the version gap) | | applied everywhere |

## 3. Findings

**MEDIUM 1: no test covers the preparation runner's count and V checks, or two V parameters that the rule names.**
- Where:
  - `run_r9_a28_prepare.py:41, 63, 73`;
  - `r9_a28.py:24` (`MIN_PAINTINGS = 30`) and `r9_a28.py:167` (groups = paintings);
  - `r9_a28.py:33-36` (`EXPECTED`, `EXPECTED_V`).
- What survives: 7 mutants, each with all tests passing. They are a cut of 29, groups = rows, changed `EXPECTED` and
  `EXPECTED_V` constants, train side = split train only (the user decision reverted), and the two runner asserts
  deleted. The unit tests always pass `min_paintings=30` explicitly, and they never exercise `main()`.
- Failure scenario: someone reruns the preparation with one of these changes. A cut of 29 or groups = rows adds Baroque
  (29 paintings), so V's style set has 8 values, and no test fails. A changed `EXPECTED` would then accept the wrong V.
- Effect on records: none. The records are pinned, I re-derived every value, and a rerun with different bytes fails
  `check_pins`.
- Fix (test only, outside the closure): add a test in `test_r9_a28.py` that calls `A.value_sets(df, dev)` with the
  default cut. One value has 30 paintings; another has 29 paintings over at least 30 rows. Add an assert that `EXPECTED`
  and `EXPECTED_V` equal the rule's numbers. Optionally, run `main()` on a monkeypatched synthetic CSV, split and dev
  set, with one painting in val.

**LOW 1 (for the user): the near-duplicate check misses same-artwork pairs under other file names.**
- Where: `run_r9_stage1_checks.py:55, 79-85`; rule §3.3 "Near-duplicates".
- Two readings: the rule's sentence says the train side is "every non-held ArtELingo painting" (49,121 paintings), and
  in the same sentence it says "790 + 122 = 912". The code and the user decision (ticket 15) use the 912 ArtELingo-28
  train-side paintings.
- Counts over all 49,121: ≥ 0.99: 0; ≥ 0.98: 0; ≥ 0.92: 72, against 3 over the 912. So the claim fields (0 / 0) are
  true on both readings.
- The top pairs are "new" read paintings that are the same artwork as a train-side painting under another file name:

  | Read painting (new) | Train-side painting | Cosine |
  |---|---|---|
  | `hugo-van-der-goes_st-hippolyte-triptych` | `dirk-bouts_st-hippolytus-triptych-1475` | 0.974 |
  | `dmitry-levitzky_portrait-of-ursula-mniszech-1782` | `…portrait-of-countess-ursula-mniszek-1782` | 0.953 |
  | `vincent-van-gogh_wheat-field-with-reaper-and-sun-1889` | `…the-reaper-1889` | 0.948 (likely a version) |

- Because of the version gap, a true duplicate scores below 0.98, so the claim's "0 / 0, none excluded" can be read as
  "no read painting was seen in training". That is not so for at least 2 of the 498 new paintings.
- Fix (text only, outside the closure): the report states the 49,121 counts (0, 0, 72) and names the two pairs, or the
  user confirms that the 912 reading is the intended disclosure. My numbers: `fr_b1_neardup_wide.json`.

**LOW 2: `near_duplicates_a28.json` is bound to no pin (the lead from area A); it is a gap, cheap to close.**
- Where: `run_r9_held.py:733-738` reads the live file and records only its SHA-256. `rederive/rd9_compare.py:224`
  compares the pass file's counts with the same live file. `run_r9_descriptive.py:326-331` checks the file against the
  pass record's SHA.
- Each of the three checks is consistent with the read, but none ties the file to stage 1. Rule §3.3's pin list does
  not name the file, so this is not a hard rule 6 gap. It does go against 25c's intent: a stage-1 fact, carried as
  written there.
- Failure scenario: the gitignored file is edited, or rewritten by a rerun of the stage-1 checks with changed code,
  between the pins and H8, H9 or H10. The claim then carries the new counts and C13 agrees. Nothing cross-checks
  `near_duplicates_source` across H8 to H10.
- Today's file is correct (0, 0, 3; SHA-256 `ba92530d433dded1c8d2b2a2037e23d015aed5361fb1249d1caa631e40da2206`).
- Cheapest fix (outside the closure): pin that SHA in `rederive/rd9_inputs.py` and read the file through `read_pinned`
  at `rd9_compare.py:224`, as the flags file is read. C13 then fails on any change, and C13 gates the claim. Add one
  test and one mutant.
- Alternative: add `results/near_duplicates_a28.json` to `pins_a28.json`. `check_pins` hashes every listed key, so
  `guard:a28_pins` covers it with no code change. This needs the pins file rewritten (`write_pins` refuses to
  overwrite), `RESULT_FILES` updated (`r9_a28_data.py:50`), `setup_a28` reading the path from the pin, and
  `test_r9_held_a28.py:518` updated. Either fix goes in before H8's ledger row; the ledger has no H8 to H10 row yet.

**LOW 3: three join and label guards survive deletion; the real data is clean on each.**
- `r9_a28_data.py:155`, two images for one painting: the loader places the last image (`image_of`, line 163) and never
  uses a caption's own `img` pointer. So a duplicated entry with a wrong image would load silently.
- `r9_a28.py:79`, translation painting vs CSV.
- `r9_a28.py:90`, one style and genre per painting.
- Real data: one image per painting, 0 painting mismatches, 0 multi-label paintings. `run_r9_stage1_checks.py:103` and
  the extraction adapter repeat the first two checks.
- Fix: one synthetic case each in `test_r9_a28_data.py` and `test_r9_a28.py`. Optionally place images by the caption's
  pointer.

**LOW 4: the 6 held paintings whose style differs reach no pinned file and no report code.**
- Rule §3.3 says they "are listed in the report". The list exists only in the gitignored, unpinned
  `results/a28_style_differs.json`; neither ticket 23's descriptive record nor the claim reads it. The content is
  correct.
- Fix: `run_r9_descriptive.py` copies the list and its SHA into its record, or the report checklist names the file
  (outside the closure).

**LOW 5: `pytest src/test/20261202_r9_scope/` fails at collection in a worktree (25 errors).**
- Where: `r9_a28.py:16` imports `src` before `r9_backbone` or `r9_common`, so in a worktree the worktree's `src` loads
  first and `r9_common`'s import guard raises. Each test file passes alone, and the main checkout is not affected.
- Fix: import `r9_backbone` first in `r9_a28.py` (outside the closure).

**Nits.**
- `rd9_compare.py:128`: the duplicate-entry assert in `reported_set_aside` survives deletion. A duplicated set-aside
  entry would pass C13 because both sides are compared as sets.
- `run_r9_a28_feasibility.py:69-71` reads the read set and V unpinned. It is a print-only diagnostic; no change is
  needed.

## 4. Hard rule 6 gaps

**No part of rule §3.3's ArtELingo-28 items or §5 is contradicted. One part is partial: the style-differs listing
(LOW 4).** Nothing unrequested is harmful: the three pinned `index.json` files are the C6 join key.

| Rule item | Status | Evidence |
|---|---|---|
| §3.3 Stores (ArtELingo-28 part) | implemented | one image per painting from the zip, 746 of 746; one caption per kept row; row ID = CSV position; counts, finite values and joins asserted (stage-1 checks, loader); re-derived (§2) |
| §3.3 Stores (SigLIP 2 on ArtELingo and SemArt, Qwen `artelingo_full`) | not B1 | area B2 |
| §3.3 Spot check | implemented | 20 images and 20 captions per store from `default_rng(0)`; minimum cosine ≥ 0.9989; I did not re-encode (no GPU) |
| §3.3 Painting join (option A) | implemented | 248 of 248 nearest; all cosines and margins reported; re-derived exactly |
| §3.3 Near-duplicates | implemented on the 912 reading | ≥ 0.99, ≥ 0.98 in the claim (`run_r9_apply_rule.py:412-414`); ≥ 0.92 with the gap note (`run_r9_descriptive.py:389-391`); reading question: LOW 1 |
| §3.3 Read set | implemented | all counts asserted and re-derived; "no painting both held and train-side" asserted and tested |
| §3.3 Labels | implemented, listing partial | mapping by name, genre 746 of 746, `other` to -1 (tested); "listed in the report": LOW 4 |
| §3.3 Translation | implemented | proxy codes `zul_Latn`, `swh_Latn`; 0 empty outputs, counted per language |
| §3.3 Language flags | implemented | four criteria, strict thresholds (tested at the edge), written 2026-10-13 18:30, before any read; flagged languages named in the claim |
| §3.3 Pins | implemented | 19 rule items plus 3 `index.json`; committed in 2fbad9f (19:27), before H8's ledger row (none yet) |
| §5 item 1 Rows and clusters | implemented (loader part) | kept rows; clusters = `np.unique` inverse of painting names; the label index before the drops is ticket 17 (not B1) |
| §5 item 2 V | implemented | re-derived; "every eligible value is a development value" asserted and tested; written and pinned |
| §5 item 3 Episodes | not B1 | tickets 17, 21 |
| §5 item 4 Bundle | loader part implemented | NaN on dropped rows, finite on kept rows (`check_a28_space`, mutants caught); the bundle level is ticket 18 |
| §5 item 5 Picks | not B1 | |

## 5. Guards whose deletion survives

**I ran 69 mutants, one at a time, each reverted at once and checked by SHA: 48 were caught and 21 survived.** Of the
21, 10 are redundant: a later check or an exception stops the same fault. The other 11 are real.
`git status` on the snapshot shows only `final_review/`. Details: `fr_b1_mutations.json`.

| Guard (file:line) | Status |
|---|---|
| `r9_a28_data.py:148` row_id join; `:152` caption painting; `:156` store paintings = read; `:160` image pointer painting; `:130-138` header, dtype, shapes; `:197-198` finite on kept, NaN on dropped; `:102-107` table columns, order, languages; `:89` `_pinned` SHA | caught |
| `r9_a28_data.py:274` `guard:pins_sha`; `:266` missing keys; `:261` backbones; `:272` missing file; `:262` results needed; `:49` `index.json` needed; `:227` no overwrite; `:219` `merged_from` | caught |
| `run_r9_held.py:772` `guard:a28_pins` (call deleted) | caught (`test_r9_held_a28.py`) |
| `r9_a28.py:65` held and train-side; `:77` translation cover; `:80` translation language; `:94` genre equals CSV; `:19` `other` to -1; `:83` strip; strict thresholds (`:146`, `:148`, `:149`, both sides); `:148` Latin-only script; `:114` median after drops; `:107` English on read paintings; `:162` V on kept rows; `:165` all three labels known; `:173` V inside development set | caught |
| `rd9_compare.py` flagged compare; near-duplicate compare; both counts; `set_aside` compare; all seeds; flags pinned; H9 and H10 included; train side 912; 746 read paintings | caught |
| `r9_a28_data.py:155` one image per painting; `r9_a28.py:79` translation painting; `:90` one label per painting | survive: LOW 3 |
| `r9_a28.py:24` cut 30; `:167` groups = paintings; `:33-36` `EXPECTED`, `EXPECTED_V`; `run_r9_a28_prepare.py:41` train side includes val; `:63`, `:73` count and V asserts | survive: MEDIUM 1 |
| `rd9_compare.py:128` duplicate set-aside entry | survives: nit |
| `r9_a28_data.py:141` count of texts vs kept rows; `:143` stores finite; `:159` pointer range; `:193` kept = not dropped; `:192` lengths; `:173` language index; `:117` backbone pinned; `:87` missing file in `_pinned`; `:270` SHA format; `:205` `_entry` missing file | survive, redundant: a later check or an exception stops the same fault |

Not done:
- the spot check's local re-encoding (it needs the encoders; no GPU);
- `hit_max_new_tokens`, which cannot be recounted from text.

Files I would remove after the round: none outside this folder. The scratchpad `fr_b/` holds only logs.
