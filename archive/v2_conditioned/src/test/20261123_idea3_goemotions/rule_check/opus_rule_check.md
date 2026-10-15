# Rule check: round 5 decision rule (idea 3, draft, before commit)

Written 2026-10-07 09:56 (Amsterdam) by a fresh Opus reviewer. Checked: `../DECISION_RULE.md` (uncommitted draft, 747
lines, SHA-256 f960ffc2101aeb1a4a8ba95e78bf33fc93a70c22f51eb47259e10ea3ebea985c, unchanged since 09:35) against the spec
(393c3c2), the handoff, round 4's and round 3's rules, round 4's rule check and final review, round 4's report §7 and
§8.4, and every code path the draft names.

My script `rc5_check.py` (output `rc5_check.json`, ignored by the repository's `*.json` rule) ran once on CPU (8
threads, no bytecode, one process, 117 s). It built round 4's seed-42 bundle and reproduced the earlier rounds' numbers.
It computed no number of G-T, G-TF or B′_G, fitted no GE head and computed no AUC or pair statistic with a GoEmotions
placement. It joined the selection captions only to count them and passed none of them to any model. GoEmotions ran
only on D3's 2,048 **scorer-train** regression captions, which the affect line had already put through the model. Two
one-off CPU calls counted selection paintings and tokenised scorer-train captions; neither used the model. I wrote only
under `rule_check/`. No `__pycache__` or `.pyc` appeared anywhere in the repository after my start marker, and the
HF cache's `.no_exist` markers are unchanged (dated 2026-09-22).

## Verdict: FIX BEFORE COMMIT (0 blocking, 5 should-fix, 14 nits)

The rule matches the spec on every settled point. Every SHA-256 matches: the 36 rows of D12, the header's three, the
seven abbreviated ones after the table, all 36 of round 3's `r3_common.INPUTS`, all 22 of round 4's `r4_common.INPUTS`
and round 1's 9. Every quoted constant reproduces bit for bit through the float32 path:

- τ, R1, AFF, B′(A1), AFF − B′(A1);
- the CLIP heads (posteriors, n_iter_ 176/154, accuracies, draw SHA);
- `pair_stats_heads` (all 13 values) and the AUC 0.7870951145887375. The AUC also reproduces from the **pipeline's**
  P, which equals `cand_R1_A0.npz` bit for bit.

The Q_CLIP path of item 4 reproduces AFF exactly. In it, τ′ = τ element by element, B′_Q = B′(A0), and the bar
comparator is "B′_G".

None of the findings below can change a number. They close gaps that would force an improvisation later (S1 to S3, S5)
and add one cheap positive control (S4).

## Blocking

None.

## Should-fix

**S1. §6.1 and §8 step 7: no code owner and no test list for the sensitivity projection.** §8 step 2's list of seed-42
code has no sensitivity path. Step 5 has the seed-42 runner end with the diagnostics. List B is "the build runner, the GO
runner, the rule application and the descriptive pass", and §6.1 must run "before any test-seed code runs". So the code
that writes `results/sensitivity.json` would be written after the carry, with no listed test. That is the pattern of
round 4's Task 3 split (final review S3). *Fix:*

- In §8 step 2 add "the sensitivity path (§6.1)" to the seed-42 code.
- Add to list A item 7: "the sensitivity path runs only when `results/carry.json` names a carried candidate and the log
  holds the phase-1 agreement record (refused otherwise). It computes the nine checks in §6.5's order with round 3's
  `r3_stats.sensitivity` and reproduces SE and x on a hand-computed example."

**S2. §8 "What the re-derivation may import": phase 2 cannot be done with the list as written.** Phase 2 must re-derive
"every GO check of §6.5" and "the hash check of §6.2" on seeds 52 to 54. The RCA checks need RCA's per-anchor arrays
from `per_anchor_seed{s}.npz`, and the hash check needs `baselines_seed{s}.json`. For the test seeds neither file is a
"stored file of D12": they are written by §6.2. Seed 42 is fine, since `per_anchor_seed42.npz` is a row of round 3's D15.
`EvalContext` covers the episodes. *Fix:* append to the list "and, in phase 2, each test seed's `run_baselines.py`
outputs (`per_anchor_seed{s}.npz` for cosine and RCA, `baselines_seed{s}.json` for the hash check; episodes through
`EvalContext`), each SHA-256 asserted against `results/build_seed{s}.json`".

I checked the rest of the list and it suffices. B is `crossfit_condition_free` on `uniform_probe_scores(load_posteriors(...),
ep, E2's three parts)`, because `run_n6.n6_terms(...)[2]` is that call. B′(A1) on seed 42 comes from `load_bundle`, and
phase 2 does not need it. `rc.unit`, `global_labels`, the draw constants and `as_int4` are a few lines each.

**S3. §4 item 5 contradicts §6.4 on the test seeds.** Item 5 runs "G-TF's [readers] on F_G" on every seed, and §4 item 3
builds F_G on every seed. §6.4 says "the candidate that was not carried is never computed on a test seed". Round 4's rule
check S4 found the same pattern, and the draft fixed the gates (§4 item 6) but not the reader. *Fix:* add to §4 item 5
"On a test seed, G-TF's reader, and F_G (§4 item 3), only if G-TF is carried; stack_G and B′_G always (B′_G is a GO
comparator)."

**S4. D5 and §5 items 4 and 5: nothing on seed 42 shows that the GE extension actually uses Q_GE.** Item 4 is an
identity test with Q_CLIP, so an extension that silently ignores its Q and returns the bundle's own fields passes it.
§6.4(e) checks Q_GE ≠ Q_CLIP, but only on the test seeds. On seed 42 such a bug would surface only as Δ_k = 0 for G-T
(a boundary flag) or as a disagreement with phase 1. *Fix:*

- **D5.** Add: "With Q = Q_GE (item 5, after release), the affect slice of stack_G equals, exactly, an independent
  recomputation from Q_GE and the affect image posterior: `einsum('nc,nkc->nk', p_img[anchor], Q_GE[candidates])` for
  i2t and `einsum('nc,nkc->nk', Q_GE[anchor], p_img[candidates])` for t2i. It differs from the bundle's affect slice on
  at least one episode, and F_G's columns 0 to 5 differ from F's on at least one episode. These are boolean
  assertions; no value is printed."
- **List A item 4.** Add "with a synthetic Q ≠ Q_CLIP only the affect slice, feature columns 0 to 5 and B′_Q change;
  an extension that ignores its Q fires".

**S5. Precedence (lines 16 to 21): sections incorporated "with their text unchanged" contradict this file.**
- **Round 4's §5 item 2.** Item 1 cites it ("as … round 4's §5 item 2 state them"). Its text requires AFF "computed
  through this round's code path: the candidate gate function with both factors set to 1 … and AFF's family run from
  those gates", which is `r4_fusion.gates_candidate` / `run_candidate`, a module D12 marks "not imported".
- **Round 4's D1, last sentence.** "Every candidate's term is AFF's: T^c = Σ_h P^c(h)·s_h on A0 …" contradicts
  G-T's and G-TF's terms. Round 4's rule check N3 is the precedent for excluding such a sentence.
- **Round 3's sections pulled in "through" round 4.** Round 3's §10 ("CPU only", "write only to this folder's
  `results/`") conflicts with D2's GPU step and this folder's `cache/`. Round 3's §6.11 (AFF's disclosure) and §7
  item 7 (the random-share control) are not this round's.

"This file governs" resolves each conflict, but an implementer or re-deriver who reads item 1 literally will look for
round 4's gate function. *Fix:* replace the list with:

> (§1 as this file's §1 restricts it; §2; D1 except its last sentence; D4; D9; D10; §5 item 1; §5 item 2's targets,
> not its code-path clause; §6.2; §6.7) … through it, round 3's §2, D1 to D12, D15 with its text, §4, §5 items 1 to 3,
> §6.1, §6.2 and §9 are part of this file too; round 3's §6.11, §7 item 7 and §10 are not (this file's §6.11 and §10
> replace them).

## Nits

- **N1. D3 tolerance (line 140), measured.** I compared a CPU rerun of D3's exact sample (positions from
  `default_rng(5)`, join as D3 states, batch 256, max_length 64) with the stored CUDA `affect_probs`:
  - max |diff| 4.470348358154297e-06, mean 3.49e-08, and 0 of 57,344 entries above 1e-5;
  - batch 256 against batch 64 on CPU: 5.66e-07;
  - the same sample shifted by one row: 0.9599.

  1e-3 is safe (224× the observed maximum) but loose. It may not catch half-precision inference (about 1e-3 to 1e-2).
  1e-4 keeps 22× headroom. Either way, replace "which we expect to stay below 1e-5" with the measured 4.5e-6, and if
  you change the tolerance, move list A item 2's 0.9e-3 / 1.1e-3 to match.
- **N2. D3: "other truncation of a long caption" is not exercised.** The longest caption in the regression sample has
  41 tokens. No scorer-train caption exceeds 58 tokens (0 of 183,694 over 64), so truncation at 64 never binds. A wrong
  max_length above 41 would pass. *Fix:* drop that error from the list, or say "a max_length below 41"; record the
  selection captions' maximum token count and the count over 64 in the GoEmotions record (descriptive).
- **N3. Header line 9 ("no caption of the selection rows had been passed to the GoEmotions model") and spec §2's
  "for the first time".** The percept branch's exploratory pilot
  (`experiment/percept_topic_pipeline:src/test/20260923_artelingo_buddy_analysis/run_affect_pilot.py`, committed
  2026-09-22) passed every English `artelingo_train` caption through the same model, with batch 256 and max_length 64.
  That includes the selection and held rows. Nothing of it is read here. *Fix:* "no caption of the selection rows had
  been passed to the GoEmotions model in the v2 line (an exploratory percept-branch pilot of 2026-09-22 pooled
  GoEmotions over every `artelingo_train` caption; this round reads nothing of it)". Add the same clause to §6.11's
  external-model disclosure.
- **N4. D2 item 2: pin what is actually loaded.** `load_goemotions` resolves `refs/main`, and `ANNOTATIONS_PATH`
  honours the `COSIR_ARTELINGO_ANNOTATIONS` environment variable. Assert that `refs/main` is d75048…26be, hash the files
  at the resolved snapshot path (not a hard-coded one), and assert `ANNOTATIONS_PATH` is the D12 path with the variable
  unset.
- **N5. D12.** The rule names `run_gonogo.EvalContext` and `run_gonogo.sha_array` (§1, D4), and item 4's genre codes
  come from `src/data/wikiart_genre.py`. Neither file is hashed, and the T1 row covers only modules of rounds 1, 3 and 4.
  Add `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` (and optionally `wikiart_genre.py`) to D12.
- **N6. §8 lapse 5 (line 626).** "No run other than the real seed-42 run computes a GE-placement result on seed 42"
  contradicts phase 1 and the final review. Write "No run of this round's implementation other than …".
- **N7. §1.** Round 4's glossary rows that apply carry round 4's section numbers: "carried candidate" points to §5 item
  8, which is item 7 here. Add "section references in those rows are read as this file's".
- **N8. D6 (line 216).** `rc_core.thresholds` returns `(taus, n)`. Write "τ′ = `rc_core.thresholds(m′)[0]`" and require
  n = 24,576. I confirmed that it returns `r3_common.TAUS` element by element (`==`) from the pipeline's margins.
- **N9. Diagnostic (d) (line 455).** "AFF's +0.566 and +0.916" are correct: bs_09_direction.json `AFF.i2t` and
  `AFF.t2i`, fused minus cf R@1, 0.5655924479166714 and 0.91552734375. R1's are +0.216 and +0.671, as §6.11 says. Name
  the keys ("fused minus counterpart R@1, `AFF.{i2t,t2i}`") so that nobody confuses these with the handoff's R1
  numbers.
- **N10. D5 (line 204).** `r4_bundle.r3_fingerprint` fingerprints `post` on A0's keys only, and round 4's `A1_FIELDS`
  do not include `post["csd"]`. Fingerprint the whole of `bundle.post` as well. I confirmed that `bundle.post["affect"]`
  *is* `r3_bundle._HEADS[60000]["post"]`, the same dict object, so D5's no-assignment rule and its identity check are
  necessary.
- **N11. Spec §4's "keeping both B′ versions can only raise the bar".** The rule does not repeat it; the report should
  not either, or should qualify it. A larger comparator set raises clause 1, the bar-margin point. It does not
  necessarily raise clause 2: the lower bound against the max-mean comparator can be higher than against another
  comparator with a noisier pairing. The GO pass checks every comparator separately, so the test is unaffected.
- **N12. §8 re-derivation.** Say where it reads the GoEmotions file's SHA-256: the record
  `cache/r5_goemotions_selection.json` and the run-log line, not `r5_common.py`, which is implementation code.
- **N13. D2 device.** A GPU file and a CPU file differ by about 5e-6 in input, which could flip a near-tie in the integer
  statistics. The device is fixed by availability before any number and the file is frozen, so this is no forking path.
  Say so in one line (D2 item 3 or §6.11). For planning: the 2,048 captions took 20.8 s on 8 CPU threads, so the
  32,413 selection captions should take about 5.5 min on CPU. The handoff estimated 10 to 30.
- **N14. List A item 2.** Also make D2 item 1's row assertion fire on a held-overlapping row set, since D2 asserts
  disjointness from `held`. Where §6 reads τ′ from `dev_seed42.json` "SHA-256 asserted", name where that SHA-256 is
  recorded (for example in `carry.json`).

## Spec fidelity (no finding)

Every settled point of spec §1 to §7 is in the rule, and nothing contradicts it:

- §1 and §2: design L parked; the GE head recipe with raw probabilities; the fallback fixed before any number.
- §3 and §4: G-T and G-TF exactly as the spec's table, and the four regression items. Comparators {B, B′(A0), B′_G,
  counterpart} with B′(A1) beside; D10, Δ_k > 0 and the 24-ranking tie band going to G-T; the AUCs and diagnostics
  descriptive (the rule adds "after `carry.json`", a strengthening).
- §5 and §6: the nine GO checks; build, freezes and readings; the disclosure and prior.
- §7: test-seed code only if carried and before any test-seed run; re-derivation independence (S2 aside) and blindness;
  no skipped tests; PYTHONDONTWRITEBYTECODE.

The lapses of round 4's report §7 and §8.4 and its final review's §5 before-reuse table are each closed, by text or by
a list A or B test. Round 4's S5 and S6 were fixed in 7f7553b, as the rule says.

Logic checked:
- **G-T.** G-T's gates are AFF's (same m, π, τ; float32), and σ* depends on B only. So G-T's counterpart differs from
  AFF's only through the term, and it removes only the condition, which is the matched-control lesson.
- **G-TF.** G-TF's counterpart is built under its own gates. B′_G credits the placement's condition-free value to a
  comparator. I found no route to a pass for a condition-free reason.
- **τ′.** τ′ is label-free and its seed-42 origin is disclosed in §6.11.
- **The carry band.** 0.05 pp × 49,152 / 100 = 24.576 (float64 24.576000000000004, so the integer literal is right). 24
  rankings = 0.048828125 pp and 25 = 0.050862630208333 pp.

## What I verified (values at full precision)

- **SHA-256 (`sha256sum`).**
  - All 36 rows of D12: the 7 HF snapshot files are hashed at the blobs `refs/main` → d75048…26be points to; the
    annotations file is 6a4e5b17…894d.
  - The header's spec, round 4 rule and round 3 rule; `git show 393c3c2:` of the spec gives bc738402…184ee.
  - The paragraph after D12: told_oracle.json 76d9ec89…70d2, per_anchor_told_oracle.npz 27e81011…6af1366, rc_core.py
    e649fff5…185c, rc_tau.json e10cf52b…02bf, bs_07_detector.py 1b287bcc…088a and .json e9d52462…85e1,
    codes_provenance.json 8e6a517b…bfaf, and `baselines_seed{42,43,45,47,48,49,50,51}.json`.
  - The 36 inputs of `r3_common.INPUTS`, the 22 of `r4_common.INPUTS` and round 1's 9: 0 mismatches.
- **Versions.** Python 3.11.15, numpy 2.2.6, scikit-learn 1.6.1, torch 2.11.0+cu130, transformers 5.6.2, tokenizers
  0.22.2. The torch, transformers and tokenizers dist-info are dated 2026-04-28; `affect_prepare.npz` 2026-10-01; its
  record says "device": "cuda".
- **Seed guard** (r4_bundle `_check_seed` after import): admits 42, 52, 53 and 54, and 9001 only with smoke; refuses 49,
  55 and 9001 without smoke. `R3.TEST_SEEDS == R4.TEST_SEEDS == (52, 53, 54)`.
- **Bundle and reader.**
  - `r4_bundle.build_bundle(42, False)`: post keys (affect, image, caption, csd).
  - The reader's P equals `cand_R1_A0.npz` `probs__a/b` bit for bit (float64, max |diff| 0.0) and `seed42_arrays.npz`
    `P__a/b`. The picks equal the stored picks.
- **AUC** (bs_07 definition, 8,192 positives): 0.7870951145887375 from the pipeline's P, from `cand_R1_A0.npz` and from
  `seed42_arrays.npz`, equal to `bs_07_detector.json`.
- **τ.** `rc_core.thresholds(m)` gives [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526,
  0.7502585816077211] with n = 24,576, all `==` TAUS. `np.percentile` with condition a first gives the same.
- **R1.**
  - Cells 116, 119 / 58, 123; σ* 0 / 0; comparator: the counterpart.
  - Bar margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852].
  - Gain statistic 2.667236328125 [2.325087836946873, 3.012361650695922].
- **AFF.**
  - Fused 19.136555989583336, counterpart 18.39599609375, comparator B′(A0).
  - Bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968].
  - Margin against the counterpart 0.7405598958333333 [0.5196896694963071, 0.9598857494832738].
  - Gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384]; either change −1.629638671875.
  - Per pair: 0.9765625, 1.45263671875, −0.32958984375.
  - Cells 39, 119 / 149, 10; σ* 0 / 0; τ_0 open counts 9,941 / 3,627.
  - AFF − R1: fused 0.21769205729166666 [0.06425880757348419, 0.3709597330984391], bar 0.25634765625
    [0.04280778303598444, 0.46195041633015954].
  - All `r1_*` and `aff_*` per-anchor arrays and the gates equal `seed42_arrays.npz` exactly.
- **B′(A1) and baselines.**
  - Means: B′(A1) 18.804931640625, B′(A0) 18.436686197916664, B 18.341064453125.
  - AFF − B′(A1): 0.33162434895833337 [0.048231414333532084, 0.6246158772581268].
  - Either per gain: 0.5238718116415958 (= 1.629638671875 / 3.110758463541667).
- **Item 4 with Q_CLIP**, D5 exactly as written (`post_Q` a new dict; `grouping_stack`;
  `seed42_features(SimpleNamespace(ctx, post_Q), A0)[0]`; `crossfit_condition_free(ctx.cos, t_n1u,
  uniform_probe_scores(post_Q, ep, A0), ctx.parity)[0]`):
  - stack, F, B′ scores and per-anchor arrays all equal the bundle's; the `_HEADS` fingerprint is unchanged.
  - `r3_fusion.reader(SimpleNamespace(F, stack_Q))` and `reader(SimpleNamespace(F_Q, stack_Q))` give AFF's P, T, m
    and π exactly.
  - τ′ == TAUS elementwise; G-TF's gates equal AFF's (float32).
  - `r4_stats.bar_comparator` with the order B′_G, B′(A0), counterpart, B returns "Bprime_G".
- **Placement function on CLIP features** (fit_one_head's draw, check rows, `LogisticRegression(C=1, max_iter=300)`,
  scatter):
  - posteriors equal `fit_one_head`'s exactly, NaN pattern included, for txt and img;
  - `classes_` 0..40; n_iter_ 176 (txt) and 154 (img); accuracies 35.72 and 9.81 (JSON round trip equal);
  - draw SHA 7be956c09bf716547df20264435388bd3645ae963df728636774e80359cdef5c; `scorer_train[pos] == draw`; check rows
    all scorer-train;
  - the head record equals told_oracle.json arm L's `head`;
  - partition_L: 183,694 rows, 41 classes, smallest 204.
- **pair_stats_heads** with the inputs of item 4 (float64 selection posteriors, `artelingo_aspect_labels` codes,
  `ctx.groups[selection]`) equals `arms.L.pairs.heads` as a JSON round trip, 13 values; the group lift is
  2.7111312041209863.
- **Rows.**
  - `artelingo_splits` scorer_train and selection equal the grid cache's (`20261013_stage_d_selection/cache/prepare.npz`)
    that `run_affect.py` and `run_posthoc_affect.py` used. So `X[scorer_train[i]] = affect_probs[i]` is the alignment
    partition_L assumes.
  - Selection: ascending and unique, equal to `ctx.selection`, 32,413 rows on 6,451 paintings, disjoint from
    scorer_train and held (0 and 0).
  - The selection join gives 32,413 non-empty strings.
  - The regression join equals the full scorer-train join at those positions.
  - `affect_probs` (183,694 × 28, float32) ranges from 4.2079060222022235e-05 to 0.9885085225105286.
- **GoEmotions call.** `goemotions_probabilities` (padding=True, truncation=True, max_length, batch slicing in input
  order) is exactly `run_affect.py`'s call. The CPU rerun numbers are in N1.
- **Other.** 14 October 2026 is a Wednesday. `.gitignore` of round 4 ignores `cache/`, `results/` and `*.npz`.
  Episode seeds 52 and later are free in the ledger.

Files: `rule_check/rc5_check.py`, `rule_check/rc5_check.json` (6 KB). Nothing over 1 GB was written.
