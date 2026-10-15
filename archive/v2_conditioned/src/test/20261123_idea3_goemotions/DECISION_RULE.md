# Decision rule: reader fix, round 5 (idea 3: GoEmotions placement of captions on AFF, developed on seed 42, fresh-seed test if carried), committed before any code

**Written** 2026-10-07 between 08:10 and 09:34 (Amsterdam), fixed after the rule check 09:59 to 10:01, from the approved spec
`docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md` (commit 393c3c2, SHA-256
bc7384027d856102ab5efc8eb24a1a7831a4ce33e660f356a66628a2dad184ee), before any implementation script of this folder
exists. The only numbers it states are earlier rounds', the brainstorm's and `told_oracle.json`'s, which it reuses, and
facts about stored files and the named code that were computed while it was written (row counts, the caption join's
count, the CLIP heads' lbfgs iteration counts, the reproduction of 0.787 and 1.1445 by the code this file names). When it
was written, no caption of the selection rows had been passed to the GoEmotions model in the v2 line (an exploratory
percept-branch pilot of 2026-09-22, `experiment/percept_topic_pipeline:src/test/20260923_artelingo_buddy_analysis/run_affect_pilot.py`,
pooled GoEmotions over every `artelingo_train` caption, selection and held rows included; this round reads nothing of
it), no GE head had been fitted, and no number of G-T, G-TF or B′_G existed.

**Precedence.** Where this file differs from the spec, from round 4's rule
(`src/test/20261122_round4_aff_vetoes/DECISION_RULE.md`, SHA-256
cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b), from round 3's rule
(`src/test/20261121_round3_affect_gate/DECISION_RULE.md`, SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925) or from earlier rules, **this file governs**. Round
4's rule is part of this file by reference in the sections this file cites (§1 as this file's §1 restricts it; §2; D1
except its last sentence, "Every candidate's term is AFF's …"; D4; D9; D10; §5 item 1; §5 item 2's targets, not its
code-path clause, since this round does not import round 4's candidate gate function; §6.2; §6.7), with their text
unchanged unless this file says otherwise. Through it, round 3's §2, D1 to D12, D15 with its text, §4, §5 items 1 to 3,
§6.1, §6.2 and §9 are part of this file too; round 3's §6.11, §7 item 7 and §10 are not (this file's §6.11 and §10
replace them). Round 4's candidates V4, V2 and V24, its v₇₅ and its random-share control are
not part of this round. The spec and the earlier rules are otherwise cited for provenance only.

**Checked** before its commit by a fresh Opus reviewer (spec §7; not an ARS round; report `rule_check/opus_rule_check.md`,
09:36 to 09:59). It found 0 blocking, 5 should-fix and 14 nit findings, all applied before the commit. It verified every
SHA-256 of D12 and of the tables of round 3's and round 4's code (67 inputs), and reproduced through the float32 path, at
full precision, τ, R1, AFF, B′(A1), AFF minus B′(A1), the CLIP heads (posteriors, iteration counts, accuracies, draw),
the 13 pair statistics, the AUC 0.7870951145887375 from the pipeline's own P (equal to round 2's stored P bit for bit),
and item 4's CLIP-placement path (AFF exactly, τ′ = τ, B′_Q = B′(A0)). It ran GoEmotions on CPU only on D3's 2,048
scorer-train captions (D3's measured tolerance), and computed no number of G-T, G-TF or B′_G.

**Status.** Two candidates built on AFF, G-T and G-TF (§3 D6), in one development family on seed 42 (§5), after
regression checks of the code and of the GoEmotions step on seed 42. At most one candidate is carried; it alone is
tested on the fresh episode seeds 52, 53 and 54 (§6), the only confirmatory step. AFF, frozen as tested in round 3, runs
beside it and enters one GO check (the paired check against AFF). B′(A1) is reported beside AFF and each candidate and
decides nothing.

**Authorisation.** The user's decisions of 2026-10-07: the handoff
`docs/superpowers/handoffs/2026-10-07-idea3-goemotions-handoff.md` §2 (round 4's kill stands and AFF stays the current
best; idea 3 before the held-split paper test, so both held reads stay available; a cheap measured step on seed 42
before any fresh-seed round; the process of rounds 3 and 4; commits to `main` without asking, scoped by explicit path,
never pushed); the spec's open points, settled one at a time in chat between 07:45 and 08:04 (idea 3 is not the parked
design L; the placement is a multinomial logistic head from the 28 GoEmotions probabilities with the CLIP caption head's
recipe; two candidates, G-T and G-TF; round 4's carry on seed 42 with AFF's paired check in GO, detection AUCs
descriptive only, and the test on seeds 52 to 54 pre-registered in this file; the comparators B, B′(A0), B′_G and each
candidate's own counterpart, B′(A1) beside; GoEmotions once on the selection captions, on the local GPU under the lock
if free, otherwise on CPU); both design sections approved; the spec committed (393c3c2, 08:05) and approved by the user with their "go" for
this file at about 08:10. The user allowed this file, and the work under it, to be committed to `main`
without asking. After its commit this file changes only with the user's approval.

**Dates.** Folder and report dates (`20261123`, `2026-11-23`; earlier rounds' `20261117`, `20261118`, `20261120`,
`20261121`, `20261122`) are sequence numbers, not calendar dates. Calendar times in this file are Amsterdam local time.

**Prior** (written before any number of this round; spec §6). We expect a small gain at best. Every agreement and every
grouping score pairs one image with one caption, so s_affect multiplies the sharper caption posterior by the image head's,
which reaches 9.81% held-out accuracy over the 41 communities; the caption side can sharpen the term only as far as the
image side allows. The comparators B′_G and each candidate's counterpart are rebuilt with the same placement and take
whatever condition-free value the sharper term has. G-T changes only the term, so its gates and picks are AFF's; we do
not expect it to move AFF's seed-42 R@1 by much in either direction. G-TF's frozen half-readers were trained on
features of the CLIP placement; on GoEmotions features their probabilities, picks and margins shift, and we cannot
predict the direction of that shift from what we have (τ′ re-centres the thresholds, not the picks). The brainstorm's
suggested detection bar (an AUC above 0.83) is not a bar here. A kill at the carry would not surprise us; a GO would.

## 1. Glossary

Round 3's glossary (its §1) and round 4's (its §1, the rows for AFF, A1, csd, Δ_k, AFF check, carried candidate, test
seeds and smoke seeds) apply; section and item references in those rows are read as this file's (round 4's §5 item 8,
the carry, is this file's §5 item 7). The terms below are new or changed.

| Term | Meaning in this file |
|---|---|
| selection rows, selection captions | `ctx.selection` of `run_gonogo.EvalContext` (32,413 rows on 6,451 paintings, ascending, equal to `artelingo_splits(ctx.data).selection`) and their ArtELingo captions |
| GoEmotions file | `cache/r5_goemotions_selection.npz` of this folder: the 28 GoEmotions sigmoid probabilities of each selection caption (D2) |
| GE head | the multinomial logistic regression from the 28 GoEmotions probabilities to the 41 `partition_L` communities, fitted with the CLIP caption head's recipe (D4) |
| placement, Q | a caption-side posterior over the 41 affect communities: float32 (308,723, 41), finite on selection rows, NaN elsewhere, in `partition_L` label order (D5) |
| CLIP placement, Q_CLIP | the standard affect caption posterior `post["affect"]["txt"]` of round 3's bundle (round 3's D2); used by AFF, by B′(A0) and by §5 item 4 |
| GE placement, Q_GE | the GE head's posterior on the selection captions (D4), the GE file `cache/r5_ge_posterior.npz` |
| post_Q, stack_Q, F_Q, B′_Q | the posteriors with the affect caption side replaced by Q, and the grouping scores, the 18 reader features and B′ rebuilt from them (D5) |
| B′_G | B′_{Q_GE}: round 3's condition-free B′(A0) recipe with the GE placement (D5) |
| G-T, G-TF | the two candidates (D6): AFF with the GE placement in the steering term only (G-T), or in the term and the reader's six affect features (G-TF) |
| τ′_0..τ′_3 | G-TF's thresholds: the 0th, 25th, 50th and 75th percentiles of G-TF's 24,576 seed-42 margins, frozen for later seeds (D6) |
| GE-placement result | any number or array computed on episodes or pairs from the GE placement (D11); refused by the guard before §5 items 1 to 4 pass |

## 2. Data, metrics and intervals

As round 4's rule §2 (and through it round 3's §2), with these changes:

- **Test-seed episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed{52,53,54}.npz`, built once each by
  §6.2 only if a candidate is carried, 12,288 episodes per seed on selection rows, loaded through
  `run_gonogo.EvalContext(seed, False)`; pooled over the test seeds as round 4's §2 says (36,864 episodes, clusters the
  anchor paintings `groups[anchor]` across seeds).
- **Metrics, intervals and precision** as round 4's §2: R@1, other-aspect rate, condition gain, either rate, per
  episode averaged over the four rankings, in percentage points; 95% percentile intervals of the anchor-painting
  bootstrap (`src.eval.aspect_metrics.cluster_bootstrap`, 5,000 resamples, seed 42, chunk 250, scaled by round 1's
  `common.point_ci`), cross-fit picks fixed before resampling; "a lower bound above 0" means strictly greater than 0;
  every threshold applies to the full-precision value; the integer comparisons of D9 and §5 item 7 are exact.
- **Rows the GoEmotions model reads:** the selection captions once (D2) and the 2,048 scorer-train captions of the
  regression sample (D3), which the affect line already passed through the model; the re-derivation reruns 1,024 of the
  selection captions on CPU (§8). Held and validation captions are not joined and not read. Held rows are not read.
- **GPU:** only the GoEmotions step of D2 may use the GPU, under the lock (D2). Everything else runs on CPU.

## 3. Definitions (the only definitions; every item below refers to them)

**D1. Inherited.** Round 4's D1 applies (round 3's D2 to D12 and D7, with round 3's D1 groupings and AFF frozen by
name), with "AFF" in round 3's D8 and D9 replaced by each candidate where this file says so. Round 4's D4 (B′(A1),
seed 42 mean R@1 18.804931640625) applies for the beside line only. AFF is round 3's candidate frozen as tested: the A0
half-readers on the 18 features F of the CLIP placement, P^c, T^c = Σ_h P^c(h)·s_h, m^c, π^c, the gate
g_t^c = 1[m^c ≥ τ_t]·1[π^c = affect] at τ_0..τ_3 = 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526,
0.7502585816077211 (`r3_common.TAUS`, round 1's `rc_tau.json`), its 224 cells and per-seed cross-fits.

**D2. The GoEmotions file.** Computed once, by this round's GoEmotions runner, in one process:
1. *Rows and captions.* rows = `ctx.selection` (asserted: ascending and unique, length 32,413, equal to
   `artelingo_splits(ctx.data).selection`, disjoint from `scorer_train` and from `held`). Captions =
   `src.data.artelingo.join_captions(ctx.data.sample_ids[rows], annotations)` with annotations read from
   `src.data.artelingo.ANNOTATIONS_PATH` (`/data/PDD/artelingo/artelingo_train.json`, D12), the join the affect line's
   post-hoc script used (`src/test/20261018_affect_factor_learning/run_posthoc_affect.py`, its `captions_sl`); 32,413
   non-empty strings in the order of rows (asserted; 32,413 when this file was written).
2. *Model and call.* `src.data.affect.load_goemotions(device=...)` (SamLowe/roberta-base-go_emotions from the local
   Hugging Face cache, snapshot d75048347613a25d77de8cf6412eaae9fa7b26be, files hashed in D12) and
   `goemotions_probabilities(captions, loaded=..., batch_size=256, max_length=64)`: 28 sigmoid probabilities per
   caption, float32, in input order, finite and in [0, 1] (asserted). Before loading, the runner asserts that the
   cache's `refs/main` names snapshot d75048347613a25d77de8cf6412eaae9fa7b26be and hashes the seven files at the
   resolved snapshot path (D12), and asserts that `COSIR_ARTELINGO_ANNOTATIONS` is unset and `ANNOTATIONS_PATH` is the
   D12 path.
3. *Device.* Before the run, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`. If it lists no
   process, the runner is launched as `flock -n -o -E 75 /tmp/gpu0.lock env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
   MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 <python> run_r5_goemotions.py --device cuda`. If a process is listed, or
   flock exits 75, the runner is launched on CPU (`CUDA_VISIBLE_DEVICES=` and the same variables, `--device cpu`); the
   log records what held the GPU (pid, command, project from `/proc/<pid>/cwd`), and nothing of another session is
   touched. A GPU run that crashes is repeated once on CPU after its partial outputs are deleted, and the crash is
   logged. The runner asserts that the device it was given is the device it uses. A GPU file and a CPU file differ by
   about 5e-6 per probability (D3), which could flip a near-tie in the integer statistics; the device is fixed by
   availability before any number and the file is then frozen, so the choice is not a forking path.
4. *Order inside the process.* First the regression sample of D3 (§5 item 2) with the same loaded model; if it fails,
   no selection caption is passed to the model, the failure record is written and the run stops. Then the selection
   captions.
5. *Output.* `cache/r5_goemotions_selection.npz` with `probs` (float32, 32,413 × 28), `rows` (int64, the rows of item
   1) and `sample_ids` (int64, `ctx.data.sample_ids[rows]`); the record `cache/r5_goemotions_selection.json` with the
   device tag (`cuda:0` and the GPU name, or `cpu`), the torch, transformers, tokenizers and scikit-learn versions, the
   model snapshot and its file SHA-256s, batch 256, max_length 64, the row and caption counts, the selection
   captions' largest token count and the count of captions over 64 tokens (descriptive), item 2's comparison (D3), the Amsterdam time, the npz's SHA-256 and the SHA-256 of the `probs` bytes. The npz's SHA-256 is written to the
   run log and committed as a constant of `r5_common.py` (a one-line commit) before any later step runs; every later step,
   the re-derivation included, asserts it. The file is never recomputed and never overwritten; whichever device produced
   it, it is the round's input.

**D3. The scorer-train regression sample (§5 item 2) and its tolerance.** Positions
`numpy.sort(numpy.random.default_rng(5).choice(183_694, size=2_048, replace=False))` into the scorer-train order of
`affect_prepare.npz` (`affect_probs` row i belongs to global row `scorer_train[i]`, `scorer_train =
artelingo_splits(ctx.data).scorer_train`, ascending; this is the alignment that `run_told_oracle.py` and
`partition_L` assume). Captions `join_captions(ctx.data.sample_ids[scorer_train[pos]], annotations)` (when this file
was written they equalled the full scorer-train join at those positions), in that order, through D2's model and call.
**Pass** if max |rerun − `affect_probs[pos]`| ≤ 1e-4 over all 2,048 × 28 entries; the maximum, the mean absolute
difference and the count of entries above 1e-5 are recorded (descriptive). *Why 1e-4:* `affect_prepare.npz` was computed
on CUDA (`affect_prepare.json`: `"device": "cuda"`) by `run_affect.py` in batches of 256 consecutive scorer-train
captions padded to the longest caption of each batch, with max_length 64, under the library versions installed in the
CoSiR environment since 2026-04-28 (torch 2.11.0+cu130, transformers 5.6.2, tokenizers 0.22.2; the record itself
stores no versions). The rerun forms other batches, so padding lengths differ, and it may run on CPU. With attention
masking, padding changes only the shapes, hence the float32 summation order, and the device changes the kernels. The
rule check measured both on this very sample before this file's commit: a CPU rerun differed from the stored CUDA
values by at most 4.470348358154297e-06 (mean 3.49e-08, no entry above 1e-5), and batch 256 against batch 64 on CPU by
5.66e-07; 1e-4 keeps about 22 times that headroom. The errors this item guards against (another model or revision,
softmax for sigmoid, misaligned rows, half-precision inference) move probabilities by more than 1e-4; the same sample
shifted by one row differed by 0.9599. Truncation is not exercised: no scorer-train caption exceeds 58 tokens (0 of
183,694 over 64), and the longest in the sample has 41, so a max_length of 41 or more would pass. The same tolerance
applies to the re-derivation's spot check (§8).

**D4. The GE head and the GE placement.**
- *Recipe:* `run_told_oracle.fit_one_head`'s, with only the input swapped. Draw =
  `numpy.random.default_rng(0).choice(scorer_train, 60_000, replace=False)` (`run_checks.PROBE_SEED`,
  `run_n6.HEAD_ROWS`); rest = `numpy.setdiff1d(scorer_train, draw)`; check rows =
  `numpy.random.default_rng(1).choice(rest, 10_000, replace=False)` (`run_n6.CHECK_ROWS`); labels lab =
  `run_told_oracle.global_labels(partition_L, scorer_train, 308_723)`; `LogisticRegression(C=1.0, max_iter=300)` with
  scikit-learn 1.6.1's other defaults (lbfgs, multinomial loss, tol 1e-4, intercept, no class weights), fitted on
  (X[draw], lab[draw]).
- *Input X:* a float32 (308,723, 28) array, NaN except on scorer-train rows (`X[scorer_train[i]] = affect_probs[i]`) and
  on selection rows (`X[rows] = probs` of the GoEmotions file); the raw probabilities, no normalisation, no scaler. The
  draw and check rows read only scorer-train rows (asserted finite). The draw's SHA-256
  (`run_gonogo.sha_array(numpy.sort(draw))`, as `fit_one_head` records it) must be told_oracle.json's arm L
  `draw_rows_sha256` 7be956c09bf716547df20264435388bd3645ae963df728636774e80359cdef5c.
- *One placement function* serves the CLIP heads (§5 item 3) and the GE head: given a full-row feature array, a
  transform (`run_checks.unit` for CLIP features; the identity for X), lab, scorer_train, the selection rows and
  max_iter, it does exactly what `fit_one_head` does for one modality (draw, check rows, fit on transform(F[draw]),
  `predict_proba(transform(F[rows]))` scattered into a NaN float32 (308,723, K) array, held-out accuracy
  100·`clf.score(transform(F[check]), lab[check])`) and returns the posterior, `classes_`, `n_iter_` and the accuracy.
  `classes_` must equal 0..40 (asserted), so columns are in `partition_L` label order.
- *Fallback, fixed before any episode number:* if the fit's `n_iter_` reaches its max_iter (300), the GE head is refitted
  once with max_iter = 3,000, everything else equal (same draw, input, C, tol and solver); both iteration counts and the
  use of the fallback are recorded, and the refit is the GE head. If the refit's `n_iter_` reaches 3,000, no GE
  posterior is written and the work stops and goes to the user. *Why:* 300 is the recipe's own cap, under which the CLIP
  heads converged (refitted with the same draw when this file was written: 176 iterations for the caption head, 154
  for the image head, their posteriors equal to `fit_one_head`'s); the fallback changes only where lbfgs stops, never
  the model. The fallback applies to the GE head only; §5 item 3 always uses max_iter 300.
- *Output (the GE placement Q_GE):* `predict_proba` on the 32,413 selection rows of X, float32 in `partition_L` order,
  NaN outside selection; rows sum to 1 within 1e-5 (asserted). Stored as `cache/r5_ge_posterior.npz` (`post_sel`
  float32 32,413 × 41, `rows`, `classes`), its SHA-256 written to the run log and committed as a constant of
  `r5_common.py` before any later step; Q_GE is rebuilt from it by scattering `post_sel` into rows. The GE head and Q_GE
  do not depend on the episode seed and are reused on every seed.
- *Placement quality* (§5): the GE head's held-out accuracy on the same 10,000 check rows, beside the CLIP caption head's
  35.72 and the image head's 9.81 (told_oracle.json arm L; majority share of the check rows 6.41, uniform
  2.4390243902439024).

**D5. Placements in the bundle.** For a placement Q (D4's shape and finiteness asserted): post_Q is a **new** dict
{"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"], "caption": post["caption"]} in A0 order.
Nothing is assigned into `bundle.post` or into round 3's process cache of the affect heads (`r3_bundle._HEADS`, whose
dict is shared by every bundle of the process); after each extension, `bundle.post["affect"]["txt"]` and the cached
`_HEADS` entry are asserted unchanged, by identity and by value (a fingerprint taken before and after). Then, exactly
as round 3's `build_bundle` forms its own fields (r3_bundle.py lines 207, 209 and 210):
- stack_Q = `common.grouping_stack(post_Q, ep, A0)` (s_affect(q, k) = p_img(q)·Q(k) for an image query and
  Q(q)·p_img(k) for a caption query, p_img being the affect image head's posterior); its image and caption slices
  equal the bundle's `stack` exactly (asserted);
- F_Q = `rb_eval.seed42_features(SimpleNamespace(ctx=ctx, post=post_Q), A0)[0]`; columns 6 to 17 (image and caption)
  equal the bundle's F exactly (asserted); columns 0 to 5 are the affect S, C, Δ, the two spreads and the arg-max match
  share; Δ^b = −Δ^a (asserted);
- B′_Q = `crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post_Q, ep, A0), ctx.parity)[0]` and its
  per-anchor arrays; condition-free and with condition gain 0 on every episode (asserted).

With Q = Q_CLIP these are the bundle's own stack, F and B′(A0) (§5 item 4). **B′_G** = B′_{Q_GE}. B (E2's k-means heads)
and B′(A1) (round 4's D4, built from the CLIP placement) are not rebuilt. The GE extension never changes a field of
round 3's or round 4's bundle (asserted by `r4_bundle.r3_fingerprint`, the same fingerprint over round 4's A1
fields and a fingerprint of the whole of `bundle.post`, csd included, before and after; `bundle.post["affect"]` is the
same dict object as `r3_bundle._HEADS[60000]["post"]`, which is why nothing may be assigned into it).

**Positive check** (with Q = Q_GE, in §5 item 5, after release): the affect slice of stack_G equals, exactly, an
independent recomputation from Q_GE and the affect image posterior p_img, `einsum('nc,nkc->nk', p_img[anchor],
Q_GE[candidates])` for i2t and `einsum('nc,nkc->nk', Q_GE[anchor], p_img[candidates])` for t2i; it differs from the
bundle's affect slice on at least one episode; and F_G's columns 0 to 5 differ from F's on at least one episode. These
are boolean assertions; no value is printed.

**D6. The candidates.** Per episode and condition, the same for both directions; `reader` is round 3's
`r3_fusion.reader(SimpleNamespace(F=..., stack=...), readers=bundle.readers)` (the frozen A0 half-readers, P^c, T^c =
`common.expected_term(stack, P)`, m^c, π^c):

| Candidate | Reader input F | Term stack | Thresholds | Gate g_t^c | Term T^c |
|---|---|---|---|---|---|
| **G-T** | F (CLIP placement): P^c, m^c, π^c are AFF's | stack_{Q_GE} | τ_0..τ_3 (D1) | `r3_fusion.gates_aff(m, π, τ)`: AFF's gate exactly (asserted) | Σ_h P^c(h)·s_h on stack_{Q_GE} |
| **G-TF** | F_{Q_GE}: P′^c, m′^c, π′^c | stack_{Q_GE} | τ′_0..τ′_3 | `r3_fusion.gates_aff(m′, π′, τ′)` | Σ_h P′^c(h)·s_h on stack_{Q_GE} |

τ′ = round 1's `rc_core.thresholds(m′)[0]` on seed 42 (its second value, the count, must be 24,576): `numpy.percentile` (linear) at 0, 25, 50 and 75 of the 24,576
seed-42 margins of G-TF, condition a's 12,288 first, in float64, the method that produced R1's `rc_tau.json`. τ′ is
written to `results/dev_seed42.json` and read from it on every later seed, that file's SHA-256 being recorded in
`results/carry.json` and asserted; it is never recomputed on a
test seed. On seed 42 τ′_0 is G-TF's smallest margin, so its τ′_0 gate is open on every affect pick there. π′^c = arg
max_h P′^c(h), ties to the first grouping in A0 order (`numpy.argmax`). Every gate is a float32 0/1 array per condition
(asserted). The gates read only the reader's outputs and treat both conditions alike, and the GE placement reads only
the caption (the task's own query or candidate), so both candidates are label-free.

**D7. Family, counterpart and cross-fits per candidate.** Round 3's D8 and D9 with the candidate's T^c and gates: the
224 cells (4 τ indices × 7 λ_u × 8 λ_a, cell number (t·7 + u)·8 + a), z(B) and z(T^c) per ranking row before any gate,
the gated term g_t^c·z(T^c), the fused score z(B) + λ_u·z(B) + λ_a·(g_t^c·z(T^c)) in float32, the nested control σ*, the
fused reader's integer min-margin cross-fit, and the matched counterpart G_cf,t = (g_t^a·z(T^a) + g_t^b·z(T^b)) / 2 from
the candidate's own term and gates, with its integer max-R@1 cross-fit; ties to the lowest cell number. Computed by
round 3's `r3_fusion.run_family(bundle, T, gates)`, which reads only B, the parity halves, T and the gates. The
development step asserts that the gates each family was run from equal, at every τ index and condition, the gates
recomputed from D6 (G-T: AFF's gates; G-TF: from τ′ and its margins and picks), and stores those gates.

**D8. Comparators, the bar comparator and margins.**
- Each candidate's condition-free comparators: B′_G, B′(A0), its own matched counterpart and B. External baselines:
  cosine and RCA (`per_anchor_seed{s}.npz`, round 3's D12).
- *Bar comparator* = whichever of those four has the largest mean R@1 over the episodes considered (full precision);
  ties to the earliest in the order B′_G, B′(A0), counterpart, B (round 4's pattern: the round's own floor first). Chosen
  once per scope (seed 42; each test seed; the pooled test seeds), as round 3's D12. A per-pair bar margin uses the
  comparator of the scope it breaks down.
- *Margin*, *bar margin* and *gain statistic* as round 3's D12, for the candidate.
- Keeping both B′(A0) and B′_G in the set can only raise the bar comparator's mean, hence clause 1's bar-margin point;
  it does not necessarily raise clause 2, since the lower bound against the max-mean comparator can be higher than
  against a comparator with a noisier pairing. The test checks every comparator separately (§6.5), so it is unaffected.
- *Beside* (descriptive, seed 42 and after the verdict): B′(A1)'s mean R@1 and each scorer's R@1 minus B′(A1), paired per
  anchor, with its interval. On seed 42 AFF minus B′(A1) is 0.33162434895833337 [0.048231414333532084,
  0.6246158772581268] (round 4's `dev_seed42.json`, `beside_aff`).

**D9. Paired difference against AFF.** As round 4's D9: per episode, d_k = R@1 of candidate k's fused reader minus R@1 of
AFF's fused reader; **Δ_k** = Σ over seed 42's 12,288 episodes of 4·d_k, summed from integer per-episode values (round
2's `r2_fusion.as_int4`, which asserts multiples of 0.25), never from float means; its point is 100·Δ_k / (4·12,288) pp
with its 95% anchor-painting interval. "Beats AFF on seed 42" means Δ_k > 0 (integer comparison).

**D10. Development bar** (round 4's D10 with D8's bar comparator), on seed 42: (1) bar margin point at least +0.5; (2)
bar margin lower bound above 0; (3) gain statistic lower bound above 0.

**D11. GE-placement results and the guard.** A *GE-placement result* is any array or number computed from Q_GE on
episodes or on selection pairs: stack_{Q_GE}, F_{Q_GE}, B′_G and its per-anchor arrays, the readers' outputs on
F_{Q_GE}, τ′, any gate or open count of G-T or G-TF, any family statistic, per-anchor array, chosen cell, R@1, margin,
gain, either rate, bar margin or Δ_k of G-T or G-TF, and any AUC or pair statistic computed with Q_GE. The GoEmotions
file, the GE head, Q_GE itself and the GE head's held-out accuracy are not GE-placement results. **No GE-placement
result is computed, written or printed before §5 items 1 to 4 have passed in order and `results/regression_check.json`
records them as passed.** The guard is not opt-in: every function of this round that can receive a placement takes a
placement object whose kind is `clip` or `ge` (a `clip` object is built only from the bundle's own Q_CLIP, asserted
equal by value; a `ge` object only from the GE file, SHA-256 asserted) and itself refuses a `ge` object until the guard
is released, whatever calls it. The measured diagnostics of
§5 are further refused until `results/carry.json` exists (or the boundary continuation of §8 has written it).

**D12. Inputs.** Read only; never written or modified. Scripts assert every SHA-256 before using the file; a mismatch
stops the run. Every input of round 3's D15 table and of round 4's D11 table that this round's imported code reads
applies with its SHA-256 (round 3's and round 4's code assert them itself; this round's code asserts them again before
its first bundle call, §8). Round 1's `common.verify_inputs()` runs on every seed. In addition (paths under the
repository root unless absolute):

| File | SHA-256 | Used for |
|---|---|---|
| `docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md` | bc7384027d856102ab5efc8eb24a1a7831a4ce33e660f356a66628a2dad184ee | the spec (provenance) |
| `src/test/20261122_round4_aff_vetoes/DECISION_RULE.md` | cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b | round 4's rule (part of this file by reference) |
| `src/test/20261122_round4_aff_vetoes/r4_common.py` | d1d4b819868aaa21603438fc7121151cd06639ec0318b610959340a989518fbb | imported (constants; sets `r3_common.TEST_SEEDS`) |
| `src/test/20261122_round4_aff_vetoes/r4_bundle.py` | 2fc452800a14768bbb81efbb0728113ce10aa4f281a4c5910252eb5c4bd634e2 | imported (`build_bundle`, `compare_a1_with_round1`, `r3_fingerprint`, the guards) |
| `src/test/20261122_round4_aff_vetoes/r4_stats.py` | b308ca43afc3e8accd35894a3c555f551c5c17db0e90d330bab8669b68f2d271 | imported (`bar_comparator` only) |
| `src/test/20261122_round4_aff_vetoes/r4_fusion.py` | 9b9b12e388dd41a01db5c6aea4277af4710d4cd4990adfefc918b3e0ccbd961f | provenance (not imported) |
| `src/test/20261122_round4_aff_vetoes/run_r4_seed42.py` | 74f6ebe0a97f83322fd2858ee61a95ac9033edd33693bea192db1c6de6149eda | provenance: the Guard and Recorder pattern (not imported) |
| `src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz` | 72fb827fa9b44360d237dd3a3d555403825cf3847282976d24073dcc12ee0c75 | R1's and AFF's seed-42 arrays, gates, cells, σ* (§5 items 1 and 4) |
| `src/test/20261122_round4_aff_vetoes/results/dev_seed42.json` | fd7b3f480d5997284f9d8cfe29fe25275ff3edd38c1ada0a9392d8396ba9e01f | AFF minus B′(A1) on seed 42 (§5 item 1) |
| `src/test/20261122_round4_aff_vetoes/.gitignore` | 65c8bc8600f811509b0d8325f4aec4d53bd0c35639e6400c460e2445bddbdbc9 | copied as this folder's `.gitignore` |
| `src/test/20261122_round4_aff_vetoes/rule_check/opus_rule_check.md` | 8c550feae32f30b954531abc563bf21cc486be7d094b38d30bc55aa3782e113c | provenance |
| `src/test/20261122_round4_aff_vetoes/final_review/final_review.md` | 5c68260241011d9e2d8137bf8d5069f6f32f50b9c30a024f8f64573df400224b | the before-reuse items (§8) |
| `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md` | 53ecffe87f9f8927f46c8755bd44053d6aa34feb34bf372c886bec52b68d5334 | provenance (its §7 and §8.4: the lapses closed in §8) |
| `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md` | 0aeb88506c6aa988a82776bd9aea5261e544a92f4b203ebc0e056142ce72fac2 | provenance (idea 3, §3.3) |
| `src/data/affect.py` | d37e306c9b8673068f6d74e74d7eff61f649ebdb53088f0d8ad7f592e17f3d33 | GoEmotions model and call (D2) |
| `src/data/artelingo.py` | 623b7b02eb03b7a81ecf246b120bb7f49e1b929194830faeebcd0d64fa3c89e1 | `join_captions`, `ANNOTATIONS_PATH` (D2, D3) |
| `src/data/artelingo_splits.py` | f130950355487b8537ba0c0b1e35230a08e2e07d3d0fddd4c3e5aad240886c13 | scorer-train, selection and held rows; aspect labels (§5 item 4) |
| `/data/PDD/artelingo/artelingo_train.json` | 6a4e5b17feecc2c3b54cd166416b75f1bbffd56bec1523f4173b245edb8e894d | the captions (D2, D3) |
| HF snapshot `models--SamLowe--roberta-base-go_emotions/snapshots/d75048347613a25d77de8cf6412eaae9fa7b26be/` (under `/data/SSD2/HF_home/hub/`): `model.safetensors` | 84d6d338b4cf63f0ed3c990a0ce748d32d1d2965c072f4645accaa71af3888c0 | the GoEmotions model (D2) |
| same snapshot: `config.json` | 3d4ef8e1465958e169761e2eb09d6e2c8d8806216973691ac40e405c97339d5c | the GoEmotions model (D2) |
| same snapshot: `tokenizer.json` | 90e2336a1cdacffe5d4328ab323aa9e5c33889026e4e4881323bebdeeb0e179d | the tokenizer (D2) |
| same snapshot: `tokenizer_config.json` | 6735f2f38dc5399eb76a2c20dcba3ef27a9b2fbba0d05b6e2966038f28aefcf9 | the tokenizer (D2) |
| same snapshot: `vocab.json` | ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e | the tokenizer (D2) |
| same snapshot: `merges.txt` | fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862 | the tokenizer (D2) |
| same snapshot: `special_tokens_map.json` | 06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f | the tokenizer (D2) |
| `src/test/20261018_affect_factor_learning/cache/affect_prepare.npz` | e25d2dcadadc23b33ac94659fb44c13e220110e64ce463def07b04343bfc4f2e | scorer-train GoEmotions probabilities (D3, D4) |
| `src/test/20261018_affect_factor_learning/cache/affect_prepare.json` | 8734bdac0ec49b07ddd53caa5aa088dd4d51d0daba9c50730b359c98395316d1 | its record (device, `affect_probs` SHA-256) |
| `src/test/20261018_affect_factor_learning/run_affect.py` | ef04c7c24810d7550a791c8191162de0664d241d5414450f849e28a847f88c65 | provenance of `affect_prepare` (batch 256, max_length 64) |
| `src/test/20261018_affect_factor_learning/run_posthoc_affect.py` | 07da2767706964f36cdf0ca282e2298b61414e6e86ed1afbaaa18862a0fb12d8 | provenance of the selection join (D2) |
| `src/test/20261111_community_told_oracle/run_told_oracle.py` | b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d | imported (`fit_one_head`, `global_labels`, `pair_stats_heads`, `roundtrip`) |
| `src/test/20261108_new_method_quick_checks/run_checks.py` | 5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610 | imported (`unit`, `PROBE_SEED`) |
| `src/test/20261108_new_method_quick_checks/run_n6.py` | 57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0 | imported (`HEAD_ROWS`, `CHECK_ROWS`) |
| `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` | 8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc | imported (`EvalContext`, `sha_array`) |
| `src/data/wikiart_genre.py` | aef1d35978305b9bc00f84897a4d8a999b729dac8bc85a3237dc01b5e00c5448 | the genre codes of `artelingo_aspect_labels` (§5 item 4) |
| `src/test/20261120_r1_levers_brainstorm/bs_09_direction.py` | 74f3fa4c88588832cd681b1cbb793b7bf6982748477df9147acdf6141346de1f | provenance of the per-direction diagnostic (not imported) |
| `src/test/20261120_r1_levers_brainstorm/results/bs_09_direction.json` | 0882262340518347d59e825f9e5444d5042f75698d2b9668f36eac2163e7fc8e | orientation for the per-direction diagnostic |
| `src/test/20261120_r1_levers_brainstorm/bs_cache.py` | 62b6d9baa534a04a7fc5365e7a2fea8f47e8719ec29890d9cafe7ae24fd3b07e | provenance: R1's P in `bs_07_detector.py` is round 2's `cand_R1_A0.npz` |
| `src/test/20261118_reader_fix_round2/results/cand_R1_A0.npz` | c707af6a101bf3c0e49ee6ce509cf31d82bfa54ee33597dbece2ec6261941407 | provenance of 0.787 (reproduced from these probabilities when this file was written) |

Already in round 3's D15 or round 4's D11 and used here as stated: `told_oracle.json` (76d9ec89…70d2: the CLIP heads'
record, the pair statistics), `per_anchor_told_oracle.npz` (27e81011…6af1366: `partition_L`), round 1's `rc_core.py`
(e649fff5…185c: `thresholds`) and `rc_tau.json`, `bs_07_detector.py` and `bs_07_detector.json` (1b287bcc…088a,
e9d52462…85e1: the AUC definition and `results.R1aff.auc_emotion`), `baselines_seed{42,43,45,47,48,49,50,51}.json` and
`codes_provenance.json` (8e6a517b…bfaf). This round's own files `cache/r5_goemotions_selection.npz` and
`cache/r5_ge_posterior.npz` are inputs of every step after the one that wrote them, by the SHA-256 constants of D2 and
D4. The pickles were written with scikit-learn 1.6.1 and numpy 2.2.6; the run uses the same versions.

## 4. The pipeline

One code path for the episode work, a function of the episode seed s (and of `smoke` for the wiring smoke test, §10),
used unchanged on seed 42 and on the test seeds; two one-time steps before it.

1. **GoEmotions step** (once, D2; §5 item 2 inside it). Writes the GoEmotions file and its record.
2. **Placement step** (once, CPU): §5 item 3, then the GE head (D4) with its fallback, Q_GE and its held-out accuracy (the
   first number of this round, computed before any episode number) to `cache/r5_ge_posterior.npz` and
   `results/placement.json`. It uses `EvalContext(42, False)` only for the data, the row sets and `fit_one_head`'s
   context, and reads none of its episode fields.
3. **Bundle:** round 4's `r4_bundle.build_bundle(s, smoke)` (round 3's bundle under the seed guard, then round 4's A1
   extension, which supplies B′(A1)), then this round's GE extension (D5) with Q_GE: stack_G, F_G and B′_G with its
   per-anchor arrays. The seed guard admits 42, 52, 53 and 54 and, with smoke, 9001 to 9003: this round's code sets
   `r3_common.TEST_SEEDS = (52, 53, 54)` in its own process before the first bundle call and asserts that round 4's
   `r4_common.TEST_SEEDS` is the same tuple (round 4's import sets the same value). No other round-3 or round-4 function
   that reads `TEST_SEEDS` or `EARLIER_SEEDS` is called. The smoke flag is never passed to the head files, the readers or
   the placement step. Per-seed caches: round 4's two cache files and a GE file beside them bound to both by SHA-256.
4. **External baselines:** as round 3's §4 item 2.
5. **Readers:** AFF's (which are G-T's) on F; G-TF's on F_G (D6). On a test seed, G-TF's reader, and F_G (item 3),
   only if G-TF is carried; stack_G and B′_G always (B′_G is a GO comparator).
6. **Gates:** AFF's at τ_0..τ_3; G-T's (equal to AFF's, asserted); G-TF's at τ′ (on seed 42 after τ′ is computed). On a
   test seed, only AFF's gates and the carried candidate's gates are computed.
7. **Families:** for AFF and for each candidate needed by the step (§5, §6.4): D7.
8. **Records:** per seed and scorer, the chosen cells of both cross-fits on both tune halves (cell number, τ index and
   value, λ_u, λ_a), σ*, the per-anchor arrays (`r1`, `gain`, `other`, `swap`, `strict`) of each fused reader and
   counterpart computed, the gates each family ran from, and B, B′(A0), B′_G, B′(A1), cosine and RCA per anchor. What may
   be computed before the test verdict is limited by §6.4.

## 5. Seed 42: the regression checks, the development step, the measured diagnostics

Items 1 to 4 are the regression checks. Items 2 and 3 need no episode and run inside the GoEmotions and placement steps
(§4 items 1 and 2) before item 1; the seed-42 runner then records items 1 to 4 in order: item 1 computed, items 2 and 3
re-asserted from their records (file SHA-256s equal to `r5_common.py`'s constants, `passed` true, the stated sample,
tolerance and rows; the GoEmotions file's `rows` equal to `ctx.selection` and its `sample_ids` to
`ctx.data.sample_ids[ctx.selection]`), item 4 computed. No GE-placement result (D11) is computed, written or printed
before items 1 to 4 pass. If any check fails, no further number is written, and the work stops and goes to the user with
the cause traced.

**Item 1. Bundle, R1, AFF and B′(A1).** Through round 3's and round 4's own functions, against round 1's
`common.load_bundle()` loaded once with its console output discarded:
- the bundle passes round 3's `r3_bundle.compare_with_round1` (the D7 redundancy check included) and round 4's
  `r4_bundle.compare_a1_with_round1`; B′(A1)'s mean R@1 is 18.804931640625 exactly; the cosine and RCA arrays of
  `per_anchor_seed42.npz` pass round 3's §4 item 2;
- R1 = round-1 R-c and AFF = round 3's targets, exactly as round 3's rule §5 items 2 and 3 and round 4's §5 item 2 state
  them, computed with round 3's `r3_fusion.reader`, `gates_r1`, `gates_aff` and `run_family`: R1's T^c, margins and
  picks equal `cand_Rc_Rb_expected_A0.npz`, τ recomputed equal to `rc_tau.json`, cells 116, 119 / 58, 123, σ* 0 / 0, bar
  margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852], gain statistic 2.667236328125
  [2.325087836946873, 3.012361650695922]; AFF's fused R@1 19.136555989583336, counterpart 18.39599609375, bar comparator
  B′(A0), bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968], margin against the counterpart
  0.7405598958333333 [0.5196896694963071, 0.9598857494832738], gain statistic 3.110758463541667 [2.780005709854805,
  3.4559584315470384], either change −1.629638671875, per-pair bar margins 0.9765625, 1.45263671875 and
  −0.32958984375, cells 39, 119 / 149, 10, σ* 0 / 0, AFF minus R1 0.21769205729166666 [0.06425880757348419,
  0.3709597330984391] (fused R@1) and 0.25634765625 [0.04280778303598444, 0.46195041633015954] (bar margin), τ_0 open
  counts 9,941 / 3,627;
- R1's and AFF's per-anchor arrays (fused and counterpart, five metrics), gates at the four τ indices (float32), chosen
  cells and σ* equal round 4's `results/seed42_arrays.npz` (keys `r1_*`, `aff_*`) exactly;
- AFF minus B′(A1) equals 0.33162434895833337 [0.048231414333532084, 0.6246158772581268] (round 4's `dev_seed42.json`).

The brainstorm scored in float64 and this pipeline in float32 (round 3's D8); round 3 and round 4 reproduced every
number above through the float32 path, so a difference is a code difference, traced before anything else runs.

**Item 2. GoEmotions** (in the GoEmotions step, D2 and D3): the regression sample reproduces `affect_probs` within
max-abs 1e-4, and the selection join gives 32,413 non-empty captions in the order of `ctx.selection`, under D2 item 1's
row assertions.

**Item 3. The placement function = the CLIP heads** (in the placement step, D4): fed the CLIP caption features
(`ctx.data.txt_features`, transform `run_checks.unit`) with lab, scorer_train and max_iter 300, the placement function
returns a posterior equal to `run_told_oracle.fit_one_head`'s `"txt"` posterior of the same process exactly (shape
(308,723, 41), float32, values and NaN pattern, `numpy.array_equal(..., equal_nan=True)`) and held-out accuracy 35.72;
fed the image features, it returns `fit_one_head`'s `"img"` posterior exactly and 9.81; `classes_` = 0..40 for both;
`fit_one_head`'s record equals told_oracle.json arm L's `head` exactly (`n_classes` 41, `draw_rows_sha256`
7be956c0…cdef5c, the two accuracies, `check_majority_share` 6.41, `uniform` 2.4390243902439024); and the draw positions
in scorer-train order satisfy `scorer_train[pos] == draw` (the mapping D4 uses for the GE input). The CLIP heads'
`n_iter_` are recorded (descriptive; 176 and 154 when this file was written).

**Placement quality** (after item 3, before any episode number; descriptive, decides nothing): the GE head's held-out
accuracy, `classes_`, `n_iter_` (and the fallback's, if used), its check majority share and uniform rate, beside 35.72 and
9.81, written to `results/placement.json` with this file's SHA-256 and the Amsterdam time.

**Item 4. The CLIP placement through this round's code.** With Q = Q_CLIP passed, as a `clip` placement object, to the
functions that later receive Q_GE:
- stack_Q, F_Q and B′_Q (scores and per-anchor arrays) equal the bundle's stack, F and B′(A0) exactly;
- G-T's and G-TF's reader outputs (P, T, m, π) equal AFF's of item 1 exactly; τ′ computed from those margins by the
  same function equals τ_0..τ_3 of D1 exactly (each element, `==`); their gates equal AFF's exactly at every τ index and
  condition;
- their families give AFF's cells (39, 119 / 149, 10), σ* (0 / 0) and per-anchor arrays exactly (equal to item 1's and to
  `seed42_arrays.npz`'s `aff_*`);
- their development records, through this round's record function (D8 to D10), give AFF's numbers of item 1 exactly,
  with bar comparator B′_Q (its arrays equal B′(A0)'s, and it comes first in D8's tie order), all three D10 clauses true
  and Δ_k = 0 against AFF (expected here; the boundary rule of §8 applies to items 5 to 7 only);
- the pair-lift code, `run_told_oracle.pair_stats_heads(Pi, Pt, labS, gS)` with Pi and Pt the bundle's affect image and
  caption posteriors on the selection rows cast to float64, labS the emotion, style and genre codes of
  `artelingo_aspect_labels(ctx.data)` on the selection rows and gS `ctx.groups[ctx.selection]`, reproduces told_oracle.json
  `arms.L.pairs.heads` exactly as JSON round trips (among them `ratio_same_over_diff` 1.1445184466303795 and the contrast
  ratios 1.070827615209179, emotion × style, and 1.0375160447836334, emotion × genre; the whole
  dict, 13 values, was reproduced when this file was written);
- the AUC code of the diagnostics (§5, diagnostic (a)) applied to AFF's P^c(affect) of item 1 gives
  0.7870951145887375 (`bs_07_detector.json`, `results.R1aff.auc_emotion`) exactly; it was reproduced when this file was
  written from round 2's stored R1 probabilities (`cand_R1_A0.npz`), which the brainstorm read.

**Item 5. Development numbers** (only after items 1 to 4 pass and the regression record is written). First D5's
positive check (boolean, nothing printed). With Q_GE: stack_G,
F_G, B′_G; for G-T and for G-TF: the reader outputs, τ′ (G-TF), the gates and their τ_0 open counts per condition
(integers), the family (D7) with the gate check of D7, fused and counterpart R@1, the chosen cells and σ*, the bar
comparator (D8) with the four comparators' means, the bar margin, the margin against the counterpart and the gain
statistic with their intervals, the either change against the counterpart, Δ_k (D9) with its point and interval, the
clauses of D10, per-pair bar margins (descriptive); B′_G's mean R@1 and B′_G minus B′(A0), paired (descriptive); B′(A1)
beside (D8). Written to `results/dev_seed42.json` with this file's SHA-256 and the Amsterdam time, and the arrays (per
scorer as §4 item 8, τ′, the readers' P, m and π for AFF and G-TF) to `results/seed42_arrays.npz`.

**Item 6. Development bar:** D10, per candidate.

**Item 7. Carry.** Let E be the set of candidates that clear the development bar and have Δ_k > 0, and M the largest
Δ_k in E. Every member of E with M − Δ_k ≤ 24 is tied (the integer literal 24: 0.05 percentage points × 4·12,288 / 100
= 24.576, so a gap of 24 is 0.048828125 pp and a gap of 25 is 0.0508626302 pp). The carried candidate is the tied member
that comes first in the order G-T, G-TF. Written to `results/carry.json` with the Δ_k, the D10 clauses, E, M, the
tied set and the SHA-256 of `results/dev_seed42.json` (which holds τ′). The console line reads "CARRY <name> (pending the phase-1 agreement, rule §8)".

**Item 8. Kill.** If E is empty, no test seed is built, the seed-42 results go to the user, and AFF stays the current
best; the user decides what follows. The console line reads "KILL (pending the phase-1 agreement, rule §8)"; the log
records the kill only after the agreement.

No other variant is computed on seed 42.

**Measured diagnostics** (seed 42, descriptive, decide nothing; computed only after `results/carry.json` is written,
written to `results/diagnostics_seed42.json`), in this order:
- **(a) Detection AUC of G-TF's P′^c(affect)** by `bs_07_detector.py`'s definition: positives are condition a of
  emotion × style and emotion × genre, negatives the other four (pair, condition) values, the scores of condition a's
  12,288 values then condition b's, 24,576 pooled, `sklearn.metrics.roc_auc_score`; beside AFF's (R1's) 0.7870951145887375.
- **(b) Feature-level AUC of Δ_affect** (feature column 2 of each condition), on the same sets: F_G's against F's.
- **(c) Emotion pair lift through the placement:** `pair_stats_heads` as in item 4 with Pt = Q_GE on the selection rows
  (float64), against the CLIP values of item 4 (`ratio_same_over_diff` and the two contrast ratios); beside the group lift
  2.7111312041209863 (told_oracle.json `arms.L.pairs.groups.lift.lift`).
- **(d) The sharper term:** for AFF, G-T and G-TF, from the fused and counterpart scores re-assembled with the cells their
  cross-fits chose (the assembly of `run_family`; `per_anchor` of the re-assembled scores must equal the family's
  arrays exactly): per aspect pair and condition c, pooled over the pair's 4,096 episodes, fused minus counterpart in
  R@1^c, gain^c and either^c (R@1^c = mean over the two directions of 1[the target of c ranks strictly first], other^c
  the same for the other aspect's candidate, gain^c = R@1^c − other^c, either^c = R@1^c + other^c), with intervals; the
  same per direction (image query i2t, caption query t2i) pooled over pairs and conditions, as `bs_09_direction.py`'s
  `per_dir` defines it (orientation: AFF's fused minus counterpart R@1 there, `bs_09_direction.json` keys `AFF.i2t`
  and `AFF.t2i`, 0.5655924479166714 and 0.91552734375; R1's are +0.216 and +0.671); each candidate minus AFF on these, paired; the
  chosen τ index, λ_u and λ_a of both cross-fits on both halves; and the either cost per unit of gain, −(either change
  against the counterpart) / gain statistic, beside AFF's 0.5238718116415958 (1.629638671875 / 3.110758463541667).

## 6. The test (the only confirmatory step; only if a candidate is carried)

1. **Sensitivity, after the carry and the phase-1 agreement, before any test-seed code runs** (round 3's §6.1 formula,
   unchanged). For each GO check of item 5, take the carried candidate's seed-42 per-episode difference (its fused R@1
   minus the comparator's; its fused gain minus 0 for the gain statistic; its fused gain minus RCA's gain; its fused R@1
   minus AFF's fused R@1) and project the pooled standard error over three seeds; write SE, 1.96·SE, the **detectable
   margin x = 2.80·SE** and the seed-42 bootstrap half-width to the log and to `results/sensitivity.json`. Used only to
   read a failed check (item 7); it never stops the round.
2. **Build.** As round 4's §6.2 (and round 3's): seeds 52, 53 and 54 (free in `docs/superpowers/episode_seed_ledger.md`),
   once each, in one invocation of this round's build runner, each with
   `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, without `--overwrite`. The 9 new per-pair
   episode SHA-256s must differ from each other and from every per-pair SHA-256 in
   `baselines_seed{42,43,45,47,48,49,50,51}.json`; on a match the test stops and goes to the user.
   `codes_provenance.json` keeps 8e6a517b…bfaf before and after each build. Build logs go to `results/build_seed{s}.log`
   and are not opened before the verdict; build records `results/build_seed{s}.json` are asserted by every later step; a
   crashed build is handled as round 3's §6.2 says. After the hash check the seed ledger gets its rows: seeds 52, 53, 54
   (test, round 5, idea 3), 55 and later free.
3. **Frozen from seed 42:** the GoEmotions file, the GE head and Q_GE, the A0 half-readers with their scalers, τ_0..τ_3,
   τ′_0..τ′_3, the affect restriction, the 224-cell family (λ grid, cell order, tie rules), the heads and posteriors, the
   recipes of B, B′(A0), B′_G and B′(A1), the method-A checkpoint. **Rerun on each test seed's own parity halves:** B,
   B′(A0), B′_G, B′(A1), the readers' probabilities on that seed's F and F_G, the gates (that seed's margins and picks
   against the frozen τ and τ′), σ*, and for the carried candidate and for AFF the fused reader's min-margin cross-fit,
   and for the carried candidate the counterpart's max-R@1 cross-fit.
4. **Order of computation.** On the test seeds, until the verdict is written to `results/test_verdict.json` (with this
   file's SHA-256 and the Amsterdam time), only these are computed and written: the per-seed bundles with the GE extension
   and the reader arrays of §4 items 3 to 6 (cached; a bundle holds B′(A1)'s arrays, which enter no comparison before the
   verdict); the per-anchor arrays of the carried candidate's fused reader and counterpart, of AFF's fused reader, of B,
   B′(A0), B′_G, cosine and RCA; the chosen cells and σ* needed to assemble them; and the pooled checks of item 5. AFF's
   counterpart is not cross-fitted, assembled or written; the candidate that was not carried is never computed on a test
   seed. The GO pass asserts, before it computes any check: (a) each seed's clusters equal `groups[anchor]` of that
   seed's episodes; (b) before each family call, the gates passed in equal, at every τ index and in both conditions, the
   gates recomputed from that method's definition (AFF: D1's τ with the seed's margins and picks on F; G-T: AFF's
   gates; G-TF: τ′ read from `dev_seed42.json` with the seed's margins and picks on F_G); (c) the build records' file and
   episode-hash checks of item 2 are re-run; (d) AFF's counterpart is not cross-fitted; (e) the placement used is Q_GE
   (the `ge` object of D11, the GE file's SHA-256 asserted, and Q_GE differs from Q_CLIP on the selection rows); (f) the
   arrays the AFF check receives as AFF's and as the candidate's are, by value, the ones AFF's and the candidate's
   family runs produced (each family run records the SHA-256 of its per-anchor arrays; the check compares the SHA-256 of
   what it receives). The open counts these assertions use are held in memory and are not written or printed before the
   verdict.
5. **GO** if, pooled over the three seeds (§2), every one of these nine checks has a 95% lower bound above 0:
   - R@1, the carried candidate's fused reader minus each of cosine, RCA, B, B′(A0), B′_G and its matched counterpart;
   - the gain statistic (D8), which is also the gain difference against cosine, B, B′(A0) and B′_G, counted once;
   - condition gain, the candidate's fused reader minus RCA;
   - **the AFF check:** R@1, the candidate's fused reader minus AFF's fused reader, paired per anchor.

   Points, intervals and pass or fail of every check are written to `results/test_verdict.json` with the verdict.
6. *(No secondary check in this round.)*
7. **NO-GO** if any check fails, including a partial pass, read as round 4's §6.7: a failed check with a pooled point
   above 0 is **inconclusive at a detectable margin of x** (its x from item 1, the realised pooled half-width beside it);
   at or below 0 it is "<candidate> did not beat <name> on fresh episodes", <name> being cosine, RCA, B, B′(A0), B′_G,
   the matched counterpart or AFF for the R@1 checks, "the condition-free comparators on condition gain" for the gain
   statistic and "RCA on condition gain" for the condition-gain check. If the AFF check is the only failed check, the
   NO-GO also reads "the candidate works, but no improvement over AFF was shown". In every NO-GO AFF stays the current
   best. For the AFF check, x may understate the margin needed (round 4's §6.7: in round 3 the analogous paired check
   realised a half-width 18% above its projection).
8. **Per-seed and per-pair results** are reported after the verdict and never change it; per-pair results are not
   tested.
9. **Frozen-cell line** (descriptive, after the verdict): on each test seed, the carried candidate's fused reader and
   counterpart scored with the cells seed 42's cross-fits chose for it (the cell chosen on seed-42 tune half h scores the
   test seed's episodes of parity 1 − h), and AFF's with its seed-42 cells (fused 39 and 119, counterpart 149 and 10),
   with the comparators of item 5.
10. **Claim licensed.** A GO shows that the carried candidate (AFF with the GoEmotions placement of captions in its term,
    and for G-TF in its reader's affect features) beats AFF and each comparator of item 5, pooled over the three aspect
    pairs, on new episodes drawn from the same 6,451 selection paintings, and so replaces AFF as the current best for the
    held-split paper test. It does not show transfer to new paintings (the held split stays reserved) or a margin on each
    aspect pair. For the paper: the groupings were built without evaluation labels; AFF and the candidates were developed
    on seed 42 (item 11).
11. **Disclosures** (reported with every number of the carried candidate). *Multiplicity:* AFF was found among about 50
    label-free variants read on seed 42 (round 3's §6.11); idea 3 was proposed from seed-42 observations (the brainstorm's
    §3.3, among them R1's margin of +0.216 with an image query and +0.671 with a caption query); the development family
    has two candidates and the carry takes the larger paired gain over AFF; τ′ is set on seed 42; the placement recipe
    was chosen by the user from three options before any number, not compared on data. The seed-42 numbers are therefore
    inflated; the fresh seeds 52 to 54 are the protection, and the frozen-cell line accompanies the test. *External
    model:* GoEmotions (SamLowe/roberta-base-go_emotions) is the affect grouping's own external source and is now applied
    at inference to every caption (a RoBERTa pass per caption). ArtELingo captions are the annotators' explanations of
    their emotion, so a caption's GoEmotions probabilities are close to its emotion label; the method reads the caption,
    which is the task's own input (query or candidate), and no evaluation label. In the v2 line this round is the first
    to pass selection captions through GoEmotions; an exploratory percept-branch pilot of 2026-09-22 had pooled
    GoEmotions over every `artelingo_train` caption, held rows included, and nothing of it is read here.

## 7. After the verdict (descriptive; decides nothing)

Computed only after `results/test_verdict.json` is written, from the cached per-seed arrays where possible:

1. Per-seed and per-pair results for the carried candidate (the measures of §6.5 per seed, and per pair pooled over
   seeds), its bar margin (D8, the bar comparator chosen per scope), its margin, gain and either change against its
   counterpart, its chosen cells per seed, and the frozen-cell line (§6.9).
2. **AFF's own seven checks** on seeds 52 to 54 (round 3's §6.5 list, its counterpart cross-fit run now), with its bar
   margin, per seed and per pair, labelled descriptive: a replication of round 3 on new episodes.
3. **B′(A1) beside AFF and the candidate:** B′(A1)'s pooled mean R@1, and AFF's and the candidate's fused R@1 minus
   B′(A1), paired, pooled, with intervals.
4. Gate open shares at each τ index, overall, per condition and per pair (label-free), for the carried candidate and
   AFF, per seed and pooled (for G-T these equal AFF's).
5. Nothing else is computed on the test seeds; the measured diagnostics of §5 are not repeated there.

## 8. Order of work, re-derivation, round 4's lapses, and timeline

- **Order:**
  1. This file checked by a fresh Opus reviewer, fixed, committed and sent to the user; the short implementation plan;
     the user chooses the execution method.
  2. Implementation of the seed-42 code (the GoEmotions runner, the placement step, the GE extension, the candidates,
     the records, the seed-42 runner, the diagnostics and the sensitivity path of §6.1) **with every test of §10 list A, all passing, before step 3**.
     The folder's `.gitignore` (a copy of round 4's, D12) is its first file: the repository's `.gitignore` ignores
     `*.json` but not `*.npz` here, and this folder's `cache/` and `results/` must stay untracked.
  3. The GoEmotions step (§4 item 1, item 2 inside it), launched by the main session; its SHA-256 committed to
     `r5_common.py`.
  4. The placement step (§4 item 2, item 3 inside it, then the placement quality); its SHA-256 committed.
  5. The seed-42 runner: input checks, items 1 to 4, the regression record, the guard released, items 5 and 6, then the
     boundary stop or items 7 and 8, then the measured diagnostics.
  6. The independent re-derivation, phase 1. Its development numbers and carry must agree before a kill is recorded and
     before the sensitivity step; its x (§6.1) is compared after the sensitivity step and before any test-seed code runs.
  7. If a candidate is carried: the sensitivity projection (§6.1; seed-42 arrays only); the comparison of x; then the
     test-seed code (the build runner, the GO runner, the rule application and the descriptive pass) **with every test
     of §10 list B, all passing**, which this file pre-registers, so it is not a departure; then the end-to-end wiring
     smoke test on seeds 9001 to 9003 (§10); the three seeds built (§6.2); the GO pass (§6.4); the re-derivation, phase
     2; the rule applied, the verdict written; the descriptive pass (§7).
  8. The whole-branch final review on the most capable model, one fix wave and a scoped re-review, **before** the report
     `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md` is committed with its `reports_sum.md` row.
- **Independent re-derivation.** An agent that has not written or read the implementation re-derives with its own code,
  in two phases, and writes only to `rederive/` of this folder. Phase 1 may start once this file is committed and the
  GoEmotions file exists. It first reproduces, with its own code, the targets of §5 item 1 (R1 and AFF), of item 3 (its
  own placement code on the CLIP features equals `fit_one_head`'s posteriors) and of item 4's τ and B′ parts, and runs
  the **CPU spot check** of the GoEmotions file: positions `numpy.sort(numpy.random.default_rng(6).choice(32_413,
  size=1_024, replace=False))` of the file's rows, their captions joined as D2 joins them, run on CPU through
  `src.data.affect` (batch 256, max_length 64), max-abs difference to the stored rows at most 1e-4 (D3's tolerance and
  reasoning); it also checks the file's `rows` and `sample_ids` against its own join. It computes no GE-placement result
  before these match. It then fits its own GE head (D4, with its own draw, mapping and scatter, calling scikit-learn's
  `LogisticRegression` directly) and re-derives Q_GE, the held-out accuracy, `n_iter_` and the fallback decision,
  stack_G, F_G, B′_G, every candidate's development numbers of item 5, the D10 clauses, the Δ_k, the carry of item 7, and
  the detectable margins x of §6.1 for the carried candidate. Phase 2 (after the GO pass, before the rule is applied): the
  hash check of §6.2; per test seed B, B′(A0), B′_G, the gates, σ*, the chosen cells of the carried candidate's two
  cross-fits and of AFF's fused cross-fit; every GO check of §6.5 (points, bounds, pass or fail). The measured
  diagnostics are re-derived by the final review, not by the re-derivation.
- **What the re-derivation may import** (and nothing else of this repository): the data loaders and frozen components
  round 3's rule §8 lists (`run_gonogo.EvalContext`, `run_checks.model_inputs` with `centered_term`,
  `run_n6.load_posteriors`, `run_told_oracle.fit_one_head` for the CLIP heads, `rb_build.load_readers("A0", False)`,
  `zscore_rows`, `crossfit_condition_free`, `uniform_probe_scores`, `cluster_bootstrap`); round 1's `common.load_bundle`
  (seed 42 only); `src.data.affect` (`load_goemotions`, `goemotions_probabilities`); `src.data.artelingo`
  (`join_captions`, `ANNOTATIONS_PATH`) and `src.data.artelingo_splits.artelingo_splits`; and the stored files of D12,
  this round's GoEmotions file among them (its SHA-256 read from `cache/r5_goemotions_selection.json` and the run-log
  line, not from `r5_common.py`, which is implementation code); and, in phase 2, each test seed's `run_baselines.py`
  outputs (`per_anchor_seed{s}.npz` for cosine and RCA, `baselines_seed{s}.json` for the hash check; episodes through
  `EvalContext`), each SHA-256 asserted against `results/build_seed{s}.json`. *Why this list:* these produce the inputs that are fixed before any choice of
  this round (episodes, CLIP heads, readers, B's ingredients, the bootstrap, the GoEmotions call and the captions);
  everything that turns the GE placement into a decision (the GE head's fit and scatter, the features, P, T, margins, τ′,
  the gates, G_cf, the 224 cells' integer statistics, σ*, both cross-fits, the assembly, the per-anchor metrics, the
  comparators, the checks, Δ_k, the carry and x) is new code. It must **not** import round 3's or round 4's re-derivation
  code (`rd3_*`, `rd4_*`, anything under their `rederive/` folders), any module of rounds 2 to 5 (`r2_*`, `r3_*`,
  `r4_*`, `r5_*`, `run_r*`), round 1's `common` functions other than `load_bundle`, `rc_core`, `rb_eval` or
  `rb_features`, or the brainstorm's modules. The controller checks the re-derivation's import lines against this list
  before the comparison and records the check in the log.
- **Blindness.** The re-derivation's results (any GE-placement result, the GE head's accuracy, the carry) stay out of the
  shared ledger, the run log and any file an implementer reads until the implementation's seed-42 runner is committed and
  its real run has written `results/carry.json` or its boundary record; until then the controller records only that
  phase 1 finished and the SHA-256 of its results file. It compares with the implementation's files only after its own
  results are written and hashed.
- **Agreement** means: every discrete quantity identical (picks, gates, chosen cells, σ*, the GE head's `classes_`,
  `n_iter_` and fallback decision, the bar comparator, Δ_k, the carry, each check's pass or fail); Q_GE identical
  (float32 values and NaN pattern); the GE head's held-out accuracy identical; τ and τ′ within 1e-15 absolute or 1e-9
  relative; every margin, gain statistic, check point and bound within 1e-9 percentage points; every per-anchor array
  exactly; the CPU spot check within 1e-4 (D3). A difference beyond these is traced to its cause before the step that
  depends on it; the computation that follows this file's text settles it, and the user is told.
- **Boundaries.** A check whose lower bound lies within 1e-12 of 0, a development-bar clause within 1e-12 of its
  threshold, a Δ_k of exactly 0, or a tie gap of exactly 24 in items 5 to 7, is reported to the user with both values
  before the step that depends on it is recorded (the stated inequalities still decide). On seed 42 the runner stops
  after `dev_seed42.json` and writes `results/boundary_seed42.json`; the continuation takes the SHA-256 of that file.
- **Round 4's lapses, closed here** (its report §7 and §8.4, its final review S1 to S3, N11, N12):
  1. The re-derivation writes its own code and imports only the list above (S1).
  2. The implementation is blind to the re-derivation's results (S2), as stated above.
  3. The §10 tests come before the runs they guard: list A before the GoEmotions step; list B before the wiring smoke
     test. No test this file lists is skipped, deferred or replaced without asking the user (S3).
  4. Every Python call, `--help` and one-liners included, runs with `PYTHONDONTWRITEBYTECODE=1` (N11).
  5. No run of this round's implementation other than the real seed-42 run computes a GE-placement result on seed 42: a dry run of the seed-42 runner
     stops when the guard would be released, and the development step is otherwise tested on synthetic bundles (N12).
     Smoke and dry runs print and log no metric value.
  6. The guard is not opt-in (D11), every resume and continuation path runs the full input check first, and the KILL and
     CARRY lines say they are pending the phase-1 agreement (items 7 and 8).
- **Round 4's final review, before-reuse items** (its §5 table), as they apply to this round:

| Item | Applies? | Closed by |
|---|---|---|
| T1: D11 hashes of `r3_bundle`/`r3_common` asserted only inside `extend_a1` | yes: this round calls `r4_bundle.build_bundle`, hence `extend_a1` | `r5_common` asserts the SHA-256 of every imported module file of rounds 1, 3 and 4 and of D12 before the first bundle call of every runner; a test patches one hash and shows that the runner stops before building |
| T2: the AFF-check guard compares by identity only | yes, in kind: this round writes its own GO checks (round 4's are bound to V4, V2 and V24) | §6.4 (b) and (f): gates recomputed from the definitions and the AFF arrays checked by value (SHA-256 recorded at each family run); a test swaps the arrays by copy, not by reference, and the check fires |
| T2: two near-vacuous `or` assertions (`test_r4_fusion.py`) | no: round 4's tests are not reused | this round's tests assert each condition on its own (no `or` inside a test assertion) |
| T3a-1: the boundary resume skips `check_inputs()` | yes, in kind | every entry and resume path of every runner of this round runs the full input check first; a test per path |
| T3a-2: the guard is opt-in; the order test stubs items 1 to 4 | yes, in kind | D11's placement-keyed guard inside every function that can take Q_GE; a test calls each such function with a `ge` object before release and expects a refusal; a test pins the order of the items list to this file's |
| T3a-3: `develop()`'s gates not checked against the verified ones | fixed in round 4 (7f7553b); applies to this round's development step | D7's gate check, the gates stored; a test with a mutated τ′ inside the development step fires |
| T3a-4: the boundary resume with a carried candidate untested | yes, in kind | tests of the continuation with a carried candidate (sensitivity written) and with a kill |
| T3a-5: the KILL line should say it is pending the agreement | yes | items 7 and 8's console lines |

  Round 4's S5 (the D10 clauses pinned at their thresholds) and S6 were fixed in round 4's code (7f7553b); this round's own
  record and development functions get the same tests (§10).
- **Time-box:** ends Wednesday 14 October. If the test cannot be completed by then, no partial verdict is issued; the
  user is told and decides.

| Day (Amsterdam) | Work |
|---|---|
| Wed 7 Oct | This rule written, checked, committed and sent to the user; the plan; the seed-42 code with list A; the GoEmotions step; the placement step; the seed-42 run; phase 1; the seed-42 verdict (carry or kill; spec §7 target) |
| Thu 8 Oct | If carried: sensitivity, the test-seed code with list B, the smoke test, seeds 52 to 54 built, the GO pass, phase 2, the verdict, the descriptive pass. In every case: the whole-branch final review and its fix wave; the report |
| by Wed 14 Oct | Time-box ends |

## 9. Outcome-to-action table

| Outcome | Action |
|---|---|
| A script check fails (a SHA-256 of this rule, round 3's or round 4's rule, a D12 input or an imported module; round 1's, round 3's or round 4's bundle check; §5 items 1 to 4; an episode-hash match; a change of `codes_provenance.json`; a condition-free, gate, placement-identity or alignment assertion) | Stop that step and report to the user; nothing is improvised |
| §5 item 2 fails (the GoEmotions rerun differs from `affect_probs` by more than 1e-4, or the join is not 32,413 selection captions) | No selection caption is passed to the model; traced to its cause; the user decides before anything else runs |
| §5 item 3 or item 4 differs from the stored or stated values | Traced to its cause; the user decides before anything else runs |
| The GE head reaches its cap after the fallback (D4) | No GE posterior is written; the user decides |
| The wiring mutation of §10 does not fire an assertion | Stop; no test seed is built; the user is told |
| No candidate clears the development bar with Δ_k > 0 (§5 item 8) | **Kill** (after the phase-1 agreement): no test seed is built; the seed-42 results go to the user; AFF stays the current best |
| A candidate is carried | After the phase-1 agreement: sensitivity, test-seed code with list B, smoke test, build, GO pass (§8 order) |
| On the test seeds, before the verdict | Only the quantities of §6.4 |
| All nine checks of §6.5 pass | **GO** within the claim of §6.10, with the disclosures of §6.11: the carried candidate replaces AFF as the current best |
| Any check of §6.5 fails, including a partial pass | **NO-GO**, read by §6.7; AFF stays the current best; the user decides what follows |
| AFF's own checks (§7 item 2) fail on the test seeds, or B′(A1) is above AFF or the candidate (its pooled mean R@1 at or above their fused R@1) | Reported only; no second verdict; the user decides what follows |
| A measured diagnostic of §5 looks good or bad | Reported only; it decides nothing |
| The re-derivation or the final review disagrees with a reported number, the carry or the verdict beyond the agreement of §8 | Traced to its cause; the computation that follows this file's text settles it; the corrected result goes to the user with the cause; the rule is not changed |
| A run crashes, or a bug is found before the rule is applied | As round 3's §9: correcting code to match this file is not a change of the rule; a crashed run is repeated after its partial outputs are deleted; a written results file is never overwritten (the corrected run writes beside it with the suffix `_fix<n>`, the log records the cause and the numbers before and after); a completed seed is never rebuilt; the GoEmotions file and the GE file are never recomputed once their SHA-256 is committed |
| Any situation this file does not cover | The user decides; this file is not changed after its commit without the user |

## 10. Process

- Scripts of this folder assert this file's SHA-256, round 4's and round 3's rules' SHA-256 and the SHA-256 of every D12
  input and imported module they read, refuse to overwrite non-smoke results, and write only to this folder's `cache/`
  and `results/` (gitignored by this folder's `.gitignore`), except that `run_baselines.py` (§6.2) writes its own outputs
  for seeds 52, 53 and 54 in `src/test/20261030_aspect_baselines/results/` and rewrites `codes_provenance.json` there,
  the smoke builds write to that folder's `results/smoke/`, and the seed ledger gets its rows.
- Module names of this folder start with `r5_` (runners `run_r5_*`, tests `test_r5_*`), so that none shares a name with a
  module of rounds 1 to 4 (`common`, `rc_core`, `rb_*`, `r2_*`, `r3_*`, `r4_*`, their `run_*` and `test_*`), whose folders
  are on `sys.path`. The folders of rounds 1 to 4, step 1, the brainstorm, the told-oracle line and the affect line are
  read only: their modules are imported by path and not modified, and their functions that write files are not called.
- **List A: tests written and passing before the GoEmotions step** (step 2 of §8):
  1. `r5_common`: the input and module SHA-256 checks (a patched hash stops the run before any build); the seed guard
     admits 42, 52, 53 and 54 and refuses 49 and 55, and admits 9001 to 9003 only with smoke; round 4's `TEST_SEEDS`
     equals this round's.
  2. GoEmotions runner, with a stub model: the rows and join of D2 item 1 (the assertions fire on an unsorted, short,
     scorer-train-overlapping or held-overlapping row set); the `refs/main` and `COSIR_ARTELINGO_ANNOTATIONS` asserts of D2 item 2; D3's sample positions and order; the tolerance at 0.9e-4 passes and at 1.1e-4
     fails; a shifted sample (misaligned rows) fails; a failed sample passes no selection caption to the model; the
     device tag, and a refusal when the device used is not the device given; batch 256 and max_length 64 passed; no
     overwrite.
  3. Placement function, on synthetic data: it equals a direct `LogisticRegression` fit with the same draw, check rows
     and scatter; NaN outside selection, float32; `classes_` other than 0..K−1 refused; the fallback (a synthetic fit
     that reaches max_iter refits once with the larger cap and records both counts; a second cap stops); the
     scorer-train position mapping (a misaligned mapping refused).
  4. GE extension: post_Q is a new dict (a mutation that assigns into `bundle.post["affect"]["txt"]` or the cached
     heads fires); the image and caption feature columns and stack slices unchanged; B′_Q condition-free; round 3's and
     round 4's fields and the whole of `bundle.post` unchanged (fingerprints); with Q_CLIP every output equals the
     bundle's; with a synthetic Q ≠ Q_CLIP only the affect slice of the stack, feature columns 0 to 5 and B′_Q change, and
     an extension that ignores its Q fires D5's positive check.
  5. Candidates: G-T reads P, m and π from F and T from stack_G (a swap of either fires); G-TF reads both from F_G; τ′ =
     `rc_core.thresholds` of G-TF's margins (condition a first); τ′ read from the record and never recomputed on a
     non-42 seed; G-T's gates equal AFF's; the counterpart is built from the candidate's own term and gates (a mutation
     that passes AFF's fires); tie tests for π′ with constructed exact ties (M28); the ρ_ctrl criterion test on a
     hand-computed example (M06).
  6. Records and carry: the comparator order B′_G, B′(A0), counterpart, B and a bar-comparator test in which the
     per-seed, pooled and per-pair choices differ (T2-M4); D10 pinned at its thresholds with round 4's final review S5
     cases (a bar margin point exactly 0.5 passes, 0.49 fails, a lower bound exactly 0 fails, clause 3 reads the gain
     statistic's lower bound and not the bar's) and their boundary flags; Δ_k from integers (a non-multiple of 0.25
     refused; a float-mean Δ refused); the carry (band 24 inclusive, 25 exclusive, Δ_k = 0 not in E, E requires all
     three clauses, a tie goes to G-T, an empty E is a kill); boundaries flagged at 1e-12, Δ_k = 0 and a gap of 24.
  7. Seed-42 runner: the guard refuses every function that can take Q_GE, each called directly with a `ge` object
     before release (D11, M20, T3a-2), and refuses the diagnostics before `carry.json`; the items run in this file's
     order; the development step checks and stores the gates it used (a mutated τ′ fires; T3a-3); every entry and
     resume path runs the full input check (T3a-1); the sensitivity path runs only when `results/carry.json` names a
     carried candidate and the log holds the phase-1 agreement record (refused otherwise), computes the nine checks in
     §6.5's order with round 3's `r3_stats.sensitivity` and reproduces SE and x on a hand-computed example; the boundary continuation with a carried candidate and with a kill
     (T3a-4); the KILL and CARRY lines say "pending the phase-1 agreement" (T3a-5); no non-smoke overwrite; a dry run
     stops at the guard and its console and log hold no decimal number (the leak check catches any decimal, one-decimal,
     ".5" and "5e-03" forms included).
  8. Diagnostics, on hand-made examples: the AUC sets and order of `bs_07_detector.py`; Δ_affect's column; the
     per-condition and per-direction definitions of (d); the re-assembled scores equal the family's arrays.
- **List B: tests written and passing before the wiring smoke test** (only if a candidate is carried; §8 step 7): the
  build runner (one invocation for 52 to 54; the hash checks against seeds 42, 43, 45, 47 to 51 and among the new
  seeds; `codes_provenance.json` before and after; build records; crash handling; ledger rows); the GO-pass assertions
  (a) to (f) of §6.4, each shown to fire under a mutation; the nine checks in this file's order, B′(A1) in no check, the
  non-carried candidate never computed, the open counts never written; the rule application (the readings of §6.7,
  including the AFF-only failure, reading `results/sensitivity.json`; a lower bound within 1e-12 of 0 goes to the user);
  the descriptive pass reproduces the GO pass's cached arrays (M27); the boundary-reported path and the phase-2
  agreement record each bound to the SHA-256 of the current `go_pooled.json` (T3-1, T3-10).
- **Smoke runs** write to `results/smoke/`, may be overwritten, are not results, and never print or log a metric value of
  any scorer on any seed: the console and the log show only shapes, counts, file names and the pass or fail of each
  assertion. Their files are opened only by the assertions and are deleted when the smoke test has passed. They never
  use the test seeds. The **end-to-end wiring smoke test** runs after §6.1 and list B: the GO pass, the rule application
  (reading `results/sensitivity.json`) and the descriptive pass on the smoke seeds 9001, 9002 and 9003 (built with
  `run_baselines.py --smoke --episodes-seed <s>`), with the carried candidate and the real GoEmotions and GE files. It must
  complete with every assertion passing before any test seed is built, and it runs three mutations of the wiring that
  must each fire an assertion: the candidate's gated term passed where the counterpart expects G_cf; AFF's fused arrays
  swapped, by copy, for the candidate's in the AFF check; and Q_CLIP passed where Q_GE is expected.
- CPU for everything except the GoEmotions step (D2): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  PYTHONDONTWRITEBYTECODE=1` on every Python call, at most 3 processes, `uptime` and `free -g` checked first. The main
  session launches every real run (`run_in_background`); subagents implement.
- Records: the log `20261123_idea3_goemotions_log.md` in this folder (times from `TZ=Europe/Amsterdam date '+%F %H:%M'`);
  the report `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md`, committed only after the whole-branch final review
  and its fix wave, with a row in `docs/reports/reports_sum.md` and `scripts/check_reports_sum.py` run;
  `.claude/<yyyymmdd>_log.md` for any edit to an existing source file; the seed ledger rows. Storage left behind (the two
  cache files, about 9 MB, and `results/`) is reported at the end. Commits go to `main` without a further request
  (authorisation above), scoped by explicit path, never pushed.
