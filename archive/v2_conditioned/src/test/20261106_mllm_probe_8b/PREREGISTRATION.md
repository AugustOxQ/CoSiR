# MLLM probe addendum: Qwen3-VL-8B-Instruct (CoSiR v2, spec §4 branch 2 test)

Written and committed on 2026-10-03, before the 8B model has scored any episode (timing smoke included).

- **Why:** E3's early MLLM probe (Qwen3-VL-2B-Instruct, seed 44) did not meet its pre-registered rule (v2: R@1
  +0.22 [−1.13, 1.53], gain −0.53 [−1.68, 0.66] against the cosine), and its resolution was about 1.2 points. Spec §6
  allows the 8B model "if time allows", and the E3 report §11 offered an 8B probe under a pre-registered addendum
  before branch 2 is closed. Since then the method repair ended in branch 3 under its own pre-registered rule
  (`src/test/20261105_method_repair_diagnostics/results/decision.json`). The user asked for this probe on 2026-10-03.
- **Binding authority:** spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §4 (decision
  branches) and §6 ("Early MLLM probe"), file SHA-256 `049a0ec6bc119a3512b5fe5488771f3e6f3f36454b18c0d7c734524927c2e61e`;
  the E3 pre-registration `src/test/20261101_aspect_factor_gonogo/PREREGISTRATION.md` §9 (the "works" rule), file
  SHA-256 `07debf24f252d5e3e1589876d50bee65e4a8d35e629f35b44dc1da1d2a7682df`.
- **Frozen:** nothing here changes after the 8B model has scored a seed-46 episode. A change is a dated addendum at
  the end of this file, committed before the result it affects.

## 1. What is held fixed from the 2B v2 probe

- **Prompt and scoring:** `src/eval/mllm_reranker.py` as committed (SHA-256
  `7779b605beaf2d6dd6b5bf592796c963879bc26d3cb69de6c8c334883088d965`): the v2 prompt (`build_messages`, explicit line
  breaks, the aspect never named), letters A to M, letter scores read in fp32 from the last hidden state, the
  candidates permuted before lettering for every episode, condition and direction with `default_rng([seed, episode])`,
  scores mapped back with `unpermute`.
- **Runner:** `src/test/20261102_mllm_probe/run_probe.py`, the v2 code path, with one change: a `--model` option
  (default unchanged, `Qwen/Qwen3-VL-2B-Instruct`). Images: `max_pixels = 256 × 28 × 28`, bf16 weights.
- **Episodes:** `build_aspect_episodes` on the **selection** rows, pairs emotion×style (third genre), emotion×genre
  (third style), style×genre (third emotion), pooled in that order, validated, all rows asserted inside the selection
  split. Val and held rows are never read.
- **Baseline:** backbone-only CLIP ViT-B/32 cosine on the same episodes (`cosine_scores`, NaN outside selection).

## 2. What changes

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (bf16), on the local RTX 3090 under the GPU lock; DAS6 only if it does not fit.
- **Episode seed: 46** (fresh; seed 44 was scored by both 2B runs; seed 45 stays reserved for a future method's single
  GO test). The seed ledger `docs/superpowers/episode_seed_ledger.md` records it.
- **Episodes per pair, fixed by measured speed before any seed-46 score:** a timing smoke runs the 8B model on the
  2B probe's smoke episodes (`--n 4`, seed 44, 12 episodes; its numbers are not read as evidence). If the measured
  time per episode × 1,800 is at most 6 hours, the run uses **600 episodes per pair** (1,800 pooled); otherwise
  **300 per pair** (900 pooled, as for 2B). The choice and the measured time are recorded in the folder log before the
  seed-46 run starts. Rationale: at 300 per pair the 2B probe could only detect a pooled gain above about 1.2 points;
  doubling the episodes narrows the intervals by roughly √2.

## 3. The rule (unchanged from E3 PREREGISTRATION §9)

- The 8B MLLM **works** iff `compare(mllm, cosine, clusters, "r1")["ci95"][0] > 0` **and**
  `compare(mllm, cosine, clusters, "gain")["ci95"][0] > 0` on the pooled seed-46 episodes, with clusters = the anchor's
  painting group, painting-clustered bootstrap, 5,000 resamples, bootstrap seed 42.
- **Decision map (spec §4):** GO was missed (E3) and the repair ended in branch 3. If the 8B MLLM works, the numbers
  point to **branch 2** (a benchmark paper: a solvable, released task that embedders and metric-from-pairs fail while an
  MLLM given the examples partly succeeds). If it does not, branch 2 stays closed, and the user decides between
  branch 3 and a new method stage.
- **Descriptive only (decide nothing):** per-pair R@1 and gain; swap; the letter preference (share of top scores per
  letter, as `posthoc_letter_bias.py`); the 2B v2 result on seed 44 as context (different episodes, not a paired
  comparison); wall time and peak memory.

## 4. Provenance

`probe.json` records the model id, max_pixels, the SHA-256 of each pair's episodes, of the instruction text, of one
rendered sample prompt per direction and of the permutations (the runner's existing fingerprint), plus the runner's and
the reranker's file SHA-256. The run resumes only from a partial file with the same fingerprint.

## Addenda

(none)
