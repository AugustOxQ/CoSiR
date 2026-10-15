# Final review, area C (comparator, GPU jobs, descriptive pass, smoke chain): round 6

> Reviewer C (Opus), 2026-10-09 19:16 to 19:55 (Amsterdam). Snapshot `/project/CoSiR-r6-fr` at bceff9c (detached).
> Tickets 10 to 15; rule §6.6, §6.7, §7, §10.2, §10.5; spec SC3, D10 to D13, D15, D18, D21, D22, D25 (D1, D24).
> Scripts `final_review/fr_c_*.py`, outputs `fr_c_*.json`; scratch
> `/tmp/claude-0/-project-CoSiR/714507bd-709f-400b-92b9-40bd3c709f3b/scratchpad/fr/c/`.

## 1. Verdict

**CONFIRMED WITH FIXES.** Every setting of §7 matches the rule character by character, and my own code (written from
§7's text) reproduces the code's tuning scores, chosen setting, λ picks (finite and λ = ∞, with parse failures),
DTS-CF and DTS-N, per-anchor arrays, the stop count and the frozen held scoring bit for bit; the §10.5 quantities
re-derive exactly on a three-seed synthetic input; the smoke end-to-end test passes. Three should-fix findings
remain, all in how the §7 budget and the §6.7 smoke are enforced: a DTS rerun forced by any later code change counts
as a late build, the 24-hour clock start is not pinned, and a smoke that never ran the GPU jobs is accepted.

## 2. What I re-derived

| Check | How | Mine vs the code's | Result |
|---|---|---|---|
| §7 settings text | `fr_c_settings.py`: W1 to W4, listing prompt, marker regex parsed from the rule; 32/128 tokens, K, model, names, 9,406/49,152, tie order, 1,024, 24 h, max_pixels | 15 of 15 equal | pass |
| Greedy decoding | snapshot `generation_config.json` (repetition_penalty 1.0); `generate_kwargs` do_sample False, temperature/top_p/top_k None; argmax check per token | as the rule | pass |
| Embedder | `ClipB32("cpu")`, raw normalised string, batch of one, no template (gpu_path §3: same model as the float32 cache) | as the rule | pass |
| Tuning (8 settings), chosen, λ picks of DTS/DTS-CF/DTS-N, failure counts, sanity, per-anchor arrays, stop hits | `fr_c_dts.py`: own parsing, CRL, failure fallback, cross-fit, z-mean CF, DTS-N names; crafted smoke-shaped world (192 episodes, 28/30 failed rows a/b, mixed markers, blank and repeated lines, short listings); code run through `run_r6_dts.main` | all equal; picks ∞ in world 1, 4/2/8 in world 2; hits 536 = 536 and 237 = 237; chosen W2 K16 both | pass |
| Held DTS path | same world as a held seed: `held_dts_scores` vs my frozen application (pick of half h on parity 1 − h, failed rows on cosine, CF gain 0) | DTS, DTS-CF, DTS-N per-anchor exact | pass |
| Stop and budget decisions | `fr_c_stop.py` on crafted seed-42 records through `run_r6_dts.main` and `run_r6_held.dts_stop_problem` | 9,405/9,406 continue, 9,407 stops; built 2026-10-10 12:29:00 in budget, 12:29:01 stops; not built yet: refused, nothing written | pass (see finding 1) |
| Clock start | git: 6beb360 at 2026-10-09 12:29:20, log line 32 | the code takes whatever `--clock-start` says | finding 2 |
| §10.5 rows (21 scorers × pooled, 3 seeds, 3 pairs × R@1, gain, swap, AFF minus) | `fr_c_desc.py`, own cluster bootstrap | 378 of 378 equal (≤ 1e-9) | pass |
| P1 to P7, S1, S2 by scope; AFF − B1 per pair; bar margin (D12 comparator per scope, per-pair breakdown); R1's seven checks by name | same | all equal | pass |
| Two-way bootstrap (both candidate sides), anchor-only interval, item reuse | own implementation of the module's definition | max abs diff 0.0; reuse equal | pass |
| FT checkpoints | SHA-256 of the three `best_params.pt` vs `FT_CKPTS`; selection vs the clipft report (LP 3e-4 ep 9, LB 3e-5 ep 8, LoRA 1e-4 ep 10) | equal | pass |
| MLLM un-permutation | read against `mllm_reranker.unpermute` and the probe's prompt order | `put_along_axis(out, perms, shown)` = `out[perm] = shown` | pass |
| Smoke chain | `test_r6_smoke_e2e.py` once | 2 passed in 579 s; nothing left in F/results, no `.pyc` | pass |

## 3. Findings

1. **should-fix**, `run_r6_dts.py:443` (`b = D.budget(start, max(sanity["time"], chosen["time"]))`). The budget is
   judged on the time of the stage records it reads. Rule §6.7 makes any change to a module DTS ran (r6_common,
   r6_episodes, r6_gpu_common, r6_gpu_inputs, r6_dts, run_r6_dts, dts_settings.json) force a rerun of the DTS stages
   (the `same_modules` guard, smoke step 1 and the held runner's `dts_stop_problem` all refuse otherwise). A rerun
   after 2026-10-10 12:29 then writes "not built within the 24-hour budget", stop = true, and the read waits for the
   user, although DTS was built in time (`fr_c_stop.json`, case `rerun_after_code_change`). §7.5 and §7.7 make
   "built" a one-time event; §6.7 re-evaluates the stop from the rerun, not the budget. A fix to r6_common.py after
   the smoke (the fix wave touches it now) is enough to trigger it. Fix: the first chosen stage that is built in
   budget writes `results/dts_first_built.json` once (time, setting, settings SHA, GPU output fingerprints); the
   stop uses that time when the current chosen record has the same setting, settings SHA and GPU outputs, and the
   rerun's time otherwise.
2. **should-fix**, `run_r6_dts.py:405-409` and `run_r6_held.py:816-841`. `--clock-start` accepts any time, and the
   held runner checks only `built` and `stop`, not the recorded `budget.clock_start`. A mistyped start (a later day)
   would let a late build pass silently, or an earlier one stop. Fix: a constant `DTS_CLOCK_START = "2026-10-09
   12:29"` (log line 32; commit 6beb360 at 12:29:20) that the stop stage requires for seed 42 and
   `dts_stop_problem` checks against the record.
3. **should-fix**, `run_r6_smoke.py:766-801, 853` with `run_r6_held.py:768` (`smoke_guard`). A family whose GPU
   outputs were not given is written `{"missing": ...}` and the record still says `"passed": true`, which the held
   runner, the apply step and the descriptive pass accept. Rule §6.7: "The GPU job scripts run on the smoke episodes
   too ... so the smoke covers their key joins." A smoke without them lets the first real join of the verbaliser,
   listing, reranker and FT outputs happen in the descriptive pass after the verdict, where a join bug can only be
   corrected through the reserve read (§9). Fix: in the default and fix/crash kinds, `passed` requires the
   verbaliser, listing, reranker and FT outputs given and checked (DTS's smoke-scale sanity or stop outcome may stay
   a "missing" DTS row, since its inputs were checked); or record `gpu_covered` and require it in `smoke_guard`.
4. nit, no code checks `results/dts_settings.json` (rule §7: the copy committed before the first seed-42 call;
   contracts §9 amendment 13:05). `list-input --for sanity` (the first seed-42 GPU input) could require that file
   with F's bytes. Otherwise a run-handoff item.
5. nit, `r6_dts.first_line` strips the answer before taking the first line, so "\nCalm" gives "calm" where the rule's
   literal first line is empty (a failure). Recorded as an agent default in the committed settings file; defensible.
   A mutation removing the strip is caught by the tests.
6. nit, the descriptive pass re-embeds every value string in its own cache (`dts_value_embeddings_descriptive.npz`),
   including those embedded on seed 42; CPU float32 results can differ by rounding across thread counts. Run
   handoff: give `external_sources.json` a byte copy of the tuning cache as `embeddings`.
7. nit, held listing jobs re-list (phrase, K) pairs already listed on seed 42 unless the run chat passes `--cache`
   with the seed-42 listing outputs (T11's nit); batched listing (16, left padding) need not equal single calls. The
   rule's "one listing per (phrase, K)" then rests on the cache. Run handoff: `--cache` on every held listing job,
   and the seed-42 listing folders in `external_sources.json`'s `listing_out` (the merge refuses a divergent answer).
8. nit, provenance records the node's library versions per run, but nothing asserts that the seed-42 and held jobs of
   one family ran on the same transformers (gpu_path hazard 6). Run handoff: run both on the same node env.

## 4. Guards

- §7.6 stop above 9,406: fires through `run_r6_dts.main` (exit 3) and `dts_stop_problem` (fired); `>=` mutation caught.
- §7.7 budget: fires (12:29:01); +1 h mutation caught.
- §7.4 sanity first and ceiling gain > 0: exit 3 / refused; `>= 0` mutation caught.
- §7.3 DTS-N target names: swapped-condition mutation caught.
- §7.4 frozen convention (half h scores parity 1 − h): mutation caught (6 tests fail).
- §7.4 tie order: `>=` mutation caught.
- §7.2 failed parse scores as cosine at every λ (λ = ∞ included): fallback deleted, caught.
- §8.3 MLLM recorded permutation (un-permute): inverse mutation caught.
- §10.5 two-way candidate stream: same-stream mutation caught.
- §8.3 GPU jobs receive no labels: `das6_sync_r6.forbidden_in_job` on a crafted job (an `anchor` array, a WikiArt path in a caption): 2 problems found, 0 on a clean job (`fr_c_guards.json`); the module's own mutation tests are T10's.
- §8.3 descriptive only after the verdict: `run_r6_descriptive.run` in real mode without a verdict: refused, exit 4, nothing written; the T13 suite's mutation tests of `check_verdict` pass.
- §6.7 wiring mutation fires and the control passes: shown by `test_r6_smoke_e2e.py` (both tests pass).
- §6.7 record SHAs: e2e asserts `module_sha256 == r6_module_shas()` and the three consumers accept it; staleness
  refusal is reviewer A's.

## 5. Hard rule 6

| ID / section | Status | Where |
|---|---|---|
| SC3 | covered; partial on the budget (findings 1, 2) | run_r6_dts.py stages, run_r6_held.dts_stop_guard |
| D10 | covered | r6_dts, r6_external (reported rows, no pass check) |
| D11 | covered | r6_dts.stop_decision, run_r6_dts.stage_stop |
| D12 | covered; partial on the budget (findings 1, 2) | r6_dts.budget, stage_stop |
| D13 | covered (per-pair rows); the disclosure is the report's | r6_descriptive.scorer_row |
| D15 | covered | r6_gpu_ft_features, r6_ft_cache, r6_external (FT-LP on CPU, LB/LoRA by row id) |
| D18 | covered | run_r6_dts.stage_tune, stage_chosen |
| D21 | covered; partial on the GPU smoke (finding 3) | das6_sync_r6, r6_gpu_inputs neutral names, run_r6_smoke |
| D22 | covered | r6_descriptive.two_way_bootstrap, swap rows; r6_external MLLM seed 52 only |
| D25 | covered | scripts/run_r6_*.sh, das6_sync_r6 (`cluster.validate_node`, /local/wding asserted) |
| D1, D24 | no code needed | |
| §6.6 | covered | smoke step 1 and held runner require a current, built, non-stopping `dts_stop.json` |
| §6.7 | partial (finding 3) | run_r6_smoke.py |
| §7.1 to §7.6 | covered | dts_settings.json, r6_gpu_verbalise/listing, r6_dts, run_r6_dts |
| §7.7 | partial (findings 1, 2) | run_r6_dts.stage_stop |
| §10.2 | covered (`cluster-selftest gpu` is a run-chat step) | wrappers set PYTHONDONTWRITEBYTECODE, OMP/MKL 8, offline HF |
| §10.5 | covered | r6_descriptive, r6_external, run_r6_descriptive |
| Contracts §8 amendments 17:22, 17:32 | applied in every consumer | r6_external.require_sources (real: four entries, exit 4), dts_stop beside the record, embedder meta, MLLM model keys, no-number rows |
| Contracts §9 and amendments 13:05, 14:50 | applied | Manifest keys, neutral names, job folders, score_seed include_pm |

Unrequested: none found.

## 6. Deferred nits

- T10 `copy_images` copies `cluster.run_data_sync`'s body to add `--copy-links`: leave (cluster helpers only, remote
  paths asserted under /local/wding).
- T10 listing check-only "k of 4" batched vs single: leave; read it at the GPU smoke (finding 7's cache keeps one
  listing per key).
- T11 list-input repeats listed pairs: leave with the run-handoff item of finding 7.
- T11 two mutations change nothing (float32 mean of two values, `count=1`): leave, equivalent mutants.
- T13 departures from plan §10 ("a two-way (anchor cluster × candidate cluster) bootstrap ... together with the
  item-reuse rate per split"; swap success §5.1), each defensible, leave: pigeonhole weights W[anchor] × mean V over
  candidates (an episode has 13 candidate paintings, so a mean is the natural crossed weight); two candidate sides
  (all 13 and the 2 targets; the targets carry R@1, so that side is the conservative one); independent anchor and
  candidate draws (the plan names two factors); example pairs off the candidate side (the plan names anchor ×
  candidate only); pooled P1 to P7, S1, S2 only (the plan asks for a sensitivity check, not per scope); reuse = 1 −
  distinct/slots, per split with seed 42's development figure; exit 5 on held arrays that disagree with the pass and
  `descriptive.json` written once (nothing guessed).
- T06 verdict-existence check (from reviewer B): leave. The only post-verdict caller with held bundles is
  `run_r6_descriptive`, whose `check_verdict` validates the verdict's content, the agreement record and both SHA-256s
  before any bundle; the held pass calls `score_seed(..., include_pm=False)`.
- Snapshot test files of area C (`test_r6_dts`, `_gpu`, `_gpu_inputs`, `_gpu_t12`, `_sync_ckpt`, `_external`,
  `_descriptive`, `_smoke`): 297 passed, 0 failed (logs in the scratch dir, `s2_*.log`).
