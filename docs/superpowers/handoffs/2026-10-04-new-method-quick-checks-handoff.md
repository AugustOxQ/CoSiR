# Handoff: implement and run the three quick checks (D0, N1, N2)

Written 2026-10-04 for a fresh agent. Read this first; it points to everything else. The user drives the next steps.

## The job

Implement and run the three development checks of the approved spec
**`docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md`** (user approved on 2026-10-04: "looks fine"),
then apply its decision table and report:

- **D0** (diagnostic): can the aspect be read from 4 example pairs when items are represented by label probes? Told vs
  inferred aspect.
- **N1**: a centered (covariance) agreement rule on the existing factor codes, against the current rule and diagonal
  KISSME on the same codes.
- **N2**: find-then-select cascade on A3 (rank by the condition-free score, rerank the top k by the condition term).

Decided with the user (do not reopen):
- **D0 threshold:** "close to Told" means Inferred keeps at least half of Told's gain over cosine. Commit the decision
  rule (spec §5, with this number) before any check is run.
- **Seeds, kept light (user, 2026-10-04):** develop on the existing seed-42 episodes; if a configuration passes, test it
  on 3 fresh episode seeds (45, 47, 48), each reported and pooled. Do not build single-look or lucky-seed ceremony
  around them (project memory `feedback_seed-handling-light.md`). The GO rule is written down before the test seeds are
  built.
- **The branch decision is NOT taken.** The repair stage's own pre-registered rule pointed to branch 3, but the user
  wants to try more methods first. Paper writing is paused.

## Read in this order

1. The spec above (self-contained: task, metrics, what was tried, the three checks, the decision table, sources).
2. `src/test/20261107_new_method_candidates/synthesis.md` §4 (ARS synthesis: why this order, required baselines, what
   a pre-registration needs) and, as needed, `scan_N1.md`, `scan_N2.md`, `scan_N6.md`.
3. `docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md` (A′: nested score, label-trained factors) and
   `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md` §6.1 (the selection versus aspect-finding trade-off).
4. `docs/reports/auto/v2/2026-10-23_aspect_episode_spike.md` (the label-probe reference D0 builds on) and its code
   `src/test/20261023_aspect_episode_spike/aspect_ceiling.py`.
5. Project memory `~/.claude/projects/-project-CoSiR/memory/project_v2-publication-plan-pending.md` (top entries).

## Where things stand (all numbers re-derived in final reviews)

| Stage | Outcome | Record |
|---|---|---|
| E3 (method A go/no-go) | NO-GO: A3 R@1 13.76, gain 0.26 [−0.04, 0.56]; its uniform control 16.72 | E3 report |
| A′ diagnostics (nested score + label-trained H3) | Pre-registered decision **branch_3**: H1 not promising (16.52 vs control 16.55, gain −0.01); H3 ceiling too low (best nested gain 0.05 < g* 0.218) | 2026-11-05 report; `src/test/20261105_method_repair_diagnostics/results/decision.json` |
| ARS review of the repair order | Major revision; its rule changes became spec §15 and the A′ pre-registration | 2026-11-04 report |
| Qwen3-VL-8B in-context probe (seed 46, 600 per pair, node404) | **works = False**: R@1 +1.07 [0.17, 1.93], gain +0.21 [−0.51, 0.94] | `src/test/20261106_mllm_probe_8b/20261106_mllm_probe_8b_log.md` |
| New-method candidates N1 to N6 + ARS literature check | All "partially exist"; N4 dropped for this window; N3, N5 behind privileged references; order D0, N1, N2, then N6 if D0 passes | `src/test/20261107_new_method_candidates/` |

## Code and data entry points

| What | Where |
|---|---|
| Episodes, validator, hashing | `src/eval/aspect_episodes.py` |
| Metrics, painting-clustered bootstrap, paired compare | `src/eval/aspect_metrics.py` (`per_anchor`, `cluster_bootstrap`, `summarize`, `compare`) |
| Cosine, agreement term (and its uniform control), 1-D cross-fit | `src/eval/aspect_scorers.py` |
| Nested score, its control, min-margin cross-fit, readings | `src/eval/aspect_nested.py` (tested in `src/test/test_aspect_nested.py`) |
| Raw-feature baselines incl. KISSME, RCA, CVS-style ("wang") | `src/eval/pair_metric_baselines.py` |
| Seed-42 development episodes and their loader (SHA-checked, NaN outside selection rows) | `src/test/20261030_aspect_baselines/results/episodes_seed42.npz`; `EvalContext` in `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` (import as in `src/test/20261105_method_repair_diagnostics/common.py`) |
| Factor checkpoints and cached codes | A3 `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` (SHA-256 dadfef1b…); C0 and SE codes `src/test/20261030_aspect_baselines/results/codes_{C0,SE}.npz`; label-trained L3, LT (diagnostics only, never candidates) `src/test/20261105_method_repair_diagnostics/checkpoints/` |
| Labels | `src.data.artelingo_splits.artelingo_aspect_labels(data)` (emotion without the catch-all, style, genre with −1 where unlabelled) |
| E1 per-anchor baseline arrays (cosine, RCA, …) for seed 42 | `src/test/20261030_aspect_baselines/results/per_anchor_seed42.npz` (keys `<scorer>__<metric>`) |
| Fresh test seeds later | build with E1's runner `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>` (also gives cosine and RCA on them) |

## Environment and rules that bite

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python`; tests `... -m pytest <file> -q`; CPU work with
  `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Never install into the env.
- **Local GPU (shared):** the user needs it for other projects' smoke tests. These checks are CPU-only; if a GPU is ever
  needed, check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and use
  `flock -n -o -E 75 /tmp/gpu0.lock <cmd>`, or use DAS6.
- **DAS6 node404** (user's reservation, 3 × 48 GB): use the cluster-run skill. `watch`, `logs` and `pull` need
  `--node node404` (the default `NODE` is node403). Launchable forms are `python main_cosir.py …` or
  `bash scripts/run_<name>.sh`; non-Hydra data goes up with a sync script on the pattern of
  `scripts/das6_sync_mllm_probe_8b.py` (run with `/usr/bin/python3`). The node env has transformers 5.16.1 (local 5.6.2).
  `cluster sync` pushes main to GitHub.
- **Data paths** can be redirected with `COSIR_ARTELINGO_FEATURES`, `COSIR_ARTELINGO_ANNOTATIONS`,
  `COSIR_WIKIART_GENRE_DIR`, `COSIR_WIKIART_DIR` (defaults unchanged).
- **Parallel trainings:** 4 factor trainings at once ran out of memory on the 24 GB card (each reserves 5 to 10 GiB);
  run at most 2 or 3.
- **Git:** main, files staged by explicit path, `bin/` and `docs/paper/` stay untracked, `docs/*.DS_Store` are not ours.
  origin/main was last pushed by `cluster sync` at b33d223; later commits are local. Push only when the user asks or
  `cluster sync` needs it.
- **Experiment folders:** next free sequence date is **20261108** (`src/test/20261108_<name>/`, `.gitignore` copied
  from `src/test/20261023_aspect_episode_spike/.gitignore`, ends with a `_log.md`). Edits to existing source files get
  an entry in `.claude/20261003_log.md`-style day logs (`git add -f`).
- **Reports:** `docs/reports/auto/v2/<sequence date>_<topic>.md` (paper-draft style, a real baseline beside every
  number, figures, no dashes as punctuation), one row in `docs/reports/reports_sum.md`, then
  `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` must print OK. Next free report date: 2026-11-06
  is the 8B probe's (report not yet written, see below), 2026-11-07 the candidates', so this work is **2026-11-08**.
- **Final review:** after the checks (and any test), one whole-branch review on the most capable model that re-derives
  the load-bearing numbers from stored arrays with independent code, then one fix wave and a scoped re-review
  (`~/.claude/rules/final-review.md`). In this project it found a real defect every time (this session: two overclaims
  in a report, and a wording error introduced by the fix wave itself).
- **Process the user uses:** ARS for reviews and literature (methodology-focus review when a decision needs checking;
  deep-research scans for novelty), superpowers writing-plans plus subagent-driven development for implementation.

## Open items left from 2026-10-03/04 (not blocking the checks)

1. **8B probe report** (`docs/reports/auto/v2/2026-11-06_mllm_probe_8b.md` + reports_sum row): not written. The log has
   every number and the verdict; the per-anchor arrays are in
   `res/cluster_jobs/mllm-probe-8b-seed46/code/outputs/mllm_probe_8b_seed46/`. The pre-registered descriptive letter
   preference (`src/test/20261102_mllm_probe/posthoc_letter_bias.py` pattern) is not computed yet.
2. **Candidates write-up** (`docs/reports/auto/v2/2026-11-07_new_method_candidates.md` + row): the material is in
   `src/test/20261107_new_method_candidates/`; no report yet.
3. **Paper (paused):** ARS intake draft `docs/paper/cosir_v2_aspect_similarity/phase0_paper_configuration.md`
   (untracked); the user has not answered the configuration questions (venue, ArtELingo-only, English-only abstract,
   authors, release).
4. The local 150-episode 8B partial (`src/test/20261106_mllm_probe_8b/results/probe_partial.npz`) was never read; keep it
   unread.

## Suggested first steps

1. Read the spec; write `src/test/20261108_<name>/DECISION_RULE.md` (spec §5 with D0's threshold) and commit it.
2. Plan the one implementation task with superpowers:writing-plans (D0 told/inferred with a genre probe added, N1 rule,
   diagonal KISSME on codes, N2 cascade with the both-in-top-k diagnostic, unit tests that fail when each rule is
   removed), implement with a subagent, review.
3. Run the three checks on CPU; apply the decision table; report to the user before building anything further.
