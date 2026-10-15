# Handoff: implement the CVPR plan's experiments E0 to E5

Written 2026-10-03 at the end of the planning session, for a fresh agent that implements the plan. Read this first.
It points to everything else.

## The job

Execute the implementation plan **`docs/superpowers/plans/2026-10-03-cosir-v2-cvpr-e0-e5.md`**: 17 tasks plus a
final whole-branch review. It implements experiments E0 to E5 of the design spec
**`docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`** (revision 2).

- **The deliverable** is the **Fri Oct 9 go/no-go decision package** (Task 15). After it, **stop**: the user chooses
  the paper branch (spec §4), and only then do E5 (Task 17, only after a GO) and the next plan (E6 onward) follow.
- **Deadlines:** CVPR abstract registration Tue Nov 10, paper Mon Nov 16, supplementary Mon Nov 23.

**Execution method.** The user had not chosen when this was written.
- The planning session recommended **subagent-driven** (`superpowers:subagent-driven-development`). The tasks pass
  exact interfaces to each other, and a mistake reaching the Oct 9 decision would pick the wrong paper branch.
- Ask the user once, then follow their choice.
- Implementation goes to Claude Code subagents sized to the task (`haiku`, `sonnet` or `opus`; see
  `~/.claude/rules/agent-routing.md`), with a review after each task. Codex only if the user asks.

## Read in this order

1. **The plan**, in full, especially the *Mechanism note*, *Global Constraints* and *Review Focus* at the top.
2. **The spec**, §3 to §11 (the problem, claims K1 to K8, benchmarks, method A, baselines, statistics, schedule) and
   §14 (the review-driven changes).
3. **Background, only if needed:**
   - the ARS review summary `docs/reports/auto/v2/2026-10-27_ars_plan_review.md`;
   - the aspect spike `docs/reports/auto/v2/2026-10-23_aspect_episode_spike.md`, whose numbers Task 4 must
     reproduce;
   - the backbone check `docs/reports/auto/v2/2026-10-25_backbone_check.md`.
4. **Project memory:** `~/.claude/projects/-project-CoSiR/memory/project_v2-publication-plan-pending.md` (all of
   the user's decisions, with dates).

## Decisions already made by the user (do not reopen)

- **The problem:** example-conditioned **aspect** similarity across modalities. Examples share an aspect with values
  different from the query's, and contrast pairs share another aspect.
- **Benchmarks:**
  - ArtELingo, with emotion × style × genre (genre from ArtGAN, 81% coverage);
  - CUB on the **50 unseen species** of the xlsa17 zero-shot split;
  - GeneCIS focus attribute;
  - SemArt;
  - Affection is only a potential extra.
- **Method A:** aspect-trained shared factors with the training-free agreement rule. Pseudo-aspect partitions are
  an accepted, flagged risk (R-pseudo), now tested by the held-out-genre test (K8).
- **Backbones:** CLIP ViT-B/32 (development) plus **Qwen3-VL-Embedding-2B** (final tables), after a fidelity check;
  PE-Core L/14 is the fallback.
- **Metrics:** R@1 and **condition gain** are co-primary; the bootstrap is clustered by painting (by species on
  CUB); the "matches" margin is ±1.0 R@1.
- **Aspect typing for K3:** emotion is subjective; style, genre, SemArt fields and CUB attributes are objective.
- **Three Oct 9 branches:**
  - GO: method paper;
  - NO-GO with the MLLM probe working: benchmark paper;
  - NO-GO otherwise: analysis paper for a workshop or a datasets-and-benchmarks track.
- **Use case (confirmed):** mood or style boards, relevance feedback, dataset curation.
- **Not in the tables:** PercepT and teacher-only (PercepT is optional, depending on the storyline).

## A finding from planning that matters for E3

A simulation while writing the plan showed:
- **The agreement rule needs aspect-block codes:** all values of an aspect must share factors, each value a different
  pattern. With one factor per value it is blind under value-disjoint episodes (condition gain exactly 0, because
  every candidate ties).
- **Raw features that mix the aspects** defeat raw pair rules the same way (0.00 mixed against 0.70 to 0.95 when each
  aspect has its own block).

This explains the aspect spike's failure. It is why the plan includes:
- the tests `test_value_onehot_codes_are_blind_under_value_disjoint_conditions` and Task 8's block-world fixture;
- grid run A6 (denser codes);
- a value-sharing diagnostic in Task 13.

**Do not "fix" a failing synthetic test by switching the fixture back to value one-hot codes or a mixed world.**

## User actions the plan needs (ask early; they block tasks)

1. **Task 5, CUB zero-shot class lists.** The `curl` is blocked by an egress hook, so the user runs it with `!`:
   ```
   ! mkdir -p /data/SSD/cub/xlsa17 && cd /data/SSD/cub/xlsa17 && curl -sSfL -o xlsa17.zip https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip && unzip -o -j xlsa17.zip 'xlsa17/data/CUB/trainvalclasses.txt' 'xlsa17/data/CUB/testclasses.txt' && wc -l *.txt
   ```
   Expect 150 and 50 lines. If the URL is dead, ask the user for a mirror.
2. **Network approvals** for HF model downloads (Qwen3-VL-2B-Instruct in Task 14; the official Qwen stack via
   `pip --target` in Task 6).
3. **The Oct 9 branch decision** after Task 15.
4. **DAS6:** only if E4 (Task 16) cannot run locally. The node is whatever the user has reserved; use the
   `cluster-run` skill.

## Environment and data (verified 2026-10-02/03)

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python` (transformers 5.6.2, which includes `Qwen3VLForConditionalGeneration`).
  Never install into this env; use `pip install --target /data/SSD2/pyenvs/<name>/`.
- **Local RTX 3090,** shared with other sessions:
  - check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` first;
  - wrap every GPU job in `flock -n -o -E 75 /tmp/gpu0.lock <cmd>`;
  - CPU jobs set `OMP_NUM_THREADS=8`.
  - On 2026-10-02 the container lost GPU access ("Failed to initialize NVML"). If that recurs, the user restarts the
    container.
- **Datasets** (`/data/SSD/`):
  - `cub/` (images plus Reed captions in `captions/extracted/text_c10`);
  - `semart/SemArt`;
  - `visual_genome/VG_100K_all`;
  - `wikiart_genre/` (ArtGAN genre CSVs; class names in the plan's `GENRE_NAMES`).
- **ArtELingo:** images at `/data/PDD/wikiart_proj/wikiart/<annotation "image">`; annotations in
  `/data/PDD/artelingo/artelingo_train.json`; CLIP B/32 features via `load_artelingo()`.
- **Features** for CLIP, SigLIP 2, PE-Core and Qwen on the ArtELingo subset and on CUB:
  `/data/SSD2/pre_extract/backbone_check/<dataset>/<model>/`.
- **Existing caches the plan reuses** (local files, gitignored):
  - `src/test/20261016_factor_learning_grid/cache/{graph.npz, grid_prepare.npz}`;
  - `src/test/20261018_affect_factor_learning/cache/affect_prepare.npz`;
  - `src/test/20261023_aspect_episode_spike/results/aspect_episodes.npz` (for Task 4's reproduction);
  - the SE, C0 and R3 checkpoints, loaded through `run_affect.model_codes`.

## Rules that bite

- **Row scope:**
  - ArtELingo development reads **selection rows only**; training reads **scorer-train rows only**.
  - Val and held rows, the CUB test species and the GeneCIS templates are **never** read in E0 to E5.
  - Every evaluation array is NaN outside its scope, and scripts assert this.
- **Pre-registration first.** Task 12's `PREREGISTRATION.md` must be committed **before** any grid run. The final
  review checks the commit order.
- **Long jobs** (grid, extraction, MLLM probe, partition banks) are launched by the main session with
  `run_in_background`. Subagents write and smoke-test the scripts.
- **Git:** stage files by explicit path (other sessions share main; leave `bin/` untracked); one commit per task step
  marked Commit, with the session's attribution lines. Main is **15 commits ahead of origin** (not pushed); push
  only if the user asks.
- **Reports and logs:**
  - Every experiment ends with a report in `docs/reports/auto/v2/` plus one row in `docs/reports/reports_sum.md`,
    then `scripts/check_reports_sum.py`.
  - Report style: a real baseline beside every number, figures, paper-draft prose, no dashes.
  - Report and folder dates are a **sequence** (the next free values are 2026-10-29 onward, as the plan assigns), not
    calendar dates.
  - Edits to existing source (only `train_factors.py` in this plan) get a change-log entry in
    `.claude/<yyyymmdd>_log.md`.
- **Final review:** after Task 15 on a NO-GO, or Task 17 on a GO, run one whole-branch review on the most capable
  model. It re-derives the load-bearing numbers (spike reproduction, GO comparisons, K8, MLLM verdict) from stored
  arrays. Then one fix wave and a scoped re-review (`~/.claude/rules/final-review.md`). On this project it has found
  a real defect every time.

## Where things stand at handoff

- **Last commit:** `ee10038` (plan written). No plan task has started.
- **Calendar:** today is Sat Oct 3, so E0 should start now. The plan's dates are E0 Oct 3 to 5, E1 Oct 4 to 6, E2
  Oct 4 to 5, E3 Oct 5 to 9, and E4 Oct 5 to 12 in parallel.
- **Session history:** the planning session's chain of reports is indexed in `docs/reports/reports_sum.md` (v2 rows
  10-21 to 10-28) and summarized in spec §2.4 and §14.
