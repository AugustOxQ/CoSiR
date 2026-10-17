# Prompted aspect views: controller handoff

2026-10-17, CoSiR. No GPU extraction or prompted-view scoring has been run. Reference-only scores in results.json and tables.md are validation artifacts, not an answer to the premise question.

## Problem
Does changing Qwen3-VL-Embedding-2B's system instruction move image and caption representations toward ArtELingo emotion, genre or style? All aspect labels are used only for scoring. This is a train-side premise check, not a held-out result.

## Files and method
- `pav_prompts.py`: all ten exact instructions, shared between modalities.
- `pav_sample.py`: calls the earlier `load_sample(min_caps=3)` once, preserving its exact sample-ID order, then draws 5,000 painting IDs with `default_rng(0).choice(..., replace=False)`. Existing sample.npz is checked and reused. The frozen file contains original feature rows, caption sample IDs, dense and actual painting IDs, labels, first-caption indices, subset IDs and adapter image indices. No later script reselects the sample.
- `pav_extract.py`: import-safe controller entry point; loads one Qwen encoder for all unfinished selected keys, sets `enc.instruction`, encodes images in batches of 32 and captions in batches of 128 (outer chunks of 4,096). No labels are supplied to the encoder. One image per subset painting; all sampled captions. Outputs float16 img.npy/txt.npy, positional ids.npz and an atomic done.json marker under the requested external root. All IDs and the adapter image mapping are asserted. The existing extractor's finite/unit-norm gate (tolerance 0.01) runs before sealing. Completed keys are revalidated and skipped. Unsealed keys are recomputed; a sealed incompatible/corrupt key raises for inspection. Nothing is deleted.
- `pav_score.py`: CPU-only, thread pools limited to eight; all references use the frozen sample. MiniBatchKMeans n_init=3, batch_size=4096, seeds 0/1/2, default remaining sklearn settings. Caption emotion AMI at k=9/20/40. Painting views at k=20, scored against majority emotion (nine classes, ties to lowest ID), known genre (all negative codes excluded), and 27 styles. Mean-caption features use all sampled captions of each painting, then L2 normalisation. GoEmotions uses raw 28-d probabilities and stays caption-only. CSD/VGG retain the earlier row-to-image consistency asserts. CLIP text/image use the cached B/32 features.
- Probes: captions of the 5,000 paintings (25,160 captions), five-fold GroupKFold by painting; painting probes use five-fold GroupKFold with one item per painting. Training-fold mean and std+1e-6; logistic regression C=0.1, max_iter=300. Report mean/sd across fold accuracies, the global majority-class rate, and the fold-training majority classifier's test accuracy. No hyperparameter tuning.
- Pairwise partition AMI is computed separately for image and mean-caption views, including available references; same-view seed 0 versus seed 1 accompanies each matrix. `--keys` restricts Qwen keys, not reference rows. Missing/unsealed keys are skipped. Default is independently loaded for shift even when omitted from `--keys`; shift is unavailable if default is missing. References from other feature spaces have no meaningful cosine-to-Qwen-default shift.
- Buddy graphs: exact cosine top-20 neighbours, self excluded, mutual within each view. Strict is intersection, union is union of the two undirected edge sets. Report both single-view graphs too. Lift divides observed agreement by the exact agreement probability of two distinct uniformly drawn paintings. For genre, both graph edges and the random denominator are conditioned on two known labels. Empty/undefined rates are JSON null. CLIP image and mean-CLIP-caption views also get paired reference graphs; CSD/VGG have only an image side.
- Results checkpoint after each completed metric, include sample fingerprint and prompt strings, and regenerate tables. Repeated runs reuse completed metrics. Partial runs accumulate view results; pairwise matrices describe views included in the current completed invocation. Keep caches immutable; if changing encoder settings/model revisions, archive the old results before rescoring.
- `pav_cpu_smoke.py`: optional offline CPU-only two-image/four-caption shape and float16 norm check; writes no feature files.

## Commands for the controller
Run from /project/CoSiR. These GPU commands are provided for review; neither has been executed here. The smoke uses exactly two keys, 16 images and 64 captions, and writes only to `/tmp/claude-0/-project-CoSiR/pav_smoke/`. Full output is `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/<key>/`.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261017_prompted_aspect_views/pav_extract.py --smoke --device cuda > /project/CoSiR/src/test/20261017_prompted_aspect_views/gpu_smoke.log 2>&1
```

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261017_prompted_aspect_views/pav_extract.py --device cuda > /project/CoSiR/src/test/20261017_prompted_aspect_views/gpu_full.log 2>&1
```

About 12 minutes/key and two hours total, plus load/I/O/gates, using the supplied 11.5 images/s and 340 captions/s. Feature arrays alone total about 4.3 GB across ten keys, plus IDs. GPU smoke should take well under a minute on the measured GPU, plus model loading. Full commands intentionally run longer than the four-minute bound applied to agent-run checks; the controller runs them after review.

After extraction:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261017_prompted_aspect_views/pav_score.py > /project/CoSiR/src/test/20261017_prompted_aspect_views/score_full.log 2>&1
```

Scoring a selection: append `--keys default emotion_focus`, or `--no-references --keys default emotion_focus` to omit the cached references. `--stage caption` runs caption clustering only; default `--stage all` runs every metric. Full Qwen CPU scoring is designed for the approximately 45-minute budget with eight threads, but its elapsed time cannot yet be verified without the extracted arrays. No dimensions, seeds, folds, iterations or sample sizes are reduced to meet that target.

## CPU validation performed
Every Python invocation used `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Long-running checks were wrapped in `timeout 230`; their output went to files followed by a short tail.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_prompted_aspect_views/pav_sample.py > src/test/20261017_prompted_aspect_views/sample.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_prompted_aspect_views/pav_score.py --reference-only --stage caption > src/test/20261017_prompted_aspect_views/reference_caption.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_prompted_aspect_views/pav_score.py --reference-only > src/test/20261017_prompted_aspect_views/reference_full.log 2>&1
```

- Sample counts passed: 99,999 captions, 19,889 paintings, 5,000 subset paintings.
- Caption CLIP text AMI mean/sd at k=9/20/40: 0.036999/0.001390, 0.046664/0.001621, 0.051961/0.002508. Each target reproduced within 0.002.
- GoEmotions AMI mean/sd: 0.196560/0.001826, 0.209299/0.000799, 0.199374/0.001734. Each target reproduced within 0.002. The scorer_train probabilities map to ascending scorer_train rows; the selection cache's rows and sample IDs are asserted exactly, as in apc_check4.
- Full reference-only scoring completed in 46.7 seconds; checkpoint resume in 6.3 seconds. Results contain all five reference views, painting probes, pairwise partition matrices and CLIP buddy graphs.
- `import pav_extract` and `pav_extract.py --help` passed without loading a model.
- Synthetic checks passed: frozen ID maps, ten prompts, nine emotion/27 style classes, exact mutual neighbours versus an independent full argsort, lift denominator, missing genres and empty graphs.
- A fake encoder contract check passed: exactly one model construction across two keys, both instructions used in both modalities, float16/shape/ID gates, and a rerun that made no encoder calls. Fixtures are explicitly fake, stored only in `validation_fake_extract/`, and never used by scoring.

## Caveats
No aspect-instruction effects have been measured. The one-word instructions still request embeddings; no words are generated. These prompts test exactly the requested wording, including variants that omit mention of captions. GoEmotions is a supervised reference on emotion and is not an unsupervised backbone comparison. Painting emotion can have majority ties. Genre coverage is incomplete and is conditioned on known labels everywhere. Reported AMI sd is across seeds, not sample uncertainty. Probe accuracy averages folds equally, matching the earlier check. Solver warnings and per-fold iteration counts remain visible in logs. The fixed image subset is selected without labels. Cosine shift is a movement diagnostic, not evidence of moving toward a particular aspect without the label-based scores.

## To remove (not done)
Nothing was deleted. Optional cleanup after review: `validation_fake_extract/` (fake embeddings only), `extractor_contract.log`, `sample.log`, `reference_caption.log`, `reference_full.log`, `reference_resume.log`, `cpu_smoke.log`, and the three planning notes `task_plan.md`, `findings.md`, `progress.md`. Keep sample.npz, source files, results.json, tables.md and this handoff log. No commits or .ccg records were created.

## Optional CPU encoder smoke outcome
The real Qwen CPU smoke passed in 59.4 seconds with CUDA hidden and offline cache access. Shapes were (2, 2048) and (4, 2048); the float16 gate found no nonfinite rows or bad norms (maximum norm deviation 0.00001144). No embeddings were saved and no prompted-view scores were produced.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_prompted_aspect_views/pav_cpu_smoke.py > src/test/20261017_prompted_aspect_views/cpu_smoke.log 2>&1
```

Final source syntax and module-prefix checks passed after the last edits. The fake fixture ID/norm checks and full reference checkpoint reuse were rechecked. `reference_final.log` is also optional cleanup; retained here.
