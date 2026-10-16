# Aspect premise checks (ArtELingo cached features)

Started and finished 2026-10-17 (Amsterdam). CPU only. Nothing decided. All numbers are in `results.json` (merged by `apc_merge.py` from `results_check*.json`).

## Problem
Premise checks for the aspect idea: do caption residuals, different feature views and ArtELingo-28 language groups carry separable partitions?

## Method
- Sample: ArtELingo train side of the painting-grouped split (scorer_train + selection rows, 216,107 captions). Paintings drawn at random (seed 0) until at most 100k captions (whole paintings). Joins use the `FeatureManager` sample IDs via `load_artelingo()` row order; style caches map by `row_to_image.npy[row]`, checked to be one image per painting and distinct across paintings.
- Check 1 sample: paintings with >= 3 captions (so the residual is not degenerate): 99,999 captions, 19,889 paintings. Check 2: no minimum: 100,000 captions, 19,896 paintings. Check 4 = check 1 sample.
- Labels: emotion has 9 classes incl. 'something else'. Painting emotion = majority of its sampled captions (ties to lowest id). Genre = WikiArt genre (ArtGAN CSVs), 10 classes, known for 16,127 of 19,889 paintings (check 1), others dropped from genre AMI. Style = WikiArt art_style, 27 classes, all paintings (available in the metadata).
- k-means: MiniBatchKMeans, n_init 3, batch 4096; AMI is mean +- sd over seeds 0, 1, 2.
- Residual: t = unit CLIP text; r_mean = normalise(t - mean over the painting's captions of t); r_proj = normalise(t - (t.v)v) with v the unit CLIP image feature.
- Probe: logistic regression (standardised, C = 0.1, 300 iterations), 5-fold GroupKFold by painting.
- Check 3: `read_set_a28.parquet`, JS = Jensen-Shannon, log base 2, mean over language pairs per painting; random = 20 permutations of the caption-to-language assignment inside the painting (group sizes kept).
- Check 4: only 28-d GoEmotions classifier probabilities are cached (not hidden features): archive `20261018_affect_factor_learning/cache/affect_prepare.npz` (scorer_train rows) and `20261123_idea3_goemotions/cache/r5_goemotions_selection.npz` (selection rows) together cover all train rows. No new features computed.

## Scripts
`apc_data.py` (loading, sampling), `apc_check1.py`, `apc_check1_probe.py`, `apc_check2.py`, `apc_check3.py`, `apc_check4.py`, `apc_merge.py`. Run with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python <script>` from this folder. Logs: `check*.log`.

## Numbers
See `results.json`; the tables are in the hand-back report. Key readings:
- Caption-level emotion AMI from raw CLIP text is low (0.04 to 0.05); the mean-subtracted residual is lower (0.02 to 0.03) while a linear probe on it still reaches 46.6% (chance 28.4%, raw t 57.3%). Projection residual r_proj is close to t.
- Genre and style AMI come mainly from the image view and fall to near zero for the mean-subtracted residual.
- Four views at k=20 give different partitions (pairwise AMI 0.11 to 0.36; same view, two seeds: 0.54 to 0.63).
- ArtELingo-28 read set: 746 paintings, 74,569 captions, 27 non-English languages; JS between languages 0.532 vs 0.480 for random same-size groups.
- GoEmotions probabilities (supervised reference): caption emotion AMI 0.197 / 0.209 / 0.199 at k = 9 / 20 / 40.

## Caveats
- The brief's "about 1.3k paintings with genre" is the English release `artelingo_genre_emotion_eng.json` (1,303 paintings, 6,515 captions); the a28 read set has 746 paintings and carries genre for all rows.
- The raw json files are English only (no language field); the a28 parquet has no English rows. 410 of its 746 paintings have English captions in the raw json (2,088 captions); an English group added changes JS little (see results.json).
- a28 emotion code -1 is read as 'something else' (inferred from the sorted-name coding in `encode_labels`; not confirmed against the extraction code); counted as a 9th bin.
- Probe uses a fixed C (no tuning), standardised features; accuracy is across painting-grouped folds. The k-means sd is across 3 seeds on one fixed sample, not sample variance.
- Genre labels cover 81% of check-1 paintings (16,127 of 19,889).
- JS: small groups (about 4 captions per painting and language) make JS large for both real and random; read the difference, not the level.
- Painting majority emotion has many ties (5 captions, 9 classes).

## To remove (not done)
Nothing needed; `__pycache__` was not written (PYTHONDONTWRITEBYTECODE=1). `check*.log` can be removed after reading.
