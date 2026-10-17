# Emotion word masking: controller handoff

2026-10-17, CoSiR. Single agent. No GPU use. No commits, deletions or CCG records. Source and real mask artifacts are in this folder. CPU scoring and fake encoder fixtures are under the explicitly requested `/tmp/claude-0/-project-CoSiR/ewm_smoke/` path. The queue and real encoder smoke have not been run.

## Question

Does the emotion_focus caption AMI lift survive removal of explicit emotion words? The earlier k=20 values were 0.121728 for default and 0.278 for emotion_focus. Masking and feature construction use no labels. Labels enter word-agreement statistics and embedding scoring only.

## Files and method

- `ewm_masks.py` defines the fixed lexicons and regexes. It loads `pav_sample.load_frozen()` and captions from `adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)`. It checks sample and painting IDs. Each regex uses Unicode whole-word boundaries and case-insensitive matching. Alternatives are longest first. Each match becomes the literal `[MASK]`. All other characters stay unchanged. Both masks are fixed before any masked embedding scores.
- `masked_captions_label.json` and `masked_captions_broad.json` contain all 99,999 masked strings in frozen sample order. JSON files are gitignored. `ewm_masks.npz` contains sample IDs, boolean `has_label_word` and `has_broad_word`, and uint32 `label_word_counts` with shape (99,999, 9). It also records the emotion names and source, masked-list and mask-spec hashes. `ewm_word_statistics.json` is the real word-only report.
- The emotion map is asserted from `src/data/artelingo_splits.py:encode_labels`, which uses sorted names, and from every raw annotation indexed by the frozen sample ID. `apc_data.load_sample` uses that encoder without catch-all exclusion. The map is 0 amusement, 1 anger, 2 awe, 3 contentment, 4 disgust, 5 excitement, 6 fear, 7 sadness, 8 something else. The assertion runs in scoring and word statistics.
- `ewm_extract.py` loads one Qwen3VLEmb for all pending views. It switches `enc.instruction` between the exact `PROMPTS['default']` and `PROMPTS['emotion_focus']`. It first encodes the first 256 unmasked captions with each instruction. It prints and requires mean cosine >0.999 against the stored float16 embeddings. These are exactly 256-caption calls. Their length-sorted batches can differ from the original 4,096-caption chunk, so the cosine gate is required.
- Full extraction encodes 99,999 masked captions for each of two masks and two keys. Outer chunks are 4,096, with `encode_texts(..., batch_size=128)`. Full output is `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/masked_<mask>/<key>/`. Each view has float16 `txt.npy` (99,999, 2048), `ids.npz` from `identity(s, 0, 99999)`, and an atomic `done.json`. Empty image identity fields are retained. No image features are written. `check_arrays` receives an empty (0, 2048) float16 image array. The seal records instruction, mask, list SHA256, mask-spec SHA256, source SHA256, shape, dtype, timing, reproduction cosines and gate.
- A sealed masked view is verified against its instruction, mask hash, spec, IDs, actual shape/dtype and a fresh finite/unit-norm gate. An incompatible seal raises. An unsealed view is recomputed in place. Nothing is deleted. `--smoke` encodes 64 captions, two keys and the label mask. All smoke writes, including mask artifacts, are isolated in the requested /tmp smoke folder. It also runs both 256-caption reproduction checks.
- `ewm_score.py` imports `pav_score.partitions`, `probe` and `summary` directly. Caption features are float32 casts of the stored float16 arrays, without extra normalisation. Caption emotion AMI uses k=9/20/40 and three seeds. Mean-caption painting features and labels exactly follow `pav_score.main`: aggregate all sampled captions per painting, select `subset_pid`, L2 normalise, and cluster at k=20. Labels are majority emotion over nine classes, first genre and first style. Negative labels are excluded from AMI. The fixed subset has 5,000 paintings and 25,160 captions.
- The four masked probes call `probe(cap[cm], s['emo'][cm], s['pid'][cm])`. This is five-fold GroupKFold by painting, train-fold scaling, LR C=0.1 and max_iter=300. Unmasked probes are copied verbatim from `pav/results.json`, with their source stated in JSON and tables.
- No-word subgroups use the two unmasked keys. Each subgroup is clustered on its own at k=9/20/40. Sizes and nine-class counts/shares appear beside the full distribution. Caption agreement means that at least one matched label word maps to the caption's own label. Captions with several distinct emotion matches count once in that denominator. Top-word counts are occurrences, lowercased. `awe-inspiring` counts as one match.
- `ewm_results.json` checkpoints after each completed metric. `ewm_tables.md` is regenerated with every checkpoint. A checkpoint binds the frozen sample, labels, prompts, source captions, masks and prior baseline scores by hash. Finished metrics are skipped. Missing masked views are reported and skipped. Unmasked caption and painting mean/sd are compared to the earlier results within 0.002. Failures are printed and recorded; scoring continues. Full round results are left for the controller. CPU validation results are only in /tmp.
- `ewm_queue.sh` runs GPU extraction under `flock -o -w 3600 -E 75 /tmp/gpu0.lock`, then CPU scoring with CUDA hidden. Scoring runs even if extraction fails. Logs are `extract.log` and `score.log`. One `EWM_DONE extract_exit=... score_exit=... <Amsterdam date>` line goes to stdout and `queue_status.txt`. The queue exits nonzero if either stage fails.
- `ewm_checks.py` checks masking units, all module imports and each Python script's `--help`. A Qwen constructor sentinel prevents accidental model loading. Its optional fake encoder contract checks one construction, instruction switching, both reproduction checks, shapes, gates, seals and skip on rerun. Fake features are retained under `/tmp/.../ewm_smoke/fake_contract_*/` and never scored as real views.

SHA256 for caption lists means UTF-8 JSON with `ensure_ascii=False` and compact separators, with list order preserved. Hashes describe string content, not the indentation of the saved JSON file.

## Exact fixed masks

Label mask: 44 forms. Each group maps to the named label.

```text
amusement: amusement, amused, amusing, amuse, amuses
awe: awe, awed, awestruck, awe-inspiring, awesome
contentment: contentment, content, contented
excitement: excitement, excited, exciting, excite, excites
anger: anger, angry, angered, angrily
disgust: disgust, disgusted, disgusting, disgusts
fear: fear, fears, feared, fearful, afraid, scared, scary, scare, scares, frightened, frightening, terrified, terrifying
sadness: sadness, sad, saddened, saddening, sadly
```

Broad mask: all label forms plus these 88 forms, for 132 total.

```text
happy, happiness, happily, joy, joyful, joyous, glad, cheerful, delight, delighted, delightful, calm, calming, calmness,
peaceful, peace, serene, serenity, relaxed, relaxing, pleasant, pleased, love, loving, lovely, lonely, loneliness,
depressed, depressing, depression, gloomy, melancholy, melancholic, upset, upsetting, worried, worry, anxious,
anxiety, nervous, uneasy, horror, horrified, horrible, horrifying, creepy, eerie, ominous, disturbing, disturbed,
gross, funny, fun, laugh, laughing, nostalgic, nostalgia, hopeful, hope, hopeless, grief, sorrow, sorrowful, mournful,
tragic, cozy, comforting, comfortable, inspired, inspiring, curious, curiosity, intrigued, intriguing, bored, boring,
annoyed, annoying, irritated, irritating, uncomfortable, unsettling, emotional, feel, feeling, feelings, feels, felt
```

The feel forms are deliberate. There is no label-mask group for the catch-all phrase. No extra synonyms or stems are added.

## Commands for the controller

These commands have not been executed. First run the real GPU smoke:

```bash
cd /project/CoSiR
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 timeout 230 flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_extract.py --smoke --device cuda > src/test/20261017_emotion_word_masking/gpu_smoke.log 2>&1
```

Inspect both reproduction cosines and the two seals. Then launch the full queue:

```bash
cd /project/CoSiR/src/test/20261017_emotion_word_masking
setsid nohup ./ewm_queue.sh > queue.log 2>&1 < /dev/null &
```

The queue changes to `/project/CoSiR` and exports the required Python/thread environment. Full extraction is deliberately not bounded to four minutes; the controller runs it. The shared GPU lock can wait up to one hour.

For CPU scoring before or after extraction:

```bash
cd /project/CoSiR
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_score.py > src/test/20261017_emotion_word_masking/score.log 2>&1
```

This writes the round's checkpoint and tables. It works with only word statistics and unmasked views available. `--stage words` writes just word statistics to that checkpoint. `--stage caption`, `--ks 9`, `--keys default`, `--texts unmasked`, and `--no-subgroups` can limit a check. Use a separate `--results` path for smoke results; artifacts and tables are written beside it.

## CPU checks performed

Every check used `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''`, under `timeout 230`. Output was redirected to local logs and only short tails were printed. No real encoder was constructed.

```bash
cd /project/CoSiR
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_checks.py --contract > src/test/20261017_emotion_word_masking/checks_final.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_masks.py > src/test/20261017_emotion_word_masking/word_statistics_timed.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_score.py --keys default --texts unmasked --no-subgroups --results /tmp/claude-0/-project-CoSiR/ewm_smoke/ewm_results.json > src/test/20261017_emotion_word_masking/unmasked_smoke.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_emotion_word_masking/ewm_score.py --keys default --texts unmasked --results /tmp/claude-0/-project-CoSiR/ewm_smoke/ewm_results.json > src/test/20261017_emotion_word_masking/subgroup_smoke.log 2>&1
```

- Masking units, all four imports/help commands, and fake extraction contract passed in 3.2 seconds. Tests cover contentment/content versus contents/discontent, sadly versus sadler, mixed case, one-token awe-inspiring, all fixed forms, feel forms, repeated matches, nine-column counts, conditional own-label agreement, and unchanged UTF-8 bytes without matches.
- The first extra fake contract attempt failed because the test restored all of `sys.modules` and unloaded Torch wrappers. Restoring only the mocked encoder entry fixed the harness. The failed `checks.log` and passing retry logs are retained. This did not load a Qwen model.
- Real word statistics and codebook assertions passed in 10.6 seconds. Label matches: 13,195 / 99,999 = 13.1951%. Broad matches: 40,586 / 99,999 = 40.5864%. Own-label agreement: 11,615 / 13,195 = 88.0258%. The two top-ten lists are printed in `word_statistics_timed.log`; complete top-30 and per-emotion shares are in `ewm_word_statistics.json`.
- Default unmasked caption and painting checks passed in 57.2 seconds. All six comparisons matched the prior mean and sd exactly. Caption means for k=9/20/40 were 0.098292625, 0.121728031 and 0.128728071. Mean-caption painting emotion/genre/style means were 0.107478392, 0.337555612 and 0.120230862. The unmasked caption probe was copied from the earlier results.
- Checkpoint resume passed in 11.8 seconds. Clustering and probe functions were replaced by raising sentinels for this check. No completed metric was recomputed. Both missing default masked views were reported and skipped. See `resume_smoke.log`.
- The default-key subgroup pass completed in 62.7 seconds. It reused the six completed baseline metrics and computed all six subgroup AMIs. No broad word: n=59,413; k=9/20/40 means 0.055716449, 0.070157667, 0.084550509. No label word: n=86,804; means 0.086873688, 0.102332012, 0.104352713. Full mean/sd and label distributions are in the /tmp checkpoint and tables. All six baseline gates still passed. The emotion_focus subgroup scores are left for the controller.
- `bash -n ewm_queue.sh` passed. All agent-run checks stayed within their 230-second bounds. The only initial failure was the optional fake test harness issue above.
- Final artifact validation passed in 0.2 seconds. All four Python files parse and have the ewm_ prefix. There is no __pycache__. Both saved caption-list hashes match the archive. Flags are boolean and the count matrix is (99,999, 9). The smoke checkpoint contains six passing baseline gates, six subgroup AMIs and both missing-view records. The final shell syntax check also passed.

## Time and storage estimates

At about 267 captions/second, each 99,999-caption mask/key pass takes 374.5 seconds, about 6.2 minutes. Four passes take about 25 minutes, plus model loading, reproduction and I/O/gates. Budget about 26–30 minutes on the earlier GPU. The smoke encodes 640 captions including reproduction, plus model loading and baseline verification. Budget under four minutes after acquiring the lock.

CPU scoring is estimated at 10–20 minutes with eight threads. This is an estimate from the default baseline and prior round. Four grouped caption probes are the main unmeasured cost. Solver warnings remain in the logs. Full scoring has no agent-imposed timeout in the controller queue.

Four feature arrays total about 1.64 GB decimal. ID archives add about 214 MB. Each masked caption JSON is about 8 MiB. The compressed mask flags/counts archive is about 370 KiB. Smoke fixtures and CPU checkpoint are separate in /tmp.

## Caveats

Masked embedding results and real reproduction cosines are pending the controller GPU run. The present word counts do not answer whether the lift survives. This is one train-side sample and one model size. Lexicons miss other emotion wording. Some listed forms have non-affective senses, including content, love and feel. `[MASK]` is literal input text; it is not a trained masked-language-model operation. Masking can affect fluency and sentence meaning. The no-word subgroups have different label distributions and are not randomised controls. Their tables include those distributions. Matching an own label means any own-label match, even if other emotion words occur too. AMI sd is variation across clustering seeds. Probe sd is variation across folds. Neither is sample uncertainty. The instructions and masks are fixed; do not change them after inspecting masked scores.

## To remove (not done)

Nothing was deleted. Optional cleanup after review: `checks.log`, `checks_retry.log`, `checks_final.log`, `word_statistics.log`, `word_statistics_timed.log`, `unmasked_smoke.log`, `resume_smoke.log`, `subgroup_smoke.log`, and `task_plan.md`, `findings.md`, `progress.md`. The controller may also remove `/tmp/claude-0/-project-CoSiR/ewm_smoke/`, including the retained fake-contract fixture directories and CPU smoke checkpoint/tables. Keep the source, real masked captions, real mask archive, word report and this log. Keep full results/tables and real extraction seals after the controller run.

## Full run (controller, 2026-10-17)

- Reviewed by the controller (whole-word masks fixed before the run; the scorer reuses the pav functions). Launched
  05:41 with `ewm_queue.sh` in a background Bash. Ended 06:15: `EWM_DONE extract_exit=0 score_exit=0`.
- Unmasked re-encode gate: cosine 0.99991 (default) and 0.99987 (emotion_focus). Four masked views sealed, 230 to 337
  captions/s. Scoring took 632 s. All 12 unmasked consistency gates PASS with difference 0.
- Key numbers (from `ewm_results.json`): caption emotion AMI k = 20, default / emotion_focus: unmasked 0.122 / 0.278;
  label mask 0.095 / 0.231; broad mask 0.069 / 0.187; unmasked on the 59,413 captions with no broad word 0.070 /
  0.197. Label words in 13.2% of captions, broad words in 40.6%.
- Report: `docs/reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md`.
