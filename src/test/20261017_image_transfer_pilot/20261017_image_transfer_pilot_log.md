# Image transfer pilot: controller handoff

2026-10-17, CoSiR. No GPU was used. No Qwen model was loaded. Full student runs and the new extraction belong to the controller. All repository writes are in this round. Smoke arrays and scores are only in the requested /tmp smoke directory. Nothing was deleted or committed. No .ccg records were created.

## Question and fixed bar

Can a small image-only student learn the emotion geometry of prompted caption embeddings and beat the prompted model on images alone? Labels are used only by the scorer. They never enter training, early stopping, sample selection or training settings.

The prespecified bar is student emotion AMI >= 0.12, and teacher-student minus matched-control-student emotion AMI > max(the two student-seed sds). The scorer prints PASS or FAIL and the numbers after all three seeds of both targets exist. It also prints teacher-student genre/style changes versus the input's image baseline. It repeats this for emotion_focus input when stage 2 is complete. The bar is pending until full student outputs exist. Baseline scores alone do not answer the research question.

## Files and method

- `itp_data.py`: shared paths, identity fingerprint, name join, fixed painting split, caption targets and sealed training-image validation. Adds the repo root, pav round and apc round to sys.path. Calls pav validation with the exact frozen IDs and prompt. The default image store is joined by unique painting name for all 19,889 paintings. Both train and evaluation use those store rows. The subset cosine gate requires mean > 0.999 and min > 0.99.
- `itp_train.py`: Linear(2048,1024), GELU, Dropout(0.1), Linear(1024,2048), then L2 normalisation. Fixed symmetric in-batch InfoNCE at temperature 0.05. AdamW lr 0.001, weight decay 0.05, batch 512, up to 200 epochs. No augmentation, projection change or tuning. Each painting target is L2(mean of its caption embeddings). Caption means use the same float32 input and float64 accumulator as pav.
- The 5,000 fixed subset paintings are excluded from training and dev. Of the 14,889 remaining paintings, default_rng(0).permutation selects the first 1,489 for dev and the other 13,400 for fitting. This partition is fixed across seeds and targets. Each epoch evaluates all dev paintings as one batch in eval mode. Lowest dev loss selects the in-memory state. Fifteen epochs without improvement stops the run. No label or evaluation feature is used in the loss or stopping rule.
- Seeds 0/1/2 seed torch, numpy and shuffled batch order. Stage 1 is default image input with emotion_focus and default caption targets: six runs. Stage 2 substitutes emotion_focus image input: six more runs. Automatic input selection skips stage 2 if its sealed extraction is absent; explicit `--inputs emotion_focus` reports failure when absent.
- Full runs write `outputs/<input>__<target>__s<seed>.npy` (5000 x 2048 float16, fixed subset order) and a JSON with config, best epoch, epochs run, losses, history and timing. The best model stays in memory; no model checkpoint is saved. The output array is the final atomic run marker. Existing outputs are checked and skipped. A metadata mismatch raises instead of silently reusing a different experiment. CPU/GPU device changes alone do not invalidate resume.
- `itp_score.py`: imports pav `partitions`, `probe`, `summary`, `mutual_pairs` and `graph_stats` unchanged. K=20 with the same three MiniBatchKMeans seeds. Emotion majority, first genre and first style exactly match pav. Negative genres are excluded. The emotion probe keeps five GroupKFold folds, scaling and solver settings. ConvergenceWarnings remain in the logs. Exact cosine top-20 overlap excludes self and compares each view with both caption target spaces. Chance is 20/4999 = 0.00400080. The buddy graph uses pav mutual pairs and reports all three label lifts and pair counts.
- Scoring checkpoints after each metric and skips completed views. Sample IDs, prompts, labels and student output hashes protect cache reuse. Per-run JSON retains k-means and fold variation. Tables show baseline rows and input/target groups with mean and population sd across the three student seeds, for every numeric metric. The AMI bar uses each run's mean over k-means seeds, then mean/sd over student seeds. Incomplete seed groups are marked and cannot satisfy the bar.
- Baseline mean AMIs are compared with `pav/results.json` within 0.003 per label. All comparisons are printed and saved. Failures do not stop scoring. Seed-sd deltas are also recorded as diagnostics. The original pav results are never modified.
- `itp_extract.py`: lazy adapter and Qwen construction. Checks every painting's dense-to-adapter image mapping. Before encoding training images, re-encodes the first 32 subset images and requires mean cosine > 0.999 against the sealed emotion_focus image view. Uses the same Qwen constructor, instruction, image loader and batches of 32 as pav; closes images even on errors. Training paintings are in ascending dense-index order. It writes only `img.npy`, `ids.npz` and atomic `done.json` to the requested external emotion_focus_trainimg folder. check_arrays receives an empty (0,2048) float16 txt array before sealing. Sealed arrays, IDs, instruction, sample fingerprint and gate are verified and skipped. Unsealed partial files are recomputed in place. Sealed corrupt outputs raise for inspection. Nothing is deleted.
- `itp_queue.sh`: controller-only queue. Stage 1 train, CPU scoring, extraction, stage 2 train, CPU scoring. Each GPU step uses flock -o -w 3600 -E 75 /tmp/gpu0.lock. All steps record their exit code. CPU scoring hides CUDA. Final scoring still runs after extraction or training failure. One ITP_DONE line is sent to stdout and queue_status.txt, with all five codes and Amsterdam time. The queue exits nonzero if any step failed.
- `itp_checks.py`: CPU import/help, shell syntax, real join and synthetic overlap/split/bar/gate checks. Planning notes are task_plan.md, findings.md and progress.md.

## Exact controller commands

These full commands intentionally exceed the agent's four-minute check limit. They have not been run here. Run stage 1 alone from /project/CoSiR:

```bash
cd /project/CoSiR
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_train.py --inputs default --device cuda > src/test/20261017_image_transfer_pilot/train_stage1.log 2>&1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_score.py > src/test/20261017_image_transfer_pilot/score_stage1.log 2>&1
```

Extraction smoke: 16 training images plus the required 32-image subset gate. All writes go under /tmp/claude-0/-project-CoSiR/itp_smoke/extract/.

```bash
cd /project/CoSiR
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_extract.py --smoke --device cuda > src/test/20261017_image_transfer_pilot/extract_smoke.log 2>&1
```

Full queue, including resume of stage 1:

```bash
cd /project/CoSiR/src/test/20261017_image_transfer_pilot
setsid nohup ./itp_queue.sh > queue.log 2>&1 < /dev/null &
```

## CPU checks run

Every Python command used /root/miniconda3/envs/CoSiR/bin/python with PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''. Each was wrapped in timeout 230. Long output went to the check logs here. Checks never ran the queue, loaded Qwen or acquired the GPU lock.

From /project/CoSiR, with the four variables exported:

```bash
timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_checks.py --mode imports > src/test/20261017_image_transfer_pilot/check_imports.log 2>&1
timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_checks.py --mode join > src/test/20261017_image_transfer_pilot/check_join.log 2>&1
timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_checks.py --mode synthetic > src/test/20261017_image_transfer_pilot/check_synthetic.log 2>&1
timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_train.py --smoke --device cpu > src/test/20261017_image_transfer_pilot/check_train_smoke.log 2>&1
timeout 230 /root/miniconda3/envs/CoSiR/bin/python src/test/20261017_image_transfer_pilot/itp_score.py --baselines-only --results /tmp/claude-0/-project-CoSiR/itp_smoke/baselines.json > src/test/20261017_image_transfer_pilot/check_baselines.log 2>&1
```

- Import of every itp module, --help of every entry point and bash -n passed in 10.5 s. The test also verifies that the Qwen implementation was not imported.
- Real join passed in 3.3 s. Shape is (19889,2048), float32, finite and unit-norm. Subset cosine mean 0.99992734; minimum 0.99856031.
- Synthetic exact top-20 overlap passed in 3.0 s. Identical spaces give 1.0. Two random spaces give 0.23373494, exactly matching independent full argsort and Python set intersection. The selected neighbour sets match argsort. Split disjointness and deterministic dev selection passed. Synthetic seed aggregation and PASS/FAIL boundaries passed.
- CPU training smoke used 2,000 fitting paintings from the fixed training partition, plus the full fixed 1,489 dev paintings. Seed 0, default input, emotion_focus target, exactly three epochs. Training loss: 5.738410 -> 4.619121 -> 4.150272. Dev: 5.984909 -> 5.410388 -> 5.177481. Best epoch 3. The timed training/evaluation part was 0.64 s, excluding data loading. Saved outputs are finite and unit-norm. Only the designated /tmp/train directory was written.
- All four full baseline views scored in 80.8 s. No reduced view check was needed. Each includes three k-means seeds, the full five-fold emotion probe, both exact overlaps and the buddy graph. ConvergenceWarnings were retained. Mean AMIs (emotion / genre / style): default_image 0.069634 / 0.448597 / 0.221676; ef_image 0.092876 / 0.409632 / 0.167824; teacher 0.217173 / 0.243924 / 0.100889; control_target 0.107478 / 0.337556 / 0.120231.
- The initial consistency implementation also applied 0.003 to seed sd. This was stricter than the requested per-label mean AMI comparison. It falsely flagged default_image style sd (delta 0.007474). Corrected the gate to compare means and keep sd as a diagnostic. Default-image mean deltas are 0.001068 / 0.000010 / 0.000081. All other baseline means and sds reproduce exactly. No training choice or threshold changed. check_baselines.log preserves the original diagnostic; the final comparison is in check_baselines_resume.log.

## Time and storage estimates

Planning estimates, not GPU measurements: about 2-10 minutes per six-run training stage, depending on early stopping and GPU load. CPU smoke does not predict GPU throughput reliably. Full emotion_focus extraction is about 22 minutes at the earlier 11.5 images/s, plus model loading and gate overhead; allow 25-30 minutes. Extraction smoke should take under a minute plus model loading. Scoring the four baselines took 81 s on this shared CPU. Allow roughly 2-4 minutes for six new student views per stage, plus baseline loading and other CPU users. Lock contention can add up to an hour per GPU step.

Each full student output is about 20.5 MB decimal; six outputs about 123 MB and twelve about 246 MB. The new external image extraction is about 61 MB plus IDs. No model checkpoints are saved.

## Caveats

This is a single frozen, train-side research pilot, not a new held-out dataset claim. Caption targets can contain explicit emotion words. K-means seed variation and student seed variation are different; both are retained. Probe fold variation is not uncertainty over independently sampled paintings. The default store is a near-identical re-encoding of the default subset view; its mean gate passes, but individual k-means partitions and seed sd can change. Target-space overlap is a geometry check, not itself an emotion score. No full student score or optional extraction has been produced here. Reusing the smoke scorer cache as production results would be inappropriate; the controller's scorer uses a separate round-local results path.

## To remove (not done)

Nothing was removed. Optional cleanup after review: the check_*.log files and task_plan.md, findings.md, progress.md in this round. The designated /tmp/claude-0/-project-CoSiR/itp_smoke/ tree contains validation outputs and can be removed by the controller later. Keep source scripts, this handoff log and all eventual production outputs/results. A corrupt or incompatible sealed extraction should be inspected and moved aside by the controller before rerunning; this code never deletes it.

Final validation: baseline checkpoint resume finished in 21.7 s. All four views were skipped as scored and all four mean-AMI consistency gates passed. No metrics were recomputed or reduced. Tables were regenerated with baseline rows and the full scalar breakdown. Synthetic checks were rerun after the gate correction in 3.0 s, including explicit pass/fail cases around the 0.003 mean tolerance. Final import/help and shell syntax checks passed. The smoke score cache remains outside the round; production itp_results.json and itp_tables.md will be created by the controller's scoring command.

## Full run (controller, 2026-10-17)

- Reviewed by the controller (training reads no labels; dev slice and early stop use the InfoNCE loss only; the
  scorer reuses the pav functions). Launched 05:40 with `itp_queue.sh` in a background Bash. Ended 06:31:
  `ITP_DONE train_stage1=0 score_stage1=0 extract=0 train_stage2=0 score_stage2=0`.
- Each student run took 1.4 to 3.6 s on the GPU. The training-image extraction ran at 10.2 images/s (1,460 s); its
  32-image subset gate gave cosine 1.0000.
- Consistency gates: all four baselines PASS (largest difference 0.0011, default_image, which reads the store rows).
- Bar: FAIL for both inputs. Default input: student 0.0740 ± 0.0004, control 0.0739 ± 0.0016. emotion_focus input:
  student 0.0736 ± 0.0031, control 0.0750 ± 0.0016. Numbers read from `itp_results.json`.
- Exploration after the fail (controller, `itp_knn.py`, log `knn.log`, `itp_knn_results.json`; CPU, 139 s): mean
  teacher target of the k nearest training images. k = 20: default input 0.091 (control targets 0.070); emotion_focus
  input 0.102 (control 0.083). k = 100 gives about the same.
- Report: `docs/reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md`.
