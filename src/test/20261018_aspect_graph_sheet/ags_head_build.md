# r10 ags 02 build log

Ticket: `.scratch/r10-aspect-graph-sheet/issues/02-head.md`.
Branch: `r10-ags-02-head`. Code: `ags_head.py`, `ags_queue.sh`.

## Completed work

Read the ticket, index, exact rows and cited spec in order, then the sibling pilot modules and
`ags_trainimg.py`. Implemented the cross-modal head, verified resume, cached AMI scoring,
reference views, Markdown/JSON sheets, seed selection and the controller queue.
The queue runs three aspect workers concurrently in each training phase; each worker appends its
exit code immediately. Scoring runs after training failures. Its final status line contains all
step codes and the Amsterdam timestamp.

Planning, findings and progress are consolidated in this ticket log to keep the requested three-file
deliverable. No unit tests, other models, CCG records, or changes in sibling round folders.
All smoke artifacts are isolated in:
`/tmp/claude-0/-project-CoSiR/c1d9190f-c197-43c9-bb25-2692e4200e97/scratchpad/ags_smoke/`.

## Definitions settled

- Follow `itp_train.CONFIG` and `Student` unchanged; two separate MLPs. Concatenate the two endpoints
  for each MLP call; dropout remains independent per element.
- Loss is the sum of the mean of four buddy terms and the mean of two own-caption terms.
  Every buddy CE direction uses sum(w*ce)/sum(w); own-caption terms are unweighted.
- Epoch train/dev summaries are batch-size-weighted means of batch losses, as in `pbp_train`.
  Dev order and batch boundaries are fixed, dropout disabled. Labels never enter training or stopping.
- Output npz is the completion marker. JSON is written before sealing; resume checks exact config,
  graph archive SHA256, output SHA256, shapes, dtype, finite values and norms.
- Per-output scores keep the mean and population sd of three k-means seeds. Row means average the
  per-output means. Student sd is reported only with three student seeds (`null` otherwise).
  Target sd uses each run's two-side mean, preserving covariance. Exact selection ties retain
  the graph selection order. A failed/missing seed-0 score prevents seed selection.
- Graph manifest uses ticket 01's exact convention: `identity.sample_sha256`, `identity.method`,
  `archives[aspect]`, `counts[aspect][side][row]`. Ticket 01 appeared during this build; the initial
  assumed row/side nesting was corrected in the head and fixture metadata before subsequent runs.
  The archive SHA and per-output config were unchanged, and existing smoke output resume passed.
- Scores verify every completed output before cache reuse and rehash reference vectors; cached
  scoring avoids all k-means work. Missing-output reporting expects seed 0 plus the picked rows'
  seeds 1/2 in a full run; the prescribed smoke expects seed 0 only.

## Commands

Every Python command below used this exact prefix (CPU only):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python
```

`<round>` = `src/test/20261018_aspect_graph_sheet`; `<smoke>` = the scratchpad above.
Each substantive execution was wrapped in `timeout 235`, keeping commands below four minutes.

```text
python <round>/ags_head.py --help                          # ags_head_help.log in round
bash -n <round>/ags_queue.sh                              # passed
python <smoke>/head_fixture.py                            # head_fixture.log
python -u <round>/ags_head.py --smoke --rows R0            # head_train_R0.log
python -u <round>/ags_head.py --smoke --rows R1 Gdown025   # head_train_remaining.log
python -u <round>/ags_head.py --smoke --score              # head_score.log
python -u <round>/ags_head.py --smoke --pick-seeds         # head_pick_seeds.log
python -u <round>/ags_head.py --smoke --rows R0 --time-epoch # head_full_epoch.log
python -u <round>/ags_head.py --smoke                      # head_resume.log
python -u <round>/ags_head.py --smoke --score              # head_rescore.log
```

Logs other than help are in the smoke folder. No full queue was launched; that is controller-owned
and requires integrating ticket 01. Initial read-only inspection found no root `AGENTS.md` file
(the user supplied its instructions), and ticket 01's code was initially absent; neither blocked
implementation. No smoke command failed.

## Fixture and training numbers

The fixture copies pilot `graphs/{fit,dev}.npz` E as R0, P as R1, and E as Gdown025 with weight .25
on EC pairs. Original source: `/project/CoSiR/src/test/20261018_purified_buddy_pilot/graphs/`.
Full node order is preserved: 13,400 fit, 1,489 dev. Smoke uses the first 5,000 fit pairs per row,
then a seeded shuffle each epoch; all dev pairs and all 5,000 evaluation paintings are used.

| Row | Full fit/dev pairs | Train loss e1 → e2 | Dev loss e1 → e2 | Best epoch | Epoch seconds | Run seconds |
|---|---:|---|---|---:|---|---:|
| R0 | 57,727 / 7,413 | 10.805952 → 7.928528 | 9.630940 → 8.826898 | 2 | 1.81 / 1.73 | 3.86 |
| R1 | 45,505 / 4,909 | 10.785647 → 8.391137 | 10.149917 → 9.373834 | 2 | 2.15 / 2.30 | 4.86 |
| Gdown025 | 57,727 / 7,413 | 10.923647 → 8.138241 | 10.070728 → 9.172200 | 2 | 2.52 / 1.97 | 4.93 |

Gdown025 has 12,222 fit / 2,504 dev EC pairs; weight sums 48,560.5 / 5,535.0.
Default-store join cosine mean .99992734, min .99856031.
All six saved arrays passed float16 (5,000, 2,048), finite and unit-norm checks (tolerance .01).
All graph and output checksums verified. Identical-output resume skipped all three rows.
Training run seconds exclude input loading and include inference/save; epoch seconds include dev.

## AMI sheet (seed 0; mean of three k-means seeds)

| Row | img/emotion | img/style | img/genre | cap/emotion | cap/style | cap/genre | Target emotion AMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | .085330 | .179773 | .423202 | .130386 | .131546 | .337337 | .107858 |
| R1 | .083854 | .174919 | .411329 | .134345 | .129675 | .339507 | .109100 |
| Gdown025 | .087947 | .182073 | .422685 | .134443 | .125324 | .331309 | .111195 |

| Reference | Emotion AMI | Style AMI | Genre AMI |
|---|---:|---:|---:|
| default/img | .069634 ± .001989 | .221676 ± .002409 | .448597 ± .003525 |
| default/cap | .107478 ± .006510 | .120231 ± .003478 | .337556 ± .004927 |
| emotion_focus/img | .092876 ± .001739 | .167824 ± .010241 | .409632 ± .008866 |
| emotion_focus/cap | .217173 ± .003044 | .100889 ± .003607 | .243924 ± .015294 |

Initial scoring: 58.72 s; cached rescore: 24.17 s, with all three outputs cached and no k-means
rerun (reference loading/hashing remains). Both durations exclude frozen-sample loading.
`--pick-seeds` produced `{"emotion": ["Gdown025", "R1"]}`. Missing smoke outputs: none.
Results: `<smoke>/ags_head_results.json`, `ags_head_tables.md`, `head_seed_rows.json`.

## Full-size epoch timing and handoff

A fresh seed-0 R0 model trained on **all 57,727 fit pairs**, then evaluated on **all 7,413 dev pairs**:
**14.890842 s/epoch**, including dev; **14.385274 s** training. Batch 512, CPU, 8 threads.
Train loss 6.738660, dev loss 7.736261. This is a measured full shuffled pass, not an extrapolation.
Stored in `<smoke>/head_full_epoch_timing.json`; the timing run does not replace smoke embeddings.
Concurrent ticket 01 work may affect shared-machine throughput; three concurrent aspects may differ.

No implementation issues remain identified. Full multi-aspect training, seeds 1/2, and queue execution
are left to the controller after ticket 01 integration; smoke validates the requested emotion fixture.
Existing untracked `src/test/20261017_prompted_aspect_views/sample.npz` remains untouched.
Deleted files: none. Removal candidates: none. Generated logs and scratchpad artifacts are retained.
Only the two code files and this build log are staged for the ticket commit; no npz/npy/json outputs.
