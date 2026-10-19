# r10 model sheet 02 build

Ticket: trainer core, branch `r10-ms-02-trainer`.

## Plan and progress

- Read binding index, ticket, spec from section 1, and reference code: complete.
- Implement rows, model construction, per-head loss, joint loop and CLI: complete.
- Run full-pair genre two-epoch regression and T1 smoke: complete, exact regression passes.
- Complete and partial resume: complete (details below).
- Record numbers and review the three-file diff: complete; committed with the ticket code.

Planning-with-files notes are consolidated here to respect the ticket's write scope.
No agents, CCG files, unit tests, or edits to earlier round code.

## Findings and definitions settled

- CPU form a calls `ags_head.pair_loss` unchanged, preserving endpoint concatenation and CE order.
- Separate runs construct img Student then cap Student after reseeding, matching stage 1 draws.
- Shared runs construct each side's trunk then aspect layers in emotion/style/genre order.
- `build_model` returns the optimizer-owned model and per-head views onto its parameters;
  this gives ticket 03 one construction seam and one `head_loss` seam to extend.
- Joint batches sum losses in aspect order, with no extra arithmetic for a single head;
  this preserves the CPU single-head operation sequence.
- Each head's epoch ends after its own full shuffled fit list, including its final short batch;
  this lets unequal graph lengths progress independently, with no wraparound within a step.
- Dev is fixed-order, eval-mode and batch-size weighted; strict improvement deep-copies just
  the head's two side layers and shared trunks, so each output uses its own best trunk state.
- The single-head shuffle uses `default_rng(seed)`, joint shuffles use
  `default_rng([seed, aspect_index])`; both NumPy and torch are seeded before model construction.
- CUDA uses the same loss expression with device-local indices, weights and CE labels;
  CPU calls the original `ags_head.pair_loss` directly to avoid rounding changes.
- Partial shared resume must replay the original training cohort to preserve the joint trajectory;
  verified files stay untouched, and a fully completed cohort skips training entirely.
- Config stores the full row dict under `row`, the sheet name under `name`, and the selected
  graph under `graph_row`; this distinguishes sheet rows from graph rows for downstream code.
- Config additionally binds active aspects and all cohort graph archive hashes, because a head
  depends on the other heads' gradients. `code_version` hashes ms_train/ms_rows/ags_head/itp_train/
  itp_data source bytes, so code changes invalidate resume without depending on a commit ID.
- Output validation reuses `ags_head.verify_output`; JSON precedes the atomic npz completion
  marker, retaining strict config, checksum, dtype, shape, finite and unit-norm checks.
- `--set` accepts JSON values or bare strings, with dotted keys such as `graphs.emotion=Fres1`;
  this permits named derived rows without editing the canonical picks.
- CPU uses torch 8 threads plus `threadpool_limits(8)`; `--max-epochs` overrides the normal 200
  or smoke 2 epoch cap, and `--smoke` always confines outputs to this ticket's scratchpad.
- Earlier graph data is read from the absolute main-checkout base; input loaders already use
  absolute `/data/SSD2/...` stores. Code imports come from this worktree.

## Commands

Read-only inspection used `cat`, `sed`, `rg`, `ls`, `pwd`, `git status --short` and
`git branch --show-current`. Binding inputs were read in the requested order. No filesystem
AGENTS.md was found; the user-supplied instructions were followed. An inspection of a presumed
worktree `prompted_aspect_views/outputs` directory found no such path; the loader's actual
absolute `/data` stores worked without any changes or symlinks.

All Python commands below used this prefix, with no GPU visible:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 235 /root/miniconda3/envs/CoSiR/bin/python
```

`<round>` = `src/test/20261019_model_sheet`.
`<smoke>` = `/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/02`.
The following arguments were executed (logs named on the right, all under `<smoke>`):

```text
-u <round>/ms_train.py --rows A1 --aspects genre --seeds 0 --device cpu --max-epochs 2 --outdir <smoke>/full
    genre_full.log
-u <round>/ms_train.py --rows T1 --seeds 0 --device cpu --smoke --max-epochs 2
    t1_smoke.log
<smoke>/verify_smoke.py
    verify_smoke.log
-u <round>/ms_train.py --row A1 --set graphs.genre=R0o --name A1 --aspects genre --seeds 0 --device cpu --max-epochs 2 --outdir <smoke>/full
    genre_resume.log
-u <round>/ms_train.py --rows T1 --seeds 0 --device cpu --smoke --max-epochs 2
    t1_resume.log
-u <smoke>/partial_resume.py
    partial_resume.log
```

The final script copies the emotion npz/JSON into `<smoke>/partial/T1`, then invokes the T1
command with `--outdir <smoke>/partial`. No files are deleted. A short stdin Python read printed
the fit/dev counts and run durations from the saved JSON. `git diff --check` and the staged diff
were checked before committing with this message (no push or merge):

```text
r10 ms 02: Trainer core: separate and trunk heads, form a, joint loop, stage 1 bitwise path
```

## Smoke numbers

Full genre: **62,906 fit / 8,045 dev pairs**, seed 0, CPU 8 threads, batch 512, two epochs.
Both train and dev values compare with Python `==` against the first two epochs of
`/project/CoSiR/src/test/20261018_aspect_graph_sheet/outputs_fu/genre__R0o__s0.json`:

| Epoch | Step | Train loss (exact) | Dev loss (exact) | Seconds |
|---|---:|---:|---:|---:|
| 1 | 123 | 5.772502180768421 | 7.159665240316824 | 16.68 |
| 2 | 246 | 4.082042087861413 | 7.192388301728452 | 16.54 |

Best epoch **1**, run **33.58 s** excluding input loading. The stored reference also has best
epoch 1 (its full run stopped at 16), so the two-epoch smoke can compare its final arrays directly.
Both `img` and `cap` pass **`np.array_equal`**, each over **10,240,000 float16 values**. This is
bitwise array equality, not a tolerance or AMI comparison.

T1: first **5,000 fit pairs per head**, full dev graphs, seed 0, two joint epochs, steps 10/20.

| Head / graph | Full fit / dev | Train e1 → e2 | Dev e1 → e2 | Best epoch | Max norm error |
|---|---:|---|---|---:|---:|
| emotion / Gres | 35,754 / 4,926 | 11.179670 → 9.155603 | 10.861981 → 10.319878 | 2 | 0.000027299 |
| style / Fres05 | 23,136 / 5,644 | 11.705056 → 10.196061 | 10.925975 → 10.360161 | 2 | 0.000029504 |
| genre / R0o | 62,906 / 8,045 | 10.457106 → 6.960823 | 8.866079 → 8.194670 | 2 | 0.000036836 |

Three heads saved; all six arrays are finite float16 `(5000, 2048)`, within the .01 norm
tolerance; per-head histories differ. T1 run **10.65 s** through the final output save,
excluding input loading. Wall times are descriptive: the genre and T1 processes briefly overlapped.

Complete resume verified and skipped genre and all three T1 heads. The genre resume also exercised
the `--row/--set/--name` interface. Partial T1 resume preserved both emotion files' SHA and mtime,
and the six arrays after cohort replay exactly equal the original T1 arrays. All smoke executions
passed; no smoke failures or numerical workarounds.

## Open points and cleanup

CUDA is implemented but deliberately not executed; the controller owns its GPU smoke. Full stage 0
emotion/style multi-seed regeneration and scoring belong to ticket 04. Forms b/c, distillation,
instructed inputs and their row definitions belong to ticket 03. No open ticket-02 implementation
issues identified. No unit tests were added.

Deleted files: none. Repo removal candidates: none. The ticket's smoke directory (logs, check
scripts and generated npz/JSON files) can be removed after controller validation; retained here.
