# r10 model sheet 03 build

Ticket: **Trainer extensions: form b, form c, distillation, D1 batches, instructed inputs**.
Branch: `r10-ms-03-ext`.

## Plan and progress

- Binding reads (index, ticket, spec from section 1, defaults 4–8, ticket 02 build, trainer/helpers): complete.
- Full row schema, forms b/c, image distillation, D1 painting batches and instructed inputs: complete.
- Six requested two-epoch CPU smokes and extension diagnostics: complete, all passed.
- Ticket 02 full genre and T1 replay: complete; all exact comparisons passed.
- Single-agent diff review and `git diff --check`: complete; committed with the code.

Planning-with-files notes are consolidated here to respect the ticket's write scope.
No agents, CCG records, unit-test suite or changes to earlier-round code/data.

## Definitions settled (each with its reason)

- Canonical T4/T5 and D1–D5 use the index's exact picks/weights; validation covers the full schema: derived rows use the same contract as canonical rows.
- Validation requires form c iff mask sharing, form b's complete three-aspect graph map, `graphs=None` only with graph terms off, distillation when graph terms are off, positive weight iff distillation, and default/instructed inputs: invalid combinations fail at the boundary.
- CPU form a with graphs and no distillation still directly returns `ags_head.pair_loss`, with no added loss arithmetic: preserves the bitwise path.
- Separate/trunk MLP construction and the exact `AdamW(models.parameters(), lr=..., weight_decay=...)` call stay in their original order: preserves parameters, dropout and optimizer trajectories.
- Form b's candidates are the union of the two other aspects' undirected buddies minus own buddies and self, separately for fit/dev: implements the specified cross-aspect population.
- CSR construction joins other graphs' node indices through painting IDs into the head's order and asserts equal unique node sets: index order is not a shared interface.
- Main loads all three graph rows for form b, including a derived separate run selecting one head: a head still needs both other aspects' candidates.
- Form b samples one node per pair endpoint uniformly over its CSR list before each epoch, using independent `default_rng([seed, aspect_index, 1])`: preserves shuffle streams and keeps NumPy sampling outside the loss step.
- Dev negatives reset `default_rng([10000, aspect_index])` for every evaluation in fixed dev-pair order: removes sampling noise from early stopping.
- Each endpoint's sampled node is embedded on the buddy term's column side and reused across all four buddy terms and both directions; missing rows get `-inf`: preserves pair weighting and the own-caption terms.
- Form c constructs image MLP, caption MLP, then masks in aspect order; applies raw MLP output times ReLU mask, then normalization: implements one shared embedding with aspect conditioning.
- Each head shares one mask across both sides; masks alone get a zero-decay AdamW group and cost `5e-4 * L1(relu(mask))` per head loss: follows the binding CSN defaults without an embedding L2 term.
- Form-c JSON saves the ReLU mask after restoring that head's best state: the 2,048 saved floats describe its output arrays, not the final joint state.
- Direct distillation reuses graph terms' 2B image outputs, student rows against teacher columns at T=.05, excluding off-diagonal occurrences of the same painting: duplicate endpoints cannot become negatives.
- Relational distillation is mean-row KL(teacher || student) at T=.05, excluding the diagonal and every same-painting column; excluded log values are zeroed before multiplication: avoids `0 * -inf` NaNs while keeping eligible probabilities unchanged.
- Distillation-only batches use 1,024 distinct local painting indices and one full shuffled 13,400-painting pass per head/epoch, including smoke (13 full batches plus 88): D1 has no graph-dependent batching.
- D1 dev loss is the sample-count-weighted mean of direct distillation on fixed 1,024/465-painting batches; strict improvements save each head, patience 15 removes it, all heads inactive ends the run: inherits deterministic head-wise early stopping on a graph-free objective.
- D1 never calls the caption model in a loss; its gradients remain None and ordinary AdamW leaves its MLP at init: fulfills the caption control without changing trunk optimizer construction.
- D1 config uses graph_row/graph_identity/graph_sha256 `none`, graph_archives `{}`, zero graph-pair counts and explicit painting counts: scorer and sheet identify it without a graph archive.
- Unique/exhaustive, pairwise-disjoint fit/dev/evaluation IDs are asserted before input/teacher reads; only fit/dev teacher rows are copied into loss tensors in each graph's node order: no evaluation-painting teacher reaches training or dev loss.
- Instructed inputs call `head.load_inputs(s, aspect, 'prompted')` per head and pass its aspect key as `input_key` to `head.make_config`: both image/caption inputs and scoring metadata agree.
- All extended loss operations, labels, weights, teachers and extra indices use the input device; CSR sampling precedes epochs/dev passes: CPU/CUDA share the loss implementation.
- Config code identity includes the instruction-image helper; extended loss descriptions name the actual terms: resume binds the complete objective.

## Commands

`<round>` = `src/test/20261019_model_sheet`.
`<smoke>` = `/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/03`.
All training/diagnostic commands used this prefix (GPU invisible, 8 CPU threads):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 235 /root/miniconda3/envs/CoSiR/bin/python
```

Arguments and redirected logs (all logs in `<smoke>`):

```text
-u <round>/ms_train.py --rows T4 T5 --seeds 0 --device cpu --smoke --max-epochs 2
    t4_t5.log (initial run)
-u <round>/ms_train.py --rows T4 T5 D1 D2 D5 --seeds 0 --device cpu --smoke --max-epochs 2
    extensions.log (final code)
-u <round>/ms_train.py --row T1 --set inputs=instructed --name I1 --seeds 0 --device cpu --smoke --max-epochs 2
    i1_smoke.log
-u <smoke>/verify_extensions.py
    verify_extensions.log
-u <round>/ms_train.py --rows A1 --aspects genre --seeds 0 --device cpu --max-epochs 2 --outdir <smoke>/full
    genre_full.log
-u <round>/ms_train.py --rows T1 --seeds 0 --device cpu --smoke --max-epochs 2 --outdir <smoke>/regression
    t1_smoke.log
-u <smoke>/verify_regression.py
    verify_regression.log
```

Initial T4/T5 passed before finalizing loss metadata and the extended path's pair-index conversion;
retained those arrays/JSON as `<smoke>/initial_outputs` (`mv`, no deletion). Final extension artifacts
are in `<smoke>/outputs`. Read-only inspection used `cat`, `sed`, `rg`, `ls`, `ps` and git status/diff;
file creation used heredocs/apply_patch (one system-Python pathlib edit of the scratch diagnostic).
`git diff --check` passed. No process hit its 235-second timeout; no smoke failures or workarounds.

## Extension smoke numbers

Seed 0, CPU 8 threads, two epochs. Graph rows train the first 5,000 fit pairs (10 steps/epoch),
full dev graphs; D1 trains all 13,400 paintings (14 steps/epoch). Every head picked epoch **2**.
Losses below are rounded to six decimals; JSON/logs retain the full values.

| Row/head | Train e1 → e2 | Dev e1 → e2 |
|---|---|---|
| T4 emotion | 11.197146 → 9.197786 | 10.893423 → 10.333800 |
| T4 style | 11.715060 → 10.230261 | 10.948600 → 10.373115 |
| T4 genre | 10.503107 → 7.013130 | 8.900038 → 8.189098 |
| T5 emotion | 12.349010 → 10.317883 | 11.878833 → 11.398205 |
| T5 style | 12.863237 → 11.502824 | 12.136006 → 11.419539 |
| T5 genre | 11.216137 → 8.271711 | 10.166000 → 9.034722 |
| D1 emotion | 4.525992 → 2.603943 | 2.717622 → 2.007664 |
| D1 style | 4.524874 → 2.625706 | 2.683794 → 2.081881 |
| D1 genre | 4.372456 → 2.531945 | 2.567010 → 1.985273 |
| D2 emotion | 14.085461 → 11.477272 | 13.498795 → 12.699688 |
| D2 style | 14.415157 → 12.252980 | 13.375108 → 12.411760 |
| D2 genre | 12.916302 → 8.854305 | 11.054018 → 9.966817 |
| D5 emotion | 11.740456 → 9.655235 | 11.257748 → 10.367734 |
| D5 style | 12.277087 → 10.595402 | 11.433983 → 10.617050 |
| D5 genre | 11.074313 → 7.828978 | 9.467688 → 8.503506 |
| I1 emotion | 12.017521 → 10.222458 | 11.487582 → 10.436082 |
| I1 style | 12.044414 → 10.682502 | 11.346085 → 10.648601 |
| I1 genre | 11.808779 → 9.054456 | 10.350428 → 9.173722 |

All **18 npz files / 36 arrays** are exactly float16 `(5000, 2048)`, finite, checksum verified,
and unit norm within .01; maximum norm error **0.00006639957427978516**.
All I1 input fields equal their aspect key. D1 config graph fields equal `none` and steps are 14/28.

D1 caption outputs are **bitwise equal on uint16 views** to each head's caption MLP at seed-0 init,
**10,240,000 values per head** (30,720,000 in total).

Form b first shuffled fit batch (512 pairs, seed 0; candidates from full graphs):

| Head | Endpoint a with / without | Endpoint b with / without | Loss with extra − loss all missing |
|---|---:|---:|---:|
| emotion | 508 / 4 | 505 / 7 | 0.0021219253540039062 |
| style | 511 / 1 | 509 / 3 | 0.0021495819091796875 |
| genre | 484 / 28 | 487 / 25 | 0.0019063949584960938 |

Each real batch has present and absent candidates. In eval mode the extra column increases the
actual head loss; replacing every candidate with missing exactly recovers form a (`torch.equal`).
Reversing both other graphs' node orders and remapping their edges leaves the candidate CSR
identical. Repeated dev sampling with the prescribed reset seeds gives identical tensors.

T5 masks have **204 / 218 / 207 zeros** (emotion/style/genre); maximum changes from initialization
are **0.02014117 / 0.02030666 / 0.02017914**. Pairwise mask L2 distances are **42.20353 / 42.15335 /
41.73100** (emotion-style / emotion-genre / style-genre), proving distinct saved masks and training.

On one real genre batch of 64 pairs, D3 composed with form b/c gives losses **13.095521926879883 /
14.069445610046387** with all **8 / 9** participating parameter gradients finite, respectively.
These are smoke diagnostics for the D17 combinations, not additional unit tests.

## Ticket 02 exact regression

Full genre: 62,906 fit / 8,045 dev pairs, 123 steps/epoch, best epoch 1. Training complete:

| Epoch | Step | Train loss (exact) | Dev loss (exact) |
|---|---:|---:|---:|
| 1 | 123 | 5.772502180768421 | 7.159665240316824 |
| 2 | 246 | 4.082042087861413 | 7.192388301728452 |

Both genre float16 arrays are **bitwise equal via `np.array_equal` on uint16 views** to
`/project/CoSiR/src/test/20261018_aspect_graph_sheet/outputs_fu/genre__R0o__s0.npz`,
**10,240,000 values per side**. All four losses above equal the stored stage-1 JSON at full
precision and the ticket-02 `genre_full.log` values at its logged precision. Runtime 31.92 s
excluding input loading, best epoch 1.

T1's **12 full-precision train/dev losses** equal ticket 02's saved JSON values, and all six
epoch log entries equal its `t1_smoke.log` (times excluded). All **six T1 arrays** additionally
match ticket 02 bitwise on uint16 views. Best epoch 2 for all heads, steps 10/20, runtime 10.60 s
excluding input loading. Its second-epoch dev losses are **10.319878227382572 /
10.36016113246104 / 8.194670393157256** (emotion/style/genre).

`verify_regression.log`: **PASS 16 exact full-precision losses; 2 stage-1 arrays and 6 ticket-02
T1 arrays bitwise equal**. Config code hashes differ from ticket 02 as expected; every result
was retrained. Final commit message:

```text
r10 ms 03: Trainer extensions: form b, form c, distillation, D1 batches, instructed inputs
```

## Open points and cleanup

CUDA is implemented but intentionally unexecuted; controller smokes T4/T5/D1/D2/D5/I1 on CUDA.
No known implementation issue. D3/D4 vary only direct weights; D17 composition has finite-gradient
smoke coverage, while the canonical six rows have full two-epoch smoke coverage.

Deleted files: **none**. Repo removal candidates: **none**. Would remove only ticket 03's scratch
logs, diagnostic scripts and generated arrays/JSON under `<smoke>` after controller validation,
including the retained `initial_outputs`; everything is retained now. No push or merge.
