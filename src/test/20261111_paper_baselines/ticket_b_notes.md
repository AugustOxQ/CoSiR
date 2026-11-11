# Ticket B implementation and validation

Fix-round correction: the original BERTopic smoke below used an incomplete MiniLM cache and is **invalid
as BERTopic evidence**. The model silently fell back to Transformer + mean Pooling without Normalize,
with max_seq_length 512 and non-unit outputs. The controller repaired the cache at revision 1110a243.
The runner now requires exactly Transformer, mean-only Pooling, Normalize, max_seq_length 256, and
unit rows within strict deviation < 1e-4; it does not pass `normalize_embeddings`. Module names,
pooling mode and sequence length are recorded in minilm.json. No real-data run was made in this fix round.

Legacy embedding caches lacking these checks are rebuilt before seed skip decisions. A seed whose
embeddings_sha256 differs from the current minilm.npy is not complete. The scorer now requires
VLM2Vec meta.json and verifies its aspect, exact instruction, emb_sha256, n=12281 and limit=None.
It hashes and loads the same byte snapshot. BERTopic sidecars are also required, with smoke=false,
the expected seed and a matching NPZ checksum.

Fix-round folder tests: **58 passed in 4.46s**, with the requested CoSiR Python and thread settings.
Nothing was deleted. The retained `fix_regression_red.log` can be removed after review.

The runner uses `/root/venvs/bertopic/bin/python`; the scorer uses
`/root/miniconda3/envs/CoSiR/bin/python`. Set `PYTHONDONTWRITEBYTECODE=1` on every invocation.
The scorer imports r17's implementation unchanged. Shared paths and decisions were not edited.
No full BERTopic seeds or full scoring were run.

## Environment and source audit

The venv was created with the requested CoSiR Python and `--system-site-packages`.
Only `/root/venvs/bertopic/bin/python -m pip install --upgrade bertopic` was used for installation.
New venv packages: `bertopic==0.17.4`, `cuda-bindings==13.5.0`, `hdbscan==0.8.44`,
`narwhals==2.28.0`, `plotly==7.1.0`, `pynndescent==0.7.0`, `sentence-transformers==6.2.0`,
`umap-learn==0.6.0`. Venv bootstrap packages: `pip==24.0`, `setuptools==79.0.1`.
Inherited core versions are in `ticket_b_validation.json` (including scikit-learn 1.6.1).
Installation log: `/root/venvs/bertopic/install.log`.

`pip check` is not clean: it reports inherited GPU, datasets and telemetry conflicts, plus
`cuda-python==12.9.7` requiring CUDA 12 bindings while the inherited Torch requires CUDA 13 bindings.
Pip installed CUDA 13 bindings in the venv only. CPU imports, encoding and BERTopic smoke succeeded.
No attempt was made to repair unrelated packages or change the CoSiR environment.

Audited installed file: `/root/venvs/bertopic/lib/python3.11/site-packages/bertopic/_bertopic.py`,
SHA-256 `4fb30201881399ad4d7d1addb6fbc702b989af0c7d9d539a17ea0c30a17ef901`.
Upstream references: [release](https://pypi.org/project/bertopic/0.17.4/),
[source](https://github.com/MaartenGr/BERTopic/blob/v0.17.4/bertopic/_bertopic.py).

- `__init__` constructs UMAP with `n_neighbors=15, n_components=5, min_dist=0.0,
  metric='cosine', low_memory=False`. The runner supplies precisely these arguments plus
  `random_state=s`. Other UMAP settings come from installed umap-learn 0.6.0 (notably `init='auto'`).
- Default HDBSCAN constructor arguments: `min_cluster_size=10, metric='euclidean',
  cluster_selection_method='eom', prediction_data=True`; the runner retains the default object.
- Default `CountVectorizer(ngram_range=(1,1))` has no stop-word filtering. Default
  `ClassTfidfTransformer()` is retained. All resolved arguments, including defaults inherited from
  UMAP and HDBSCAN, are captured in the seed sidecar and `ticket_b_validation.json`.
- `_reduce_to_n_topics` calls agglomerative clustering with `self.nr_topics - self._outliers`.
  Thus production passes **21 with an outlier topic, 20 without**, leaving 20 real topics.
  The runner refuses a different BERTopic version until this source audit is updated.
- `reduce_outliers` defaults to `strategy='distributions', threshold=0`. Only remaining outliers
  go through `strategy='embeddings'`, with the same document embeddings and default threshold 0.
  Reassignments are used directly; no extra topic update or new topic is introduced.

## Runs and outputs

The permitted queue smoke ran once with `--limit 2000 --seed 0 --out` pointing to
`/tmp/claude-0/-project-CoSiR/a8d4e0fd-c558-4101-b584-e2c7ba0d2480/scratchpad/bertopic_smoke`.
It exited 0 in 0.46 queue minutes (about 28 seconds including imports).
MiniLM encoding took 7.701 seconds; BERTopic fit took 13.927 seconds, and the whole seed took 14.174 seconds.
HDBSCAN found **16 real topics**. Reduction passed **17**, including `-1`, and kept 16 real topics
under the explicit smoke exception. Outliers: **651** after HDBSCAN, **651** after reduction,
**0** after default reassignment, **0** at the embedding stage (that pass was unnecessary).
The invalid fallback MiniLM max sequence length was **512**; longest smoke document: 202 tokens;
documents exceeding the limit: 0. Encoding did not add normalization.

The scratch folder contains `minilm.npy`, `minilm.json`, `topics_seed0.npz`, `topics_seed0.json`.
The compact metadata are preserved in `ticket_b_validation.json`; topic arrays stay in scratch.

The queued real candidate gate exited 0 in **110.211 seconds** (1.86 queue minutes), within the 235-second
bound. All three candidate means matched r18 **exactly** (absolute difference 0, tolerance 1e-9):
emotion **0.10658729378204021**, style **0.40664322288431654**, genre **0.5001038781216712**.
The real label loader also confirmed 12,281 held rows and 9,949 labelled genre rows.

Tests: the requested CoSiR pytest command passed **27 tests in 3.89s**; the BERTopic venv passed
**11 pure-helper tests in 0.17s**. Tests cover invalid name order, non-unit/non-finite/malformed rows,
invalid topic ids/counts, relabel partition invariance, default and embedding reassignment paths,
production refusal of fewer than 20 topics, signed differences and r10 verdict boundaries,
all-aspect gate failure before output, and scratch-only smoke scoring with the actual r17 scorer.

CLI details:

- BERTopic `--out` names the directory containing `minilm.npy` and `topics_seed<s>.npz` directly.
  `--limit` requires a separate output directory. `--seed` selects one seed; default is all three.
  Completed seeds are validated and skipped; `--force` refreshes both embeddings and selected seeds.
- Scorer `--gate-only` loads only real candidate partitions/labels and r18's reference, prints the gate, and
  writes no cells. The task's gate invocation is wrapped in `timeout 235s` through the same CPU queue.
- Scorer `--only` writes the three selected comparator cells; it replaces the output file rather than
  merging with a previous invocation. Run without `--only` to produce all six cells together.
- `--record` copies the completed cells to this round's records folder. `PB_SMOKE=1` requires scratch
  `PB_OUTPUTS`, uses r17's 40 draws, and rejects `--record`. No records were written in this task.
- Input SHA-256s cover comparator files and available sidecars, r18 ranges/reference, ordered names,
  labels and both partition arrays. Array hashes include dtype and shape. Candidate arrays are those
  returned directly by `_cand_parts`; no copies are written into older rounds.

## Choices for the controller's decisions.md

These are proposed entries, not edits to the controller-owned decisions file. Prescribed choices are
included where the implementation resolves an API detail that can affect the result.

| Choice | Main alternative | Where | Could affect verdict/claim? |
|---|---|---|---|
| Current BERTopic 0.17.4 and installed component defaults, with a version audit guard | Older BERTopic/UMAP versions or tutorial presets | `pb_bertopic.make_model` | yes |
| Complete cached MiniLM on CPU: Transformer, mean Pooling, Normalize, max sequence length 256; validate unit rows without encode-time normalization | Accept an incomplete-cache fallback, change truncation length, or aggregate chunks | `document_embeddings` | yes |
| Account for `-1` in `nr_topics`: 21 when present, 20 otherwise | Always pass 20, which would leave 19 real topics with outliers | `reduction_count` | yes |
| Stop production if HDBSCAN finds fewer than 20 real topics; smoke retains the available count | Tune HDBSCAN or substitute k-means to force 20 | `reduction_count` | yes |
| Default distributions reassignment then embeddings only for survivors, default threshold 0; stop if any remain | c-TF-IDF/probability reassignment or unconditional argmax fallback | `run_seed` | yes |
| Sorted original topic ids map to 0..19 | Relabel by frequency or first occurrence | `relabel` | no, AMI is permutation invariant |
| First N documents in input order for smoke | Random smoke subsample | `load_docs` | smoke only |
| BERTopic seeds occupy the unseeded comparator's three state slots, shared across aspects | Resample UMAP seeds independently or refit per aspect | `load_bertopic`, r17 `paired_cell` | intervals may differ |
| VLM2Vec rows are checked against strict r17 unit tolerance and cast to float32, without renormalization | Normalize again before clustering | `load_vlm2vec` | potentially |
| All aspect candidate means must match r18 within 1e-9 before cells are written | Gate each aspect only when its cell is written | `candidate_gate`, `main` | avoids partial invalid output |
| Reuse r17 `paired_cell` as given: d=c* minus comparator, 2,000 draws (40 smoke), BOOT_SEED=0; r18 Qwen validation range; outcome taken directly | Reimplement bootstrap, reverse sign, or use r17's CLIP margin | `score_cell` | yes |

## Cleanup

Nothing was deleted. No removal is required. The scratch smoke folder and venv install log are disposable
after controller review; retain the venv for full runs. Failed atomic writes, if any, would leave temporary
files for inspection; none were observed in the smoke output.
