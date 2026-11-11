# r25 class-count extension

The paper's Table 2 now uses each aspect's class count: emotion 9, style 27, genre 10.
This extension retains all K=20 behavior and adds the class-count cells beside it, AMI only.
The controller runs full scoring after VLM2Vec extraction; this task runs only BERTopic and the candidate gate.

## Implementation

- `pb_partitions.py` supplies lightweight class counts, aspect K pairs and filenames for both environments.
  The scorer asserts labelled class counts equal r17's `CLASS_COUNT` and these constants; the independent
  checker makes the same assertion with NumPy. Shared `pb_paths.py` remains unchanged.
- The runner defaults to targets 20, 9, 10, 27, seeds 0/1/2. It uses a **fresh model fit per (seed, K)**;
  no reduction is chained after another target's reduction. MiniLM embeddings are reused from the checked cache.
  Each target passes `nr_topics=K+1` with outliers or K without, then applies the unchanged reassignment steps.
  Production stops if HDBSCAN finds fewer than the requested K real topics.
- New files are `topics_k9_seed<s>.npz`, `topics_k10_seed<s>.npz`, `topics_k27_seed<s>.npz` and JSON sidecars.
  Sidecars include requested `k` and actual `n_topics` as well as the existing provenance and count fields.
  Existing `topics_seed<s>.npz` and JSON remain the K=20 files; old K=20 sidecars without `k` are accepted as 20.
- `--k 20` selects the previous runner target explicitly, including for smoke runs; omitting `--k` selects all four.
  `--limit`, `--out`, `--seed`, `--force` and the model/cache safeguards remain available.
- The scorer loads candidate partitions at both K values, checks all six r18 candidate references before writing
  any cells, and uses each cell's own r18 validation range. It emits 12 cells, or 6 with `--only`.
  Each `score_cell` still calls the unchanged r17 paired bootstrap with the same draws, seed and sign.
- BERTopic partitions are shared across aspects only when K agrees (K=20); each class-count aspect uses its own K.
  VLM2Vec rows are loaded once per aspect and clustered independently at each K with r17's helper.
- `pb_check.py` independently calls sklearn MiniBatchKMeans at each aspect's two K values, then computes sklearn
  AMI on labelled rows for both comparators. It supports the scorer's JSON list and a controller-style `{cells: ...}`
  wrapper. `--stale` retains the historical K=20 fallback comparison only.

## Tests and authorized runs

Both full folder suites passed: **75 tests** in CoSiR (12.12s) and **75 tests** in BERTopic (12.18s).
The tests use stand-in stores/models and cover nine independent new fits, byte-for-byte K=20 preservation,
K-aware loader checks, class-count agreement, all six gates, 12/6-cell output and six independent sklearn refits.
These two test suites ran concurrently; no real comparator scoring or checker run was started.

The BERTopic command runs from this round's folder through `res-queue cpu --est 15m --label r25:bertopic`,
under nohup in a persistent tool session. Output is retained at `logs/bertopic_classcount.log`; `.exit` records its status.
The all-six gate runs afterwards through `res-queue cpu --est 10m --label r25:score --no-wait`.
Run results, hashes and the K=20 preservation check are recorded in `classcount_validation.json`.

BERTopic completed with queue exit 0 in **1.78 minutes**, skipping all three K=20 seeds.
HDBSCAN topics and pre-reassignment outliers reproduced identically across K for each seed:

| Target K | Seed | HDBSCAN real topics | Outliers before reassignment | Outliers after distributions | Outliers at embedding stage | Final K |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0 | 92 | 5384 | 0 | 0 | 9 |
| 9 | 1 | 87 | 5102 | 0 | 0 | 9 |
| 9 | 2 | 91 | 5134 | 0 | 0 | 9 |
| 10 | 0 | 92 | 5384 | 0 | 0 | 10 |
| 10 | 1 | 87 | 5102 | 0 | 0 | 10 |
| 10 | 2 | 91 | 5134 | 0 | 0 | 10 |
| 27 | 0 | 92 | 5384 | 0 | 0 | 27 |
| 27 | 1 | 87 | 5102 | 0 | 0 | 27 |
| 27 | 2 | 91 | 5134 | 0 | 0 | 27 |

Outlier counts were unchanged by reduction in every run; embeddings reassignment was unnecessary.
`nr_topics` passed was 10, 11 or 28 respectively (all runs contained -1 initially).
SHA-256 checks confirmed both MiniLM files and all six legacy K=20 files remained byte-for-byte unchanged.

The subsequent gate exited 0 in **224.237 seconds** (3.77 queue minutes). All six candidate means
matched r18 exactly, with absolute difference 0 (tolerance 1e-9):

| Aspect | K | Candidate mean AMI |
|---|---:|---:|
| emotion | 9 | 0.1024127745008577 |
| emotion | 20 | 0.10658729378204021 |
| style | 27 | 0.4109255361955107 |
| style | 20 | 0.40664322288431654 |
| genre | 10 | 0.5100704595257602 |
| genre | 20 | 0.5001038781216712 |

Full scoring and the independent checker were not run on real comparator data, as instructed.

## Result-shaping choices and alternatives

| Choice | Main alternative | Effect |
|---|---|---|
| Fit a fresh default BERTopic model for every (seed, K), with identical embeddings and seeded UMAP | Deep-copy a single fitted model before each target reduction | Both avoid reduction chains; chosen for a direct, auditable fit history |
| Explicit side-env class counts 9/27/10, asserted against r17 and actual labels in scorer/checker | Import r17 into BERTopic's side env or derive counts without a fixed protocol check | Matches the protocol and fails on changed labels |
| Keep legacy K=20 names and accept missing sidecar k as 20 only | Rename or rewrite existing artifacts | Preserves prior partitions and evidence unchanged |
| Use class-specific K and validation range for every new cell; retain K=20 cells independently | Use one global K or the K=20 range for all cells | May change AMI, intervals and verdicts; required by the new paper decision |
| Preserve the three BERTopic UMAP seeds as the unseeded comparator's three state slots at each K | Bootstrap UMAP seeds independently | Retains r17's agreed interval protocol |

## Cleanup

Nothing was deleted. The retained logs and exit-status files are disposable after controller review;
keep the generated partitions, embedding cache and validation evidence for scoring and reproducibility.
