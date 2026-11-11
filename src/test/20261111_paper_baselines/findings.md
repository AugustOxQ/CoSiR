# Ticket B findings

- r17 paired_cell accepts comparator [3,n] as an unseeded arm: averages all three states in every draw, resamples only rows and c* seeds. BERTopic UMAP seeds occupy these three state slots.
- r17 pins scikit-learn 1.6.1; BERTopic is isolated in a system-site-packages venv.
- r17 unit tolerance is strict deviation < 1e-3; clustering consumes float32.
- Candidate partitions must come directly from ns_clu_cells._cand_parts, with no renormalization or persistence in older rounds.
- All three aspect gates must pass before writing cells. r17 clustering draws: 2,000 (40 in smoke), BOOT_SEED=0.
- Installed BERTopic 0.17.4 source uses UMAP(n_neighbors=15, n_components=5, min_dist=0., metric='cosine', low_memory=False); HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True).
- _reduce_to_n_topics subtracts self._outliers from nr_topics, so production requests 21 with outliers, 20 without. Smoke found 16 real topics and requested 17 including -1.
- Default reduce_outliers strategy is distributions, threshold 0. Embeddings strategy also defaults to threshold 0; fail if any outlier remains after both prescribed passes.
- Smoke: first 2,000 input docs, seed 0, HDBSCAN 16 topics / 651 outliers, after distributions 0, embeddings pass unnecessary. MiniLM 7.701 seconds; fit/reduction/reassignment 14.174 seconds; queue 0.46 minutes.
- In installed sentence-transformers 6.2.0, cached MiniLM reports max_seq_length=512; longest smoke doc 202 tokens, none exceeds the limit. Do not override defaults.
- pip check reports inherited environment conflicts (GPU/data/telemetry packages); venv installation additionally exposes the inherited cuda-python 12.9.7 vs torch's cuda-bindings>=13 requirement. CPU BERTopic smoke succeeds; no base packages changed.
- Real gate PASS in 110.211s; candidate mean equals r18 exactly for all aspects: emotion 0.10658729378204021, style 0.40664322288431654, genre 0.5001038781216712. Genre label count 9,949 passed.
