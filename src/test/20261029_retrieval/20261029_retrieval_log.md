# 20261029 retrieval run log

- rt_metrics speed, 12,281 x 2,048 random float32 rows, 8 workers, before the fix (float32 products): retrieval qe_k 0 2.9 s, qe_k 5 5.0 s, graded 4.3 s; largest worker 318 MB peak.
- Same, after the fix (float64 similarities, row-wise sums for the nDCG discounts): qe_k 0 3.1 s, qe_k 5 5.7 s, graded 4.2 s; largest worker 437 MB peak (limit 2 GB, no block size change).

## Amendment 1 (PercepT validation source)

The saved full-scale PercepT validation outputs used different mapper weights from the sample-scale held P3 comparator (13,400 fit paintings). Validation probabilities are now exported by rt_percept_val.py using the same per-seed sample-scale mapper as held, with matching weight hashes checked on export and load.
