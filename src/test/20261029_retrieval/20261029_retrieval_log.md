# r11 retrieval (2026-10-29): run log

- rt_metrics speed, 12,281 x 2,048 random float32 rows, 8 workers, before the fix (float32 products): retrieval qe_k 0 2.9 s, qe_k 5 5.0 s, graded 4.3 s; largest worker 318 MB peak.
- Same, after the fix (float64 similarities, row-wise sums for the nDCG discounts): qe_k 0 3.1 s, qe_k 5 5.7 s, graded 4.2 s; largest worker 437 MB peak (limit 2 GB, no block size change).
