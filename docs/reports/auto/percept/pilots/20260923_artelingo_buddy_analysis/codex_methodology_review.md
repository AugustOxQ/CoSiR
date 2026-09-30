# Methodology review

## Finding: tied labels are silently treated as a majority

**`load_dedup_features()` line 139; `majority()` lines 175–176 (used at line 220).** `Counter.most_common(1)` does not establish a strict majority. When two or more emotions have the same maximal count, it returns the one first encountered in the input order. The script neither detects nor reports that tie, so the assigned per-painting ``majority_emotion`` can change with annotation-row order and be presented as a genuine majority. This value feeds Sections 1–5, the valence label, and the anger subset, producing plausible but order-dependent statistics.

The same tie behavior also affects the per-painting genre and Arabic/Chinese reductions at lines 155 and 170–171.

I found no label-array alignment issue in Sections 1–5: each paired list is constructed from the same index list (or from the full shared painting order). The contingency table counts aligned ``zip(genre_sub, emo_sub)`` pairs; its sorted row/column label iteration cannot change those counts. Section 4 correctly compares \(P(\mathrm{top\ community}\mid\mathrm{anger})\) with \(P(\mathrm{top\ community})\), and Section 5 uses the exact ``idx_l`` subset for both other-language and English metrics. Empty counters would raise rather than silently yield a label; they are not a silent-result risk on these construction paths.
