# SemArt check (r16 decide, 2026-11-03)

**Problem.** Before the freeze, the user asked whether a dataset on disk can carry another c* test (decision C
reading: results good, the Qwen emotion retrieval cell below the teacher was unexpected). SemArt (21,382 paintings
from the Web Gallery of Art, not WikiArt) is the one fit. The user asked for a feasibility check first (14:05).

**What ran.** `sc_check.py` (CPU, 8 threads, 47 s), reading existing stores only:
1. Near copies: the default Qwen image of each SemArt painting against the 49,121 ArtELingo train and validation
   paintings, with the MMArt cut's own matmul (`mp_extract.nearest_rows`). The SemArt and ArtELingo Qwen stores were
   built by the same extractor on 2026-10-03 (same format, unit rows); no new parity run.
2. Artists: SemArt's author field against the artists of ArtELingo's 42,969 train paintings (r15's unseen-artist
   reference, `mt_data.artelingo_train_names`). Three match tiers (exact slug, first surname plus first given name,
   surname only); a painting counts as new only when no tier matches; "UNKNOWN MASTER" authors are left out.
3. First pass (14:11) found name misses among the largest "new" authors: RAFFAELLO Sanzio (raphael), TIZIANO Vecellio
   (titian), GOYA Y LUCIENTES (francisco-goya), K?BKE (christen-kobke; the CSV has '?' for some letters). Fix: two
   aliases, the first surname token only, '?' as a wildcard. Rerun (14:13). The top 120 new authors were then read by
   eye; spot checks against the full ArtELingo store (all 61,402 paintings, 1,101 artists) confirm Poussin, Giotto,
   Duccio, Ingres, Guercino, Domenichino, Gossart and Grünewald are absent from ArtELingo.

**Results** (`sc_check.json`, per painting `sc_nearcopy.parquet`).

| Measure | Value |
|---|---|
| Nearest ArtELingo cosine, median | 0.862 |
| At cosine >= 0.85 / 0.90 / 0.95 / 0.99 | 12,176 / 6,292 / 3,542 / 1,670 |
| Kept at the 0.95 cut | 17,840 |
| Near-copy pairs inside SemArt at 0.95 | 1,095 |
| Authors: exact / given+surname / surname only / no match | 236 / 118 / 188 / 2,739 |
| Kept paintings by a new artist (unknown masters out) | 11,162, by 2,717 artists |
| Of those, nearest cosine >= 0.90 / >= 0.93 | 1,249 / 217 |

Type counts of the new-artist set: religious 4,101, portrait 1,686, landscape 1,548, mythological 1,175, genre 885,
still-life 823, historical 344, interior 323, other 253, study 24.

**Findings.**
- The 0.95 cut does not remove every copy on SemArt: the two sets scan paintings separately, so the same painting can
  sit below 0.95 (Leonardo's Last Supper 0.944; El Greco's Annunciation 0.909; Leonardo's Annunciation 0.904). On
  MMArt both sides were the same WikiArt files, so 0.95 sufficed there. The full kept set (17,840) still holds copies
  of ArtELingo paintings; how many is not checked.
- The new-artist set avoids most of this by construction. Its residual close pairs are mostly workshop or family
  versions (Pieter Brueghel the Younger after his father, 0.947) and name misses not yet found.
- SemArt's labels: type (10 classes, close to WikiArt genre), school (26), period (22, 50-year bins). No emotion or
  style label.

**Open for a spec (not decided).** Test set (the new-artist set; a stricter cut on it); read-out picks (SemArt's own
1,069 validation paintings, or the ArtELingo picks); encoders (Qwen and SigLIP 2 stores exist; SigLIP 1 and CLIP L/14
need extraction); the Qwen genre teacher view (about 35 GPU minutes, estimate from report 12-21, not checked).
