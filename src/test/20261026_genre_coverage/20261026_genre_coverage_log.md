# 2026-10-02 genre coverage check (spec §5.3)

**Question.** Can WikiArt genre be the third ArtELingo aspect? The labels on disk (ArtELingo-28) covered only 1,160 of
61,402 paintings.

**Data.** ArtGAN's WikiArt genre CSVs (`genre_train.csv`, 45,503 lines; `genre_val.csv`, 19,492 lines). The user
downloaded them into `/data/SSD/wikiart_genre/`. `genre_class.txt` returned 404 upstream.

**Method.** `genre_coverage.py` joins on the file stem (= ArtELingo painting id). It reads painting ids, split
membership and the CSVs only: no model, no evaluation result, no held episode. The id-to-name mapping comes from
majority vote against the ArtELingo-28 genre names.

**Result** (`genre_coverage.json`).
- 64,994 labelled WikiArt images; 1 conflicting entry, dropped.
- Mapping purity is 1.0 for 9 of 10 ids. Id 5 has no ArtELingo-28 example and is `nude_painting` by elimination
  and alphabetical order (inferred).
- Coverage is 81.0% of scorer-train, 81.5% of selection, 80.6% of val and 81.0% of held paintings.
- The smallest genre is illustration or nude: 837 paintings in scorer-train, 149 in selection, 296 in held.
- **Spec rule** (≥ 50% of selection and held paintings, ≥ 30 per genre in each): **KEEP.**
