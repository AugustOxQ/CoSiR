# Painting-Grouped Split Validation Log

## Problem
ArtELingo split required leakage-free grouping by painting and identical image vectors to ensure train/val/held sets have no overlapping paintings or images. The previous row-based split leaked 99.5% of held-out rows.

## What Was Run

```bash
cd /project/CoSiR-v2 && PYTHONPATH=/project/CoSiR-v2 /root/miniconda3/envs/CoSiR/bin/python src/test/20261008_painting_grouped_split/check_split.py
```

## Results

### Data Load
- Loaded 308,723 rows
- 61,402 paintings
- 9 emotions

### Grouping
- Computed leakage groups via union-find on paintings and image vector hashes
- Result: 61,402 groups (each painting maps to exactly one group)

### Split Distribution
| Part  | Rows       | %    | Paintings | Groups |
|-------|-----------|------|-----------|--------|
| train | 216,107   | 70.0 | 42,969    | 42,969 |
| val   | 30,872    | 10.0 | 6,152     | 6,152  |
| held  | 61,744    | 20.0 | 12,281    | 12,281 |

**Row shares match targets exactly: 70.0% / 10.0% / 20.0%**

### Leakage Check

All 6 leakage metrics returned 0:
- ✓ held_rows_painting_in_train: 0
- ✓ held_rows_image_in_train: 0
- ✓ held_rows_painting_in_val: 0
- ✓ held_rows_image_in_val: 0
- ✓ val_rows_painting_in_train: 0
- ✓ val_rows_image_in_train: 0

### Largest Group Size
12 rows (no group larger than this)

### Emotion Distribution

| Emotion       | Train | Val  | Held  |
|---------------|-------|------|-------|
| amusement     | 21592 | 3076 | 6220  |
| anger         | 3117  | 479  | 879   |
| awe           | 35267 | 5012 | 10010 |
| contentment   | 61326 | 8639 | 17295 |
| disgust       | 10386 | 1496 | 2998  |
| excitement    | 17968 | 2600 | 5110  |
| fear          | 19780 | 2854 | 5591  |
| sadness       | 22751 | 3263 | 6642  |
| something else| 23920 | 3453 | 6999  |

All emotions distributed proportionally across parts.

### Runtime
4.262 seconds

## Conclusion

✅ **SUCCESS**: The painting-grouped split achieves:
- **Zero leakage** across all 6 metrics
- **Perfect row balance** matching target fractions exactly
- **Proportional emotion distribution** across train/val/held
- **Deterministic and seed-controlled** via seed=42

The split is ready for Tasks 3, 6, and 7.

## Note: residual near-duplicates (final-review measurement)

"Zero leakage" above means zero leakage on exact keys: the painting slug and the bit-identical image vector. The final whole-branch review measured residual near-duplicates on this seed-42 split:
- 24 held rows (0.04%) have a train image at CLIP cosine ≥ 0.99;
- 63 held rows (0.10%) have one at cosine ≥ 0.98.

These are real WikiArt duplicates stored under different painting slugs. One example is `camille-pissarro_boulevard-montmartre-spring-rain` and its `…-1897` counterpart. The final review found that this affects no conclusion. The `split_leakage` docstring in `src/data/splits.py` records the same caveat.
