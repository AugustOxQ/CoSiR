# Leiden development reading (r10 decide 13), 2026-10-20

Exploration on development data; a direction, never a result. Asked by the user in the claim-test grill (Q3): "if we
simply replace k-means by Leiden in our case, what will happen?"

## Question

The sheet scores image-side AMI with k-means at k = 20 on the 5,000 evaluation paintings. What do the same output
vectors give under Leiden, and is a change due to Leiden's method or to its own community count?

## Protocol (option 3a of the grill)

- Vectors: the saved stage 2 sheet outputs (D2, T1; model seeds 0, 1, 2), PercepT core's comparator arm per aspect
  (P3 / P2 / P3, 40 topic probabilities, 3 seeds), the instructed Qwen image (target) and the default Qwen image
  (image encoder, for all three aspects; the sheet's style encoder bar uses CLIP). Each head is scored against its
  own aspect's label only.
- Leiden: L2-normalised float32 vectors; union 20-NN graph by cosine, undirected, unweighted, simple;
  `leidenalg.ModularityVertexPartition` (resolution 1.0), seeds 0, 1, 2; AMI with labels < 0 masked.
- Control: k-means (`pav_score.partitions`, the sheet's three seeds) at c = Leiden's rounded mean count.
- Code: `ld_probe.py` (Codex, gpt-6.1-sol medium; read by Claude). Command:
  `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261020_leiden_dev_reading/ld_probe.py`
  (183 s CPU). Outputs: `outputs/ld_results.json`, `outputs/ld_table.md`; test run in `outputs/test/`.

## Regression note

k = 20 AMI recomputed here matches the sheet to 0 on most vector sets; the largest difference is 0.0037 (D2 style,
seed 2). The probe re-normalises the float16 outputs in float32, while the sheet only casts them, so k-means can
settle differently. Not traced further (exploration).

## Result (mean over model seeds; Leiden averaged over its three seeds)

| Row | Aspect | k-means k = 20 | Leiden | Leiden count | k-means at Leiden's count |
|---|---|---:|---:|---:|---:|
| D2 | emotion | 0.0962 | 0.0815 | 11.8 | 0.0937 |
| D2 | style | 0.3684 | 0.3378 | 12.2 | 0.3505 |
| D2 | genre | 0.4540 | 0.4689 | 15.0 | 0.4529 |
| PercepT core | emotion | 0.0772 | 0.0725 | 16.7 | 0.0756 |
| PercepT core | style | 0.2286 | 0.2298 | 13.3 | 0.2215 |
| PercepT core | genre | 0.4359 | 0.4435 | 16.7 | 0.4442 |
| Target | emotion | 0.0929 | 0.0811 | 12.0 | 0.0917 |
| Target | style | 0.3952 | 0.3566 | 13.7 | 0.3922 |
| Target | genre | 0.4956 | 0.5375 | 13.7 | 0.5086 |

T1 and the image encoder are in `outputs/ld_table.md`.

## Reading

- Leiden at resolution 1.0 finds 10 to 18 communities on these graphs, fewer than 20.
- Emotion and style: Leiden lowers AMI for our heads and the target. k-means at the same count stays near k = 20,
  so the loss comes from Leiden's partition, not from its count.
- Genre: Leiden raises AMI for every row; the target gains most (+0.042).
- D2 minus PercepT core: emotion +0.019 (k-means) and +0.009 (Leiden); style +0.140 and +0.108; genre +0.018
  and +0.025. D2 stays above PercepT core under both clusterings on all three aspects.
- Under Leiden, D2 emotion (0.0815) ties the target (0.0811).

## Storage

Outputs are under 1 MB; nothing to delete.
