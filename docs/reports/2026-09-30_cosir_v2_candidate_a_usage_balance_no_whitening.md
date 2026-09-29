# CoSiR v2 Candidate A: usage balance without whitening on ArtELingo

**Verdict: the usage-balance penalty alone fixes the measured mass concentration while preserving raw-feature reconstruction.** In one successful seed-42 run on all 308,723 ArtELingo training pairs, raw CLIP features with `lambda_usage_balance=0.1` gave **7.90% top-two-factor mass share**, down from 87.7% in Task 3, with **0.5606/0.5154** image/text relative reconstruction L2, close to Task 3's 0.5542/0.5097 on the same raw target. All 32 factors were active, shared across modalities by the declared threshold, and community-spanning. The full-rank and reduced-rank whitened runs also balanced usage, but their reconstruction errors stayed near 0.99 on their respective whitened targets. This raw-input ablation supports whitening as the source of that reconstruction cost under the tested configuration. It does not establish that the factors are semantically useful or generalize beyond training data.

## Direct comparison

| Metric | Task 3 (raw, no usage-balance) | Task 4 (full-rank whiten + usage-balance) | Task 5 (K=433 reduced-rank whiten + usage-balance) | **Task 6 (raw + usage-balance)** |
|---|---:|---:|---:|---:|
| Top-2 mass share | 87.7% | 8.66% | 7.76% | **7.90%** |
| Max single-factor share | — | 4.35% | 3.88% | **4.06%** |
| Modality-private factors | 1/32 | 0/32 | 0/32 | **0/32** |
| Dead factors | 0/32 | 0/32 | 0/32 | **0/32** |
| Community-spanning fraction | 19/32 (59.4%) | 25/32 (78.1%) | 31/32 (96.9%) | **32/32 (100%)** |
| Reconstruction rel. L2 (img/txt) | 0.5542/0.5097 | 0.9929/0.9931 | 0.9912/0.9915 | **0.5606/0.5154** |
| Matched/shuffled cosine | 0.9263/0.6060 | 0.8900/0.5881 | 0.9026/0.5791 | **0.9237/0.6005** |

The reconstruction target is raw CLIP in Tasks 3 and 6, and PCA-whitened CLIP in Tasks 4 and 5. Thus the Task 3-to-Task 6 difference is a direct same-target comparison; the whitened-target errors describe different reconstruction problems. Against Task 3, Task 6's relative L2 rose by only 0.0064 for images and 0.0057 for text. Matched-minus-shuffled cosine was 0.3232, versus 0.3203 in Task 3.

## Method and diagnostics

The reproducible script is `src/test/20260930_factor_usage_balance_no_whitening/run_revalidation.py`. It loads cached image/text CLIP features through `FeatureManager` at `/data/SSD2/pre_extract/artelingo/features` and checks the unique, in-range stored sample IDs before the positional join to `/data/PDD/artelingo/artelingo_train.json`. As in Tasks 3–5, it builds the content graph from raw features, trains default Block 1 Stage 1, and detects communities from its embeddings. The graph had 3,130,544 edges and detection produced 21 communities.

Factor training received the **original raw 512-dimensional image/text arrays**, with no PCA fit or transform. `FactorTrainingConfig(lambda_usage_balance=0.1)` kept 32 factors, seed 42, 2,000 edge-sampled epochs, batch size 1,024, Adam learning rate `1e-3`, and the other Task 3–5 loss weights. There was no hyperparameter tuning. The first five and last five sampled-loss means were 0.708994 and -0.147568; all 2,000 losses were finite (range -0.156840 to 0.935807). A negative total loss is valid because the usage-balance term minimizes negative entropy.

Diagnostics were computed on all 308,723 returned code pairs. As in earlier reports, a factor is dead if both modality means are below `1e-4`; an active factor is modality-private if its smaller modality mean is at most 5% of the larger; a factor spans communities if no one community receives at least half its combined image/text activation mass. No factor crossed any of those failure thresholds. The largest single-community mass share across factors ranged from 15.6% to 34.7%. Factor activation-mass shares ranged from 2.11% to 4.06%; factors 25 and 30 were the largest two, together holding 7.90%. Reconstruction was measured as the mean per-row `||reconstruction − raw input||₂ / ||raw input||₂`; the shuffled cosine used the same seed-42 text-row permutation as the prior runs.

The first execution trained with the same settings but stopped before emitting diagnostics: its copied loss-trace parser did not accept a minus sign and therefore rejected the negative loss values as an incomplete trace. The parser was corrected and the identical configuration rerun once. The table reports that successful run; no data, model, loss weight, or preprocessing choice changed after seeing results.

This remains an in-sample, single-seed diagnostic. The community-spanning rule is an operational mass threshold, not a semantic-factor test. The factors have not been named or evaluated on held-out examples. The successful run took 116.5 seconds overall, including 9.3 seconds for the graph, 1.8 for Stage 1, 80.5 for community detection, and 23.6 for factor training plus full-code generation.
