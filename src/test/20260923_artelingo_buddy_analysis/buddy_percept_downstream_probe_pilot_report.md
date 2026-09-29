# Buddy versus PercepT: image-only downstream probe pilot

## Method

The train and held-out patch caches follow the fresh painting order returned by each split's `pipeline.load_dedup_features()`. Both saved snapshots were checked for identical unique painting-ID sets and reindexed by painting ID into those cache orders before labels were used. Aligned held-out emotions and genres from both snapshots were checked against labels computed from the held-out pipeline. Buddy's independently reclustered held-out communities were not used; PercepT's held-out topics were checked but were not mapper targets.

Two unchanged `AttentionPoolingMapper` instances from the sibling PercepT Stage 2 pilot were trained independently on the full 61,402-painting patch cache, with 19 buddy outputs or 67 PercepT outputs. Both used Adam, learning rate 1e-3, 100 full-batch epochs, and **single-label cross-entropy**. This is a deliberate change from the original Stage 2 pilot's BCE-with-logits objective: both systems here supply one hard integer label per painting, whereas the original DEC targets were soft multi-hot. Cross-entropy gives these two systems the same hard-label task. Each frozen mapper's held-out softmax is predicted from image patches alone. The control is the untrained mean of each painting's 50 patch tokens (512-D).

Each feature set got the same `LogisticRegression(max_iter=2000)` probe with default settings. Emotion used a seed-42 stratified 50/50 split: 4,682 probe-train and 4,683 probe-eval paintings. Genre used all 159 annotated paintings in seed-42 shuffled `StratifiedKFold(n_splits=5)`, with `cross_val_predict` and pooled out-of-fold scoring. The smallest genre class had 5 paintings, so the actual `n_splits` was 5 (the largest feasible value up to 5). AMI uses `pipeline.external_metrics`; accuracy is the fraction of correct predictions.

## Results

| feature set | emotion AMI | emotion accuracy | genre AMI | genre accuracy |
|---|---:|---:|---:|---:|
| buddy topic softmax | 0.0231 | 0.3350 | 0.2625 | 0.4403 |
| PercepT topic softmax | 0.0221 | 0.3359 | 0.2973 | 0.4465 |
| mean-pooled control | 0.0685 | 0.3882 | 0.3399 | 0.5660 |

Final full-batch train cross-entropy: buddy **2.409179**; PercepT **2.454176**.

## Verdict

The matched audit measured buddy occupancy at **0/19 empty, 3/19 below 1%** and PercepT occupancy at **21/67 empty, 50/67 below 1%**. Those are frozen-topic occupancy numbers, not probe scores.

**Emotion:** mean-pooled control AMI 0.0685, accuracy 0.3882; majority-class accuracy 0.3237. Buddy retains 33.7% of control AMI and 17.5% of its accuracy lift over the majority class: less than most by the stated half-of-both rule. PercepT retains 32.3% of control AMI and 18.9% of its accuracy lift over the majority class: less than most by the stated half-of-both rule. 

**Genre:** mean-pooled control AMI 0.3399, accuracy 0.5660; majority-class accuracy 0.2453. Buddy retains 77.2% of control AMI and 60.8% of its accuracy lift over the majority class: most by the stated half-of-both rule. PercepT retains 87.5% of control AMI and 62.7% of its accuracy lift over the majority class: most by the stated half-of-both rule. 

Neither bottleneck leads on both metrics for both outcomes. Buddy minus PercepT is +0.0009 emotion AMI, -0.0009 emotion accuracy, -0.0347 genre AMI, and -0.0063 genre accuracy. Emotion favors mixed; genre favors PercepT. The direction is not consistent between outcomes. This single seed and single probe split establish the observed direction, but cannot establish reliability across mapper seeds or probe splits.
