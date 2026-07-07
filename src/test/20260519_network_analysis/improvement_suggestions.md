# CoSiR Improvement Suggestions
**Source:** `condition_informativeness.ipynb` analysis, epoch 999, experiment `20260602_054458_sweep_73jsta4o`  
**Date:** 2026-06-02  
**Headroom being closed:** oracle R@1 = 80.2% vs best deployable (type_des) = 60.9% → **19.3pp gap**

---

## Quick-reference priority table

| ID | Side | Change | Targets | Expected impact | Effort |
|---|---|---|---|---|---|
| L1 | Loss | Oracle distillation on 512-dim combiner output | nearest_rep=45.4%, predictor-oracle gap 24pp | Very high — closes most of 19.3pp gap | Medium |
| L2 | Loss | Type-contrastive loss on condition space | silhouette=0.019, direction_sim=0.746 | High — enables all condition-based improvements | Low |
| L3 | Loss | Delta direction orthogonality loss | direction_sim=0.746 | High — forces type-specific combiner behavior | Low |
| C1 | Training | Upsample caption 2.5× | caption predictor gap = 24pp | High for caption, free otherwise | Very low |
| M1 | Model | Increase label_dim 16→32 or 64 | silhouette=0.019, no room for type structure | Medium — prerequisite for L2 to work well | Low |
| L4 | Loss | Per-type retrieval loss | description dominating impression/aesthetic | Medium — makes conditions query-type aware | Medium |
| L5 | Loss | Predictor diversity regularization | predictor bias toward description (35%) | Medium — prevents safe-averaging | Low |
| M2 | Model | Type-conditional scalar gate in combiner | direction_sim=0.746 | Medium — structural fix for direction collapse | Medium |
| M3 | Model | Partitioned representatives (K per type) | nearest_rep fails, condition-space routing broken | Medium — structural fix for routing | High |
| C2 | Training | Type-anchor label embedding init | silhouette=0.019 at step 0 | Low-medium — helps convergence speed | Very low |

---

## Diagnostic summary (what each number means)

| Metric | Value | Normal would be | Meaning |
|---|---|---|---|
| Silhouette score (condition space) | 0.019 | > 0.25 | Conditions have almost no type-cluster structure in 16-dim space |
| Within-type cosine similarity | 0.080 | > 0.3 | Statistically real but practically tiny type signal |
| Effect consistency Cohen's d | 0.88 | > 0.5 = good | Same-type conditions produce consistent combiner outputs — combiner works |
| Direction similarity (off-diag) | 0.746 | < 0.3 | All condition types push images in nearly the same direction |
| Matched cond diagonal advantage | +6.0pp | > 15pp | Type-matching helps a little, but description dominates imp+aes |
| Predicted vs Gaussian noise z | -0.22 | > 2.0 | Learned conditions are NOT significantly better than noise in absolute terms |
| Oracle vs Gaussian noise z | +9.19 | > 2.0 | Oracle DOES carry real signal — the gap is reachable |
| nearest_rep overall R@1 | 45.4% | > avg_all (60.8%) | Condition-space routing is actively harmful |
| Predictor-oracle type agreement | 31.4% | > 60% | Predictor barely better than chance at picking the right type |
| Caption training samples | 1,423 | ~3,500 (balanced) | Caption is 2.5× underrepresented vs other types |

---

## L1 — Oracle distillation on 512-dim combiner output

### The problem
The predictor is trained to predict `label_emb[i]` — the learned 16-dim condition vector for training sample i — via stop-gradient distillation. But `nearest_rep = 45.4%` (the worst deployable strategy) proves that even when the predictor outputs a good condition in 16-dim space, it does not translate into selecting a useful representative. The 16-dim condition space and actual retrieval quality are decoupled.

The underlying issue: `label_emb[i]` is supervised by the contrastive retrieval loss, but only indirectly. The learning signal that reaches the label embeddings is diffuse — a condition vector ends up encoding many factors beyond "which representative would help this image." The predictor therefore mimics a noisy target.

### The fix
Bypass the 16-dim condition space entirely for predictor supervision. Instead, supervise the predictor to directly reproduce the oracle representative's 512-dim combiner output for each image:

```python
# At predictor training time (stop-gradient on combiner):
with torch.no_grad():
    oracle_rep_idx = find_oracle_rep(img_i)           # best K representative for this image's GT text
    oracle_output  = combiner(img_i, representatives[oracle_rep_idx])  # [512]
    oracle_output  = F.normalize(oracle_output, dim=-1)

pred_cond    = predictor(img_i)                       # [label_dim]
pred_output  = F.normalize(combiner(img_i, pred_cond), dim=-1)  # [512]

L_pred = 1 - (pred_output * oracle_output).sum()     # cosine distance in 512-dim space
```

The predictor now learns "produce the transformed embedding that the oracle representative would produce," not "produce a condition vector that matches the oracle's 16-dim label." The target is a 512-dim embedding — directly in the space where retrieval happens.

### Why this helps more than the current approach
Current supervision path: `img → predictor → label_emb_target` (16-dim, indirectly tied to retrieval)  
New supervision path: `img → predictor → combiner → oracle_512_target` (directly tied to retrieval)

This changes the predictor's job from "navigate a poorly-structured 16-dim space" to "produce an embedding that looks like the best possible transformed image." The combiner is the same frozen/jointly-trained network in both cases.

### Implementation notes
- `oracle_rep_idx` should be computed per GT text (not per image first-text) to avoid the bias seen in the aggregation section where oracle_UB=51.0% < avg_all=60.8%.
- Can be computed cheaply at batch construction time using the CA cache (`per_rep_gt_rank`).
- The cosine loss target can be computed once per epoch and cached, or computed online.
- Warmup: continue using the current 16-dim distillation for the first N epochs (where the combiner is not yet stable), then switch.

### Dependencies
Pairs well with L2 (the 16-dim condition space can still be improved in parallel) and M1 (larger label_dim gives more expressive predicted conditions). Can be done independently of all others.

---

## L2 — Type-contrastive loss on condition space

### The problem
Silhouette = 0.019. The 16-dim condition vectors learn almost no type-cluster structure. During training, the only signal that reaches `label_emb[i]` is the contrastive retrieval loss — which does not explicitly reward type-discriminative conditions, only retrieval-quality conditions. Since all types help retrieval to some extent (avg_cond beats random with z=+25), the loss sees no reason to push types apart.

### The fix
Add a contrastive loss on condition vectors within each batch, treating type as the positive/negative signal:

```python
# Within a batch, condition vectors grouped by type:
# Positives: pairs (c_i, c_j) where type_i == type_j
# Negatives: pairs (c_i, c_k) where type_i != type_k

def type_contrastive_loss(label_embs, types, temperature=0.07):
    # label_embs: [B, label_dim]
    # types: [B] int
    label_n = F.normalize(label_embs, dim=-1)
    sim = label_n @ label_n.T / temperature          # [B, B]
    
    # Mask: positive = same type (excluding self), negative = different type
    type_match = types.unsqueeze(0) == types.unsqueeze(1)  # [B, B]
    self_mask  = torch.eye(len(types), dtype=torch.bool)
    pos_mask   = type_match & ~self_mask
    neg_mask   = ~type_match
    
    # InfoNCE: for each anchor, positives vs all negatives
    L = 0
    for i in range(len(types)):
        if not pos_mask[i].any(): continue
        pos_sims = sim[i][pos_mask[i]]
        neg_sims = sim[i][neg_mask[i]]
        logits   = torch.cat([pos_sims, neg_sims])
        labels   = torch.zeros(len(logits)); labels[:len(pos_sims)] = 1.0 / len(pos_sims)
        L += -(labels * F.log_softmax(logits, dim=0)).sum()
    return L / len(types)

L_total = L_retrieval + lambda_type * type_contrastive_loss(label_embs_batch, types_batch)
```

`lambda_type` around 0.1–0.5 works as a starting point; tune by monitoring silhouette score during training.

### Why this helps
This is the root-cause fix for the weak condition structure. Once conditions cluster by type, the predictor has a learnable target (predict which type region to aim for), nearest_rep becomes useful (routing in condition space has meaning), and the per-type analysis metrics all become interpretable.

### Implementation notes
- This loss applies to `label_emb_all[batch_indices]` — the per-sample conditions being jointly trained.
- Does NOT apply to the predictor output yet; the predictor will benefit automatically once `label_emb` becomes better-structured.
- Requires that each batch contains multiple samples per type. Ensure batch construction stratifies by type (not purely random).
- Monitor: silhouette score during training as a diagnostic. Target > 0.15 before moving to the next stage.

### Dependencies
Prerequisite for M3 (partitioned representatives). Synergistic with M1 (larger label_dim gives more dimensions to separate). Should be deployed alongside L3 since both target direction_sim=0.746.

---

## L3 — Delta direction orthogonality loss

### The problem
Direction similarity = 0.746 between type-mean delta vectors (mean shift applied to images). All four condition types push image embeddings in nearly the same geometric direction in 512-dim space. This is a combiner-side mode collapse: the architecture learned one useful transformation axis, and all conditions just dial the intensity up/down.

This explains why matched conditions only give +6.0pp advantage (vs ideal ~15pp+) — if all conditions push in the same direction, there is no type-specific alignment to exploit.

### The fix
Penalise alignment of the mean delta vectors across types during training:

```python
def delta_orthogonality_loss(combiner, combine_emb, type_means, tau=0.3):
    # Compute mean delta per type on a batch
    deltas = []
    for t in range(4):
        out_t  = combiner(combine_emb, type_means[t:t+1].expand(len(combine_emb), -1))
        delta_t = (out_t - combine_emb).mean(0)   # [512] mean shift
        deltas.append(F.normalize(delta_t, dim=0))
    
    L = 0
    for i in range(4):
        for j in range(i+1, 4):
            cos_sim = (deltas[i] * deltas[j]).sum()
            L += F.relu(cos_sim - tau)             # only penalise if above threshold
    return L / 6   # 6 pairs

L_total = L_retrieval + lambda_orth * delta_orthogonality_loss(combiner, batch_emb, type_means)
```

`tau=0.3` is a lenient target (from current 0.746). Can tighten to 0.1 once the loss converges.

### Why this helps
Forces the combiner to learn that impression-type conditions should geometrically move images toward impression-aligned text embeddings, not in the same direction as description-aligned text embeddings. This structurally enables the type-matching advantage (currently only +6pp) to grow.

### Implementation notes
- `type_means` can be a running mean updated during training (detached from gradients) or recomputed each epoch.
- Efficient: compute per-type mean delta on a random subsample of the batch (e.g., 64 images) not the full training set.
- This loss acts on the combiner parameters, not on label_emb — gradient flows through combiner weights.
- Monitor: track direction similarity (off-diagonal mean delta cosine sim) after each epoch.

### Dependencies
L3 and L2 are complementary: L2 separates condition vectors in input space; L3 separates their effects in output space. Deploy together.

---

## C1 — Upsample caption training samples 2.5×

### The problem
Caption type has 1,423 training samples vs ~3,565 for description/impression/aesthetic — a 2.5× imbalance. Despite this, caption achieves the strongest oracle (93.5%) and the strongest matched condition (83.6%), indicating the caption-specific transformation is highly learnable. But the predictor has the largest gap for caption: oracle 93.5% − predictor 69.5% = **24pp**, compared to description's gap of 17pp.

The predictor has simply seen far fewer caption examples to learn from during training.

### The fix
Weighted sampler in the DataLoader:

```python
# Compute per-sample weight
type_counts  = {t: (train_types == t).sum().item() for t in range(4)}
max_count    = max(type_counts.values())
sample_weights = torch.tensor([max_count / type_counts[t.item()] for t in train_types])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
```

Or alternatively, repeat caption samples 2–3× in the dataset construction.

### Why this helps
The caption condition is the most type-specific (caption condition gives 83.6% for caption queries; description gives only 69.3% — a 14.3pp gap that disappears for impression/aesthetic). There is strong signal for the predictor to learn, just insufficient examples. Closing the 24pp predictor gap for caption alone would significantly improve overall performance since caption queries are 25% of the test set.

### Implementation notes
- Zero code risk: no architecture changes, just sampler modification.
- Monitor per-type predictor performance separately during training to verify caption improves.
- Risk: oversampling caption too aggressively may hurt impression/aesthetic if the combiner starts over-specialising. Stay at 2–3× maximum.

### Dependencies
Independent of all other changes. Should be done first alongside any training run as a free improvement.

---

## M1 — Increase label_dim from 16 to 32 or 64

### The problem
16 dimensions must simultaneously encode: (a) which caption type this sample belongs to, (b) which specific representative is most useful, (c) image-specific conditioning signal, and (d) magnitude of the required transformation. With 4 types, K=30 representatives, and per-sample variation, 16 dimensions are severely underpowered. The silhouette=0.019 is partly a consequence of this — in 16 dims, type structure cannot cleanly coexist with the other signals.

### The fix
Change `label_dim` in the experiment config from 16 to 32 or 64:

```yaml
# configs/train/default.yaml
label_dim: 64   # was 16
```

Everything downstream (combiner, predictor, label_emb initialisation) is parameterised by `label_dim` and adapts automatically.

### Why this helps
More dimensions give the condition space room to develop independently-encoded axes for type, for representative routing, and for instance-specific variation. The orthogonality loss (L3) and type-contrastive loss (L2) become more effective because they have more degrees of freedom to work with.

64 dims is not large relative to the 512-dim embedding space — the information bottleneck from 512→64→512 is still significant, which is desirable (conditions can't simply pass through all image information unchanged).

### Implementation notes
- Requires retraining from scratch — not compatible with continuing from epoch 999.
- 32 vs 64: start with 32 to validate the benefit before committing to 64. The marginal gain from 32→64 is likely smaller than 16→32.
- No change to combiner or other_proj architecture required (both accept `label_dim` as input).
- May slow training slightly due to larger predictor input, but negligible at this scale.

### Dependencies
Prerequisite for L2 (more dimensions make type clustering easier). Beneficial to combine with L1.

---

## L4 — Per-type retrieval loss

### The problem
The current contrastive loss evaluates retrieval against all 4 text types simultaneously. A training image is optimised to retrieve against a pool of 4 texts (one per type). This means the gradient signal does not distinguish whether the condition is helping caption retrieval vs impression retrieval — it just rewards higher similarity to the GT image overall.

This is why description condition wins for impression queries (+7.7pp over impression's own condition): the loss never explicitly penalises "impression condition fails to serve impression queries."

### The fix
Add a parallel loss that evaluates retrieval separately for each type:

```python
# For each training image with known type t:
# Apply condition that SHOULD work for type t, and evaluate against type-t texts only

def type_specific_retrieval_loss(combiner, img_emb, other_proj_out, label_embs, 
                                  types, txt_embs, txt_types, txt2img):
    L = 0
    for t in range(4):
        # Image indices that have type t condition
        img_mask  = (types == t)
        if not img_mask.any(): continue
        
        # Text indices of type t
        txt_mask  = (txt_types == t)
        
        # Apply type-t label embedding to type-t images
        out_t = combiner(img_emb[img_mask], None, label_embs[img_mask])  # conditioned
        out_t_n = F.normalize(out_t, dim=-1)
        
        # Compare against type-t text gallery
        gallery_t = F.normalize(other_proj_out[txt_mask], dim=-1)
        sims_t    = out_t_n @ gallery_t.T   # [N_t, M_t]
        
        # Standard contrastive loss on this type-t subset
        L += contrastive_loss(sims_t, img_to_txt_map(img_mask, txt_mask, txt2img))
    
    return L / 4

L_total = (1 - lambda_type) * L_retrieval_all + lambda_type * L_type_specific
```

### Why this helps
Explicitly rewards impression conditions for helping impression retrieval, and penalises impression conditions that help description retrieval at the expense of impression retrieval. This breaks the "description is a safe average" equilibrium the current training has settled into.

### Implementation notes
- `lambda_type` = 0.3–0.5 is a starting point. If too high, the per-type loss can hurt recall for mixed queries.
- Requires per-sample type labels during training, which already exist (`train_sample_types`).
- Per-type batch construction: ensure each batch contains at least 16 samples per type for the per-type loss to have sufficient negatives.
- Can be combined with the existing `LabelContrastiveLoss` by extending its forward pass.

### Dependencies
Most effective after L3 (once delta directions are more orthogonal, type-specific retrieval loss has something to reward/penalise).

---

## L5 — Predictor diversity regularization

### The problem
The predictor outputs description-type conditions for 35% of images (vs 25% expected). For impression queries (the weakest type, CLIP=36.8%), the predictor defaults to a near-description condition rather than learning an impression-specific prediction. This safe-averaging reduces the predictor's per-image value: predictor overall = 58.0% vs avg_all = 60.8% — the predictor is actually **worse than a fixed global average**.

### The fix
Add a diversity loss on predicted conditions within each batch:

```python
def predictor_diversity_loss(pred_conds, tau=0.3):
    # pred_conds: [B, label_dim]
    pred_n  = F.normalize(pred_conds, dim=-1)
    sim_mat = pred_n @ pred_n.T                          # [B, B]
    mask    = ~torch.eye(len(pred_conds), dtype=torch.bool)
    pairwise = sim_mat[mask]
    return F.relu(pairwise - tau).mean()                 # penalise pairs too similar

L_total = L_retrieval + lambda_div * predictor_diversity_loss(pred_conds_batch)
```

Alternatively, a softer version via entropy maximisation on type-affinity soft assignments:

```python
def predictor_type_entropy_loss(pred_conds, type_means):
    # pred_conds: [B, label_dim]; type_means: [4, label_dim]
    pred_n  = F.normalize(pred_conds, dim=-1)
    tmeans_n = F.normalize(type_means, dim=-1)
    logits  = (pred_n @ tmeans_n.T) * 5.0                # [B, 4] sharp
    probs   = F.softmax(logits, dim=-1).mean(0)           # [4] batch-average type distribution
    entropy = -(probs * (probs + 1e-8).log()).sum()
    return -entropy  # maximise entropy = uniform type usage
```

The entropy version is cleaner: it encourages the predictor to distribute predictions equally across all 4 type regions over a batch, while still allowing per-image specialisation.

### Implementation notes
- `tau=0.3` for the diversity loss. The entropy version is self-calibrating.
- `lambda_div` = 0.05–0.1. This is a soft regularizer, not a dominant loss.
- Should be applied to the predictor parameters only (stop gradient on label_emb to avoid circular effects).
- The entropy version naturally uses `type_means` which are available throughout training.

### Dependencies
Works best after L2 (once condition space has type structure, encouraging uniform type usage is more meaningful).

---

## M2 — Type-conditional scalar gate in combiner

### The problem
The combiner uses a single learned `dynamic_scalar` — a per-sample gate that interpolates between input and delta. This produces one number per image regardless of condition type. With one scalar, all condition types must share the same gating behavior, which structurally pushes them to produce similar delta directions (since the gate magnitude is the only free variable per-type).

### The fix
Replace the single scalar with a type-conditional scalar bank. The predicted type (from the predictor's type affinity) selects which scalar to use:

```python
class TypeConditionalScalar(nn.Module):
    def __init__(self, label_dim, n_types=4):
        super().__init__()
        # Instead of one scalar, project condition to 4 potential scalars
        self.scalar_net = nn.Linear(label_dim, n_types)
        self.type_means_buffer = None  # registered during init
    
    def forward(self, condition):
        # condition: [B, label_dim]
        scalars   = torch.sigmoid(self.scalar_net(condition))  # [B, 4]
        # Soft-select scalar based on condition's type affinity
        if self.type_means_buffer is not None:
            cond_n    = F.normalize(condition, dim=-1)
            tmeans_n  = F.normalize(self.type_means_buffer, dim=-1)
            weights   = F.softmax((cond_n @ tmeans_n.T) * 5, dim=-1)  # [B, 4]
            scalar    = (weights * scalars).sum(-1, keepdim=True)      # [B, 1]
        else:
            scalar = scalars.mean(-1, keepdim=True)
        return scalar
```

This allows each condition type to have a different gating intensity, without requiring explicit type labels at inference time.

### Implementation notes
- Requires architecture change — needs retraining.
- Adds only 4 × label_dim parameters (64 if label_dim=16) — negligible.
- The `type_means_buffer` should be updated as a running average during training.
- Alternative simpler version: just increase the scalar network depth from 1-layer to 2-layer with a per-condition bias.

### Dependencies
More impactful after L3 (once directions are orthogonal, different scalars per type become meaningful rather than redundant).

---

## M3 — Partitioned representatives (K per type)

### The problem
K=30 representatives span all 4 types without structural partitioning. When the predictor outputs a condition in 16-dim space and we search for the nearest representative, we're navigating a space where type structure is near-absent (silhouette=0.019). This is why `nearest_rep=45.4%` is the worst strategy — the condition-space geometry doesn't map to which representative would help.

### The fix
Partition representatives into type-specific banks, initialised by K-means within each type:

```python
# During representative computation: K_per_type = K // 4 representatives per type
K_per_type = 8   # total K=32

type_representatives = []
for t in range(4):
    type_conds = label_emb_all[train_types == t]
    # K-means on type-t conditions
    kmeans = KMeans(n_clusters=K_per_type).fit(type_conds)
    type_representatives.append(torch.tensor(kmeans.cluster_centers_))

representatives = torch.cat(type_representatives)  # [K_per_type*4, label_dim]
type_of_rep = torch.tensor([t for t in range(4) for _ in range(K_per_type)])  # [K]
```

At inference, the predictor first routes to a type, then searches within that type's K_per_type representatives:

```python
# Two-stage selection:
type_probs    = softmax(pred_cond @ type_means.T)       # [4] soft type assignment
dominant_type = type_probs.argmax()
rep_bank      = representatives[type_of_rep == dominant_type]  # [K_per_type, label_dim]
nearest_rep   = rep_bank[(pred_cond @ rep_bank.T).argmax()]    # [label_dim]
```

### Why this helps
Gives the predictor a structured routing problem: first decide which type, then which representative within that type. The within-type search space is 8 instead of 30, which is easier to navigate even with imperfect condition-space geometry.

### Implementation notes
- High effort: requires changes to representative computation, condition analysis (CA) cache, and inference code.
- The type routing can be soft (weighted average of type-specific nearest reps) at inference for smoother gradients.
- Prerequisite: L2 should improve condition clustering first, otherwise type routing will be only marginally better.
- Best implemented after L2+L3 demonstrate improved condition structure.

### Dependencies
Requires L2 (type-structured condition space) to work effectively. Also benefits from M1 (larger label_dim).

---

## C2 — Type-anchor label embedding initialisation

### The problem
`label_emb` is initialised from N(0, 1) — pure random. The first N epochs of training must therefore learn type structure from scratch against a contrastive loss that only weakly rewards it. This is a slow convergence problem, not a fundamental capacity problem.

### The fix
Initialise each label embedding with its type-mean anchor plus small noise:

```python
# Instead of: nn.Embedding(N_train, label_dim) with default init
# Use:
type_mean_init = type_means[train_types]                    # [N_train, label_dim]
label_emb_weight = type_mean_init + 0.1 * torch.randn_like(type_mean_init)
label_emb.weight.data = label_emb_weight
```

The noise (0.1 × std) prevents all embeddings of the same type from being identical (which would block gradient flow), while still giving the optimiser a type-discriminative starting point.

The type means themselves must be computed once before training begins (e.g., from a PCA of a pre-computed embedding space, or from a short warm-up run).

### Implementation notes
- Zero computational cost at training time.
- Requires type labels to be available at initialisation (they are, via `train_sample_types`).
- The type means used for initialisation can be the cluster centroids of a pre-trained embedding space, or simply random unit vectors fixed per type (preserving separation but not alignment).
- Monitor: silhouette score at epoch 10 should be clearly above 0.019 if this is working.

### Dependencies
Synergistic with L2 (type-contrastive loss drives the embeddings to maintain and strengthen the initial type structure). Has limited standalone value — most useful as a convergence accelerator alongside L2.

---

## Implementation roadmap

### Phase 1 — Low-risk, high-ROI (do first, no retraining needed for C1)
| Step | Action |
|---|---|
| C1 | Add weighted sampler for caption (2.5× oversample) |
| L5 | Add predictor entropy diversity loss (entropy version, λ=0.05) |
| C2 | Add type-anchor label embedding init for next run |

### Phase 2 — Core structural fixes (next training run)
| Step | Action |
|---|---|
| M1 | Increase label_dim to 32 |
| L2 | Add type-contrastive loss on label_emb (λ=0.2) |
| L3 | Add delta orthogonality loss (τ=0.4, λ=0.1) |
| C1 | Apply weighted sampler |

### Phase 3 — Predictor fix (after Phase 2 stabilises)
| Step | Action |
|---|---|
| L1 | Switch predictor distillation target to 512-dim oracle combiner output |
| L4 | Add per-type retrieval loss component (λ=0.3) |

### Phase 4 — Architecture changes (if Phase 2+3 plateau)
| Step | Action |
|---|---|
| M2 | Type-conditional scalar gate |
| M3 | Partitioned representatives with two-stage routing |

---

## Key metrics to track per epoch during Phase 2+3

| Metric | Current | Phase 2 target | Phase 3 target |
|---|---|---|---|
| Silhouette score (condition space) | 0.019 | > 0.15 | > 0.25 |
| Direction similarity (off-diagonal) | 0.746 | < 0.5 | < 0.3 |
| Predictor type-affinity agreement with oracle | 31.4% | > 45% | > 60% |
| Matched cond diagonal advantage | +6.0pp | > 10pp | > 15pp |
| Predictor overall R@1 | 58.0% | > 63% | > 70% |
| Oracle UB R@1 | 80.2% | stable | stable |
| Headroom (oracle − best_deployable) | 19.3pp | < 15pp | < 10pp |
