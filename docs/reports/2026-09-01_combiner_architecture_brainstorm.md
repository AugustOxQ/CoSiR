# Combiner architecture brainstorm: injecting a tiny buddy vector into frozen CLIP space

**Date:** 2026-09-01

**Scope:** literature-backed architecture recommendations only; no training result is claimed here.

**Question:** What fusion module best turns a 2--16-D, per-sample buddy condition into a controlled change of a frozen 512--1024-D CLIP/SigLIP joint embedding?

## Executive recommendation

Prioritize a **zero-initialized, residual low-rank additive adapter** as the first new family, followed by a **zero-initialized FiLM residual adapter**. Both impose the right inductive bias for CoSiR: the buddy is a small geometric coordinate, not a second high-bandwidth modality, so it should select a small movement inside the already useful frozen joint space rather than cause the model to decode a new full embedding. They are also cheap enough that capacity, identity preservation, and rank can be ablated cleanly.

The present `Combiner_new` is a reasonable descendant of composed-image-retrieval (CIR) combiners, but its original problem is materially different: CIR combines two rich, semantically expressive embeddings (reference image and modification text). CoSiR instead combines one 512--1024-D frozen embedding with a 2--16-D graph coordinate. A full concat/decode network can work, but it gives the tiny coordinate an unnecessarily indirect path and lets the output leave the backbone geometry wholesale.

There is also one important implementation correction: **the current production combiner is not identity/no-op at initialization.** Its `dynamic_scalar` final linear is zero-initialized, hence `s=0.5`, but `label_decoder`, `general_decoder`, and `combiner_layer` are ordinary random MLPs. Thus its initial result is `normalize(0.5*x + 0.5*delta_random(x,z))`, not `normalize(x)`. The proposals below deliberately fix that property. This is not a claim that the existing implementation is erroneous; it is a consequential mismatch with the project's stated identity-init design rule and should be isolated in any architecture comparison.

## What the checked repository actually implements

This section reflects the code at the time of writing, not a paraphrase.

- `CoSiRModel` freezes the Hugging Face CLIP/SigLIP-family backbone unless configured otherwise, auto-detects its joint feature width, sends the selected pooled side to `Combiner_new`, and applies an identity-initialized `Linear(feature_dim, feature_dim)` to the other side. It supports either `combine_side="img"` or `"txt"`; `general_full` is passed to the combiner but production `Combiner_new` does not use it. `ConditionPredictor` is an independent MLP from the selected CLIP-side embedding to the condition vector, trained for deployment when the graph/table is absent. See [cosirmodel.py](../../src/model/cosirmodel.py) and [condition_predictor.py](../../src/model/condition_predictor.py).
- `Combiner_new` has three `GeLUNetGradual` towers: `z: label_dim -> 128`, `x: feature_dim -> 128`, then `[z';x']: 256 -> feature_dim`. It forms `delta`, and returns `normalize((1-s)x + s*delta)`, where `s` is a clipped sigmoid in `[0.1, 0.9]`. `num_heads` is accepted but unused. `hidden_dim` is also accepted but unused by this class: the live bottleneck is always 128. The unused `CombinerGated` is a FiLM-like alternative, but it is not the production training path and has a different forward signature. See [combiner.py](../../src/model/combiner.py).
- `GeLUNetGradual` uses geometrically interpolated widths, LayerNorm/GELU/dropout between linear layers. `ConditionPredictor`, in contrast, does use `hidden_dim`: for each hidden layer it is `Linear -> LayerNorm -> GELU -> Dropout`, followed by a final linear output.
- All current model YAMLs use `hidden_dim: 1024`, `combine_side: "img"`, and a frozen backbone. `clip_base` and `siglip_base` use `embedding_dim: 16, num_layers: 6`; `clip_large`, `siglip_large`, `siglip2_base`, and `siglip2_large` use `embedding_dim: 2, num_layers: 4`. The model class is modality-agnostic, and existing reports document text-side experiments, but the checked default variant YAMLs are all image-side. See [configs/model](../../configs/model).
- The project framing is not “a learned arbitrary code”: buddy initialization is a content-aware geometric initialization from a cross-modal mutual-kNN graph. The publication plan explicitly frames that graph as the source of the trainable per-sample condition vector. In particular, prior intrinsic work found 16-D a useful graph-geometry compromise: 72% kNN preservation at 16-D versus 63% at 8-D and 79% at 32-D on the Impressions diagnostic; 2-D retained only 6.7%. See the publication [design §1--2](../superpowers/specs/2026-08-04-buddy-publication-plan-design.md) and the earlier [dimension study](2026-06-09_buddies_dim_hparam_study.md).

### Capacity reference point

At `feature_dim=512`, `buddy_dim=8`, and the current fixed 128 bottleneck, `Combiner_new` has roughly **1.0M parameters at four layers** and **1.4M at six layers** (including linear biases and normalization scales; the unused `hidden_dim=1024` does not change this). The counts below compare against that range. They exclude the per-sample buddy table, which is common to all families.

For storage, float32 buddy vectors cost `4 * d` bytes/example before optimizer state: approximately 32 MB / 1M examples at 8-D, 64 MB at 16-D, and 128 MB at 32-D. With Adam's parameter plus two moment tensors, the practical training-state footprint is about 4x the raw-vector number if the manager keeps all three in the same precision. This makes dimension a real systems axis, but not a reason to force 2-D when the graph itself is not 2-D.

## Ranked shortlist

The ranks are a recommendation for the first empirical sequence, not a claim of universal dominance. `x` denotes the frozen pooled joint-space feature, `z` the buddy vector, and every output is L2-normalized as today.

| Rank | Family | Suggested first configuration | Approx. module parameters (512-D `x`, 8-D `z`) | Why it is in this order |
|---:|---|---|---:|---|
| 1 | Low-rank residual dictionary / conditional adapter | rank 16, one small `z -> a` MLP, scalar residual gate | 10--25K | Directly expresses a low-dimensional shift and has the cleanest identity/init and rank ablation. |
| 2 | FiLM residual modulation | `z -> (gamma,beta)` with hidden 32--64 | 35--70K | Strong conditional-modulation precedent; preserves a direct frozen-feature path. |
| 3 | Content-aware low-rank (LoRA-style) residual adapter | rank 8--16, `U diag(a(z)) V^T x` | 20--40K | Lets the condition choose a feature-dependent movement without a full decoder. |
| 4 | Highway/gated residual MLP | 1--2 residual blocks, 128--256 internal width | 150--400K | A fair, safer evolution of the current content-dependent gating idea. |
| 5 | Conditional channel gate (SE-style) plus additive low-rank shift | 2-layer channel gate, rank-8 beta basis | 40--90K | Particularly conservative when most useful action is reweighting existing CLIP dimensions. |
| 6 | Tiny hypernetwork over a shared low-rank adapter | generate rank coefficients or diagonal only, rank 8--16 | 20--80K | More expressive conditional weight adaptation, but higher optimization/overfit risk. |
| 7 | Token cross-attention using `general_full` | 1 block, 1--2 heads, head dim 64 | 100--250K | Only candidate that can exploit token detail; lower priority because pooled-vector conditioning is otherwise a degenerate attention problem. |

## Candidate details

### 1. Low-rank residual dictionary / conditional adapter — first choice

Use a learned basis `B in R^(D x r)` and a small map `a(z) in R^r`, then return `normalize(x + g(x,z) * B a(z))`. A simple first version makes `a` a linear map or `8 -> 32 -> r` MLP and `g` a scalar in `[0, g_max]`; a slightly more structured version makes the columns of `B` orthonormal or periodically penalizes `B^T B - I`. This is an output-space adapter, not a decoder of an entire replacement embedding.

**Why it fits CoSiR.** The modification lies in an at-most-`r` subspace, exactly matching the intuition that a small graph coordinate should make a small number of semantic/geometric moves. `z` initially preserves its graph geometry through a short, low-distortion map, rather than being immediately expanded through several random nonlinear layers. The frozen `x` has an explicit identity highway, works identically for image or text pooled features, and the rank exposes a meaningful capacity knob.

**Identity init.** Initialize `B` normally/orthogonally but initialize the final `z -> r` linear to zero, or initialize a nonnegative residual gate `g` to zero. Then the residual is exactly zero and the output is `normalize(x)` (assuming the backbone feature is already nonzero). Use one mechanism, not both, to avoid a completely dead first step; a zero final coefficient layer still receives useful gradients through `B` only after its output weights begin to move, while a gate initialized at a small positive value (for example 0.01) improves early gradient flow but is no longer exact identity. A practical exact-identity alternative is zero-initialize `B` while leaving `a(z)` nonzero.

**Capacity and sweeps.** About 9K parameters for `r=16` with a linear coefficient map; about 10--25K with a 32--64 hidden MLP and gate, versus 1.0--1.4M today. Start with **one coefficient MLP of 1--2 linear layers**, `r in {8,16,32}`, and `buddy_dim in {8,16,32}`. Do not start at 2-D unless the question is explicitly whether the graph can be compressed that far; existing geometry evidence says it cannot preserve local neighborhoods well.

**Literature.** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) motivates frozen-base adaptation with low-rank updates; its rank-deficiency evidence is qualitative support for testing small ranks, not a prescription that CoSiR must use the same ranks. [LoRA+](https://arxiv.org/abs/2402.12354) is relevant if optimizing factorized matrices proves slow: it shows factor learning rates can matter at large width. [GeneCIS](https://arxiv.org/abs/2306.07969) grounds the broader conditional-similarity setting in which a representation should adapt to a condition rather than replace the base representation.

**Risk:** if `B` becomes a generic high-energy escape direction, it can still distort CLIP geometry; monitor residual norm/cosine shift and constrain or regularize `B` before increasing rank.

### 2. FiLM residual modulation — best simple baseline

Compute `gamma(z), beta(z) in R^D`, but use them as a residual transform: `y = x + g * (gamma(z) ⊙ x + beta(z))`, then normalize. A 32--64-wide MLP is enough; optionally restrict `gamma` with `tanh` and use a separate small scale for beta. This repairs and simplifies the intent of the unused `CombinerGated`: the condition controls feature-wise affine modulation without first asking a tiny vector to synthesize an entire `D`-vector through a deep decoder.

**Why it fits CoSiR.** FiLM is designed for a compact conditioning input that determines how an existing feature representation is used. Multiplicative modulation is especially attractive for a frozen CLIP space: it can select/reweight dimensions already present in `x`, while beta allows controlled moves where selection alone is insufficient. The same formula works for either joint-space modality. Because `z` is only read by a shallow conditioner, its meaningful local geometry is less likely to be erased.

**Identity init.** Zero-initialize the last gamma and beta projection(s), yielding `gamma=beta=0`; set the outer residual multiplier to one or omit it. Do **not** use `y=(1+gamma)⊙x+beta` without zeroing the generated outputs. Bound gamma (e.g. `0.1*tanh`) after the zeroed last projection rather than clipping a random initial gamma.

**Capacity and sweeps.** `8 -> 64 -> 2*512` is about 66K parameters; hidden 32 is about 34K. Use **one conditioner MLP with 2 linear layers**, no stack of FiLM blocks at first. Test `buddy_dim {8,16,32}`; test gamma-only, beta-only, and both once, because that ablation says whether the graph condition is selecting existing CLIP dimensions or requires a direction outside them.

**Literature.** [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) introduces feature-wise affine conditioning and shows robustness to architectural changes. [Modulating Early Visual Processing by Language](https://arxiv.org/abs/1707.00683) supplies the related conditional-normalization formulation. [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) supports the narrower channel-recalibration intuition.

**Risk:** a per-channel gamma can overfit individual samples or suppress too many dimensions; tanh bounds, a residual-norm penalty, and logging gamma/beta norms are important.

### 3. Content-aware low-rank (LoRA-style) residual adapter

Use `y = x + g * U diag(a(z)) V^T x`, with `U,V in R^(D x r)` and `a(z) in R^r`. Unlike candidate 1, the buddy selects a low-rank *linear transformation of the current CLIP feature*, so the same condition can move different input embeddings differently. A bilinear variant may concatenate a low-dimensional `V^T x` with `z` before predicting the rank coefficients.

**Why it fits CoSiR.** This retains low-rank control while recognizing that a graph condition should interact with the sample content; it is therefore closer to the current `dynamic_scalar` motivation without paying for a full `x,z -> D` decoder. The input and output are both joint-space vectors, so it is modality-agnostic. The condition's own geometry remains visible in `a(z)` and can be probed directly.

**Identity init.** Initialize `U,V` normally/orthogonally and zero-initialize the final coefficient map producing `a(z)`; alternatively zero `U`. The latter gives exact identity and immediately trains `U` from nonzero `a(z)` gradients, so it is a good default. Keep the residual gate fixed at one initially; the zero factor already guarantees identity.

**Capacity and sweeps.** `r=16` costs about `2*512*16 = 16.4K` for `U,V`, plus <2K for a shallow coefficient map and optional gate. Use **one factorized block**, not a stack: `r {4,8,16,32}`, `buddy_dim {8,16}`, then 32 only if rank-16 saturates. It is a much better test of whether content interaction matters than adding depth to the existing tower.

**Literature.** [LoRA](https://arxiv.org/abs/2106.09685) is the frozen-model/factorized-update antecedent. [HyperNetworks](https://arxiv.org/abs/1609.09106) motivates condition-generated parameters; this formulation deliberately generates only `r` coefficients rather than a dense `D x D` matrix. [FiLM](https://arxiv.org/abs/1709.07871) is the complementary example of conditioning a computation rather than encoding the condition as a replacement feature.

**Risk:** bilinear terms can make condition gradients scale with `||x||`; normalize/standardize the input as the existing pipeline expects and use gradient clipping if residual norms spike.

### 4. Zero-initialized highway residual MLP — fairest comparison to current Combiner_new

Build one or two residual blocks `h = x + T(x,z) ⊙ F(x,z)` where `F` is an MLP with a 128--256 bottleneck and `T` is a scalar or channel gate. This is deliberately a more restrained version of concat-then-decode: it preserves `x` as the carrier and asks the MLP only to predict a residual. A practical form is `F = W2 GELU(W1 [LN(x); Pz])`, where `Pz` is a shallow projection of `z`.

**Why it fits CoSiR.** It retains the current family enough to answer whether poor results stem from the mixing topology rather than from the ability to model nonlinear x--z interactions. Residual/highway paths are designed to make an identity function easy, especially important with frozen representations and a trainable table. It can be run symmetrically on either combine side.

**Identity init.** Zero-initialize `W2` and its bias in every residual block; initialize the transform gate to a negative bias or simply omit it initially. Then each block is exactly identity. Do not use a convex mix with a randomly initialized alternate output, which is the current combiner's initialization issue.

**Capacity and sweeps.** One `D -> 128 -> D` block is about 131K parameters plus the small z projection; two 256-wide blocks are about 525K. Try **1 then 2 blocks**, internal widths `{128,256}`, `buddy_dim {8,16}`. This candidate is the upper capacity control; there is little structural reason to begin with 4--6 blocks for a 2--16-D condition.

**Literature.** [Highway Networks](https://arxiv.org/abs/1505.00387) provides the transform/carry mechanism and identity-biased gates. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) provides the residual optimization rationale. [CLIP4Cir](https://arxiv.org/abs/2308.11485) is the closest provenance for CoSiR's current combiner, but combines rich image and text features after task-oriented CLIP fine-tuning; it is support for retaining a nonlinear control, not evidence for deep decoding of a tiny graph coordinate into a frozen feature space.

**Risk:** it can become a near-full replacement map as width/depth rises; retain a residual-shift metric and do not treat 4--6 blocks as the default.

### 5. Conditional channel gate (SE-style) plus low-rank additive shift

Generate a bounded channel gate `q(z) in (0,2)^D` and a small low-rank additive shift: `y = q(z) ⊙ x + B a(z)`. This separates two hypotheses: the buddy may mainly select which pre-existing joint-space coordinates matter (gate), while a rank-limited beta term handles necessary translation. It is less flexible than full FiLM but may be more geometry-conservative.

**Why it fits CoSiR.** A frozen joint embedding is likely to contain useful semantic factors already; a low-dimensional graph coordinate may need to emphasize a subset rather than invent a different representation. It is also a natural way to prevent the buddy from bypassing the backbone geometry. This is more principled for a persisted 2--16-D table than creating a fresh high-dimensional semantic vector.

**Identity init.** Parameterize `q=1+epsilon*tanh(q_raw)` with `epsilon <= 0.1`, zero-init the last gate projection so `q=1`, and zero-init the additive basis or its final coefficient map. Then `y=x` exactly.

**Capacity and sweeps.** A 32--64 hidden gate plus rank-8 beta basis is about 40--90K. Use **one gate**, not a stack; `buddy_dim {8,16}`, beta rank `{0,8,16}`. The rank-0 branch is valuable: it asks whether selection alone is enough.

**Literature.** [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) establishes efficient channel recalibration. [FiLM](https://arxiv.org/abs/1709.07871) supplies the more general affine-conditioning view. [Conditional Batch Normalization](https://arxiv.org/abs/1707.00683) is a related example of a compact language vector controlling channel-wise parameters.

**Risk:** pure channel rescaling cannot rotate the representation; if beta rank 0 loses badly, do not keep tuning the gate indefinitely.

### 6. Restricted hypernetwork — useful second-wave expressiveness test

Let a small hypernetwork map `z` to the coefficients of a shared adapter, not to a dense `D x D` matrix: for example, `z -> a(z) in R^r` in the factorized adapter above, or `z -> d(z) in R^r` controlling a block-diagonal/basis bank. The key restriction is that the table generates **only O(r)** or **O(rD)** modulation values, never an arbitrary per-sample full model.

**Why it fits CoSiR.** Hypernetworks are the cleanest formulation of “the condition chooses how a fixed model adapts.” They can represent different local transformations for different graph neighborhoods without giving each sample a costly stored adapter. Restricting the generated object maintains the low-rank prior, bounds the parameter count, and preserves compatibility with both combine sides.

**Identity init.** Make the generated coefficients a zero-initialized final head, or generate a residual around fixed identity coefficients. If generating a diagonal, output `1 + 0.1*tanh(d_raw)` with zero head. Never initialize a generated full dense matrix near random.

**Capacity and sweeps.** A `8 -> 64 -> 16` coefficient generator plus an `r=16` adapter is only about 20K; a generator that emits both U/V scales or channel gates is 40--80K. Use **one 2-layer generator**, `r {8,16}`, `buddy_dim {8,16}`. It should be tried only after the simpler rank adapter establishes that conditional coefficients help.

**Literature.** [HyperNetworks](https://arxiv.org/abs/1609.09106) is the primary source for generating another network's weights. [LoRA](https://arxiv.org/abs/2106.09685) provides the low-rank restraint. [FiLM](https://arxiv.org/abs/1709.07871) is the lower-variance special case where the generated parameters are affine scales and shifts.

**Risk:** a hypernetwork can memorize the sample table through nonlinear coordinates; restrict what it emits and compare against a linear `z -> coefficients` baseline before attributing a gain to “dynamic weights.”

### 7. One-block cross-attention over actual encoder tokens — only if token information is the hypothesis

Use `general_full`, which `CoSiRModel` already computes but the production combiner ignores. Project `z` into one or a few query tokens; let them cross-attend to the selected side's projected image patches or text-token sequence; map the attended context to a residual on the pooled `x`. The pooled feature must remain the residual carrier: `y=x+W_o Attn(Q(z),K(general_full),V(general_full))`.

**Why it fits, and why it is last.** This is the only shortlisted family that can let a buddy select *where* in an image/text sequence to read. That could matter if the graph condition corresponds to a local object/word rather than a global direction. But when only a single pooled vector is used, attention over its arbitrary chunks is not semantically meaningful; it is merely a costly gated linear map. Moreover, text token sequences and vision patch grids differ, so one shared block is modality-agnostic only in tensor shape, not necessarily in token statistics. This family should therefore test a real token-level hypothesis, not replace the simple pooled fusion baseline by default.

**Identity init.** Zero-initialize the final attention output projection `W_o` and residual bias. Then attention may compute internally but its residual is exactly zero. Keep a pre-norm residual layout and one block.

**Capacity and sweeps.** One query, one/two heads, head width 64 costs roughly 100--250K including Q/K/V/output projections and a small output MLP, depending on whether projections are shared. Test **one block only**, heads `{1,2}`, `buddy_dim {8,16}`. Token count, not buddy dimension, drives its activation cost.

**Literature.** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) establishes query/key/value cross-attention. [FiLM](https://arxiv.org/abs/1709.07871) remains the more appropriate compact-condition baseline. [GeneCIS](https://arxiv.org/abs/2306.07969) motivates conditional similarity but does not itself demonstrate that CoSiR needs token-level fusion.

**Risk:** higher memory/variance and a modality-specific token-distribution mismatch can erase the desired symmetry; reject it if it does not beat pooled low-rank adapters under equal retrieval and shift budgets.

## What CIR literature does and does not transfer

The repository attribution to CLIP4Cir and GeneCIS is real and useful. CLIP4Cir trains a Combiner for composed retrieval, and its public training recipe uses a high-dimensional projection/hidden configuration; later CIR work explicitly describes a CLIP4Cir-like adaptive combiner as combining separately guided image/text features with learned convex weights and a nonlinear concatenation branch. See [CLIP4Cir](https://arxiv.org/abs/2308.11485), its [reference implementation](https://github.com/ABaldrati/CLIP4Cir), [GeneCIS](https://arxiv.org/abs/2306.07969), and [Cross-modal Feature Alignment and Fusion for CIR](https://openaccess.thecvf.com/content/CVPR2024W/CVFAD/papers/Wan_Cross-modal_Feature_Alignment_and_Fusion_for_Composed_Image_Retrieval_CVPRW_2024_paper.pdf).

The transferable conclusion is modest: late fusion plus a residual/nonlinear composition branch is a credible baseline for conditional retrieval. The non-transferable leap is that the same deep concat decoder is automatically appropriate when the condition has only 2--16 dimensions and is a graph coordinate. In CIR, both inputs carry dense semantic content; CoSiR's buddy carries a deliberately compressed relation signal. This is why the first ablation should compare **residual topology and rank**, rather than simply tune the present MLP's number of layers.

## Depth: what is justified

There is no literature result that can dictate an optimal layer count for this exact graph-conditioned frozen-embedding problem. The relevant structural argument is stronger than a generic “shallower is better” claim:

1. `z` has 2--16 coordinates, and its initialization already has a geometric meaning. A 4--6-layer nonlinear expansion is more likely to reparameterize that geometry than to extract new information from it.
2. The backbone cannot co-adapt to rescue an unstable combiner, because it is frozen. The adaptation module should therefore bias toward a small, smooth movement of a good representation.
3. Current `num_layers` is applied separately to all three production towers, so 4--6 means substantially more than one shallow fusion MLP. In contrast, the `hidden_dim=1024` configuration knob is not used by `Combiner_new` at all.

Recommended depth by family:

| Family | Recommended initial depth | Escalate only if |
|---|---|---|
| Low-rank dictionary / LoRA-style adapter | 0--1 coefficient MLP hidden layer | rank and 1-layer models plateau while residual norm remains safely small. |
| FiLM / SE gate | 1 hidden layer (two linears) | gamma/beta-only ablations show real benefit but underfit systematically. |
| Highway residual MLP | 1 block, then 2 | a one-block residual beats linear/rank adapters but leaves a stable capacity gap. |
| Restricted hypernetwork | 1 hidden layer | linear coefficient generation loses but does not show table memorization. |
| Token attention | exactly 1 block initially | a token-local diagnostic demonstrates information unavailable in pooled `x`. |

The existing 4--6 layer setting is defensible as a historical CIR-derived baseline, not as a presently justified default. The fair experiment is to hold identity initialization, output normalization, optimizer, and shift regularization fixed while comparing 1--2 residual blocks to the current tower, rather than infer depth from a cross-family comparison.

## Buddy dimensionality: guidance and a focused sweep

Information-bottleneck work provides a principle--retain task-relevant information while limiting unnecessary information--but **does not supply a universal dimension such as 8 or 16**. [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410) supports treating capacity as a controlled trade-off; it is not evidence that a particular latent width is optimal here. Similarly, LoRA's useful low ranks motivate a rank ablation for the *adapter*, not direct transfer of language-model rank rules to a per-sample graph embedding.

CoSiR has stronger local evidence than either generic literature heuristic: the prior graph study found a roughly linear participation ratio through 32 dimensions and substantial local-neighborhood loss at 2-D. Therefore:

- **Primary sweep:** `buddy_dim in {8,16,32}`. This asks whether 16 is the graph-geometry/storage sweet spot and whether 32's intrinsic preservation translates to retrieval.
- **Diagnostic lower bound:** include `2` only as a deliberately lossy condition, ideally for the best two fusion families. It should not be the default for large/SigLIP configs merely because the backbone is wider.
- **Do not sweep every cross-product first.** Fix adapter rank at 16 for candidate 1/3, and test 8/16/32 dimensions. If 16 beats 8 and 32 does not help, then test ranks 8/16/32 at 16-D. This separates graph-information capacity from fusion capacity.
- **Predictor compatibility matters.** `ConditionPredictor` must reconstruct `z` from `x` at deployment. A buddy dimension that improves oracle/table-conditioned retrieval but sharply worsens predicted-condition retrieval is not automatically a deployment win; report both tiers.

The geometry-preservation concern favors shallow, approximately linear maps near the buddy input, and favors evaluating whether learned `a(z)` or FiLM parameters preserve neighbor relationships from the initialized table. It does not forbid nonlinear fusion, but it makes a deep random expansion a hypothesis needing evidence rather than a neutral default.

## Which axes are most likely to matter

1. **Fusion family / residual topology — highest priority.** The present architecture can replace half of the frozen feature with a random-initialized decoded branch at step zero, and then uses deep concat/decode despite a tiny condition. Whether the module has an exact identity path and restricts its movement is likely to dominate both stability and retrieval. This is also the axis most directly implicated by the project’s existing architecture-asymmetry findings.
2. **Buddy dimensionality — second priority.** Unlike a generic latent sweep, it changes how much of the cross-modal graph geometry can exist in the table and incurs real per-sample storage cost. Existing intrinsic evidence makes 2 versus 8/16 plausibly consequential; 16 versus 32 is more likely a marginal but worthwhile question.
3. **Adapter rank/width — third priority, coupled to family.** In low-rank families this is the honest capacity knob and likely more interpretable than MLP depth. It should be swept after a family and buddy dimension are selected.
4. **Depth — fourth priority.** For this low-rank conditioning signal and frozen carrier, 1 versus 2 blocks is worth testing; 4 versus 6 is unlikely to be the most informative first question. The current number of layers changes several towers simultaneously, which makes it a poor first diagnostic.

## Minimal, interpretable ablation order

1. Establish an **identity-initialized current-topology control**: `y=normalize(x + F(x,z))` with the final residual projection zeroed, one 128-wide block. This separates identity initialization from the rest of the architecture.
2. Compare it with candidate 1 (rank-16 low-rank dictionary) and candidate 2 (64-hidden FiLM), all at `buddy_dim=16`, with the same predictor, other-side projection, training protocol, and L2 normalization.
3. For the winner(s), evaluate `{8,16,32}` buddy dimensions and report both table/oracle and predictor/deployable retrieval, residual norm, cosine shift from `x`, and condition-prediction error.
4. Only then sweep rank `{8,16,32}` or residual block count `{1,2}`. Reserve hypernetwork and token attention for a clear failure mode: respectively, evidence that linear conditional coefficients are insufficient, or evidence that pooled features discard condition-relevant token locality.

The critical success criterion is not only top-1 retrieval: the winning family should improve the appropriate retrieval tier **without** a disproportionate increase in conditioned-vs-frozen cosine shift, condition-table drift from buddy initialization, or collapse of condition diversity. This keeps the architecture aligned with the paper-facing claim that buddies contribute content-aware graph geometry rather than an unconstrained per-sample lookup trick.

## References

- Baldrati, Bertini, Uricchio, and Del Bimbo. [Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features (CLIP4Cir)](https://arxiv.org/abs/2308.11485), 2023.
- Vaze, Carion, and Misra. [GeneCIS: A Benchmark for General Conditional Image Similarity](https://arxiv.org/abs/2306.07969), CVPR 2023.
- Perez et al. [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871), AAAI 2018.
- de Vries et al. [Modulating Early Visual Processing by Language](https://arxiv.org/abs/1707.00683), NeurIPS 2017.
- Hu, Shen, and Sun. [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html), CVPR 2018.
- Srivastava, Greff, and Schmidhuber. [Highway Networks](https://arxiv.org/abs/1505.00387), 2015.
- He et al. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.
- Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
- Ha, Dai, and Le. [HyperNetworks](https://arxiv.org/abs/1609.09106), ICLR 2017.
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
- Alemi et al. [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410), ICLR 2017.
