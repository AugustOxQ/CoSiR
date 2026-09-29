# Independent Adversarial Review: PercepT Baseline Implementation & Buddy-vs-PercepT Evaluation

**Reviewer:** Antigravity (Independent Adversarial Review)  
**Date:** 2026-09-27  
**Branch:** `experiment/percept_topic_pipeline`  
**Primary Reviewed Documents:**
- `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` (Master Report)
- `docs/reports/2026-09-27_key_advantages.md` (Executive Summary)
- Core PercepT and Buddy scripts and pilot reports in `src/test/20260922_percept_topic_pipeline/` and `src/test/20260923_artelingo_buddy_analysis/`
- Reference Paper: *Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data* ([arXiv:2606.03345v1](https://arxiv.org/html/2606.03345v1))

---

## Executive Summary & Verdict

### 1. Is the PercepT Implementation Sound?
**Verdict: Unsound — compromised by two critical mathematical/logical implementation bugs and one fundamental dataset modeling divergence.**

1. **Inverted Center Pruning (The 180° Pruning Bug):** PercepT paper §4.3 explicitly states that DEC drives unassigned/outlier centers to *large* norms and provides an algorithm to *discard* centers with norms exceeding a threshold $\tau$. The implementation in `run_percept_stage1_pilot.py` line 260 did the exact opposite: it sorted centroid norms in descending order and **kept the 67 highest-norm centers**, while pruning the compact, densely populated low-norm centers. The "severe occupancy collapse" (50/67 centers with <1% occupancy, 21 empty centers) that forms the core empirical indictment of PercepT across all project reports is a direct consequence of keeping the outlier centers the paper designed a filter to discard.
2. **Reconstruction Loss Dimension-Scaling Bug:** In `run_percept_stage1_pilot.py` lines 221–223, reconstruction loss was computed via PyTorch's `F.mse_loss(reconstruction, inputs)` with default reduction `'mean'`, which divides by the input feature dimension $D = 2,816$. Meanwhile, DEC's clustering loss `F.kl_div` with `reduction='batchmean'` does not divide by cluster count $K$. Because inputs are unit-norm, the MSE is $\approx 0.00008$, attenuating the reconstruction gradient by a factor of 2,816 relative to KL ($\approx 0.25$). PercepT paper's $\lambda_R = 1.0$ was formulated for sample-level squared or unsquared L2 norm (magnitude $\approx 0.2$). Dividing by $D=2,816$ effectively crippled the reconstruction anchor by $\sim 3,000\times$, allowing unanchored DEC to collapse cluster assignments during training.
3. **Destruction of Multi-Caption Diversity:** In the PercepT paper (§4.1, §4.4), the method forms topics over image-caption pairs $(I, C)$ because an artwork evokes multiple distinct emotional responses from different annotators. Stage 2 targets are defined as multi-hot indicators $O[i] = 1$ if *any* caption of image $I$ was assigned to topic $i$. This codebase pre-averaged all captions for each painting into a single mean vector before Stage 1, eliminating multi-caption emotional diversity, reducing paintings to single vectors, and forcing Stage 2 to invent an ad-hoc probability threshold that resulted in 0.00% multi-label targets.

### 2. Is the Comparison and Reporting Fair?
**Verdict: Unfair and Systematically Biased against PercepT.**

1. **Gross Stage 2 Cherry-Picking:** Both the Master Report (§6a) and `key_advantages.md` (§1) claim that "End-to-end (Stage 1 + Stage 2), buddy wins outright", citing buddy's Stage 2 macro AUC of **0.5978** beating "PercepT's own cited Stage 2 result (**0.5690**) by +0.0288." This comparison cites an initial, un-swept smoke-test run (`percept_stage2_pilot_report.md`) with an untuned learning rate (`1e-3`) and a degenerate target threshold. In the project's own subsequent sweep and 14-seed extended stress test (`percept_stage2_extended_seed_pilot_report.md`), PercepT's tuned Stage 2 mapper achieved macro AUC **0.8256** (4-seed mean) and **0.8290** (14-seed mean, range 0.8266–0.8315). The reports completely concealed this 0.8290 result from the executive summary and headline comparison tables to claim an end-to-end victory for buddy.
2. **The "433× Loss-Scale Distortion" Artifact:** The Master Report (§6) and `key_advantages.md` cite a "433× loss-scale distortion" when feeding buddy's 32-D embedding into PercepT as proof that PercepT's recipe is fragile. The physical squared reconstruction error $||h - \hat{h}||_2^2$ differed by only $2.32\times$ ($0.225$ vs $0.522$). The 433× shift was an artifact of $D$ changing from 2,816 to 32 in the denominator of `F.mse_loss` ($2,816 / 32 = 88\times$).
3. **Asymmetric Out-of-Sample Evaluation:** Buddy's Stage 1 Pareto-clearing numbers (emotion 0.1249, genre 0.2404) were obtained by running Leiden community detection directly on the held-out test split (re-clustering the test set). PercepT was evaluated strictly out-of-sample by assigning test points to frozen train centroids.
4. **Bespoke Conjunctive Pareto Bar:** The Pareto bar (emotion > 0.1236 AND genre > 0.1954) was constructed directly from buddy's own internal development experiments. PercepT's faithful recipe produced a genre AMI of 0.3288 (dramatically outperforming buddy's 0.2404), but was classified as a failure because emotion AMI was 0.1092. Under the paper's own harmonic mean evaluation metric, PercepT faithful (0.1639) and Buddy Attention-h1 (0.1644) are virtually identical.
5. **Genre Sample Size Instability:** Only $n=159$ held-out paintings have genre annotations. Gating on genre AMI across 40–67 clusters with 159 points creates high sampling variance that was treated as a rigid deterministic gate.

---

## Detailed Numbered Findings

### Category A: Implementation Bugs in PercepT Codebase

#### Finding 1: 180° Inversion in Center Pruning (`prune_centers`)
- **File & Lines:** `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`, lines 257–262 (and imported across all subsequent pilots):
  ```python
  def prune_centers(centers: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
      """Keep the 67 highest-norm final centers as the documented pruning proxy."""
      norms = torch.linalg.vector_norm(centers.detach(), dim=1)
      surviving = torch.argsort(norms, descending=True)[:N_SURVIVING_CLUSTERS]
      return centers.detach()[surviving], surviving.cpu().numpy()
  ```
- **Analysis:**
  - In PercepT's published paper (arXiv:2606.03345v1, Section 4.3, *Dynamic Selection of Perception Experiences*), the authors explain:
    > *"We dynamically filter irrelevant clusters by measuring the norm of each cluster center $\mu_i$: the DEC loss drives centers with few assigned samples to large norms, so we use those norms to discard them. We use Algorithm 1 to dynamically set a filtering threshold $\tau$ by finding the point of sharp transition via the maximum finite difference of sorted norms. We then filter out cluster centers with norms larger than $\tau$."*
  - The mathematics of DEC confirms this: when a center $\mu_j$ receives negligible soft assignment weight ($p_{ij} \approx 0, q_{ij} > 0$), the DEC gradient $\frac{\partial L_{KL}}{\partial \mu_j} = -2 \sum_i (1 + ||z_i - \mu_j||^2)^{-1} (p_{ij} - q_{ij}) (z_i - \mu_j)$ is repulsive, pushing unpopulated centers away from the origin and away from the data manifold into high-norm peripheral space.
  - The implementation in this repo did the exact opposite of the paper: it set `descending=True` and kept the 67 centers with the **largest** norms, while pruning the compact, densely populated centers near the data manifold.
  - In `percept_stage1_faithful_recipe_pilot_report.md`, before pruning, 23/100 centers were populated (>1% occupancy). By keeping the 67 highest-norm centers, 47/67 surviving centers were empty/small (<1%), and only 20 populated centers survived. The 33 centers that were discarded were the low-norm populated centers.
  - This bug directly caused the headline "occupancy collapse" (50/67 centers below 1%, 21 zero-occupancy centers) that is cited dozens of times across the reports as proof that PercepT fails.
- **Severity:** **Bug that fundamentally changes conclusions** (Critical).

---

#### Finding 2: Reconstruction Loss Scaling Bug via `F.mse_loss`
- **File & Lines:** `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`, lines 221–223; `run_percept_stage1_faithful_recipe_pilot.py`, lines 164–166:
  ```python
  kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
  reconstruction_loss = F.mse_loss(reconstruction, inputs)
  total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss
  ```
- **Analysis:**
  - `F.mse_loss(..., reduction='mean')` computes the mean squared error averaged over both the batch dimension $N$ and the feature dimension $D = 2,816$:
    $$\text{MSE} = \frac{1}{N \cdot D} \sum_{i=1}^N \sum_{d=1}^D (h_{id} - \hat{h}_{id})^2$$
  - In contrast, `F.kl_div(..., reduction='batchmean')` computes:
    $$\text{KL} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{ik} \log \frac{p_{ik}}{q_{ik}}$$
    which is summed over the $K$ clusters and divided only by $N$.
  - Because inputs $h_i$ are unit-normalized ($||h_i||_2 = 1$), $\sum_{d=1}^D h_{id}^2 = 1$. The maximum possible MSE for $\hat{h}=0$ is $1/D = 1/2816 \approx 0.000355$. For a well-trained autoencoder, the MSE is $\approx 0.00008$.
  - In DEC/IDEC literature (Guo et al. 2017 Eq. 9; Xie et al. 2016) and PercepT Section 4.1, reconstruction loss $L_R$ is defined as sample-wise squared error $||h - \hat{h}||_2^2$ or unsquared error $||h - \hat{h}||_2$ (not divided by $D$). The sample-wise squared error is $D \times \text{MSE} \approx 2816 \times 0.00008 \approx 0.225$, which is on the exact same numerical scale as KL ($\approx 0.25$).
  - With $\lambda_R = 1.0$ applied to `F.mse_loss`, the reconstruction loss was $\approx 0.000085$, which was $\mathbf{760\times}$ smaller than the KL loss ($0.065$). The gradient backpropagated from reconstruction into the encoder was attenuated by 2,816.
  - The authors noted in `2026-09-23_artelingo_percept_stage1_report.md` §III.1: *"At convergence, KL was 0.064992 while reconstruction was 0.000085, so the reconstruction anchor was roughly 760 times smaller and DEC's self-sharpening effectively dominated the joint loss."* Instead of recognizing that PyTorch's `F.mse_loss` averages over feature dimensions, they concluded that PercepT's formulation is inherently flawed, which led them to invent `LAMBDA_BALANCE=1000`.
- **Severity:** **Bug that fundamentally changes conclusions** (Critical).

---

#### Finding 3: Caption Pre-Averaging Destroys Multi-Caption Emotional Granularity
- **File & Lines:** `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`, lines 112–125; `run_percept_stage2_pilot.py`, lines 101–114.
- **Analysis:**
  - In PercepT's paper (§4.1, §4.4), Stage 1 clusters image-caption pairs $(I, C)$. In ArtELingo, each painting has multiple captions written by different annotators expressing different emotions (e.g. awe, fear, contentment).
  - The paper's Stage 2 multi-hot ground truth is defined explicitly as:
    $$O[i] = 1 \quad \text{if any caption of image } I \text{ is assigned to cluster } c_i$$
    This models the reality that an artwork evokes diverse perceptual experiences across different viewers.
  - This repo's script `run_percept_stage1_pilot.py` mean-pooled all RoBERTa token embeddings across all captions of a painting into a single average affect vector before training:
    ```python
    return embedding_sums / embedding_counts[:, None]
    ```
  - This reduced each painting to a single point in Stage 1, completely destroying the multi-caption distribution.
  - Then, in Stage 2 (`run_percept_stage2_pilot.py`), because there was only one vector per painting, the authors had no multi-caption assignments. Instead, they invented an artificial probability threshold `q > 2.0 / 40` on the single DEC output. When this threshold produced 0.00% multi-labeled samples, they claimed in `buddy_stage2_pilot_report.md` line 16 that *"both realized Stage 2 target sets are single-label in practice"*, obscuring the fact that their own pre-averaging step destroyed the multi-label nature of the data.
- **Severity:** **Methodological distortion that mischaracterizes the baseline** (Major).

---

#### Finding 4: DEC Target Distribution $P$ Update Frequency
- **File & Lines:** `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`, lines 216–226.
- **Analysis:**
  - In standard DEC (Xie et al. 2016, Guo et al. 2017), the target distribution $P$ is treated as a fixed target for an epoch or multiple iterations and updated periodically (e.g. every few hundred iterations) to prevent unstable positive-feedback loops.
  - In `run_percept_stage1_pilot.py`, $P$ is recomputed at every single full-batch epoch directly from the current $Q$:
    ```python
    q = soft_assignments(latent, centers)
    p = target_distribution(q)
    kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
    ```
  - While $P$ is detached (`.detach()`), updating $P$ continuously on every gradient step accelerates self-reinforcing cluster shrinkage, which, when combined with the crippled reconstruction anchor (Finding 2), exacerbated the collapse.
- **Severity:** **Implementation divergence that worsens stability** (Minor / Moderate).

---

### Category B: Fairness and Accuracy of Comparison and Reporting

#### Finding 5: Gross Stage 2 Cherry-Picking: Comparing Buddy's 0.5978 to a Discarded 0.5690 Smoke Test While Concealing 0.8290
- **Report Claims:**
  - `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`, lines 37–41 & 383–391:
    > *"Stage 2 wiring, done and validated... PercepT's own unmodified image-only attention-pooling mapper, retrained against buddy's frozen topics instead of DEC's, reaches macro AUC 0.5978 — beating its own train-marginal baseline by +0.098 and beating PercepT's own cited Stage 2 result (0.5690) by +0.0288. This is a genuine, working, better-performing end-to-end replacement for PercepT's full two-stage pipeline..."*
  - `docs/reports/2026-09-27_key_advantages.md`, lines 11–16:
    > *"End-to-end (Stage 1 + Stage 2), buddy wins outright. Buddy's image-only Stage 2 mapper reaches macro AUC 0.5978 against PercepT's own cited 0.5690 — a +0.0288 margin..."*
- **Analysis:**
  - What was PercepT's actual Stage 2 result in this repository?
  - `percept_stage2_pilot_report.md` (dated 2026-09-23 02:54:48) was an initial smoke-test run with an untuned learning rate (`1e-3`) and a threshold `q > 2.0/40` that yielded 0.5690.
  - The repository's own Stage 2 closing report (`docs/reports/2026-09-23_artelingo_percept_stage2_report.md`, Section III.1) explicitly states:
    > *"That initial apparent success came with an important negative finding... The pilot was thus an effectively single-label classifier, not the genuine multi-label mapping the Stage-2 design intended. This prompted the threshold sweep rather than treating 0.5690 as the final answer."*
  - The subsequent sweep (`percept_stage2_sweep_pilot_report.md`) swept thresholds and learning rates, finding that `q > 1.2/40` and `lr=3e-3` achieved macro AUC **0.8256** (Part C).
  - The extended seed stress test (`percept_stage2_extended_seed_pilot_report.md`, dated 2026-09-23 09:02:42) tested this across 14 seeds, achieving:
    - **Mean macro AUC: 0.8290**
    - **Min / Max: 0.8266 / 0.8315**
    - **Standard Deviation: 0.0014**
    - **95% Confidence Interval: [0.8283, 0.8298]**
  - All 14 seeds of PercepT scored above 0.826!
  - Yet, three days later, `buddy_stage2_pilot_report.md` and the master report cited **only** the 0.5690 smoke-test number as "PercepT's cited Stage 2 result", compared it to buddy's single-seed **0.5978**, and declared that "buddy wins outright" by +0.0288.
  - Neither the Master Report nor `key_advantages.md` discloses anywhere that PercepT's validated Stage 2 mapper reached 0.8290. Citing an explicitly discarded pilot number to claim an end-to-end victory is a severe reporting violation.
- **Severity:** **Severe reporting distortion and comparison unfairness** (Critical).

---

#### Finding 6: The "433× Loss-Scale Distortion" in Candidate 2 Is an Artifact of $D$-Division in `F.mse_loss`
- **Report Claims:**
  - `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`, line 303:
    > *"The loss-scale audit explains why: the reconstruction/KL ratio shifted 433× from the original recipe, because a 32-D input into an architecture built for 2,816-D makes the 128-D latent overcomplete (4× wider than the input)..."*
  - `docs/reports/2026-09-27_key_advantages.md`, line 57:
    > *"a fixed reconstruction/KL loss balance ($\lambda_R=1$) that this investigation's own candidate-2 pilot showed is not portable to a different input width without a full loss-scale audit (a 433× ratio shift when the input changed from 2,816-D to 32-D)."*
- **Analysis:**
  - In `percept_on_buddy_embedding_pilot_report.md` Table in Section "Loss-scale audit":
    - Original 2,816-D input: Final DEC MSE = $0.000080$, $\text{MSE} \times \text{dim} = \mathbf{0.225280}$, KL = $0.391414$. Ratio $\text{MSE}/\text{KL} = 0.000204$.
    - Buddy 32-D input: Final DEC MSE = $0.016305$, $\text{MSE} \times \text{dim} = \mathbf{0.521753}$, KL = $0.184118$. Ratio $\text{MSE}/\text{KL} = 0.088556$.
    - Ratio shift: $0.088556 / 0.000204387 = \mathbf{433.28\times}$.
  - Notice the actual physical squared reconstruction error $||h - \hat{h}||_2^2 = \text{MSE} \times \text{dim}$:
    $$0.521753 / 0.225280 = \mathbf{2.32\times}$$
  - The true physical error ratio between the two models was only $2.32\times$. The remaining factor of $187\times$ in the $433\times$ shift was caused entirely by:
    1. $2,816 / 32 = \mathbf{88\times}$ due to dividing by $D$ in `F.mse_loss`.
    2. $0.3914 / 0.1841 = \mathbf{2.13\times}$ due to different final KL values.
    $$2.32 \times 88 \times 2.13 = 433.28\times$$
  - Presenting this mathematical artifact of PyTorch's `reduction='mean'` as an intrinsic architectural pathology of PercepT ("non-portable without a full loss-scale audit") is scientifically incorrect.
- **Severity:** **Flawed causal attribution and misleading reporting** (Moderate / Major).

---

#### Finding 7: Asymmetric Out-of-Sample Evaluation Protocol
- **Code & Report Locations:**
  - `src/test/20260923_artelingo_buddy_analysis/run_attention_h1_embedding_snapshot_pilot.py`, lines 394–397.
  - `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`, lines 124–130.
- **Analysis:**
  - In `run_attention_h1_embedding_snapshot_pilot.py`, buddy's held-out emotion and genre AMIs (0.1249 and 0.2404) were computed using `heldout_community_post`:
    ```python
    heldout_community_post = community_and_metrics(
        arch, heldout_embedding_post_np, "final-heldout-attn1", heldout_pipeline,
        affect_pilot, str(device), len(heldout_paintings), single_modality,
    )
    ```
  - This function constructs a mutual-kNN graph over the 9,365 held-out test paintings and runs Leiden community detection directly on the test set. It finds an independent, test-specific partition (18 communities) with arbitrary community IDs that have no connection to the train community IDs.
  - In contrast, PercepT was evaluated by freezing its 67 train centroids and mapping each test point to the nearest train centroid.
  - Re-clustering the entire test set transductively with Leiden allows global density optimization over test points, which is a fundamentally easier and more privileged task than out-of-sample inductive projection onto frozen centroids.
  - When buddy was eventually forced to perform out-of-sample assignment onto frozen train communities via k-NN transfer (`run_heldout_label_transfer_pilot.py`), its numbers were tested at $k=20$, but the core baseline figures that established the Pareto bar came from independent test re-clustering.
- **Severity:** **Fairness concern and protocol asymmetry** (Moderate).

---

#### Finding 8: Structurally Biased Conjunctive Pareto Bar
- **Report Locations:** `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`, line 112; `docs/reports/2026-09-27_key_advantages.md`, line 19.
- **Analysis:**
  - The investigation defined its success criterion as a strict conjunction:
    $$\text{emotion AMI} > 0.1236 \quad \mathbf{AND} \quad \text{genre AMI} > 0.1954$$
  - How were these thresholds chosen? As shown in `docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md` lines 51 & 101, $0.1236$ was the exact train emotion AMI of buddy's earlier "Late fusion — union" run, and $0.1954$ was the train genre AMI of "Hierarchical refinement".
  - This bar was calibrated to match internal buddy experiments. When Buddy Attention-h1 cleared emotion at 0.1249, it cleared by a microscopic margin of $+0.0013$. In multi-seed testing (`attention_h1_baseline_seed_stress_pilot_report.md`), seed 7 missed the emotion bar (0.1213).
  - Meanwhile, PercepT's faithful recipe produced a genre AMI of **0.3288** — vastly outperforming buddy's **0.2404** by $+0.0884$ (a +36.8% relative gain). However, because its emotion AMI was $0.1092$ (missing $0.1236$ by $0.0144$), it was classified as an outright failure.
  - In PercepT's original paper (§5.4.2), the objective is evaluated by the **harmonic mean** of emotion and genre scores. Under the paper's own harmonic mean metric:
    - PercepT faithful recipe: $\text{HM}(0.1092, 0.3288) = \mathbf{0.1639}$
    - Buddy Attention-h1: $\text{HM}(0.1249, 0.2404) = \mathbf{0.1644}$
    The two systems are functionally tied on overall semantic grounding ($\Delta = 0.0005$). The narrative that buddy "decisively clears while PercepT fails" is an artifact of imposing a rigid conjunctive threshold tailored to buddy's exact operating point.
- **Severity:** **Evaluation bias and selective framing** (Moderate).

---

#### Finding 9: Small Sample Size of Genre Ground Truth ($n=159$) Unaccounted for in Pareto Gating
- **File & Lines:** `run_buddy_percept_matched_silhouette_audit_pilot.py`, lines 80–86; `buddy_percept_matched_silhouette_audit_pilot_report.md`, lines 20–21.
- **Analysis:**
  - While the held-out split contains 9,365 paintings for emotion evaluation, **only 159 paintings** have genre annotations.
  - In PercepT, evaluating Adjusted Mutual Information between 67 predicted clusters and genre ground truth over $N=159$ paintings means there are, on average, only $\approx 2.37$ samples per cluster.
  - Adjusted Mutual Information on small sample sizes with large contingency tables has high variance. Small changes in seed or clustering boundaries cause genre AMI to fluctuate dramatically (e.g. PercepT genre AMI was 0.3288 in the faithful report, 0.3453 in the snapshot refit, 0.2617 in K=60/40, and 0.2274 on node404).
  - Despite this known high variance, genre AMI > 0.1954 was treated as a rigid gate where missing by 0.003 was classified as an unambiguous failure.
- **Severity:** **Statistical fragility in evaluation gating** (Moderate).

---

#### Finding 10: Mischaracterization of `LAMBDA_BALANCE` as an "Unprincipled Ad-Hoc Hack"
- **Report Locations:** `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` line 26; `docs/reports/2026-09-27_key_advantages.md` line 26.
- **Analysis:**
  - The reports repeatedly criticize PercepT's K=60/40 success as relying on *"an admitted, ad hoc `LAMBDA_BALANCE=1000` regularizer with no principled derivation, invented specifically to stop collapse."*
  - In modern deep clustering and unsupervised representation learning literature (e.g., SwAV [Caron et al., NeurIPS 2020], SeLa [Asano et al., ICLR 2020], DEC variants, and information-theoretic clustering), adding an entropy or KL penalty $KL(\text{Uniform} || \frac{1}{N}\sum_i q_i)$ to prevent degenerate partition solutions is a standard, mathematically principled, and ubiquitous technique.
  - More importantly, PercepT's paper did not need an explicit balance loss because it had two other anti-collapse mechanisms: (1) a properly scaled reconstruction anchor, and (2) an outlier pruning rule. Because the reimplementation broke both of these mechanisms (Findings 1 & 2), adding the balance regularizer was the only way to stabilize the model. Criticizing the baseline for requiring a fix to a problem created by implementation bugs is circular and unfair.
- **Severity:** **Unfair characterization of baseline and literature context** (Moderate).

---

### Category C: Verified Sound / Clean Aspects (No Issue Found)

#### Finding 11: Train / Held-Out Split Isolation (No Data Leakage)
- **Files Checked:** `run_percept_stage1_pilot.py`, `run_buddy_percept_matched_silhouette_audit_pilot.py`, `run_pipeline.py`.
- **Details Checked:**
  - Verified whether training sets (CLIP embeddings, RoBERTa embeddings, patch features) overlap with held-out validation/test splits.
  - Checked `assert_matching_paintings` calls in audit scripts: 61,402 train paintings and 9,365 held-out paintings have empty intersection ($A \cap B = \emptyset$).
  - Autoencoder pretraining, K-means initialization, DEC joint training, PCA fits, and mapper training are strictly isolated to train-split data.
- **Conclusion:** **Checked train/held-out split isolation; found no data leakage.**

---

#### Finding 12: Quantitative Report Citation Accuracy Against Source Pilot Logs
- **Files Checked:** All tables and numerical claims in `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` and `docs/reports/2026-09-27_key_advantages.md` cross-referenced with:
  - `attention_h1_baseline_seed_stress_pilot_report.md`
  - `attention_h1_noise_schedule_pilot_report.md`
  - `attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md`
  - `attention_h1_dec_hybrid_pilot_report.md`
  - `attention_h1_vmf_dec_hybrid_pilot_report.md`
  - `attention_h1_decoupled_cluster_head_pilot_report.md`
  - `attention_h1_decoupled_cluster_head_detached_pilot_report.md`
  - `reconstruction_anchored_cluster_head_pilot_report.md`
  - `heldout_label_transfer_pilot_report.md`
  - `buddy_percept_matched_silhouette_audit_pilot_report.md`
  - `buddy_percept_downstream_probe_pilot_report.md`
  - `percept_on_buddy_embedding_pilot_report.md`
  - `dshared_capacity_sweep_pilot_report.md`
  - `soft_stage2_target_pilot_report.md`
  - `buddy_stage2_pilot_report.md`
  - `percept_stage1_faithful_recipe_pilot_report.md`
  - `percept_stage1_cluster_count_sweep_v2_pilot_report.md`
  - `learned_student_arch_sweep_pilot_report.md`
- **Details Checked:**
  - Every quoted mean, range, seed clearance count, silhouette score, AMI value, and AUC value matches the source pilot report Markdown files exactly to the quoted number of decimal places.
  - No transcription errors, fabricated numbers, or arithmetic calculation mistakes were found in the master report's citations of the pilot reports.
- **Conclusion:** **Checked all quantitative citations against raw pilot reports; found 100% numerical fidelity.** (The issue is which numbers were selected for comparison, not whether the numbers match the files).

---

#### Finding 13: Seeding and Independence of Multi-Seed Runs
- **Files Checked:** `run_attention_h1_baseline_seed_stress_pilot.py`, `run_percept_stage1_seed_stress_pilot.py`, `run_percept_stage1_extended_seed_pilot.py`.
- **Details Checked:**
  - Verified whether seed stress tests genuinely re-initialize random number generators and neural network weights.
  - In each script, `torch.manual_seed(seed)`, `np.random.seed(seed)`, and CUDA seeds are reset prior to constructing the model architecture, pretraining, and K-means.
  - Temporary arrays and states are not carried over between loop iterations.
- **Conclusion:** **Checked seeding logic; confirmed multi-seed stress runs are statistically independent.**

---

## Summary Matrix of Findings

| ID | Category | Target / Citation | Summary Description | Severity |
|---|---|---|---|---|
| **1** | Implementation Bug | `run_percept_stage1_pilot.py:257-262` | 180° Inversion of Center Pruning: kept 67 largest-norm centers (outliers) instead of discarding them per paper §4.3 | **Bug that fundamentally changes conclusions** |
| **2** | Implementation Bug | `run_percept_stage1_pilot.py:221-223` | Reconstruction loss divided by $D=2,816$ via `F.mse_loss`, attenuating reconstruction anchor by 2,816× and causing DEC collapse | **Bug that fundamentally changes conclusions** |
| **3** | Methodological Bug | `run_percept_stage1_pilot.py:112-125` | Pre-averaging captions destroyed image-caption pair structure and multi-label ground truth distribution | **Methodological distortion** |
| **4** | Implementation Bug | `run_percept_stage1_pilot.py:216-226` | Target distribution $P$ updated every single batch step instead of periodically, exacerbating self-training instability | **Implementation divergence** |
| **5** | Reporting / Fairness | Master Report §6a; `key_advantages.md` §1 | Stage 2 head-to-head cited untuned smoke-test 0.5690 to claim buddy "wins outright", suppressing PercepT's validated 0.8290 | **Severe reporting distortion** |
| **6** | Reporting / Fairness | Master Report §6; `key_advantages.md` §1 | "433× loss-scale distortion" in Candidate 2 was an artifact of $D=32$ vs $D=2816$ in `F.mse_loss`, not architectural fragility | **Flawed causal attribution** |
| **7** | Reporting / Fairness | `run_attention_h1_embedding_snapshot_pilot.py` | Asymmetric evaluation: buddy re-clustered test set transductively with Leiden; PercepT projected onto frozen centroids | **Fairness concern** |
| **8** | Reporting / Fairness | Master Report §2; `key_advantages.md` §1 | Conjunctive Pareto bar tailored to buddy; ignores that PercepT leads genre (0.3288 vs 0.2404) and ties on paper's harmonic mean (0.164) | **Evaluation bias** |
| **9** | Reporting / Fairness | `buddy_percept_matched_silhouette_audit_pilot.py` | Genre evaluation uses only $n=159$ held-out paintings across 67 clusters, creating high finite-sample estimator variance | **Statistical fragility** |
| **10** | Reporting / Fairness | Master Report §2; `key_advantages.md` §1 | Mischaracterized `LAMBDA_BALANCE` as an unprincipled hack; standard in clustering literature and only needed due to Bugs #1 & #2 | **Unfair characterization** |
| **11** | Integrity Check | All train / test split pipelines | Checked train/held-out split isolation; verified zero painting overlap ($n_{\text{train}}=61,402$, $n_{\text{heldout}}=9,365$) | **Checked; no issue found** |
| **12** | Integrity Check | Master report tables vs raw pilot reports | Checked all numerical values in Master Report & Key Advantages against raw pilot reports; 100% citation accuracy | **Checked; no issue found** |
| **13** | Integrity Check | Multi-seed pilot execution loops | Checked RNG seeding across multi-seed stress tests; confirmed fresh initialization and run independence | **Checked; no issue found** |

---

## Recommendations for the Project

1. **Fix the Pruning Logic:** Re-run PercepT's Stage 1 with the paper's actual pruning logic: discard centers with norms exceeding threshold $\tau$ (or retain the lowest-norm centers), rather than keeping the highest-norm centers.
2. **Correct the Reconstruction Loss Reduction:** In the DEC joint objective, compute reconstruction loss as sample-wise squared error (`reduction='none'`, summed over $D$, then mean over $N$), or scale `LAMBDA_RECONSTRUCTION` by $D = 2,816$ so that the reconstruction anchor operates at the $\mathcal{O}(0.2)$ scale intended by the paper.
3. **Retract the Stage 2 "Outright Win" Claim:** Remove the claim that buddy beats PercepT by $+0.0288$ in Stage 2 (0.5978 vs 0.5690). The comparison must either compare buddy to PercepT's validated multi-label Stage 2 model (0.8290) or explicitly document why the 0.5690 smoke test is not representative.
4. **Report Harmonic Mean Alongside the Conjunctive Bar:** Report the harmonic mean of emotion and genre AMI (PercepT's paper standard), which shows a balanced 0.164 vs 0.164 parity, providing a more balanced assessment of topic quality.
