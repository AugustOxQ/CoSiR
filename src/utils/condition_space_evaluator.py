import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
import json
from tqdm import tqdm


class CoSiRAutomaticEvaluator:
    """
    Automatic evaluation system for CoSiR.

    Args:
        model: CoSiR model
        precomputed: (image_embs, text_embs, captions_flat, img_to_cap_map)
            - image_embs: [N_images, D]
            - text_embs:  [N_texts,  D]
            - captions_flat: list of N_texts strings
            - img_to_cap_map: dict {img_idx: [cap_idx, ...]}
        conditions: [N, D] label embeddings from the training set
        device: computation device
        representatives: [N_reps, D] cluster representative conditions
        hdbscan_labels: [N] int array, -1 = noise
    """

    def __init__(
        self,
        model,
        precomputed,
        conditions,
        device,
        representatives=None,
        hdbscan_labels=None,
    ):
        self.model = model
        self.device = device
        self.image_embs, self.text_embs, self.captions_flat, self.img_to_cap_map = (
            precomputed
        )
        self.conditions = conditions.to(device)
        self.image_embs = self.image_embs.to(device)
        self.text_embs = self.text_embs.to(device)
        self.n_images = len(self.image_embs)
        self.n_texts = len(self.text_embs)

        self.representatives = (
            representatives.to(device)
            if isinstance(representatives, torch.Tensor)
            else None
        )
        self.hdbscan_labels = (
            np.array(hdbscan_labels)
            if hdbscan_labels is not None and not isinstance(hdbscan_labels, np.ndarray)
            else hdbscan_labels
        )

        # Side-aware setup: which embeddings get combined, which are the query side
        combine_side = getattr(model, "combine_side", "txt")
        if combine_side == "txt":
            self.combine_embs = self.text_embs    # embeddings fed to model.combine
            self.query_embs = self.image_embs     # embeddings used as retrieval queries
            self.n_combine = self.n_texts
            self.n_query = self.n_images
            # GT: for each query image, list of matching text indices
            self.query_to_gt: dict = {
                img_idx: self.img_to_cap_map[img_idx].tolist()
                for img_idx in range(self.n_images)
            }
        else:
            self.combine_embs = self.image_embs
            self.query_embs = self.text_embs
            self.n_combine = self.n_images
            self.n_query = self.n_texts
            # Derive text→image map from the image→text map
            _t2i: dict = {}
            for img_idx in range(self.n_images):
                for txt_idx in self.img_to_cap_map[img_idx].tolist():
                    _t2i[txt_idx] = img_idx
            # GT: for each query text, list containing its single matching image index
            self.query_to_gt = {
                txt_idx: [_t2i[txt_idx]]
                for txt_idx in range(self.n_texts)
            }

        print(f"Evaluator initialized with:")
        print(f"  - {self.n_images} images, {self.n_texts} texts")
        print(f"  - {len(self.conditions)} conditions")
        if self.representatives is not None:
            print(f"  - {len(self.representatives)} representatives")
        print(f"  - Device: {self.device}, combine_side: {combine_side}")

    # ========== 1. Magnitude–Effect Correlation ==========
    def compute_magnitude_effect_correlation(self, n_samples=200, n_texts_sample=100):
        """
        Pearson correlation between condition norm and modulation strength.

        Measures whether larger-magnitude conditions produce stronger shifts in
        text embedding space. Ideal: r > 0.8.
        """
        print("\n[1/5] Computing Magnitude–Effect Correlation...")
        self.model.eval()

        n_samples = min(n_samples, len(self.conditions))
        n_texts_sample = min(n_texts_sample, self.n_texts)

        if n_samples < 2:
            return {"correlation": 0.0, "p_value": 1.0}

        condition_indices = torch.randperm(len(self.conditions))[:n_samples]
        sampled_conditions = self.conditions[condition_indices]
        norms = torch.norm(sampled_conditions, dim=1).cpu().numpy()

        combine_indices = torch.randperm(self.n_combine)[:n_texts_sample]
        sampled_combine_embs = self.combine_embs[combine_indices]

        effects = []
        with torch.no_grad():
            for condition in tqdm(sampled_conditions, desc="  Processing"):
                cond_expanded = condition.unsqueeze(0).repeat(len(sampled_combine_embs), 1)
                combine_embs_mod = self.model.combine(
                    sampled_combine_embs, None, cond_expanded
                )
                delta = torch.norm(combine_embs_mod - sampled_combine_embs, dim=1).mean()
                effects.append(delta.cpu().item())

        effects = np.array(effects)

        if norms.std() < 1e-8 or effects.std() < 1e-8:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = pearsonr(norms, effects)

        print(f"  → Pearson r: {correlation:.4f} (p={p_value:.4e})")
        return {"correlation": float(correlation), "p_value": float(p_value)}

    # ========== 2. Condition Distance–Semantic Correlation ==========
    def compute_condition_distance_semantic_correlation(
        self, n_unique=50, n_texts_sample=100
    ):
        """
        Spearman correlation between pairwise condition distances and pairwise
        semantic distances of their modulated text spaces.

        Nearby conditions should produce similar retrieval behaviour.
        Works for any condition dimensionality (no polar-grid assumption).
        Ideal: ρ > 0.7.
        """
        print("\n[2/5] Computing Condition Distance–Semantic Correlation...")
        self.model.eval()

        n_unique = min(n_unique, len(self.conditions))
        n_texts_sample = min(n_texts_sample, self.n_texts)

        # Fixed sample for reproducibility across epochs
        n_combine_sample = min(n_texts_sample, self.n_combine)
        combine_indices = torch.arange(n_combine_sample, device=self.device)
        sampled_combine_embs = self.combine_embs[combine_indices]

        condition_indices = torch.randperm(len(self.conditions))[:n_unique]
        sampled_conditions = self.conditions[condition_indices]

        # Compute mean-pooled modulated embedding per condition → [n_unique, D]
        modulated_means = []
        with torch.no_grad():
            for condition in tqdm(sampled_conditions, desc="  Modulating"):
                cond_expanded = condition.unsqueeze(0).repeat(n_combine_sample, 1)
                mod = self.model.combine(sampled_combine_embs, None, cond_expanded)
                modulated_means.append(F.normalize(mod.mean(dim=0), dim=0))

        modulated_means = torch.stack(modulated_means)  # [n_unique, D]

        # Pairwise condition L2 distances
        cond_dists = torch.cdist(sampled_conditions, sampled_conditions)  # [n, n]

        # Pairwise cosine distances of modulated spaces
        sim_matrix = modulated_means @ modulated_means.T  # [n, n]
        semantic_dists = 1 - sim_matrix

        # Upper triangle only (exclude diagonal)
        idx = torch.triu_indices(n_unique, n_unique, offset=1)
        cond_dists_flat = cond_dists[idx[0], idx[1]].cpu().numpy()
        semantic_dists_flat = semantic_dists[idx[0], idx[1]].cpu().numpy()

        if len(cond_dists_flat) < 2:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = spearmanr(cond_dists_flat, semantic_dists_flat)

        print(f"  → Spearman ρ: {correlation:.4f} (p={p_value:.4e})")
        return {"spearman_rho": float(correlation), "p_value": float(p_value)}

    # ========== 3. Conditional Retrieval Gain ==========
    def compute_conditional_retrieval_gain(self, n_conditions=10, n_test_images=100):
        """
        R@K with random conditions vs no condition.

        Ideal: positive absolute gain at R@1/R@5/R@10.
        """
        print("\n[3/5] Computing Conditional Retrieval Gain...")
        self.model.eval()

        n_conditions = min(n_conditions, len(self.conditions))
        n_test_queries = min(n_test_images, self.n_query)

        condition_indices = torch.randperm(len(self.conditions))[:n_conditions]
        sampled_conditions = self.conditions[condition_indices]
        test_query_indices = torch.randperm(self.n_query)[:n_test_queries].tolist()

        all_ranks_baseline = []
        all_ranks_conditional = []

        with torch.no_grad():
            for query_idx in tqdm(test_query_indices, desc="  Testing queries"):
                query_emb = self.query_embs[query_idx : query_idx + 1]
                gt_indices = self.query_to_gt[query_idx]

                sims_base = F.cosine_similarity(
                    query_emb.expand(self.n_combine, -1), self.combine_embs, dim=1
                )
                best_rank_base = min(
                    (sims_base > sims_base[gt]).sum().item() for gt in gt_indices
                )
                all_ranks_baseline.append(best_rank_base)

                for condition in sampled_conditions:
                    cond_expanded = condition.unsqueeze(0).repeat(self.n_combine, 1)
                    combine_embs_mod = self.model.combine(
                        self.combine_embs, None, cond_expanded
                    )
                    sims = F.cosine_similarity(
                        query_emb.expand(self.n_combine, -1), combine_embs_mod, dim=1
                    )
                    best_rank = min(
                        (sims > sims[gt]).sum().item() for gt in gt_indices
                    )
                    all_ranks_conditional.append(best_rank)

        if not all_ranks_baseline or not all_ranks_conditional:
            return {}

        def recall_at_k(ranks, k):
            return float((np.array(ranks) < k).mean() * 100)

        result = {}
        for k in [1, 5, 10]:
            r_base = recall_at_k(all_ranks_baseline, k)
            r_cond = recall_at_k(all_ranks_conditional, k)
            result[f"R@{k}_baseline"] = r_base
            result[f"R@{k}_conditional"] = r_cond
            result[f"R@{k}_absolute_gain"] = r_cond - r_base

        result["mean_rank_baseline"] = float(np.mean(all_ranks_baseline))
        result["mean_rank_conditional"] = float(np.mean(all_ranks_conditional))
        result["mean_rank_improvement"] = float(
            result["mean_rank_baseline"] - result["mean_rank_conditional"]
        )

        print(
            f"  → R@1: {result['R@1_baseline']:.2f}% → {result['R@1_conditional']:.2f}%"
            f" (+{result['R@1_absolute_gain']:.2f}%)"
        )
        return result

    # ========== 4. Retrieval Diversity ==========
    def compute_retrieval_diversity(self, n_conditions=12, n_test_images=20, k=10):
        """
        Jensen–Shannon Divergence between retrieval distributions of different conditions.

        Samples conditions from the actual condition set (no polar-grid assumption).
        Ideal: mean JSD > 0.1.
        """
        print("\n[4/5] Computing Retrieval Diversity...")
        self.model.eval()

        n_conditions = min(max(2, n_conditions), len(self.conditions))
        n_test_queries = min(n_test_images, self.n_query)
        k = min(k, self.n_combine)

        condition_indices = torch.randperm(len(self.conditions))[:n_conditions]
        test_conditions = self.conditions[condition_indices]
        test_query_indices = torch.randperm(self.n_query)[:n_test_queries].tolist()

        distributions = [Counter() for _ in range(n_conditions)]

        with torch.no_grad():
            for query_idx in tqdm(test_query_indices, desc="  Testing queries"):
                query_emb = self.query_embs[query_idx : query_idx + 1]

                for cond_idx, condition in enumerate(test_conditions):
                    cond_expanded = condition.unsqueeze(0).repeat(self.n_combine, 1)
                    combine_embs_mod = self.model.combine(
                        self.combine_embs, None, cond_expanded
                    )
                    sims = F.cosine_similarity(
                        query_emb.expand(self.n_combine, -1), combine_embs_mod, dim=1
                    )
                    top_k_indices = torch.topk(sims, k=k)[1].cpu().tolist()
                    for idx in top_k_indices:
                        distributions[cond_idx][idx] += 1

        jsds = []
        for i in range(n_conditions):
            for j in range(i + 1, n_conditions):
                all_indices = sorted(
                    set(distributions[i].keys()) | set(distributions[j].keys())
                )
                if not all_indices:
                    continue
                di = np.array([distributions[i][x] for x in all_indices], dtype=float)
                dj = np.array([distributions[j][x] for x in all_indices], dtype=float)
                di /= di.sum() + 1e-8
                dj /= dj.sum() + 1e-8
                jsd = jensenshannon(di, dj)
                if not np.isnan(jsd):
                    jsds.append(jsd)

        result = {
            "mean_jsd": float(np.mean(jsds)) if jsds else 0.0,
            "std_jsd": float(np.std(jsds)) if jsds else 0.0,
        }
        print(f"  → Mean JSD: {result['mean_jsd']:.4f} ± {result['std_jsd']:.4f}")
        return result

    # ========== 5. Best-Condition Upper Bound ==========
    def compute_best_condition_upper_bound(self, n_test_images=20):
        """
        For fixed test images, try every representative condition and keep the best R@K.
        Compare to no-condition baseline to get retrieval boost upper bound.

        Uses a fixed set of images (first n_test_images) for epoch-to-epoch comparability.
        """
        print("\n[5/5] Computing Best-Condition Upper Bound...")
        self.model.eval()

        if self.representatives is None:
            print("  ⚠ No representatives provided, skipping.")
            return {}

        n_test_queries = min(n_test_images, self.n_query)
        test_query_indices = list(range(n_test_queries))  # fixed, not random

        all_ranks_baseline = []
        all_ranks_best = []

        with torch.no_grad():
            for query_idx in tqdm(test_query_indices, desc="  Testing queries"):
                query_emb = self.query_embs[query_idx : query_idx + 1]
                gt_indices = self.query_to_gt[query_idx]

                # Baseline
                sims_base = F.cosine_similarity(
                    query_emb.expand(self.n_combine, -1), self.combine_embs, dim=1
                )
                best_rank_base = min(
                    (sims_base > sims_base[gt]).sum().item() for gt in gt_indices
                )
                all_ranks_baseline.append(best_rank_base)

                # Best of representatives
                best_rank_rep = self.n_combine
                for rep in self.representatives:
                    cond_expanded = rep.unsqueeze(0).repeat(self.n_combine, 1)
                    combine_embs_mod = self.model.combine(
                        self.combine_embs, None, cond_expanded
                    )
                    sims = F.cosine_similarity(
                        query_emb.expand(self.n_combine, -1), combine_embs_mod, dim=1
                    )
                    rank = min((sims > sims[gt]).sum().item() for gt in gt_indices)
                    best_rank_rep = min(best_rank_rep, rank)
                all_ranks_best.append(best_rank_rep)

        def recall_at_k(ranks, k):
            return float((np.array(ranks) < k).mean() * 100)

        result = {}
        for k in [1, 5, 10]:
            r_base = recall_at_k(all_ranks_baseline, k)
            r_best = recall_at_k(all_ranks_best, k)
            result[f"R@{k}_baseline"] = r_base
            result[f"R@{k}_best_condition"] = r_best
            result[f"R@{k}_boost"] = r_best - r_base

        result["mean_rank_baseline"] = float(np.mean(all_ranks_baseline))
        result["mean_rank_best"] = float(np.mean(all_ranks_best))
        result["mean_rank_boost"] = float(
            result["mean_rank_baseline"] - result["mean_rank_best"]
        )

        print(
            f"  → R@1: {result['R@1_baseline']:.2f}% → {result['R@1_best_condition']:.2f}%"
            f" (boost +{result['R@1_boost']:.2f}%)"
        )
        return result

    # ========== 6. Condition Space Quality ==========
    def compute_condition_space_quality(self):
        """
        Intrinsic geometry of the condition space.
        - Silhouette score (uses pre-computed HDBSCAN labels, no extra clustering)
        - PCA effective dimensionality (95% variance)
        - Near-origin ratio
        """
        print("\n[6/6] Computing Condition Space Quality...")

        conditions_np = self.conditions.detach().cpu().numpy()

        # Silhouette: use HDBSCAN labels to avoid re-clustering
        silhouette = 0.0
        if self.hdbscan_labels is not None:
            valid = self.hdbscan_labels != -1
            unique_valid = np.unique(self.hdbscan_labels[valid])
            if valid.sum() >= 2 and len(unique_valid) >= 2:
                try:
                    silhouette = float(
                        silhouette_score(
                            conditions_np[valid], self.hdbscan_labels[valid]
                        )
                    )
                except Exception as e:
                    print(f"  ⚠ Silhouette failed: {e}")

        # PCA effective dimensions
        if len(conditions_np) > conditions_np.shape[1]:
            pca = PCA()
            pca.fit(conditions_np)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_effective_dims = int(np.argmax(cumsum >= 0.95) + 1)
            variance_ratios = pca.explained_variance_ratio_[:5].tolist()
        else:
            n_effective_dims = conditions_np.shape[1]
            variance_ratios = []

        # Near-origin ratio
        norms = np.linalg.norm(conditions_np, axis=1)
        near_origin_ratio = float((norms < 0.5).mean())

        result = {
            "silhouette_score": silhouette,
            "n_effective_dims": int(n_effective_dims),
            "pca_variance_ratios": variance_ratios,
            "near_origin_ratio": near_origin_ratio,
            "total_conditions": len(conditions_np),
        }

        print(f"  → Silhouette: {silhouette:.4f}")
        print(f"  → Effective dims: {n_effective_dims}/{conditions_np.shape[1]}")
        print(f"  → Near-origin: {near_origin_ratio*100:.1f}%")
        return result

    # ========== Main ==========
    def evaluate_all(self, save_path=None, verbose=True):
        if verbose:
            print("=" * 70)
            print(" " * 20 + "CoSiR Automatic Evaluation")
            print("=" * 70)

        results = {
            "metadata": {
                "n_images": self.n_images,
                "n_texts": self.n_texts,
                "n_conditions": len(self.conditions),
                "device": str(self.device),
            }
        }

        try:
            results["magnitude_effect"] = self.compute_magnitude_effect_correlation(
                n_samples=200, n_texts_sample=100
            )
            results["condition_distance_correlation"] = (
                self.compute_condition_distance_semantic_correlation(
                    n_unique=50, n_texts_sample=100
                )
            )
            results["retrieval_gain"] = self.compute_conditional_retrieval_gain(
                n_conditions=10, n_test_images=100
            )
            results["diversity"] = self.compute_retrieval_diversity(
                n_conditions=12, n_test_images=20, k=10
            )
            results["best_condition_upper_bound"] = (
                self.compute_best_condition_upper_bound(n_test_images=20)
            )
            # results["space_quality"] = self.compute_condition_space_quality()

        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            results["error"] = {"message": str(e), "traceback": traceback.format_exc()}

        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {save_path}")

        if verbose and "error" not in results:
            self._print_summary(results)

        return results

    def _print_summary(self, results):
        print("\n" + "=" * 70)
        print(" " * 25 + "EVALUATION SUMMARY")
        print("=" * 70)

        checks = []

        if "magnitude_effect" in results:
            checks.append(
                (
                    "Magnitude–Effect Correlation (r)",
                    results["magnitude_effect"]["correlation"],
                    0.8,
                    "≥",
                )
            )

        if "condition_distance_correlation" in results:
            checks.append(
                (
                    "Condition Distance–Semantic Corr (ρ)",
                    results["condition_distance_correlation"]["spearman_rho"],
                    0.7,
                    "≥",
                )
            )

        if "retrieval_gain" in results:
            checks.append(
                (
                    "R@1 Gain (random condition, %)",
                    results["retrieval_gain"]["R@1_absolute_gain"],
                    0.0,
                    "≥",
                )
            )

        if "diversity" in results:
            checks.append(
                (
                    "Retrieval Diversity (Mean JSD)",
                    results["diversity"]["mean_jsd"],
                    0.1,
                    "≥",
                )
            )

        if (
            "best_condition_upper_bound" in results
            and results["best_condition_upper_bound"]
        ):
            checks.append(
                (
                    "R@1 Boost — best condition (%)",
                    results["best_condition_upper_bound"]["R@1_boost"],
                    5.0,
                    "≥",
                )
            )

        if "space_quality" in results:
            checks.append(
                (
                    "Condition Space Silhouette",
                    results["space_quality"]["silhouette_score"],
                    0.3,
                    "≥",
                )
            )

        passed = 0
        for name, value, threshold, op in checks:
            ok = value >= threshold if op == "≥" else value <= threshold
            status = "✓ PASS" if ok else "✗ FAIL"
            passed += ok
            print(f"{name:45s}: {value:7.4f}  (thr: {op}{threshold:.2f})  {status}")

        print("=" * 70)
        score = (passed / len(checks)) * 100 if checks else 0
        print(f"Overall: {passed}/{len(checks)} passed ({score:.1f}%)")
        print("=" * 70)

        if "retrieval_gain" in results:
            rg = results["retrieval_gain"]
            print(f"\nRetrieval Gain (random conditions):")
            print(f"  Baseline  R@1: {rg['R@1_baseline']:.2f}%")
            print(
                f"  Cond.     R@1: {rg['R@1_conditional']:.2f}%  (+{rg['R@1_absolute_gain']:.2f}%)"
            )

        if (
            "best_condition_upper_bound" in results
            and results["best_condition_upper_bound"]
        ):
            ub = results["best_condition_upper_bound"]
            print(f"\nBest-Condition Upper Bound (first 20 test images):")
            print(f"  Baseline  R@1: {ub['R@1_baseline']:.2f}%")
            print(
                f"  Best-rep  R@1: {ub['R@1_best_condition']:.2f}%  (boost +{ub['R@1_boost']:.2f}%)"
            )
            print(
                f"  Mean rank: {ub['mean_rank_baseline']:.1f} → {ub['mean_rank_best']:.1f}"
            )
