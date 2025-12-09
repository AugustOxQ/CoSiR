import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr, kstest
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import json
from tqdm import tqdm


class CoSiRAutomaticEvaluator:
    """
    使用预计算embeddings的自动评估系统

    Args:
        model: CoSiR模型
        precomputed: (image_embs, text_embs, captions_flat, img_to_cap_map)
            - image_embs: [N_images, 512] torch.Tensor
            - text_embs: [N_texts, 512] torch.Tensor
            - captions_flat: list of N_texts个captions
            - img_to_cap_map: dict {img_idx: [cap_idx1, ..., cap_idx5]}
    """

    def __init__(self, model, precomputed, conditions, device):
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

        print(f"Evaluator initialized with:")
        print(f"  - {self.n_images} images")
        print(f"  - {self.n_texts} text embeddings")
        print(f"  - Device: {self.device}")

    # ========== 1. 半径-效果相关性 ==========
    def compute_radius_effect_correlation(self, n_samples=200, n_texts_sample=100):
        """
        测量半径和调制强度的Pearson相关系数

        理想值: r > 0.8
        """
        print("\n[1/6] Computing Radius-Effect Correlation...")
        self.model.eval()

        # Validate inputs
        n_samples = min(n_samples, len(self.conditions))
        n_texts_sample = min(n_texts_sample, self.n_texts)

        if n_samples < 2:
            print("  ⚠ Warning: Need at least 2 samples for correlation")
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "radii_mean": 0.0,
                "radii_std": 0.0,
                "effects_mean": 0.0,
                "effects_std": 0.0,
                "n_samples": n_samples,
            }

        # 随机采样conditions
        condition_indices = torch.randperm(len(self.conditions))[:n_samples]
        sampled_conditions = self.conditions[condition_indices]

        # 计算半径
        radii = torch.norm(sampled_conditions, dim=1).cpu().numpy()

        # 随机采样texts
        text_indices = torch.randperm(self.n_texts)[:n_texts_sample]
        sampled_text_embs = self.text_embs[text_indices]

        # 计算每个condition的平均调制强度
        effects = []

        with torch.no_grad():
            for condition in tqdm(sampled_conditions, desc="  Processing conditions"):
                # 用这个condition调制所有采样的texts
                cond_expanded = condition.unsqueeze(0).repeat(len(sampled_text_embs), 1)
                text_embs_mod = self.model.combine(
                    sampled_text_embs, None, cond_expanded
                )

                # 计算平均变化幅度
                delta = torch.norm(text_embs_mod - sampled_text_embs, dim=1).mean()
                effects.append(delta.cpu().item())

        effects = np.array(effects)

        # 计算Pearson相关系数
        # Check for zero variance to avoid division by zero
        if radii.std() < 1e-8 or effects.std() < 1e-8:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = pearsonr(radii, effects)

        result = {
            "correlation": float(correlation),
            "p_value": float(p_value),
            # "radii_mean": float(radii.mean()),
            # "radii_std": float(radii.std()),
            # "effects_mean": float(effects.mean()),
            # "effects_std": float(effects.std()),
            # "n_samples": n_samples,
        }

        print(f"  → Correlation: {correlation:.4f} (p={p_value:.4e})")
        return result

    # ========== 2. 角度-语义单调性 ==========
    def compute_angular_semantic_monotonicity(
        self, n_angles=12, n_texts_sample=150, test_radius=2.0
    ):
        """
        测量角度差异和语义差异的单调性

        理想值: Spearman ρ > 0.7
        """
        print("\n[2/6] Computing Angular-Semantic Monotonicity...")
        self.model.eval()

        # Validate inputs
        n_angles = max(2, n_angles)  # Need at least 2 angles
        n_texts_sample = min(n_texts_sample, self.n_texts)

        # 生成均匀分布的角度
        test_angles = torch.linspace(0, 2 * np.pi, n_angles + 1, device=self.device)[
            :-1
        ]

        # 生成test conditions (固定半径)
        test_conditions = torch.stack(
            [
                test_radius * torch.cos(test_angles),
                test_radius * torch.sin(test_angles),
            ],
            dim=1,
        )

        # 随机采样texts
        text_indices = torch.randperm(self.n_texts)[:n_texts_sample]
        sampled_text_embs = self.text_embs[text_indices]

        with torch.no_grad():
            # 对每个angle，调制texts
            all_modulated = []
            for condition in tqdm(test_conditions, desc="  Modulating texts"):
                cond_expanded = condition.unsqueeze(0).repeat(len(sampled_text_embs), 1)
                text_embs_mod = self.model.combine(
                    sampled_text_embs, None, cond_expanded
                )
                all_modulated.append(text_embs_mod)

            all_modulated = torch.stack(all_modulated)  # [n_angles, n_texts, 512]

            # 计算任意两个角度之间的语义距离
            angle_diffs = []
            semantic_dists = []

            for i in range(n_angles):
                for j in range(i + 1, n_angles):
                    # 角度差
                    angle_diff = torch.abs(test_angles[i] - test_angles[j])
                    angle_diff = min(angle_diff.item(), 2 * np.pi - angle_diff.item())
                    angle_diffs.append(angle_diff)

                    # 语义距离（平均余弦距离）
                    cos_sim = F.cosine_similarity(
                        all_modulated[i], all_modulated[j], dim=1
                    ).mean()
                    semantic_dist = 1 - cos_sim.item()
                    semantic_dists.append(semantic_dist)

        # 计算Spearman相关系数
        if len(angle_diffs) < 2:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = spearmanr(angle_diffs, semantic_dists)

        result = {
            "spearman_rho": float(correlation),
            "p_value": float(p_value),
            # "n_angles": n_angles,
            # "test_radius": test_radius,
            # "mean_angle_diff": float(np.mean(angle_diffs)) if angle_diffs else 0.0,
            # "mean_semantic_dist": (
            #     float(np.mean(semantic_dists)) if semantic_dists else 0.0
            # ),
        }

        print(f"  → Spearman ρ: {correlation:.4f} (p={p_value:.4e})")
        return result

    # ========== 3. 条件化检索提升 ==========
    def compute_conditional_retrieval_gain(self, n_conditions=10, n_test_images=100):
        """
        对比使用condition vs 不使用condition的检索性能

        指标: R@1, R@5, R@10的提升百分比
        """
        print("\n[3/6] Computing Conditional Retrieval Gain...")
        self.model.eval()

        # Validate inputs
        n_conditions = min(n_conditions, len(self.conditions))
        n_test_images = min(n_test_images, self.n_images)

        # 随机采样conditions
        condition_indices = torch.randperm(len(self.conditions))[:n_conditions]
        sampled_conditions = self.conditions[condition_indices]

        # 随机采样test images
        test_img_indices = torch.randperm(self.n_images)[:n_test_images].tolist()

        all_ranks_baseline = []
        all_ranks_conditional = []

        with torch.no_grad():
            for img_idx in tqdm(test_img_indices, desc="  Testing images"):
                img_emb = self.image_embs[img_idx : img_idx + 1]  # [1, 512]

                # Ground truth caption indices (MS-COCO: 5 captions per image)
                gt_cap_indices = self.img_to_cap_map[
                    img_idx
                ]  # list of 5 caption indices

                # ===== Baseline: 无condition =====
                sims_baseline = F.cosine_similarity(
                    img_emb.expand(self.n_texts, -1), self.text_embs, dim=1
                )  # [n_texts]

                # 找ground truth的最佳rank
                gt_ranks_baseline = []
                for gt_idx in gt_cap_indices:
                    rank = (sims_baseline > sims_baseline[gt_idx]).sum().item()
                    gt_ranks_baseline.append(rank)

                if gt_ranks_baseline:
                    best_rank_baseline = min(gt_ranks_baseline)
                    all_ranks_baseline.append(best_rank_baseline)
                else:
                    continue

                # ===== Conditional: 对每个condition =====
                for condition in sampled_conditions:
                    cond_expanded = condition.unsqueeze(0).repeat(self.n_texts, 1)
                    text_embs_mod = self.model.combine(
                        self.text_embs, None, cond_expanded
                    )

                    sims_cond = F.cosine_similarity(
                        img_emb.expand(self.n_texts, -1), text_embs_mod, dim=1
                    )

                    # 找ground truth的最佳rank
                    gt_ranks_cond = []
                    for gt_idx in gt_cap_indices:
                        rank = (sims_cond > sims_cond[gt_idx]).sum().item()
                        gt_ranks_cond.append(rank)

                    if gt_ranks_cond:
                        best_rank_cond = min(gt_ranks_cond)
                        all_ranks_conditional.append(best_rank_cond)

        # Check if we have any valid ranks
        if not all_ranks_baseline or not all_ranks_conditional:
            print("  ⚠ Warning: No valid retrieval results found")
            return {
                "R@1_baseline": 0.0,
                "R@1_conditional": 0.0,
                "R@1_absolute_gain": 0.0,
                "R@1_relative_gain_%": 0.0,
                "R@5_baseline": 0.0,
                "R@5_conditional": 0.0,
                "R@5_absolute_gain": 0.0,
                "R@5_relative_gain_%": 0.0,
                "R@10_baseline": 0.0,
                "R@10_conditional": 0.0,
                "R@10_absolute_gain": 0.0,
                "R@10_relative_gain_%": 0.0,
                "mean_rank_baseline": 0.0,
                "mean_rank_conditional": 0.0,
                "mean_rank_improvement": 0.0,
            }

        # 计算R@K
        def recall_at_k(ranks, k):
            if not ranks:
                return 0.0
            return (np.array(ranks) < k).mean() * 100  # 转为百分比

        result = {}
        for k in [1, 5, 10]:
            r_baseline = recall_at_k(all_ranks_baseline, k)
            r_conditional = recall_at_k(all_ranks_conditional, k)

            # 绝对提升
            absolute_gain = r_conditional - r_baseline
            # 相对提升
            relative_gain = (absolute_gain / (r_baseline + 1e-8)) * 100

            result[f"R@{k}_baseline"] = float(r_baseline)
            result[f"R@{k}_conditional"] = float(r_conditional)
            result[f"R@{k}_absolute_gain"] = float(absolute_gain)
            result[f"R@{k}_relative_gain_%"] = float(relative_gain)

        # Mean Rank
        result["mean_rank_baseline"] = float(np.mean(all_ranks_baseline))
        result["mean_rank_conditional"] = float(np.mean(all_ranks_conditional))
        result["mean_rank_improvement"] = float(
            result["mean_rank_baseline"] - result["mean_rank_conditional"]
        )

        print(
            f"  → R@1: {result['R@1_baseline']:.2f}% → {result['R@1_conditional']:.2f}% "
            f"(+{result['R@1_absolute_gain']:.2f}%)"
        )
        print(
            f"  → R@5: {result['R@5_baseline']:.2f}% → {result['R@5_conditional']:.2f}% "
            f"(+{result['R@5_absolute_gain']:.2f}%)"
        )

        return result

    # ========== 4. 检索多样性 ==========
    def compute_retrieval_diversity(
        self, n_conditions=12, n_test_images=20, k=10, test_radius=2.0
    ):
        """
        测量不同conditions检索结果的多样性

        使用Jensen-Shannon Divergence (JSD)
        理想值: JSD > 0.1
        """
        print("\n[4/6] Computing Retrieval Diversity...")
        self.model.eval()

        # Validate inputs
        n_conditions = max(2, n_conditions)  # Need at least 2 for diversity
        n_test_images = min(n_test_images, self.n_images)
        k = min(k, self.n_texts)

        # 在不同角度均匀采样conditions
        angles = torch.linspace(0, 2 * np.pi, n_conditions + 1, device=self.device)[:-1]
        test_conditions = torch.stack(
            [test_radius * torch.cos(angles), test_radius * torch.sin(angles)], dim=1
        )

        # 随机选择test images
        test_img_indices = torch.randperm(self.n_images)[:n_test_images].tolist()

        # 对每个condition，收集检索到的caption indices的分布
        condition_caption_distributions = [Counter() for _ in range(n_conditions)]

        with torch.no_grad():
            for img_idx in tqdm(test_img_indices, desc="  Testing images"):
                img_emb = self.image_embs[img_idx : img_idx + 1]

                # 对每个condition检索
                for cond_idx, condition in enumerate(test_conditions):
                    cond_expanded = condition.unsqueeze(0).repeat(self.n_texts, 1)
                    text_embs_mod = self.model.combine(
                        self.text_embs, None, cond_expanded
                    )

                    sims = F.cosine_similarity(
                        img_emb.expand(self.n_texts, -1), text_embs_mod, dim=1
                    )

                    # Top-k
                    top_k_indices = torch.topk(sims, k=min(k, self.n_texts))[1]

                    # 收集top-k caption indices
                    for idx in top_k_indices.cpu().tolist():
                        condition_caption_distributions[cond_idx][idx] += 1

        # 计算两两之间的JSD
        jsds = []
        for i in range(n_conditions):
            for j in range(i + 1, n_conditions):
                # 构建共同的vocabulary (caption indices)
                all_indices = set(condition_caption_distributions[i].keys()) | set(
                    condition_caption_distributions[j].keys()
                )

                if len(all_indices) == 0:
                    continue

                # 构建分布向量
                dist_i = np.array(
                    [
                        condition_caption_distributions[i][idx]
                        for idx in sorted(all_indices)
                    ]
                )
                dist_j = np.array(
                    [
                        condition_caption_distributions[j][idx]
                        for idx in sorted(all_indices)
                    ]
                )

                # 归一化
                dist_i = dist_i / (dist_i.sum() + 1e-8)
                dist_j = dist_j / (dist_j.sum() + 1e-8)

                # 计算JSD
                jsd = jensenshannon(dist_i, dist_j)
                if not np.isnan(jsd):
                    jsds.append(jsd)

        result = {
            "mean_jsd": float(np.mean(jsds)) if jsds else 0.0,
            "std_jsd": float(np.std(jsds)) if jsds else 0.0,
            # "min_jsd": float(np.min(jsds)) if jsds else 0.0,
            # "max_jsd": float(np.max(jsds)) if jsds else 0.0,
            # "n_conditions": n_conditions,
            # "n_pairs": len(jsds),
        }

        print(f"  → Mean JSD: {result['mean_jsd']:.4f} ± {result['std_jsd']:.4f}")
        return result

    # ========== 5. 语义一致性 ==========
    def compute_semantic_coherence_score(self, n_bins=8, n_images_per_bin=5, k=10):
        """
        评估同一角度区间的语义一致性

        使用简化版本：计算同一bin内检索结果的相似度
        """
        print("\n[5/6] Computing Semantic Coherence...")
        self.model.eval()

        # Validate inputs
        n_images_per_bin = min(n_images_per_bin, self.n_images)
        k = min(k, self.n_texts)

        # 分angle bins
        angles = torch.atan2(self.conditions[:, 1], self.conditions[:, 0])
        angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1, device=self.device)

        bin_coherences = []

        with torch.no_grad():
            for i in range(n_bins):
                bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])

                if not bin_mask.any():
                    continue

                bin_conditions = self.conditions[bin_mask]

                # 从这个bin随机采样几个conditions
                n_sample = min(3, len(bin_conditions))
                sampled_conditions = bin_conditions[
                    torch.randperm(len(bin_conditions))[:n_sample]
                ]

                # 随机选择test images
                test_img_indices = torch.randperm(self.n_images)[
                    :n_images_per_bin
                ].tolist()

                # 收集这个bin所有检索到的text embeddings
                bin_retrieved_embs = []

                for img_idx in test_img_indices:
                    img_emb = self.image_embs[img_idx : img_idx + 1]

                    for condition in sampled_conditions:
                        cond_expanded = condition.unsqueeze(0).repeat(self.n_texts, 1)
                        text_embs_mod = self.model.combine(
                            self.text_embs, None, cond_expanded
                        )

                        sims = F.cosine_similarity(
                            img_emb.expand(self.n_texts, -1), text_embs_mod, dim=1
                        )

                        top_k_indices = torch.topk(sims, k=min(k, self.n_texts))[1]

                        # 收集这些text的原始embeddings（未调制）
                        bin_retrieved_embs.append(self.text_embs[top_k_indices])

                if len(bin_retrieved_embs) > 1:
                    # 合并所有retrieved embeddings
                    bin_retrieved_embs = torch.cat(
                        bin_retrieved_embs, dim=0
                    )  # [N, 512]

                    # 计算平均pairwise cosine similarity
                    # 归一化
                    bin_retrieved_embs = F.normalize(bin_retrieved_embs, dim=1)

                    # 相似度矩阵
                    sim_matrix = bin_retrieved_embs @ bin_retrieved_embs.T

                    # 去除对角线
                    n = len(sim_matrix)
                    mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                    mask.fill_diagonal_(False)

                    coherence = sim_matrix[mask].mean().item()
                    bin_coherences.append(coherence)

        result = {
            "mean_coherence": float(np.mean(bin_coherences)) if bin_coherences else 0.0,
            "std_coherence": float(np.std(bin_coherences)) if bin_coherences else 0.0,
            # "n_bins_evaluated": len(bin_coherences),
            # "coherences_per_bin": [float(c) for c in bin_coherences],
        }

        print(
            f"  → Mean Coherence: {result['mean_coherence']:.4f} ± {result['std_coherence']:.4f}"
        )
        return result

    # ========== 6. Condition Space质量 ==========
    def compute_condition_space_quality(self):
        """
        评估condition space本身的质量

        包括：
        - Silhouette score (聚类质量)
        - 有效维度数
        - 分布均匀性
        """
        print("\n[6/6] Computing Condition Space Quality...")

        conditions = self.conditions.detach().cpu().numpy()

        # 1. Silhouette score
        # Ensure we have enough samples (at least 2*n_clusters)
        n_clusters = min(8, max(2, len(conditions) // 20))  # More conservative
        if n_clusters >= 2 and len(conditions) >= 2 * n_clusters:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(conditions)
                # Check if we have at least 2 different labels
                if len(np.unique(labels)) >= 2:
                    silhouette = silhouette_score(conditions, labels)
                else:
                    silhouette = 0.0
            except Exception as e:
                print(f"  ⚠ Warning: Silhouette score computation failed: {e}")
                silhouette = 0.0
        else:
            silhouette = 0.0
            n_clusters = 0

        # 2. 有效维度（通过PCA）
        if len(conditions) > conditions.shape[1]:
            pca = PCA()
            pca.fit(conditions)
            # 保留95%方差需要的维度数
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_effective_dims = int(np.argmax(cumsum >= 0.95) + 1)
            variance_ratios = pca.explained_variance_ratio_.tolist()
        else:
            n_effective_dims = conditions.shape[1]
            variance_ratios = []

        # 3. 极坐标分布统计（2D特有）
        if conditions.shape[1] == 2:
            angles = np.arctan2(conditions[:, 1], conditions[:, 0])
            radii = np.linalg.norm(conditions, axis=1)

            # 角度均匀性（Kolmogorov-Smirnov test）
            # 转换到[0, 1]
            angles_normalized = (angles + np.pi) / (2 * np.pi)
            ks_stat, ks_pval = kstest(angles_normalized, "uniform")

            # 半径分布统计
            radius_stats = {
                "mean": float(radii.mean()),
                "std": float(radii.std()),
                "min": float(radii.min()),
                "max": float(radii.max()),
                "median": float(np.median(radii)),
            }

            # 条件在原点附近的比例
            near_origin_ratio = (radii < 0.5).mean()
        else:
            ks_stat, ks_pval = None, None
            radius_stats = None
            near_origin_ratio = None

        result = {
            "silhouette_score": float(silhouette),
            "n_clusters": n_clusters,
            "n_effective_dims": int(n_effective_dims),
            "pca_variance_ratios": variance_ratios[:5],  # 前5个
            "angle_uniformity_ks_stat": float(ks_stat) if ks_stat is not None else None,
            "angle_uniformity_p_value": float(ks_pval) if ks_pval is not None else None,
            "radius_stats": radius_stats,
            "near_origin_ratio": (
                float(near_origin_ratio) if near_origin_ratio is not None else None
            ),
            "total_conditions": len(conditions),
        }

        print(f"  → Silhouette: {silhouette:.4f}")
        print(f"  → Effective Dims: {n_effective_dims}/{conditions.shape[1]}")
        if radius_stats:
            print(f"  → Radius: {radius_stats['mean']:.2f} ± {radius_stats['std']:.2f}")

        return result

    # ========== 主评估函数 ==========
    def evaluate_all(self, save_path=None, verbose=True):
        """
        运行所有自动化评估指标
        """
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
            # 1. 半径-效果相关性
            results["radius_effect"] = self.compute_radius_effect_correlation(
                n_samples=200, n_texts_sample=100
            )

            # 2. 角度-语义单调性
            results["angular_monotonicity"] = (
                self.compute_angular_semantic_monotonicity(
                    n_angles=12, n_texts_sample=150, test_radius=2.0
                )
            )

            # 3. 检索性能提升
            results["retrieval_gain"] = self.compute_conditional_retrieval_gain(
                n_conditions=10, n_test_images=100
            )

            # 4. 检索多样性
            results["diversity"] = self.compute_retrieval_diversity(
                n_conditions=12, n_test_images=20, k=10
            )

            # 5. 语义一致性
            results["coherence"] = self.compute_semantic_coherence_score(
                n_bins=8, n_images_per_bin=5, k=10
            )

            # # 6. Condition space质量
            # results["space_quality"] = (
            #     self.compute_condition_space_quality()
            # )  # DEBUG: This is too slow, we should use a faster method

        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            results["error"] = str(e)

        # 保存结果
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {save_path}")

        # 生成summary
        if verbose and "error" not in results:
            self.print_summary(results)

        return results

    def print_summary(self, results):
        """
        打印评估摘要
        """
        print("\n" + "=" * 70)
        print(" " * 25 + "EVALUATION SUMMARY")
        print("=" * 70)

        # 定义阈值
        thresholds = {
            "radius_correlation": (0.7, "≥"),
            "angular_monotonicity": (0.6, "≥"),
            "r1_absolute_gain": (5.0, "≥"),
            "diversity_jsd": (0.1, "≥"),
            "coherence": (0.5, "≥"),
            "silhouette": (0.3, "≥"),
        }

        # 检查每个指标
        checks = []

        if "radius_effect" in results:
            checks.append(
                (
                    "Radius-Effect Correlation",
                    results["radius_effect"]["correlation"],
                    thresholds["radius_correlation"],
                )
            )

        if "angular_monotonicity" in results:
            checks.append(
                (
                    "Angular-Semantic Monotonicity (Spearman ρ)",
                    results["angular_monotonicity"]["spearman_rho"],
                    thresholds["angular_monotonicity"],
                )
            )

        if "retrieval_gain" in results:
            checks.append(
                (
                    "R@1 Absolute Gain (%)",
                    results["retrieval_gain"]["R@1_absolute_gain"],
                    thresholds["r1_absolute_gain"],
                )
            )

        if "diversity" in results:
            checks.append(
                (
                    "Retrieval Diversity (Mean JSD)",
                    results["diversity"]["mean_jsd"],
                    thresholds["diversity_jsd"],
                )
            )

        if "coherence" in results:
            checks.append(
                (
                    "Semantic Coherence",
                    results["coherence"]["mean_coherence"],
                    thresholds["coherence"],
                )
            )

        if "space_quality" in results:
            checks.append(
                (
                    "Silhouette Score",
                    results["space_quality"]["silhouette_score"],
                    thresholds["silhouette"],
                )
            )

        # 打印每个检查
        passed = 0
        for name, value, (threshold, op) in checks:
            if op == "≥":
                status = "✓ PASS" if value >= threshold else "✗ FAIL"
                passed += value >= threshold
            else:
                status = "✓ PASS" if value <= threshold else "✗ FAIL"
                passed += value <= threshold

            print(f"{name:45s}: {value:7.4f}  (thr: {op}{threshold:5.2f})  {status}")

        print("=" * 70)
        score = (passed / len(checks)) * 100 if checks else 0
        print(f"Overall Score: {passed}/{len(checks)} checks passed ({score:.1f}%)")
        print("=" * 70)

        # 额外的关键信息
        if "retrieval_gain" in results:
            print(f"\nRetrieval Performance:")
            print(
                f"  Baseline R@1:     {results['retrieval_gain']['R@1_baseline']:.2f}%"
            )
            print(
                f"  Conditional R@1:  {results['retrieval_gain']['R@1_conditional']:.2f}%"
            )
            print(
                f"  Improvement:      +{results['retrieval_gain']['R@1_absolute_gain']:.2f}%"
            )

        if "space_quality" in results and results["space_quality"].get("radius_stats"):
            rs = results["space_quality"]["radius_stats"]
            print(f"\nCondition Space Statistics:")
            print(
                f"  Radius: {rs['mean']:.2f} ± {rs['std']:.2f} (range: [{rs['min']:.2f}, {rs['max']:.2f}])"
            )
            print(
                f"  Near origin: {results['space_quality']['near_origin_ratio']*100:.1f}%"
            )
