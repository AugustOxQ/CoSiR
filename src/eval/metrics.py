"""Recall and ranking metric computation for CoSiR evaluation."""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm

from .config import EvaluationConfig


class RankingMetric:
    """Shared recall/ranking helpers used by RecallMetrics and OracleMetrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def calculate_recall_metrics(
        self, inds: torch.Tensor, mappings: torch.Tensor, captions_per_image: int
    ) -> Tuple[int, int, float]:
        """Return (mean_rank, median_rank, mAP×100) from sorted retrieval indices."""
        AP_scores, all_ranks = [], []

        for query_idx in range(inds.size(0)):
            correct = mappings[query_idx].tolist()
            query_inds = inds[query_idx]

            if isinstance(correct, int):
                ranks = [(query_inds == correct).nonzero(as_tuple=True)[-1].item() + 1]
            else:
                ranks = [(query_inds == c).nonzero(as_tuple=True)[-1].item() + 1 for c in correct]

            all_ranks.extend(ranks)
            AP = sum(j / r for j, r in enumerate(sorted(ranks), start=1)) / captions_per_image
            AP_scores.append(AP)

        return (
            int(np.mean(all_ranks)),
            int(np.median(all_ranks)),
            round(float(np.mean(AP_scores)) * 100, 1),
        )

    def _compute_recall_from_indices(
        self,
        inds_tti: torch.Tensor,
        inds_itt: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        num_texts: int,
        num_images: int,
        captions_per_image: int,
        prefix: str,
    ) -> Dict[str, float]:
        """Build recall@K/mAP/ranking dict from precomputed sorted indices."""
        inds_tti = inds_tti.to(self.config.device)
        inds_itt = inds_itt.to(self.config.device)

        t2i_recall = []
        for k in self.config.k_vals:
            correct = torch.eq(inds_tti[:, :k], text_to_image_map.unsqueeze(-1)).any(dim=1)
            t2i_recall.append(round(correct.sum().item() / num_texts * 100, 1))

        i2t_recall = []
        for k in self.config.k_vals:
            topk = inds_itt[:, :k]
            correct = torch.zeros(num_images, dtype=torch.bool, device=self.config.device)
            for i in range(captions_per_image):
                correct |= torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            i2t_recall.append(round(correct.sum().item() / num_images * 100, 1))

        meanR_t2i, medR_t2i, mAP_t2i = self.calculate_recall_metrics(inds_tti, text_to_image_map, 1)
        meanR_i2t, medR_i2t, mAP_i2t = self.calculate_recall_metrics(inds_itt, image_to_text_map, captions_per_image)

        del inds_tti, inds_itt
        torch.cuda.empty_cache()

        metrics: Dict[str, float] = {}
        for i, k in enumerate(self.config.k_vals):
            metrics[f"{prefix}/i2t_R{k}"] = i2t_recall[i]
            metrics[f"{prefix}/t2i_R{k}"] = t2i_recall[i]
        metrics.update({
            f"{prefix}/i2t_meanR": meanR_i2t,
            f"{prefix}/i2t_medR": medR_i2t,
            f"{prefix}/i2t_mAP": mAP_i2t,
            f"{prefix}/t2i_meanR": meanR_t2i,
            f"{prefix}/t2i_medR": medR_t2i,
            f"{prefix}/t2i_mAP": mAP_t2i,
        })
        return metrics


class RecallMetrics(RankingMetric):
    """I2T and T2I recall@K, mAP, and ranking for raw (unconditioned) retrieval."""

    def compute_all_recalls(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute both I2T and T2I recall from unnormalized embeddings."""
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        i2t_pfx = f"{prefix}/i2t" if prefix else "i2t"
        t2i_pfx = f"{prefix}/t2i" if prefix else "t2i"
        return {
            **self.compute_i2t_recall(image_embeddings, text_embeddings, image_to_text_map, i2t_pfx),
            **self.compute_t2i_recall(image_embeddings, text_embeddings, text_to_image_map, t2i_pfx),
        }

    def compute_i2t_recall(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "i2t",
    ) -> Dict[str, float]:
        """For each image, rank all captions and compute recall@K."""
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        dist = (text_embeddings @ image_embeddings.T).T  # [num_images, num_texts]
        if self.config.cpu_offload:
            dist = dist.cpu()
        inds = torch.argsort(dist, dim=1, descending=True).to(self.config.device)

        recall_scores = []
        for k in self.config.k_vals:
            topk = inds[:, :k]
            correct = torch.zeros(num_images, dtype=torch.bool, device=self.config.device)
            for i in range(captions_per_image):
                correct |= torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            recall_scores.append(round(correct.sum().item() / num_images * 100, 1))

        mean_rank, median_rank, mean_ap = self.calculate_recall_metrics(inds, image_to_text_map, captions_per_image)
        metrics = {f"{prefix}_R{k}": recall_scores[i] for i, k in enumerate(self.config.k_vals)}
        metrics.update({f"{prefix}_meanR": mean_rank, f"{prefix}_medR": median_rank, f"{prefix}_mAP": mean_ap})
        return metrics

    def compute_t2i_recall(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_to_image_map: torch.Tensor,
        prefix: str = "t2i",
    ) -> Dict[str, float]:
        """For each caption, rank all images and compute recall@K."""
        num_texts = text_embeddings.shape[0]

        dist = text_embeddings @ image_embeddings.T  # [num_texts, num_images]
        if self.config.cpu_offload:
            dist = dist.cpu()
        inds = torch.argsort(dist, dim=1, descending=True).to(self.config.device)

        recall_scores = []
        for k in self.config.k_vals:
            correct = torch.eq(inds[:, :k], text_to_image_map.unsqueeze(-1)).any(dim=1)
            recall_scores.append(round(correct.sum().item() / num_texts * 100, 1))

        mean_rank, median_rank, mean_ap = self.calculate_recall_metrics(inds, text_to_image_map, 1)
        metrics = {f"{prefix}_R{k}": recall_scores[i] for i, k in enumerate(self.config.k_vals)}
        metrics.update({f"{prefix}_meanR": mean_rank, f"{prefix}_medR": median_rank, f"{prefix}_mAP": mean_ap})
        return metrics


class OracleMetrics(RankingMetric):
    """Oracle and predictor-based evaluation over label conditions."""

    def compute_oracle_recall_average(
        self,
        model,
        label_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_full: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "oracle",
        aggregation: str = "max",
    ) -> Dict[str, float]:
        """Oracle upper bound: try all label conditions and keep the best per query.

        aggregation='max'  — best label per query (upper bound).
        aggregation='mean' — average similarity across all labels.
        """
        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]
        combine_side = getattr(model, "combine_side", "txt")

        image_embeddings = image_embeddings.to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)
        text_full = text_full.to(self.config.device)

        # Project the "other side" (the side not being combined)
        if combine_side == "txt":
            _img_proj = model.project_other(image_embeddings)
            image_norm = _img_proj / _img_proj.norm(dim=-1, keepdim=True)
            text_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        else:
            image_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            _txt_proj = model.project_other(text_embeddings)
            text_norm = _txt_proj / _txt_proj.norm(dim=-1, keepdim=True)

        combine_source = text_embeddings if combine_side == "txt" else image_embeddings
        n_to_combine = num_texts if combine_side == "txt" else num_images

        print(f"Evaluating oracle (combine_side={combine_side})...")

        running_max_tti: Optional[torch.Tensor] = None
        running_max_itt: Optional[torch.Tensor] = None

        # Welford online mean/variance accumulators (CPU, avoids stacking K full sim matrices)
        _wf_n = 0
        _wf_mean: Optional[torch.Tensor] = None
        _wf_M2: Optional[torch.Tensor] = None

        # Sampled columns for pairwise-diversity diagnostic
        _div_cols = torch.randperm(num_images)[: min(1000, num_images)]
        _div_samples: list = []

        with torch.no_grad():
            for label_id in tqdm(range(-1, len(label_embeddings)), desc="Evaluating oracle"):
                if label_id == -1:
                    combined = combine_source.detach().clone()
                else:
                    label_emb = label_embeddings[label_id].expand(n_to_combine, -1).to(self.config.device)
                    batches = []
                    for i in range(0, n_to_combine, self.config.batch_size):
                        end = min(i + self.config.batch_size, n_to_combine)
                        batches.append(model.combine(combine_source[i:end], None, label_emb[i:end]))
                    del label_emb
                    torch.cuda.empty_cache()
                    combined = torch.cat(batches, dim=0)

                combined = combined / combined.norm(dim=-1, keepdim=True)
                sims_cpu = (combined @ image_norm.T if combine_side == "txt" else text_norm @ combined.T).cpu()
                del combined
                torch.cuda.empty_cache()

                # Running max for oracle upper bound
                if running_max_tti is None:
                    running_max_tti = sims_cpu.clone()
                    running_max_itt = sims_cpu.T.clone()
                else:
                    running_max_tti = torch.maximum(running_max_tti, sims_cpu)
                    running_max_itt = torch.maximum(running_max_itt, sims_cpu.T)

                # Welford running mean/variance
                _wf_n += 1
                sims_f = sims_cpu.float()
                if _wf_mean is None:
                    _wf_mean = sims_f.clone()
                    _wf_M2 = torch.zeros_like(sims_f)
                else:
                    delta = sims_f - _wf_mean
                    _wf_mean.add_(delta / _wf_n)
                    _wf_M2.add_(delta * (sims_f - _wf_mean))

                _div_samples.append(sims_cpu[:, _div_cols].reshape(1, -1))
                del sims_cpu, sims_f

            # Diagnostics
            sim_variance = _wf_M2 / max(_wf_n - 1, 1)
            print(f"Sim variance — mean: {sim_variance.mean():.4f}, max: {sim_variance.max():.4f}")
            del sim_variance, _wf_M2

            flat = torch.cat(_div_samples, dim=0).float()
            flat_norm = flat / flat.norm(dim=-1, keepdim=True)
            pairwise_sim = flat_norm @ flat_norm.T
            mask = ~torch.eye(len(_div_samples), dtype=torch.bool)
            print(f"Pairwise label diversity — mean off-diag: {pairwise_sim[mask].mean():.4f}")
            del flat, flat_norm, pairwise_sim, mask, _div_samples

            if aggregation == "mean":
                dist_tti, dist_itt = _wf_mean, _wf_mean.T
            else:
                dist_tti, dist_itt = running_max_tti, running_max_itt
                del _wf_mean

            inds_tti = torch.argsort(dist_tti, dim=1, descending=True)
            inds_itt = torch.argsort(dist_itt, dim=1, descending=True)

            metrics = self._compute_recall_from_indices(
                inds_tti, inds_itt, text_to_image_map, image_to_text_map,
                num_texts, num_images, captions_per_image, prefix,
            )

            del image_norm, text_norm, image_embeddings, text_embeddings, text_full
            del inds_tti, inds_itt
            torch.cuda.empty_cache()

            return metrics

    def compute_non_oracle_recall_txt(
        self,
        model,
        label_embeddings: Optional[torch.Tensor],
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_full: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "txt_non_oracle",
    ) -> Dict[str, float]:
        """Non-oracle T2I/I2T: predict condition from text query via condition predictor."""
        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        image_norm = (image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)).to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)
        text_full = text_full.to(self.config.device)

        print("Evaluating with txt-only condition predictor...")

        with torch.no_grad():
            combined_batches = []
            for i in range(0, num_texts, self.config.batch_size):
                end = min(i + self.config.batch_size, num_texts)
                cond = model.predict_condition(text_embeddings[i:end])
                combined_batches.append(model.combine(text_embeddings[i:end], None, cond))
                del cond
                torch.cuda.empty_cache()
            combined = torch.cat(combined_batches, dim=0)

            dist_tti = combined @ image_norm.T
            if self.config.cpu_offload:
                dist_tti = dist_tti.cpu()
            inds_tti = torch.argsort(dist_tti, dim=1, descending=True)

            dist_itt = dist_tti.T
            if self.config.cpu_offload:
                dist_itt = dist_itt.cpu()
            inds_itt = torch.argsort(dist_itt, dim=1, descending=True)

            return self._compute_recall_from_indices(
                inds_tti, inds_itt, text_to_image_map, image_to_text_map,
                num_texts, num_images, captions_per_image, prefix,
            )

    def compute_non_oracle_recall_img(
        self,
        model,
        label_embeddings: Optional[torch.Tensor],
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_full: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "img_non_oracle",
    ) -> Dict[str, float]:
        """Non-oracle T2I/I2T: predict condition from image via condition predictor.

        Note: O(N_images × N_texts) — uses each image's predicted condition to
        score all text queries against that image.
        """
        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        image_norm = (image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)).to(self.config.device)
        image_embeddings = image_embeddings.to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)
        text_full = text_full.to(self.config.device)

        print("Evaluating with img-only condition predictor...")

        # Predict one condition per image (loop bound must be num_images, not num_texts)
        all_conditions = []
        for i in range(0, num_images, self.config.batch_size):
            end = min(i + self.config.batch_size, num_images)
            cond = model.predict_condition(image_embeddings[i:end])
            all_conditions.append(cond)
            del cond
            torch.cuda.empty_cache()
        all_conditions = torch.cat(all_conditions, dim=0)

        # Build [num_images, num_texts] sim matrix: for each image, combine all texts
        # with that image's condition and score against the image
        with torch.no_grad():
            sim_matrix = torch.zeros(num_images, num_texts)
            for img_idx in tqdm(range(num_images), desc="Computing per-image similarity"):
                cond_i = all_conditions[img_idx].unsqueeze(0)
                for i in range(0, num_texts, self.config.batch_size):
                    end = min(i + self.config.batch_size, num_texts)
                    combined = model.combine(text_embeddings[i:end], None, cond_i.expand(end - i, -1))
                    combined = combined / combined.norm(dim=-1, keepdim=True)
                    sim_matrix[img_idx, i:end] = image_norm[img_idx] @ combined.T
                    del combined
                    torch.cuda.empty_cache()

            dist_tti = sim_matrix.T
            if self.config.cpu_offload:
                dist_tti = dist_tti.cpu()
            inds_tti = torch.argsort(dist_tti, dim=1, descending=True)

            dist_itt = sim_matrix
            if self.config.cpu_offload:
                dist_itt = dist_itt.cpu()
            inds_itt = torch.argsort(dist_itt, dim=1, descending=True)

            return self._compute_recall_from_indices(
                inds_tti, inds_itt, text_to_image_map, image_to_text_map,
                num_texts, num_images, captions_per_image, prefix,
            )

    def compute_non_oracle_recall_imgtxt(
        self,
        model,
        label_embeddings: Optional[torch.Tensor],
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_full: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "both_non_oracle",
    ) -> Dict[str, float]:
        """Non-oracle T2I/I2T: predict condition from paired (image, text) via predictor.

        ConditionPredictor takes a single embedding, so we pass the combine-side
        (text) embedding; image context is implicitly encoded via the pairing structure.
        """
        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        image_norm = (image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)).to(self.config.device)
        image_embeddings = image_embeddings.to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)
        text_full = text_full.to(self.config.device)

        print("Evaluating with imgtxt condition predictor...")

        with torch.no_grad():
            combined_batches = []
            for i in range(0, num_images, self.config.batch_size):
                end = min(i + self.config.batch_size, num_images)
                # Each image is paired with its captions (contiguous in text_embeddings)
                txt_batch = text_embeddings[i * captions_per_image : end * captions_per_image]
                cond = model.predict_condition(txt_batch)
                combined_batches.append(model.combine(txt_batch, None, cond))
                del cond
                torch.cuda.empty_cache()
            combined = torch.cat(combined_batches, dim=0)

            dist_tti = combined @ image_norm.T
            if self.config.cpu_offload:
                dist_tti = dist_tti.cpu()
            inds_tti = torch.argsort(dist_tti, dim=1, descending=True)

            dist_itt = dist_tti.T
            if self.config.cpu_offload:
                dist_itt = dist_itt.cpu()
            inds_itt = torch.argsort(dist_itt, dim=1, descending=True)

            return self._compute_recall_from_indices(
                inds_tti, inds_itt, text_to_image_map, image_to_text_map,
                num_texts, num_images, captions_per_image, prefix,
            )

    def compute_predictor_recall(
        self,
        model,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "pre_original",
    ) -> Dict[str, float]:
        """Single-pass retrieval using the model's condition predictor (O(N), no search)."""
        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]
        combine_side = getattr(model, "combine_side", "txt")

        image_embeddings = image_embeddings.to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)

        if combine_side == "txt":
            image_norm = model.project_other(image_embeddings)
            image_norm = image_norm / image_norm.norm(dim=-1, keepdim=True)
            text_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        else:
            image_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            text_norm = model.project_other(text_embeddings)
            text_norm = text_norm / text_norm.norm(dim=-1, keepdim=True)

        print(f"Evaluating predictor recall (combine_side={combine_side})...")

        with torch.no_grad():
            source = text_embeddings if combine_side == "txt" else image_embeddings
            n = num_texts if combine_side == "txt" else num_images

            combined_batches = []
            for i in range(0, n, self.config.batch_size):
                end = min(i + self.config.batch_size, n)
                pred = model.predict_condition(source[i:end])
                combined_batches.append(model.combine(source[i:end], None, pred))
                del pred
                torch.cuda.empty_cache()
            combined = torch.cat(combined_batches, dim=0)
            combined = combined / combined.norm(dim=-1, keepdim=True)

            sims = combined @ image_norm.T if combine_side == "txt" else text_norm @ combined.T
            if self.config.cpu_offload:
                sims = sims.cpu()

            inds_tti = torch.argsort(sims, dim=1, descending=True)
            inds_itt = torch.argsort(sims.T, dim=1, descending=True)
            del sims, combined
            torch.cuda.empty_cache()

        return self._compute_recall_from_indices(
            inds_tti, inds_itt, text_to_image_map, image_to_text_map,
            num_texts, num_images, captions_per_image, prefix,
        )
