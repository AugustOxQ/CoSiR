"""Evaluation pipeline: embedding extraction, training and test evaluators."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EvaluationConfig, MetricResult, TestEvaluationDetail
from .metrics import RecallMetrics, OracleMetrics


# ---------------------------------------------------------------------------
# Label embedding utilities
# ---------------------------------------------------------------------------

def replace_with_most_different(label_embeddings: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Replace each embedding with one of its k most dissimilar neighbours."""
    batch_size = label_embeddings.size(0)
    k = max(1, min(k, batch_size - 1))

    normalized = F.normalize(label_embeddings, p=2, dim=-1)
    dist = 1 - torch.matmul(normalized, normalized.T)
    dist.fill_diagonal_(-float("inf"))  # exclude self

    topk_indices = torch.topk(dist, k=k, dim=1).indices  # [B, k]
    rand_k = torch.randint(0, k, (batch_size,), device=label_embeddings.device)
    selected = topk_indices[torch.arange(batch_size), rand_k]
    return label_embeddings[selected]


def sample_label_embeddings(label_embeddings: torch.Tensor) -> torch.Tensor:
    """Sample embeddings so no element stays at its own index."""
    batch_size = label_embeddings.size(0)
    unique = torch.unique(label_embeddings, dim=0)
    if unique.size(0) == 1:
        return label_embeddings

    out = torch.empty_like(label_embeddings)
    for i in range(batch_size):
        diff_mask = ~torch.all(unique == label_embeddings[i], dim=1)
        choices = unique[diff_mask]
        if choices.size(0) > 0:
            out[i] = choices[torch.randint(0, choices.size(0), (1,))]
        else:
            out[i] = label_embeddings[i]
    return out


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model, processor, dataloader: DataLoader, config: EvaluationConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
    """Extract image/text embeddings from a test dataloader.

    Returns (img_emb, txt_emb, txt_full, raw_text, text_to_image_map, image_to_text_map).
    """
    all_img, all_txt, all_txt_full, all_raw = [], [], [], []
    i2t_map: List[List[int]] = []
    t2i_map: List[int] = []
    text_idx = img_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            image, raw_text = batch
            image_input = image.to(config.device)
            B = image_input["pixel_values"].size(0)
            cpi = getattr(dataloader.dataset, "captions_per_image", 5)

            # Flatten captions: [cpi, B] list-of-lists → flat list length B*cpi
            raw_text_list = []
            for b in range(B):
                if cpi == 1:
                    raw_text_list.append(raw_text[b])
                else:
                    raw_text_list.extend(raw_text[i][b] for i in range(cpi))
            all_raw.extend(raw_text_list)

            tokenizer_kwargs: Dict[str, Any] = dict(
                text=raw_text_list, return_tensors="pt", padding="max_length", truncation=True
            )
            if config.max_text_length is not None:
                tokenizer_kwargs["max_length"] = config.max_text_length
            text_input = processor(**tokenizer_kwargs).to(config.device)

            for _ in range(B):
                i2t_map.append(list(range(text_idx, text_idx + cpi)))
                t2i_map.extend([img_idx] * cpi)
                text_idx += cpi
                img_idx += 1

            img_emb, txt_emb, _, txt_full = model.encode_img_txt(image_input, text_input)
            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())
            all_txt_full.append(txt_full.cpu())

            del img_emb, txt_emb, txt_full, image_input, text_input
            torch.cuda.empty_cache()

    return (
        torch.cat(all_img, dim=0),
        torch.cat(all_txt, dim=0),
        torch.cat(all_txt_full, dim=0),
        all_raw,
        torch.LongTensor(t2i_map).to(config.device),
        torch.LongTensor(i2t_map).to(config.device),
    )


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _format_results(
    config: EvaluationConfig, metrics: Dict[str, float], epoch: Optional[int] = None
) -> MetricResult:
    if epoch is not None:
        metrics = {**metrics, f"{config.metric_prefix}epoch": epoch}
    return MetricResult(metrics=metrics, epoch=epoch)


def _print_results(config: EvaluationConfig, results: MetricResult, title: str = "Evaluation Results"):
    if not config.print_metrics:
        return
    sep = "=" * 50
    print(f"\n{sep}\n{title}")
    if results.epoch is not None:
        print(f"Epoch: {results.epoch}")
    print(sep)
    for key, value in results.metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# TrainEvaluator
# ---------------------------------------------------------------------------

class TrainEvaluator:
    """Batch-wise rank evaluation on the training set."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        model,
        feature_manager,
        embedding_manager,
        dataloader: DataLoader,
        device=None,
        epoch: int = 0,
        max_batches: Optional[int] = None,
    ) -> MetricResult:
        """Evaluate combined vs raw retrieval rank over up to max_batches chunks."""
        device = device or self.config.device
        if max_batches is None:
            max_batches = self.config.train_max_batches

        model.eval()
        total_rank_raw = total_rank_comb = total_rank_shuffled = 0.0
        total_chunks = 0

        with torch.no_grad():
            for batch_id, _ in enumerate(tqdm(dataloader, desc="Training evaluation")):
                if batch_id >= max_batches:
                    print(f"Epoch {epoch}: stopping after {max_batches} batches")
                    break

                data = feature_manager.get_features_by_chunk(batch_id)
                img = data["img_features"].to(device, non_blocking=True)
                txt = data["txt_features"].to(device, non_blocking=True)
                txt_full = data["txt_full"].to(device, non_blocking=True) if "txt_full" in data else None
                img_full = data["img_full"].to(device, non_blocking=True) if "img_full" in data else None
                sample_ids = data["sample_ids"].tolist()

                combine_side = getattr(model, "combine_side", "txt")
                if combine_side == "txt":
                    combine_feat, combine_full = txt, txt_full
                    anchor = img
                else:
                    combine_feat, combine_full = img, img_full
                    anchor = txt

                label_emb = embedding_manager.get_embeddings(sample_ids).to(device)
                comb_emb = model.combine(combine_feat, combine_full, label_emb, epoch=epoch)
                label_emb_neg = replace_with_most_different(label_emb)
                comb_emb_neg = model.combine(combine_feat, combine_full, label_emb_neg, epoch=epoch)

                total_rank_raw += self._mean_rank(anchor, combine_feat)
                total_rank_comb += self._mean_rank(anchor, comb_emb)
                total_rank_shuffled += self._mean_rank(anchor, comb_emb_neg)
                total_chunks += 1

                del img, txt, txt_full, img_full, label_emb, comb_emb, comb_emb_neg, label_emb_neg
                torch.cuda.empty_cache()

        n = total_chunks
        if n == 0:
            raise RuntimeError("TrainEvaluator: no batches processed — check feature_manager path and max_batches config")
        mean_raw = total_rank_raw / n
        mean_comb = total_rank_comb / n
        mean_shuffled = total_rank_shuffled / n

        print(
            f"Epoch {epoch}: Mean Rank — Raw: {mean_raw:.2f}, "
            f"Combined: {mean_comb:.2f}, Shuffled: {mean_shuffled:.2f}"
        )

        metrics = {
            "val/mean_rank_raw": mean_raw,
            "val/mean_rank_comb": mean_comb,
            "val/mean_rank_comb_shuffled": mean_shuffled,
        }
        return _format_results(self.config, metrics, epoch)

    @staticmethod
    def _mean_rank(a: torch.Tensor, b: torch.Tensor) -> float:
        """Mean rank of diagonal elements in the [B, B] cosine similarity matrix."""
        a = F.normalize(a.float(), p=2, dim=-1).cpu()
        b = F.normalize(b.float(), p=2, dim=-1).cpu()
        sim = a @ b.T  # [B, B]
        inds = torch.argsort(sim, dim=1, descending=True)
        ranks = [(inds[i] == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(len(a))]
        return float(np.mean(ranks))


# ---------------------------------------------------------------------------
# TestEvaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Full test-set evaluation with oracle and predictor-based metrics."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.recall = RecallMetrics(self.config)
        self.oracle = OracleMetrics(self.config)
        # In-memory cache: keyed by id(dataloader), stores raw backbone embeddings.
        # Backbone is frozen so these never change across epochs.
        self._backbone_cache: Dict[int, Tuple] = {}

    def _get_or_extract_embeddings(
        self,
        model,
        processor,
        dataloader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
        """Return backbone embeddings, using in-memory then disk cache to avoid re-extraction."""
        cache_key = id(dataloader)
        if cache_key in self._backbone_cache:
            print("Reusing cached test embeddings (in-memory).")
            return self._backbone_cache[cache_key]

        disk_path: Optional[Path] = None
        if self.config.cache_dir:
            disk_path = Path(self.config.cache_dir) / "test_backbone_embeddings.pt"
            if disk_path.exists():
                print(f"Loading cached test embeddings from {disk_path} ...")
                saved = torch.load(disk_path, map_location="cpu", weights_only=False)
                result = (
                    saved["img_emb"], saved["txt_emb"], saved["txt_full"],
                    saved["raw_text"], saved["t2i_map"], saved["i2t_map"],
                )
                self._backbone_cache[cache_key] = result
                return result

        print("Extracting embeddings from test data...")
        result = extract_embeddings(model, processor, dataloader, self.config)

        if disk_path is not None:
            print(f"Saving test embeddings to {disk_path} ...")
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "img_emb": result[0], "txt_emb": result[1], "txt_full": result[2],
                "raw_text": result[3], "t2i_map": result[4], "i2t_map": result[5],
            }, disk_path)

        self._backbone_cache[cache_key] = result
        return result

    def evaluate(
        self,
        model,
        processor,
        dataloader: DataLoader,
        label_embeddings: Optional[torch.Tensor] = None,
        epoch: int = 0,
        device: Optional[str] = None,
        return_detailed_results: bool = False,
        use_oracle: bool = False,
        oracle_aggregation: str = "max",
    ) -> Union[MetricResult, TestEvaluationDetail]:
        """Run all test metrics: oracle/predictor retrieval + raw baseline."""
        img_emb, txt_emb, txt_full, raw_text, t2i_map, i2t_map = self._get_or_extract_embeddings(
            model, processor, dataloader
        )

        # --- Oracle or non-oracle conditioned retrieval ---
        if use_oracle:
            metrics_oracle = self.oracle.compute_oracle_recall_average(
                model, label_embeddings, img_emb, txt_emb, txt_full,
                t2i_map, i2t_map, "oracle", aggregation=oracle_aggregation,
            )
            oracle_groups = {"oracle": metrics_oracle}
        else:
            m_txt = self.oracle.compute_non_oracle_recall_txt(
                model, label_embeddings, img_emb, txt_emb, txt_full, t2i_map, i2t_map, "txt_non_oracle"
            )
            m_img = self.oracle.compute_non_oracle_recall_img(
                model, label_embeddings, img_emb, txt_emb, txt_full, t2i_map, i2t_map, "img_non_oracle"
            )
            m_imgtxt = self.oracle.compute_non_oracle_recall_imgtxt(
                model, label_embeddings, img_emb, txt_emb, txt_full, t2i_map, i2t_map, "both_non_oracle"
            )
            oracle_groups = {"oracle": m_txt, "oracle_img": m_img, "oracle_imgtxt": m_imgtxt}

        # --- Raw (unconditioned) baseline ---
        print("Running raw evaluation...")
        metrics_raw = self.recall.compute_all_recalls(img_emb, txt_emb, t2i_map, i2t_map, "raw")

        # --- Condition predictor (single forward pass, no representative search) ---
        metrics_pred: Dict[str, float] = {}
        if hasattr(model, "condition_predictor"):
            print("Running predictor evaluation...")
            metrics_pred = self.oracle.compute_predictor_recall(
                model, img_emb, txt_emb, t2i_map, i2t_map, "pre_original"
            )

        # --- Combine all metric groups under named wandb sections ---
        def _prefix(d: dict, group: str) -> dict:
            pfx = f"{group}/"
            return {
                f"test_{group}/{k[len(pfx):]}" if k.startswith(pfx) else f"test_{group}/{k}": v
                for k, v in d.items()
            }

        all_metrics: Dict[str, float] = {}
        for group, d in oracle_groups.items():
            all_metrics.update(_prefix(d, group))
        all_metrics.update(_prefix(metrics_raw, "raw"))
        all_metrics.update(_prefix(self._diff(oracle_groups.get("oracle", {}), metrics_raw, "raw", "diff"), "diff"))
        if metrics_pred:
            all_metrics.update(_prefix(metrics_pred, "pre_original"))
            all_metrics.update(_prefix(self._diff(metrics_pred, metrics_raw, "raw", "pre_diff"), "pre_diff"))

        results = _format_results(self.config, all_metrics, epoch)
        if self.config.print_metrics:
            _print_results(self.config, results)

        if not return_detailed_results:
            return results
        return TestEvaluationDetail(
            results=results,
            all_img_emb=img_emb,
            all_txt_emb=txt_emb,
            all_raw_text=raw_text,
            text_to_image_map=t2i_map,
            image_to_text_map=i2t_map,
        )

    def encode_data_only(
        self, model, processor, dataloader: DataLoader, device: Optional[str] = None
    ) -> Tuple:
        """Return embeddings + raw sorted indices without running metrics."""
        img_emb, txt_emb, txt_full, raw_text, t2i_map, i2t_map = self._get_or_extract_embeddings(
            model, processor, dataloader
        )
        img_norm = F.normalize(img_emb, p=2, dim=1)
        txt_norm = F.normalize(txt_emb, p=2, dim=1)
        dist = img_norm @ txt_norm.T
        inds_tti = torch.argsort(dist.T, dim=1, descending=True)
        inds_itt = torch.argsort(dist, dim=1, descending=True)
        return img_emb, txt_emb, txt_full, raw_text, t2i_map, i2t_map, inds_tti, inds_itt

    @staticmethod
    def _diff(m1: Dict, m2: Dict, prefix2: str, new_prefix: str) -> Dict[str, float]:
        """Compute m1[k] - m2[corresponding_k] for all matching keys."""
        out = {}
        for key in m1:
            if "/" not in key:
                continue
            name = key.split("/", 1)[1]
            ref_key = f"{prefix2}/{name}"
            if ref_key in m2:
                out[f"{new_prefix}/{name}"] = m1[key] - m2[ref_key]
        return out


# ---------------------------------------------------------------------------
# EvaluationManager — unified entry point
# ---------------------------------------------------------------------------

class EvaluationManager:
    """Single entry point for training and test evaluation."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.train_evaluator = TrainEvaluator(self.config)
        self.test_evaluator = TestEvaluator(self.config)

    def evaluate_train(self, model, feature_manager, embedding_manager, dataloader, device=None, epoch=0, max_batches=None) -> MetricResult:
        return self.train_evaluator.evaluate(model, feature_manager, embedding_manager, dataloader, device, epoch, max_batches)

    def evaluate_test(self, model, processor, dataloader, label_embeddings=None, epoch=0, device=None, return_detailed_results=False, use_oracle=False, oracle_aggregation="max") -> Union[MetricResult, TestEvaluationDetail]:
        return self.test_evaluator.evaluate(model, processor, dataloader, label_embeddings, epoch, device, return_detailed_results, use_oracle, oracle_aggregation)

    def encode_test_data(self, model, processor, dataloader, device=None) -> Tuple:
        return self.test_evaluator.encode_data_only(model, processor, dataloader, device)


# ---------------------------------------------------------------------------
# Backward-compatibility wrappers
# ---------------------------------------------------------------------------

def inference_train(model, dataloader, feature_manager, embedding_manager, device, epoch=0, k_vals=None, max_batches=25) -> Dict[str, Any]:
    config = EvaluationConfig(device=device, train_max_batches=max_batches, k_vals=k_vals or [1, 5, 10], print_metrics=True)
    result = EvaluationManager(config).evaluate_train(model, feature_manager, embedding_manager, dataloader, device, epoch, max_batches)
    return result.metrics


def inference_test(model, processor, dataloader, label_embeddings, epoch, device) -> Dict[str, Any]:
    config = EvaluationConfig(device=device, print_metrics=True)
    result = EvaluationManager(config).evaluate_test(model, processor, dataloader, label_embeddings, epoch, device)
    return result.metrics


def encode_data(model, processor, dataloader, device) -> Tuple:
    config = EvaluationConfig(device=device)
    return EvaluationManager(config).encode_test_data(model, processor, dataloader, device)
