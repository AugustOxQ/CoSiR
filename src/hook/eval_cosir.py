import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoProcessor
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

from src.dataset import (
    CoSiRValidationDataset,
)
from src.model import CoSiRModel
from src.eval import EvaluationManager, EvaluationConfig, MetricResult
from src.utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    get_representatives,
)


class CoSiREvaluator:
    """Evaluation and analysis interface for trained CoSiR models"""

    def __init__(self, experiment_path: str, device: Optional[str] = None):
        """
        Initialize evaluator with trained experiment

        Args:
            experiment_path: Path to experiment directory or experiment name
            device: Device to use for evaluation
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load experiment
        if Path(experiment_path).exists():
            # Direct path to experiment directory
            self.experiment_dir = Path(experiment_path)
            self.experiment_name = self.experiment_dir.name
        else:
            # Experiment name - search in experiments directory
            exp_manager = ExperimentManager()
            experiment = exp_manager.load_experiment(experiment_path)
            self.experiment_dir = experiment.directory
            self.experiment_name = experiment.name

        # Load experiment config
        self.config = self._load_config()

        # Initialize components
        self.model = None
        self.feature_manager = None
        self.embedding_manager = None
        self.processor = None

        print(f"Loaded experiment: {self.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration"""
        import ast

        # Try config.json first (if manually saved)
        config_path = self.experiment_dir / "configs/config.json"
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            with open(config_path) as f:
                config = json.load(f)
                # If config is a string (stored incorrectly), parse it
                if isinstance(config, str):
                    config = ast.literal_eval(config)
                return config

        # Try experiment_metadata.json (saved by ExperimentManager)
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        if metadata_path.exists():
            print(f"Loading metadataconfig from: {metadata_path}")
            with open(metadata_path) as f:
                metadata = json.load(f)
                config = metadata.get("config", {})
                # If config is a string (stored incorrectly), parse it
                if isinstance(config, str):
                    config = ast.literal_eval(config)
                return config

        raise FileNotFoundError(f"Config not found in {config_path} or {metadata_path}")

    def load_model(self) -> CoSiRModel:
        """Load the trained model"""
        # Initialize CLIP processor
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"]["clip_model"]
        )

        # Initialize model
        self.model = CoSiRModel(
            label_dim=self.config["model"]["embedding_dim"],
        )

        # Load trained combiner weights
        artifacts_path = self.experiment_dir / "checkpoints" / "final_model.pt"
        if artifacts_path.exists():
            combiner_state = torch.load(artifacts_path, map_location=self.device)
            self.model.combiner.load_state_dict(combiner_state)
            print(f"Loaded model weights from: {artifacts_path}")
        else:
            print("Warning: No trained model weights found, using initialized weights")

        self.model.to(self.device)
        self.model.eval()

        return self.model

    def load_feature_manager(self) -> FeatureManager:
        """Load feature manager with cached features"""
        feature_config = {
            "storage_dir": self.config["featuremanager"]["storage_dir"],
            "sample_ids_path": self.config["featuremanager"]["sample_ids_path"],
            "primary_backend": self.config["featuremanager"]["primary_backend"],
            "chunked_storage": {
                "enabled": self.config["featuremanager"]["chunked_storage"]["enabled"],
                "chunk_size": self.config["featuremanager"]["chunk_size"],
                "compression": self.config["featuremanager"]["chunked_storage"][
                    "compression"
                ],
            },
            "cache": {
                "l1_size_mb": self.config["featuremanager"]["cache"]["l1_size_mb"],
                "l2_size_mb": self.config["featuremanager"]["cache"]["l2_size_mb"],
                "l3_path": self.config["featuremanager"]["cache"]["l3_path"],
            },
        }

        self.feature_manager = FeatureManager(
            features_dir=feature_config["storage_dir"],
            config=feature_config,
        )
        print(f"Loaded feature manager")

        return self.feature_manager

    def load_embedding_manager(self) -> TrainableEmbeddingManager:
        """Load embedding manager with trained embeddings"""
        if self.feature_manager is None:
            raise RuntimeError("Feature manager must be loaded first")

        # Load sample IDs
        sample_ids_path = self.config["featuremanager"]["sample_ids_path"]
        sample_ids_list = torch.load(sample_ids_path, map_location="cpu")

        # Initialize embedding manager
        self.embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids_list,
            embedding_dim=self.config["model"]["embedding_dim"],
            storage_mode=self.config["embeddingmanager"]["storage_mode"],
            device=self.device,
            initialization_strategy="random",  # Will be overridden by loading
            embeddings_dir=str(self.experiment_dir / "final_embeddings"),
            # Cache settings
            cache_l1_size_mb=self.config["embeddingmanager"]["cache_l1_size_mb"],
            cache_l2_size_mb=self.config["embeddingmanager"]["cache_l2_size_mb"],
            enable_l3_cache=self.config["embeddingmanager"]["enable_l3_cache"],
            auto_sync=False,  # Not needed for evaluation
        )

        # Load final embeddings
        final_embeddings_dir = self.experiment_dir / "final_embeddings"
        if final_embeddings_dir.exists():
            print(f"Loading final embeddings from: {final_embeddings_dir}")
            # The embedding manager will automatically load from the embeddings_dir
        else:
            print("Warning: No final embeddings found")

        return self.embedding_manager

    def load_all(self) -> Tuple[CoSiRModel, FeatureManager, TrainableEmbeddingManager]:
        """Load all components (model, features, embeddings)"""
        print("Loading all components...")

        feature_manager = self.load_feature_manager()
        model = self.load_model()
        embedding_manager = self.load_embedding_manager()

        print("All components loaded successfully!")
        return model, feature_manager, embedding_manager

    def get_sample_embedding(self, sample_id: int) -> torch.Tensor:
        """Get embedding for a specific sample"""
        if self.embedding_manager is None:
            raise RuntimeError("Embedding manager not loaded")

        return self.embedding_manager.get_embeddings([sample_id])

    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all sample IDs and their embeddings"""
        if self.embedding_manager is None:
            raise RuntimeError("Embedding manager not loaded")

        return self.embedding_manager.get_all_embeddings()

    def get_sample_features(self, sample_ids: list) -> Dict[str, torch.Tensor]:
        """Get CLIP features for specific samples"""
        if self.feature_manager is None:
            raise RuntimeError("Feature manager not loaded")

        return self.feature_manager.load_features(sample_ids)

    def get_representatives(self, num_representatives: int = 50) -> torch.Tensor:
        """Get representative embeddings using the same method as training"""
        _, label_embeddings_all = self.get_all_embeddings()
        return get_representatives(label_embeddings_all.cpu(), num_representatives)

    def create_evaluation_dataset(
        self, annotation_path: str, image_path: str, ratio: float = 1.0
    ) -> DataLoader:
        """Create evaluation dataset"""
        if self.processor is None:
            raise RuntimeError("Model must be loaded first to get processor")

        eval_set = CoSiRValidationDataset(
            annotation_path=annotation_path,
            image_path=image_path,
            processor=self.processor,
            ratio=ratio,
        )

        return DataLoader(
            eval_set,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False,
            num_workers=self.config["train"]["num_workers"],
        )

    def evaluate_on_dataset(
        self, dataloader: DataLoader, num_representatives: int = 50
    ) -> Tuple | MetricResult:
        """Run evaluation on a dataset using the evaluation manager"""
        if self.model is None or self.embedding_manager is None:
            raise RuntimeError("Model and embedding manager must be loaded")

        # Create evaluation config from experiment config
        evaluation_config = EvaluationConfig(**self.config["evaluation"])

        evaluator = EvaluationManager(evaluation_config)

        # Get representatives
        representatives = self.get_representatives(num_representatives)

        # Run evaluation
        detailed_results = evaluator.evaluate_test(
            model=self.model,
            processor=self.processor,
            dataloader=dataloader,
            label_embeddings=representatives,
            epoch=-1,  # Indicate this is final evaluation
            return_detailed_results=True,
        )

        return detailed_results

    def get_model_predictions(
        self, dataloader: DataLoader, num_representatives: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Get model predictions for analysis"""
        if self.model is None or self.embedding_manager is None:
            raise RuntimeError("Model and embedding manager must be loaded")

        representatives = self.get_representatives(num_representatives)

        all_similarities = []
        all_labels = []
        all_indices = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Getting predictions"):
                inputs, labels, indices = batch

                # Move to device
                pixel_values = inputs["pixel_values"].to(self.device)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Get model outputs
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compute similarities with representatives
                similarities = torch.mm(
                    outputs.normalized_combined_features,
                    representatives.T.to(self.device),
                )

                all_similarities.append(similarities.cpu())
                all_labels.append(labels)
                all_indices.append(indices)

        return {
            "similarities": torch.cat(all_similarities, dim=0),
            "labels": torch.cat(all_labels, dim=0),
            "indices": torch.cat(all_indices, dim=0),
            "representatives": representatives,
        }


def load_cosir_evaluator(
    experiment_path: str, device: Optional[str] = None
) -> CoSiREvaluator:
    """Convenience function to create and load a CoSiR evaluator"""
    evaluator = CoSiREvaluator(experiment_path, device)
    evaluator.load_all()
    return evaluator


# Example usage functions for notebooks
def quick_eval(
    experiment_path: str,
    test_annotation_path: str,
    test_image_path: str,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Quick evaluation on test set"""
    evaluator = load_cosir_evaluator(experiment_path, device)
    test_loader = evaluator.create_evaluation_dataset(
        test_annotation_path, test_image_path
    )
    return evaluator.evaluate_on_dataset(test_loader)


def get_embeddings_for_analysis(
    experiment_path: str, device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get all embeddings for analysis"""
    evaluator = load_cosir_evaluator(experiment_path, device)
    return evaluator.get_all_embeddings()
