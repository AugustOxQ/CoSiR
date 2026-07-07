"""
Experiment lifecycle management for CoSiR training runs.

Provides three classes:
  ExperimentPaths   — standardized path layout under an experiment directory
  ExperimentContext — per-run state: metrics, artifacts, checkpoints, notes
  ExperimentManager — registry of all runs; creates and loads ExperimentContext objects
"""
import json
import shutil
import tarfile
import time
import pickle
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import matplotlib.pyplot as plt


class ExperimentPaths:
    """Centralized path management for experiments"""

    def __init__(self, base_dir: Path):
        self.base = base_dir

        # Standard paths
        self.checkpoints = base_dir / "checkpoints"
        self.logs = base_dir / "logs"
        self.plots = base_dir / "plots"
        self.embeddings = base_dir / "embeddings"
        self.features = base_dir / "features"
        self.results = base_dir / "results"
        self.artifacts = base_dir / "artifacts"
        self.configs = base_dir / "configs"
        self.scripts = base_dir / "scripts"

    def get_epoch_dir(self, epoch: int) -> Path:
        """Get epoch-specific directory"""
        epoch_dir = self.base / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        return epoch_dir

    def get_plot_path(self, plot_name: str, epoch: Optional[int] = None) -> Path:
        """Get standardized plot path"""
        if epoch is not None:
            return self.plots / f"{plot_name}_epoch_{epoch}.png"
        else:
            return self.plots / f"{plot_name}.png"

    def get_checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        """Get checkpoint path"""
        if epoch is None:
            return self.checkpoints / "latest.pt"
        else:
            return self.checkpoints / f"checkpoint_epoch_{epoch}.pt"


class ExperimentContext:
    """Context manager for experiment lifecycle"""

    def __init__(
        self,
        name: str,
        directory: Path,
        config: Dict[str, Any],
        tags: List[str],
        description: Optional[str] = None,
        parent_experiment: Optional[str] = None,
    ):

        self.name = name
        self.directory = directory
        self.config = config
        self.tags = tags
        self.description = description or ""
        self.parent_experiment = parent_experiment

        # Experiment state
        self.status = "created"
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.current_epoch = 0

        # Metrics and artifacts
        self.metrics_history = []
        self.artifacts = {}
        self.notes = []

        # Paths
        self.paths = ExperimentPaths(directory)

        # Create directories first
        self._create_directories()

        # Save initial config
        self._save_config()

    def __enter__(self):
        self.status = "running"
        self.start_time = time.time()
        self._save_status()

        # Create run-specific log
        run_log = {"started": self.start_time, "config": self.config, "tags": self.tags}

        with open(self.paths.logs / "run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if exc_type is None:
            self.status = "completed"
        else:
            self.status = "failed"
            # Log error information
            self._log_error(exc_type, exc_val, exc_tb)

        self._save_status()
        self._save_final_summary()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.checkpoints,
            self.paths.logs,
            self.paths.plots,
            self.paths.embeddings,
            self.paths.features,
            self.paths.results,
            self.paths.artifacts,
            self.paths.configs,
            self.paths.scripts,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_metadata(
        cls, directory: Path, metadata: Dict[str, Any]
    ) -> "ExperimentContext":
        """Create ExperimentContext from saved metadata"""
        context = cls(
            name=metadata["name"],
            directory=directory,
            config=metadata.get("config", {}),
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
            parent_experiment=metadata.get("parent_experiment"),
        )

        context.status = metadata.get("status", "unknown")
        context.created_time = metadata.get("created", time.time())

        # Load existing data
        context._load_existing_data()

        return context

    def _save_config(self):
        """Save current configuration"""
        config_file = self.paths.configs / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_status(self):
        """Save current status"""
        status_info = {
            "status": self.status,
            "created": self.created_time,
            "started": self.start_time,
            "ended": self.end_time,
            "current_epoch": self.current_epoch,
            "last_updated": time.time(),
        }

        status_file = self.directory / "status.json"
        with open(status_file, "w") as f:
            json.dump(status_info, f, indent=2, default=str)

    def _log_error(self, exc_type, exc_val, exc_tb):
        """Log error information"""
        import traceback

        error_info = {
            "timestamp": time.time(),
            "error_type": exc_type.__name__ if exc_type else None,
            "error_message": str(exc_val) if exc_val else None,
            "traceback": traceback.format_tb(exc_tb) if exc_tb else None,
        }

        error_file = self.paths.logs / "error.json"
        with open(error_file, "w") as f:
            json.dump(error_info, f, indent=2, default=str)

    def _save_final_summary(self):
        """Save final experiment summary"""
        summary = {
            "name": self.name,
            "status": self.status,
            "created": self.created_time,
            "started": self.start_time,
            "ended": self.end_time,
            "duration": (
                self.end_time - self.start_time
                if self.start_time and self.end_time
                else None
            ),
            "total_epochs": self.current_epoch,
            "final_metrics": self.get_final_metrics(),
            "config": self.config,
            "tags": self.tags,
            "description": self.description,
            "artifacts": list(self.artifacts.keys()),
            "notes": self.notes,
        }

        summary_file = self.directory / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _load_existing_data(self):
        """Load existing data from previous runs"""
        # Load metrics history
        if self.paths.logs.exists():
            metrics_files = sorted(self.paths.logs.glob("metrics_epoch_*.json"))
            for metrics_file in metrics_files:
                try:
                    with open(metrics_file) as f:
                        metric_data = json.load(f)
                        self.metrics_history.append(metric_data)
                except:
                    continue

        # Update current epoch
        if self.metrics_history:
            self.current_epoch = max(m.get("epoch", 0) for m in self.metrics_history)

        # Load artifacts info
        artifacts_file = self.directory / "artifacts.json"
        if artifacts_file.exists():
            with open(artifacts_file) as f:
                self.artifacts = json.load(f)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """Log metrics for current epoch/step."""
        if epoch is None:
            epoch = self.current_epoch

        metric_entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics.copy(),
        }

        self.metrics_history.append(metric_entry)

        # Save to file
        if step is None:
            metrics_file = self.paths.logs / f"metrics_epoch_{epoch}.json"
        else:
            metrics_file = self.paths.logs / f"metrics_epoch_{epoch}_step_{step}.json"

        with open(metrics_file, "w") as f:
            json.dump(metric_entry, f, indent=2, default=str)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        embedding_manager: Any,
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Save comprehensive checkpoint."""
        if epoch is None:
            epoch = self.current_epoch

        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "embeddings_checkpoint": embedding_manager.create_checkpoint(),
            "optimizer_state": optimizer_state,
            "config": self.config,
            "timestamp": time.time(),
            "experiment_name": self.name,
        }

        if additional_data:
            checkpoint["additional_data"] = additional_data

        checkpoint_path = self.paths.checkpoints / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Keep link to latest checkpoint
        latest_path = self.paths.checkpoints / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()

        # Create relative symlink
        try:
            latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback for systems that don't support symlinks
            shutil.copy2(checkpoint_path, latest_path)

        # Keep only last N checkpoints to save space
        self._cleanup_old_checkpoints(keep_last=5)

    def load_checkpoint(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Load checkpoint."""
        if epoch is None:
            checkpoint_path = self.paths.checkpoints / "latest.pt"
        else:
            checkpoint_path = self.paths.checkpoints / f"checkpoint_epoch_{epoch}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu")

    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Keep only the last N checkpoints"""
        checkpoint_files = []
        for f in self.paths.checkpoints.glob("checkpoint_epoch_*.pt"):
            try:
                epoch_num = int(f.stem.split("_")[-1])
                checkpoint_files.append((epoch_num, f))
            except ValueError:
                continue

        checkpoint_files.sort(key=lambda x: x[0])  # Sort by epoch number

        if len(checkpoint_files) > keep_last:
            for _, old_checkpoint in checkpoint_files[:-keep_last]:
                try:
                    old_checkpoint.unlink()
                except OSError:
                    pass

    def save_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "pickle",
        description: Optional[str] = None,
        folder: str = "artifacts",
    ):
        """Save experiment artifact.

        artifact_type: 'pickle' | 'json' | 'torch' | 'numpy' | 'png' | 'figure'
        folder: one of the standard subdirectory names (artifacts, plots, checkpoints, …)
        """
        available_folders = [
            "artifacts",
            "plots",
            "results",
            "checkpoints",
            "logs",
            "embeddings",
            "features",
            "configs",
            "scripts",
        ]
        if folder not in available_folders:
            raise ValueError(
                f"Invalid folder '{folder}'. Must be one of: {available_folders}"
            )

        target_dir = getattr(self.paths, folder)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if "." in name:
            base_name, ext = name.rsplit(".", 1)
            timestamped_name = f"{base_name}_{timestamp}.{ext}"
        else:
            timestamped_name = f"{name}_{timestamp}"

        if artifact_type == "pickle":
            if not timestamped_name.endswith(".pkl"):
                timestamped_name = f"{timestamped_name}.pkl"
            artifact_path = target_dir / timestamped_name
            with open(artifact_path, "wb") as f:
                pickle.dump(data, f)

        elif artifact_type == "json":
            if not timestamped_name.endswith(".json"):
                timestamped_name = f"{timestamped_name}.json"
            artifact_path = target_dir / timestamped_name
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif artifact_type == "torch":
            if not timestamped_name.endswith(".pt"):
                timestamped_name = f"{timestamped_name}.pt"
            artifact_path = target_dir / timestamped_name
            torch.save(data, artifact_path)

        elif artifact_type == "numpy":
            if not timestamped_name.endswith(".npy"):
                timestamped_name = f"{timestamped_name}.npy"
            artifact_path = target_dir / timestamped_name
            np.save(artifact_path, data)

        elif artifact_type in ("png", "figure"):
            if not timestamped_name.endswith(".png"):
                timestamped_name = f"{timestamped_name}.png"
            artifact_path = target_dir / timestamped_name
            if hasattr(data, "savefig"):
                data.savefig(artifact_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
            elif hasattr(data, "save"):  # PIL Image
                data.save(artifact_path)
            elif artifact_type == "png":
                plt.imsave(artifact_path, data)
            else:
                raise ValueError("Data must be a matplotlib figure for 'figure' artifact type")

        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

        # Register artifact
        self.artifacts[name] = {
            "path": str(artifact_path),
            "type": artifact_type,
            "description": description or "",
            "created": time.time(),
            "size_bytes": artifact_path.stat().st_size,
        }

        artifacts_file = self.directory / "artifacts.json"
        with open(artifacts_file, "w") as f:
            json.dump(self.artifacts, f, indent=2, default=str)

    def load_artifact(self, name: str) -> Any:
        """Load experiment artifact"""
        if name not in self.artifacts:
            raise ValueError(f"Artifact {name} not found")

        artifact_info = self.artifacts[name]
        artifact_path = Path(artifact_info["path"])
        artifact_type = artifact_info["type"]

        if artifact_type == "pickle":
            with open(artifact_path, "rb") as f:
                return pickle.load(f)
        elif artifact_type == "json":
            with open(artifact_path) as f:
                return json.load(f)
        elif artifact_type == "torch":
            return torch.load(artifact_path, map_location="cpu")
        elif artifact_type == "numpy":
            return np.load(artifact_path)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

    def add_note(self, note: str, category: str = "general"):
        """Add a note to the experiment"""
        note_entry = {"timestamp": time.time(), "category": category, "note": note}
        self.notes.append(note_entry)

        notes_file = self.paths.logs / "notes.json"
        with open(notes_file, "w") as f:
            json.dump(self.notes, f, indent=2, default=str)

    def get_final_metrics(self) -> Dict[str, float]:
        """Get metrics from the last logged epoch"""
        if not self.metrics_history:
            return {}
        final_entry = max(self.metrics_history, key=lambda x: x.get("epoch", 0))
        return final_entry.get("metrics", {})

    def get_metrics_dataframe(self):
        """Get metrics as pandas DataFrame for analysis"""
        records = []
        for entry in self.metrics_history:
            record = {
                "epoch": entry.get("epoch", 0),
                "step": entry.get("step"),
                "timestamp": entry.get("timestamp", 0),
            }
            record.update(entry.get("metrics", {}))
            records.append(record)

        import pandas as pd

        return pd.DataFrame(records)

    def plot_metrics(
        self, metric_names: Optional[List[str]] = None, save_path: Optional[Path] = None
    ):
        """Plot experiment metrics"""
        df = self.get_metrics_dataframe()
        if isinstance(df, list) or (hasattr(df, "empty") and df.empty):
            print("No metrics to plot")
            return

        if metric_names is None:
            import pandas as pd

            metric_names = [
                col
                for col in df.columns
                if col not in ["epoch", "step", "timestamp"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

        fig, axes = plt.subplots(
            len(metric_names), 1, figsize=(10, 3 * len(metric_names))
        )
        if len(metric_names) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_names):
            if metric in df.columns:
                ax.plot(df["epoch"], df[metric], marker="o")
                ax.set_title(f"{metric} vs Epoch")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.paths.get_plot_path("metrics_history")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class ExperimentManager:
    """Comprehensive experiment lifecycle management"""

    def __init__(self, base_experiments_dir: Union[str, Path] = "experiments"):
        self.base_dir = Path(base_experiments_dir)
        self._ensure_directory_exists(self.base_dir)

        self.current_experiment = None
        self.experiment_registry = self._load_registry()

    def _ensure_directory_exists(self, directory_path: Path):
        """Recursively create directory and all parent directories if they don't exist"""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                f"Permission denied: Cannot create directory {directory_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {directory_path}: {str(e)}")

    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry"""
        registry_file = self.base_dir / "experiment_registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                return json.load(f)
        return {"experiments": {}, "tags": defaultdict(list)}

    def _save_registry(self):
        """Save experiment registry"""
        registry_file = self.base_dir / "experiment_registry.json"
        with open(registry_file, "w") as f:
            json.dump(self.experiment_registry, f, indent=2, default=str)

    def create_experiment(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        parent_experiment: Optional[str] = None,
    ) -> ExperimentContext:
        """Create new experiment with automatic naming and directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is None:
            name = f"exp_{timestamp}"
        else:
            name = f"{timestamp}_{name.replace(' ', '_')}"

        # Ensure unique name
        counter = 1
        original_name = name
        while (self.base_dir / name).exists():
            name = f"{original_name}_{counter}"
            counter += 1

        exp_dir = self.base_dir / name
        exp_dir.mkdir(exist_ok=True)

        experiment = ExperimentContext(
            name=name,
            directory=exp_dir,
            config=config or {},
            tags=tags or [],
            description=description,
            parent_experiment=parent_experiment,
        )

        self._create_standard_structure(experiment)
        self._save_experiment_metadata(experiment)
        self._register_experiment(experiment)

        self.current_experiment = experiment
        return experiment

    def load_experiment(self, experiment_name: str) -> ExperimentContext:
        """Load existing experiment by name (partial match supported)."""
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            matches = [
                d
                for d in self.base_dir.iterdir()
                if d.is_dir() and experiment_name in d.name
            ]
            if len(matches) == 1:
                exp_dir = matches[0]
                experiment_name = exp_dir.name
            else:
                raise ValueError(
                    f"Experiment {experiment_name} not found. "
                    f"Matches: {[m.name for m in matches]}"
                )

        metadata = self._load_experiment_metadata(exp_dir)
        experiment = ExperimentContext.from_metadata(exp_dir, metadata)

        self.current_experiment = experiment
        return experiment

    def _create_standard_structure(self, experiment: ExperimentContext):
        """Create standard directory structure"""
        directories = [
            "checkpoints",
            "logs",
            "plots",
            "embeddings",
            "training_embeddings",
            "final_embeddings",
            "features",
            "results",
            "artifacts",
            "configs",
            "scripts",
        ]

        for dir_name in directories:
            (experiment.directory / dir_name).mkdir(exist_ok=True)

    def _save_experiment_metadata(self, experiment: ExperimentContext):
        """Save experiment metadata"""
        metadata = {
            "name": experiment.name,
            "created": experiment.created_time,
            "config": experiment.config,
            "tags": experiment.tags,
            "description": experiment.description,
            "parent_experiment": experiment.parent_experiment,
            "status": experiment.status,
            "directory": str(experiment.directory),
        }

        metadata_file = experiment.directory / "experiment_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_experiment_metadata(self, exp_dir: Path) -> Dict[str, Any]:
        """Load experiment metadata"""
        metadata_file = exp_dir / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        else:
            # Fallback for legacy experiments
            return {
                "name": exp_dir.name,
                "created": time.time(),
                "config": {},
                "tags": [],
                "description": "",
                "parent_experiment": None,
                "status": "unknown",
            }

    def _register_experiment(self, experiment: ExperimentContext):
        """Register experiment in global registry"""
        exp_info = {
            "created": experiment.created_time,
            "status": experiment.status,
            "tags": experiment.tags,
            "description": experiment.description,
            "parent_experiment": experiment.parent_experiment,
        }

        self.experiment_registry["experiments"][experiment.name] = exp_info

        for tag in experiment.tags:
            if experiment.name not in self.experiment_registry["tags"][tag]:
                self.experiment_registry["tags"][tag].append(experiment.name)

        self._save_registry()

    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        parent: Optional[str] = None,
        sort_by: str = "created",
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering by tags, status, or parent."""
        experiments = []

        for exp_name, exp_info in self.experiment_registry["experiments"].items():
            if tags and not any(tag in exp_info.get("tags", []) for tag in tags):
                continue
            if status and exp_info.get("status") != status:
                continue
            if parent and exp_info.get("parent_experiment") != parent:
                continue

            experiments.append(
                {
                    "name": exp_name,
                    "created": exp_info.get("created"),
                    "status": exp_info.get("status", "unknown"),
                    "tags": exp_info.get("tags", []),
                    "description": exp_info.get("description", ""),
                    "parent_experiment": exp_info.get("parent_experiment"),
                }
            )

        if sort_by == "created":
            experiments.sort(key=lambda x: x["created"], reverse=True)
        elif sort_by == "name":
            experiments.sort(key=lambda x: x["name"])
        elif sort_by == "status":
            experiments.sort(key=lambda x: x["status"])

        return experiments

    def archive_experiment(
        self,
        experiment_name: str,
        archive_path: Optional[Path] = None,
        remove_original: bool = True,
    ):
        """Archive a completed experiment as a .tar.gz file."""
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")

        if archive_path is None:
            archive_dir = self.base_dir / "archived"
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / f"{experiment_name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(exp_dir, arcname=experiment_name)

        print(f"Archived experiment to {archive_path}")

        if experiment_name in self.experiment_registry["experiments"]:
            self.experiment_registry["experiments"][experiment_name]["status"] = "archived"
            self.experiment_registry["experiments"][experiment_name]["archive_path"] = (
                str(archive_path)
            )
            self._save_registry()

        if remove_original:
            shutil.rmtree(exp_dir)
            print(f"Removed original experiment directory")

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare config and final metrics across multiple experiments."""
        comparison: Dict[str, Any] = {
            "experiments": {},
            "common_config": {},
            "config_differences": {},
            "metric_comparison": {},
        }

        for exp_name in experiment_names:
            try:
                exp = self.load_experiment(exp_name)
                comparison["experiments"][exp_name] = {
                    "config": exp.config,
                    "status": exp.status,
                    "created": exp.created_time,
                    "final_metrics": exp.get_final_metrics(),
                }
            except Exception as e:
                comparison["experiments"][exp_name] = {"error": str(e)}

        if len(comparison["experiments"]) > 1:
            configs = [
                exp_data.get("config", {})
                for exp_data in comparison["experiments"].values()
                if "config" in exp_data
            ]
            if configs:
                common_keys = set(configs[0].keys())
                for config in configs[1:]:
                    common_keys &= set(config.keys())

                for key in common_keys:
                    values = [config[key] for config in configs]
                    if all(v == values[0] for v in values):
                        comparison["common_config"][key] = values[0]
                    else:
                        comparison["config_differences"][key] = {
                            exp_name: config.get(key)
                            for exp_name, config in zip(
                                comparison["experiments"].keys(), configs
                            )
                        }

        return comparison

    def cleanup_failed_experiments(self, older_than_days: int = 7):
        """Delete failed experiments older than the specified number of days."""
        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        to_remove = []
        for exp_name, exp_info in self.experiment_registry["experiments"].items():
            if (
                exp_info.get("status") == "failed"
                and exp_info.get("created", 0) < cutoff_time
            ):
                to_remove.append(exp_name)

        for exp_name in to_remove:
            exp_dir = self.base_dir / exp_name
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
                print(f"Cleaned up failed experiment: {exp_name}")

            if exp_name in self.experiment_registry["experiments"]:
                del self.experiment_registry["experiments"][exp_name]

        if to_remove:
            self._save_registry()
