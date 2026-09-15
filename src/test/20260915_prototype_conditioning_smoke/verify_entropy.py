#!/usr/bin/env python3
"""
Verify prototype_bank state was saved in checkpoints.
Usage: python verify_entropy.py
Outputs: confirms prototype_bank state_dict and config are present in checkpoints.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.model.prototype_bank import PrototypeBank


def verify_checkpoint(exp_dir: Path, num_prototypes: int) -> dict:
    """
    Load a checkpoint and verify prototype_bank state is present.

    Args:
        exp_dir: Path to experiment directory (contains checkpoints/ subdirectory)
        num_prototypes: expected num_prototypes value for this run

    Returns:
        dict with verification results
    """
    checkpoints_dir = exp_dir / "checkpoints"

    # Find the phase_1_model checkpoint
    checkpoint_files = sorted(checkpoints_dir.glob("phase_1_model_*.pt"))
    if not checkpoint_files:
        return {
            "status": "CHECKPOINT_NOT_FOUND",
            "checkpoint_path": None,
            "has_prototype_bank": False,
            "entropy": None,
        }

    # Use the latest checkpoint
    checkpoint_path = checkpoint_files[-1]
    print(f"[P={num_prototypes}] Loading: {checkpoint_path.name}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        return {
            "status": f"CHECKPOINT_LOAD_ERROR: {e}",
            "checkpoint_path": str(checkpoint_path),
            "has_prototype_bank": False,
            "entropy": None,
        }

    # Check for prototype_bank state
    has_prototype_bank = "prototype_bank_state_dict" in checkpoint
    has_proto_config = "prototype_bank_config" in checkpoint

    if not has_prototype_bank or not has_proto_config:
        return {
            "status": "MISSING_PROTOTYPE_BANK_STATE" if not has_prototype_bank else "MISSING_PROTOTYPE_BANK_CONFIG",
            "checkpoint_path": str(checkpoint_path),
            "has_prototype_bank": has_prototype_bank,
            "has_proto_config": has_proto_config,
            "entropy": None,
        }

    # Verify the checkpoint contains the expected num_prototypes
    proto_config = checkpoint["prototype_bank_config"]
    actual_num_prototypes = proto_config.get("num_prototypes")

    if actual_num_prototypes != num_prototypes:
        return {
            "status": f"NUM_PROTOTYPES_MISMATCH (expected {num_prototypes}, got {actual_num_prototypes})",
            "checkpoint_path": str(checkpoint_path),
            "has_prototype_bank": has_prototype_bank,
            "entropy": None,
        }

    try:
        # Reconstruct PrototypeBank
        prototype_bank = PrototypeBank(
            num_prototypes=proto_config["num_prototypes"],
            condition_dim=proto_config["condition_dim"],
            query_dim=proto_config["query_dim"],
            temperature_init=proto_config.get("temperature_init", 1.0),
        )

        # Load state dict
        prototype_bank.load_state_dict(checkpoint["prototype_bank_state_dict"])
        prototype_bank.eval()

        # Compute entropy with fresh/zero counts
        # (The _usage_counts will be zeros, giving uniform distribution entropy = log(P))
        prototype_bank._usage_counts = torch.zeros(num_prototypes)
        entropy = prototype_bank.usage_entropy().item()
        log_p = float(torch.log(torch.tensor(num_prototypes)).item())

        return {
            "status": "OK",
            "checkpoint_path": str(checkpoint_path),
            "has_prototype_bank": True,
            "entropy": entropy,
            "log_p": log_p,
            "entropy_ratio": entropy / log_p,  # Should be close to 1.0 with zero counts
        }

    except Exception as e:
        return {
            "status": f"ERROR_DURING_VERIFICATION: {str(e)}",
            "checkpoint_path": str(checkpoint_path),
            "has_prototype_bank": has_prototype_bank,
            "entropy": None,
        }


def main():
    """Run checkpoint verification for all three P values."""

    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION: prototype_bank state")
    print("="*70)

    results = {}
    for p in [8, 16, 32]:
        # Find the latest experiment directory for this P
        results_dirs = list(Path(f"/tmp/exp18_smoke_p{p}").glob("202609*_CoSiR_Experiment"))
        if not results_dirs:
            print(f"\n[P={p}] No experiment directory found")
            results[p] = {"status": "NO_EXPERIMENT_DIR", "entropy": None}
            continue

        exp_dir = sorted(results_dirs)[-1]  # Use latest
        print(f"\n[P={p}] Experiment: {exp_dir.name}")
        result = verify_checkpoint(exp_dir, p)
        results[p] = result

        if result["status"] == "OK":
            print(f"  Status: OK")
            print(f"  Checkpoint: {Path(result['checkpoint_path']).name}")
            print(f"  Entropy (zero counts): {result['entropy']:.4f}")
            print(f"  log(P): {result['log_p']:.4f}")
            print(f"  Ratio: {result['entropy_ratio']:.4f} (should be ~1.0 with uniform/zero counts)")
        else:
            print(f"  Status: {result['status']}")
            print(f"  Checkpoint path: {result.get('checkpoint_path', 'N/A')}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for p in [8, 16, 32]:
        r = results[p]
        if r["status"] == "OK":
            print(f"P={p:2d}: OK - entropy={r['entropy']:.4f} (log({p})={r['log_p']:.4f})")
        else:
            print(f"P={p:2d}: {r['status']}")

    return results


if __name__ == "__main__":
    main()
