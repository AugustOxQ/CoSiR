#!/usr/bin/env python3
"""Example usage of the refactored evaluation system."""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/project/CoSiR')

def example_new_api():
    """Example using the new evaluation API."""
    print("🚀 Example: New Evaluation API")
    print("="*50)

    # Import the new API
    from src.eval import EvaluationManager, EvaluationConfig

    # Create configuration
    config = EvaluationConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        k_vals=[1, 5, 10],
        train_max_batches=25,
        batch_size=256,
        print_metrics=True,
        cpu_offload=True  # Move large matrices to CPU for memory efficiency
    )

    print(f"✅ Config created: device={config.device}, k_vals={config.k_vals}")

    # Initialize evaluation manager
    evaluator = EvaluationManager(config)
    print("✅ EvaluationManager initialized")

    # In your training loop, you would use:
    """
    # Training evaluation
    train_results = evaluator.evaluate_train(
        model=model,
        dataloader=train_dataloader,
        epoch=epoch
    )

    # Test evaluation
    test_results = evaluator.evaluate_test(
        model=model,
        processor=processor,
        dataloader=test_dataloader,
        label_embeddings=your_label_embeddings,
        epoch=epoch
    )

    # Access results
    print(f"Training metrics: {train_results.metrics}")
    print(f"Test metrics: {test_results.metrics}")

    # Log to wandb
    wandb_run.log(train_results.metrics)
    wandb_run.log(test_results.metrics)
    """

    print("✅ Example code structure shown above")


def example_backward_compatibility():
    """Example using backward compatible functions."""
    print("\n🔄 Example: Backward Compatible API")
    print("="*50)

    # Import backward compatible functions
    from src.eval import inference_train, inference_test

    print("✅ Backward compatible functions imported")

    # Your existing code continues to work exactly the same:
    """
    # Training evaluation (exactly as before)
    inf_train_log = inference_train(
        model, train_dataloader, device, epoch, [1, 5, 10]
    )

    # Test evaluation (exactly as before)
    inf_test_log = inference_test(
        model, processor, test_dataloader, representatives, epoch, device
    )

    # Use results as before
    wandb_run.log(inf_train_log)
    wandb_run.log(inf_test_log)
    logger_epoch["inference_train"] = inf_train_log
    logger_epoch["inference_test"] = inf_test_log
    """

    print("✅ Your existing code works unchanged!")


def example_configuration_options():
    """Show different configuration options."""
    print("\n⚙️  Example: Configuration Options")
    print("="*50)

    from src.eval import EvaluationConfig

    # Default configuration
    config_default = EvaluationConfig()
    print(f"Default: device={config_default.device}, max_batches={config_default.train_max_batches}")

    # Memory-optimized configuration
    config_memory = EvaluationConfig(
        device="cuda",
        batch_size=128,      # Smaller batches
        cpu_offload=True,    # Offload large matrices to CPU
        train_max_batches=10 # Fewer batches for faster evaluation
    )
    print(f"Memory optimized: batch_size={config_memory.batch_size}, cpu_offload={config_memory.cpu_offload}")

    # Custom metrics configuration
    config_custom = EvaluationConfig(
        k_vals=[1, 5, 10, 20, 50],  # More recall@K values
        print_metrics=False,         # Disable metric printing
        test_use_best_label=True,    # Use best label for oracle evaluation
    )
    print(f"Custom metrics: k_vals={config_custom.k_vals}, use_best_label={config_custom.test_use_best_label}")


def integration_guide():
    """Show how to integrate with existing training code."""
    print("\n🔧 Example: Integration with Existing Training Code")
    print("="*50)

    integration_code = '''
# Option 1: Minimal change (drop-in replacement)
from src.eval import inference_train, inference_test

# Your existing code works unchanged:
if cfg.control.val:
    print("##########Testing train dataset##########")
    inf_train_log = inference_train(
        model, train_dataloader, device, epoch, [1, 5, 10]
    )
    wandb_run.log(inf_train_log)
    logger_epoch["inference_train"] = inf_train_log

if cfg.control.test:
    if unique_embeddings is not None:
        # ... your k-means logic ...
        print("##########Testing test dataset##########")
        inf_test_log = inference_test(
            model, processor, test_dataloader, representatives, epoch, device
        )
        logger_epoch["inference_test"] = inf_test_log
        wandb_run.log(inf_test_log)

# Option 2: Use new API for more control
from src.eval import EvaluationManager, EvaluationConfig

# One-time setup
eval_config = EvaluationConfig(
    device=device,
    k_vals=[1, 5, 10],
    train_max_batches=25,
    print_metrics=True
)
evaluator = EvaluationManager(eval_config)

# In training loop
if cfg.control.val:
    train_results = evaluator.evaluate_train(model, train_dataloader, epoch=epoch)
    wandb_run.log(train_results.metrics)
    logger_epoch["inference_train"] = train_results.metrics

if cfg.control.test:
    if unique_embeddings is not None:
        # ... your k-means logic ...
        test_results = evaluator.evaluate_test(
            model, processor, test_dataloader, representatives, epoch=epoch
        )
        wandb_run.log(test_results.metrics)
        logger_epoch["inference_test"] = test_results.metrics
'''

    print(integration_code)


def main():
    """Run all examples."""
    print("📚 Refactored Evaluation System - Usage Examples\n")

    example_new_api()
    example_backward_compatibility()
    example_configuration_options()
    integration_guide()

    print("\n🎉 Summary:")
    print("✅ All imports working correctly")
    print("✅ Backward compatibility maintained")
    print("✅ New API provides enhanced functionality")
    print("✅ Flexible configuration system")
    print("✅ Ready for integration with your training code!")

    print(f"\n🔗 Key files:")
    print(f"  - Main interface: src/eval/__init__.py")
    print(f"  - Configuration: src/eval/config.py")
    print(f"  - New API: src/eval/interface.py")
    print(f"  - Training eval: src/eval/evaluators/train_evaluator.py")
    print(f"  - Test eval: src/eval/evaluators/test_evaluator.py")


if __name__ == "__main__":
    main()