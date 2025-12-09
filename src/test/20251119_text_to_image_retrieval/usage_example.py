"""
Example usage of text-to-image retrieval functions in training script
This shows how to integrate the new functions into train_cosir.py
"""

# In train_cosir.py, after evaluation (around line 465), add:

"""
# Import at the top of train_cosir.py (already imported via src.utils)
from src.utils import (
    visualize_angular_semantics_fast,  # existing
    visualize_angular_semantics_text_to_image_fast,  # NEW
)

# After line 465 where all_raw_image is obtained:
all_raw_image = test_set.get_all_raw_image()

# Existing visualization (image-to-text)
fig3 = visualize_angular_semantics_fast(
    label_embeddings_all.cpu().numpy(),
    model,
    (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
    device=device,
)
experiment.save_artifact(
    name=f"angular_semantics_fast_{epoch}",
    data=fig3,
    artifact_type="figure",
    folder="plots",
    description=f"Angular semantics visualization (image-to-text) at epoch {epoch}",
)

# NEW: Text-to-image visualization
fig4 = visualize_angular_semantics_text_to_image_fast(
    label_embeddings_all.cpu().numpy(),
    model,
    (all_img_emb, all_txt_emb, all_raw_text, text_to_image_map, all_raw_image),
    device=device,
)
experiment.save_artifact(
    name=f"angular_semantics_text_to_image_{epoch}",
    data=fig4,
    artifact_type="figure",
    folder="plots",
    description=f"Angular semantics visualization (text-to-image) at epoch {epoch}",
)

# Optional: Log to wandb
logger.log_metrics({
    "vis/angular_semantics_img_to_txt": wandb.Image(fig3),
    "vis/angular_semantics_txt_to_img": wandb.Image(fig4),
})

plt.close("all")
"""

# Standalone usage example
"""
import torch
from src.utils import retrieve_image_with_condition_fast

# After obtaining precomputed data from evaluation
precomputed_data = (
    all_img_emb,        # [N_images, 512]
    all_txt_emb,        # [N_texts, 512]
    all_raw_text,       # list of N_texts strings
    text_to_image_map,  # [N_texts] tensor mapping text idx to image idx
    all_raw_image       # list of N_images PIL Images
)

# Retrieve images for a specific text with different conditions
query_text_id = 42
conditions = [
    torch.tensor([0.0, 0.0]),   # Neutral
    torch.tensor([1.0, 0.0]),   # Right
    torch.tensor([0.0, 1.0]),   # Up
    torch.tensor([-1.0, 0.0]),  # Left
]

for i, condition in enumerate(conditions):
    top_images = retrieve_image_with_condition_fast(
        model=model,
        precomputed_data=precomputed_data,
        condition=condition,
        query_text_id=query_text_id,
        k=5,
        device=device
    )

    print(f"Condition {i}: {condition.numpy()}")
    print(f"  Query: {all_raw_text[query_text_id]}")
    print(f"  Retrieved {len(top_images)} images")

    # Save or display images
    for j, img in enumerate(top_images):
        img.save(f"retrieved_cond{i}_rank{j}.png")
"""
