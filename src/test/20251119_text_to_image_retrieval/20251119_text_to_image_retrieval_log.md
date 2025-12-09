# Text-to-Image Retrieval Functions Implementation Log
Date: 2025-11-19

## Problem Description
The existing codebase had image-to-text retrieval functions (`retrieve_with_condition_fast` and `visualize_angular_semantics_fast`) that:
- Fix a single image as query
- Retrieve top-k texts using different condition angles
- Visualize how different conditions affect text retrieval

The task was to create the inverse: text-to-image retrieval functions that:
- Fix a single text as query
- Retrieve top-k images using different condition angles
- Visualize how different conditions affect image retrieval

## Investigation Steps

### 1. Understanding Existing Functions
Analyzed the existing functions in `/project/CoSiR/src/utils/tools.py`:

**`retrieve_with_condition_fast`** (lines 166-210):
- Takes precomputed data: `(image_embs, text_embs, all_captions_flat, img_to_captions_map)`
- Fixes a query image
- Modulates ALL text embeddings with the condition
- Computes cosine similarity between image and modulated texts
- Returns top-k matching captions

**`visualize_angular_semantics_fast`** (lines 213-286):
- Uses 12 angle bins to partition the condition space
- For each angle bin, retrieves top-20 captions using the bin center condition
- Creates a 3x4 grid visualization showing caption results for each angle

### 2. Understanding Data Structures
From `/project/CoSiR/src/hook/train_cosir.py` (lines 456-465):
```python
(
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    text_to_image_map,
    image_to_text_map,
    test_results,
) = test_results_detailed

all_raw_image = test_set.get_all_raw_image()
```

From `/project/CoSiR/src/eval/processors/embeddings.py` (lines 65-96):
- `image_to_text_map`: list of text indices for each image (5 captions per image)
- `text_to_image_map`: image index for each text (each text maps to 1 image)
- `all_raw_text`: list of all caption strings
- `all_raw_image`: list of PIL Image objects (from `get_all_raw_image()`)

## Solution Implemented

### 1. New Function: `retrieve_image_with_condition_fast`
Location: `/project/CoSiR/src/utils/tools.py` (lines 289-345)

**Key Changes from Original:**
- **Input**: `precomputed_data = (image_embs, text_embs, all_raw_text, text_to_image_map, all_raw_image)`
- **Query**: Fix a single text instead of image (`query_text_id`)
- **Modulation**: Modulate the QUERY text embedding (not all texts) with condition
- **Similarity**: Compute similarity between modulated text and ALL images
- **Return**: Top-k images (PIL Image objects or arrays)

**Logic Flow:**
```python
# Select query text
txt_emb = text_embs[query_text_id:query_text_id + 1]  # [1, 512]

# Modulate query text with condition
txt_emb_modulated = model.combine(txt_emb, None, condition)

# Compute similarity with all images
similarities = F.cosine_similarity(
    txt_emb_modulated.expand(len(image_embs), -1),
    image_embs,
    dim=1
)

# Get top-k images
top_k_indices = torch.topk(similarities, k)
top_k_images = [all_raw_image[idx] for idx in top_k_indices]
```

### 2. New Function: `visualize_angular_semantics_text_to_image_fast`
Location: `/project/CoSiR/src/utils/tools.py` (lines 348-429)

**Key Changes from Original:**
- Fix a single query text instead of image
- Retrieve top-9 images (for 3x3 grid) per angle bin
- Display images instead of text captions
- Use `GridSpecFromSubplotSpec` to create inner 3x3 image grid in each subplot

**Visualization Structure:**
- 3x4 grid (12 subplots for 12 angle bins)
- Each subplot shows:
  - Title with angle range and query text snippet
  - 3x3 grid of top-9 retrieved images for that angle's condition

## Testing

### Test Script
Created: `/project/CoSiR/src/test/20251119_text_to_image_retrieval/test_text_to_image_retrieval.py`

**Test 1: `retrieve_image_with_condition_fast`**
- Created mock data with 50 images, 250 texts (5 per image)
- Tested with 5 different conditions: [0,0], [1,0], [0,1], [-1,0], [0,-1]
- Fixed query text ID = 10
- Retrieved k=5 images per condition
- ✓ All tests passed

**Test 2: `visualize_angular_semantics_text_to_image_fast`**
- Generated 36 conditions using polar grid (12 angles × 3 radii)
- Created visualization with mock images (random colored squares)
- Saved to: `test_visualization.png`
- ✓ Visualization created successfully

### Test Results
```
================================================================================
✓ All tests passed!
================================================================================
```

## Files Modified

### 1. `/project/CoSiR/src/utils/tools.py`
**Added Functions:**
- `retrieve_image_with_condition_fast` (lines 289-345)
- `visualize_angular_semantics_text_to_image_fast` (lines 348-429)

**Changes:**
- Added two new functions following the same pattern as existing functions
- Both functions are automatically exported via `from .tools import *` in `src/utils/__init__.py`

### 2. Test Files Created
- `/project/CoSiR/src/test/20251119_text_to_image_retrieval/test_text_to_image_retrieval.py`
- `/project/CoSiR/src/test/20251119_text_to_image_retrieval/test_visualization.png`
- `/project/CoSiR/src/test/20251119_text_to_image_retrieval/20251119_text_to_image_retrieval_log.md`

## Usage Example

### In Training Script
```python
from src.utils import (
    retrieve_image_with_condition_fast,
    visualize_angular_semantics_text_to_image_fast,
)

# After evaluation, prepare data
precomputed_data = (
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    text_to_image_map,
    all_raw_image  # from test_set.get_all_raw_image()
)

# Visualize text-to-image retrieval across conditions
fig = visualize_angular_semantics_text_to_image_fast(
    conditions_2d=label_embeddings_all.cpu().numpy(),
    model=model,
    precomputed_data=precomputed_data,
    save_path="text_to_image_angular.png",
    device=device
)

# Or retrieve specific images with a condition
top_images = retrieve_image_with_condition_fast(
    model=model,
    precomputed_data=precomputed_data,
    condition=torch.tensor([1.0, 0.0]),
    query_text_id=42,
    k=10,
    device=device
)
```

## Key Differences: Image-to-Text vs Text-to-Image

| Aspect | Image-to-Text (Original) | Text-to-Image (New) |
|--------|-------------------------|---------------------|
| **Query** | Fixed image | Fixed text |
| **Modulation** | Modulate ALL texts | Modulate QUERY text only |
| **Similarity** | Image vs modulated texts | Modulated text vs ALL images |
| **Return** | Top-k captions (strings) | Top-k images (PIL Images) |
| **Visualization** | Text list display | 3x3 image grid |
| **Data tuple** | (img_emb, txt_emb, captions, img_to_txt_map) | (img_emb, txt_emb, captions, txt_to_img_map, raw_imgs) |

## Notes

1. **Modulation Strategy**: The key difference is WHERE modulation happens:
   - Image-to-text: Modulate database (all texts) to match query (image)
   - Text-to-image: Modulate query (text) to match database (all images)

2. **Data Dependency**: The text-to-image version requires `all_raw_image` which must be obtained via `test_set.get_all_raw_image()` after evaluation

3. **Performance**: Both versions use precomputed embeddings for efficiency, avoiding re-encoding during visualization

4. **Extensibility**: The functions follow the same pattern, making it easy to add variations (e.g., different visualization layouts, more angle bins, etc.)
