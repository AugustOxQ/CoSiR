# ArtELingo retrieval evaluation split

- Read the existing flat val and test JSON files independently; keep training JSON and its converter unchanged.
- Group by painting in source order, drop paintings with fewer than five captions, use exactly the first five captions, and emit one row per painting with the shared image, art_style, painting, and image_id equal to painting.
- Treat inconsistent image or art_style within a painting and duplicate output image paths as errors.
- Point periodic training evaluation at the grouped val output; reserve grouped test for final evaluation.
- Verify output structure and unique image/image_id values by loading both output files; report kept and dropped counts and exact config diff.
- Commit source changes and archive the task according to the supplied AGENTS instructions.
