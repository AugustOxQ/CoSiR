# Requirements

Create `src/test/20260923_artelingo_buddy_analysis/run_single_modality_pilot.py`.

- Dynamically import the sibling pipeline and affect-pilot scripts.
- Reuse the named loaders, metric helpers, affect encoder, and normalizer.
- Build image-only, text-only, and affect-only mutual-kNN graphs with `pipeline.K`,
  repair minimum degree and connectivity using the existing helpers, and use Leiden
  seed 42.
- Evaluate full-graph emotion and genre-subset metrics, then write the specified
  four-row Markdown report at runtime.
- Do not execute the GPU workload during development.
