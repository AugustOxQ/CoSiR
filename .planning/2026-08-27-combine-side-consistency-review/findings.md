# Findings

- `combine_side: "img"` is configured at `configs/model/clip_base.yaml:12`.
- Core training, oracle metrics, snapshots, automatic evaluator, and post-hoc diagnostics consistently condition image and project/use text for that setting; all corresponding `txt` branches are inverse-correct.
- The non-oracle helpers in `src/eval/metrics.py:297-462` are text-hardcoded and are dispatched by `src/eval/pipeline.py:305-322` when `use_oracle=False`. They are not the training snapshot route, which requests `use_oracle=True`.
- The full evidence is in `docs/superpowers/scratch/2026-08-27_codex_combine_side_review.md`.
