# ArtELingo retrieval evaluation review

## Findings

- Critical: none in either final review.
- Initial warnings about optimized-Python assertions, direct output truncation, and the stale config metadata comment were fixed. Five focused tests pass.
- Residual warning (Codex): two fixed output paths cannot be replaced as one atomic pair. Both complete splits are built and validated first; each output is then replaced atomically. A failure between the two replacements could leave files from different converter runs. This is accepted for an offline preparation step and does not affect the current generated outputs.
- External cluster configuration at `/root/.claude/skills/cluster-run/configs/dataset/artelingo_cluster.yaml` still points to the flat val file. It is outside the requested worktree/config scope and must be updated before using the grouped split on that cluster.

## Verification

- `python -m unittest discover -s src/test/20260928_artelingo_retrieval_eval -p 'test_*.py' -v`: 5 passed.
- `python -m py_compile` on converter and test: passed.
- Source-to-output comparison: val 2,469 kept / 650 dropped; test 4,975 kept / 1,271 dropped. Every output has five string captions in source order, a unique image and ID, and `image_id == painting`.
- No model import or image read was used for verification.
