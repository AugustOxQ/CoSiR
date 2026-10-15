# 20261031 pseudo-partitions (E2) log

**Problem.** Method A needs a label-free training episode bank. We needed three pseudo-partitions (affect, image, caption; k=64) over scorer-train rows and three banks (AIC, AI, IC) of 65,536 episodes.

**Steps.**
1. TDD for `kmeans_partition` and `build_episode_bank` (`src/train/pseudo_partitions.py`, commit fa703f7).
2. `build_partitions.py` asserts row alignment of the cached arrays with `artelingo_splits().scorer_train`, copies affect/image/local_groups/graph, builds the caption partition and the banks, validates 1,000 episodes per set, and computes the AMI table (commit 632eb68). Smoke run: 14.6 s.
3. Full run (controller, background): 637.5 s, exit 0.
4. Report with an AMI heatmap and a shuffled-label null.

**Results.** All banks validated. AMI: affect/emotion 0.2035, image/style 0.3176, image/genre 0.3970, caption/genre 0.1613; shuffled-label null about 0.00001. Details in `docs/reports/auto/v2/2026-10-31_pseudo_partitions.md`.

**Issues.**
- Affect/emotion AMI is 0.2035 against a recorded 0.1960: the earlier prepare kept the catch-all "something else" emotion. With it kept we get 0.19601969782711, the record to 13 digits.
- The image partition carries more genre information (0.397) than style (0.318), so the AI "held-out genre" bank is not blind to genre. The report recommends E3 wording for K8. No code or rule was changed.
