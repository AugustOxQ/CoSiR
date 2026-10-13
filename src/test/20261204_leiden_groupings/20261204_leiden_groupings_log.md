# 20261204 Leiden groupings: side check log

Exploratory side check outside the research loop (user, 2026-10-14). Seed 42 development episodes only; decides
nothing about AFF (no more AFF tuning, user). Handoff: `docs/superpowers/handoffs/2026-10-14-leiden-groupings-handoff.md`.

## Decisions (2026-10-14, 01:10 to 01:16)

| Decision | Who decided |
|---|---|
| Check (1): screen Leiden on the CLIP graphs at about 64 groups (1′), then the full AFF rebuild if it passes | user |
| Check (2): screen the attention-H1 communities, then the full rebuild if it passes | user |
| H1 labels: map the stored `train_community_post` labels to scorer-train rows by painting name (development only) | user |
| H1 slot: H1 alone, replacing all three groupings, every aspect told to it ("exactly follow" buddy vs PercepT); the other three slots dropped | user, after seeing the recommendation |
| "Passes": reader margin at least A0's in the same run | agent default |
| Resolution search for 1′: r = 1.0, doubling, then up to 8 bisections; the r nearest 64 groups (ties to the smaller r) | agent default |
| Cheap build: one script, the A0 reproduction check as the only test | user ("minimal to satisfy a cheap check") |

## Build

`lg_screen.py` (commit 1b471a4) reuses round 3's told-oracle screen (`20261116_grouping_step1_style/run_step1.py`)
by import. Stages `group`, `heads`, `eval`; outputs in `results/`. Checks built in:

- At r = 1.0 the image graph reproduces run_step1's `image_leiden` partition (SHA-256).
- A0 reproduces run_step1's stated A0 margins (told +1.64 [1.37, 1.92], reader +0.35 [0.15, 0.57]).
- In the H1 arm, reader and told terms are equal (one grouping).

## Run

Started 01:18, finished 01:42 (group 749 s, heads 287 s, eval 403 s). All built-in checks passed.

| Grouping | Resolution | Groups | Search |
|---|---|---|---|
| image Leiden | 5.047 | 64 | 12 resolutions, 9–17 s each |
| caption Leiden | 2.375 | 64 | 7 resolutions, 41–74 s each (graph 134 s) |
| H1 (`train_community_post`) | — | 19 | all 36,518 scorer-train paintings found in the snapshot |

`results/lg_eval.txt`, R@1 margins over the matched control, 12,288 seed-42 development episodes:

```
A0:  told +1.64 [+1.37, +1.92]  reader +0.35 [+0.15, +0.57]  bar +0.31 [+0.10, +0.53] (B_prime)  pick 54.7
A3r: told +1.70 [+1.41, +2.01]  reader +0.22 [+0.06, +0.38]  bar +0.22 [+0.06, +0.38] (counterpart)  pick 53.8
     minus A0: told +0.06 [-0.19, +0.31] reader -0.14 [-0.36, +0.09]
H1:  told -0.46 [-0.63, -0.28]  reader -0.46 [-0.63, -0.28]  bar -0.46 [-0.63, -0.28] (counterpart)  pick 100.0
     minus A0: told -2.10 [-2.43, -1.78] reader -0.81 [-1.09, -0.53]
```

- **Verdict against the pass bar:** neither passes (reader +0.22 and −0.46 against +0.35). No full AFF rebuild runs,
  as the user's routes say ("screen, then rebuild if promising").
- **H1 diagnostic:** the nested cross-fit picked weight 0 for H1's conditioned term in both folds (`T_picks` cells
  [0, 0]), so its fused score equals B. Its condition-free version gained +0.51, +0.48 and +0.38 over B on the three
  aspect pairs; that is the whole negative margin.

## Reports

- Auto report: `docs/reports/auto/v2/2026-12-04_leiden_groupings.md`.
- User-read report: `docs/user_read/2026-10-14_leiden_groupings.md`.

## Storage

`results/` holds 44 MB (`lg_heads.npz` 38 MB, `lg_group.npz` 7 MB, JSON and logs). Kept as the run's outputs; nothing
of 1 GB or more.
