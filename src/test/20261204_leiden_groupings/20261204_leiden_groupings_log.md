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

## 2026-10-14 02:00 onward: the user's reading and the follow-ups

The user's reading (01:55): Leiden ties, so take Leiden for coherence; H1 was expected to be bad. Do both: rebuild AFF
on Leiden, retrain H1 (direction log entry "2026-10-14 · Leiden groupings side check").

**H1 retrained** (`lg_h1r.py`, Sonnet builder; 22 s on GPU under the lock): scorer-train paintings only, 20 communities,
AMI 0.49 with the old mapped labels. Deviations in `results/h1r.json` (plateau stop on scorer-train recall, ran all
200 epochs; GoEmotions from the stored per-row `affect_probs`). Screen (`lg_screen.py heads`, `eval`; old eval kept as
`lg_eval_v1.*`): H1r alone −0.27 [−0.48, −0.07], minus reference −0.63 [−0.94, −0.32]; the cross-fit again gave its
conditioned term weight 0 in both folds. Fails the bar; no rebuild (the cheap rule).

**AFF rebuilt on Leiden** (`lg_rebuild.py`, Sonnet builder; frozen r9 worktree `/project/CoSiR-lg-r9` at b84db1e):
image and caption k-means replaced by Leiden at 64 everywhere (B included); picks and τ refit; CLIP's content graph
reused. Run 02:07 to 02:54 (A3 749 s on GPU, 3.93 GiB). The score step's summary crashed (point_ci returns a dict);
fixed, and the result recomputed from the saved arrays (`--step result`). τ_0..3 = 0.0001, 0.142, 0.342, 0.611.

```
AFF seed-42 dev, R@1 points: bar margin Leiden +0.50 [+0.30, +0.71] | k-means +0.70 [+0.46, +0.94] | paired -0.20 [-0.44, +0.05];
AFF R@1 Leiden 18.58, k-means 19.14, paired -0.56 [-0.80, -0.30]; Leiden over B'(A0) +0.50 [+0.30, +0.71],
over its matched control +0.53 [+0.34, +0.72]
```

The k-means reference reproduces AFF's stated development margin (+0.70 [+0.46, +0.94]).

## 2026-10-14 close: the user's reading and storage

The user's reading of the rebuild and the decision are in the direction log entry ("Meaning (C), rebuild", "Decided"):
kept as a record, AFF's recipe unchanged, Leiden parked. Storage (user approved): deleted the rebuild's intermediates
(heads, banks, posteriors, graph, partitions, about 400 MB) and the builder's check runs (`_check`, `_check2`, 37 MB);
`results/rebuild/` keeps the checkpoint, per-episode arrays and JSON records (1.2 MB); `--step result` still works.
The frozen worktree `/project/CoSiR-lg-r9` (82 MB) removed; recreate at b84db1e if needed. Nothing of 1 GB or more.
