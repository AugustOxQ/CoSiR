# r30: a shared UMAP map and an emotion valence check for r21 (DEC on c*'s heads)

A small loop (exploration) from saved r21 outputs, seed 0. Decided in the "r30 decide" chat, 2026-11-13. Outputs:
`/data/SSD2/cosir_r21/20261111_dec_heads/r30_followup/`.

## Why

- User's reading of r21 (2026-11-13 16:30): in the UMAPs emotion changes the most, but only between c* and long DEC
  plain (50 DEC epochs) on validation; the two big emotion groups long DEC forms may be positive against negative
  emotions (user's idea).
- r21's UMAPs fit one map per panel, so panels do not line up. The user said yes to one map fitted on c* with each
  arm projected into it (after seeing ours).
- Both start after r31's runs end (user, 2026-11-13 16:30).

## Design (agent, shown to the user in chat 16:45)

- `r30_followup.py umap`: per aspect and split, `umap.UMAP(**r21 SETTINGS)` fitted on c*'s rows; every other arm
  placed with `.transform()`. `plot`: r21's layout and colours, the same axes within a row.
- `r30_followup.py valence`: emotion head of every arm; unit rows; `ns_metrics.partitions(x, 2, 'artelingo')`
  (k-means k = 2, states 0 to 2; group tables use state 0). Caption votes from `rt_data.emotion_dist`: positive =
  amusement, awe, contentment, excitement; negative = anger, disgust, fear, sadness; other = something else. Valence
  label = the largest of the three shares; a tie of the top two (within 1e-12) is excluded from the valence AMI. AMI
  of the 2 groups against valence, emotion (9), style (27), genre (10, labelled rows); the gap in mean positive share.
- What would support the user's guess: long DEC's 2-group split matches valence clearly better than c*'s split on
  validation, and better than it matches style or genre. One seed, no intervals: descriptive.

## Build, 2026-11-13

- 16:37 to 16:40: Codex `gpt-6.1-terra` medium wrote `r30_followup.py` and `test_r30_followup.py`; smoke with
  `--limit 600` in the chat's scratchpad (about 1 min); 2 tests pass.
- 16:44: controller review of the diff. One fix: valence ties used exact float equality; equal caption counts can
  differ in the last bit after summing fractions, so ties now use a 1e-12 tolerance (a 4-to-4 case checked). The
  palette equals r21's. Smoke figure checked: panels line up, legend readable.
- Smoke note (600 test rows, not a result): c*'s own 2-group split already puts 82% positive paintings in one group,
  so c*'s split is the needed baseline.
- Choices that could change a number (Codex hand-back, accepted): UMAP gets r21's loaded rows unchanged (as r21's
  figures); k-means gets unit rows (as r21's scoring); summary AMIs are means over 3 states, the JSON keeps each state;
  ties count in the group share denominators and are reported apart.

## Run

- `r30_queue.sh` (background, started 16:47): waits until a job labelled `r29:*` or `r31:*` (r31 runs r29's runner)
  has been seen in `res-queue` and then none for 60 minutes in a row; gives up when none is seen by 2026-11-14 11:00.
  Then `umap`, `valence`, `plot`, each `res-queue cpu --short --label r30:<step>`. Logs: `<outputs>/logs/`.
