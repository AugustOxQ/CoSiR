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

- `r30_queue.sh` (background, started 16:42): waits until a job labelled `r29:*` or `r31:*` (r31 runs r29's runner)
  has been seen in `res-queue` and then none for 60 minutes in a row; gives up when none is seen by 2026-11-14 11:00.
  Then `umap`, `valence`, `plot`, each `res-queue cpu --short --label r30:<step>`. Logs: `<outputs>/logs/`.
- 2026-11-14 02:47 to 02:50: the wait saw the first r31 job at 17:30 and the last at about 01:47; 60 quiet minutes,
  then `umap` (2.3 min), `valence` (0.3 min), `plot` (0.2 min), all exit 0. r31 had ended: its reports were committed
  at 02:18 and "r32 decide" opened. UMAP warnings only (legacy layout default; `n_jobs` set to 1 by `random_state`).

## Results (seed 0, descriptive)

- Shared map (fit on c*): the four test arms land almost where c* is on all three aspects; long DEC on validation also
  keeps c*'s layout (emotion: the contentment edge gets denser). Figures `shared_umap_<aspect>.png`.
- Valence check, k = 2 groups of the emotion head (`valence.md`, `valence.json`). AMI of the two groups, mean of 3 states:

| Split, arm | Valence | Emotion (9) | Style (27) | Genre (10) | Positive share gap |
|---|---:|---:|---:|---:|---:|
| validation, c* | 0.0402 | 0.0392 | 0.0622 | 0.3136 | 0.152 |
| validation, long DEC plain | 0.0459 | 0.0377 | 0.0751 | 0.3255 | 0.137 |
| test, c* | 0.0389 | 0.0423 | 0.0592 | 0.3079 | 0.146 |
| test, the three DEC forms and λ = 0 | 0.0388 to 0.0399 | 0.0403 to 0.0416 | 0.0552 to 0.0605 | 0.3047 to 0.3112 | 0.140 to 0.150 |

- Add-on (exploration, agent, 02:51; `r30_islands.py` -> `islands.json`, about 1 min, no slot): are the two UMAP
  islands of long DEC (r21's own emotion map) the same as the k = 2 groups? Yes: AMI 0.879, sizes 3,363 / 2,789 against
  3,362 / 2,790; the islands are 5.7 UMAP units apart at their closest points. c*'s map has no such gap; its k = 2
  groups and a 2-way k-means of its map agree at AMI 0.572.
- The two groups (long DEC, validation, state 0): group A, 3,362 paintings: portrait 0.39, genre painting 0.22,
  religious painting 0.19 of the genre-labelled; mean positive caption share 0.563, negative 0.330. Group B, 2,790:
  landscape 0.43, abstract 0.17, cityscape 0.15, still life 0.09; positive 0.700, negative 0.182. c*'s k = 2 groups
  have the same make-up (portrait 0.36 / landscape 0.47; positive 0.564 / 0.716).
- Reading (agent): the split is by subject (paintings of people against landscapes, abstracts, cities and objects),
  already present in c*; long DEC pulls the two halves apart into islands. Valence follows the subject split only in
  part (people paintings carry more negative captions); the split's valence AMI is 0.046 against 0.040 for c*, and its
  genre AMI is 0.33.
- 02:52: figure `r30_islands_fig.py` -> `islands_emotion.png` (r21's own emotion UMAPs on validation, c* and long DEC, coloured by genre and by positive caption share); copied to the report assets `docs/reports/assets/2027-01-20_dec_emotion_split/` (with the three shared maps and r21's emotion UMAP) and `docs/user_read/assets/2026-11-14_dec_emotion_split/`.

## Reports (step 9), 2026-11-14

- 02:55 to 03:15: auto report [2027-01-20_dec_emotion_split](../../../docs/reports/auto/percept/2027-01-20_dec_emotion_split.md)
  (Codex `gpt-6.1-sol` medium draft, Claude review and claim review: five brief numbers recomputed from the JSON files,
  all match) and user-read report [2026-11-14_dec_emotion_split](../../../docs/user_read/2026-11-14_dec_emotion_split.md)
  (fidelity check `gpt-6.1-terra` medium: no number errors, two fixes; cold read `gpt-6-luna` medium, the first try made
  no tool call and was rerun with the text inline; one fix wave).
- Correction to the add-on above: the 5.7 UMAP units come from sampled point pairs (Codex read the code), so they are an
  upper bound on the closest distance, not an exact minimum.
- Agent choices: under k-means state 0 alone, validation valence AMI is 0.0430 (c*) against 0.0459 (long DEC), genre
  0.3010 against 0.3255; the reading holds. Raw-row k-means and another tie rule are not checked.
