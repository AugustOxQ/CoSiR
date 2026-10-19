# 2026-10-19 · Image-side ceiling per aspect (r10 decide 10, step 3)

**Question.** Image-side emotion AMI sits at 0.07 to 0.09 for every method so far. How high can it go from these
image features at all? The user chose this check before the stage 2 grill (decision A, option b, 2026-10-19 17:30).

**Method.** `ec_ceiling.py`, CPU, 8 threads. A logistic probe learns each aspect's painting label (emotion: the
majority of 9 classes; style and genre: the painting's label) from standardized image features on the 13,400 fit
paintings. The 1,489 dev paintings pick C by accuracy. On the 5,000 evaluation paintings, the probe's class
probabilities are clustered at k = 20 (MiniBatchKMeans, seeds 0 to 2, the image-side AMI protocol of `pc_score.py`),
and the argmax is scored directly. "Raw" is the same k-means on the feature itself. Labels train the probe here: a
diagnostic only, never part of a method. Development data only; no held data.

**Grid.** Run 1, C in {0.01, 0.1, 1, 10}, picked C = 0.01 (its lowest) in 9 of 9 cases
(`outputs/ec_results_grid1.json`, `ec_ceiling_grid1.log`). Run 2, C in {0.0003, 0.001, 0.003, 0.01}
(`outputs/ec_results.json`, `ec_ceiling.log`): an interior C for 7 of 9; emotion on Qwen default and Qwen prompted
picked 0.0003, but dev accuracy is flat there (0.494 against 0.493, 0.503 against 0.501).

## Results (run 2)

Image-side AMI, k = 20, mean over 3 k-means seeds (sd at most 0.008). Genre: 4,053 labelled paintings.

| Aspect | Feature | Raw | Probe, k = 20 | Probe, argmax | Accuracy (majority) | C |
|---|---|---:|---:|---:|---:|---:|
| Emotion | Qwen default | 0.070 | 0.121 | 0.123 | 0.480 (0.352) | 0.0003 |
| Emotion | CLIP | 0.065 | 0.105 | 0.107 | 0.463 (0.352) | 0.003 |
| Emotion | Qwen emotion_focus | 0.093 | 0.130 | 0.132 | 0.487 (0.352) | 0.0003 |
| Style | Qwen default | 0.222 | 0.527 | 0.534 | 0.640 (0.151) | 0.001 |
| Style | CLIP | 0.300 | 0.483 | 0.496 | 0.613 (0.151) | 0.01 |
| Style | Qwen style_word | 0.395 | 0.569 | 0.576 | 0.679 (0.151) | 0.01 |
| Genre | Qwen default | 0.449 | 0.657 | 0.691 | 0.849 (0.218) | 0.001 |
| Genre | CLIP | 0.412 | 0.605 | 0.634 | 0.814 (0.218) | 0.003 |
| Genre | Qwen genre_word | 0.496 | 0.658 | 0.690 | 0.848 (0.218) | 0.001 |

Run 1 gave the same order; emotion was lower there (0.101, 0.103, 0.116) because C = 0.01 was too high.

## Reading (ours)

Our network on default Qwen images (0.081 / 0.348 / 0.458) closes this share of the gap from raw Qwen default
images to the Qwen default probe: emotion 21% (0.011 of 0.051), style 41% (0.126 of 0.305), genre 4% (0.009 of
0.208).

- Emotion has room, but little in absolute terms: about +0.04 to the probe. The images carry a weak emotion signal
  (probe accuracy 0.48 against a majority rate of 0.35).
- Genre has the most unused room: our heads stay near raw image clustering, while a linear readout reaches 0.66.
- A linear probe is not an upper bound: a nonlinear probe or a tuned encoder may go higher (not checked).

## Step 3 note: candidates for stage 2

1. Pick and train for the image side: stage 1 picked heads on the mean of both sides; the best image-side emotion
   head so far is 0.089 (row 8), not the stage 1 pick (0.081).
2. The aspect-instructed image as the trunk's input at test time: raw emotion 0.093 already passes both bars with no
   training, and its probe reaches 0.130 (a claim question: the user was unsure on 2026-10-18).
3. For genre, a training signal beyond the image's own neighbours: the image-graph head (0.458) adds 0.009 over raw
   Qwen images; the other genre graphs scored lower on the image side (0.435 to 0.454, rows 7 to 9).

Storage: about 200 KB (script, logs, two JSONs); nothing over 100 MB.
