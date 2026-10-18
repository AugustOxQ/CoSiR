# Aspect graph follow-up: the residual gains hold on three seeds, the image graph alone gives the best genre head, and heads on prompted inputs come close to the prompted view on its stronger side without passing it

> 2026-10-18 (Amsterdam), local CPU; queue 19:27 to 21:00. The file name uses CoSiR's sequence date. Spec: none
> (small loop, design in chat, approved by the user at 19:15); parent spec
> [aspect graph sweep](../../../superpowers/specs/2026-10-18-aspect-graph-sweep-design.md) (draft). Decision rule:
> none (descriptive). Run log:
> [20261018_aspect_graph_sheet_log.md](../../../../src/test/20261018_aspect_graph_sheet/20261018_aspect_graph_sheet_log.md),
> section "Follow-up 1". Train side only; labels used only to score.

## Results brief

**What we tried.** A short follow-up to stage 1 of the
[aspect graph sheet](2026-12-11_aspect_graph_sheet.md), with the same head and scoring, in 31 CPU runs. (a) Seeds 1
and 2 for the reference heads that had one seed. (b) A genre head on the image graph alone. (c) A head that reads
the aspect's prompted image and caption embeddings instead of the default ones, on the aspect graph and on the best
graph per aspect.

**Outcome against the bar.** No bar (exploration). Target AMI is the mean of the image-side and caption-side AMI for
the aspect's own label (k = 20, 5,000 evaluation paintings), mean ± sd over 3 seeds.
- The residual graphs stay ahead with three seeds: emotion 0.137 ± 0.002 vs 0.118 ± 0.002 for the aspect graph,
  style 0.260 ± 0.003 vs 0.213 ± 0.005.
- Genre: the image graph alone gives 0.428 ± 0.003, the image-weighted fusion 0.420 ± 0.002, the caption graph
  0.402 ± 0.005.
- Prompted inputs raise the residual heads: emotion 0.151 ± 0.003 (default inputs 0.137), style 0.284 ± 0.002
  (0.260). On the aspect graph the change is smaller (emotion +0.004, style +0.009, genre +0.002).
- Against the prompted view clustered directly (mean of both sides): style is above it (0.284 vs 0.265), emotion is
  just below (0.151 vs 0.155), genre is level (0.424 vs 0.424; the image graph alone, default inputs, 0.428).
- On the prompted view's stronger side, no head passes it on the mean of three seeds: emotion captions 0.216 vs
  0.217, style images 0.393 vs 0.395 (one seed 0.397), genre images 0.458 vs 0.496. Style gains on the other side
  (captions 0.175 vs 0.135); emotion stays below there too (images 0.087 vs 0.093).

**What it does not show.** Development data only: one frozen sample, the 2B Qwen model, ArtELingo. Stage 1's scores
on the same 5,000 paintings picked the rows for (c), so the seed sd does not cover that selection. A head on prompted
inputs needs one fixed instruction per aspect at test time (one more Qwen pass per aspect and image), so (c) measures
the gap to the prompted view; it does not test the stage 2 trunk on default inputs. The genre image graph got no
prompted-input head.

**Options for next.**
- a. PercepT core in two versions (already decided as the next step): a build, then a GPU run. It tells where
  PercepT stands on this protocol.
- b. Stage 2, with the graph per aspect updated (emotion and style residual, genre image graph alone) and the trunk
  input as an arm: default inputs, or the prompted inputs per aspect. A build of one to two days, then a night of
  runs.
- c. One more CPU follow-up (about an hour): prompted inputs on the genre image graph, and the effect of coverage
  (the emotion residual graph leaves 57% of paintings without a pair).

**Your reading.** (user, before ours)

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The residual gain over the aspect graph is stable across three seeds, at about ten times the
aspect-graph head's seed sd for emotion and style. Fairly sure, within the selection caveat. For genre, side fusion adds nothing over the image
graph alone (0.428 vs 0.420). Stage 1's gap on the prompted view's stronger side is mostly the input: with prompted
inputs the residual heads come within 0.003 of it (emotion 0.216 vs 0.217, style 0.393 vs 0.395); with default
inputs they stay below (0.194, 0.348). So the head does not sharpen the aspect on that side; for style it carries
the aspect to the other side.
Medium sure. For stage 2: on that side, the best default-input heads are about 0.02 (emotion) and 0.05 (style) below
the prompted view.

**Our recommendation.** PercepT core next, as decided. In the stage 2 spec, carry emotion residual λ = 1, style
residual λ = 0.5 and the genre image graph alone, and make the trunk input (default or prompted) an arm. Our view.

</details>

## What ran

- **Stages and types:** one exploration step (no bar), a small loop with its design in chat.
- **Data and seeds:** the frozen sample of stage 1 (19,889 paintings, train side): graphs and training on 13,400 fit
  paintings, early stopping on 1,489 dev paintings, scores on the fixed 5,000 evaluation paintings. Head seeds as
  below; AMI with 3 k-means seeds per view.
- **Runs (31):** (a) seeds 1 and 2 for the aspect graph (R0) and full pruning (R1) of emotion and style, and for R1
  of genre (10); (b) genre on the image graph alone (R0o), seeds 0 to 2 (3); (c) prompted inputs on R0 and the best
  row per aspect (emotion residual λ = 1, style residual λ = 0.5, genre image 0.75 + caption 0.25), seeds 0 to 2
  (18). Row definitions: [ags_rows.md](../../../../src/test/20261018_aspect_graph_sheet/ags_rows.md).
- **Prompted inputs:** the aspect's prompted image embedding (emotion_focus, style_word, genre_word) and the painting
  mean of its prompted caption embeddings, the same views that the graphs use. Default inputs as in stage 1.
- **Code:** commit `02d0466`. Codex (`gpt-6.1-sol`, medium) added the input option to `ags_head.py` (the default
  config is unchanged and was checked equal to stage 1's), `ags_fu.py` (run list, scoring, combined sheet) and
  `ags_fu_queue.sh`. The controller read the diff and ran a genre smoke from main. Local CPU, three processes.

## Results

Target AMI with mean ± population sd over 3 head seeds; image and caption columns give the aspect's own label, mean
over seeds. The prompted view is stronger on captions for emotion and on images for style and genre. Rows with
"stage 1" took all three seeds from stage 1.

| Aspect | Graph | Input | Image side | Caption side | Target AMI |
|---|---|---|---:|---:|---|
| emotion | aspect graph (R0) | default | 0.085 | 0.151 | 0.118 ± 0.002 |
| emotion | full pruning (R1) | default | 0.089 | 0.157 | 0.123 ± 0.002 |
| emotion | residual λ = 1 (stage 1) | default | 0.081 | 0.194 | 0.137 ± 0.002 |
| emotion | aspect graph (R0) | prompted | 0.085 | 0.159 | 0.122 ± 0.003 |
| emotion | residual λ = 1 | prompted | 0.087 | 0.216 | **0.151 ± 0.003** |
| emotion | prompted view, no training | — | 0.093 | 0.217 | 0.155 |
| emotion | default view, no training | — | 0.070 | 0.107 | 0.089 |
| style | aspect graph (R0) | default | 0.267 | 0.160 | 0.213 ± 0.005 |
| style | full pruning (R1) | default | 0.283 | 0.160 | 0.221 ± 0.003 |
| style | residual λ = 0.5 (stage 1) | default | 0.348 | 0.172 | 0.260 ± 0.003 |
| style | aspect graph (R0) | prompted | 0.289 | 0.156 | 0.222 ± 0.004 |
| style | residual λ = 0.5 | prompted | 0.393 | 0.175 | **0.284 ± 0.002** |
| style | prompted view, no training | — | 0.395 | 0.135 | 0.265 |
| style | default view, no training | — | 0.222 | 0.120 | 0.171 |
| genre | caption graph (R0, stage 1) | default | 0.435 | 0.369 | 0.402 ± 0.005 |
| genre | full pruning (R1) | default | 0.437 | 0.370 | 0.404 ± 0.002 |
| genre | image 0.75 + caption 0.25 (stage 1) | default | 0.454 | 0.386 | 0.420 ± 0.002 |
| genre | image graph alone (R0o) | default | 0.458 | 0.397 | **0.428 ± 0.003** |
| genre | caption graph (R0) | prompted | 0.437 | 0.371 | 0.404 ± 0.003 |
| genre | image 0.75 + caption 0.25 | prompted | 0.458 | 0.390 | 0.424 ± 0.004 |
| genre | prompted view, no training | — | 0.496 | 0.353 | 0.424 |
| genre | default view, no training | — | 0.449 | 0.338 | 0.393 |

The views' sd over k-means seeds is at most 0.007. All labels and sides:
[ags_fu_tables.md](../../../../src/test/20261018_aspect_graph_sheet/ags_fu_tables.md).

**Verdict under the rule:** none (descriptive).

## Analysis

- **The stage 1 order holds with three seeds.** The aspect-graph heads moved little from their one-seed values
  (emotion 0.120 to 0.118, style 0.218 to 0.213). Full pruning stays between the aspect graph and the residual
  graph. The residual gaps (emotion +0.019, style +0.047) are about ten times the aspect-graph head's seed sd
  (0.002 and 0.005). These ratios describe the spread; they are not a test.
- **For genre, the image side alone is enough.** The image graph alone beats the image-weighted fusion by 0.008
  (about three times its seed sd of 0.003), on both sides (image 0.458 vs 0.454, captions 0.397 vs 0.386). Stage 1 gave the two graphs
  the same lift (5.35); the head separates them.
- **Prompted inputs help most where the graph is residual.** On the residual rows the head gains 0.014 (emotion) and
  0.023 (style) over default inputs; on the aspect graph 0.004 and 0.009. Our reading: on the aspect graph the
  pairs already follow the prompted view, so the input adds little; on the residual graph the pairs and the input
  carry different parts of the aspect. Not tested.
- **The gain over the prompted view comes from the other side.** With prompted inputs, the residual heads come
  within 0.003 of the prompted view on its stronger side (emotion captions 0.216 vs 0.217, style images 0.393 vs
  0.395) and stay below it on the mean of three seeds (one style seed reaches 0.397). Their gain is on the weaker side: style captions 0.175 vs 0.135, emotion images 0.087 vs 0.093 (still below).
  Genre's image side stays below the prompted view with both inputs (0.458 vs 0.496).
- **Heads on prompted inputs train longer.** Their best dev loss came at epoch 3 to 12 (emotion residual 7 to 12);
  every default-input head of stage 1 and of this follow-up had it at epoch 1 to 4.
- **The cost of the other labels.** The prompted-input emotion head keeps less genre on the image side than the
  prompted image itself (0.327 vs 0.410), as the default-input residual head did in stage 1 (0.304). The emotion
  residual pairs cross genre (stage 1, lift 0.69), and the head follows them.

## Deviations

- None from the design in chat. Agent defaults in the code: outputs in a new folder (`outputs_fu/`), output names with
  `__prompted` for prompted inputs, a combined sheet that reads stage 1's scores read-only.
- The stage 1 run log's two slips (39 head runs; 16.8%) are corrected in the log, with a note.

## Files and storage

- **Outputs:** `src/test/20261018_aspect_graph_sheet/`: `ags_fu.py`, `ags_fu_queue.sh`, `ags_fu_tables.md`, the run
  log (in git); `outputs_fu/` (31 head output pairs), `ags_fu_results.json`, `queue_fu_*.log` (gitignored).
- **Storage:** `outputs_fu/` (1.2 GB) waits for the storage summary at the end of the loop, beside stage 1's
  `outputs/` (1.1 GB); see the disk ledger rows dated 2026-10-18.
