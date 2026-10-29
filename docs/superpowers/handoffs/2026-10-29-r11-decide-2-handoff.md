# Handoff: r11 decide 2: choose the retrieval protocol (step 4, decision B), then a short grill and a short spec

## Brief
- Done: decision C on r10 and its wrap-up (main merged and pushed); Qwen store for Impressions; Affection on disk (4 of 5 sources); step 3 facts on retrieval.
- State: c* is the frozen recipe for 11-10; its held per-aspect embeddings exist (5 seeds); no retrieval code yet.
- Next: show the three retrieval protocol options, the user picks one, then a short grill and a short spec.
- Blocked: nothing. The Affection store runs at 19:00 (about 75 min).

> Written 2026-10-29 16:20 by "r11 decide" (cut at 200k context). Loop step reached: after step 3 (a facts brief,
> no deep research), before step 4 (decision B). Direction log entry:
> [2026-10-29](../direction_log.md) (r10's read, C and A).

## 1. Read in this order

1. §2 to §4 below: the decisions and the three options.
2. The direction log entry of 2026-10-29 and "Now".
3. `~/.claude/references/grill.md` and `spec-templates.md`, before steps 5 and 6.
4. Only if needed: literature notes `docs/reports/literature/2026-12-04_conditional_similarity_update.md` (closest
   works) and `2026-12-05_percept_neighbourhood.md` (what a reviewer asks for).

Skills: `grilling` and `domain-modeling` at step 5.

## 2. Decided by the user, not to be reopened

- **Decision C on r10** (2026-10-29, before ours): "I think overall good, we can have a bigger sweep and check later on
  genre, but now the table is okey." Recorded in the direction log.
- **Decision A, r11 to r13** (2026-10-29 about 01:35, in "r10 reports 5"): r11 retrieval (a fast build and test,
  first results, a few small optimization steps; a good result before 11-10); r12 more datasets in clustering and
  retrieval; r13 a large sweep and analysis after 11-10. Data side job beside r11; method and setup sections drafted
  during r11 and r12.
- **The recipe:** c* freezes at 11-10; c* without the caption endpoints is an r13 analysis item (user, after seeing
  the recommendation, 2026-10-29).
- **r11's size:** a medium loop with a short grill and a short spec (user, after seeing the recommendation).
- **Retrieval's query form** (2026-10-25, user, after seeing the recommendation): "image queries with an aspect name
  only". No text queries: c* has no text side.
- **Storage:** all recommendations approved (2026-10-29); done (§6).
- **Affection's extraction** waits for 19:00, so r11 has the GPU first (user, 2026-10-29).
- **PolyArt:** a later choice (user, 2026-10-29).

## 3. Where things stand

- **Step 3 facts** (a Sonnet facts brief, 2026-10-29; read-only):
  - c*'s held per-aspect embeddings: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40/{emotion,style,genre}__s{0..4}.npz`,
    key `img`, (12281, 2048) float16, rows in the order of
    `src/test/20261020_claim_test/outputs/held/held_paintings.json` (no name array in the npz; join by position).
    Norm on disk not checked. Weights: `.../outputs/full/weights/k11=40/{aspect}__s{seed}.pt`.
  - Baselines on held paintings: the plain Qwen image (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/img.npy`,
    61,402 rows, join by `index.json` painting name); the instructed Qwen images
    (`.../qwen3vl_emb_2b_aspect_prompts/{emotion_focus_held,style_word_held,genre_word_held}/`, 12,281 rows, `ids.npz`
    names, held hash matches); CLIP and SigLIP stores beside them. PercepT core gives only 40-topic probabilities on
    held paintings (`src/test/20261020_claim_test/outputs/held/percept/core__k40/P1__s*.npz`), no embedding.
  - Labels: `src/test/20261020_claim_test/ct_held.py:86` `painting_labels(names)`. Emotion: the majority of a
    painting's caption emotions (about 5 captions; 9 classes; ties to the smallest id). Style: 27 classes. Genre: 10
    classes, −1 on 2,332 held paintings (exclude). Per-painting emotion counts can be rebuilt from
    `/data/PDD/artelingo/artelingo_train.json`.
  - No mAP, P@k or aspect-query retrieval code exists. `eg_v5.py:116` is an image-to-own-caption sanity check, and
    `src/eval/factor_gates.py` is old paired recall; neither fits.
  - Literature: PercepT has no retrieval. The closest work on the metric side is CLAY (CVPR 2026): a text keyword
    condition, image-to-image, training-free, mAP against labels (Stanford40 action, location, mood; CLIP-B 43.0 /
    47.0 / 53.0, CLAY 66.0 / 55.4 / 57.9). Also GeneCIS (R@1 among a small gallery), TPIPS (zero-shot
    Qwen3-VL-Embedding is a strong row), CSD (WikiArt style retrieval, mAP@k). No work found that discovers aspects and
    then retrieves by a discovered aspect; a reviewer could build "LLM-named criteria, then CLAY" as a baseline
    (note 12-04).
- **Data side job:** Impressions Qwen store built (1,442 images, 14,947 texts; no class labels for AMI). Affection:
  annotations from the user; 4 of 5 image sources on disk (COCO, Visual Genome, Flickr30k; Emotional Machines from
  figshare, CC BY 4.0); 63,199 images, 392,331 texts; the You et al. (FI) set has no copy (a Baidu link in the WSCNet
  README is untested). Log and code on branch `r11-data`:
  `src/test/20261029_r11_data/20261029_r11_data_log.md`.

## 4. Open points for the next step

a. **Step 4, decision B (key, user first): the retrieval protocol.** Three options from step 3 (agent):
   1. **Label retrieval on held ArtELingo.** Query: a held painting plus an aspect name. Gallery: the other held
      paintings (12,281; genre on the 9,949 labelled). A hit is relevant when it shares the query's label for that
      aspect. c* ranks by its aspect head; baselines: the plain Qwen image (the same ranking for every aspect), the
      instructed Qwen image per aspect (the teacher), CLIP L/14 and SigLIP 2 images, PercepT core's topic vector.
      Metric: mAP (as CLAY) and P@10, 5 seeds, a paired bootstrap over queries. Cheap: all inputs exist.
   2. **Option 1 plus an aspect-switch table.** The same queries, scored with the labels of every aspect: does the
      emotion head retrieve by emotion more than by style? This shows that the aspect name changes the hits, the
      paper's point; no new inputs.
   3. **Option 1 with graded emotion relevance.** Emotion relevance by the overlap of the two paintings' caption
      emotion counts (nDCG), since the majority label is noisy (about 5 captions each). Style and genre stay binary.
   Our recommendation, shown after the user's pick: option 2 (option 1 is the claim; the switch table is nearly free
   and carries the story); option 3 as a later optimization step if emotion mAP is too noisy.
b. **The grill (key):** the claim and its bar (for example, c* above the plain Qwen image on all three aspects, and
   level with or above the instructed image); which baselines are in the main table; k for P@k.
c. **Held data (raise before the step runs):** the first results read held paintings, which is fine. r11's "small
   optimization steps" tune, so they must score on development paintings (the 29,232 development paintings or the
   6,152 validation paintings), never held (`held-data.md`).
d. **Minor:** a literature check on the multi-task storyline (proposed in "r10 reports 5", not decided);
   `deep-research` on Opus at deep depth, beside the build.
e. **Minor:** the FI images for Affection: the user may try the WSCNet Baidu link
   (`https://pan.baidu.com/s/1YLludq6JYQZcCIEXgiMH2Q`, code in the README) or email the Affection authors.

## 5. Code and pitfalls

- New round folder, new module prefix (CoSiR CLAUDE.md, "Pitfalls"); the r10 folders' code is frozen.
- c*'s held npz rows join by position to `held_paintings.json`; the plain Qwen store joins by painting name. Assert
  the name order before any comparison (constitution C6).
- Genre −1 rows: drop them from genre queries and from the genre gallery.
- The metric code is verdict code: ticket review and unit tests (`matt-chain.md`).
- In this shell a bare `cluster` is Graphviz; use `~/.claude/skills/cluster-run/cluster`. node402's reservation ends
  about 2026-10-30 23:30; r10's inputs there are deleted. The user gets a new reservation.

## 6. State at handoff

- Running: `queue_affection_extract.sh` (pid 1545349, nohup, from the worktree `/project/CoSiR-wt-r11-impressions`):
  waits for 19:00, then extracts `affection_local` with Qwen under the GPU lock (about 75 min, heavy); status in
  `src/test/20261029_r11_data/queue_status.txt` of that worktree, output
  `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/`. Keep the worktree until it ends; then merge `r11-data`
  into main.
- Uncommitted: nothing in the project. `claude-config`: Codex ledger rows and the disk ledger, for `sync.sh`.
- On disk over 1 GB: `/data/SSD/affection/` (about 4.5 GB: annotations, the Emotional Machines tarball 2.8 GB and its
  1.3 GB image folder; keep: r12's dataset; the tarball can go after the store is built). r10's kept outputs are in
  the disk ledger.
