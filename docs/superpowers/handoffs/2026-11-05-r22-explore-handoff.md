# Handoff: r22 explore: design and run the text and cross-modal retrieval exploration (core plus p, q, r), then report it

## Brief
- Done: decision A for r22. A small exploration first, core plus SemArt, the MMArt perspective table and word queries.
- State: no text retrieval result exists yet. c*'s caption side is saved in the weights. The container GPU failed its check before the restart.
- Next: check the GPU, run or collect the literature check, then run the exploration on development data.
- Blocked: nothing after the restart.

> Written 2026-11-05 16:55 by tab "r22 decide". Loop step reached: after decision A, small loop, design in chat next. Direction log
> entry: [New scores read](../direction_log.md) ("Decided in the r22 decide chat").

## 1. Read in this order

1. This file, sections 2 to 5.
2. The direction log's "Now".
3. `src/test/20261029_retrieval/rt_metrics.py` and `rt_data.py`: mAP and the validation and held lists to reuse.

Skills: none.

## 2. Decided by the user, not to be reopened

All 2026-11-05, in the r22 decide chat.

- **Directions:** text to image, image to text, text to text (user).
- **Hit rule:** both the aspect label and the known pair, a painting and its own caption, scored with R@K (user).
- **Text:** ArtELingo captions and MMArt perspective captions (user). ArtELingo first, since its text exists (agent default).
- **Paper role:** a main claim, abstract unchanged (user). Our condition: the exploration must earn it; the fallback is supplementary.
- **Size:** a small exploration first, then decide (user). So type exploration, small loop, one chat, development data only.
- **Scope:** the core plus p, q and r below (user, after seeing the recommendation).
- **Pair-hit limit:** if c* wins on label hits and falls below plain text on pair hits, the user decides after the results whether the paper
  states it as a limit (user). Do not treat it as decided.
- **Start:** after the container restart (user).

## 3. Where things stand

**The core, Qwen, ArtELingo validation, one seed:**

- All three directions, both hit rules, three aspects.
- Rows: c*'s caption side; plain Qwen (the user's "zero-shot model"); the teacher's text; GoEmotions RoBERTa on text to text for emotion.
- A free variant: captions through c*'s image side, since Qwen puts text and images in one space.
- Leak check: the share of captions that name their label, and the scores without those captions.

**The extras:**

- p. SemArt text-to-painting benchmark, on SemArt's own validation split (1,069 paintings by the 2018 paper's split; not checked here).
- q. MMArt, 4 perspectives by 3 heads, on the 5,769 validation paintings: does each perspective do best in its matching head?
- r. Word queries ("sadness", "portrait"): text retrieval with no caption leak; a new short input for the caption side.

Not in scope now: s (Affection captions with the photo model, after the core), t (caption-side retrain on single captions, image side
frozen, only if the rule triggers), u (c* score joined with plain score, weight picked on validation, only if pair hits are weak).

**What changes direction, set before we look (shown to the user, who said yes to the scope):**

1. c* is above both main comparators (plain Qwen and the teacher's text) on label hits in most cells: a medium claim loop follows, on held
   ArtELingo and the MMArt test rows.
2. c* is level or below: try t, then read again.
3. It is still not above: the work goes to the supplementary and the paper keeps image-to-image retrieval.

**Timing (our estimate):** exploration read by 11-06, claim test read by about 11-10, paper text 11-11 to 11-16. Deadlines: abstract 11-10
(unchanged by this work), paper 11-16, supplementary 11-23.

**Data, checked 2026-11-05:**

| What | Where (`/data/SSD2/pre_extract/`) | Notes |
|---|---|---|
| Plain Qwen, ArtELingo | `artelingo_full/qwen3vl_emb_2b/` `img.npy` (61,402), `txt.npy` (308,723), float16 | `index.json` key `texts`: sample_id, painting, img; held paintings are in it (61,744 captions) |
| Teacher's text, emotion | `artelingo_full/qwen3vl_emb_2b_aspect_prompts/emotion_focus_s2txt/` | 146,980: train 116,108, val 30,872; none on held |
| Teacher's text, style and genre | same folder, `style_word/`, `genre_word/` | 99,999 captions; whether they cover validation is not checked; about 295 captions per second |
| RoBERTa | `artelingo_full/roberta_go_emotions_s2/txt.npy` (146,980, 768) | train and validation; none on held |
| SemArt | `semart/qwen3vl_emb_2b/txt.npy` (21,382) | also CLIP B/32 and SigLIP 2 text |
| Affection | `affection_local/qwen3vl_emb_2b/txt.npy` (526,749) | for option s |
| MMArt text | none on any encoder | raw: `/data/SSD/mmart/raw/data/data.parquet` (`e_emotional`, `e_formal`, `e_narrative`, `e_historical`, `e_unified`); splits: `/data/SSD/mmart/mmart_index.parquet` (train 40,155, val 5,769, held 11,528, new 16,782) |

**c* weights:** `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/<aspect>__s<seed>.pt`, keys `img.trunk.0.*`,
`img.output.*`, `cap.trunk.0.*`, `cap.output.*` (emotion seed 0 checked). The forward pass is `TrunkHead` in
`src/test/20261019_model_sheet/ms_train.py`. The other encoders are under `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/`,
not checked for the caption side.

**Literature check:** launched at 16:42 from the r22 decide chat, Opus `high`, deep depth, since our notes do not cover the area. A restart
ends it. If `docs/reports/literature/2027-01-08_text_cross_modal_retrieval.md` exists, read it. If not, launch it again with the brief in
section 7. Do not wait for it: new comparators only add rows (user said start after the restart, and the rows are additive).

## 4. Open points for the next step

Minor, agent defaults unless the results need the user:

- The read-out for text and cross-modal cells, picked on validation as in r11. Centering per modality may matter.
- Which perspective matches which head in option q. `e_emotional` to emotion and `e_formal` to style are clear; show the whole 4 by 3 table,
  so no pick is needed.
- The order of the other open items (r20 figure, paper chat, genre baseline row, r21, item 17, the note to the collaborator and supervisor)
  waits until after the exploration is read.

## 5. Code and pitfalls

- New round folder, `src/test/20261105_text_retrieval/`, with a new module prefix, `tx_` (taken: `ns_`, `tg_`, `rt_`, `mt_`, `bs_`, `rs_`).
- Queue GPU text extraction with `res-queue gpu --short --label r22:text`; check `res-queue status` first.
- Validation only: no tuning, read-out pick or selection reads held rows. An exploration result is never quoted as a result.
- **Label on each side:** an ArtELingo caption has its own emotion label; a painting's emotion is the plurality of its captions; style and genre
  belong to the painting; genre has missing entries.
- **Pair hits in text to text:** exclude the query caption itself.
- **Caption mean:** the direction log says the caption side trained on each painting's caption mean; a single caption is a new input. Not
  checked in the code. Read `ab_train` and `rs_train`'s inputs and record what is true.
- **Content:** c*'s emotion and style graphs subtract content, so pair hits are the harder test.
- **MMArt captions are generated:** the README names Qwen3-VL-8B-Instruct for the historical perspective, so check how the others were
  made. A caption from a Qwen model may favour Qwen encoders. The one emotional caption read names its label ("contentment").
- **Comparators:** our notes describe TPIPS as image to image; CLAY uses text only for its condition. Neither has a known caption query path.
- `masked_label` and `masked_broad` hold `default` and `emotion_focus` subfolders with masked caption embeddings, from
  `src/test/20261017_emotion_word_masking/`; their row set is not checked.
- Reports use the next sequence date; literature note 2027-01-08 is reserved for the check above. Report rows go in `reports_sum.md`.
- Bulk outputs go to `/data/SSD2/cosir_r22/`, smoke runs to the scratchpad.

## 6. State at handoff

- **GPU:** `nvidia-smi` failed at 16:50 with `Failed to initialize NVML: Unknown Error`, kernel and library both 595.99.02, so a host
  `systemctl daemon-reload` is the likely cause and a restart should fix it. Run the GPU check of `container.md` first.
- Running: the literature check only (see section 3). No queue job, no Codex job.
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: nothing new.

## 7. The literature brief, to launch again if needed

`deep-research`, model `opus`, effort `high`, background, deep depth. Save its report as
`docs/reports/literature/2027-01-08_text_cross_modal_retrieval.md` and add its index row.

Read first: in `docs/reports/literature/`, the notes `2026-10-21_cvpr_literature_review.md`, `2026-11-28_design_l_novelty.md`,
`2026-12-04_conditional_similarity_update.md`, `2026-11-25_held_claim_check.md`, `2026-12-05_percept_neighbourhood.md`, and grep
`2027-01-03_paper_related_work_sources.md` for cross-modal, C-STS and SemArt.

Context for the agent: c* sits on a frozen multimodal encoder, uses no labels, and has one trunk and three aspect heads (emotion, style, genre)
on the image side and the same on the caption side, trained on aspect neighbour graphs from instruction-conditioned embeddings. Retrieval so
far is image to image with label hits. Test sets: ArtELingo, MMArt, possibly SemArt and Affection.

Questions:

1. How does the field score aspect-specific or attribute-conditional retrieval across text and image, and text to text? Hit definitions, metrics
   (mAP, R@K, MedR, nDCG) and any standard protocol. Include C-STS and follow-ups, and cross-modal hashing's label-sharing mAP.
2. Which comparators would a CVPR reviewer expect, for cross-modal and for text to text? Consider CLIP, SigLIP, instruction-following
   embedders (Qwen3-VL-Embedding, GME, VLM2Vec), instruction-tuned text embedders (INSTRUCTOR), emotion text encoders (GoEmotions RoBERTa) and
   art-specific models. Public weights? Runnable in hours without training?
3. What published cross-modal numbers exist on paintings and affect: SemArt Text2Art and successors, ArtEmis or ArtELingo retrieval, an MMArt
   baseline, Affection? Best MedR or R@K with split and gallery size; could our protocol be comparable?
4. How do papers treat captions that name the label in text-query retrieval? Is masking standard? Class-name prompts against real captions?
5. Closest prior work, 2024 to 2026: one model with several aspect spaces, scored across modalities, without labels. The strongest comparator,
   and anything that weakens a novelty claim.

End with a ranked list of comparators to run and metrics to report, each with one line of why. No advice on our model's design.
