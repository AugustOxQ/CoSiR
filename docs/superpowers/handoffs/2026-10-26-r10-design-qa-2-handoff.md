# Handoff: r10 design Q&A 2: go on answering the user's design questions and adjusting the two framework figures

## Brief
- Done: two framework figures of the current model, low detail and detailed (formulas, λ example, InfoNCE matrix), with build scripts, in `docs/reports/assets/2026-10-26_condition_free_line/`; questions (a) to (g) on the heads and losses, the mean caption, the residual graphs and the stopping rule answered from the code.
- State: results unchanged (held emotion: Qwen 0.1029 against the PercepT bar 0.0874); the figures are in the repo but not yet in the stage report or on the page.
- Next: further questions and figure adjustments as the user asks; three notes wait for the decide 23 chat (section 4).
- Blocked: nothing.

> Written 2026-10-26 17:05 by "r10 design Q&A". Loop step reached: side chat; step 2 (decision A) stays open in the decide 23 chat. Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The first Q&A handoff, `2026-10-26-r10-design-qa-handoff.md`, section 1: the ordered list of sources (stage report design section, glossary, specs, code lines). It still holds.
3. The two figures (view with Read): `docs/reports/assets/2026-10-26_condition_free_line/framework.png` and `framework_detail.png`, and their scripts `build_framework_figure.py` (SVG helpers, the first figure) and `build_framework_detail_figure.py` (imports the first; formulas, matrix, λ example). Run either from its folder with the CoSiR python; `rsvg-convert` renders the PNG at 1.6x.
4. Code read in the first chat, when a question needs the exact computation: `ms_train.py` (`TrunkSide` 31, `symmetric` 79, `distill_loss` 102, `head_loss` 119, the stopping loop 384 to 417, `save_output` 246), `ct2_graphs.py` (`_partition` 70 to 97), `ags_graphs.py` (`top` 59, `reciprocal` 48, `similarity_sd` 142), `itp_train.py` line 17 (settings), `rb_rows.py` lines 8 to 13 (recipes).

Skills: none.

## 2. Decided by the user, not to be reopened

- This chat explains and draws; every research decision goes to the "r10 decide 23" tab (user, 2026-10-26 16:19, first Q&A handoff).
- The two framework figures go into the project (user, 2026-10-26, this chat); whether they enter the stage report or the dev-log page is for the decide 23 chat.
- The archive is read only when the user asks about the conditioned line.

## 3. Where things stand

Facts verified in the code during the Q&A (each beside its source):

- The image and caption sides are two separate `TrunkSide` modules; they share no weights (`ms_train.py` 73).
- Per aspect, the buddy loss averages four InfoNCE terms (image-image, image-caption, caption-image, caption-caption) over the batch of that aspect's buddy pairs; the teacher term uses the image head only, both endpoints, same painting masked, temperature 0.05 (`head_loss`, `distill_loss`). The own-caption terms exist in the code and have weight 0 in recipe R6-5 (`rb_rows.py` 10).
- Painting-mean caption: mean over the painting's caption embeddings, then L2-normalised (`ct2_data.py` 255, 294). Caption language not checked.
- Residual graphs: content(i, j) = ½ [cos_D / sd_D + cos_I / sd_I] over the default caption-mean and default image spaces; emotion score = cos_E / sd_E − λ_i · content with λ_i the share of i's plain emotion buddies that are also content buddies (mutual top-20 in D or in I); style score = cos_S / sd_S − 0.5 · content; genre: plain mutual top-20 in the genre-instructed image space. sd = std of cos over 200,000 random pairs, seed 0 (`ct2_graphs._partition`).
- Stopping: per aspect, the training loss on the stopping paintings' pairs after each epoch; a new minimum saves the heads and the trunk; patience 15, at most 200 epochs; AdamW 1e-3, weight decay 0.05, batch 512 (`itp_train.CONFIG`, the loop in `ms_train.py`).

## 4. Open points for the next step

Notes for the decide 23 chat (minor; nothing to decide here):

1. The stage report's model figure (`model_structure.png`, panel 2) says "content similarity subtracted" for all three graphs; the genre graph is plain.
2. Each aspect's saved best state includes the trunk at that aspect's best epoch, and scoring reloads it per aspect (`ms_train.save_output`, `ct2_d2.py` 63, `rb_student.py` 308, `eg_train3.py` 196). So the three outputs come from three trunk snapshots; "one trunk" describes the architecture, not the weights as scored. The one Qwen pass per image is unaffected. Whether scoring with one shared snapshot changes AMI is not checked.
3. Whether the two framework figures replace or join `model_structure.png` in the stage report and on the dev-log page.

## 5. Code and pitfalls

- The figure scripts write beside themselves (`OUT = Path(__file__).parent`); the detailed script imports the first, so keep them in one folder. Text in `box()` is plain; use `math=True` (detailed script) for `_{}` and `^{}` sub- and superscripts. SVG collapses spaces unless `xml:space="preserve"` (set in `formula()`).
- The Qwen attention-map viewer (round `src/test/20261021_attention_maps/`): `att_viewer.py --root /data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention --port 8770` (port 8765 belongs to a MultiMAE page), opened through Cursor's port forward (`container.md`).
- Round-folder modules go on `sys.path`; read them, do not import them outside their folder.

## 6. State at handoff

- Running: the "r10 decide 23" chat (open, decision A); a page server on port 8768 (scratchpad copy of the two figures); the attention viewer on port 8770; the "r10 stage report 3" chat's servers on 8766 and 8790. No subagent, GPU lock or cluster job.
- Uncommitted: nothing in the project after this cut's commit; in `/project/cosir-devlog`, whatever the decide 23 chat left (not this chat's).
- On disk over 1 GB: unchanged (the attention store, 8.1 GB, has its ledger row).
