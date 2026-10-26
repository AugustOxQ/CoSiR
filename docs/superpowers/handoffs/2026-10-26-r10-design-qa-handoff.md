# Handoff: r10 design Q&A: explain the current model design to the user, question by question, from the stage report, the specs and the code

## Brief
- Done: nothing yet; a side chat, opened from "r10 decide 23" at the user's request (2026-10-26 16:19).
- State: the current recipe (no own-caption terms, teacher share 1/5) passes held emotion on four encoders (Qwen 0.1029 against the strongest PercepT entry 0.0874); decision A on the next experiment is open in the decide 23 chat.
- Next: read the design sources below, then answer the user's questions about the model structure; change nothing.
- Blocked: nothing; waits for the user's first question.

> Written 2026-10-26 16:19 by "r10 decide 23". Loop step reached: step 2 of the next loop (decision A), open in the
> decide 23 chat. Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The stage report's design section, `docs/reports/stage/2026-10-26_condition_free_line.md` lines 168 to 291 ("The
   design, as it stands on 2026-10-26"): the four views, the graphs, the network, the losses, test-time reading, the
   other encoders, the cost. Its figure: `docs/reports/assets/2026-10-26_condition_free_line/model_structure.png`
   (view it with Read). The rest of the report only when a question needs it: "Where we started" (lines 34 to 167,
   PercepT and the first pilots), "The steps, in date order" (318 to 907), "What the analyses say about the design"
   (933 to 998), "Glossary" (1098).
3. `GLOSSARY.md`: recipe, teacher share, matched arm, frozen input, seed range, the PercepT versions.
4. The specs, §3 Design of each, when a question goes below the report's level:
   - `docs/superpowers/specs/2026-10-19-stage2-model-sheet-design.md` (lines 70 to 132): the network as first built
     (trunk 2048 to 1024, heads 1024 to 2048), the loss terms, the temperature, the stopping rule.
   - `docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md` (74 to 199): the recipe switches (own-caption
     terms off, teacher weight 0.25), the other encoders' input layer.
   - `docs/superpowers/specs/2026-10-25-encoders-and-genre-design.md` (79 to 192): the genre variants (teacher
     multiplier, relational term, adaptor), stage iii.
5. The code, when the user asks for the exact computation. All of it lives in dated round folders under `src/test/`;
   `src/model/`, `src/train/` and `src/eval/` are the archived conditioned line's and are not this model:
   - `src/test/20261019_model_sheet/ms_train.py`: `TrunkSide` (line 31), `TrunkHead` (38), `MaskHead` (48),
     `distill_loss` (102, the teacher term), `head_loss` (119, the buddy and own-caption InfoNCE terms per head).
   - `src/test/20261025_recipe_backbones/rb_student.py`: `WideTrunkSide` (51, the input layer for an encoder of
     another width), `loss_patch` (72, the recipe switches); `rb_rows.py` (lines 8 to 13, the six recipes; `R6-5` is
     the current one).
   - `src/test/20261025_encoders_genre/eg_train3.py`: `GenreLoss` (65), `switched_head_loss` (90, the relational
     term and the teacher multiplier beside the direct term); `eg_variants.py` (the variant names: GT teacher
     multiplier, GR relational weight, GA adaptor; ladders in line 5).
   - Graphs: `src/test/20261020_claim_test/ct2_graphs.py` (`build`, line 99; `_mutual`, 64) and
     `src/test/20261018_aspect_graph_sheet/ags_graphs.py` (the graph types, residual graphs included).
   - The instruction strings: `src/test/20261017_prompted_aspect_views/pav_prompts.py`.
   - Feature extraction: `scripts/extract_features.py`; stores under `/data/SSD2/pre_extract/<dataset>/<backbone>/`.
6. `~/.claude/rules/occam-razor-chat.md` for the replies (point first, one idea per sentence, field terms, no new
   names).

Skills: none.

## 2. Decided by the user, not to be reopened

- This chat explains; it changes nothing: no edit to code, specs, reports or the direction log, no experiment, no GPU
  work, no commit. If a question shows a mistake in the report or the page, say so and note it for the decide 23 chat.
- Every decision of the user's is made and recorded in the "r10 decide 23" tab (the loop rule). If the user states a
  decision here, say that it goes in that tab and do not act on it.
- The archive (`archive/v2_conditioned/`) is read only when the user asks about the conditioned line or a named old
  result; say when an answer comes from it, with its date.

## 3. Where things stand

The design in six lines; every number is in the stage report's design section, with its source report.

- Input: Qwen3-VL-Embedding-2B ("Qwen") embeddings of the default image and of the painting-mean default caption. One
  instructed Qwen view per aspect (emotion focus form; style and genre one-word form) builds one buddy graph per aspect
  (mutual top-20 neighbours on painting means; emotion and style residual graphs, genre a plain graph; training
  paintings only, no labels).
- Network: two sides (image, caption), each a two-layer MLP: a shared trunk 2048 to 1024, then one head per aspect
  1024 to 2048. A student on another encoder changes only the image input layer.
- Loss, current recipe: per aspect, buddy InfoNCE terms (weight 1) on that aspect's graph, plus a direct
  distillation term of the image head toward the instructed Qwen image of that aspect (weight 0.25, temperature
  0.05); own-caption terms removed; nominal teacher share 1/5. Each aspect stops on a label-free loss.
- Development variant, not in the recipe: a relational genre term (KL between the genre head's batch similarity rows
  and the instructed genre image's) at weight 8 reaches genre AMI 0.5022 on fresh seeds against the recipe's 0.4694
  (29,232 development paintings; picked among seven variants on one seed; no held test).
- Test time: one default-image encoder pass, the trunk, the three heads; k-means (20 clusters) per head, AMI against
  the label. No instruction and no caption at test time.
- Results: held emotion (12,281 paintings, 5 seeds; bar: strongest PercepT entry 0.0874, pass needs +0.0100 with the
  interval above zero): Qwen 0.1029, SigLIP 1 0.1036, SigLIP 2 0.1024, CLIP L/14 0.0996 pass; CLIP B/16 0.0945 and
  CLIP B/32 0.0885 inconclusive; DINOv2-large fails at sample scale. k-means only; every student needs Qwen at training.

## 4. Open points for the next step

- The user's questions, as they come. The user said they are "a bit lost" about the model structure: start from the
  figure and the six lines above, then go as deep as asked. Give the source of each fact (report section, spec line or
  code line); say "not checked" for anything not verified in a file.
- Nothing to decide here.

## 5. Code and pitfalls

- Code names and their plain names: D2 = the first shared model (model sheet, 2026-10-19 and 10-20); R6-5 = the current
  recipe (no own-caption terms, teacher weight 0.25); Gres, Fres05, R0o = the emotion, style and genre graph types;
  GT-m, GR-w, GA-w = the genre variants (teacher multiplier m, relational weight w, adaptor weight w); P3 = PercepT core
  with 40 topics; "matched arm" and "frozen input" as in the glossary. Use the plain names in replies.
- "Teacher share" is nominal (weights, not measured loss values): 1/5 both for D2 (0.5 of 2.5) and for R6-5 (0.25 of
  1.25).
- Report names keep sequence dates (12-24, 12-25); the real dates are 2026-10-25 and 2026-10-26. The stage report's
  step dates are real.
- Held data: readable, never used for tuning or selection (`~/.claude/rules/held-data.md`); the development split is the
  ArtELingo train side (29,232 paintings at full scale, 5,000 in older reports).
- Round-folder modules carry prefixes (`ms_`, `ct2_`, `rb_`, `eg_`) because the folders go on `sys.path`; read them
  with Read or grep, do not import or run them. Python, if a check needs it: `/root/miniconda3/envs/CoSiR/bin/python`
  with `PYTHONDONTWRITEBYTECODE=1`. No GPU work; delete nothing.
- Context: this is a side chat, not a phase chat; when the context meter prints, tell the user the size and offer a cut.

## 6. State at handoff

- Running: the "r10 decide 23" chat (open, waits for the user's decision A); two preview servers of the "r10 stage
  report 3" chat (ports 8766 and 8790); no subagent, Codex job, GPU lock or cluster job.
- Uncommitted: in the project, nothing; in `/project/cosir-devlog`, the staged page (on purpose; the deploy waits for the
  user's word in the decide 23 chat).
- On disk over 1 GB: unchanged (the rounds' outputs on /data/SSD2, kept; the shared feature stores).
