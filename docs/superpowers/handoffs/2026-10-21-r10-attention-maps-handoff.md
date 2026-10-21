# Handoff: a side chat that maps Qwen's attention under the four views on the 5,000 development paintings

## Brief
- Done: decide 17 read the D2 ablation (decision C); the user asked for this side task before the next run.
- State: the four Qwen views differ in structure (an instruction keeps 31% to 42% of the default image's top-20 neighbours); their attention is not looked at yet.
- Next: settle the open design points with the user, then extract the maps, build the HTML viewer, write the summary.
- Blocked: nothing. Run mode is "night": ask the main chat for "day" so this chat may ask the user.

> Written 2026-10-21 15:20 by "r10 decide 17". Loop step reached: a side task opened during decide 17's wrap-up
> (after decision C, before decision A); decide 17 goes on in its own tab. Direction log entry: none for this side
> task; the last one is [2026-10-21 · Claim test read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The four views as measured so far: auto report
   [12-22](../../reports/auto/percept/2026-12-22_d2_ablation.md), section "Four Qwen views", and its data
   `src/test/20261021_d2_ablation/outputs/analysis/views.md` (CKA, top-20 neighbour overlap, AMI and probe per view,
   the instruction offsets).
3. Code to reuse (read, do not change): the four prompts `src/test/20261017_prompted_aspect_views/pav_prompts.py`
   (`PROMPTS`: `emotion_focus`, `style_word`, `genre_word`); caption extraction `pav_extract.py` and the frozen sample
   `pav_sample.py` (`load_frozen`) beside it; image extraction `src/test/20261017_image_transfer_pilot/itp_extract.py`
   (`encode`); the 5,000 evaluation paintings and their order `src/test/20261021_d2_ablation/ab_analysis.py`
   (`check_order`, `eval_rows`: `s['subset_pid']`).
4. Before the HTML page: `container.md`, "Showing the user an HTML page".

Skills: none.

## 2. Decided by the user, not to be reopened

The request (user, 2026-10-21 about 15:20, decide 17), verbatim:

> before the real next run, I need to deviate a little and start a side visualization chat that visualize the
> attention heatmap over image and caption (and cross) from qwen under four views, across 5k development data.
> That is:
> 1. extract 5k pairs' attention map under four different prompt ways (default + three aspects), and have
>    image-only, text-only, and image-text both as input. So 5k x four prompt ways x 3 modalities
> 2. Build a html interative page to let us slide over different index, choose the prompt, choose the modality and
>    check the attention.
> 3. Give your own summarization of statistics/analysis of these attention maps

- Data: the 5,000 development evaluation paintings of the D2 ablation, in `s['subset_pid']` order. No held data.
- Loop size: small (an exploration of hours): a design in chat, the run, a short auto report with a half-page
  user-read report (`research-loop.md`).

## 3. Where things stand

- Backbone: Qwen3-VL-Embedding-2B; the embedding is the pooled last-token state. Its image and caption stores for
  the four views exist (the report's "Four Qwen views" names them).
- Known view facts (image side, 5,000 paintings): CKA with the default view 0.673 (style) to 0.795 (genre); top-20
  neighbour overlap with the default view 0.307 (style), 0.419 (emotion), 0.423 (genre); emotion against style
  0.209. An instruction changes its own aspect's k-means AMI far more than its probe (style +0.1735 against +0.030).
- The GPU check at 15:20 passed: driver 595.99.02, the 3090 free.

## 4. Open points for the next step

Ask the key ones answer-first (the user answers before seeing the recommendation).

a. **Key: which attention.** Candidates: the last (pooling) token's attention to every input token, per layer and
   averaged over heads; attention rollout over layers; a gradient-weighted map. What is stored decides the viewer's
   controls (a layer slider or one map).
b. **Key: which text.** A painting has several ArtELingo captions; text-only and image+text need one text (or a
   fixed few). The caption stores use the mean over captions, which attention cannot use.
c. **Key: what "cross" shows** in image+text: text tokens to image patches, image patches to text tokens, or the
   pooling token's split between the two.
d. **Minor: device and time.** About 60,000 forward passes with eager attention (flash and SDPA return no weights):
   estimate on a 100-painting pilot first. A local heavy GPU run of over an hour does not run on weekdays from 10:00
   to 19:00 (`shared-resources.md`); DAS6 or an evening start are the other routes.
e. **Minor: storage.** Estimate the size from the pilot. Bulk arrays go to `/data/SSD2` under a folder CoSiR owns
   (for example `/data/SSD2/pre_extract/artelingo/`), not the system disk (`storage.md`).
f. **Minor: the viewer.** 60,000 maps are too many for one page or an Artifact: a local page that loads one painting's
   maps at a time, served through Cursor.

The summary (item 3) is the agent's design; candidates: the share of attention on image, instruction and caption
tokens; entropy and spatial spread of the image maps; per-painting change from the default prompt (for example
Jensen-Shannon divergence); differences by emotion, style and genre label.

## 5. Code and pitfalls

- A new dated round folder, `src/test/20261021_attention_maps/`, with a new module prefix (`att_`): round folders go
  on `sys.path`.
- Joins are positional (constitution C6): use the extraction's sample IDs and `s['subset_pid']`, never `range(n)`.
- Change no code in `src/test/20261021_d2_ablation/`: `ab_row.json` binds a code hash.
- GPU work under the lock: `flock -n -o -E 75 /tmp/gpu0.lock <cmd>`; check `nvidia-smi` first.
- The report takes the next free sequence name in `docs/reports/reports_sum.md` at write time; decide 17 may also
  add rows today.
- A pilot's or smoke run's files over 100 MB go at the task's end; list the rest in the storage summary.

## 6. State at handoff

- Running: nothing of this task. Decide 17 continues in its own tab (wrap-up, decision A, item 5's grill).
- Uncommitted: nothing for this task after decide 17's commit.
- On disk over 1 GB: nothing new for this task.
