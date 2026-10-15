# Handoff: show the user the cost table for the two Leiden checks, get the route for each, then build and run cheaply

## Brief
- Done: the k-shot check, reported (AFF needs 3 example pairs). The glossary is fixed. Records searched for two Leiden checks.
- State: CLIP Leiden at resolution 1.0 already lost (reader margin −0.32). Old attention-H1 labels exist, but H1 saw the held paintings.
- Next: show the cost table (section 3), let the user pick a route per check, then build and run.
- Blocked: the user's route choice, and which grouping slot the H1 communities replace.

> Written 2026-10-14 01:04 by the "AFF questions" side chat (outside the r9 loop). Cut at the context meter (about 200k
> tokens) on the user's word. Direction log entry: `docs/superpowers/direction_log.md`, "2026-10-14 · K-shot side check"
> (the Leiden matched-count idea is in its "Parked ideas").

## 1. Read in this order

1. This handoff. Sections 3 and 4 carry the two record searches; they are not saved elsewhere.
2. `docs/reports/stage/2026-10-13_aff_model_card.md` §3.2 (groupings) and §3.6 to §3.8 (reader, steering term, gate).
3. Only for the route the user picks: the code paths in section 5.

## 2. Decided by the user, not to be reopened

- These are **side experiments outside the research loop**: no loop protocol, as simple and cheap as possible, reuse
  cached data, features and code (user, 2026-10-14).
- **Two checks** (user, 2026-10-14):
  - (1) replace CLIP's k-means groupings (image, caption) with Leiden on their corresponding graph;
  - (2) "exactly follow" the buddy × PercepT experiment: take the attention-H1 version's Leiden communities and use them
    in today's aspect episodes.
- The k-shot check is a separate auto report; the user discusses it after round 9 ends (user, 2026-10-14).
- No more AFF tuning (user, earlier). These checks are questions; they do not change AFF.

## 3. Where things stand

**Cost table** (from the records; nothing run yet):

| Check | Cheap screen: told and reader margins (CPU) | Full AFF rebuild on seed 42 |
|---|---|---|
| (1) Leiden on the CLIP k-NN graphs, resolution 1.0 | already done (2026-11-16 arm A3): lost | about 45–60 min, plus an adapter |
| (1′) the same at about 64 groups (matched to k-means) | about 10 min, plus a resolution argument | about 45–60 min |
| (2) attention-H1 communities, old labels mapped to rows | about 10 min, plus a mapping script | about 45–60 min |
| (2′) H1 rebuilt on the 36,518 scorer-train paintings | porting the old code, plus GPU tens of minutes (no recorded runtime) | the same |

- **The cheap screen** is how the affect grouping was chosen. It uses the told term and N6's hard reader on base B, and
  the margins are over the condition-free counterpart. It is not AFF: no practice bank, no half-readers, no cells, no τ.
- **The full rebuild** runs the r9 recipe chain on new groupings and compares with AFF's seed-42 pins.

**Check (1), what exists.** `src/test/20261116_grouping_step1_style/` arm A3 (`run_step1.py --set descriptive`):
- **How it was built.** 20-NN union graph, Leiden RBConfiguration, resolution 1.0, seed 42, groups under 200 rows
  merged. Image at painting level (36,518 nodes): 17 groups. Caption at row level (183,694 rows): 30 groups.
- **Result.** Told margin +1.39 [1.09, 1.69] against k-means +1.64. Reader margin +0.04 [−0.20, 0.27] against +0.35
  (paired −0.32 [−0.57, −0.06]).
- **Agreement and cost.** AMI with the k-means groupings is 0.57 and 0.51. The run took 7.5 min on CPU (caption Leiden
  297 s).
- **Why k-means at all.** k-means 64 was inherited from stage (d) and never compared. The user said so on 2026-10-05.

**Check (2), what exists.** Attention-H1 is a small learned student:
- **Inputs.** CLIP image and text features (PCA to 50 d) plus the 28 GoEmotions probabilities, at painting level.
- **Model.** 32-d projections mixed by 1-head self-attention.
- **Training.** Two InfoNCE terms whose positive pairs are the edges of the buddy graph (K = 20, union of image and
  text 20-NN) and of the affect mutual-kNN graph (K = 20).
- **Communities.** A mutual 20-NN graph on the 32-d embedding, then Leiden ModularityVertexPartition, seed 42, no
  resolution: 19 communities on train.
- **Recorded AMI.** Emotion 0.135, genre 0.240. No style AMI was recorded.
- **Sources.** Weekly `docs/reports/weekly/2026-09-30_percept_buddy_to_v2.md` §2.5 (lines 145–165), and
  `docs/reports/auto/percept/pilots/20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot_pilot_report.md`.
- **Labels on disk.** `/project/CoSiR-buddy_prototype_conditioning/src/test/20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz`
  (18 MB, untracked; worktree of branch `experiment/percept_topic_pipeline`). It holds `train_paintings` (61,402
  names), the 32-d embeddings and `train_community_pre/post`. There is no model checkpoint. The scripts
  (`run_learned_student_arch_sweep_pilot.py`, `run_attention_h1_embedding_snapshot_pilot.py`) exist only on that
  branch. `detect_communities` is in `src/conditional_buddy/prototype_seed.py`.
- **The catch.** The old "train" was the whole `artelingo_train.json` (61,402 paintings). That file holds today's
  scorer-train, selection, val and held rows (183,694 + 32,413 + 30,872 + 61,744 = 308,723). So H1 saw the 12,281 held
  paintings and the selection paintings, without labels (CLIP features and GoEmotions of their captions). Mapped labels
  (by painting name to the 183,694 scorer-train rows) are fine only for a development-only check. They can never
  feed a held read.

## 4. Open points for the next step

1. **Key, ask the user first:** a route per check: the cheap screen only, or the screen and then the full AFF rebuild
   if it looks promising. Our default is the screen first, which is the cheapest.
2. **Key:** for check (2), map the old labels (minutes; development-only) or rebuild H1 on scorer-train paintings
   (clean; porting plus GPU time). Our default is to map, given "cheap" and "development-only".
3. **Key:** for check (2), which slot the H1 communities take. The options:
   - replace both image and caption groupings (H1 mixes content and affect);
   - replace affect;
   - a fourth grouping, as round 8 did.
   The screen's told mapping must change to match. Ask; there is no safe default.
4. **Minor:** for check (1′), the target group count. Our default is the resolution that gives about 64 groups per
   side, found by the r8 search rule (`20261129_r8_grouping/r8_recipe.py` `recipe_at_resolution`, DECISION_RULE M3).

## 5. Code and pitfalls

- **Screen:** `src/test/20261116_grouping_step1_style/run_step1.py`. `RESOLUTION = 1.0` is hard-coded at line 88. The
  groupings come from `painting_grouping` (line 322) and `caption_grouping` (line 373). `evaluate_general` (line 596)
  takes any tuple of groupings and any told mapping. A new grouping needs a per-row label array over the 183,694
  scorer-train rows, seed 42, joined by the extraction sample IDs (`.claude/CLAUDE.md`, "Sample ID consistency").
- **Full rebuild:** the r9 recipe chain, `src/test/20261202_r9_scope/run_r9_stage2.py --step {groups,a3,reader,baselines,cells,record}`.
  - Swap `fit_partitions` (`r9_recipe_groups.py:107`) to load the new labels.
  - Write to a non-pinned output folder, and skip the CLIP bit-equality checks against the pins.
  - Recorded step times: bank about 210 s; heads 118 s; A3 retrain 659 s on GPU (3.9 GiB); reader chain about 820 s;
    cells about 210 s.
  - Round 8's and round 7's runners only add a fourth slot; replacing a grouping needs a new runner there.
- **r9-scope is shared and under active build** (r9 build 9). Import its code from a fixed commit (a worktree), never
  from a tree that is changing. Commit only your own files, by path.
- **The GPU is busy:** at 01:04, pid 1235147 held 13.2 GB (likely an r9 build job). Check `nvidia-smi` and take the
  lock (`flock -n -o -E 75 /tmp/gpu0.lock ...`). The cheap screens are CPU only.
- **Codex:** 5-hour window at 30% left at 01:04 (it resets 04:38), shared with r9 jobs. The adapters are small, so
  Claude can write them directly.
- **Never read held annotations.** The mapped H1 labels saw held paintings without labels: development episodes only.
- **New folder** `src/test/20261204_leiden_groupings/` (next CoSiR sequence date), module prefix `lg_`. Keep the log in
  that folder. A short auto report plus a half-page user-read report, as the k-shot check had, unless the user asks for less.

## 6. State at handoff

- Running: nothing of this chat's. The r9 build chats run their own Codex and GPU jobs.
- Uncommitted: nothing of this chat's (`src/test/20261129_r8_grouping/results*` are not ours).
- On disk over 1 GB: nothing.
