# RedCaps Phase 2 — VLM Judge Design

**Date:** 2026-06-23
**Status:** Approved (no commit per user)
**Branch:** `experiment/conditional_buddy`
**Precursors:** `docs/reports/2026-06-23_redcaps_buddy.md`,
`src/test/20260622_buddy_analysis/phase2_vlm.py` (Impressions Phase 2)

## Goal

The most direct, type-free test that condition buddies share *specific* content:
ask Qwen2.5-VL whether a candidate caption is a GOOD match for an anchor **image**,
comparing the anchor's buddy captions against same-topic and random negatives. On
RedCaps this is cleaner than Impressions — no same-photo or caption-style confound.

## Components & data flow

New `src/test/20260623_redcaps_buddy/phase2_vlm.py`, adapted from the Impressions
version (the `QwenAnnotator` good/bad prompt already targets "Reddit").

- Load `redcaps_buddy.load_data()` + `build_graphs(K=30)` → **raw B/E** graphs
  (NOT the `ensure_connected`-bridged graph; bridge edges are artificial).
- Adjacency = all buddy neighbours (every RedCaps edge is cross-content; no
  same-photo filter).
- Per anchor (eligible = ≥1 buddy), take ≤ `max_buddies` (=6) buddies. For each
  buddy caption draw **two negatives**:
  - **subreddit-matched**: a caption from the **anchor's subreddit**, different
    sample, not a buddy (hard — same topic, tests specific content beyond topic);
  - **plain-random**: any-subreddit caption, not a buddy (easy floor).
- Judge with Qwen2.5-VL via vLLM, reusing `SYSTEM_PROMPT`, `_build_user_prompt`,
  `_parse_response`, `_encode_image_base64`. Candidates shuffled per anchor,
  temperature 0, `top_k=1`, retries on parse/API failure.

## Metrics (per graph B and E)

`buddy_good_rate`, `subreddit_random_good_rate`, `plain_random_good_rate`, and
paired per-anchor diffs (buddy − each negative) with bootstrap 95% CI. Success =
**buddy ≫ subreddit-random ≫ plain-random**. Output `phase2_vlm_{B,E}.json` +
a grouped-bar figure `phase2_vlm.png` under `assets/redcaps_buddy/`.

## Params & server

`--n_anchors 150`, `--max_buddies 6`, `--seed 42`, `--dry_run` (no server) for a
sanity check first. Server via `src/test/automatic_annotator/launch_vllm.sh`
(Qwen2.5-VL-7B on :8000), launched in the background. All 150K images exist on disk
(DINO pass: 0 missing).

## Report

Add "Result 3 — VLM judge (Phase 2)" to `docs/reports/2026-06-23_redcaps_buddy.md`
with the B/E gradient, mirroring the Impressions Phase 2 section.

## Not doing

Anything touching training; judging beyond ~150 anchors; using the bridged graph.
