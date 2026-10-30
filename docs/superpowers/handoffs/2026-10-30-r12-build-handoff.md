# Handoff: build r12 (frozen c* on Affection, CRL bars) from its approved spec, then open the run chat

## Brief
- Done: r12 decided and specced: frozen c* on Affection photos (emotion; confirmation) and CRL bars on ArtELingo and Affection; the spec was reviewed (Opus xhigh) and approved.
- State: on ArtELingo, c* beats its teacher in clustering (+0.0145 AMI) and matches it in retrieval; the Affection store has 63,199 of 85,007 images (FI not yet in).
- Next: tickets, build and final review of the spec's code; then open the run chat.
- Blocked: nothing (PolyArt waits for the user's link; not in r12).

> Written 2026-10-30 21:20 by the r12 decide chat. Loop step reached: step 6 done (spec approved), step 7 next.
> Direction log entry: [2026-10-30](../direction_log.md) (Read, Meaning C, Decided A, Idea B).

## 1. Read in this order

1. The spec, from §1 on: [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md).
   Its decision rule is §6 with §3.4 and §3.5; it inherits measurement and the read-out rule from the
   [r11 spec](../specs/2026-10-29-retrieval-design.md).
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any Codex job.
3. The facts brief, `src/test/20261030_affection_confirmation/ac_facts_brief.md` (Codex, 2026-10-30; checked in the
   decide chat and by the spec's reviewer). It names the code paths with line numbers. Its GPU rates are superseded by
   spec §9, and its "23% ties" by spec §3.4 (22.6% on test under the spec's rule).
4. The data log, `src/test/20261029_r11_data/20261029_r11_data_log.md` (the stores, FI's location, the adapter).

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D9, D14's stage 2 type) and the direction log, 2026-10-30. The spec was approved from its
brief at 21:17, with no container restart.

## 3. Where things stand

- Affection's Qwen default store: `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/`, 63,199 images and 392,331
  texts, four sources. FI's 21,808 images are at `/data/SSD/affection/fi_t4sa/dataset/benchmark/FI/images/<emotion>/`
  and are not yet in the adapter's `image_roots` (per-emotion subfolders).
- Instructed views and a CLIP L/14 store for Affection do not exist yet. The instructed extractors
  (`CT/ct2_extract.py`, `RB/rb_extract_a28.py` in the facts brief) are hard-coded to ArtELingo.
- r11's outputs that stage 0a replays: `/data/SSD2/cosir_r11/20261029_retrieval/outputs/` (stage 4 scores and picks,
  `stage5/verdict.json`, CRL texts and lists in `crl/`). r10's: `src/test/20261027_recipe_sweep/` (`DECISION_RULE.md` §2.4,
  `rs_stage4.py`, `rs_vh.py:171` for the label rule).
- Run mode: night (`run-mode`). GPU check at 21:18: the driver answers (595.99.02), no process on the GPU, no lock.

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to the
run chat, not to the user.

## 5. Code and pitfalls

- New round folder `src/test/20261030_affection_confirmation/`, module prefix `ac_` (round folders go on `sys.path`).
- Keys are `<dataset>/<image_name>`. 1,640 file names repeat across sources, so never join by name alone or by store
  position. The annotation file starts with FI rows, so FI is appended after the existing store rows. Record the old
  store's hashes and commit them before the change (stage 0a(i)).
- Labels: alphabetical codes, ties to the first (r10's `emotion_labels`); the class counts in spec §3.4 are a gate.
- PercepT core inference asserts its training CUDA device (`src/test/20261029_retrieval/rt_percept_val.py:58`). Use
  the full-scale mapper, `src/test/20261020_claim_test/outputs/stage2/full/` (r11's amendment A2).
- c* runs on CPU from `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/{aspect}__s{seed}.pt`
  (r11's `rt_data.CSTAR_ROOT`, `rt_stage0.py:208`).
- Verdict code (labels, picks, outcomes, bootstrap) gets unit tests and an Opus `high` ticket review; the rest gets the
  controller's diff read and one end-to-end smoke run (`agent-routing.md`, "Builds").
- Heavy GPU (FI about 28 min, instructed views about 42 min, CLIP L/14 about 10 min) belongs to the run chat. Light
  checks of 200 images may run in the build, under the GPU lock.
- Unattended: delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat's (older untracked files in `archive/` and `src/test/` predate it).
- On disk over 1 GB: nothing new; the ledger is current (`/project/claude-config/storage/disk_ledger.md`).
