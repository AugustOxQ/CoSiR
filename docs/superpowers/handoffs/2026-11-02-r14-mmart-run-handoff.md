# Handoff: at 19:00 start the MMArt feature extraction queue, verify its stores, record the results

## Brief
- Done: MMArt (PolyArt) prepared: 74,234 WikiArt paintings indexed with ArtELingo's split; extractor built and smoke-tested (parity ≥ 0.9997 on 7 views).
- State: test = 16,782 paintings not in ArtELingo; val = 5,769 ArtELingo val paintings with MMArt labels.
- Next: at 19:00 run the 7-view extraction (about 2 h, local GPU), then parity, near-copy and store checks.
- Blocked: nothing.

> Written 2026-11-02 15:20 by the MMArt prep chat (a side chat beside r14, cut at the context meter). Loop step
> reached: data preparation for a later loop; no research decision open. Direction log: two MMArt ideas added to
> "Parked ideas" (2026-11-02).

## 1. Read in this order

1. This handoff.
2. The round log, [20261102_mmart_prep_log.md](../../../src/test/20261102_mmart_prep/20261102_mmart_prep_log.md):
   what MMArt is, the overlap with ArtELingo, the labels, the user's decisions, the smoke.
3. The docstrings of `src/test/20261102_mmart_prep/mp_extract.py` and `mp_extract_queue.sh` (commands, outputs).

Skills: none.

## 2. Decided by the user, not to be reopened

All 2026-11-02, recorded in the round log:
- The same split as ArtELingo (index column `split`: train / val / held / new). The new paintings are the test set.
- MMArt's `dominant_emotion` is its own label set (it matches our ArtELingo emotion label on 43% of held paintings).
- The aspects stay emotion, style and genre with the ArtELingo teacher instructions. Perspective retrieval and
  perspective names as instructions are parked (direction log).
- Tonight: only the seven views of the 16,782 new paintings (plus the 200-painting parity set). No val teacher views,
  no caption embeddings: the decide chat that designs the MMArt test decides those.
- Extraction after 19:00 (user: "no extraction now, we do that after 19:00").

## 3. Where things stand

- Code: merged into `r14-backbones-sweep` and pushed (1da403f, log 2e93fa7). 16 CPU tests pass.
- Index: `/data/SSD/mmart/mmart_index.parquet` (sha256 in `mp_index_summary.json`). Captions and labels:
  `/data/SSD/mmart/raw/data/data.parquet`. Images: the local WikiArt files (`image_path`), byte-identical to MMArt's.
- Smoke (14:57): 14 stores, parity min cosine 0.99971 to 1.00000 against the ArtELingo stores (bar 0.999), check 14 of
  14. Rates: Qwen about 88 ms/image, so the full queue takes about 2 h (19:00 to about 21:00).
- No full store exists yet: `/data/SSD2/pre_extract/mmart/` does not exist.

## 4. Open points for the next step

None for the user. The steps:
1. GPU check at about 18:55: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` empty and
   `pgrep -af gpu0.lock` shows no holder. If another session holds the GPU, wait for it (the user queued this run):
   the queue script uses `flock -n`, so start it only when the lock is free.
2. Wait until 19:00 Amsterdam time in a background Bash, then launch the queue in the same background command, so the
   chat is notified when it ends:
   `bash -c 'sleep $(( $(TZ=Europe/Amsterdam date -d "19:00" +%s) - $(date +%s) )); bash /project/CoSiR/src/test/20261102_mmart_prep/mp_extract_queue.sh'`
   (start it before 19:00; if it is already past 19:00, launch the queue script directly). The last line is
   `MP QUEUE DONE ok in <s>s` or `MP QUEUE FAILED at <step> rc=<rc>`; the log is under
   `/data/SSD2/pre_extract/mmart/logs/`.
3. On success, read `mp_parity.json` (every view `pass: true`), `mp_stores_check.json` (`ok: true`, 14 stores) and
   `mp_nearcopy.json` (counts at 0.85 to 0.99) in the round folder. Commit the three JSON files with `git add -f`
   (`*.json` is gitignored) and push.
4. Add a "Run" section to the round log: times, seconds per store, parity minima, near-copy counts at each cut, the
   final test count (16,782 minus the exact copies (34) and the near-copies at ≥ 0.85; note any overlap of the two).
   Commit and push.
5. In the direction log's "Now" paragraph, replace "PolyArt waits for the user's link." with one sentence: MMArt
   (PolyArt) stores are ready, the test count, and a link to the round log. Re-read the file right before the edit
   (the r14 run chat edits it too); commit only that change.
6. On a failure: read the log tail; a parity miss or a check failure is traced (which view, which rows) before any
   rerun. A sealed store is never overwritten; a `.pending` folder from a crash stops the queue: inspect it and list
   it on `.scratch/pending_deletions.md` rather than deleting it (unattended chat).
7. Chat end per `routines.md`: storage summary (the new stores total about 0.4 GB, `keep`: the MMArt test rests on
   them; the smoke folder in the old chat's scratchpad is already summarized), commit and push, then end with the
   chat-done line. No next chat: the next decide chat finds MMArt through the direction log.

## 5. Code and pitfalls

- `mp_extract.py extract-all` skips sealed stores, so a rerun after a crash resumes; it refuses a `.pending` folder.
- Smoke runs need `--smoke N <scratch dir>`; never point `MP_STORE_ROOT` at the real root for a smoke.
- `mp_stores.load(view, keys)` joins by MMArt key: new keys from the MMArt store, held keys from the ArtELingo stores
  (teacher views from `*_held`), train/val keys default-image views only (teacher views raise: none exist for val).
- `held_paintings.json` and other frozen inputs are gitignored and live only in `/project/CoSiR` (not in worktrees).
- The r14 run chat works in the same checkout: commit only your own files; if `.git/index.lock` exists, wait.
- Heavy-GPU rule: this run is about 2 h of heavy load, allowed from 19:00 on weekdays; it ends long before 11:00.

## 6. State at handoff

- Running: nothing of this chat (no background job, subagent or Codex job; worktree removed after the merge).
- Uncommitted: nothing of this chat (old untracked files of other rounds remain, as before).
- On disk over 1 GB: nothing new. The 24 GB of image shards were deleted (disk ledger row, 2026-11-02);
  `/data/SSD/mmart` is 113 MB (`data.parquet` and the index).
- GPU check at 14:57: the smoke ran on the GPU without error; run mode: night.
