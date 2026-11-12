# r29 decision lines

One line per result-shaping choice (`~/.claude/references/loop-chats.md`, "Decision lines"):
when · the choice · the main alternative · where · who · could it change a verdict or claim.

| When | Choice | Main alternative | Where | Who | Verdict? |
|---|---|---|---|---|---|
| 2026-11-12 22:53 | Gate (iii) compares per-painting caption counts with the Qwen3 full store's index (all 308,723 captions) | the sealed Qwen3 default store's subset (ids.npz) | r29_captions.py gate_iii_checks | build chat (ticket 01) | no |
| 2026-11-12 22:53 | The Qwen3 caption source's store hash is the sha256 of the Qwen3 full store's index.json | the sealed default store's ids.npz | r29_captions.py store_hash | build chat (ticket 01) | no |
| 2026-11-12 22:53 | caption_rows returns the stored per-caption rows unnormalised (float32); each caller normalises as the protocol it follows does | unit rows from the loader | r29_captions.py caption_rows | build chat (ticket 01) | maybe (a caller that forgets the step changes text cells; ticket 04 follows r23's rule) |
| 2026-11-12 23:04 | Outcome rule strict: WIN when the lower bound > 0, LEVEL when > −r, else BELOW; a missing, negative or NaN r crashes | a WIN-only fallback when r is missing | r29_image_cells.py:100 | build chat (ticket 06) | no (spec §3.4 wording) |
| 2026-11-12 23:04 | Test cells refuse unless the picks record is committed unchanged in HEAD | only check that the file exists | r29_image_cells.py:260 | build chat (ticket 06) | no |
| 2026-11-12 23:04 | LEVEL unit r per image cell: the candidate's own five-seed validation range (AMI at the class count; mAP at R0) | the old model's recorded ranges | r29_image_cells.py:225-232 | build chat (ticket 06) | yes (§3.4 names the candidate's range) |
| 2026-11-12 23:04 | Untold (SC7-type) read-out by r11's rule with the candidate's own R0 ranges (ns_picks.readout_doc); Q1 read-out by r11's stage-4 rule (bs_artret.pick_cstar) | r17's ns_picks.pick, which reads the old model's recorded ranges | r29_image_cells.py:232 | build chat (ticket 06) | maybe (to be checked in review) |
| 2026-11-12 23:04 | Comparator read-outs not recomputed: plain = r14's recorded pick for the encoder; teacher = r11's stored instructed pick | recompute them for r29 | r29_image_cells.py:241 | build chat (ticket 06) | no (§3.7 freezes them) |
