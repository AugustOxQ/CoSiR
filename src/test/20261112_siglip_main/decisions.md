# r28 decision lines

One line per result-shaping choice (`~/.claude/references/loop-chats.md`, "Decision lines"):
when · the choice · the main alternative · where · who · could it change a verdict or claim.

| When | Choice | Main alternative | Where | Who | Verdict? |
|---|---|---|---|---|---|
| 2026-11-12 19:40 | Encoder pick: the highest mean of the six validation cells (AMI 9/27/10 and plain-cosine mAP, seed means), as the handoff's agent default | a per-cell vote | handoff §4 item 1 | run chat (default from paper chat) | maybe |
| 2026-11-12 19:35 | Text-pass timing sample: 2,000 of 308,723 caption rows, RandomState(0) | the first 2,000 rows | r28_time.py:main | worker C | no |
| 2026-11-12 19:35 | Text-pass timing includes the tokenizer, as r19's image pass included the processor | forward pass only | r28_time.py:main | worker C | maybe (a slightly lower Qwen3 text time) |
| 2026-11-12 19:44 | MMArt cells taken from r17 (and r18 for + TURTLE) records, not rescored; the same ids give the paper's Qwen3 rows exactly | rescore from r15 exports | r28_mmart.py | worker B | no |
| 2026-11-12 19:44 | EmoSet read-out tolerance r = r14's committed photo-model seed range (0.00695) for every arm, as r14 did; picks under r/2 and 2r kept in the record | each arm's own range | r28_photos.py:r_map_for | worker B | maybe |
| 2026-11-12 19:44 | Affection "paintings only" read-out tolerance r = the arm's own five-seed validation range (r12's rule for c*); a gate replays the Qwen3 pick | r12's Qwen3 range | r28_photos.py:r_map_for | worker B | maybe |
| 2026-11-12 19:46 | Validation AMI: r17's clustering call (MiniBatchKMeans, n_init 3, states 0-2, float16 rows as stored); mAP: rt_metrics.retrieval with qe_k=0 | none (r17 and t5 agree) | r28_valpick.py:score_encoder | worker A | no |
| 2026-11-12 19:46 | Qwen3 c* scored on validation as an information row, not a pick candidate | leave it out | r28_valpick.py:PICK_ENCODERS | run chat | no |
