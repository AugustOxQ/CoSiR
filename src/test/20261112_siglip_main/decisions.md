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
| 2026-11-12 19:54 | Graph stores the r28 rows need are reused from r14's outputs (after a code and build check) instead of rebuilt in r28 | rebuild in r28 | r28_trainer.py:56-66 | Codex T1 (reviewer: archives identical) | no |
| 2026-11-12 19:54 | "instructed_graphs" removal keeps r10's single graph on the Qwen3 plain embedding (default_view, k=40) | a graph on the SigLIP 1 image | r28_trainer.py training_context | paper chat default, Codex T1 | maybe |
| 2026-11-12 19:58 | Settings check uses the handoff's fallback (mean of the six validation cells, seed 0, c* wins ties within 1e-12): r10's stage 3 criterion needs P exports (29,232 rows, 23,080 of them training paintings), Qwen3-only r_a and seeds 1-3, none of which exist for these runs | r10's stage 3 criterion | r28_settings.py docstring, TIE | worker F | maybe |
| 2026-11-12 19:58 | Ablation scorer maps r28 arm names onto t5's removal names with an assert; validation exports of graph-removal rows found by glob and checked against the run record | edit t5 | r28_ablation.py T5_NAME; r28_settings.py | worker F | no |
