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
| 2026-11-12 19:58 | Text-direction protocol taken from r23 (20261106_text_claim), whose score files rebuild all nine printed Qwen3 text cells; the brief pointed at r22 | r22's validation exploration | r28_text.py header | worker D | no |
| 2026-11-12 19:58 | Caption/image tower replay accepted at float16 resolution (>=99.5% equal, rest one float16 step, max 6.1e-5) instead of 1e-5 | 1e-5 absolute (cannot hold on float16 stores) | test_r28_text.py | worker D | no |
| 2026-11-12 19:58 | Plain SigLIP 1 text rows reused from r23's side row (own text tower, picked on validation), not rerun | rescore | r28_text.py cells() | worker D | no |
| 2026-11-12 19:58 | Per-seed image mAP for SigLIP 1 from r14 per-query files (r17 keeps only means) | none | r28_text.py per-seed | worker D | no |
| 2026-11-12 20:00 | Figure candidates: plain comparison is plain SigLIP 1 (unit rows), counterpart of plain Qwen3; seed 0 and every Qwen3 rule unchanged | none | r28_qual.py load | worker G | maybe (changes every neighbour list) |
| 2026-11-12 20:00 | Cluster figure: k-means (MiniBatchKMeans, state 0) and r20's per-class gain rule refitted on the SigLIP 1 rows | none (Qwen3 used r20's candidates built the same way) | r28_qual.py partitions, r20cand | worker G | maybe |
| 2026-11-12 20:04 | CLAY and CRL read-out tolerance in units of SigLIP 1's own c* validation mAP ranges (ns_picks.ranges(set, 'siglip1')) | Qwen3 c* ranges, as r11's CRL used (same CRL pick under both) | r28_clay_crl.py make_picks | worker E | maybe |
| 2026-11-12 20:04 | CLAY text set by r18's rules unchanged (set A won everywhere); CRL uses set A's 50 values and template | none | r28_clay_crl.py | worker E | no |
| 2026-11-12 20:04 | SigLIP 1 picks on validation: CLAY none+k10 (as on CLIP L/14), CRL center+k10 (CLIP L/14: none+k10) | none | clay_crl/picks_siglip1.json | worker E | maybe |
| 2026-11-12 23:14 | Table 5 removal cells = removal minus c*, the difference of the five-seed means before rounding, rounded to 3 decimals; + for a gain, 0.000 for under 0.0005 | the difference of the rounded means (can differ by 0.001) | records/ablation_siglip1.json deltas_vs_cstar | controller | maybe |
| 2026-11-12 23:30 | Correction to the 19:58 line (review part 2, finding 4): the fallback rests on P alone (r10's stage 3 needs exports on 29,232 rows, 23,080 of them training paintings, which these runs do not write); r_a is a fixed constant of the rule and the seed set was set to {0} by the brief, so neither is a missing input | none (same outcome) | r28_settings.py docstring | controller | no |
