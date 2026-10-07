# Lightweight CLIP fine-tuning comparator (linear probe, last block, LoRA): run log

Folder date 20261124 is a sequence number. Times are Amsterdam local time (`TZ=Europe/Amsterdam date`); cluster job
tags are UTC. Spec `docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md` (f472cc3); plan
`docs/superpowers/plans/2026-10-07-clip-lightweight-ft.md` (470f6aa); SDD ledger
`.superpowers/sdd/2026-10-07-clip-lightweight-ft/progress.md` (tasks, reviews, rulings).

## Timeline

| Time | Event |
|---|---|
| 2026-10-07 to 08:09 | The user adopted the comparator (three lightweight variants, no full fine-tuning; nodes 404, 405, 411, up to 9 GPUs) and kept the paper's framing on hold; design approved in chat; spec and memo update committed (f472cc3), plan (470f6aa); execution subagent-driven |
| 2026-10-07 08:12 | Workspace and briefs; Tasks 1 (data) and 4 (evaluation) dispatched in parallel |
| 2026-10-07 08:25 to 08:28 | Image cache built (`ft_data.py build-cache`, commit 13ca045): 49,121 images (scorer-train, val, selection paintings), 7.39 GB uint8 at `/data/SSD2/pre_extract/artelingo_clip224/`; 61,744 held rows excluded; CLIP-processor normalisation check max diff 2.4e-7 |
| 2026-10-07 08:29 to 08:33 | Data synced to node404, node405, node411 (9.30 GB each: annotations, feature cache, image cache, CLIP weights) |
| 2026-10-07 | Tasks 1 to 4 reviewed (Task 2 Opus, others Sonnet or Opus); fix rounds for Tasks 2, 3, 4; all clean (ledger) |
| 2026-10-07 08:53 to 08:57 | The user approved pushing `main` (109 commits ahead) through `cluster sync` (first sync at 63efe4a, 08:53); in-job checks on all three nodes passed (RTX A6000 ×3 each; cache SHA ok; CLIP snapshot ok) |
| 2026-10-07 09:05 to 09:08 | GPU smoke jobs per variant passed at 65bb2f4 (exit 0; a harmless multiprocessing temp-dir cleanup traceback on NFS at exit) |
| 2026-10-07 09:09 to 09:29 | Nine runs at 65bb2f4: LP {1e-4, 3e-4, 1e-3} on node404, LB {3e-6, 1e-5, 3e-5} on node405, LoRA {3e-5, 1e-4, 3e-4} on node411, one A6000 each; all succeeded. Epoch times: LP 0.5 s, LB about 16 s, LoRA about 74 s for the first epoch and about 51 s later |
| 2026-10-07 09:29 to 09:33 | Pulled. Incident: the first node411 pull also fetched 9 old `test_backbone_embeddings.pt` files (29.95 GB) from node411's results folder onto the system disk (`cluster pull --tag` on a script job falls back to the whole results folder); removed by birth time, originals remain on node411 |
| 2026-10-07 09:35 | Selection by val retrieval (spec §4): LP lr 3e-4 epoch 9 (selection 0.1168; plain CLIP 0.0668), LB lr 3e-5 epoch 8 (0.1334), LoRA lr 1e-4 epoch 10 (0.1515). Edges: LB's best lr is its grid's largest, LoRA's best epoch is the last |
| 2026-10-07 09:35 to 09:37 | Evaluation (`ft_eval.py`, `results/eval.{json,log}`), command: `ft_eval.py --features LP=res/cluster_jobs/20261007-070950-65bb2f4/code/outputs/clipft/LP_lr3e-4/features.npz LB=res/cluster_jobs/20261007-071151-65bb2f4/code/outputs/clipft/LB_lr3e-5/features.npz LoRA=res/cluster_jobs/20261007-071251-65bb2f4/code/outputs/clipft/LoRA_lr1e-4/features.npz CLIPcache=res/cluster_jobs/20261007-071151-65bb2f4/code/outputs/clipft/LB_lr3e-5/features_epoch0.npz --out results/eval.json`: pooled seeds 49 to 51, R@1 LP 15.00, LB 14.97, LoRA 15.14 against plain CLIP 13.04, B 18.07, B′(A0) 18.29, AFF 18.88; fine-tuned minus plain +1.93 to +2.10; fine-tuned minus B′(A0) −3.15 to −3.32; AFF minus fine-tuned +3.74 to +3.91 (all intervals exclude 0). The cache-path untrained reference equals plain CLIP within −0.01 |
| 2026-10-07 09:47 | Independent recompute (own code, `rederive/`; commits e98f21d, c5708ab): selection identical; 531 compared quantities agree with `results/eval.json` (max difference 0.0 pp); only the selected runs' `features.npz` reproduce the fine-tuned rows |
| 2026-10-07 10:33 | Whole-branch final review (Opus): CONFIRMED WITH FIXES (`final_review/final_review.md`): data, selection, all 3,595 numeric values of `eval.json` and end-to-end reconstruction from `best_params.pt` re-derived with its own code; 0 blocking, 4 should-fix, 13 nits; the first diff package was empty (a controller shell slip) and was regenerated during the review. Fix wave applied to the report (writer) and to this log (N10: the evaluation command) |
| 2026-10-07 10:34 | Scoped re-review of the fix wave (Sonnet): 16 of 16 addressed, no new breakage; one minor (§3.4 provenance wording) fixed by the controller (ruling in the ledger). Report, figures (`figure_data.json` force-added), `reports_sum.md` row, final review, CVPR memo §2 outcome and this log committed |
