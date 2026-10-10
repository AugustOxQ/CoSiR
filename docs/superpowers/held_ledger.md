# Held-read ledger (CVPR plan spec §10)

Every read of a final test split is one row. Final scripts check this file and refuse to run a second time.
Budget: 1 main + 1 reserve read per dataset for the CVPR paper; since 2026-10-09 (constitution C5, amendment 2;
plan §16) one read per pre-registered final method and backbone, plus one reserve read for a fix after a
final-review finding. Nothing is tuned on held data.

| # | Date | Dataset and split | Purpose | Episode / data SHA-256 | Script SHA-256 | Report |
|---|---|---|---|---|---|---|
| H1 | 2026-09-29 | ArtELingo held rows, value (label) episodes, seed 42, 1,024 per label | repaired-factor condition eval | emotion e62ab41f…, style 3a58cf9d… | see report | [condition eval](../reports/auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md) |
| H2 | 2026-09-30 | ArtELingo held rows, same episodes as H1 | stage (d) final test | as H1 | see report | [stage (d) final](../reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md) |
| H3 | 2026-10-01 | ArtELingo held rows, value episodes, seed 43, 8,192 per label | affect factor-learning held test | emotion abd1ca38…, style ee87686c… | 11d5c73f… | [affect held](../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md) |
| H4 | 2026-10-02 | CUB standard test split (5,794 images, all 200 species; includes the 50 zero-shot test species) | backbone-check attribute probes and retrieval (diagnostic, no CoSiR model) | n/a | see report | [backbone check](../reports/auto/v2/2026-10-25_backbone_check.md) |
| H5 | 2026-10-10 | ArtELingo held rows, aspect episodes, seeds 52 to 54, 4,096 per pair | AFF held read, round 6 (rule c394b60); GPU jobs on the held episodes (rule §8.3): verbaliser (W1) and value listing on seeds 52 to 54 (r6_gpu_verbalise.py 8cadb279114a56b628ab2cc6a536657c1479e9ec6f87dfd47fae23ec41ab6ee0, r6_gpu_listing.py 128219dbdf7feaf0c4c9b044433055b6d7f1a5b7528bc7848ba68de3076632b2), LB and LoRA held features (r6_gpu_ft_features.py 6213e85ccd93a83958ea44eca8bace8a93a8fc7d32754ec22ee6714ad94f284b), MLLM reranker on seed 52 (r6_gpu_rerank.py 35426d28e5a1aaee27505581ab3ee48d475e4a0808a3e3fe95114c4c4e2a5d86), shared r6_gpu_common.py 5c1194e6f43488df41ac114612e8727f203fd46acbf75ffbd202f63b569ccd6a | (pending) | a82ca8bc128eb344a7bdda623e106701d0c69a6c63894eb1a1b5b2f818775e4d | (pending) |

**CVPR budget status:** ArtELingo: H5 is AFF's one read (aspect episodes; H1 to H3 were value episodes), reserve H5-R unused; CUB 0 of 2
(H4 disclosed); SemArt 0 of 2; GeneCIS 0 of 2.
