# r26 decisions

One line per result-shaping choice: when · the choice · the main alternative · where · who · could it change a verdict
or claim. User decisions (2026-11-11, paper chat) are in the handoff's section 2 and are not repeated here.

| When | Choice | Main alternative | Where | Who | Changes a claim? |
|---|---|---|---|---|---|
| 2026-11-11 05:45 | Stage 2 on CLIP L/14 patch tokens with attention pooling (PercepT's main version), tokens in the B/32 store's form: `visual_projection(last_hidden_state)`, all 257 tokens (CLS included), no post-layernorm, no normalisation | Stage 2 on the L/14 global vector (exists already) | `po_paths.CLIP_L14_PATCH`, `po_extract.py` | run chat (handoff default) | maybe |
| 2026-11-11 05:45 | Both topic cuts, K = 40 and strict Algorithm 1 (gamma 2); only K = 40 fills the paper row; strict is reported with its collapse check | K = 40 only | `po_paths.CUTS` | run chat (handoff default) | no |
| 2026-11-11 05:45 | ModernBERT caption embedding: attention-mask mean of `last_hidden_state`, special tokens included, max length 64, fp32 (pc_roberta.py's pooling for RoBERTa); model `cirimus/modernbert-base-go-emotions` at revision 808f9db, loaded as its base `ModernBertModel` (the classifier head is not used) | CLS token or the classifier's pooled output | `po_extract.py` | run chat (handoff default) | maybe |
| 2026-11-11 05:45 | No emotion map before Eq. 2: ModernBERT (768) already matches CLIP L/14's width (768), as in PercepT's own setup; arm B4 needed a fit-PCA only because RoBERTa (768) differed from B/32 (512) | fit-PCA 768 to 768 (a rotation plus centring) as in B4 | `po_percept.py` | run chat | maybe |
| 2026-11-11 05:45 | Scoring with r17's scorer (`ns_metrics.percept_partitions`, `ns_data.labels`, r17's held order), as r25 and the Table 2 rows; mean over 5 seeds and 3 k-means states; collapsed seeds (collapse rule, GLOSSARY rows 9 and 10) are left out of the mean and counted | ct_score's own path (same reading, different code) | `po_score.py` | run chat | no |
| 2026-11-11 05:45 | Outputs on `/data/SSD2/cosir_r26/` (system disk kept free) | the round folder's `outputs/` | `po_paths.OUT` | run chat | no |
| 2026-11-11 05:58 | Scoring adds the class-count reading: k-means at each aspect's class count (emotion 9, style 27, genre 10, r17's `class_count`), fitted per aspect; the k = 20 reading is kept for the supplementary. Relayed by the paper chat; the user's decision is in the paper README (row of 2026-11-11, cosir-paper commit 71be7b4) | k = 20 only (the handoff's protocol) | `po_score.py` | user (paper chat); run chat applied it | no |
