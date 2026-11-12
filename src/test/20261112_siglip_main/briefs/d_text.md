# r28 job D: Table 3 text directions and per-seed values on SigLIP 1 (handoff §4 items 2 and 5). Verdict code.

Read first: `briefs/common.md`, the handoff's items 2 and 5, the inventory's Q2, and the paper tables
`docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables/main_retrieval.tex` and `main_clustering.tex` (read
only) for each cell's protocol.

## Part 1, text directions (Table 3)
For our SigLIP 1 model (c*, arm `k11=40`, five seeds; weights
`/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/weights/k11=40/<aspect>__s<seed>.pt`): text to
image, image to text and text to text label mAP on ArtELingo test, at the read-out picked on validation, with exactly
the protocol of the Qwen3 rows (`src/test/20261105_text_retrieval/` and its log; `src/test/20261111_table3_rows/`).
The caption tower reads the Qwen3 plain caption embedding (as r14 trained it); the image tower reads SigLIP 1.
Held exports have only images: compute the caption side on test by running the saved caption towers on the test
captions' Qwen3 plain embeddings (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/`), joined by sample IDs (C6).
Check that the r14 validation exports' caption side equals what your caption-tower inference gives on validation
(a replay, within 1e-5): it proves the inference path before you trust it on test.

Plain SigLIP 1 rows in each text direction, with SigLIP 1's own text tower (store
`/data/SSD2/pre_extract/artelingo_full/siglip_so400m/txt.npy`), as the plain rows of the table were made (read-out
picked on validation the same way, if the plain rows have one).

## Part 2, per-seed values
For ours on SigLIP 1 and for the instructed Qwen3 (3 passes): every Table 2 image cell (class-count AMI) and every
Table 3 cell (image and text directions), per seed, from the records that made the seed means (r17
`src/test/20261103_new_scores/records/verdicts.json` and your Part 1 records). Check that each seed mean you rebuild
equals the table's value (rounded); list any that does not.

## Write
`r28_text.py`, `test_r28_text.py` (the join and the replay), outputs
`/data/SSD2/cosir_r28/20261112_siglip_main/text/`, small records `records/text.json` and `records/per_seed.json`.
GPU inference is small (`res-queue gpu --no-wait --short --label r28:text -- <cmd>`). In the hand-back, give each new
cell beside the current Qwen3 cell.
