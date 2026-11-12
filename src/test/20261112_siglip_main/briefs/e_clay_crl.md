# r28 job E: CLAY and CRL on SigLIP 1 (handoff §4 item 7), and + TURTLE on MMArt (item 8). Verdict code.

Read first: `briefs/common.md`, the handoff's items 7 and 8, the inventory's Q2, and the paper tables
`docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables/{main_clustering,main_retrieval,mmart}.tex` (read
only) for the CLAY, CRL and + TURTLE rows.

## Part 1, CLAY and CRL
Today both run on CLIP L/14 (CLAY: r18 `src/test/20261103_outside_methods/om_clay.py`, `om_paths.py` around line
100; CRL: `src/test/20261029_retrieval/rt_crl.py`). Run each on SigLIP 1, with SigLIP 1's image features
(`/data/SSD2/pre_extract/artelingo_full/siglip_so400m/`, `/data/SSD2/pre_extract/mmart/siglip_so400m/`) and SigLIP
1's text tower in place of CLIP L/14's:
- the same texts: CRL's saved value lists (from Qwen3-VL-8B-Instruct) and CLAY's text sets, unchanged;
- the text embeddings: encode those texts with SigLIP 1's text tower the way the image store's text side was made
  (`scripts/extract_features.py`, backbone `siglip_so400m`: model id, prompt template if any, normalisation, max
  length). Write them to `/data/SSD2/cosir_r28/20261112_siglip_main/clay_crl/text/`;
- the same picks made the same way, on SigLIP 1 validation (whatever CLAY and CRL picked on CLIP L/14 validation:
  read the code and logs; the test split never chooses);
- the encoder switch as a parameter (default CLIP L/14, so the existing rows replay unchanged). Show that the CLIP
  L/14 path still reproduces one existing CLIP L/14 cell of each method (within 1e-6) after your change.
Cells: Table 2 (ArtELingo test AMI at the class counts 9 / 27 / 10), Table 3 (image mAP at the picked read-out),
Table 4 (MMArt AMI at 9 / 20 / 10 and mAP). Seeds as the CLIP L/14 rows used.

## Part 2, + TURTLE on MMArt, SigLIP 1
Job B reports whether this row already exists; the controller tells you. If it does not, run it as the Qwen3 MMArt
TURTLE row was made (r18 TURTLE code, `om_data.py` encoder selection). r18's TURTLE jobs took 32 to 57 min each: write
the runner, prove it on a smoke slice, and hand back the full command (the controller queues it).

## Write
`r28_clay_crl.py` (or a thin wrapper per method), `test_r28_clay_crl.py` (the encoder switch and the replay),
outputs under `/data/SSD2/cosir_r28/20261112_siglip_main/clay_crl/`, small records `records/clay_crl.json`. GPU text
encoding is short (`res-queue gpu --no-wait --short --label r28:clay -- <cmd>`). In the hand-back, give each new cell
beside the current CLIP L/14 cell of the same method and the plain SigLIP 1 cell.
