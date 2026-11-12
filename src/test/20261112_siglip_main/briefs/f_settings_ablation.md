# r28 job F: scorers for the settings check and the ablation on SigLIP 1 (handoff §4 items 9 and 10). Verdict code.

Read first: `briefs/common.md`, the handoff's items 9 and 10, `src/test/20261027_recipe_sweep/DECISION_RULE.md`
§3.3 and §4.3 (stage 3), and `src/test/20261111_ablation_classcount/t5_ablation_ami.py` with its log.

The training runs do not exist yet (job T1 builds the trainer; the controller runs it). They will be written in r14's
layout under the root `R28 = /data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1/`: `R28/weights/<arm>/`,
`R28/train/<arm>/R6-5/<aspect>__<row>__s<seed>.npz|json` (validation export, 6,152 paintings),
`R28/held/<arm>/<aspect>__s<seed>.npz|json` (held export), arm = the row name (`k11=10`, `k11=20`, `k11=80`,
`k6=0.0005`, `instructed_graphs`, `content_correction`, `buddy`, `relational`). The reference c* (`ref`, arm `k11=40`)
is r14's, not retrained: root `R14 = /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/`, five seeds.
Build and test both scorers on R14's `k11=40` files now (they are the same format).

## Part 1, settings check (`r28_settings.py`)
Rows `k11=10`, `k11=20`, `k11=80`, `k6=0.0005` against c*, SigLIP 1, seed 0, validation only.
- Criterion: r10's stage 3 pick criterion (§3.3, §4.3: Ā, ē against the comparator, M̄ with r_a, the joint bootstrap,
  "Qualifies", "PICK", "KEEP"), unchanged except: the comparator is c* (`ref`, seed 0) in G's place, and the seed set
  is {0}. Work out whether the criterion's inputs exist for SigLIP 1 runs: P (29,232 rows: which rows are they? the
  validation captions?), the partitions ("the three states"), r_a. If P and the partitions can be built from the
  validation exports exactly as r10 built them, compute the criterion. If an input is missing, use the fallback the
  handoff names: the mean of the six validation cells (AMI at 9 / 27 / 10 and plain-cosine mAP, `r28_valpick.py`'s
  functions) with c* winning ties. Say which applies and why; that is a decision line.
- Output per row: its values, c*'s, the difference, and the outcome: "c* wins or ties" or "<row> wins".
- Test: c* against itself gives zero differences and "c* wins or ties"; one test of the bootstrap on a tiny fixture.

## Part 2, ablation scorer (`r28_ablation.py`)
Table 5 on SigLIP 1: per removal row and for c*, five seeds, ArtELingo test, AMI at the class counts 9 / 27 / 10 and
plain-cosine mAP, exactly as `t5_ablation_ami.py` and the paper's `Tables/ablation.tex` (read only) scored the Qwen3
version; seed means and per-seed values. Import t5's and r17's functions; change only the roots and the encoder.
Replay check: scoring R14's c* held exports gives the SigLIP 1 c* class-count AMI already in r17's records
(`src/test/20261103_new_scores/records/verdicts.json`; 0.099 / 0.441 / 0.528 rounded) and the plain-cosine mAP that
the Qwen3 t5 scoring would give on SigLIP 1 c* (report it).
The test split is scored here, never used to choose.

## Write
`r28_settings.py`, `r28_ablation.py`, `test_r28_settings.py` (and `test_r28_ablation.py` for the root and join), the c*
replay values in `records/ablation_ref_replay.json`. Hand back the two exact scoring commands the controller runs after
the training runs, and how long each takes.
