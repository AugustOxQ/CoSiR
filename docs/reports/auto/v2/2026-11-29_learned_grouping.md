# Round 8: neither learned groupings nor the encoder screen passes; AFF stays best

> 2026-10-13 · Specs: [main line](../../../superpowers/specs/2026-10-12-r8-learned-grouping-design.md), [encoder screen](../../../superpowers/specs/2026-10-12-r8-encoder-screen-design.md) · [Decision rule](../../../../src/test/20261129_r8_grouping/DECISION_RULE.md) @ fdaee39 (amendment 1, SHA-256 18b4e063…) · [Run log](../../../../src/test/20261129_r8_grouping/20261129_r8_grouping_log.md)

## Results brief

**What we tried.** We learned new groupings on frozen CLIP features, each in CSD's place as the fourth grouping. A
separate screen tried fixed image encoders. Each grouping got a reader scorer and extended AFF (AFF that also admits
the new grouping), tested against AFF, our current best.

- **Amendment 1, made after a result** (2026-10-13 01:13): the view check failed at minimum cosine 0.99983 against
  0.9999, for 1 of 256 paintings. The user lowered the threshold and chose 0.999 after seeing the recommendation. The
  whole run restarted from stage 0, and the check passed. The verdict does not depend on it: all six view-grouping
  scorers fail the gate.
- **Time-cap drops:** the three largest fair-clustering (FCMI) weights, 0.40 to 1.60, hit the 60-minute cap (D21),
  1 to 3 epochs short. Only weights 0.05 to 0.20 were scored.

**Outcome against the bar.** No pick in either track, and the main line is not promising. Under rule §8, round 8 ends
at the gate and AFF stays current best. The bar: pooled lower bound above 0, and Σ(4d) ≥ −49 (about −0.30 R@1 points)
on both emotion pairs.

- Main line: all 20 scored scorers are below AFF. The best is extended AFF on FCMI weight 0.20: −0.466 [−0.651, −0.288] R@1 points.
- Screen: the DINOv2 reader is the closest to the bar: +0.285 [−0.012, +0.592] R@1 points, emotion × style Σ(4d) −63 (about −0.38).

**What it does not show.** This is development seed 42 only, after repeated selection on that seed, with no
multiplicity correction (rule C9). The unscored weights 0.40 to 1.60 remain unknown. The screen supports no method
claim. No held read was made.

**Options for next.** Round 8's held-read cut-off is Fri 2026-10-23 12:00 (rule §8); the CVPR abstract is due 2026-11-10.

- Close the grouping line and write the CVPR paper with AFF. Cost: paper work, no new experiment.
- A new loop on the DINOv2 reader. Cost: its own spec, rule and fresh seeds; several days (our estimate). It tests whether the miss persists on fresh seeds.
- A fair-clustering follow-up with a longer cap. Cost: a GPU night, scoring and a new rule (our estimate). It tests the weights 0.40 to 1.60.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The tested learned groupings fail clearly against AFF on this seed. DINOv2 is a narrow miss, not a
pass. A longer fair-clustering cap would complete the coverage, but the completed weights all lose to AFF, so they
give no trend towards a pass.

From the descriptive evidence: stronger own-group negatives remove style together with content. Fair clustering
carries little style despite low content agreement. DINOv2 retains content and genre; better style separation is not
established. Fitted CLIP placement makes the best main scorer worse.

**Our recommendation.** Close this round and write the paper with AFF. Keep DINOv2 as a later engineering question
that needs a new spec and fresh seeds. We would not spend the paper window on a longer fair-clustering sweep. The
user chooses the next idea.

</details>

## What ran

- **Stages:** stage 0 regression; stage 1 FCMI feasibility; learned groupings (new idea); fixed encoders (sweep); gate; post-verdict descriptive steps.
- **Data and seeds:** development seed 42 (scorer-train fits and selection episodes); no held row or held image read. Images take the grouping's own assignment; captions keep CLIP heads.
- **Code:** main checkout `fdaee39`, `src/test/20261129_r8_grouping/`, local GPU and CPU queues. Stage 0, stage 1 and every encoder extraction check passed.

| IDs used in the tables | Meaning |
|---|---|
| V100 / V67 / V33 | Contrastive head with view positives; own-content-group negative shares 100, 67, 33% |
| NB100 / NB67 / NB33 | Same shares, nearest CLIP neighbour outside the content group as positive |
| FC1 … FC6 | Fair clustering (FCMI); weight sweep below |
| NBC | Triggered nearest-CSD-neighbour head |
| AL / DI / SG | ALADIN-ViT from StyleBabel / DINOv2-L / SigLIP 2 |
| R1 / XAFF | Reader scorer / extended AFF |
| R0own / NRK / TP | Raw CSD with own assignment / NrKmeans on CLIP / told style prompts (privileged reference) |
| Z prefix / H prefix | Random grouping of matching sizes / CLIP-head placement arm |

| Time | Event |
|---|---|
| 2026-10-12 23:29 | First start |
| 2026-10-13 01:13 to 01:27 | Amendment 1; stop, then restart from stage 0 |
| 2026-10-13 09:06 | GPU queue done; its failed steps are the rule drops |
| 2026-10-13 11:20 | Gate verdict written |
| 2026-10-13 11:31 | CPU queue done; 81 of 81 steps ok |

## Results

No candidate passes the gate. Pass needs both: the SC3 pooled 95% lower bound strictly above 0, and SC4 Σ(4d) ≥ −49
on the emotion × style and the emotion × genre episodes.

SC3 pools 12,288 episodes. SC4 uses 4,096 episodes per pair.
For either statistic, R@1 points = Σ(4d) / n × 25, so the SC4 bar is about −0.299 R@1 points against AFF.
Dropped scorers have no gate estimate.

### Main line

| Scorer | SC3, R@1 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_V100 | −0.688 [−0.917, −0.459] | −165 | −204 | no |
| XAFF_V100 | −0.745 [−0.961, −0.529] | −180 | −228 | no |
| R1_V67 | −0.755 [−0.989, −0.522] | −189 | −220 | no |
| XAFF_V67 | −0.722 [−0.936, −0.510] | −187 | −234 | no |
| R1_V33 | −0.802 [−1.031, −0.572] | −175 | −225 | no |
| XAFF_V33 | −0.775 [−0.988, −0.556] | −184 | −224 | no |
| R1_NB100 | −0.643 [−0.878, −0.407] | −166 | −216 | no |
| XAFF_NB100 | −0.720 [−0.927, −0.507] | −186 | −223 | no |
| R1_NB67 | −0.696 [−0.926, −0.463] | −164 | −229 | no |
| XAFF_NB67 | −0.732 [−0.943, −0.519] | −178 | −214 | no |
| R1_NB33 | −0.584 [−0.817, −0.354] | −157 | −191 | no |
| XAFF_NB33 | −0.745 [−0.968, −0.526] | −186 | −238 | no |
| R1_FC1 | −0.488 [−0.726, −0.236] | −87 | −167 | no |
| XAFF_FC1 | −0.547 [−0.761, −0.329] | −101 | −180 | no |
| R1_FC2 | −0.576 [−0.808, −0.343] | −130 | −188 | no |
| XAFF_FC2 | −0.629 [−0.831, −0.419] | −120 | −193 | no |
| R1_FC3 | −0.478 [−0.706, −0.255] | −144 | −129 | no |
| XAFF_FC3 | −0.466 [−0.651, −0.288] | −111 | −130 | no |
| R1_FC4 to R1_FC6, XAFF_FC4 to XAFF_FC6 | — | — | — | Dropped: D21 wall-time cap |
| R1_NBC | −0.539 [−0.782, −0.296] | −128 | −203 | no |
| XAFF_NBC | −0.653 [−0.879, −0.437] | −143 | −241 | no |

### Encoder screen

| Scorer | SC3, R@1 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_AL | −0.303 [−0.553, −0.043] | −125 | −151 | no |
| XAFF_AL | −0.492 [−0.712, −0.268] | −142 | −210 | no |
| R1_DI | +0.285 [−0.012, +0.592] | −63 | 19 | no |
| XAFF_DI | +0.138 [−0.124, +0.410] | −69 | 19 | no |
| R1_SG | +0.059 [−0.262, +0.375] | −35 | −64 | no |
| XAFF_SG | −0.075 [−0.348, +0.196] | −82 | −104 | no |

### References and strongest random controls

These decide nothing.

| Scorer | SC3, R@1 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_R0own | −0.173 [−0.462, +0.126] | −10 | −150 | no |
| XAFF_R0own | −0.254 [−0.513, +0.000] | −49 | −167 | no |
| R1_NRK | −0.564 [−0.806, −0.308] | −81 | −235 | no |
| XAFF_NRK | −0.600 [−0.815, −0.382] | −99 | −224 | no |
| R1_TP | −0.549 [−0.799, −0.288] | −85 | −236 | no |
| XAFF_TP | −0.578 [−0.805, −0.355] | −133 | −237 | no |
| R1_ZNB33 (control, main line) | −0.537 [−0.779, −0.296] | −120 | −196 | no |
| R1_ZDI (control, screen) | −0.629 [−0.867, −0.391] | −132 | −217 | no |

XAFF_R0own's upper bound is +0.00005 and its style sum sits exactly on the bar. Each control row is the track's
highest SC3 point; by the style plus genre sum, R1_ZV67 (−305) is above R1_ZNB33 (−316) in the main line.

**Verdict under the rule** (§8, row "No pick in either track"): round 8 ends at the gate; AFF stays current best;
report. The main line is not promising, and both random-control flags are null. The screen's fallback record makes it
ineligible. No held read is made and no `HELD_RULE.md` is written.

## Analysis

Diagnostics decide nothing.
Every evaluated main-line scorer has a pooled interval wholly below AFF and fails the emotion-pair check.
The DINOv2 reader is the closest to the bar: its interval crosses zero and its style sum fails. Extended AFF on DINOv2
misses by a similar margin (style −69). The SigLIP 2 reader passes the style sum (−35) but fails genre (−64), and its
interval crosses zero.

All six CLIP-neighbour scorers failed, so the trigger selected their strongest scorer; the CSD probe failed too:

| Trigger field | Value |
|---|---|
| Source | R1_NB33 |
| Strength | 0.33 |
| Fired | yes; NBC ran, neither scorer passed |
| Independent re-derivations (C13) | Both agree; 0 disagreements in each (also `rederive/gate_rederive.json`) |

Our reading: content suppression did not give a useful style grouping (style AMI 0.010 to 0.261 for the learned
groupings, NBC included; 0.341 for raw CSD). AMI is adjusted mutual information, unscaled. Content AMI is label-free; style and genre
AMIs use labels after the verdict. Own-map agreement compares cross-fitted assignments with training groups. JSON
`heldout` means training halves, not held paintings.

| Grouping | Final groups | AMI: content (paintings) | AMI: style (rows) | AMI: genre (rows) | Own-map agreement (share) |
|---|---:|---:|---:|---:|---:|
| V100 | 17 | 0.087 | 0.054 | 0.066 | 0.594 |
| V67 | 17 | 0.158 | 0.096 | 0.137 | 0.544 |
| V33 | 17 | 0.273 | 0.184 | 0.237 | 0.498 |
| NB100 | 17 | 0.275 | 0.172 | 0.291 | 0.426 |
| NB67 | 17 | 0.317 | 0.179 | 0.327 | 0.464 |
| NB33 | 17 | 0.358 | 0.206 | 0.359 | 0.512 |
| FC1 | 17 | 0.014 | 0.010 | 0.010 | 0.999 |
| FC2 | 17 | 0.027 | 0.017 | 0.014 | 0.996 |
| FC3 | 16 | 0.072 | 0.040 | 0.046 | 0.996 |
| NBC | 17 | 0.273 | 0.261 | 0.200 | 0.612 |
| AL | 17 | 0.331 | 0.227 | 0.295 | 0.786 |
| DI | 17 | 0.435 | 0.209 | 0.449 | 0.835 |
| SG | 17 | 0.409 | 0.286 | 0.395 | 0.811 |
| R0own | 17 | 0.401 | 0.341 | 0.330 | 0.809 |
| NRK | 17 | 0.418 | 0.263 | 0.212 | 0.972 |
| TP | 23 | 0.300 | 0.286 | 0.183 | 1.000 |

The post-gate arm (`harm_run.json`) targets the strongest main scorer. Fitted CLIP image heads replace own assignment;
caption heads stay unchanged. Descriptive values were not independently re-derived.

| Descriptive target | SC3 versus AFF, R@1 points [95%] | SC4 style / genre sums | Pass |
|---|---:|---:|---|
| XAFF_HFC3 | −0.616 [−0.834, −0.406] | −147 / −215 | no |

## Deviations

- **Amendment 1:** made after a result; facts and ownership are in the brief. Rule §7's flags do not list amendments,
  so the brief names it first. The same minimum passed the new bar. The rule's report flags list no view drop; the old
  run's records are in `results_m4_09999/`.
- **Time cap (D21):** the kills also fail the process checks. These are mechanical drops, not retrieval results. The GPU
  queue logs them as failed steps (exit 247); the gate went on under the rule (M7, M16).

| Setting | Weight | Wall time, s (rounded) | Epochs recorded at stop / planned | Result |
|---|---:|---:|---:|---|
| FC1 | 0.05 | 3,096 | 300 / 300 | completed |
| FC2 | 0.10 | 3,133 | 300 / 300 | completed |
| FC3 | 0.20 | 3,330 | 300 / 300 | completed |
| FC4 | 0.40 | 3,601 | 299 / 300 | dropped: cap |
| FC5 | 0.80 | 3,601 | 297 / 300 | dropped: cap |
| FC6 | 1.60 | 3,601 | 299 / 300 | dropped: cap |

The retained sweep covers weights 0.05 to 0.20 only. Run time grows with the weight. The build estimated about 54.5
minutes for the 0.20 weight alone (measured 55.5); it did not estimate the larger weights. Loss logs show partial
final epochs for FC4 and FC6; "recorded" does not mean complete.

**Report flags (rule §7, `gate_verdict.json.report_flags`).** ALADIN uses the released StyleBabel ViT weights, rather than the screen spec's older ICCV model. Final counts are tabulated: stage-one fair clustering merged below target; prompts keep their count. FCMI uses unit shared-encoder representations and GPU training. Declared agent defaults.

**Known gaps.** Fair clustering "also reported as a comparator" (spec D2) is not computed or enforced separately; the
candidate rows give the available comparison. The encoder budget "at most 3 h" (ED18) is not enforced; the full
extraction passes took 147 s (AL), 547 s (DI) and 2,540 s (SG).
The joins from the FCMI and encoder steps to the grouping step first ran at full size in this run; their identity and
order assertions passed.

**Minor final-review caveats.** Coverage limits, with no observed gate disagreement:

- The regression checks another recipe entry point. The reviewer re-derived the actual builder: identical to raw CSD.
- Unit tests missed mutations to episode alignment and extended-AFF admission; runtime re-derivation and known-answer checks cover them.
- A non-weight encoder crash would block the main line as well as the screen. No such crash occurred here.
- Descriptive steps lack batch-hash checks against verdict inputs (write-once files reduce the risk) and mostly lack direct tests.
- Placement-to-label provenance is unchecked. Re-derivation covers images, not fitted text heads.
- The review warned about the 60-minute cap; the three larger weights hit it.

Sources (local, outside git): [first review](../../../../.scratch/r8-grouping/notes/final_review.md), [second review](../../../../.scratch/r8-grouping/notes/final_review_2.md), [claim review of this report](../../../../.scratch/r8-grouping/notes/claim_review.md).

## Files and storage

All outputs remain; the results trees are untracked (local, outside git).

- **Results:** [`results/`](../../../../src/test/20261129_r8_grouping/results/) (7.2 GB): verdict, trigger, stage checks, groupings, heads, readers, seed-42 arrays, re-derivations, logs, descriptive JSON. The old run's `results_m4_09999/` (2.1 GB) sits beside it.
- **Views:** `/data/SSD2/pre_extract/artelingo/r8_clip_views/views.npy` with metadata. Encoder caches: `/data/SSD2/pre_extract/artelingo/r8_enc_{aladin,dinov2l,siglip2}/`.
- **FCMI checkpoints:** `/data/SSD2/cosir_r8/fcmi/` FC1 to FC6 (about 27 GB), the amendment archives (8 GB) and the smoke folder (66 GB). The third weight's feasibility result is `results/stage1_fcmi.json`; there is no separate `fcmi/FC3/d21.json`.
- **Storage:** every item is on `.scratch/pending_deletions.md`. The next wrap-up's storage summary decides them with the user and records them in `/project/claude-config/storage/disk_ledger.md`.
