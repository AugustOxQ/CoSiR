# Round 8: neither learned groupings nor the encoder screen passes; AFF stays best

> 2026-10-13 · Specs: [main line](../../../superpowers/specs/2026-10-12-r8-learned-grouping-design.md), [encoder screen](../../../superpowers/specs/2026-10-12-r8-encoder-screen-design.md) · [Binding decision rule](../../../../src/test/20261129_r8_grouping/DECISION_RULE.md) (amended hash recorded in verdict) · [Run log](../../../../src/test/20261129_r8_grouping/20261129_r8_grouping_log.md)

## Results brief

**What we tried.** We tested learned groupings on frozen CLIP features and a separate screen of fixed encoders against AFF, our current best.
Amendment 1, made after results, lowered the view-check threshold from 0.9999 to 0.999.
The real check failed at minimum cosine 0.99983, for 1 of 256 paintings.
The user chose to lower it; the user chose the new value after seeing the recommendation.
The whole run restarted from stage 0; the new check passed and all view groupings stayed in.
The three largest fair-clustering weights were dropped by the 60 minute wall-time cap, near completion. Only low weights were scored.

**Outcome against the bar.** No pick in either track; the main line is not promising under the rule.
The DINOv2 reader scored +0.285 [−0.012, +0.592] R@1 points against AFF.
Its emotion × style sum was -63 against a bar of -49.
The rule ends the round at the gate. AFF stays best; no held read occurs and no `HELD_RULE.md` is written.

**What it does not show.** This is development seed 42 only, after repeated selection.
The unscored high-weight settings remain unknown. The screen supports no method claim.

**Options for next.**

- Close this grouping line and put the time into the CVPR paper with AFF. Cost: paper work, with no new experiment. It would consolidate the supported result.
- Start a new loop around the DINOv2 reader, the only scorer near the bar. Cost: its own spec, rule, fresh seeds and several days of work (our estimate). It tests whether the miss persists.
- Start a fair-clustering follow-up with a longer cap. Cost: another GPU night plus scoring and a new rule (our estimate). It completes the high weights. The completed weights all lose to AFF, so they give no trend towards a pass.

The [direction log](../../../superpowers/direction_log.md) took up both parked ideas here; a repeat needs a new loop. Any paper-facing held verdict must meet the 2026-10-23 cut-off; the CVPR abstract date in that plan is 2026-11-10.

**Your reading.** Pending; no result-reading decision from the user is recorded here.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The tested learned groupings are a clear failure against AFF on this seed. DINOv2 is a narrow miss, not a pass.
A longer fair-clustering cap would repair coverage, but the low-weight scores give weak grounds to expect success.

Our reading: stronger own-group negatives remove style with content. Fair clustering carries little style despite low content agreement. DINOv2 retains content and genre; better style separation is not established. Fitted CLIP placement worsens the source’s pooled point. These are our readings of descriptive evidence.

**Our recommendation.** Close this round and write the paper with AFF. Keep DINOv2 as a later engineering question that needs a new spec and fresh seeds.
We would not spend the remaining paper window on a longer fair-clustering sweep. The user chooses the next idea.

</details>

## What ran

- **Stages/types:** regression; feasibility; learned groupings (new idea); fixed encoders (engineering sweep); gate; post-verdict diagnostics. Research-loop steps 8 and 9 finished on the header date.
- **Data:** scorer-train fits and selection episodes. Images use own assignment; captions keep CLIP heads. Each grouping gets a reader scorer and extended AFF.
- **Code:** restarted from main checkout `fdaee39` in `src/test/20261129_r8_grouping/`, on the local GPU and CPU queues. Regression, stage-one feasibility, and all encoder extraction checks passed.

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
| 2026-10-13 01:14 | Stopped for amendment |
| 2026-10-13 01:27 | Restart |
| 2026-10-13 09:06 | GPU queue done; failed steps were the rule drops |
| 2026-10-13 11:20 | Gate verdict written |
| 2026-10-13 11:31 | CPU queue done; 81 of 81 steps ok |

## Results

No candidate passes the binding gate. The pooled difference must have its 95% lower bound strictly above 0; both emotion-pair sums must reach the bar stated above.

SC3 pools 12,288 episodes. SC4 uses 4,096 episodes per pair.
For either statistic, R@1 points = sum4d / n × 25.
Thus the SC4 bar is about −0.299 R@1 points against AFF.
Dropped scorers have no gate estimate.

### Main line

| Scorer | SC3 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_V100 | −0.688 [−0.917, −0.459] | -165 | -204 | no |
| XAFF_V100 | −0.745 [−0.961, −0.529] | -180 | -228 | no |
| R1_V67 | −0.755 [−0.989, −0.522] | -189 | -220 | no |
| XAFF_V67 | −0.722 [−0.936, −0.510] | -187 | -234 | no |
| R1_V33 | −0.802 [−1.031, −0.572] | -175 | -225 | no |
| XAFF_V33 | −0.775 [−0.988, −0.556] | -184 | -224 | no |
| R1_NB100 | −0.643 [−0.878, −0.407] | -166 | -216 | no |
| XAFF_NB100 | −0.720 [−0.927, −0.507] | -186 | -223 | no |
| R1_NB67 | −0.696 [−0.926, −0.463] | -164 | -229 | no |
| XAFF_NB67 | −0.732 [−0.943, −0.519] | -178 | -214 | no |
| R1_NB33 | −0.584 [−0.817, −0.354] | -157 | -191 | no |
| XAFF_NB33 | −0.745 [−0.968, −0.526] | -186 | -238 | no |
| R1_FC1 | −0.488 [−0.726, −0.236] | -87 | -167 | no |
| XAFF_FC1 | −0.547 [−0.761, −0.329] | -101 | -180 | no |
| R1_FC2 | −0.576 [−0.808, −0.343] | -130 | -188 | no |
| XAFF_FC2 | −0.629 [−0.831, −0.419] | -120 | -193 | no |
| R1_FC3 | −0.478 [−0.706, −0.255] | -144 | -129 | no |
| XAFF_FC3 | −0.466 [−0.651, −0.288] | -111 | -130 | no |
| R1_FC4 | — | — | — | Dropped: D21 wall-time cap |
| XAFF_FC4 | — | — | — | Dropped: D21 wall-time cap |
| R1_FC5 | — | — | — | Dropped: D21 wall-time cap |
| XAFF_FC5 | — | — | — | Dropped: D21 wall-time cap |
| R1_FC6 | — | — | — | Dropped: D21 wall-time cap |
| XAFF_FC6 | — | — | — | Dropped: D21 wall-time cap |
| R1_NBC | −0.539 [−0.782, −0.296] | -128 | -203 | no |
| XAFF_NBC | −0.653 [−0.879, −0.437] | -143 | -241 | no |

### Encoder screen

| Scorer | SC3 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_AL | −0.303 [−0.553, −0.043] | -125 | -151 | no |
| XAFF_AL | −0.492 [−0.712, −0.268] | -142 | -210 | no |
| R1_DI | +0.285 [−0.012, +0.592] | -63 | 19 | no |
| XAFF_DI | +0.138 [−0.124, +0.410] | -69 | 19 | no |
| R1_SG | +0.059 [−0.262, +0.375] | -35 | -64 | no |
| XAFF_SG | −0.075 [−0.348, +0.196] | -82 | -104 | no |

### References and strongest random controls

These decide nothing; controls shown are the pooled best per track.

| Scorer | SC3 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_R0own | −0.173 [−0.462, +0.126] | -10 | -150 | no |
| XAFF_R0own | −0.254 [−0.513, +0.000] | -49 | -167 | no |
| R1_NRK | −0.564 [−0.806, −0.308] | -81 | -235 | no |
| XAFF_NRK | −0.600 [−0.815, −0.382] | -99 | -224 | no |
| R1_TP | −0.549 [−0.799, −0.288] | -85 | -236 | no |
| XAFF_TP | −0.578 [−0.805, −0.355] | -133 | -237 | no |

| Scorer | SC3 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_ZNB33 | −0.537 [−0.779, −0.296] | -120 | -196 | no |

| Scorer | SC3 points [95%] | SC4 style Σ4d | SC4 genre Σ4d | Pass |
|---|---:|---:|---:|---|
| R1_ZDI | −0.629 [−0.867, −0.391] | -132 | -217 | no |

**Verdict under the rule:** “No pick in either track: Round 8 ends at the gate; AFF stays current best; report.”
The main line is not promising; both random-control flags are null without a pick.
The fallback record makes the screen ineligible; `held_rule_required` does not activate a read without a pick.

## Analysis

Diagnostics decide nothing.
Every evaluated main-line scorer has a pooled interval wholly below AFF and fails the emotion-pair check.
DINOv2 alone is near the bar; its interval crosses zero and its style sum fails. SigLIP’s positive point also fails.

All six CLIP-neighbour scorers failed, so the trigger selected their strongest scorer; the CSD probe failed too:

| Trigger field | Value |
|---|---|
| Source | R1_NB33 |
| Strength | 0.33 |
| Fired | yes; NBC ran, neither scorer passed |
| Independent re-derivations (C13) | Both agree; 0 disagreements in each (also `rederive/gate_rederive.json`) |

Content suppression did not give a useful style grouping. AMI is adjusted mutual information, unscaled. Content AMI is label-free; style/genre AMIs use labels after the verdict.
Own-map agreement compares cross-fitted assignments with training groups. JSON `heldout` means training halves, not held paintings.

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


The post-gate arm (`harm_run.json`) targets the strongest main scorer. Fitted CLIP image heads replace own assignment; caption heads stay unchanged. Descriptive values were not independently re-derived.

| Descriptive target | SC3 versus AFF, R@1 points [95% interval] | SC4 style / genre sums | Pass |
|---|---:|---:|---|
| XAFF_HFC3 | −0.616 [−0.834, −0.406] | -147 / -215 | no |


## Deviations

- **Amendment 1:** after-results change and ownership are in the brief. The same minimum passed. `report_flags.V_drops` is empty; old records remain in `results_m4_09999/`.
- **Time cap (D21):** kills also fail process and cluster checks. These are mechanical drops, not retrieval results. The GPU queue calls them failed steps; the gate proceeded under the rule.

| Setting | Weight | Wall time, s (rounded) | Epochs recorded at stop / planned | Result |
|---|---:|---:|---:|---|
| FC1 | 0.05 | 3,096 | complete | completed |
| FC2 | 0.10 | 3,133 | complete | completed |
| FC3 | 0.20 | 3,330 | complete | completed |
| FC4 | 0.40 | 3,601 | 299 / 300 | dropped: cap |
| FC5 | 0.80 | 3,601 | 297 / 300 | dropped: cap |
| FC6 | 1.60 | 3,601 | 299 / 300 | dropped: cap |

The retained sweep therefore covers weights 0.05 to 0.20 only.
Runtime rises across the completed weights. The build estimate was about 55 minutes,
from the stage-one weight only; the run log gives 54.5 minutes more precisely.
It did not estimate the whole sweep.
Epoch counts follow the log. Loss logs show partial final epochs for the first and last drops; “recorded” does not mean complete.

**Report flags (rule §7, `gate_verdict.json.report_flags`).** ALADIN uses the released StyleBabel ViT weights, rather than the screen spec’s older ICCV model. Final counts are tabulated: stage-one fair clustering merged below target; prompts keep their count. FCMI uses unit shared-encoder representations and GPU training. Declared agent defaults.

**Known gaps.** Fair clustering “also as a comparator” (D2) is not separately computed or enforced. Candidate rows give the available comparison; no standalone retrieval comparator ran.
The encoder budget “at most 3 h” (ED18) is not enforced.
The measured full-pass times were:

| Encoder | Full extraction pass, s (rounded) |
|---|---:|
| AL | 147 |
| DI | 547 |
| SG | 2,540 |

Pass times do not enforce a budget; extraction checks passed.
The FCMI-to-grouping and encoder-to-grouping joins (06 → 07 and 04 → 07) first ran at full size in this run; their identity and order assertions passed.

**Minor final-review caveats.** Coverage limits, with no observed gate disagreement:

- The regression checks another recipe entry point. The reviewer re-derived the actual builder: identical to raw CSD.
- Unit tests missed mutations to episode alignment and extended-AFF admission; runtime re-derivation and known-answer checks cover them.
- A non-weight encoder crash would block the main line as well as the screen. No such crash occurred here.
- Descriptive index globs lack batch-hash checks against verdict inputs. Write-once files reduce the risk.
- Most descriptive calculations lack direct tests; the real-layout smoke supplies limited coverage.
- Placement-to-label provenance is unchecked. Re-derivation covers images, not fitted text heads.
- The review's FCMI timing risk materialised as the cap drops above.

Sources: [minor rows, first review](../../../../.scratch/r8-grouping/notes/final_review.md) and [second review](../../../../.scratch/r8-grouping/notes/final_review_2.md).

## Files and storage

Outputs remain.

- **Results:** [`results/`](../../../../src/test/20261129_r8_grouping/results/): verdict, trigger, stage checks; `fcmi/`, `groupings/`, heads/readers; `seed42/` arrays/indices; `rederive/`, logs, descriptive JSON and `harm_run.json`. Details stay in the run log.
- **Views:** `/data/SSD2/pre_extract/artelingo/r8_clip_views/views.npy` with metadata. Encoder caches: `/data/SSD2/pre_extract/artelingo/r8_enc_{aladin,dinov2l,siglip2}/`.
- **FCMI checkpoints:** `/data/SSD2/cosir_r8/fcmi/`, completed/capped runs and amendment archives. The third weight's completed feasibility result is `results/stage1_fcmi.json`; there is no separate `fcmi/FC3/d21.json`.
- **Storage:** retain for review; see `/project/claude-config/storage/disk_ledger.md`. Ledger unchanged.
