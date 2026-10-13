# Learned groupings did not beat AFF; the encoder screen also ends at the gate

> 2026-10-13 · Full report: [Round 8, learned groupings](../reports/auto/v2/2026-11-29_learned_grouping.md).
> The full report's file name uses CoSiR's sequence date.

## Summary

- Neither track produced a passing scorer. AFF remains our current best method.
- The user lowered the image check threshold after its real failure. The whole run restarted and the check passed.
- The three largest fair-clustering weights hit the time cap. Their retrieval results remain unknown.
- DINOv2 gave the closest result, but failed both the pooled and emotion × style checks.
- This was development seed 42 only. There was no held read, so this round supports no new test claim.

## How we got here

CoSiR ranks captions for a painting from example pairs. The examples show an aspect, such as emotion, style or genre, without naming it.
A **reader** infers which grouping the examples share. **AFF**, or affect steering, adds the reader's term only when it picks affect.

AFF remains best, but its style × genre result leaves room to improve. Earlier work suggested that the style encoder mixes style with content.
This round asked whether a different grouping could help the scorer use style without harming the emotion pairs.

The first run started at 2026-10-12 23:29. It stopped at 2026-10-13 01:14 for the threshold amendment and restarted at 01:27.
The GPU queue ended at 09:06. The gate verdict was written at 11:20; the CPU queue ended at 11:31 with every step ok.

> Source: full report, What ran and Deviations.

## Learned groupings on frozen CLIP

**Problem.** A grouping that mixes style with content can steer the scorer towards the wrong aspect.

**Idea.** We tried two ways to learn a grouping from frozen CLIP features. The first used a contrastive head.
Its positives were views of the same painting, or a nearby painting from another content group.
Its negatives came in different shares from the query's own content group.

The second used **fair clustering**. It treated content groups as a protected attribute and tried to make the new clusters independent of them.
We swept its fairness weight. Both builders replaced the style grouping; the rest of the scorer used the existing pipeline.

An image went to the nearest group centre in the builder's space. Captions kept their CLIP heads.
Each grouping got a reader scorer and **extended AFF**, which also admits the new grouping when the reader picks it.
Random groupings of matching sizes served as controls.

**Did it work.** No tested scorer passed. Each had to beat AFF pooled, with its interval wholly above zero.
It also had to keep both emotion pairs within the rule's allowed loss. Every scored learned grouping failed both checks.
The rule's weaker “promising” conditions failed too.

The image check changed after results. The real check's minimum cosine was 0.99983, against the original threshold of 0.9999.
One painting out of 256 failed. The user chose to lower the threshold and accepted 0.999 after seeing the recommendation.
The whole run restarted from the regression stage. The check then passed, so the view groupings stayed in.

The three largest fair-clustering weights stopped at the 60 minute cap, near the final epoch.
Only weights 0.05 to 0.20 reached scoring. This limits coverage: failure of the tested weights does not settle the untested weights.

**Key evidence.** The strongest completed learned-grouping scorer still scored below AFF.
The diagnostic change to CLIP image heads also scored below AFF. All values below are pooled R@1 points against AFF.

| Comparison | Difference from AFF |
|---|---:|
| Best learned grouping: extended AFF with fair clustering | −0.466 [−0.651, −0.288] |
| Same grouping with fitted CLIP image heads | −0.616 [−0.834, −0.406] |

The nearest-CLIP-neighbour groupings all failed, so the planned probe with CSD neighbours ran. It failed too.
Independent checks before the probe and before the verdict both agreed with the pipeline.

The diagnostics offer a possible explanation, not another verdict. The view heads removed style agreement together with content agreement.
Fair clustering removed more content, but its style agreement was low. Better separation from content alone did not give a useful grouping.

> Full report: Results, Analysis, and Deviations. The threshold amendment and time-cap drops are disclosed there in full.

## Fixed encoders as another source

**Problem.** Learning on CLIP might limit the available style signal. Another encoder could give a better grouping without a learned head.

**Idea.** We screened ALADIN-ViT from StyleBabel, DINOv2-L and SigLIP 2. Each fixed encoder supplied features for the same grouping recipe.
Each grouping then got the same two scorers. This was an engineering screen, not a new method claim.

**Did it work.** No encoder produced a passing scorer. The DINOv2 reader came closest.
Its pooled point was above AFF, but its interval crossed zero. It also lost too much on emotion × style.
The screen could proceed to held data only with a pick and a failed main line. There was no pick.

**Key evidence.** The closest scorer missed two independent checks.

| DINOv2 reader against AFF | Result | Required |
|---|---:|---|
| Pooled R@1 points, with interval | +0.285 [−0.012, +0.592] | Lower bound above zero |
| Emotion × style, integer sum of differences | −63 | At least −49 |

The second row uses the rule's integer units, not R@1 points. The full report gives the conversion and every scorer.
DINOv2 retained strong agreement with content and genre. The diagnostics do not show that better style separation caused its score.

> Full report: Encoder screen in Results, Analysis, and report flags in Deviations.

<details>
<summary>Our advice (open after giving your reading)</summary>

This is our view; you decide the next idea.

Close this round and put the time into the paper with AFF. The tested learned groupings are clearly below AFF on this seed.
The DINOv2 result is a narrow miss, not evidence of a passing method.

A later DINOv2 loop could test whether the near miss persists. It needs its own spec, rule and fresh seeds.
A longer fair-clustering cap would complete the missing settings, but the completed weights give weak grounds to expect a pass.

</details>

## Next

This round ends at the gate under its binding rule. AFF remains current best. No held data was read and no held-read rule was written.
Your reading of the results is still pending.

Your next choice is to close this grouping line and write the paper with AFF, start a new DINOv2 loop, or test fair clustering with a longer cap.
The paper option needs writing time. Either experiment option needs a new rule and another period of build and run work.
The full report gives costs and the existing paper cut-off. None of these options has been started here.

## Glossary

- **AFF:** affect steering, our current best scorer.
- **Reader:** model that infers which grouping the example pairs share; R1 is its scorer.
- **Extended AFF:** AFF whose gate also admits the new style grouping; full report: XAFF.
- **CSD:** the existing style encoder; its neighbour probe is NBC in the full report.
- **Fair clustering:** clusters made less dependent on content groups; full report: FCMI, settings FC1 to FC6.
- **Own assignment:** an image takes the nearest group centre in the builder's space.
- **Gate:** development checks that decide whether a candidate can proceed to held data; full report: SC3 and SC4.
- **R@1:** share of rankings with the target strictly first, in percentage points.
- **Held read:** a pre-registered test on held paintings, after a candidate passes the development gate.
