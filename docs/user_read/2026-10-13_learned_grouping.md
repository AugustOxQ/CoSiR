# Learned groupings did not beat AFF; the encoder screen also ends at the gate

> 2026-10-13 · Full report: [Round 8, learned groupings](../reports/auto/v2/2026-11-29_learned_grouping.md) (its file
> name uses CoSiR's sequence date). Why we tried it: [the round 8 spec](../superpowers/specs/2026-10-12-r8-learned-grouping-design.md), brief.

## Summary

- No scorer in either track passed the gate. AFF stays our current best method.
- After its first real result, you lowered the bar of one input check. The run restarted and the check passed. The verdict does not depend on it.
- The three largest fair-clustering weights hit the 60-minute training cap. Their retrieval results are unknown.
- The DINOv2 reader had the highest pooled score. But its interval crosses zero, and it loses too much on emotion × style.
- All results are on development seed 42, after many earlier tries on that seed. No held data was read in this round.

## How we got here

CoSiR ranks captions for a painting. A few example pairs show the aspect that matters, such as emotion, style or
genre, without its name. A **reader** guesses which grouping of paintings the examples share. **AFF** (affect
steering), our current best, uses the reader's guess only when it picks affect.

Round 7 showed a large possible gain on style × genre episodes when the scorer was told the true styles. Your reading
was that CSD, our style encoder, mixes style with genre. So round 8 learned a new grouping that cuts across content,
and put it in CSD's place as the fourth grouping.

Round 8 had two tracks: learned groupings (the main line) and a screen of fixed image encoders. Both had to pass one
gate, fixed in a decision rule before the run. The run took one night, from 2026-10-12 23:29 to 2026-10-13 11:31,
with one restart (below).

## Learned groupings on frozen CLIP

**Problem.** A grouping that mixes style with other aspects can steer the scorer towards the wrong aspect.

**Idea.** We tried two ways to learn a grouping from frozen CLIP features:

- A **contrastive head**. Its positives were flips and crops of the same painting, or the nearest painting with other
  content. A fixed share of its negatives (100%, 67% or 33%) came from the query's own content group.
- **Fair clustering**. It treats the content groups as a protected attribute and pushes the new clusters to be
  independent of them. We swept its fairness weight from 0.05 to 1.60.

Each painting went to the nearest group centre in the new space. Each new grouping got two scorers: a reader scorer,
and **extended AFF**, which also uses the new grouping when the reader picks it. Random groupings of the same sizes
served as controls.

**Did it work?** No. Each scorer had to beat AFF on all episodes pooled, with the 95% interval wholly above zero. It
also had to keep its loss on the emotion × style and emotion × genre episodes within about 0.30 R@1 points. Every
scored learned grouping failed both checks. The rule's weaker "promising" test also failed, so the main line ends here.

One input check changed after its real result. Before it built the flips and crops, the run checked that 256 stored
images give back their stored CLIP features (cosine at least 0.9999). One painting gave 0.99983.

You read this as noise and lowered the bar; you chose 0.999 after seeing our recommendation. The whole run restarted,
and the check passed. The verdict does not depend on this: all six scorers on the flip-and-crop groupings fail the gate.

The three largest fair-clustering weights (0.40, 0.80 and 1.60) hit the 60-minute training cap a few epochs before the
end. Training took longer at larger weights. So only weights 0.05 to 0.20 were scored. Their failure does not settle
the larger weights.

**Key evidence.** The best learned grouping still scored below AFF.

| Pooled R@1 points against AFF, 95% interval | Result |
|---|---:|
| Best learned grouping: extended AFF on fair clustering, weight 0.20 | −0.466 [−0.651, −0.288] |
| The same, with paintings placed by fitted CLIP heads (a diagnostic) | −0.616 [−0.834, −0.406] |

All six scorers on the nearest-neighbour groupings failed, so a planned probe ran with CSD neighbours instead of CLIP neighbours. It
failed too. Two independent recomputations of the probe's trigger and of the verdict agreed with the pipeline.

The diagnostics measure how much each grouping shares with content, style and genre. They give no verdict; our
reading of them is in the advice below.

> Full report: Results, Analysis and Deviations. The threshold change and the cap drops are given there in full.

## Fixed encoders as another source

**Problem.** The main line kept CLIP as the image encoder. Another encoder could give a better grouping with no
learned head.

**Idea.** We screened three fixed encoders: ALADIN-ViT (a style encoder from StyleBabel), DINOv2-L and SigLIP 2. Each
gave features to the same grouping recipe, and each grouping got the same two scorers. It was an engineering screen and
supports no method claim.

**Did it work?** No. The DINOv2 reader had the highest pooled point, above AFF, but its interval crosses zero. It also
lost too much on emotion × style. The screen could go to a held read only with a pick and a failed main line. There
was no pick.

**Key evidence.** The DINOv2 reader missed both checks.

| DINOv2 reader against AFF | Result | Required |
|---|---:|---|
| Pooled R@1 points, 95% interval | +0.285 [−0.012, +0.592] | lower bound above zero |
| Emotion × style, sum of score differences | −63 (about −0.38 R@1 points) | at least −49 (about −0.30) |

> Full report: Encoder screen in Results, Analysis, and the report flags in Deviations.

<details>
<summary>Our advice (open after giving your reading)</summary>

This is our view; you decide the next idea.

Our reading of the diagnostics: separating the groups from content did not give a useful style grouping. Stronger
negatives from the query's own content group removed style together with content. Fair clustering carried little style, although it shared little with content. DINOv2
keeps content and genre, and better style separation is not shown as the cause of its score.

Close this round and put the time into the paper with AFF. The tested learned groupings are clearly below AFF on this
seed. The DINOv2 reader missed the bar narrowly and does not show a passing method.

A later DINOv2 loop could test whether the miss persists. It needs its own spec, rule and fresh seeds. A longer
fair-clustering cap would complete the missing weights, but the completed weights give weak grounds to expect a pass.

</details>

## Next

This round ends at the gate under its decision rule. AFF stays the current best. No held data was read in this round,
and no held-read rule was written. Your reading of the results is still to come.

You choose the next step. The full report gives three options:

- Close the grouping line and write the paper with AFF. Cost: writing time, no new experiment.
- A new loop on the DINOv2 reader. Cost: a new spec, rule and fresh seeds; several days (our estimate).
- Fair clustering with a longer cap. Cost: another GPU night, scoring and a new rule (our estimate).

None is started. A paper-facing held verdict must come by the 2026-10-23 cut-off. The CVPR abstract is due 2026-11-10.

## Glossary

- **AFF:** affect steering, our current best scorer.
- **Reader:** the model that guesses which grouping the example pairs share; its scorer is R1 in the full report.
- **Extended AFF:** AFF that also uses the new grouping; XAFF in the full report.
- **CSD:** our earlier style encoder; its neighbour probe is NBC in the full report.
- **Content group:** paintings with similar content; the new groupings try to cut across them.
- **Contrastive head:** V100, V67, V33 (flip and crop positives) and NB100, NB67, NB33 (neighbour positives).
- **Fair clustering:** clusters made independent of content groups; FCMI in the full report, weights FC1 to FC6.
- **Gate:** the two development checks a scorer must pass to go on to held data; SC3 (pooled) and SC4 (emotion pairs).
- **Decision rule:** the bars and steps fixed before the run; changed after results only by you (amendment 1 here).
- **R@1:** share of rankings with the target strictly first, in percentage points.
- **Seed 42:** our development episodes. Fresh seeds are kept for a held read.
- **Held read:** a pre-registered test on held paintings, after a scorer passes the gate.
