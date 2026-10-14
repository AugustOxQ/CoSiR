*CoSiR update #2 · 13 to 14 October 2026*
*Our current method gains only on emotion; without labels, no source we tried separates style from genre.*

Recap: CoSiR tests whether a model can rank images and captions by one aspect shown only through example pairs. Current best: one-sided affect steering (AFF), +0.60 [+0.46, +0.73] R@1 points over B′(A0), the strongest pre-registered condition-free score (held ArtELingo paintings). Project primer: <link to the start page on the dev-log site>.

*Since update #1:*
1. Learned groupings on frozen CLIP, and other image encoders in place of the style encoder: none beat AFF. (did not work)
2. AFF split by the aspect the example pairs show (development episodes): +2.88 over B′(A0) when they show emotion, −0.09 on style and −0.69 on genre. (check)
3. Fewer example pairs: AFF keeps its lead with 3 pairs and falls below B′(A0) with 1 (−1.73); the emotion gain holds at every count. (check)
4. Why style and genre fail: the style encoder's groups match the true style and genre labels about equally (AMI 0.34 and 0.33). (check)
5. Leiden instead of k-means for the image and caption groupings: a tie on the margin over B′(A0), kept as a record. (check)

Figure 1: `docs/reports/assets/2026-10-14_aff_scope_limits/csd_style_genre_ami.png`

*Implications:* the method depends on its feature source. It gains where an outside emotion classifier supplies the grouping, and nowhere else. This rests on learning without aspect labels, which also sets the baselines, the training target and how the condition is used.
*Next:* AFF as a fixed recipe on two more backbones and on more datasets, with results in the paper by 6 November.
*Feedback requested:* should the method train with aspect labels, so that style can be separated from genre?
*Full update:* <link to the page on the dev-log site; kept out of this public repo, see /project/cosir-devlog/deploys.log>
