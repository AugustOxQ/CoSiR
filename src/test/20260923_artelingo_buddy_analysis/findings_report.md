# ArtELingo buddy-graph vs. ground-truth-label analysis
Generated automatically overnight, 2026-09-22 12:40:27.

**Setup:** 61402 unique painting-nodes (deduped from 308723 English caption-rows), buddy graph K=20, alpha=0.5, Leiden community detection (28 communities found), seed=42.
**Caveat on K:** chosen as a smaller-than-default judgment call for this node count (~61k, below any scale Experiment 16.1 validated K for) — not itself validated. Revisit if results look off.

## 1. Genre <-> Emotion ground-truth correlation (prerequisite baseline)
n = 1144 paintings with genre labels.
AMI(genre, emotion) = 0.0723, V-measure = 0.0865, ARI = 0.0776
Chi-square(genre, emotion): chi2=421.07, dof=64, p=5.53e-54
Interpretation guide: if AMI here is already substantial, any 'buddy predicts emotion' result below must be read alongside 'buddy predicts genre' before concluding buddy structure carries affect specifically.

## 2. Buddy community vs. genre / emotion / valence
Community vs. genre (n=1144): AMI=0.4384, V-measure=0.4572, ARI=0.3257
Community vs. emotion (n=61402, full graph): AMI=0.0593, V-measure=0.0600, ARI=0.0271
Community vs. coarse valence (n=61402, 3-way pos/neg/other): AMI=0.0270, V-measure=0.0272, ARI=0.0031

## 3. Content-stratified test: does community still separate emotion WITHIN genre?
(The sharpest test — controls for genre so a positive result can't just be content leaking through.)

| genre | n | AMI(community, emotion) within genre |
|---|---:|---:|
| landscape | 249 | 0.0608 |
| portrait | 243 | 0.0510 |
| genre_painting | 196 | 0.0462 |
| religious_painting | 116 | 0.0248 |
| abstract_painting | 106 | -0.0114 |
| cityscape | 88 | 0.0663 |
| sketch_and_study | 71 | 0.0300 |
| still_life | 40 | 0.1369 |
| illustration | 35 | 0.1199 |

## 4. Rare-class (anger) behavior under buddy smoothing
n(anger, full graph majority-vote) = 422 / 61402
Anger paintings' most common community: id=0, holds 148/422 of all anger paintings (35.1%), while that community holds 11.1% of ALL paintings — concentrated (above base rate).
Interpretation: concentration well above the community's overall size share means buddy structure is NOT just diluting the rare class into arbitrary neighbors — worth a closer look either way.

## 5. Cross-lingual emotion consistency (English buddy graph vs. Arabic/Chinese labels)
Arabic (n=59221): community vs. Arabic emotion AMI=0.0572, V-measure=0.0579  |  community vs. English emotion on SAME subset: AMI=0.0591
Chinese (n=61402): community vs. Chinese emotion AMI=0.0603, V-measure=0.0611  |  community vs. English emotion on SAME subset: AMI=0.0593

Interpretation: if AMI(community, other-language emotion) is comparable to AMI(community, English emotion) on the same paintings, that's evidence buddy structure captures something language-independent about the image, not an English-caption-vocabulary artifact.

## 6. Robustness check: majority-vote tie sensitivity

Codex's independent methodology review (`codex_methodology_review.md`) correctly
flagged that `Counter.most_common(1)` resolves ties (e.g. a 2-2-1 emotion split
across a painting's ~5 annotations) by first-occurrence order rather than
detecting them. 15,926 / 61,402 paintings (25.9%) have a tied majority emotion;
23.3% of the genre-labeled subset (266/1144) is tied too.

Restricting Section 2's "community vs. emotion" comparison to the 45,476
non-tied paintings only: AMI=0.0758, V-measure=0.0768 (vs. AMI=0.0593,
V-measure=0.0600 on the full 61,402, ties included). The signal is not an
artifact of tie noise — if anything it's slightly *sharper* once ties are
excluded. Conclusion: the weak-but-nonzero community-vs-emotion result in
Section 2 is robust to this issue, not an artifact of it.
