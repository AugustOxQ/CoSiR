*CoSiR: primer for the update posts*
_Created 12 October 2026 · last updated 13 October 2026_

*The problem.* Image and text models such as CLIP give one fixed similarity between an image and a caption. CoSiR asks for a similarity that holds in one respect only (an aspect: emotion, style or genre), where the user shows the aspect by examples instead of naming it.

*The setting.* An episode has a query (a painting or a caption), 4 example pairs that share one aspect, 4 contrast pairs that share another, and 13 candidates in the other modality. One candidate shares the query's value of the shown aspect (the target), one shares its value of the other aspect. The examples never show the query's own value. Each episode is scored twice: as built, and with examples and contrasts swapped. The swap makes the other aspect's candidate the target.

Figure: `docs/reports/assets/2026-10-07_weekly/setup_overview.png` (the task at a glance; here the example pairs share a feeling and the contrast pairs an art style).

The data is ArtELingo: WikiArt paintings with captions written by viewers. We use its English part, 308,723 image and caption pairs over 61,402 paintings. Each painting has an art style and, for most paintings, a genre. Each caption carries the emotion its writer felt, so two captions of one painting can carry different emotions.
• Emotion: 8 values, one per caption: amusement, awe, contentment, excitement, anger, disgust, fear and sadness. A ninth label, "something else", is not used as a value.
• Art style: 27 values, one per painting, such as Impressionism, Baroque and Ukiyo-e. Episodes use the 23 styles that have at least 30 development paintings.
• Genre: 10 values, one per painting, such as portrait, landscape and still life. About 81% of the paintings have a genre.

Figure: `docs/reports/assets/2026-10-12_devlog_primer/artelingo_example.png` (one ArtELingo painting with its viewers' captions).

The paintings are split once, and all captions of a painting stay in its split.
• Training: 183,694 pairs over 36,518 paintings. Our methods learn their parts here, without the emotion, style or genre labels. CLIP itself stays frozen.
• Development: 32,413 pairs over 6,451 paintings. We use them to choose methods, and to test a chosen method once on new episodes (fresh episodes).
• Held: 61,744 pairs over 12,281 paintings, kept back for the final test and read once.
A few blend weights of each method are set on development episodes. A validation split of 30,872 pairs is not used in this line of work.

*How we measure.* Each episode is scored in both conditions: as built and swapped. Each condition is scored in both directions: an image query ranks captions, and a caption query ranks images.
• R@1: how often the target ranks first, in percent. Chance is 7.7% (1 in 13).
• Other-aspect rate: how often the other aspect's candidate ranks first.
• Condition gain: R@1 minus the other-aspect rate, in points. A score that ignores which aspect the examples show ranks the same way in both conditions. So its condition gain is exactly 0. A gain above 0 shows that the first place follows the examples.
• Margin: the difference in R@1 points between two methods on the same episodes, with a 95% interval. A margin is established when the whole interval is above 0.

Figure: `docs/reports/assets/2026-10-12_devlog_primer/condition_gain.png` (condition gain; left, why a score that ignores the aspect has a gain of 0; right, two scores on the held paintings).

*What we compare against.* Before every test we write down the rule that decides it: the comparators and the margin that counts as a pass. The rule is fixed before the test episodes are built, so we call it pre-registered. A new method must beat three comparators in it. Each one tests a different source of a gain.
• Cosine on frozen CLIP ViT-B/32: the backbone alone. Beating it shows that the method adds something to the backbone.
• B′(A0): the strongest score in the rule that ignores which aspect the examples show. It adds two terms to cosine, both learned without labels. One compares the query and a candidate relative to the episode's 8 example items. The other rewards a candidate that falls in the same clusters as the query. The clusters group the training captions and images three ways: by mood, by image features and by caption features.
• Why B′(A0) is strong: it cannot choose between the two aspects, but it often finds a candidate that shares some aspect with the query. On the held paintings it reaches 18.84 R@1, against 13.25 for cosine. Beating it shows that the gain comes from reading the aspect, not from the examples and the clusters alone.
• The matched control: our method with one change, so that it cannot tell the example pairs from the contrast pairs. Our method adds a term that depends on which pairs are the examples. The control computes that term both ways, as built and swapped, and adds their average. It keeps every part of the method, but it ranks the same way in both conditions. Beating it shows that the gain comes from following the examples, not from the extra term itself.

*Where we stand.* Our current method (AFF) starts from an aspect-ignoring score of the same kind as B′(A0). When it is confident that the example pairs share a mood, it adds a term that ranks the candidates mostly by mood.

On the held paintings, AFF beats B′(A0) by +0.60 [+0.46, +0.73] R@1 points (19.44 against 18.84). Its condition gain is 3.06 points. It passed every check of its pre-registered rule, including those against cosine and its matched control. A stronger aspect-ignoring score, B′(A1), adds a pretrained style encoder to B′(A0). It was only a secondary check, and AFF's margin over it, +0.14 [−0.03, +0.30], is not established.
