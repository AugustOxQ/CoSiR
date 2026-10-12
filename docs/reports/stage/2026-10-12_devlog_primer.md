*CoSiR: primer for the update posts*

*The problem.* Image and text models such as CLIP give one fixed similarity between an image and a caption. CoSiR asks for a similarity that holds in one respect only (an aspect: emotion, style or genre), where the user shows the aspect by examples instead of naming it.

*The setting.* An episode has a query (a painting or a caption), 4 example pairs that share one aspect, 4 contrast pairs that share another, and 13 candidates in the other modality. One candidate shares the query's value of the shown aspect (the target), one shares its value of the other aspect. The examples never show the query's own value, and swapping examples and contrasts swaps the target. Data: ArtELingo, WikiArt paintings with viewer captions; 12,281 paintings are held back for the final test.

*How we measure.* R@1: how often the target ranks first, in percent. Condition gain: R@1 minus how often the other aspect's candidate ranks first; it is exactly 0 for any score that ignores the examples' aspect. Margins are in R@1 points with a 95% interval.

*What we compare against.*
• Cosine on frozen CLIP ViT-B/32, the backbone alone.
• B′(A0): our strongest pre-registered score that uses the examples but ignores which aspect they show.
• The matched control: the method itself with only the condition removed.

*Where we stand.* One-sided affect steering (AFF), +0.60 [+0.46, +0.73] R@1 over B′(A0) on held ArtELingo paintings (19.44 against 18.84), with a condition gain of 3.06. Updated 12 October 2026.

*Terms used in the posts.*
• Aspect, value: a respect (emotion, style, genre) and one setting of it (sad, Baroque, landscape).
• Fresh episodes: new episode draws used for one test. Held paintings: read once, for the paper.
• AFF: adds a label-free affect term to the condition-free score, only when a learned reader is confident the examples share the affect grouping.

Figure: `docs/reports/assets/2026-10-07_weekly/setup_overview.png` (the task at a glance).
