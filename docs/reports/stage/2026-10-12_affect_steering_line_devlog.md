*CoSiR update #1 · 2 to 12 October 2026*
*One-sided affect steering passed its held test on ArtELingo; the style × genre loss remains open.*

Recap: CoSiR tests whether a model can rank images and captions by one aspect shown only through example pairs. Current best: one-sided affect steering (AFF), +0.60 [+0.46, +0.73] R@1 points over B′(A0), the strongest pre-registered condition-free score (held ArtELingo paintings, 36,864 episodes). Project primer: see the pinned message.

*Since the start of the aspect task:*
1. Aspect episodes, condition gain and matched controls defined as the task and protocol. (worked)
2. Metrics from the example pairs, trained shared factors and in-context Qwen3-VL (2B, 8B): none chose the shown aspect. (did not work)
3. Label-free groupings, a learned reader and a confidence gate: +0.44 on development episodes, below the +0.5 bar. (did not work)
4. AFF, fresh test: +0.59 [+0.46, +0.73] over B′(A0), seven of seven checks. (worked)
5. Vetoes on AFF's gate, and GoEmotions placement of captions: neither beat AFF. (did not work)
6. Fine-tuned CLIP as a comparator: 15.32 to 15.80 R@1 on held paintings, against 19.44 for AFF. (comparator)
7. AFF, held test: +0.60 over B′(A0), seven of seven checks with Holm; not established over B′(A1) (+0.14) or over its parent without the affect restriction (+0.02). (worked)
8. Label-free style groupings for style × genre: none passed the development gate. (did not work)

Figure 1: `docs/reports/assets/2026-10-12_affect_steering_line/held_comparison.png`
Figure 2: `docs/reports/assets/2026-10-12_affect_steering_line/per_pair_fresh_held.png`

*Implications:* the ArtELingo claim holds on held paintings, but the gain sits on the two emotion pairs, and AFF is below B′(A0) on style × genre (−0.43).
*Next:* a learned style grouping on frozen CLIP (round 8), with a held-read verdict by Friday 23 October.
*Feedback requested:* should the paper lead with the task and protocol, with AFF as a modest method, or with AFF?
