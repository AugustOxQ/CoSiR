<!-- Dev-log post for update #5, in the short three-line form the user asked for (2026-10-19 19:45). Links are
placeholders: this repo is public; the real link goes in /project/cosir-devlog/deploys.log after the deploy. Attach the
structure figure: docs/reports/assets/2026-10-28_recipe_sweep/model_structure.png -->

*CoSiR update #5 (27 to 28 October):* a search of our recipe's settings picked 40 graph neighbours instead of 20. On 12,281 held-out paintings, the picked network (one read of the plain Qwen image, no instruction) beats the instructed Qwen image on emotion, AMI 0.1066 vs 0.0921. It beats it narrowly on style, 0.4066 vs 0.4026. On genre it is level under our margin rule, not a win, 0.5001 vs 0.5062, and the whole difference interval lies below zero. On emotion it also stays above our re-implementation of PercepT, 0.1066 vs 0.0874.
The search compared about 60 configurations on one seed, the starting recipe was chosen after its held genre result, and all results are k-means on one image encoder and one dataset.
Full update (same PIN as before): <link to the page on the dev-log site>. Feedback welcome: with the style win this narrow and genre only level, is the comparison with the instructed Qwen image strong enough to lead the paper, or should the main claim stay on emotion?
