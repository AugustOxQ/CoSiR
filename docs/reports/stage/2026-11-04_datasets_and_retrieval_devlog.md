<!-- Dev-log post for update #6, in the short three-line form the user asked for (2026-10-19 19:45). Links are
placeholders: this repo is public; the real link goes in /project/cosir-devlog/deploys.log after the deploy. Attach the
structure figure: docs/reports/assets/2026-11-04_datasets_and_retrieval/photo_model_structure.png -->

*CoSiR update #6 (29 October to 3 November):* our painting model now also retrieves by aspect. On 12,281 held paintings, with a comparison method picked on validation, it is level with the instructed Qwen image on emotion and style. On emotion, mAP is 0.2691 vs 0.2684. On genre it wins, 0.6151 vs 0.6124. A separate photo model, fine-tuned on unlabelled photos, beats the instructed image on EmoSet, a test set that shaped no earlier choice. Its AMI is 0.4571 vs 0.4390. The main comparisons also hold on three more encoders and on MMArt paintings.
Emotion retrieval is the main limit: it fell below the instructed image with plain cosine, on CLIP L/14 and on two MMArt encoders. The photo training came after a failed test on the same Affection photos. Three rounds are running to fill the gaps in outside comparisons.
Full update (same PIN as before): <link to the page on the dev-log site>. Feedback welcome: in the coverage table, which empty cell would a reviewer ask about first?
