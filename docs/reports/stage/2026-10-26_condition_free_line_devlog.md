<!-- Dev-log post for update #4, in the short three-line form the user asked for (2026-10-19 19:45). The longer
template draft stays in .scratch/stage_pages/2026-10-26_condition_free_line_devlog.draft.md. Links are placeholders:
this repo is public; the real link is in /project/cosir-devlog/deploys.log. Attach the structure figure:
docs/reports/assets/2026-10-26_condition_free_line/model_structure.png -->

*CoSiR update #4 (19 to 26 October):* one shared network, trained on graphs built from the Qwen image with an aspect instruction and read once on the plain image, gives emotion, style and genre spaces with no instruction at test time.
On 12,281 held-out paintings it leads our re-implementation of PercepT on emotion (AMI 0.1029 vs 0.0874 for the strongest PercepT entry), and the lead holds on four of seven image encoders; every student still needs Qwen at training, and the lead is k-means only. A new genre term reaches the instructed Qwen genre image on development data (0.5022 vs 0.4989), not yet on held-out data.
Full update (same PIN as before): <link to the page on the dev-log site>. Feedback welcome: is a held-out lead on four of seven encoders, with Qwen needed at training, enough to present the recipe as encoder-general?
