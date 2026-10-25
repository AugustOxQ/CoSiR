# CoSiR glossary

The project's terms, written by the grill (research loop step 5) and used by briefs and reports. Created 2026-10-09 for round 6.

| Term | Meaning |
|---|---|
| PercepT, core reimplementation (code tag `percept_core`) | our PercepT code with the four result-changing fixes of the [code audit](docs/reports/auto/percept/2026-12-04_percept_code_audit.md) (image-caption pairs, the paper's Eq. 2 fusion, its topic-count algorithm, multi-hot Stage 2 targets), on our own features; the PercepT row of the main tables on our protocol (user, 2026-10-17) |
| PercepT, faithful reimplementation (code tag `percept_faithful`) | PercepT with all eight audit changes and its own encoders (CLIP L/14, ModernBERT tuned on GoEmotions); needed before submission, run on our protocol and in the paper's own setting to show it reproduces the paper's numbers (user, 2026-10-17) |
| PercepT, paper settings, strict (code tag `percept_paper_strict`) | PercepT core's arms P1 (CLIP + RoBERTa) and P4 (Qwen Stage 1, CLIP patches) with the paper's training settings (the third 500-wide autoencoder layer, latent noise 0.1, batch 512, cosine learning rate to 1e-5, no balance term, at most 200 joint epochs; Stage 2 batch 64, lr 1e-3, cosine) and the paper's Algorithm 1 topic cut (γ = 2); only the backbone differs from the paper. "Collapsed" when it keeps under 10 or over 90 topics, or one topic holds over half the fit pairs; a collapsed run leaves the bar (user, 2026-10-20) |
| PercepT, paper settings, K = 40 (code tag `percept_paper_k40`) | as the strict version, but keeps the 40 of 100 topics with the most fit pairs (PercepT core's A3) instead of Algorithm 1; "collapsed" only by the one-topic rule (user, 2026-10-20) |
| recipe | the student's training loss: which terms it keeps (buddy terms on the instructed graphs, own-caption terms, the teacher term toward the instructed Qwen image) and their weights; since report 12-24 the recipe is the one without own-caption terms at teacher share 1/5 (code name R6-5) (user, 2026-10-25) |
| teacher share | the teacher term's weight divided by the sum of all term weights, where the buddy terms count 1 and the own-caption terms 1; a nominal share, not the measured share of the loss value; 1/5 for D2 (0.5 of 2.5) and for the current recipe (0.25 of 1.25) (user, 2026-10-25) |
| matched arm | an encoder's own PercepT arm: PercepT core (P3, 40 topics) with the claim test's Qwen Stage 1 topics and its Stage 2 trained on that encoder's default image; beside the student on the same encoder it drops only the student (user, 2026-10-25) |
| frozen input | an encoder's default image embedding clustered directly, with no training (in tables: the frozen image, F with the encoder's number); the starting level of that encoder (user, 2026-10-25) |
| seed range | the spread (maximum minus minimum) of D2's AMI over model seeds 0, 1 and 2 on the 5,000 old development paintings (report 12-22): emotion 0.0046, style 0.0082, genre 0.0128; the unit for a "small" effect and for the guards (user, 2026-10-25) |
| dollar weight | token use priced at API rates (cache reads, cache writes and output weighted by their price) to compare runs of agent work; not what the Max plan charges ([engine switch check](docs/reports/auto/process/2026-11-26_engine_switch_check.md)) |

The conditioned line's terms (aspect, episode, reader, AFF, B′ scorers, held read and the rest) are archived: [glossary of the conditioned line](archive/v2_conditioned/GLOSSARY.md).
