# CoSiR glossary

The project's terms, written by the grill (research loop step 5) and used by briefs and reports. Created 2026-10-09 for round 6.

| Term | Meaning |
|---|---|
| PercepT, core reimplementation (code tag `percept_core`) | our PercepT code with the four result-changing fixes of the [code audit](docs/reports/auto/percept/2026-12-04_percept_code_audit.md) (image-caption pairs, the paper's Eq. 2 fusion, its topic-count algorithm, multi-hot Stage 2 targets), on our own features; the PercepT row of the main tables on our protocol (user, 2026-10-17) |
| PercepT, faithful reimplementation (code tag `percept_faithful`) | PercepT with all eight audit changes and its own encoders (CLIP L/14, ModernBERT tuned on GoEmotions); needed before submission, run on our protocol and in the paper's own setting to show it reproduces the paper's numbers (user, 2026-10-17) |
| PercepT, paper settings, strict (code tag `percept_paper_strict`) | PercepT core's arms P1 (CLIP + RoBERTa) and P4 (Qwen Stage 1, CLIP patches) with the paper's training settings (the third 500-wide autoencoder layer, latent noise 0.1, batch 512, cosine learning rate to 1e-5, no balance term, at most 200 joint epochs; Stage 2 batch 64, lr 1e-3, cosine) and the paper's Algorithm 1 topic cut (γ = 2); only the backbone differs from the paper. "Collapsed" when it keeps under 10 or over 90 topics, or one topic holds over half the fit pairs; a collapsed run leaves the bar (user, 2026-10-20) |
| PercepT, paper settings, K = 40 (code tag `percept_paper_k40`) | as the strict version, but keeps the 40 of 100 topics with the most fit pairs (PercepT core's A3) instead of Algorithm 1; "collapsed" only by the one-topic rule (user, 2026-10-20) |
| dollar weight | token use priced at API rates (cache reads, cache writes and output weighted by their price) to compare runs of agent work; not what the Max plan charges ([engine switch check](docs/reports/auto/process/2026-11-26_engine_switch_check.md)) |

The conditioned line's terms (aspect, episode, reader, AFF, B′ scorers, held read and the rest) are archived: [glossary of the conditioned line](archive/v2_conditioned/GLOSSARY.md).
