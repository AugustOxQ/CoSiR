# CoSiR glossary

The project's terms, written by the grill (research loop step 5) and used by briefs and reports. Created 2026-10-09 for round 6.

| Term | Meaning |
|---|---|
| PercepT, core reimplementation (code tag `percept_core`) | our PercepT code with the four result-changing fixes of the [code audit](docs/reports/auto/percept/2026-12-04_percept_code_audit.md) (image-caption pairs, the paper's Eq. 2 fusion, its topic-count algorithm, multi-hot Stage 2 targets), on our own features; the PercepT row of the main tables on our protocol (user, 2026-10-17) |
| PercepT, faithful reimplementation (code tag `percept_faithful`) | PercepT with all eight audit changes and its own encoders (CLIP L/14, ModernBERT tuned on GoEmotions); needed before submission, run on our protocol and in the paper's own setting to show it reproduces the paper's numbers (user, 2026-10-17) |
| dollar weight | token use priced at API rates (cache reads, cache writes and output weighted by their price) to compare runs of agent work; not what the Max plan charges ([engine switch check](docs/reports/auto/process/2026-11-26_engine_switch_check.md)) |

The conditioned line's terms (aspect, episode, reader, AFF, B′ scorers, held read and the rest) are archived: [glossary of the conditioned line](archive/v2_conditioned/GLOSSARY.md).
