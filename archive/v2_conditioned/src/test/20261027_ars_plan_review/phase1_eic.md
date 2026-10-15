## Contract Paraphrase

D1 (methodology_rigor, mandatory, scored by the methodology seat): the panel needs one seat to judge whether the study design, data handling, statistical reporting and reproducibility provisions would satisfy a careful computer-vision referee. Because the stage note says the manuscript is a plan written before its main results exist, the question here is whether the designed experiments, held-out protocol and statistics can settle each claim one way or the other. I do not score this dimension, but my contribution judgement under D6 depends on its outcome.

D2 (domain_accuracy, mandatory, scored by the domain seat): statements about the field must match current evidence, with prior methods and benchmarks described as their authors and later replications report them, and with domain terminology and quoted numbers used correctly. A misdescribed nearest neighbour in the literature would also distort the novelty case I weigh under D6, so I will read the domain seat's findings next to my own.

D3 (argumentative_coherence, mandatory, scored by the devil's advocate and methodology seats): the central thesis has to hold together, with each claim following from the evidence offered for it and no contradiction or fallacy that would bring down the main argument. For a plan carrying evidence appendices, this includes whether the stated claims stay consistent with the appended diagnostics, negative results among them.

D4 (cross_disciplinary_relevance, high priority, scored by the perspective seat): readers from neighbouring areas such as information retrieval, metric learning or multimodal language modelling should be able to follow the framing and definitions, and any claim reaching beyond vision should be substantiated. A block on this high-priority dimension would by itself push the panel outcome toward major revision.

D5 (writing_and_structure, normal priority, my seat): I judge how the document and the paper it plans are organised, how clearly they are written, whether planned figures and tables are legible and purposeful, and whether the planned paper respects CVPR conventions such as the page limit, anonymity and the split between main paper and supplement. For a plan, good structure means a reader can trace every claim to the experiment, baseline and decision rule meant to test it.

D6 (venue_fit_and_contribution, mandatory, my seat): I judge whether the planned paper belongs at CVPR and whether its contribution would be original and significant for that audience. For a paper that proposes a new task with its own benchmark, that means a testable difference from the nearest existing tasks, a benchmark whose construction and held-out protocol can bear a headline claim, and a comparison against the strongest current alternatives. Following the stage note, absent results are not grounds for a block in themselves, while a design that leaves the contribution claim untestable is; I also weigh how likely each contribution claim is given the appended evidence, negative results included. A fatal block here sends the synthesizer toward rejection, and a repairable one toward major revision.

## Scoring Plan

### D5: writing_and_structure

dimension_id: D5
what_to_look_for: A claims-to-evidence map in which each planned claim names its experiment, baseline, held-out split and decision rule; task, condition and metric terms defined once and used consistently; planned figures and tables that each make one point with the baseline shown beside it; a main-paper outline that fits the CVPR page limit with an explicit main-versus-supplement split; and appendices that are cross-referenced from the claims they support and marked as diagnostic evidence rather than final results.
what_triggers_block: The organisation makes it impossible to tell which planned experiment tests which claim, or core terms are undefined or used so inconsistently that a claim admits two readings, or the planned main paper cannot fit the CVPR page limit and no plan says what moves to the supplement.
what_triggers_warn: Localised problems that still leave every claim traceable, such as terminology drifting between sections, appendix evidence not referenced from the claim it supports, a planned figure or table missing its baseline or legend, redundant passages, or a page budget that is tight but never stated.

### D6: venue_fit_and_contribution

dimension_id: D6
what_to_look_for: A task definition separated from the nearest existing formulations (conditional similarity learning, composed image retrieval, instruction-conditioned or universal multimodal embedding retrieval, attribute-specific retrieval) by a difference an experiment can show; a stated reason existing benchmarks cannot measure it, and a planned benchmark whose labels, size and held-out protocol can bear the headline claim; planned baselines that include the strongest current alternatives on the same splits; a significance argument tied to what the CVPR audience would build on; and an honest reading of the appended evidence, negative results included, against each contribution claim.
what_triggers_block: The design leaves the central contribution claim untestable, because no planned experiment could show the task's distinction from existing tasks, or the benchmark or held-out protocol allows leakage or tuning on test data, or the strongest current alternatives are absent so a positive result would not establish the contribution; missing results alone do not trigger this.
what_triggers_warn: The contribution is testable but under-argued, for example the delta to the closest prior task or benchmark is asserted rather than shown, a relevant but not strongest baseline family is missing, the benchmark rests on a narrow label source or few conditions, significance is planned on a single dataset or aspect, or the appended evidence makes a secondary claim unlikely and the plan does not revise it.
what_triggers_fatal: The central task and benchmark substantively duplicate existing published work so no original contribution remains, or the appended evidence already refutes the central claim and the plan keeps no other testable contribution, or the work lies outside CVPR scope so that no revision could make it fit.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
