## Contract Paraphrase

D1 (methodology_rigor, mandatory, my owned dimension). I read this as the question of whether the study design, the handling of data splits, the statistical reporting and the reproducibility affordances would satisfy a careful peer reviewer in this field. Because the contract notes that the submission is a pre-results plan with diagnostic evidence attached, I will judge the protocol as designed rather than the absence of final numbers: each claim needs a named experiment, the strongest real baseline run under matched conditions, an evaluation split that is touched only after every design choice is frozen, a declared primary metric, an uncertainty procedure, and enough specification that another group could rerun it.

D2 (domain_accuracy, mandatory, owned by the domain seat). This dimension asks whether claims agree with current evidence in the field, whether prior methods are described faithfully, and whether domain terms and reported results are correct. I do not score it. From a methodology standpoint it matters to me only where a misdescribed prior method would change which baseline is the right comparison.

D3 (argumentative_coherence, mandatory, owned by the devil's advocate seat and also eligible for me). I read this as whether the central thesis hangs together, whether the evidence offered actually licenses each claim, and whether any reasoning error undermines the main argument. My angle is the inferential chain from experiment to conclusion: whether the planned and appended evidence, including results that came out negative or null, is consistent with the claims as worded, and whether each claim is scoped to what its test can show.

D4 (cross_disciplinary_relevance, high priority, owned by the perspective seat). This asks whether framing, definitions and implications can be followed by readers from neighbouring fields, and whether any claim that crosses fields is backed up. I do not score it; I note only that undefined evaluation terms would also hurt reproducibility.

D5 (writing_and_structure, normal priority, owned by the editor seat). This covers how the manuscript is organised, how clearly it explains itself, the quality of its figures and tables, and whether it follows the target venue's conventions. I do not score it.

D6 (venue_fit_and_contribution, mandatory, owned by the editor seat). This asks whether the work suits the configured venue and offers an original and significant contribution for its audience. I do not score it, although a contribution that cannot be tested under the planned protocol would surface under my D1 and D3 commitments instead.

## Scoring Plan

### D1: methodology_rigor
dimension_id: D1
what_to_look_for: For every planned claim, a pre-specified experiment with one primary metric and a decision rule, the strongest real baseline evaluated on the identical split, features and tuning budget, a held-out test split with an explicit access budget that is consumed only after design freeze and is disjoint from any data used for the appended diagnostics, a fixed number of seeds with paired tests or bootstrap intervals and a stated correction across the multiple aspects or conditions compared, and reproducibility details covering split construction, sample identifiers, label provenance, encoder versions and released code.
what_triggers_block: Any claim whose planned test cannot separate success from failure but could still be repaired by redesigning the protocol before results are collected, for example a held-out split already used for model or hyperparameter selection, baselines given a smaller tuning budget or a different split, no declared primary metric or decision threshold, or effect estimates planned without seed variance or a paired significance procedure.
what_triggers_warn: The design is sound in outline yet under-specified in places, such as an unfixed number of seeds or bootstrap resamples, no stated multiplicity correction across conditions, informal accounting of how often the held-out data will be touched, a weaker secondary baseline missing, appended diagnostic numbers given without uncertainty, or reproducibility items such as code release and split files left undecided.
what_triggers_fatal: The headline claim is untestable by construction and no repair fits the timeline, because the only available evaluation labels are circular with the training signal or the sole test set has been irrecoverably contaminated by repeated adaptive reuse with no fresh held-out data obtainable, so that no planned run could refute the claim.

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: Whether each stated contribution follows from the evidence planned or appended for it, whether negative and null diagnostic results are acknowledged and reconciled with the claims rather than omitted, whether gains credited to a component are backed by an ablation that isolates that component, whether null results are interpreted with an equivalence margin or power argument instead of as proof of no effect, and whether claim wording stays consistent across sections.
what_triggers_block: A headline claim is contradicted or left without support by the appended evidence, such as an unacknowledged negative result that bears directly on it, or a gain attributed to a component with no planned ablation that isolates it, while a narrower reframing or an added control could still restore consistency.
what_triggers_warn: Claims mostly follow from the evidence but some are stated more strongly than their tests allow, such as a null result read as equivalence, a gain seen on a favourable subset generalised to the whole benchmark, selective reporting among conditions, or claim wording that drifts between sections.
what_triggers_fatal: The core thesis is internally contradictory or is refuted by the plan's own appended evidence with no reformulation the planned experiments could still support, so the central contribution would remain false or circular even if every planned run succeeded.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
