## Contract Paraphrase

D1 (methodology_rigor). Read adversarially, this dimension asks whether the planned design, data handling, held-out protocol and statistics would survive a reviewer who wants to explain away a positive result. The attacks I would bring are leakage between the data used for tuning and the data used for the final claim, tests too weak to detect the effect a claim needs, variance across seeds or splits that goes unreported, and gaps that would stop a third party from re-running the protocol. Because the contract frames the manuscript as a plan written before its main results, a missing result is not a defect here; a protocol that cannot tell success from failure is. The methodology seat scores this dimension, not me, but any such weakness I find still goes into my findings.

D2 (domain_accuracy). This dimension asks whether the claims match what the field currently knows, whether prior methods and baselines are shown at their strongest rather than as straw men, and whether domain terms and quoted prior numbers are right. As devil's advocate I would test whether a stronger or more recent competitor was left out, and whether a prior result is described in a way its own authors would not accept. The domain seat owns the score.

D3 (argumentative_coherence). This is the dimension I own and score. It asks whether the core thesis holds together, whether the evidence offered actually supports each claim, and whether any fallacy undermines the central argument. For a pre-results plan I read coherence as three tests. First, each headline claim must come with a pre-specified comparison, baseline and decision rule whose outcome could go against it. Second, the claims must agree with the plan's own appended diagnostic evidence, negative results included, instead of quietly reframing what did not work. Third, the chain from evidence to claim must not be circular, must not generalise past what was tested, and must not rest on an unstated premise. The contract tells me a missing result is not a block by itself, while a design that leaves a claim untestable is; I also weigh how likely each claim is to hold given the appended evidence.

D4 (cross_disciplinary_relevance). This dimension asks whether framing, definitions and implications reach readers in adjacent fields, and whether claims that cross into another discipline are backed rather than asserted. My challenge would be whether a construct borrowed from a neighbouring field is used as if it were settled when it is contested there, and whether the stated implications for other communities follow from anything tested. The perspective seat scores it.

D5 (writing_and_structure). This dimension covers organisation, clarity, figure and table quality, and adherence to venue conventions. Adversarially, I would ask whether a reader can find each claim, its planned test and its supporting evidence without rebuilding the argument themselves, and whether a figure or table quietly shows something weaker than the text around it says. The editor seat scores it.

D6 (venue_fit_and_contribution). This dimension asks whether the work fits the configured venue and makes an original, significant contribution for that readership. My stress test would be the "so what" question: does the contribution still stand once the closest prior work is placed beside it, or is the novelty mostly a new name for a known setup? The editor seat owns this mandatory score.

## Scoring Plan

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: For each headline claim, a pre-specified test, baseline and decision rule whose outcome could refute it; definitions and the problem statement that stay consistent across sections; appended evidence, negative results included, weighed honestly rather than reframed; and inferences free of circularity, overgeneralisation or an unexamined premise.
what_triggers_block: A headline claim is left untestable by the planned design because no pre-specified comparison, baseline or decision rule could refute it, or a headline claim is contradicted by the plan's own appended evidence without being revised or scoped down.
what_triggers_warn: A secondary claim lacks a decisive test, an appended negative result is acknowledged but its consequence for a claim is understated, or the wording of a claim reaches beyond what its planned test can establish.
what_triggers_fatal: The central thesis is internally contradictory or circular, with the confirming evidence built from the very assumption it asserts, so no outcome of the planned experiments could support or refute it and only a new thesis would repair it.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
