## Contract Paraphrase

D1 (methodology_rigor): This dimension belongs to the methodology seat. I read it as asking whether the planned experiments, data splits, statistics and released artifacts would let a careful vision reviewer reproduce the numbers and trust them. From the domain side it matters to me only where a planned protocol departs from how the field's established conditional-similarity and composed-retrieval benchmarks are normally run, because a nonstandard protocol can make otherwise correct domain claims incomparable with prior work.

D2 (domain_accuracy): This is my dimension. It asks whether what the manuscript says about the field is true: prior methods described as they actually work and as they actually perform, terms such as condition, aspect, notion of similarity, composed query and instruction used in their accepted sense, and the claims the plan sets out to test consistent with what current evidence makes plausible, including the appended diagnostics and their negative results. Because the contract describes a pre-results plan, I judge whether the planned baselines and benchmarks are the ones the field would require to settle each claim, and I do not penalise results that do not exist yet.

D3 (argumentative_coherence): This dimension asks whether the central thesis holds together and whether each claim follows from the evidence offered, without circular or overreaching inference. An argument can be internally consistent and still rest on a mistaken premise about prior work; I will raise such premises under D2 and leave the logical structure to the seats that own D3.

D4 (cross_disciplinary_relevance): This dimension asks whether readers from neighbouring areas, such as information retrieval, the cognitive science of similarity, or interpretability research, can follow the framing, and whether claims that reach into those areas are backed. I will only note places where a term borrowed from such an area is used inaccurately.

D5 (writing_and_structure): This dimension asks whether the document is well organised and clear, whether figures and tables inform, and whether the conventions of a vision venue are followed. It is outside my scoring remit; I mention it only where unclear exposition hides what is being compared with what.

D6 (venue_fit_and_contribution): This dimension asks whether the work fits its target venue and offers something original and significant to that readership. Originality overlaps with my literature check: if a recent conditional embedder or sparse concept method already does what is proposed, that bears on both D2 and the contribution judgement, but the fit and significance verdict belongs to the editor seat.

## Scoring Plan

### D2: domain_accuracy

dimension_id: D2
what_to_look_for: Whether prior work is represented accurately in method and in reported numbers across conditional similarity (Conditional Similarity Networks, attribute-specific embeddings, GeneCIS-style conditional benchmarks), composed image retrieval (FashionIQ, CIRR, CIRCO, and zero-shot methods such as Pic2Word, SEARLE and LinCIR), 2025 to 2026 instruction-following multimodal embedders (E5-V, VLM2Vec, GME, MM-Embed and similar VLM-based embedders prompted with the condition as text) and sparse or concept-level CLIP codes (SpLiCE, sparse autoencoders on CLIP features); whether field terms keep their accepted meaning; whether every claim is paired with the strongest current domain alternative as a planned baseline and every cited benchmark is planned under its published protocol; and whether each claim stays plausible given the appended diagnostic evidence, negative results included.
what_triggers_block: A central novelty or superiority claim omits or misdescribes a directly overlapping prior method, so the claimed gap may not exist; or the planned comparison leaves out the strongest current domain alternative, such as an instruction-following multimodal embedder given the condition as a text instruction, so the claim cannot be decided; or a cited benchmark is planned under a split, protocol or metric that differs from its published one without saying so; or the appended evidence, negative results included, contradicts a headline claim that the plan still asserts with no revision path.
what_triggers_warn: Coverage gaps or inaccuracies that leave every central conclusion intact: secondary related work missing, a prior method's numbers or setting misstated in a way that does not change the comparison, conditional similarity and composed retrieval terminology conflated, recent embedder capabilities characterised from outdated sources, or a claim worded more strongly than the appended evidence supports while remaining testable as planned.
what_triggers_fatal: The core contribution is already published in substance by unacknowledged prior work that solves the same problem under comparable conditions, leaving no defensible novelty; or the central argument rests on a domain factual error that revision cannot repair, such as a nonexistent or misattributed citation carrying the main claim; or the appended evidence directly refutes the central claim and the plan offers no reformulated claim that survives it.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
