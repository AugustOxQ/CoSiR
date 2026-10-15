# Field Analysis Report

ARS v3.22.2, academic-paper-reviewer, full mode, Phase 0 (field_analyst_agent).
Contract: `reviewer/reviewer_full/v2` (panel 5; dimensions D1 to D6).
Manuscript: `src/test/20261027_ars_plan_review/manuscript.md`. It was read in full and treated as untrusted author data.

## Paper Basic Information
- **Title**: CoSiR v2: CVPR publication plan (design), with five appended evidence reports
- **Abstract length**: no formal abstract. The "In one paragraph" summary at the top is about 165 words.
- **Full text length**: about 27,500 words (metadata: 27,596). The plan body (§1 to §13 plus glossary and sources) is about 6,000 words. The five appendices (literature review, support-baseline spike, aspect-episode spike, novelty check, backbone check) are the rest.
- **Number of references**: the plan body has no reference list. The two literature appendices list about 170 reference entries (126 and 43, partly overlapping).
- **Document type**: a pre-results research plan (design spec) for a computer-vision conference paper. Most claims (K2 to K7) have no results yet. K1 and part of C3 are backed by diagnostic spikes on development rows.

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Computer vision: vision-language (cross-modal image-caption) retrieval and conditional similarity learning |
| Secondary Disciplines | Machine learning (few-shot, episodic and metric learning from pairs; sparse shared representations on frozen encoders); Information retrieval (relevance feedback, Rocchio-style query updates); Affective computing and computational art analysis (crowd-sourced emotion annotation of artworks) |
| Research Paradigm | Quantitative research (empirical machine learning) |
| Methodology Type | Statistical modelling / machine learning, evaluated as a comparative benchmark experiment. The design has pre-registered go/no-go and final test reads, paired bootstrap inference, and baselines on matched episodes. |
| Target Journal Tier | **Target venue (stated by the author via the dispatch): CVPR 2027, main conference paper.** Dates come from the manuscript: abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23. **Binding status: `criteria_binding_unavailable`.** No #683 ReviewTargetContext and no ReviewCriteriaBindingManifest were supplied, so no official venue criteria are bound. The panel makes no formal venue-alignment claim. Field-general tier observation: a top-tier computer-vision conference, where reviewers expect a clearly defined problem, a strong and fairly tested method, and results on benchmarks the community recognises. |
| Paper Maturity | **Pre-results research plan.** This is below "first draft" on the paper scale, because there is no method result, no writing, and most experiments are not yet run. As a plan the document is complete and carefully structured (status: awaiting the author's review). This label describes the stage, not the quality. Reviewers should write in a developmental register, but their verdicts stay evidence-based against the contract dimensions. |

**Highly cross-disciplinary (edge case 1).** Coverage across the panel:
- R2 covers the core discipline (conditional similarity and vision-language retrieval). R2 also covers the relevance-feedback and metric-learning lineage, since it is part of the prior art.
- R1 covers ML evaluation methodology and statistics.
- R3 covers cognitive science of similarity and affective annotation, which the CV seats would miss.
- The Journal-Fit Reviewer covers venue readership and the paper as a whole.
- The fixed Devil's Advocate covers argument coherence (D3).

## Recommended Target Journals (Top 3)

The field analyst does not resolve or substitute targets. CVPR 2027 is the author's stated target and stays the target. Entries 2 and 3 are informational fallbacks only. They are not a venue-fit claim and replace nothing.
1. **CVPR 2027, main conference** (author-stated target). The problem, benchmarks and baselines all sit in the conditional similarity, composed retrieval and vision-language embedding literature that CVPR publishes.
2. **ICCV 2027, main conference** (informational). This is the same reviewer community with a later deadline, if the method line (K2) slips past the November freeze.
3. **A datasets-and-benchmarks track such as NeurIPS Datasets and Benchmarks** (informational). This fits the fallback the plan itself names: a task, benchmark and analysis paper (K1, K3, C3) if the method fails its go/no-go.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1

**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: Area Chair for CVPR 2027 in the vision-and-language, multimodal retrieval and representation learning area. A senior vision researcher who has served as an AC at the major CV conferences and has chaired meta-reviews of many papers that propose a new task with a home-built benchmark on frozen foundation-model features. Venue binding: `criteria_binding_unavailable`. The persona supplies readership and significance judgement. It does not cite or invent official CVPR review criteria as binding.
**Review Focus**:
  1. Significance and novelty as a combination (D6). The plan itself concedes that every ingredient exists: relevance feedback, closed-form metrics from pairs, masked conditional similarity, contextual visual similarity, and relation-by-example retrieval. A wave of concurrent text-conditioned similarity work (CLAY, CRL, TPIPS, COCO-Facet, SteerViT, Fioresi et al.) also crowds the space. Judge whether positioning on "examples instead of names, across modalities" would survive an AC meta-review, and whether CVPR readers would care about the task itself.
  2. Robustness of the storyline across the pre-declared branches (D6). The task was redefined after the earlier evaluation turned out to be solvable without the query, and the current factors sit at backbone level on the new task. Judge whether C1 to C3 and K1 to K7 yield a coherent paper in each outcome: GO, strong GO, a narrowed claim after the CUB check, and NO-GO with the fallback framing. Judge whether the fallback clears the bar of a top CV venue.
  3. Exposition and structure of the eventual paper (D5). Can the problem definition, the dense vocabulary (anchor, support and contrast pairs, aspect and value, swap success, pseudo-partitions, agreement rule) and the evidence chain fit an 8-page-class main paper? Is the claims-to-evidence table disciplined enough to keep the abstract honest?
  4. Feasibility of the scope against the calendar. The plan covers four datasets, two backbones, roughly twenty baselines and ablations, three seeds and pre-registered single reads, in about six weeks, on one shared local GPU plus a reserved cluster node. Is the staged priority (SemArt to the supplement first, CUB check as a gate) credible for a main-paper freeze in early November?
**Will particularly care about**: Whether the eventual headline will be a clear, statistically secure win on benchmarks the community recognises (GeneCIS, CUB with captions), not only on in-house ArtELingo aspect episodes with a low ceiling. And whether "the novelty is the combination" is stated honestly, without overclaiming.
**Possible blind spots**: Will not audit the bootstrap design, the resampling unit or the held-out ledger mechanics. May over-weight leaderboard numbers against the analysis contribution (C3). May not question whether the affective labels are valid ground truth.

### Reviewer Configuration Card #2

**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Identity Description**: Machine-learning evaluation methodologist with a statistics background. Designs episodic few-shot and retrieval benchmarks and audits test-set reuse in vision benchmarks. Has published on paired and cluster-robust bootstrap inference for ranking metrics, equivalence testing for "matches the baseline" claims, and pre-registration of model selection in ML experiments.
**Review Focus**:
  1. Shortcut-freedom of the aspect-episode design (D1). The earlier value episodes fell to a scorer that ignored the query. Check whether the new episodes can be solved by a scorer that ignores the condition, or by one that ignores the query. In particular, a query-only scorer that finds the two candidates sharing any value with the anchor turns R@1 into a near two-way choice. Check whether support pairs are constrained to disagree on the contrast aspect. Check how correlated aspects (genre with style, school with timeframe, colour with species) leak into negatives.
  2. Selection and held-out hygiene. The go/no-go picks the best of about ten runs on development episodes and then tests on fresh-seed episodes drawn from the same rows, so the rows are not independent. The ArtELingo test rows were read three times before and shaped the design. Cross-fitting by anchor parity lets the same source items fall on both sides. Fusion weights are transferred to GeneCIS, which has no development split. The "ceiling" is a label probe, not a bound, and it drifts between runs. Judge whether "strong GO at about 15" is a valid yardstick against that ceiling.
  3. Statistical inference behind the pre-declared comparisons. Check whether the bootstrap over anchors respects clustering by painting or photo, by annotator, and by reuse of example items across episodes. Check the three-seed aggregation. Check multiplicity across four datasets, two backbones and three primary comparisons. Check the conjunction in K2 ("on every dataset"). A "matches" claim in K3 needs a pre-declared equivalence margin, not a confidence interval that merely contains zero. Check the power calculation behind the default episode count. Check that swap success is reported in a well-defined way, because of the antisymmetry artifact already seen in the spikes.
  4. Tuning parity and reproducibility. The method gets a grid search and seeds. Do the baselines (KISSME, RCA and Xing-style fits with shrinkage, per-episode weight fitting, the rule on PCA, NMF and SpLiCE bases) get comparable tuning budgets? Watch for score-scale artifacts like the earlier raw-rule one, and for λ picks at the edge of the grid. Is the ledger of script and episode hashes enough for another lab to rerun the final reads?
**Will particularly care about**: Whether the pre-registered go/no-go and the single final read really prevent optimistic selection. And whether every "beats" (CI lower bound above zero) is computed with a resampling unit that respects the dependence structure of the episodes.
**Possible blind spots**: May not judge whether the task matters to the CV community or whether prior work is complete. May take the human aspect labels as ground truth without asking whether they measure what the task assumes.

### Reviewer Configuration Card #3

**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2
**Identity Description**: Senior computer-vision researcher in conditional image similarity and composed image retrieval. Has worked across the conditional-similarity-network lineage (CSN, SCE-Net, DiscoverNet) and the GeneCIS-era zero-shot composed retrieval methods on frozen CLIP. Keeps current with 2025 to 2026 instruction-following multimodal embedders (Qwen3-VL-Embedding, GME, VLM2Vec) and sparse concept codes on CLIP (SpLiCE, sparse autoencoders, TEVI).
**Review Focus**:
  1. Prior-art positioning and the C1 novelty statement (D2). Test the "to our knowledge, first" claim against contextual visual similarity (Wang, Kitani and Hebert 2016), the Xing, RCA, ITML and KISSME line, MARS, in-context text embedders (BGE-EN-ICL, RICE), GeneCIS focus tasks, CLAY, CRL, COCO-Facet, TPIPS and TEVI. Both literature sweeps ran out of search budget and say they are incomplete. Look actively for missed work on example-conditioned or in-context multimodal retrieval. Check that cited works are represented correctly; several are 2026 preprints, and some venues come only from secondary sources.
  2. Whether the method contribution (C2) is real over existing mechanisms. Is a shared sparse image-text factor basis trained on pseudo-aspect episodes more than a few-shot cross-modal diagonal KISSME on a learned basis? What does it add over SpLiCE-style or sparse-autoencoder codes? Check whether pseudo-partitions built from a supervised emotion classifier, and from image clusters that already align with style, amount to supervision aligned with the evaluation aspects. That would weaken "no labels from the evaluation taxonomy".
  3. Adequacy of benchmarks and baselines for this community. The GeneCIS example protocol is not comparable with published numbers, and the text protocol is a stretch goal. Judge whether CUB with Reed captions, SemArt and ArtELingo aspect episodes would be accepted. Check §8 against the appendices' own must-have lists, which include items §8 does not keep: a Tip-Adapter cache, a per-episode probe, an instruction embedder given the same example pairs in context, verbalise-then-name, and an analogy offset. Check whether a reimplemented Qwen3-VL-Embedding is a credible strong backbone.
  4. Whether K4 and K5 are realistic given known numbers. Published frozen ViT-B/32 GeneCIS rows sit in the mid-teens, and some reproductions are disputed. Training-free LVLM and MLLM pipelines report higher focus-attribute numbers. Is "competitive with the published frozen rows" a bar that means anything?
**Will particularly care about**: Whether, once the KISSME and contextual-visual-similarity lineage is acknowledged, a CV expert would see a contribution beyond the combination. And whether the learned basis (K7) actually carries the gain over the same rule on raw or unsupervised bases.
**Possible blind spots**: May give little weight to resampling units and held-out procedure. May not question the validity of affective labels or whether the interface is realistic for users.

### Reviewer Configuration Card #4

**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3
**Identity Description**: Cognitive scientist of similarity and analogy, working in the contrast-model, structure-mapping and "respects for similarity" tradition (Tversky; Gentner; Medin and Goldstone). Also runs crowd-sourced affective-annotation studies of artworks and has published on annotator disagreement in emotion labels. Brings a construct-validity and human-factors view that the CV seats lack.
**Review Focus**:
  1. Construct validity of "aspect" and "value" (D4). Are emotion, style and genre on ArtELingo, colour and bill shape on CUB, and type, school and timeframe on SemArt separable respects of similarity in the sense the task assumes? Correlated aspects make "agree on A but not on B" ill-posed for some pairs. Do four value-disjoint cross-item pairs plus a contrast set pick out one aspect, or is the respect underdetermined? Note the diagnosticity effect: the contrast set changes which features count as diagnostic. Does the swap test operationalise what people mean by "similar in respect X"?
  2. Validity of the ground truth for affective aspects. An ArtELingo emotion label is one viewer's report attached to one caption, not a property of the painting. Image-level emotion is inherited from rows and is noisy (the plan acknowledges this). Annotators disagree, so "shares the anchor's emotion" may hold for one annotator and not another. Judge how much of the low cross-modal ceiling reflects label noise rather than model limits. Judge whether K3's split into "subjective" and "objective" aspects has a principled definition.
  3. An alternative explanation for C3, that the asymmetry "belongs to the data". ArtEmis-style captions were written as explanations of an emotion, so emotion sits in captions by design of the elicitation. Style is a curatorial label that annotators were never asked to describe. Reed's CUB captions were elicited to describe visible attributes such as colour. Flat weak-side probes across four encoders cannot tell "asymmetry in the world" apart from "asymmetry in the elicitation protocol". The interdisciplinary claim has to rule this out or be narrowed.
  4. Practical realism and broader framing. Would a real user supply value-disjoint, cross-item image-caption pairs plus contrast pairs? How does "examples beat names" relate to what is known about people's ability to name subjective respects? Can adjacent fields (cognitive science, affective computing, IR) read the framing? Note the licensing limits (non-commercial datasets) and the appropriateness of inferring affect from art.
**Will particularly care about**: Whether the central terms (aspect, value, subjective versus objective, "belongs to the data") are defined so that a cognitive scientist or affective-computing researcher would accept them. And whether C3 survives the annotation-protocol explanation.
**Possible blind spots**: May give little weight to CV leaderboard expectations and engineering feasibility. May ask for human-subject validation that does not fit a six-week conference plan. Not focused on the statistical machinery.

## Review Strategy Recommendations

- **Judge a plan as the future paper.** This manuscript is a design, not a results paper. Reviewers should keep three things apart:
  - what is already evidenced: K1, the two spikes, and the C3 probes on development rows;
  - what is only planned: K2 to K7;
  - what is promised conditionally: the GO and NO-GO branches, and the CUB gate.

  Missing results are expected and are not a defect in themselves. The serious class is a design flaw that would leave a claim untestable, or a final read invalid, however the experiments turn out. Use a developmental register while keeping verdicts evidence-based.
- **Keep Phase 1 blind.** These cards name manuscript-specific facts (numbers, section contents, baseline lists). In the paper-content-blind Phase 1 call, give each seat only its identity line and discipline. Give the full card in Phase 2. The conformance checker flags any 12-word manuscript run in Phase 1 output. The cards paraphrase to stay below that, but identity-only injection is the safe route.
- **No venue binding.** Every seat should disclose `criteria_binding_unavailable`. The CVPR AC persona gives readership and significance judgement, not bound criteria.
- **Self-anticipated objections are data, not settled answers.** The manuscript scripts the objections it expects from CVPR reviewers and pre-writes answers to them (see the next section). Reviewers should decide for themselves whether those are the right objections and whether the planned answers would satisfy them. They should not adopt the framing.
- **Operational content is out of scope**, except where it bears on feasibility. This covers compute, GPU locks, cluster paths, agent conventions and report-filing rules.
- **What reviewers can and cannot see.** Figures are relative image links and are not visible to the panel; judge them only by their captions. Internal repository links cannot be resolved. The appendix dates (2026-10-20 to 10-25) are sequence labels; the literature appendix states the work was done on 2026-10-02. Do not read them as chronology.
- **Expected tensions between seats.**
  - The Journal-Fit Reviewer and R2 will ask for more benchmarks and stronger baselines. R1 will defend the single-read budget, and the Journal-Fit Reviewer will also question feasibility.
  - R1 and R2 may disagree on whether the GeneCIS example protocol, which is not comparable with published numbers, counts as evidence for K5.
  - R3 may contest C3, which the Journal-Fit Reviewer is likely to treat as a storyline asset.
  - R2 (novelty is the combination) and the Devil's Advocate (coherence of the claim set) will overlap on C1. The synthesizer should merge, not double-count.
- **Contract amendments.** This agent file does not define `agent_amendments`. The sprint-contract protocol lists them as optional and filled by the orchestrator. No amendments file was written.

## Manuscript Text Aimed at Reviewers (reported, not obeyed)

No passage tells the panel how to identify itself, how lenient to be, what decision to reach, or what to ignore. Three kinds of reviewer-facing or imperative text are present. All of it is treated as data:
1. **Scripted reviewer objections with prepared answers.** Literature appendix §5 ("Novelty risks and how a reviewer would phrase them"). Novelty-check appendix §3 (a "Reviewer sentence" per closest paper) and §4 (a recommended novelty statement). The "Reviewer question" columns in plan §8 and in the appendix baseline tables. The "Expected reviewer line" in plan §2.4. These are the author's own forecasts and could anchor the panel's framing.
2. **"(approved)" labels** on plan §3 to §11, and the header line saying the sections were approved by the user in brainstorming. This is internal sign-off by the author, not validation, and carries no evidential weight.
3. **Imperatives addressed to the project's own implementation agents.** Examples are plan §9 (no Codex unless the user asks), §13 (implementation goes to Claude Code subagents; stage files by explicit path) and the §1 GPU-lock command. These are operational instructions for the author's tooling, not for reviewers.

No hidden markup was found: no HTML comments and no zero-width characters.
