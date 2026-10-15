# Three-way scan for T2 (supervision for splitting a mixed structure without labels: identifiability)

Scan date 2026-10-05. Mode: ARS deep-research three-way-scan (bibliography + source verification). Thread: `research_brief.md` §7 T2, designs P0, L, G, E (§5), separation signals (§6), evidence (§2).

**Read scope.** Every paper's title, authors, venue, year and id were confirmed on an arXiv abs page, a PMLR page or a NeurIPS proceedings page opened with WebFetch. For nine papers (Locatello 2019, Locatello 2020, iVAE, TCL, Gresele, Lyu, von Kügelgen, Daunhawer, Yao) the abstract was read directly; for four of them (Locatello 2020, iVAE, von Kügelgen, Daunhawer) assumption and theorem details come from WebFetch summaries of the ar5iv full-text rendering. Those details are a small model's summary of the text, not a verbatim reading by me, and are labelled "full text via fetch summary". Direct PDF fetches returned undecodable binary, so no locator was checked against the PDF itself. Papers marked `abstract_only` carry no method-weakness assessment. Retrieved text was treated as data.

**AI-disclosure.** This scan was produced by a Claude agent (Sonnet 5.5) using WebSearch and WebFetch. A human should read Daunhawer et al. 2023, Locatello et al. 2020 and von Kügelgen et al. 2021 in full before any assumption listed below is cited in a paper.

**Queries run (WebSearch).** (1) Hyvarinen Morioka Nonlinear ICA of temporally dependent stationary sources permutation-contrastive learning; (2) Multi-View Causal Representation Learning with Partial Observability Yao ICLR 2024 identifiability; (3) Understanding Latent Correlation-Based Multiview Learning and Self-Supervision: An Identifiability Perspective Lyu Fu; (4) identifiability multimodal contrastive learning modality-specific factors partially shared latent block identifiability 2024 text image; (5) "Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA" NeurIPS 2016 proceedings. The papers named in the brief, plus FactorCL and Shu et al., were opened directly by arXiv id (id taken from memory, then confirmed by the page content; one remembered id, 1605.09522, was wrong and discarded).

**Sources searched.** arXiv abs and ar5iv pages, PMLR, NeurIPS proceedings, general web search results.

**Not searched.** Google Scholar, Semantic Scholar, OpenReview forums (reviews), ACL Anthology, CVF; no query on identifiability with noisy or distant labels (weak-label theory), on sparse-mechanism or mixture-of-experts identifiability (relevant to head competition), on identifiability of conditional-similarity or attribute-conditioned embeddings, or on art-domain disentanglement. The absence claims below are bounded by this.

---

## Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations
Source: ICML 2019 / arXiv 1811.12359 | Year: 2019 | Link: https://arxiv.org/abs/1811.12359
- WHY: Disentanglement methods were claimed to recover independent factors without supervision; the authors ask whether that is possible at all and whether it helps downstream.
- HOW: A theorem showing that unsupervised learning of disentangled representations is impossible without inductive biases on models and data, plus a large study (12,000+ models, per the abstract).
- WHAT: Methods enforce their designed properties, but identifying well-disentangled models needs supervision; more disentanglement did not reduce downstream sample complexity (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Weakly-Supervised Disentanglement Without Compromises
Source: ICML 2020 / arXiv 2002.02886 | Year: 2020 | Link: https://arxiv.org/abs/2002.02886
- WHY: Replace the impossible unsupervised setting with a cheap weak signal: pairs of observations that share some factors.
- HOW: Pairs (x1, x2) where x2 keeps a random subset S of x1's factors and resamples the rest; neither S nor its size is annotated (abstract); the method estimates the number of changed factors at run time.
- WHAT: Knowing only how many factors changed, not which, suffices to learn disentangled representations (abstract). Assumptions (full text via fetch summary, Theorem 1, §4): factors independent, p(z) = prod p(z_i); generator a diffeomorphism; continuous factors; P(S ∩ S' = {i}) > 0 for every factor i (each factor must at some time be the only one shared); a fixed known k; infinite data. Identification is of each factor up to permutation and coordinate-wise reparameterisation.
  - Method weaknesses: reader-inferred (not stated by the authors as such), from the assumptions above. (a) The independence assumption fails when emotion correlates with style or genre, and the result then no longer applies as stated. (b) The condition P(S ∩ S' = {i}) > 0 fails for two factors that always change or stay together, which would leave them unseparated. Author-acknowledged (fetch summary of the limitations discussion, locator not confirmed): the assumptions of unlimited data, known k and smooth invertible generator may not hold in practice.

## Weakly Supervised Disentanglement with Guarantees
Source: ICLR 2020 / arXiv 1910.09772 | Year: 2020 | Link: https://arxiv.org/abs/1910.09772
- WHY: Weak supervision is used for disentanglement without a theory of when it works.
- HOW: A framework based on distribution matching that predicts when restricted labelling, match-pairing and rank-pairing guarantee disentanglement.
- WHAT: The abstract gives the framework and the three techniques only; it reports no numbers.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Variational Autoencoders and Nonlinear ICA: A Unifying Framework (iVAE)
Source: AISTATS 2020 / arXiv 1907.04809 | Year: 2020 | Link: https://arxiv.org/abs/1907.04809
- WHY: A VAE can match the data distribution without recovering the true latents; identifiability fails.
- HOW: Condition a factorised latent prior on an additionally observed variable u (a class label, time index, environment).
- WHAT: The true joint over observed and latent variables is identifiable up to simple transformations (abstract). Assumptions (full text via fetch summary, Theorem 1): conditional prior factorial with each factor in an exponential family; u takes enough distinct values for a variability matrix to be invertible (nk + 1 points); injective mixing; non-vanishing noise characteristic function. Result: identification up to an invertible linear map of the sufficient statistics, and up to permutation and component-wise transformations under further conditions (Proposition 1 notes Gaussian location-only changes cannot get past the linear map). The fetch summary also reports a corrigendum on the discrete-observation proof.
  - Method weaknesses: reader-inferred. The auxiliary variable must change the distribution of the latents (variability); a label that is merely a view or source tag, not a variable the factors depend on, does not obviously meet this. Author-acknowledged: the Gaussian location-only case (Proposition 1, per fetch summary).

## Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA (TCL)
Source: NeurIPS 2016 (proceedings page confirmed by search) / arXiv 1605.06336 | Year: 2016 | Link: https://arxiv.org/abs/1605.06336
- WHY: Nonlinear ICA models proposed earlier were not identifiable.
- HOW: Train a network to discriminate time segments; nonstationarity of the sources over segments is the auxiliary signal.
- WHAT: TCL combined with linear ICA estimates the nonlinear ICA model up to point-wise transformations of the sources, uniquely (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Nonlinear ICA of Temporally Dependent Stationary Sources (PCL)
Source: AISTATS 2017 (PMLR 54) | Year: 2017 | Link: https://proceedings.mlr.press/v54/hyvarinen17a.html
- WHY: TCL needs nonstationarity; many signals are stationary but temporally dependent.
- HOW: Permutation-contrastive learning: logistic regression to discriminate a short window of the data from a temporally permuted window.
- WHAT: The first rigorous, general identifiability proof for nonlinear ICA with temporally dependent, non-Gaussian, stationary sources (page summary).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning (GCL)
Source: AISTATS 2019 / arXiv 1805.08651 | Year: 2019 | Link: https://arxiv.org/abs/1805.08651
- WHY: Unify TCL and PCL under one auxiliary-variable framework.
- HOW: Discriminate real (x, u) pairs from pairs with u randomised; logistic regression or a neural network.
- WHAT: A proof of identifiability and of consistency of the estimator (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA
Source: UAI 2019 | Year: 2019 | Link: https://arxiv.org/abs/1905.06642 (the abs page gives the title as "The Incomplete Rosetta Stone Problem: Multi-View Nonlinear ICA"; the longer subtitle is a memory-based variant and is not used)
- WHY: Single-view nonlinear ICA cannot undo an arbitrary mixing; multiple views might.
- HOW: Several noisy views, each a different mixing of the same independent sources.
- WHAT: Independent latent sources with arbitrary mixing are recoverable if multiple, sufficiently different noisy views are available (abstract). Treats all sources as shared.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Understanding Latent Correlation-Based Multiview Learning and Self-Supervision: An Identifiability Perspective
Source: ICLR 2022 (Spotlight) / arXiv 2106.07115 | Year: 2022 | Link: https://arxiv.org/abs/2106.07115
- WHY: Correlation-maximising deep multiview methods (CCA-style, and self-supervised ones such as BYOL, Barlow Twins) lack identifiability theory.
- HOW: Each view is a nonlinear mixture of shared and private (view-specific) components.
- WHAT: Latent correlation maximisation extracts the shared components; with suitable regularisation, private information can be provably separated from the shared part in each view (abstract). Includes a finite-sample analysis.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style
Source: NeurIPS 2021 / arXiv 2106.04619 | Year: 2021 | Link: https://arxiv.org/abs/2106.04619
- WHY: Explain why contrastive learning with hand-chosen augmentations works.
- HOW: Augmentation changes only a style block; content is invariant. Both a generative and a discriminative setting.
- WHAT: The invariant content block is identified up to an invertible map, allowing statistical and causal dependence among latents (abstract). Assumptions (full text via fetch summary, §3 and Theorem 4.4): only style variables change between views (Assumption 3.1); random subsets of style variables change, each with positive probability and smooth full-support conditional density (Assumption 3.2 and condition iii); the content dimension n_c is known; causal dependence of style on content and non-factorised marginals are allowed. Style is not identified.
  - Method weaknesses: author-acknowledged (fetch summary of §6, locator not confirmed): assumptions may not hold exactly in practice; continuous latents assumed while real data have discrete factors; guarantee is asymptotic and at the global optimum; real image augmentations cause additional variation beyond the intended latents. Reader-inferred: the known-n_c requirement means the split sizes must be fixed in advance.

## Identifiability Results for Multimodal Contrastive Learning
Source: ICLR 2023 / arXiv 2303.09166 | Year: 2023 | Link: https://arxiv.org/abs/2303.09166
- WHY: Prior multi-view results assume one generative mechanism for all views. Image and text come from distinct mechanisms, each with its own modality-specific factors.
- HOW: A generative model with shared (content) factors and modality-specific latent variables per modality; the symmetric multimodal contrastive loss.
- WHAT: Contrastive learning block-identifies the latent factors shared between modalities, even with nontrivial dependencies between factors (abstract). Further detail (full text via fetch summary of ar5iv; locators not confirmed): Theorem 1 gives block identifiability of the shared block (not individual factors); assumptions are that shared factors are invariant across modalities, modality-specific factors vary, mixing is smooth and invertible, and the number of shared dimensions is known; modality-specific factors are not identified. The discussion reports that without a capacity constraint the encoder can also encode non-shared factors, since the loss alone does not force them out. Experiments on Multimodal3DIdent (image and text; shape and position shared, some factors image-only, text phrasing text-only); the fetch summary reports good recovery of shared factors, and 48% to 80% accuracy on the discrete text factor, which the authors attribute to discrete factors violating the continuity assumption. The abstract also reports a corroborating image/text dataset.
  - Method weaknesses: reader-inferred, conditions under which our use of it would be distorted. (a) A shared factor expressed far more strongly in one modality than the other (our emotion: 56.9% from captions against 35.2% from images, brief §2) is still "shared" in theory, but the theory is asymptotic and gives no guarantee about how much of it a finite-sample encoder retains. (b) A property that is nearly image-only (our style, 60.8% from images against 25.4% from captions) is by definition modality-specific here, so the theory says the image-caption objective does not identify it. (c) Known shared-block size and the capacity condition mean an overlarge encoder returns shared and unshared factors mixed. Author-acknowledged (fetch summary): discrete factors violate the continuity assumption.

## Multi-View Causal Representation Learning with Partial Observability
Source: ICLR 2024 / arXiv 2311.04056 | Year: 2024 | Link: https://arxiv.org/abs/2311.04056
- WHY: Earlier results identify only what is shared by all views (their intersection). Real multimodal data have latents shared by some views but not others.
- HOW: Each view is a nonlinear mixture of a subset of latents, which may be causally related; contrastive learning with one encoder per view; graphical "identifiability algebra" to read off which latents are identified.
- WHAT: The information shared across every subset of views is identifiable up to a smooth bijection (abstract), under milder assumptions than earlier work; verified on numerical, image and multimodal data.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Identifiable Multimodal Causal Representation Learning under Partial Latent Sharing
Source: arXiv preprint (no venue found) | Year: 2026 | Link: https://arxiv.org/abs/2605.19135
- WHY: Multimodal data share only some latents; need component-wise identifiability without parametric priors.
- HOW: A Wasserstein-based method to recover the partially shared latent structure; handles undercomplete observations.
- WHAT: Component-wise identifiability guarantees for the causal latent representation; improvements over existing approaches on synthetic and realistic data (abstract). Preprint submitted 18 May 2026, not peer reviewed as far as this search shows; treat as context only.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Factorized Contrastive Learning: Going Beyond Multi-view Redundancy (FactorCL)
Source: NeurIPS 2023 / arXiv 2306.05268 | Year: 2023 | Link: https://arxiv.org/abs/2306.05268
- WHY: Standard multimodal contrastive learning assumes the shared information is sufficient for the task; task-relevant information is often unique to one modality.
- HOW: Split each modality's representation into shared and unique parts; maximise mutual-information lower bounds for task-relevant parts and minimise upper bounds for task-irrelevant parts; multimodal augmentations stand in for task relevance without labels.
- WHAT: State-of-the-art on six benchmarks (abstract). This is an empirical method, not an identifiability result.
  - Method weaknesses: not assessed (read scope: abstract_only).

---

## Cross-paper synthesis

- **Common WHY.** Without extra structure, a representation cannot be pinned to the true factors (Locatello 2019; the nonlinear-ICA non-identifiability behind iVAE and Rosetta Stone). Every positive result adds one thing: an auxiliary variable (TCL, PCL, GCL, iVAE), a second view (Rosetta Stone, Lyu), a pair that changes only some factors (Locatello 2020, Shu), an augmentation that changes only style (von Kügelgen), or a second modality with its own private factors (Daunhawer, Yao).
- **Divergent HOW.** Two families. (1) Component-wise results (iVAE, TCL, PCL, Locatello 2020, Rosetta Stone) need independent factors, and in return identify each factor. (2) Block results (von Kügelgen, Daunhawer, Yao, Lyu) allow dependent factors and identify only a block: what is invariant across the two views or the augmentation. The second family is the one that matches our data, since style, genre and emotion are correlated, but it can only separate "what is shared" from "what is not".
- **Strongest WHAT.** Block identifiability of the shared part, with the private or style part explicitly not identified (von Kügelgen; Daunhawer; Lyu; Yao for every subset of views). Daunhawer also shows the identified block is the part shared by image and text, not the part that is strong in one modality.
- **Unresolved gap (to our knowledge, within this search).** I found no identifiability result for splitting the shared block itself into several properties (style versus genre) without extra variation that changes one and keeps the other; none for competing heads, conditional non-redundancy, or anchoring to a noisy source such as a GoEmotions classifier; and no result for views that are deterministic functions of the same rows (CLIP image, CLIP caption and GoEmotions on the same caption are not independent noisy views). Nothing in the verified set tests the artwork or affect domain.

## Signal map

Cells give what the literature supports. Everything not tied to a citation is reader-inferred. "None found" is bounded by this search.

| Signal (brief §6) | Theory | Empirical evidence | Assumptions it needs | None found? |
|---|---|---|---|---|
| source provenance | Closest: auxiliary variable (iVAE, GCL) or source as a view (Yao 2024: info shared by subsets of views; Lyu 2022: shared versus private per view). Reader-inferred that the source tag is a view, not a conditioning variable. | In-project: hierarchical refinement kept real emotion signal (brief §2), no external evidence. | Views must be different mixings of the same latents with private noise (Gresele); iVAE needs the tag to change the latent distribution. Our sources are derived from the same inputs (GoEmotions and CLIP caption share the caption), so independence of view-specific noise is doubtful (reader-inferred). | Theory exists for views, not for a source tag over derived features. |
| modality | Daunhawer 2023: shared block identified, modality-specific not; Lyu 2022: shared extracted, private separable with regularisation; Yao 2024: every subset of views. | Daunhawer: Multimodal3DIdent image/text, shared factors recovered. In-project: weak modality per aspect across four backbones (brief §2). | Shared block size known; capacity constraint, otherwise unshared factors leak into the code (Daunhawer, fetch summary). | Nothing on how much of a weakly shared factor survives in finite samples. |
| painting membership | Analogue of Locatello 2020 pairs and of von Kügelgen's content-invariant pairs (see section below). Block level only: painting-level versus viewer-level. | None found for ArtELingo-like rows. | Locatello 2020: independent factors, each factor sometimes the only one shared, known k. Block results: shared block fixed, rest varies, block size known. | No theory separates style from genre inside the painting block. |
| non-redundancy | None positive. Locatello 2019: a factorisation or independence penalty without an inductive bias does not recover the true factors (reader-inferred application). | Locatello 2019: 12,000+ models, disentanglement scores did not identify good models without supervision (abstract). In-project: DEC variants lowered separation and agreement (brief §2). | Needs a real inductive bias. | Not identifiable on its own. |
| conditional non-redundancy | None found. | None found. | n/a | None found. |
| agreement versus disagreement between groupings | Yao 2024: info shared by subsets of views is identifiable, so "shared by some groupings only" is a block-level object; Daunhawer and Lyu for the two-view case. Reader-inferred mapping from groupings to views. | In-project: late fusion merged communities, hierarchical refinement kept emotion (brief §2). | Views as mixtures of a latent subset with partial observability; our groupings are outputs of a clustering step, not views of a latent (reader-inferred). | No theory for groupings as such. |
| head competition | None found within this search (no query on sparse-mechanism or mixture-of-experts identifiability was run). | SCE-Net, DiscoverNet are T1 papers, not checked here. | n/a | None found. |
| augmentation invariance | von Kügelgen 2021: the augmentation-invariant block (content) is identified; style, which changes, is not. | Authors' Causal3DIdent experiments (abstract); they acknowledge real augmentations add variation beyond the intended latents. | Augmentation changes only style (3.1), each style dimension changes with positive probability (3.2), n_c known. Our use is the reverse: to isolate style we need an augmentation that changes content and keeps style (crops?) and keeps colour fixed; whether crops leave WikiArt style invariant is untested (reader-inferred). | Theory fits content extraction; style extraction is an inversion of the theorem, not shown. |
| anchoring to the source | None found. Closest: Locatello 2019 (an inductive bias is required), so an anchor is an inductive bias, not a guarantee (reader-inferred). Noisy-label identifiability was not searched. | In-project only (brief §2: content absorbs affect in fused embeddings). | n/a | None found. |

## The painting rows as "pairs sharing some factors"

Rows of one painting share the image and differ in the viewer's caption and emotion (about 5 rows per painting, brief §6). This is the closest analogue to Locatello 2020 pairs, and to von Kügelgen's pairs where only a "style" block changes. What the theory covers and requires:

1. **What would be identified.** Under the block theorems (von Kügelgen 2021, Daunhawer 2023), a contrastive encoder trained on same-painting row pairs would identify the block invariant across the pair, which is the painting-level block (style, genre, and also object content), and would not identify the block that changes (the viewer-level part, mainly emotion). That separates painting-level from viewer-level properties. It does not separate style from genre, because both are in the invariant block and no pair varies one while holding the other (reader-inferred from the theorems; consistent with the Locatello 2020 condition P(S ∩ S' = {i}) > 0 failing for factors that always move together).
2. **Independence.** Locatello 2020 needs independent factors. Emotion correlates with genre and style in art, so only the block theorems apply, and they identify blocks, not individual factors (reader-inferred).
3. **What counts as shared.** Emotion has a painting-level part (the consensus emotion of a painting) and a viewer-level part. The theory puts the consensus part in the shared block and only the idiosyncratic part in the varying block, so the split will not isolate emotion. The brief's ceiling check (rows of one painting sharing an affect group, §10 check 3) measures exactly this part (reader-inferred).
4. **Known block size and a pair structure.** The block results need the invariant block dimension fixed in advance; the pairs must be formed from painting ids, which come from the split and need no evaluation labels. The caption of each row also describes the painting content, so caption-side invariance within a painting is partial; the theory's "only the changing block varies" is approximate here (reader-inferred).
5. **Evidence.** None found for painting-grouped rows or any art-domain pair-based identifiability, within this search.

## Bearing on the designs P0, L, G, E

- **P0 (one grouping per source, no merging).** No identifiability result is needed or supported: P0 never splits. The theory explains why a source grouping is dominated by whatever the source carries (Daunhawer: only shared content is stable in an image-text objective), and offers no way to fix it. Literature-supported negative: Locatello 2019 (no free split).
- **L (refine groupings jointly; conditional non-redundancy, placeability, staying near the source).** Conditional non-redundancy and source anchoring have no identifiability support (none found). Placeability across modalities corresponds to Daunhawer's shared block, which is what a cross-modal agreement objective identifies; that supports the placeable part and gives no guarantee for the weakly shared part (emotion from images). The conditional non-redundancy penalty is an independence-type regulariser and, by Locatello 2019, needs an inductive bias to be meaningful (reader-inferred).
- **G (multiplex graph, shared Leiden partition with layer-specific sub-groupings).** Layers map onto views. Yao 2024 is the nearest theory: what is shared by subsets of layers is a well-defined target. It assumes views are mixtures of latent subsets; a kNN layer over derived features is not (reader-inferred). No result on community detection recovering shared versus layer-specific structure was in this search (that belongs to T1).
- **E (two-tower trunk with property heads; modality, painting membership, head competition, augmentation, anchoring).** The best-supported pieces are two: the image-caption objective identifies the shared block and discards modality-specific factors (Daunhawer), which predicts that style, being mostly image-only, will be dropped or leak depending on capacity; and a painting-pair objective identifies painting-level versus viewer-level blocks (reader-inferred from von Kügelgen and Daunhawer). Head competition and anchoring have no theory. FactorCL (NeurIPS 2023) is the only verified work that explicitly keeps unique per-modality information with a label-free objective; it is empirical with no identifiability proof, so it would be a design precedent, not a guarantee.
- **Practical reading (reader-inferred).** To make style a dominant property of some grouping, the literature points to an extra signal that varies genre or content while holding style fixed (an augmentation, or a source), since no verified result splits a block without one. A style source that is not trained on WikiArt style labels (T4) plays the role of that extra signal.

## Brief citation check

| Brief says | Verified | Correction |
|---|---|---|
| Locatello et al. 2019 (ICML), negative result for unsupervised disentanglement | Yes: "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations", ICML 2019, arXiv 1811.12359 | none; the brief gives no title |
| Locatello et al. 2020, pairs sharing some factors | Yes: "Weakly-Supervised Disentanglement Without Compromises", ICML 2020, arXiv 2002.02886 | the brief had no venue; it is ICML 2020 |
| iVAE, Khemakhem et al. 2020 | Yes: "Variational Autoencoders and Nonlinear ICA: A Unifying Framework", Khemakhem, Kingma, Monti, Hyvärinen, AISTATS 2020, arXiv 1907.04809 | none |
| Time-contrastive / permutation-contrastive learning, Hyvärinen and Morioka | Yes, two papers: TCL (NeurIPS 2016, arXiv 1605.06336) and PCL ("Nonlinear ICA of Temporally Dependent Stationary Sources", AISTATS 2017). GCL (Hyvärinen, Sasaki, Turner, AISTATS 2019, arXiv 1805.08651) is the auxiliary-variable generalisation and is not by Hyvärinen and Morioka | cite the right paper per method |
| Gresele et al. 2019 "The Incomplete Rosetta Stone" | Yes, UAI 2019, arXiv 1905.06642, authors Gresele, Rubenstein, Mehrjou, Locatello, Schölkopf | the abs page title is "The Incomplete Rosetta Stone Problem: Multi-View Nonlinear ICA" |
| von Kügelgen et al. 2021 (NeurIPS), content and style | Yes: "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style", NeurIPS 2021, arXiv 2106.04619 | none |
| Daunhawer et al. ICLR 2023 "Identifiability results for multimodal contrastive learning" | Yes, arXiv 2303.09166; authors Daunhawer, Bizeul, Palumbo, Marx, Vogt | none |

## Not verified

None of the shortlisted 14 failed verification. Items seen only in search results and not opened, listed for context and not cited above for any claim: "Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions" (arXiv 2411.03755); "Hierarchical Contrastive Learning for Multimodal Data" (arXiv 2604.05462, shared, partially shared and modality-specific components); "On Finite-Sample Identifiability of Contrastive Learning-Based Nonlinear ICA" (arXiv 2206.06593); "Contrastive Learning Inverts the Data Generating Process" (arXiv 2102.08850).
Details the fetch summaries gave but I could not confirm against the paper text: the section and theorem locators above, and the exact wording of Daunhawer's assumptions and the 48% to 80% text-factor accuracy (all marked "full text via fetch summary").
