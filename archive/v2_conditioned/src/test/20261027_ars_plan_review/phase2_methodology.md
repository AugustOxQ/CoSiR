contract_role: methodology
## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "Any claim whose planned test cannot separate success from failure but could still be repaired by redesigning the protocol before results are collected"
block_class: repairable

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: warn
trigger: "Claims mostly follow from the evidence but some are stated more strongly than their tests allow"

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

Reviewer 1 (methodology): an ML evaluation methodologist who designs episodic retrieval benchmarks and audits test-set reuse. The manuscript is a pre-results plan for a CVPR paper on example-conditioned aspect similarity across modalities, with five appended evidence reports. Following the contract note, I judge whether the planned design, baselines, held-out protocol and statistics can decisively support or refute each claim (K1 to K7, C1 to C3), and how the appended evidence, including its negative results, bears on them. No venue criteria were bound for this panel, so this card makes no venue-alignment claim. I treated the scripted reviewer objections and the "(approved)" labels as author data; none of them tried to instruct the panel.

As a pre-results document, the plan is unusually disciplined. It has a hash-logged single-read held budget, cross-fitted fusion weights and paired bootstrap intervals. It reports swap success beside R@1, and it states its negative diagnostics openly; those diagnostics have already forced one task redesign. Three design gaps, however, leave claims testable only in a weaker sense than they are stated:

1. The primary metric and the GO rule can be passed by a scorer that ignores the condition (W1).
2. The anchor-level bootstrap ignores heavy reuse of paintings and species across episodes, so "CI lower bound above 0" is anti-conservative (W2).
3. "Matches" in K3 has no equivalence margin (W3).

Each gap matches my D1 block trigger, and each can be repaired at E13, before any final read. D1 is therefore block (repairable), not fatal: CUB, SemArt and GeneCIS still offer unread test data, and no evaluation label is circular with the training signal.

On D3, the plan acknowledges its negative evidence (K1, and factors at CLIP level on aspect episodes). But K2, K7 and C3 are worded more broadly than their planned tests can show (W4, W5), so D3 is warn.

### S1: A single-read held protocol with a hash ledger
Every final read is logged with script and episode SHA-256, and each final script refuses a second run. The reserve read is limited to a pre-registered fix and may never be used for a second attempt at a better number. The plan also discloses that ArtELingo's held rows shaped the earlier design and mitigates this with fresh-seed episodes on a new task. This is stronger held-out hygiene than most CV submissions show.
**Evidence Anchor**: text: §10 Ledger "Each final script refuses a second run."

### S2: Negative diagnostics are reported and acted on
Both spikes report paired 95% CIs. When the evidence went against the method, they said so plainly: the factor models sit within 0.3 points of CLIP on aspect episodes, and value episodes were solved without the query. Those results changed the task definition and the K2 status line ("factors are at CLIP level today"). This is the right inferential posture for a go/no-go plan.
**Evidence Anchor**: text: aspect-episode spike, Verdict "On cross-modal aspect episodes no factor model selects the aspect, although the task is learnable."

### S3: The baseline set targets the specific threats to the method claim
Tier 1 includes several baselines that answer the obvious "few-shot diagonal KISSME" objection directly:
- raw-feature metric-from-pairs fits (KISSME and RCA with shrinkage, Xing, Wang et al. per-query weights);
- the same rule on PCA, NMF and SpLiCE bases (K7);
- a value prototype;
- the earlier factor recipes.

All of these run on the same features and episodes.
**Evidence Anchor**: text: §8 Baselines "All baselines run on the same features and episodes."

### S4: Score-scale artifacts are caught rather than reported as evidence
The support spike traced raw_naive's CLIP-level score to a factor term about 1e-3 against a cosine and withdrew it as evidence. Cross-fitting by anchor parity is used for every tuned weight.
**Evidence Anchor**: text: support-baseline spike, Result 4 "It is a scale artifact, not a test of the naive rule on raw"

### S5: Aspect episodes remove the K1 support-only shortcut
With value-disjoint examples, the value prototype that solved label episodes falls to CLIP level. That is direct evidence that the new design blocks the old shortcut.
**Evidence Anchor**: table: aspect-episode spike Result 1, Value prototype (cv) row, pooled 11.08, minus CLIP −0.04 [−0.26, +0.17]

### S6: The CUB unseen-species split tests the pseudo-partition risk
Factor training uses the 150 training species. Development uses 30 of them, kept out of factor training, and the final test uses the 50 unseen species. This is a clean out-of-distribution test of R-pseudo (the risk that the model learns the episode format rather than aspects).
**Evidence Anchor**: text: §5.2 CUB row "development on 30 of the 150 training species, kept out of factor training"

### S7: Reproduction checks precede new scorers
Each spike first reproduced the stored episodes and ranks (SHA-256) before running new scorers. The backbone check reproduced the CLIP aspect cells exactly and documented probe drift. My GRIM recomputations below found every per-direction R@1 cell I checked to be reachable at the stated episode count.
**Evidence Anchor**: text: support-baseline spike, What we tested "reproduced the stored episodes (SHA-256), CLIP only, C0 and SE ranks exactly before any new scorer was run"

### W1: Condition-blind scorers can pass the primary metric and the GO rule
**Problem**: Each aspect episode places two aspect candidates (p_A and p_B) among 11 negatives that share neither value, and the primary metric averages R@1 over both conditions. Take a scorer that ignores the condition but ranks candidates sharing any value with the query above the rest. Under condition A it ranks p_A first with some probability q; under condition B it ranks p_B first with probability 1 − q. Its condition-averaged R@1 can therefore reach 50%. That exceeds the label-probe ceiling of about 23, the GO bar (CI above backbone-only at 11.1) and the strong-GO mark of about 15.

All §10 primary comparisons and the GO rule use R@1 alone. Swap success is shown "at matched R@1" but enters no decision. No planned control runs the trained factors with the condition removed (uniform w, or w from the swapped condition). A GO, or a K2 pass, could therefore come from better generic cross-modal matching on emotion and style. Such a result would not support C2's claim that the basis lets four example pairs set the similarity. K1 found that the old episodes could be solved without the query; this is the mirror risk, episodes solved without the condition.
**Evidence Anchor**: text: §5.1 Common protocol "R@1, the mean over both directions and all conditions" and "one sharing aspect A with the anchor (p_A), one sharing aspect B (p_B), and 11"
**Why it matters**: This decides whether the headline number measures conditional similarity at all, the exact failure that ended the label-episode benchmark.
**Suggestion**:
- Add "ours with uniform weights" and "ours under the swapped condition" as Tier-1 controls.
- Pre-declare "ours vs ours without the condition" as a primary comparison.
- Add a condition-sensitivity requirement to GO, for example a CI lower bound above 0 for condition-correct minus condition-swapped R@1, or the condition-averaged pairwise accuracy of W9 above 50%.
- Report a condition-blind probe ceiling (posteriors of both aspects combined) beside the aspect-aware one.
**Severity**: Major
**Confidence**: 4 — core expertise: shortcut analysis of episodic retrieval benchmarks

### W2: The bootstrap over anchors ignores item reuse, so "CI lower bound above 0" is anti-conservative
**Problem**: §10 resamples anchors, but episodes reuse items from a fixed pool. Each episode holds 30 distinct items, with 4,096 anchors per aspect pair:
- A held ArtELingo painting appears in about 10 episodes per aspect pair (4,096 × 30 / 12,281).
- A SemArt test painting appears in about 115 (4,096 × 30 / 1,069), and each painting is an anchor about four times.
- CUB's final test holds 50 species, roughly 2,950 photos by our estimate from 11,788 images over 200 species, and colour and bill shape are largely species-level.

Captions also share annotators, and rows share paintings. The anchor bootstrap treats the item pool as fixed. Its intervals answer "would this hold for new episodes over these items", not "for new paintings or species", and they will be too narrow for the latter. The support spike's own caveat notes that parity cross-fitting lets paintings straddle the two halves.
**Evidence Anchor**: text: §10 Statistics "paired bootstrap over anchors (5,000 resamples)"
**Why it matters**: Every pre-declared "beats" and the GO decision rest on this interval. The false-positive rate of K2 and of GO is therefore above nominal by an unknown amount, most of all on CUB and SemArt.
**Suggestion**:
- State the inference target.
- Use a painting-level cluster bootstrap (species-level on CUB) that resamples items and regenerates or reweights episodes, or a two-way anchor × item bootstrap.
- Base the power calculation (W12) on the same unit, and report CUB results per species.
**Severity**: Major
**Confidence**: 4 — core expertise: cluster-robust bootstrap for ranking metrics

### W3: "Matches" in K3 has no equivalence margin
**Problem**: §10 defines "beats" as a CI lower bound above 0, but "matches" is left to a per-aspect-type declaration with no criterion. If "matches" is read as a CI that contains 0, then a noisy or underpowered comparison passes by default, and the claim can be refuted only by a significant loss. K3's "match it on objective ones" is then not a testable equivalence claim.
**Evidence Anchor**: text: §4 claims table, K3 "Examples beat naming the aspect on subjective aspects and match it on objective ones"
**Why it matters**: K3 carries the "examples versus names" argument for the task. As planned, half of it cannot fail.
**Suggestion**: Pre-declare an equivalence margin δ in R@1 points per dataset before E14 (for example a fixed fraction of the backbone-to-ceiling gap). Test it with two one-sided tests, or require the 90% CI to fall inside ±δ, and power the episode count for it. Otherwise, word K3 as "not significantly different" and drop "match".
**Severity**: Major
**Confidence**: 5 — core expertise: equivalence testing for "matches the baseline" claims

### W4: K2 and K7 are worded more broadly than the pre-declared comparisons can show
**Problem**: K2 claims a win over three baseline families "in both directions, on every dataset". §10, however:
- pools directions into one primary metric;
- makes the unsupervised-basis comparison descriptive, although that comparison is the K7 test and central to C2;
- states no rule for partial passes, such as three of four datasets or one backbone.

GeneCIS is image to image and has no aspect episodes, so "both directions, on every dataset" cannot hold there as worded. The CUB development check narrows claims if A fails, but the held reads have no equivalent rule.
**Evidence Anchor**: text: §4 K2 "the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset" and §10 "All other comparisons are descriptive."
**Why it matters**: The claims table will feed the abstract. As written, K2 and K7 could be reported from pooled or descriptive numbers that cannot carry them.
**Suggestion**:
- Either promote "ours vs the rule on the best unsupervised basis (chosen on development)" to a primary comparison, or state K7 as descriptive.
- Add per-direction tests, or narrow K2 to pooled directions.
- List the datasets K2 covers, leaving GeneCIS to K5.
- Pre-register what is claimed under each pattern of passes.
**Severity**: Major
**Confidence**: 4 — core expertise: aligning pre-registered tests with claim wording

### W5: C3's "belongs to the data" cannot be separated from how the emotion labels were made
**Problem**: C3 rests on weak-side probes that stay flat across four encoders (emotion from images 35.1 to 36.6, majority class 31.8). Each ArtELingo row, however, is one viewer's emotion. A painting carries about five rows (308,723 rows over 61,402 paintings), and those rows can disagree. An image-only predictor can at best output a painting's modal emotion, so its row-level accuracy is capped by the share of rows that agree with that mode, whatever the encoder. Flat image probes are therefore what the label construction alone predicts.

E12 plans more probes of the same kind. Those cannot separate "the pixels do not hold emotion", which is the appendix's wording, from "image-level emotion is per-viewer". The style half of C3 is less exposed, because style is a painting-level label.
**Evidence Anchor**: text: §4 C3 "the effect persists across four backbones" and backbone check Result 1 "a better encoder cannot read what the pixels or words do not"
**Why it matters**: C3 is one of three contributions and a pillar of the NO-GO fallback. Its causal reading is not yet supported, although the descriptive asymmetry stands.
**Suggestion**: On scorer-train rows, compute the label-construction ceiling for image-to-emotion (the mean share of each painting's modal emotion). Report the weak-side probes as a fraction of that ceiling, and score image-side emotion against painting-level majority or distribution labels as well. Unless the normalised gap remains, word C3 as an asymmetry of the annotated data and its elicitation.
**Severity**: Major
**Confidence**: 3 — adjacent: label-noise ceilings; construct validity is Reviewer 3's remit

### W6: GeneCIS is evaluated during development although it has no development split
**Problem**: §10 lists GeneCIS with no development split and a 1 + 1 read budget. Yet E7 (Oct 13 to 20) runs the example protocol and baselines on GeneCIS focus attribute and outputs a "GeneCIS table", before the E14 final reads (Oct 24 to Nov 1). E9 also runs Qwen "on every dataset", and the example protocol draws its supports from the benchmark's own templates. As scheduled, E7 is either a test read outside the ledger or a step whose output is undefined.
**Evidence Anchor**: text: §11 E7 "GeneCIS focus attribute: example protocol, COCO-trained factors, baselines; text protocol as stretch" and "GeneCIS table"
**Why it matters**: GeneCIS is the one community-standard benchmark in the plan. If it is read during development, K5 cannot be presented as a held-out result, and the plan's own "each held-out test set is read once" promise fails for it.
**Suggestion**: Limit E7 to pipeline checks on COCO, plus at most a small, disclosed slice of GeneCIS templates excluded from the final read. Move every reported GeneCIS number to E14, and log any earlier read in the ledger. Apply the same rule to E9 and E10.
**Severity**: Major
**Confidence**: 3 — core expertise: test-set reuse audits; E7's intended output is not fully specified

### W7: Fresh-seed GO episodes reuse the selection rows, so best-of-ten selection still inflates
**Problem**: The plan says fresh seed-43 episodes keep the best-of-ten pick from inflating the GO result. But each selection painting appears about 19 times per episode set (4,096 × 30 / 6,451), so the seed-42 and seed-43 sets are recombinations of the same items. Fresh seeds remove episode-sampling noise but not item-level overfit of the chosen run, and the selection rows have already been read many times. The plan also does not say how the method's own fusion weight is set in the GO comparison (the baseline's weight is cross-fitted). Finally, "seed 43" names both an episode seed and a training seed.
**Evidence Anchor**: text: §6 Go/no-go "from the same rows, so picking the best of about ten runs does not inflate the result"
**Why it matters**: The risk is limited to a false GO, because the held reads still protect the paper. It does waste the schedule.
**Suggestion**: Split the selection paintings into two disjoint halves: pick on one, run the GO test on the other. Cross-fit the method's weight the same way as the baselines', and rename the two seed roles.
**Severity**: Minor
**Confidence**: 4 — core expertise: selection bias in model selection

### W8: The strong-GO yardstick uses a ceiling that is neither a bound nor stable
**Problem**: The label-probe ceiling is described in the appendices as a diagnostic, not a strict bound. It drifted by up to 0.25 points between identical runs, and it uses an aspect-aware scorer that a condition-blind scorer could exceed (W1). "A third of the way to the ceiling" therefore has no fixed meaning.
**Evidence Anchor**: text: §6 Strong GO "a third of the way to the ceiling of about 23"
**Why it matters**: The label carries no decision, but it will frame how the eventual numbers are read.
**Suggestion**: State strong GO as a pre-declared absolute effect over the best baseline, with its CI. Use converged, fixed-thread ceilings for description only.
**Severity**: Minor
**Confidence**: 4 — core expertise: benchmark headroom analysis

### W9: Swap success has a null that depends on the scorer, and its unit is unstated
**Problem**: Swap success is 0 for every condition-blind scorer and about 25% for independent random scores. For antisymmetric scores it is about 50%, which the spike observed at chance-level R@1. Its value cannot be read without knowing the scorer's form. Its unit is also ambiguous: C0's 4.43 is unreachable as a share of 4,096 anchors but reachable over 8,192 anchor-directions (receipt AR11).
**Evidence Anchor**: text: aspect-episode spike Result 2 "about 50% for any ordering: it reached 51.6 and 51.8 while R@1 was 8.0 and 8.6"
**Why it matters**: K6 and the condition-sensitivity question both lean on this metric.
**Suggestion**: Add the condition-averaged pairwise accuracy, the mean of P(p_A above p_B under A) and P(p_B above p_A under B). It equals exactly 50% for every condition-blind scorer and about 50% for random antisymmetric ones, so it has a single null. State whether the unit is anchors or anchor-directions.
**Severity**: Minor
**Confidence**: 4 — core expertise: metric design for paired conditional tests

### W10: The headline CI omits training-seed variance
**Problem**: Bootstrapping anchors over per-anchor seed means conditions on the three trained models, so the training procedure's between-seed variance never enters the interval. The claim, however, is about the method.
**Evidence Anchor**: text: §10 Seeds "The headline is the 3-seed mean, with its CI from bootstrapping anchors"
**Why it matters**: A favourable seed can carry the mean. Three seeds cannot give a strong seed-level interval, but they can show consistency.
**Suggestion**: Require each pre-declared comparison to pass for every seed, or report the minimum-seed effect, next to the pooled CI. Report the between-seed SD of each effect.
**Severity**: Minor
**Confidence**: 4 — core expertise: seed variance in ML evaluation

### W11: The multiplicity family is not defined
**Problem**: §10 names three primary comparisons. It does not say whether K2 is an intersection test over four datasets × two backbones (no correction needed, but every test must pass) or a claim on any subset (which needs a correction). Nor does it say how the per-aspect-type "beats" and "matches" declarations enter the family. Which raw metric-from-pairs baseline counts as "the best" is also not fixed before the final read.
**Evidence Anchor**: absence: §10 Statistics — expected a declared family and correction or intersection rule for the primary comparisons across datasets, backbones and aspect types; checked §4 claims table, §10, §11 E13, §12 risks
**Why it matters**: Without the family, a mixed outcome can be reported either way.
**Suggestion**: Use an intersection-union rule for "every dataset" and Holm across backbones and aspect types for anything weaker. Fix "the best" baseline on development.
**Severity**: Minor
**Confidence**: 4 — core expertise: multiplicity in pre-registered ML comparisons

### W12: The power calculation is not specified
**Problem**: The plan gives no target effect, α, power or resampling unit. Computed on anchor-level selection variance, it will overstate power under the dependence in W2. Beyond the size of the item pool, extra anchors add little information; SemArt, for instance, has 1,069 test paintings.
**Evidence Anchor**: text: §10 "come from a power calculation on selection variance (default 4,096 anchors per aspect pair)"
**Why it matters**: The episode count fixes the precision of every final read, and each read happens once.
**Suggestion**: Specify the minimal effect of interest and the margin from W3. Compute power by simulation under the cluster bootstrap, and report item counts as well as anchor counts.
**Severity**: Minor
**Confidence**: 4 — core expertise: power analysis for episodic benchmarks

### W13: Baseline tuning budgets and a grid-edge rule are not specified
**Problem**: The method gets about ten grid runs over partitions, L and loss weights, plus seeds. §8 fixes cross-fitted fusion weights for the baselines but no grids for their internal hyperparameters: KISSME shrinkage and rank, RCA shrinkage, Xing step count, PCA and NMF dimension, SpLiCE sparsity. Both spikes also picked λ at a grid edge.
**Evidence Anchor**: text: support-baseline spike, Caveats "every pick for the prototype and probe terms sat at that edge"
**Why it matters**: K2's win over metric-from-pairs fits is only as strong as those fits are tuned.
**Suggestion**: Pre-register a grid for each baseline that is at least as large as the method's. Extend any grid whose pick lands on an edge, and report every pick.
**Severity**: Minor
**Confidence**: 3 — core expertise: tuning parity; the baseline grids may exist outside the plan

### W14: The score form and its transfer across datasets are not fixed
**Problem**: §6 writes the score at raw scale, while the spikes fuse per-episode z-scores with λ. The spikes showed raw-scale artifacts of both kinds: a negligible raw term, and factor rows that dominate the cosine (16 to 18% swap at chance R@1). GeneCIS fusion weights come from other datasets, but its image-to-image cosines sit on a different scale from image-to-caption cosines.
**Evidence Anchor**: text: §8 "they are chosen on the development split and frozen. GeneCIS, which has no development" with §6 "s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y)" against the aspect spike "The final score is z(cos) + λ · z(term)"
**Why it matters**: A transferred weight on the wrong scale can bias K5 for the method or against it.
**Suggestion**: Fix one fusion form for every method and baseline; per-episode z-scoring makes weights transferable. Report GeneCIS sensitivity to the transferred weight.
**Severity**: Minor
**Confidence**: 3 — core expertise: score fusion; the final form may be intended but is not stated

### W15: Third-aspect and correlated-aspect values of the candidates are not controlled
**Problem**: §5.1 does not state that p_A must differ from the anchor on aspect B. With three aspects, it also does not say whether p_A, p_B and the negatives are balanced on the third aspect. Genre correlates with style, and on CUB colour correlates with species. p_B may therefore share the anchor's third-aspect value more often than p_A, which gives condition-blind scorers a systematic tilt toward one condition.
**Evidence Anchor**: text: §5.1 "With three aspects, all three aspect pairs"
**Why it matters**: An imbalance like this shifts per-condition R@1 and the swap metrics in ways that have nothing to do with conditioning.
**Suggestion**: State exclusivity, constrain or balance the third aspect across p_A, p_B and the negatives, and report R@1 split by whether the target shares the third aspect.
**Severity**: Minor
**Confidence**: 3 — core expertise: episode construction; the code may already enforce exclusivity

### W16: A guessed genre name can handicap the privileged-names baseline
**Problem**: The name for genre id 5 was inferred by elimination. The privileged-names baseline projects items onto the true value names, so a wrong name lowers only the naming side of the K3 comparison on genre.
**Evidence Anchor**: text: §5.3 "is inferred by elimination and alphabetical order"
**Why it matters**: The bias runs in the direction of the claim being tested.
**Suggestion**: Verify the name against the ArtGAN or WikiArt metadata, or exclude id 5 from genre episodes. Report K3 on genre without it as a check.
**Severity**: Minor
**Confidence**: 3 — adjacent: label provenance

### W17: Text and tables disagree in the backbone appendix
**Problem**: Four statements in the backbone appendix do not match its own tables or §2.3:
- The text says CUB colour probes are "within 2.3 points", but SigLIP 2 reads 61.9 against 64.5, a gap of 2.6.
- "ArtELingo's gaps of 21 to 35 points" omits the style gaps of 44.8 (SigLIP 2) and 45.6 (PE).
- "About 62 minutes of GPU time" disagrees with the timing table, which sums to 5,257 s, about 87 minutes.
- "37,738 selection paintings with 92,413 captions" conflicts with §2.3, which gives selection as 6,451 paintings and 32,413 rows. The 92,413 equals 60,000 scorer-train probe rows plus 32,413 selection rows, so the label is wrong, not the hygiene.
**Evidence Anchor**: table: backbone check Result 3, Primary colour (15, 21.0) row, SigLIP 2 cell 61.9 / 64.5
**Why it matters**: None of these changes a decision. They matter because E16 promises to re-derive every load-bearing number.
**Suggestion**: Correct the four statements, and generate appendix prose numbers from the JSON.
**Severity**: Minor
**Confidence**: 5 — direct recomputation from the manuscript's tables

### W18: The subjective/objective split for K3 is fixed after the development comparisons
**Problem**: E10 runs the naming baselines on development (Oct 12 to 20) before the E13 pre-registration, where "beats" or "matches" is declared per aspect type. The held read then replicates a pattern derived on development, which is valid confirmation. But the assignment of aspects to "subjective" and "objective" can be fitted to the development outcomes, which makes K3's explanation circular.
**Evidence Anchor**: text: §11 E13 "Pre-registration and power for the final reads"
**Why it matters**: K3's interpretive half depends on that assignment.
**Suggestion**: Fix the subjective/objective assignment now, from a criterion that does not depend on model results (for example inter-annotator agreement on the aspect), and keep it whatever E10 shows.
**Severity**: Minor
**Confidence**: 3 — core expertise: pre-registration timing

### W19: CUB's final-test species were already read in the backbone check
**Problem**: The backbone check ran its CUB probes and retrieval on the standard 5,794-image test split. That split spans all 200 species, so it includes the 50 unseen species reserved for the final CUB test (about a quarter of its images, by our estimate). Those reads informed the backbone choice and C3's "colour is symmetric". E0's choice of the third attribute group uses the same probe criterion, with no split stated. §10 discloses prior reads only for ArtELingo.
**Evidence Anchor**: text: backbone check Result 3 "Retrieval sanity check, 5,794 test images, first caption each"
**Why it matters**: The contamination is light, since these were probes and not the method, but it is undisclosed. It touches a test split the plan calls fresh, and it feeds C3's CUB evidence.
**Suggestion**: Log and disclose the read. Run E0's choice and all CUB probes on training species only, and recompute C3's CUB evidence on training species.
**Severity**: Minor
**Confidence**: 3 — inference: 5,794 matches CUB's standard test split size

### Statistical reporting judgement, recomputation note and questions

**Statistical reporting** (Step 4a: paired intervals, effect sizes, power, multiplicity, assumptions): PARTLY_MEETS.
- Effect sizes are R@1 differences with paired 95% CIs in both spikes.
- The backbone check gives no intervals and uses one seed, yet it carries C3 and the backbone choice.
- Power, multiplicity, the resampling unit and the equivalence margin are not yet specified (W2, W3, W11, W12).

This is decision-bearing for D1, and every gap can be fixed at E13.

**Recomputation.** I ran GRIM on twelve reported values (AR1 to AR12).
- Every per-direction R@1 cell I checked is reachable at 4,096 episodes under round-half.
- One cell (AR10) is reachable under round-half but not under truncation, so it is marked not computable.
- The swap values in AR11 and AR12 depend on the unit (see W9).

I also rebuilt the pooled values from their cells and found them consistent: aspect CLIP 1,823/16,384 = 11.13%, cross-modal ceiling 3,783/16,384 = 23.09%, and genre coverage 81.04%, 81.48% and 81.01%. No receipt is a mismatch. The inconsistencies I found are in the prose (W17).

**Questions for the authors.**
1. Must p_A differ from the anchor on aspect B, and is the third aspect controlled?
2. Does E7 compute any metric on GeneCIS templates?
3. Is swap success computed per anchor or per anchor-direction?
4. Which fusion form (raw β, or per-episode z-scores with λ) will the final reads use?
5. Will the GO decision also require condition sensitivity?

## Arithmetic Receipts

### AR1
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Emotion row, CLIP i2t 9.42
reported_inputs: R@1 9.42 percent, two decimals; 4,096 aspect episodes; R@1 defined as the share of episodes where the target ranks first; direction named (i2t)
assumptions: each of the 4,096 episodes contributes one 0 or 1 top-1 outcome per aspect condition and named direction, as the share-of-episodes definition states; rounding rule unstated, so round-half and truncation are both checked
derivation: 9.42/100 x 4096 = 385.84; k = 386 gives 9.4238 percent and k = 385 gives 9.3994 percent
derived_value_or_range: 386/4096 = 9.4238 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [9.415, 9.425) under round-half; [9.42, 9.43) under truncation
nearest_achievable: 385/4096 = 9.3994 and 386/4096 = 9.4238
status: consistent

### AR2
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Emotion row, CLIP t2i 10.86
reported_inputs: R@1 10.86 percent, two decimals; 4,096 aspect episodes; share-of-episodes definition; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 10.86/100 x 4096 = 444.83; k = 445 gives 10.8643 percent and k = 444 gives 10.8398 percent
derived_value_or_range: 445/4096 = 10.8643 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [10.855, 10.865) under round-half; [10.86, 10.87) under truncation
nearest_achievable: 444/4096 = 10.8398 and 445/4096 = 10.8643
status: consistent

### AR3
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Art style row, CLIP t2i 13.45
reported_inputs: R@1 13.45 percent, two decimals; 4,096 aspect episodes; share-of-episodes definition; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 13.45/100 x 4096 = 550.91; k = 551 gives 13.4521 percent and k = 550 gives 13.4277 percent
derived_value_or_range: 551/4096 = 13.4521 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [13.445, 13.455) under round-half; [13.45, 13.46) under truncation
nearest_achievable: 550/4096 = 13.4277 and 551/4096 = 13.4521
status: consistent

### AR4
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Emotion row, Ceiling i2t 23.58
reported_inputs: R@1 23.58 percent, two decimals; same 4,096 aspect episodes; share-of-episodes definition; direction named (i2t)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 23.58/100 x 4096 = 965.84; k = 966 gives 23.5840 percent and k = 965 gives 23.5596 percent
derived_value_or_range: 966/4096 = 23.5840 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [23.575, 23.585) under round-half; [23.58, 23.59) under truncation
nearest_achievable: 965/4096 = 23.5596 and 966/4096 = 23.5840
status: consistent

### AR5
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Emotion row, Ceiling t2i 24.29
reported_inputs: R@1 24.29 percent, two decimals; same 4,096 aspect episodes; share-of-episodes definition; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 24.29/100 x 4096 = 994.92; k = 995 gives 24.2920 percent and k = 994 gives 24.2676 percent
derived_value_or_range: 995/4096 = 24.2920 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [24.285, 24.295) under round-half; [24.29, 24.30) under truncation
nearest_achievable: 994/4096 = 24.2676 and 995/4096 = 24.2920
status: consistent

### AR6
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Art style row, Ceiling i2t 21.41
reported_inputs: R@1 21.41 percent, two decimals; same 4,096 aspect episodes; share-of-episodes definition; direction named (i2t)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 21.41/100 x 4096 = 876.95; k = 877 gives 21.4111 percent and k = 876 gives 21.3867 percent
derived_value_or_range: 877/4096 = 21.4111 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [21.405, 21.415) under round-half; [21.41, 21.42) under truncation
nearest_achievable: 876/4096 = 21.3867 and 877/4096 = 21.4111
status: consistent

### AR7
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Art style row, Ceiling t2i 23.07
reported_inputs: R@1 23.07 percent, two decimals; same 4,096 aspect episodes; share-of-episodes definition; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 23.07/100 x 4096 = 944.95; k = 945 gives 23.0713 percent and k = 944 gives 23.0469 percent
derived_value_or_range: 945/4096 = 23.0713 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [23.065, 23.075) under round-half; [23.07, 23.08) under truncation
nearest_achievable: 944/4096 = 23.0469 and 945/4096 = 23.0713
status: consistent

### AR8
procedure_id: grim
evidence_anchor: table: support-baseline spike Result 3, Style t2i (images) row, Prototype + query 47.31
reported_inputs: R@1 47.31 percent, two decimals; 4,096 art-style label episodes; R@1 as the share of episodes where the positive ranks first; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode and named direction, as the share-of-episodes definition states; rounding rule unstated, both round-half and truncation checked
derivation: 47.31/100 x 4096 = 1937.82; k = 1938 gives 47.3145 percent and k = 1937 gives 47.2900 percent
derived_value_or_range: 1938/4096 = 47.3145 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [47.305, 47.315) under round-half; [47.31, 47.32) under truncation
nearest_achievable: 1937/4096 = 47.2900 and 1938/4096 = 47.3145
status: consistent

### AR9
procedure_id: grim
evidence_anchor: table: support-baseline spike Result 3, Emotion t2i (images) row, SE 15.84
reported_inputs: R@1 15.84 percent, two decimals; 4,096 emotion label episodes; share-of-episodes definition; direction named (t2i)
assumptions: one 0 or 1 top-1 outcome per episode and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 15.84/100 x 4096 = 648.81; k = 649 gives 15.8447 percent and k = 648 gives 15.8203 percent
derived_value_or_range: 649/4096 = 15.8447 percent
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
rounding_interval: [15.835, 15.845) under round-half; [15.84, 15.85) under truncation
nearest_achievable: 648/4096 = 15.8203 and 649/4096 = 15.8447
status: consistent

### AR10
procedure_id: grim
evidence_anchor: table: aspect-episode spike Result 3, Art style row, CLIP i2t 10.77
reported_inputs: R@1 10.77 percent, two decimals; 4,096 aspect episodes; share-of-episodes definition; direction named (i2t)
assumptions: one 0 or 1 top-1 outcome per episode, aspect condition and named direction; rounding rule unstated, both round-half and truncation checked
derivation: 10.77/100 x 4096 = 441.14; k = 441 gives 10.7666 percent and k = 442 gives 10.7910 percent
derived_value_or_range: round-half interval [10.765, 10.775) contains 441/4096 = 10.7666; truncation interval [10.77, 10.78) contains no attainable value
comparison_rule: consistent when an attainable k/4096 x 100 falls in the rounding interval under every candidate rounding rule; not_computable when the verdict depends on the rule
status: not_computable
not_computable_reason: rounding_rule_ambiguous

### AR11
procedure_id: grim
evidence_anchor: text: aspect-episode spike Result 2 "C0 4.43"
reported_inputs: swap success 4.43 percent, two decimals; 4,096 aspect episodes; defined as the share of anchors meeting both pairwise orders; two retrieval directions scored
assumptions: rounding rule unstated, both round-half and truncation checked; the paper does not state whether the denominator is 4,096 anchors or 8,192 anchor-directions
derivation: at N = 4096, 4.43/100 x 4096 = 181.45 and 181/4096 = 4.4189, 182/4096 = 4.4434, neither in [4.425, 4.435) nor [4.43, 4.44); at N = 8192, 363/8192 = 4.4312 lies in both intervals
derived_value_or_range: unreachable at N = 4096; 363/8192 = 4.4312 percent at N = 8192
comparison_rule: consistent when an attainable k/N x 100 falls in the rounding interval for the analytic N the paper states; not_computable when the verdict depends on an unstated N
status: not_computable
not_computable_reason: analytic_n_ambiguous

### AR12
procedure_id: grim
evidence_anchor: text: aspect-episode spike Result 2 "R3 6.65"
reported_inputs: swap success 6.65 percent, two decimals; 4,096 aspect episodes; defined as the share of anchors meeting both pairwise orders; two retrieval directions scored
assumptions: rounding rule unstated, both round-half and truncation checked; the paper does not state whether the denominator is 4,096 anchors or 8,192 anchor-directions
derivation: at N = 4096, 6.65/100 x 4096 = 272.38 and 272/4096 = 6.6406, 273/4096 = 6.6650, neither in [6.645, 6.655) nor [6.65, 6.66); at N = 8192, 545/8192 = 6.6528 lies in both intervals
derived_value_or_range: unreachable at N = 4096; 545/8192 = 6.6528 percent at N = 8192
comparison_rule: consistent when an attainable k/N x 100 falls in the rounding interval for the analytic N the paper states; not_computable when the verdict depends on an unstated N
status: not_computable
not_computable_reason: analytic_n_ambiguous
