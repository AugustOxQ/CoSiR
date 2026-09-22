# ArtELingo stage report: fusion-mechanism investigation

## I. Why this investigation started

The preceding ArtELingo affect investigation established that an off-the-shelf affect signal and a properly converged Deep Embedded Clustering (DEC) method could each yield real but modest emotion gains, but neither made a Pareto improvement over the content-only buddy graph: both gained emotion structure by sacrificing much of its genre advantage. It therefore closed with a more specific question: could a smarter **fusion mechanism**, rather than a better encoder or a better clustering method applied to one signal, recover emotion structure while keeping most of content's genre structure? This investigation tests that question directly.

## II. Glossary

**AMI (Adjusted Mutual Information).** The primary score here, measuring how much two groupings of the same paintings agree—for example, discovered communities and human genre labels. It is adjusted for agreement expected by chance, so a score near zero means no better-than-chance structural agreement and a larger score means stronger agreement.

**V-measure.** A related 0–1 clustering-agreement score: the harmonic mean of homogeneity (whether each found group mostly contains one true category) and completeness (whether each true category mostly falls in one found group).

**Buddy graph.** A graph whose nodes are paintings and whose edges connect paintings that are mutually similar under an embedding. It supplies the neighborhood structure later partitioned into communities.

**Mutual-kNN.** Mutual K-nearest neighbors: an edge is retained only when each endpoint places the other among its K nearest neighbors. This is a strict, reciprocal local-neighborhood criterion.

**Leiden community detection.** An algorithm that divides a graph into densely connected communities without being told their number in advance.

**K and alpha.** K is the number of nearest-neighbor candidates considered when constructing a buddy graph. Alpha is the fixed coefficient that balances the image and text contributions of the content graph; these pilots use K=20 and alpha=0.5 unless stated otherwise.

**Early versus late fusion.** Early, or feature, fusion combines content and affect feature vectors before nearest-neighbor search, so the two signals jointly determine every candidate similarity. Late, or edge/graph-level, fusion first makes a separate graph for each view and only then combines their retained edges.

**Union graph and graph intersection.** A union graph retains any edge that appears in either view; an intersection retains only edges present in both. They test, respectively, permissive and strict forms of graph-level agreement.

**Hierarchical or nested clustering; parent and child community.** A hierarchical partition first fixes broad parent communities, then allows smaller child communities only within each parent. Here content communities are the parents and affect may make children, never merge across parents.

**Matched-granularity control.** A comparison partition that makes the same-sized splits as the tested refinement but without its proposed signal. It is essential because AMI is not monotone under refinement: splitting a cluster into smaller pieces can raise or lower chance-corrected AMI even if the split adds no information. A same-size-split control distinguishes real signal from a metric change caused solely by making more labels.

**Similarity Network Fusion (SNF).** A multi-view method that repeatedly diffuses each view's local similarities and reinforces them with the other view, intended for views that describe a shared local manifold.

**Co-regularized multi-view spectral clustering.** A clustering method that encourages the low-dimensional spectral representations of multiple graph views to agree, assuming they share meaningful large-scale cluster directions.

**Linear CCA (Canonical Correlation Analysis) and canonical component.** Linear CCA finds one direction in each of two feature sets whose values across matched paintings are as correlated as possible; each paired direction is a canonical component. A held-out canonical correlation is that correlation on paintings excluded from fitting. A permutation null repeats the fit after scrambling content–affect pairings, providing the correlation expected without genuine correspondence.

**Conditional or partial residual.** The part of one view left after predicting it from the other, here by regressing affect features on content features. Testing this leftover asks whether affect has useful structure beyond what overlaps content.

**Two-teacher contrastive student.** A learned shared embedding trained to respect positive neighbor pairs supplied by two teacher graphs, one content and one affect. Its InfoNCE loss makes a positive pair more similar than the other examples in its batch; those other examples are **in-batch negatives**. A **projection head** maps a teacher's input features to the common embedding. A **gate** or **gating network** learns how much each projected view contributes for a painting.

**Effective rank and representation collapse.** Effective rank summarizes how many embedding directions carry meaningful variance; a very low value can indicate collapse to a few directions. **Gradient share** isolates each loss term's contribution to the update and reports its fraction of the gradient magnitude. Together with gate saturation, these diagnostics show whether one teacher is silently dominating the other.

**Pareto improvement / Pareto bar.** In this investigation, a method clears the Pareto bar only if it beats the best prior multi-signal emotion AMI and the best prior multi-signal genre AMI simultaneously, rather than improving one while worsening the other. The content-only and affect-only results remain single-signal reference ceilings rather than a joint-fusion target.

## III. The five things tried

### 1. Late fusion — union and intersection

Late fusion builds the content and GoEmotions affect buddy graphs independently in their native feature spaces and then combines their edge sets, unlike early fusion's blending of feature vectors before neighbor search. The union graph reached emotion AMI=0.1236 and genre AMI=0.1394, exceeding early fusion's best 0.1160 / 0.0867 on both axes. It nonetheless remained far below the content-only genre result of 0.4384, so it did not escape the basic trade-off. The raw intersection was degenerate: 98.96% of nodes were isolated before repair; after the necessary repairs its AMIs were 0.0510 emotion and 0.2399 genre, which are not interpretable as a useful shared graph.

The union's genre cost was not a smooth dilution of every content boundary. The content graph's 28 communities became 21 under union, and the smallest community grew from 6 to 324 members. Affect edges therefore bridge particular content communities strongly enough for Leiden to merge them outright, explaining why its improved emotion AMI comes with a sharp genre loss.

### 2. Hierarchical refinement

This mechanism fixed the content partition as a hard parent constraint, then allowed GoEmotions-based Leiden clustering only within a parent. Predeclared guards required a sufficiently large parent and cross-seed stability before a split could stand. Of 28 parents, 23 were eligible, stable, and genuinely split; they covered 61,298 of 61,402 paintings (99.8%).

The result demonstrated real, control-verified affect signal: hierarchical refinement reached emotion AMI=0.1072, whereas the exactly size-matched random-split Control A reached 0.0362, a 0.0709 margin that is well beyond the predeclared 0.02 bar. It did not protect genre sufficiently: genre AMI=0.1954 was below the 80%-retention floor of 0.3507 and only marginally different from Control A's 0.2014. Thus most genre cost came from the granularity change itself—28 parents becoming roughly 400 or more final labels—not specifically from affect. The content re-split Control B, which was not size-matched, is not a substitute for that attribution control.

### 3. Classical joint-fusion candidates: SNF and co-regularized spectral clustering

The second two-way brainstorm assessed SNF and co-regularized spectral clustering but deliberately did not run them. This was an evidence-based deprioritization, not an oversight: both mechanisms require reinforcing, locally agreeing cross-view structure, whereas the content and affect mutual-kNN graphs had a near-empty intersection.

For SNF, the specific falsifiable prediction was a smoothed union: repeated diffusion would add off-block mass or affect bridge paths and reduce content-block contrast, rather than discover a new sparse shared backbone. It would only merit a pilot if it first produced materially more high-confidence pairs supported by both views without reducing content-community conductance. For co-regularized spectral clustering, the prediction was that a strong agreement penalty would rotate unrelated content and affect eigenspaces toward a compromise, while a weak penalty would return two nearly separate solutions. It would require nonrandom leading-eigenspace alignment before construction. Neither condition was supported by the observed edge evidence, so both were set aside in favor of a cheaper test of whether shared signal existed globally rather than locally.

### 4. The CCA audit

The CCA plus conditional-residual audit was the pivot point because it tested that global shared-signal question before authorizing a learned joint representation. Its top held-out canonical correlation was 0.7285, far above both the predeclared 0.15 bar and the corresponding permutation-null 95th percentile of 0.0699. All ten retained components also exceeded their own null thresholds. This is real, substantial, held-out-replicated linear correlation.

This reconciles the earlier near-orthogonal framing rather than overturning it. Mutual-kNN asks a strict local question—whether a pair is in each other's top 20—whereas CCA asks a softer global question about aligned continuous variation. The two views are nearly orthogonal at the former scale but share genuine structure at the latter. The local caveat remained important: a joint-CCA-space graph recalled only 7.18% of held-out content neighbors and 7.83% of affect neighbors, versus random floors of about 0.1% (0.11% and 0.12%). That is roughly a 65× lift over chance (0.0718/0.0011 and 0.0783/0.0012), but far from strong local preservation.

The residual result was also a caution, not confirmation of an independent affect axis. After removing the linearly content-predictable affect component, residual-affect-only emotion AMI was 0.0914, below both the raw affect ceiling of 0.1180 and the predeclared 80%-retention bar of 0.0944. The emotion-useful affect signal therefore substantially overlaps the content-correlated component; it is not mainly an orthogonal leftover, as hierarchical refinement alone might have suggested.

### 5. Learned two-teacher student — Stage 1 and Stage 2

The positive CCA audit licensed a minimal learned student: independently normalized content and GoEmotions inputs were mapped by linear projection heads into a shared 32-dimensional space, combined through a scalar gate, and trained with equally weighted symmetric two-teacher InfoNCE losses. Stage 2 was a controlled capacity-only comparison: it changed only the heads to small two-layer MLPs with 64 hidden units.

| stage and split | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Stage 1 — linear, train | 0.1284 | 0.1289 | 0.2799 | 0.3014 |
| Stage 1 — linear, held-out | 0.1095 | 0.1123 | 0.2901 | 0.4240 |
| Stage 2 — MLP, train | 0.1230 | 0.1235 | 0.2319 | 0.2553 |
| Stage 2 — MLP, held-out | 0.1046 | 0.1072 | 0.2087 | 0.3531 |

Stage 1 was the only method in this investigation to clear both predeclared Pareto targets on train: emotion AMI=0.1284 and genre AMI=0.2799. Its held-out result held up reasonably: genre AMI improved slightly to 0.2901, while emotion AMI fell to 0.1095, missing the 0.1236 bar by about 11%. This is a materially gentler generalization gap than the supervised BERT ceiling pilot in the prior investigation, although it is still not a held-out Pareto result.

The run's mechanical predeclared rule labeled Stage 1 **Collapsed** because final content recall (0.1314) was below the true epoch-0, pre-training value (0.1690). That label is too coarse to describe the observed model: final gradient share was 0.4988, converging near a 50/50 balance, and gate saturation was 0% at every checkpoint. It is better characterized as a genuine, non-degenerate two-teacher trade-off. The rule's limitation is that recall direction alone cannot distinguish healthy rebalancing from one-sided starvation.

Stage 2 is a clean negative result. More capacity made both AMI axes worse on both train and held-out splits, despite remaining non-collapsed by all four diagnostics: both teacher recalls retained signal, gradient shares stayed balanced, gate saturation stayed at 0%, and effective rank remained healthy. This directly rejects the claim that the teachers were capacity-limited. Per the predeclared evidence-based rule from both brainstorms, the investigation stopped escalating architecture here—not because it ran out of ideas, but because the evidence argued that deeper heads or cross-attention were not the useful next step.

## IV. Cross-method summary

Unless noted otherwise, this table reports train-split AMI. That is a caveat: the Stage 1 and Stage 2 held-out values above are the only held-out data points available within this investigation.

| method | emotion AMI | genre AMI |
|---|---:|---:|
| Content-only | 0.0593 | 0.4384 |
| GoEmotions-only (affect ceiling) | 0.1180 | 0.0396 |
| Early fusion (best point) | 0.1160 | 0.0867 |
| Late fusion — union | 0.1236 | 0.1394 |
| Late fusion — intersection | 0.0510 | 0.2399 |
| Hierarchical refinement | 0.1072 | 0.1954 |
| Learned student, Stage 1 (linear) | 0.1284 | 0.2799 |
| Learned student, Stage 2 (MLP) | 0.1230 | 0.2319 |

Stage 1 is the Pareto-best point among the multi-signal methods tried: it has the highest genre AMI of any method with non-trivial emotion AMI and the highest emotion AMI of any method with non-trivial genre AMI. It is the empirical frontier of this investigation, while the content-only and affect-only rows remain single-signal references rather than a joint solution.

## V. What this means, and what is still open

No method found a clean escape from the content–affect trade-off: every approach gave up real genre structure to gain emotion structure. The Stage 1 learned two-teacher student found the best available point on that frontier by a clear margin, using the simplest learned-fusion architecture tested. Stage 2 showed that added head capacity made the result worse, not better.

The following are explicitly open and were not tasked in this investigation:

- Stage 1's held-out emotion AMI of 0.1095 narrowly misses the 0.1236 Pareto bar. A small, principled loss-weight adjustment between teachers, informed by the now-available gradient-share diagnostic rather than another architectural change, was not tested.
- Content-anchored re-ranking—letting affect break ties only inside a narrow content-similarity band—was the third-ranked idea from the first brainstorm and was never run.
- Reconnecting these findings to Experiment 18's RedCaps retrieval-versus-interpretability trade-off remains untested. Every result here is ArtELingo-only; RedCaps has no emotion labels for a comparable GoEmotions affect teacher, so any reconnection needs a label-free affect proxy rather than GoEmotions directly.

## VI. Process note

This investigation used the same brainstorm-then-verify cycle as the earlier six-pilot affect investigation: a two-way brainstorm between the user, this session, and an independently dispatched Codex pass; a locked, fact-specific implementation brief; independent code review of every script before execution; and GPU execution retained by this session rather than delegated.
