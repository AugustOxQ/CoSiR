# More graph neighbours lift held emotion and style above the instructed image; genre stays below by less than one seed range

> 2026-10-28 · [Full report 12-28 (CoSiR numbers its reports in sequence, not by date)](../reports/auto/percept/2026-12-28_recipe_sweep.md) · Previous briefing: [relational genre, 2026-10-27](2026-10-27_relational_genre.md)

## Summary

- G, the previous model, passes the held emotion claim against PercepT at both training scales.
- The recipe search picks c*: G with 40 graph neighbours, instead of 20; other changes give small or negative gains.
- On held paintings, c* beats the instructed Qwen image on emotion and narrowly on style.
- Genre remains below the instructed image by less than one seed range, the normal spread between model seeds.
- The relational term carries genre; the instructed graphs, the buddy terms and the content correction carry emotion and style.
- The pick replaces G under the fixed rule; the held results support claims only for Qwen, k-means and ArtELingo.

## How we got here

CoSiR aims to support the paper's emotion, style and genre claims with one condition-free image-text model.
The CVPR abstract is due on 2026-11-10.
The model reads an image without an aspect instruction and gives separate outputs, called heads, for each aspect.
Its teacher is the instructed Qwen image: an embedding made with an instruction for an aspect.

G is the previous model, with a genre loss that copies the teacher's pattern of similarities between paintings.
Its sharper teacher target gives the nearest paintings more weight.
Against the instructed image, G won on emotion and was within the agreed margin on style.
Its genre was below the instructed image by less than one seed range.
The seed range is the spread of scores between seeds, which set the random choices in training.

This loop asks whether G passes the emotion claim against PercepT, the published method that CoSiR seeks to improve.
It then asks whether a stronger recipe exists, whether the pick beats the instructed image, and which parts carry its result.
A recipe is the set of training losses and their settings.

## G passes the held emotion test against PercepT

**Problem.** G's emotion score was known, but its margin over PercepT had not passed the formal claim test.

**Idea.** Test G against PercepT core, our corrected project baseline, at both sample and full training scales.
The sample scale uses less training data; the full scale uses the full training set.
Use held paintings, kept apart from the development paintings that serve recipe choices.
Score clusters with AMI, adjusted mutual information, which measures agreement with labels.
The claim needs a large enough gain, with an uncertainty interval above zero.
The interval reflects uncertainty across paintings and model seeds.
This checks whether the model chosen for genre still supports the emotion claim for the paper.

**Did it work?** Yes, at both scales. This was close to known because G's held emotion score had already been read.

**Key evidence.** Held emotion AMI gains over PercepT core, with five seeds.

| Training scale | G minus PercepT core | Claim's required gain | Result |
|---|---:|---:|---|
| Sample | +0.0188 | at least +0.0100 | Pass; interval above zero |
| Full | +0.0164 | at least +0.0100 | Pass; interval above zero |

> Full report: [Results](../reports/auto/percept/2026-12-28_recipe_sweep.md#results).

## The recipe search picks more graph neighbours

**Problem.** Most of G's settings had never been varied. We wanted the strongest recipe across all aspects together.

**Idea.** First vary each of thirteen settings alone on development paintings, at the same seed.
Then test a joint grid over the four settings with the best gains.
This asks whether useful changes add up when combined.
Finally test the top three candidates on fresh seeds, against G on those same seeds.
These seeds are fresh to these runs, but earlier rounds also used them to choose models.

Rank candidates by mean gain: each aspect's gain divided by its seed range, then averaged across aspects.
This gives each aspect a comparable unit, despite their different score scales.
A guard limits losses on any aspect, so a large style gain cannot hide a large genre loss.
Confirmation also needs the mean gain's interval above zero.

The winning change broadens the graph from twenty to forty neighbours per painting.
Graph neighbours are paintings linked as nearby in the instructed image views; these links guide what the model learns.
Broader links could supply more useful positive examples and explain the style gain. That explanation remains untested.
The pick, c*, keeps every other setting of G.

**Did it work?** Yes. More graph neighbours won alone and after the joint grid.
Other settings gave small or negative gains; combinations did not improve on the graph change alone.
The mean-gain intervals overlap, so the rule picks c* without proof that it clearly beats the runner-up.
The first search selected among about sixty configurations, which inflates its best scores.

**Key evidence.** Development confirmation on seeds one to three; AMI means and mean gain against G.

| Candidate | Emotion AMI | Style AMI | Genre AMI | Mean gain, in seed ranges | Qualifies? |
|---|---:|---:|---:|---:|---|
| G, reference | 0.1062 | 0.3931 | 0.5050 | reference | reference |
| c*: more graph neighbours only | 0.1089 | 0.4072 | 0.5019 | +0.683 | Yes |
| More neighbours, style relational loss, narrower shared network | 0.1085 | 0.4020 | 0.4990 | +0.369 | Yes |
| More neighbours, softer teacher target | 0.1082 | 0.3991 | 0.5006 | +0.272 | No; interval crosses zero |

The relational loss copies the teacher's pattern of similarities; a softer target spreads weight more widely across paintings.
Most of c*'s confirmed gain comes from style. Its genre difference from G is uncertain.

> Full report: [Stage 1](../reports/auto/percept/2026-12-28_recipe_sweep.md#stage-1-node402-knob-curves), [Stage 2](../reports/auto/percept/2026-12-28_recipe_sweep.md#stage-2-node402-top-runs) and [Stage 3](../reports/auto/percept/2026-12-28_recipe_sweep.md#stage-3-local-confirmation).

## The held pick beats the instructed image on emotion and style

**Problem.** Development gains choose a recipe; they do not establish its held performance against the teacher.

**Idea.** Choose c* before its held test, then train it at full scale with five seeds.
Compare each head with the instructed Qwen image for that aspect, under the same k-means cluster method.
This tests whether a model that needs no aspect instruction at use can beat the aspect-specific image embeddings it learns from.
A win needs the difference's whole interval above zero.
For genre, the agreed margin permits a shortfall smaller than one seed range.
The test covers the ArtELingo painting dataset.
The held paintings and teacher scores had been read before; G became the recipe after its held genre result.

**Did it work?** Partly. Emotion and style beat the instructed image; the style win has little room above zero.
Genre is below the instructed image by less than one seed range, and its whole interval is below zero.
The style win against the teacher does not establish a style gain over G: that difference's interval crosses zero.

**Key evidence.** Held AMI on 12,281 paintings, with genre labels on 9,949; differences have 95% intervals.

| Aspect | c* | Instructed Qwen image | c* minus instructed image | Reading |
|---|---:|---:|---|---|
| Emotion | 0.1066 | 0.0921 | +0.0145 | Win; interval above zero |
| Style | 0.4066 | 0.4026 | +0.0041 [+0.0004, +0.0081] | Narrow win |
| Genre | 0.5001 | 0.5062 | −0.0060 [−0.0103, −0.0015] | Below by less than the 0.0128 seed range |

c* also passes the emotion claim against PercepT core, with a gain of +0.0192.
Against G, its held style gain is +0.0009; its genre mean is unchanged at the reported precision.
The large development style gain therefore does not carry through as a clear held gain over G.

> Full report: [Stage 4](../reports/auto/percept/2026-12-28_recipe_sweep.md#stage-4-held-full-scale).

## Which parts carry the result (ablation)

**Problem.** We knew c* works, but we did not know which of its parts carry the result.

**Idea.** Remove one part at a time from c* and retrain, with nothing else changed or retuned.
The eight parts do the following:

- The relational term makes the genre head copy the teacher's pattern of similarities between paintings.
- The sharper teacher row gives the nearest paintings more weight in that pattern.
- The direct teacher makes each head copy the instructed Qwen image for its aspect.
- The buddy terms bring linked paintings together through their images and captions.
- The caption endpoints let the buddy terms use captions as well as images.
- The content correction adjusts the emotion and style graphs for content.
- The instructed graphs link nearby paintings in the instructed image view for each aspect.
- The shared trunk processes the input for all heads; removing it gives each aspect its own trunk.

Removing the caption endpoints leaves only images in the buddy terms.
Removing the instructed graphs makes all aspects use one graph from the default image view.

The removals ran on development paintings with three seeds and on held paintings with five seeds at full scale.
The ablation decides nothing.

**Did it work?** Yes: the relational term carries genre.
On held paintings, removing the relational term from c* changes genre AMI by −0.0440.
The instructed graphs, the buddy terms and the content correction carry emotion and style.
Removing the instructed graphs from c* changes style AMI by −0.1903 and emotion AMI by −0.0327.
Removing the buddy terms or the content correction changes both aspects in the same direction.

On held paintings, removing the direct teacher from c* changes emotion AMI by +0.0035 and style AMI by +0.0041.
Removing the caption endpoints from c* changes style AMI by +0.0072.
All three intervals are above zero.
On development paintings, removing the direct teacher from c* changes emotion AMI by +0.0022, with its interval above zero.
Removing the caption endpoints from c* changes style AMI by +0.0042, with its interval above zero.

The held numbers for c* without the caption endpoints already exist.
A later held test of that recipe would be close to known.

**Key evidence.** The table compares held AMI of c* with each part removed, minus c*, over five seeds and 12,281 held paintings.

| Removal | Emotion Δ | Style Δ | Genre Δ |
|---|---:|---:|---:|
| Relational term | +0.0009 | 0.0000 | −0.0440 |
| Sharper teacher row | +0.0016 | +0.0008 | −0.0038 |
| Direct teacher | +0.0035 | +0.0041 | −0.0031 |
| Buddy terms | −0.0277 | −0.1512 | −0.0015 |
| Caption endpoints | +0.0009 | +0.0072 | −0.0034 |
| Content correction | −0.0090 | −0.0752 | +0.0072 |
| Instructed graphs | −0.0327 | −0.1903 | +0.0054 |
| Shared trunk | −0.0001 | +0.0023 | −0.0001 |

> Full report: [Ablation](../reports/auto/percept/2026-12-28_recipe_sweep.md#ablation).

## Advice

<details><summary>Our advice (open after giving your reading)</summary>

Our view: we trust the emotion result most.
c* beats the instructed Qwen image by about three seed ranges and PercepT core by +0.0192.
The style win passes the rule but is narrow.
c*'s held style gain over G is +0.0009, with its interval across zero.
Genre stays within the agreed margin of the instructed image.

The ablation shows that the instructed graphs, the buddy terms and the content correction each carry many seed ranges on style.
The direct teacher and the caption endpoints look unneeded because development and held paintings agree on the direction for emotion and style, respectively.
The outcomes fit our earlier expectations.

Our recommendation: first try dropping the caption endpoints, because this costs little and may become the recipe for the other tests.
Then test the other image encoders before the machine reservation ends.
Draft the abstract in parallel, since the emotion claim holds with either recipe.

</details>

## Next

c* replaces G under the fixed decision rule.
The held outcomes do not trigger any next experiment automatically.
You give your reading, then choose the next work from these options:

- Drop the caption endpoints from c*.
  On development paintings, this gains +0.315 seed ranges over c*, with an interval above zero.
  It costs under an hour and could give the recipe for later tests.
  Its held numbers already exist, so a held test would be close to known.
- Test the recipe with SigLIP 2, SigLIP 1 and CLIP L/14, other image encoders.
  This costs about half a day of GPU work and shows whether the recipe transfers to other image encoders.
- Draft the CVPR abstract from the supported claims.
  This costs writing time and fixes what the paper can claim.

Aspect-named retrieval, the second task, has no result yet.

These choices can overlap, as our advice proposes. The current evidence covers Qwen, k-means and ArtELingo only.

> Full report: [Options for next and Our recommendation](../reports/auto/percept/2026-12-28_recipe_sweep.md#results-brief).

## Glossary

- G: the previous recipe, with a relational genre loss and sharper teacher target; GS-0.02 in the previous report.
- c*: G with graph neighbours increased from 20 to 40; the pick that replaces G.
- Recipe: the training losses and their settings.
- Head: the model output for one aspect: emotion, style or genre.
- Instructed Qwen image (T): an image embedding made with an instruction for one aspect; the held comparison bar.
- PercepT core: the project's corrected PercepT baseline; the emotion comparison uses its own clusters.
- Seed range: an earlier model's score spread across seeds; emotion 0.0046, style 0.0082, genre 0.0128.
- Mean gain: gain over G divided by the seed range for each aspect, then averaged across aspects.
- Graph neighbours (mutual top-k, k11): nearby paintings linked in the instructed views to guide training.
- Relational loss: a term that makes the model copy the teacher's pattern of similarities between paintings.
- Direct teacher: this term makes each head copy the instructed Qwen image for its aspect.
- Buddy terms: these terms bring linked paintings together through their images and captions.
- Caption endpoints: these let the buddy terms use captions as well as images.
- Content correction: this adjusts the emotion and style graphs for content.
- Instructed graphs: these link nearby paintings in the instructed image view for each aspect.
- Shared trunk: this network processes the input for all heads.
- Sharper teacher row: this target gives the nearest paintings more weight in the teacher's pattern of similarities.
- Development paintings: the paintings used to compare recipes and choose the pick.
- Held paintings: the separate paintings used for claim tests; this split had been read in earlier rounds.
- AMI: adjusted mutual information, the agreement between clusters and labels.
- k-means: the cluster method used for the main comparisons.
- Ablation: tests that remove recipe parts to find which carry the result.
- T-WIN and T-LEVEL: the full report's outcomes for a win over T or a shortfall within the agreed margin.
- PE-GO: the held emotion claim passes against PercepT core.
- V4b: the completed held-output check, which passed.
