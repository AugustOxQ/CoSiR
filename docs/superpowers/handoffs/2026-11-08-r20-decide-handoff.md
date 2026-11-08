# Handoff: r20: design the qualitative cluster figure and a page-1 teaser figure with the user, build them, then the retrieval check of the eight ablation removals

## Brief
- Done: the user read the what-is-missing table and set the writing plan (2026-11-08). The paper chat writes in its own tab.
- State: no qualitative figure exists. The eight ablation removals of c* have clustering AMI only (held, five seeds); their held exports and weights are saved.
- Next: design both figures with the user, build them from saved outputs, then design and run the retrieval check.
- Blocked: nothing.

> Written 2026-11-08 23:58 by tab "r24 decide". Loop step reached: r20 after decision A, small loop (design in chat,
> run, short auto report and half-page user-read report). Direction log entry:
> [2026-11-08 · What-is-missing table read](../direction_log.md).

## 1. Read in this order

1. The direction log's "Now" and its 2026-11-08 entry: what the user decided.
2. The [experiment overview](../../user_read/2026-11-03_experiment_overview.md), to-do item 20 (near lines 108, 228 and
   266): example paintings per cluster, c* against the plain image.
3. The section "Ablation" of [the recipe sweep report](../../reports/auto/percept/2026-12-28_recipe_sweep.md) (near
   lines 180 to 215) and its "Files and storage" section.
4. `docs/reports/assets/palette.md` (method colours) and the paper's figures,
   `docs/paper/cvpr2027_aspect_spaces/figures/` (fig1, fig2, `captions.md`, `src/`), for the figure style.

Skills: `dataviz` for the figure work.

## 2. Decided by the user, not to be reopened

- r20 is a small loop: the qualitative figure, example paintings per cluster, c* against the plain image (plan of
  2026-11-03, to-do 20).
- r20 also makes a page-1 teaser figure: one painting and its nearest neighbours in the emotion, style and genre spaces
  (user, 2026-11-08, after seeing the recommendation).
- The retrieval check of the eight ablation removals runs now, in this chat (user, 2026-11-08, after seeing the
  recommendation).
- After 11-16, not now: the photo model from random weights, r21 (head variants), TURTLE selection and TEMI, MMArt's
  other texts, the second instructed embedder.
- D1: no text words the limit on c*'s pair numbers until the user has talked to the other authors.

## 3. Where things stand

**Saved outputs** (all under `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/`; contents not checked beyond the
file names):

- c* held exports, per aspect and seed: `full/held/k11=40/{emotion,style,genre}__s{0..4}.{npz,json}`.
- The eight removals beside it, same layout: `full/held/k11=40-no_{buddy,caption_endpoints,content_correction,direct,instructed_graphs,relational,shared_trunk,sharp_row}/`.
- Weights: `full/weights/<same names>/<aspect>__s<seed>.pt`. Cluster partitions: `stage4/leiden/`,
  `stage4/partitions/`, and `ablation/cache/held/`.
- The held clustering results of the removals: `ablation/held.json` (report 2026-12-28, five seeds, 12,281 held
  paintings).

**For the figure:** r17 set the cluster counts ([report 2027-01-04](../../reports/auto/percept/2027-01-04_new_scores.md)).
The painting images' path was not checked here.

## 4. Open points for the next step (with the user, key ones answered first)

- **Key, figure design:** which aspects, which set (ArtELingo held or MMArt), how many clusters, and the rule that picks
  clusters and paintings. The rule must be fixed before the paintings are seen and stated in the caption, so that the
  figure is not picked by hand.
- **Key, teaser design:** which painting, how many neighbours per space, and the comparator (the plain image's
  neighbours beside them, or not).
- **Key, the retrieval check:** which cells (image retrieval in label mAP, as in r17; text retrieval would need the
  caption side and a short GPU pass), seeds 0 to 4, c* as the reference, no bar (exploration).
- **Minor:** image rights. Prefer paintings in the public domain for a paper figure; check each painting's artist dates.

## 5. Code and pitfalls

- New round folder in `src/test/` (`yyyymmdd_r20_figures/`) with its log, and a new module prefix (`r20_`), since round
  folders go on `sys.path`.
- GPU work through `res-queue gpu`; CPU-heavy work through `res-queue cpu` (global `shared-resources.md`).
- Sample IDs join by position (constitution C6): use the IDs from the export, never `range(n)`.
- Paper figures go to the paper folder (`docs/paper/`, its own private repo); the user-read report may carry a copy.
- Delete nothing without the user's approval.

## 6. State at handoff

- Running: nothing. The paper chat runs in its own tab.
- Uncommitted: nothing after this cut's commit.
- On disk over 1 GB: nothing new.
