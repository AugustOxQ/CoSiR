# Handoff: round 2 of the grill (backbones, comparator, the recipe rule, minor batch), then the spec of decide 17's two-part loop

## Brief
- Done: decision C on the D2 ablation, the wrap-up, decision A (two parts), grill round 1 (fresh sample, recipe rule, ArtELingo-28's role).
- State: part 1 checks the caption-free variants on all 29,232 train-side paintings outside the old sample; part 2 builds code for other backbones and ArtELingo-28.
- Next: grill round 2 (three key questions, then the minor batch), the closing summary, then the spec.
- Blocked: nothing.

> Written 2026-10-25 02:32 by "r10 decide 17" (cut at the context meter, 200k). Loop step reached: step 5 (the grill),
> round 1 done. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md) (its Idea (B) line is
> filled at the grill's close).

## 1. Read in this order

1. This handoff.
2. The facts the grill rests on: `.scratch/r10-decide-17/grill_facts.md` (Codex, reviewed: fresh sample, backbones,
   ArtELingo-28 labels), then `.scratch/r10-decide-16/f2_facts.md` §2 (stores), §3 (what D2 needs per piece), §5
   (hard-wired Qwen and 2048-width code) and `.scratch/r10-run-12/a28/a28_overlap_facts.md`.
3. The direction log's top entry (decisions C and A) and the ablation's results brief,
   [12-22](../../reports/auto/percept/2026-12-22_d2_ablation.md).
4. Framing note [12-21](../../reports/auto/percept/2026-12-21_advantages_framing.md), summary item 7: "any backbone"
   only if a SigLIP 2 or CLIP student stays above PercepT core.
5. Before the grill: `~/.claude/references/grill.md`; before the spec: `spec-templates.md` and the templates it names.

Skills: grilling, domain-modeling (the grill runs as grill-with-docs).

## 2. Decided by the user, not to be reopened

- **Decision C** (2026-10-21) and **decision A** (2026-10-25): in the direction log's top entry. Two parts in one
  medium loop. Part 1, on development paintings: the caption-free variants ("image endpoints only", "no own-caption
  terms") beside D2 with the teacher at 1/3, all three aspects; the genre information-ceiling probe (literature note
  12-20, idea 1); the four-view statistics sheet (norms, anisotropy, default-to-instructed cosine, per-label neighbour
  purity, silhouette, hubness). Part 2: code for other backbones and ArtELingo-28, with the recipe as a setting.
- **Retrieval for 11-10:** image queries with an aspect name only (user, after seeing the recommendation).
- **Q1, the fresh sample:** all 29,232 train-side paintings outside the 19,889-painting sample (23,080 train, 6,152
  validation), not a 5,000 draw. Why: "b will be a more comprehensive choice" (user, before the recommendation; ours
  was also b, for the power on the small emotion gain). Training stays on the 13,400 fit paintings with the old
  1,489 early-stopping paintings. The old 5,000 are reported beside it.
- **Q2, how part 1 picks part 2's recipe:** a rule fixed in the spec, "a more flexible fixed rule" (user, after seeing
  the recommendation; first answer was "descriptive", changed after the what-if). The rule's form and thresholds are
  round 2's first key question; the run chat applies it, so part 2's runs need no wait.
- **Q3, ArtELingo-28:** (a) a test set: train as now, read the default image of the 746 read-set paintings (248 held,
  498 outside our store), score image AMI against its labels; and (c) a culture test, descriptive only: per-language
  painting labels from about 3 to 4 votes each, no claim (user, before the recommendation; ours was a only; the user
  confirmed c's scope after the what-if). The parked idea "culture as the condition" stays parked: c only describes.

## 3. Where things stand

- Facts from the grill note: all 29,232 have the default image and captions and the three instructed images
  (`_s2img`); emotion and style labels on all; **genre coverage not checked** (the WikiArt genre CSVs are outside the
  job's read roots: check it before quoting a count). The claim test's stage 2 used them to train D2 (train) and to
  stop it early (validation); no choice between variants or recipes read them.
- ArtELingo-28: only 100 of 746 paintings have a strict majority emotion, 730 a unique plurality; its "other" emotion
  has no class in ours; our "nude painting" genre is absent; the 27 styles match. Its default Qwen image differs from
  ours on the 248 shared paintings (mean cosine 0.934, minimum 0.830; cause not checked). No instructed store exists.
- Backbones: SigLIP 2 so400m covers all train-side and held paintings, images and captions (74.7 images/s);
  CLIP B/32 has only the legacy caption-indexed store (a new `artelingo_full/clip_b32` store needs an allow-list line;
  about 106 images/s); PE-Core is not wired in `src/data/feature_extract.py`; DINOv2-large has 49,121 non-held images,
  no captions. Qwen default runs at 11.5 images/s.
- Not yet scheduled: nothing runs. The local GPU was free at the last check (2026-10-21).

## 4. Open points for the next step

Round 2, key (answer first, one line of why, then our recommendation, then a what-if):

a. **The recipe rule's form** (Q2 follow-up). Candidates to put to the user: replace D2 when a variant is above D2 at
   the matched teacher share on at least one aspect with its interval above 0 and on no aspect below it by more than
   one seed range; or a ranked rule (the best mean over the three aspects, ties to D2); or per-aspect recipes (a
   recipe per head). Say what each leads to for the paper and for part 2.
b. **Which backbones** for the student's input: SigLIP 2 (stores exist), CLIP B/32 (a new store, about 10 minutes),
   PE-Core (encoder code first), DINOv2-large (images only). Cost and claim per option.
c. **The comparator for "any backbone"**: PercepT core with that backbone's image in its Stage 2 (matched input;
   its Stage 1 still needs Qwen's instructed captions, f2 §4), the backbone's own frozen image, or PercepT core on
   Qwen as now. Held or development data for that test (`held-data.md`: held is fine with no selection).

Minor batch (with recommendations; they end in the spec's "Check these"):

d. Part 1's design: a 2×2 of caption terms (on, off) × teacher share (1/5, 1/3), for both caption-free variants;
   5 seeds; a fixed-epoch genre export beside early stopping (the no-own-caption-terms genre stopped at epoch 1 or 2);
   the ceiling probe trains on the fit paintings and scores on both development samples; scoring cost on 29,232
   estimated before the run.
e. ArtELingo-28: which images for the 248 shared paintings (its own files or ours); "other" with "something else"
   or dropped; the plurality rule and ties; instructed images on the 746 for the instructed bar (about 3.5 minutes).
f. Device and timing: part 1 under 1 GPU hour plus 1 to 3 CPU hours; part 2's extractions; the 11:00 end and the
   weekday 10:00 to 19:00 rule (`shared-resources.md`).

Then: ask once about the html-plan page (grill.md item 7), the closing summary, the Idea (B) line in the direction
log, the spec (`spec-templates.md`), and the restart question before the build chat (`loop-chats.md`, "How to cut").

## 5. Code and pitfalls

- Change no code in `src/test/20261021_d2_ablation/`: `ab_row.json` binds a code hash. A new round folder and module
  prefix for part 1.
- Joins are positional (constitution C6); ArtELingo-28's store uses `row_id`, ours `sample_id` (f2 §5).
- Every D2 path asserts 2048-wide inputs (f2 §5): part 2 must open the first layer to other widths while the teacher
  and outputs stay 2048.
- Report sequence names: 12-23 is the attention maps report; the next is 12-24.
- The local viewer of the attention side chat runs at http://localhost:8765/ (pid 3903, not this chat's).

## 6. State at handoff

- Running: nothing of this chat. The viewer above belongs to the side chat.
- Uncommitted: nothing after this commit (`.scratch/` is gitignored).
- On disk over 1 GB: nothing new.
