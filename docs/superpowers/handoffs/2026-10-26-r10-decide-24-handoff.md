# Handoff: r10 decide 24: the grill of the genre loop (the held bar, the selection rule, the anchors arm), then the spec and the decision rule; context cut

## Brief
- Done: decisions A and B are made and recorded: one full loop that sweeps the relational genre term (weights 16 and 32; relational alone and a sharper teacher temperature at weight 8; teacher anchors), confirms on fresh seeds and ends with a held genre claim test of the selected variant. The literature check of the term is saved (12-26). The stage report is approved by the user and its page is deployed.
- State: held genre, full scale, 9,949 paintings: the current recipe about 0.4535 against the instructed Qwen genre image 0.5062 (PercepT core's best about 0.4270); on development the weight-8 term closes the gap (0.5022 against the instructed 0.4989).
- Next: the grill (step 5): the held genre bar, the selection rule, the anchors arm's role and the ladder's stop; then the spec with the decision rule; then the cut to the build chat.
- Blocked: nothing.

> Written 2026-10-26 18:55 by "r10 decide 23". Loop step reached: after decision B, before step 5 (the grill). Direction
> log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md) (its "Now" holds decisions A and B).

## 1. Read in this order

1. This handoff.
2. The direction log, `docs/superpowers/direction_log.md`: "Now" and the 2026-10-26 entry (the user's reading of the
   genre result, decisions A and B).
3. The literature note, `docs/reports/literature/2026-12-26_relational_genre_term.md`: Summary, Q2, Open and the
   controller's note (the term's exact form).
4. The genre round: `docs/reports/auto/percept/2026-12-25_encoders_genre.md` lines 96 to 130 (stage iii table, fresh
   seeds) and 208 to 225 (diagnostics); its spec `docs/superpowers/specs/2026-10-25-encoders-and-genre-design.md`,
   §3 stage iii (lines 98 to 147), §6 (250 to 275), §4 rows D25 to D27, D34, D35; its rule
   `src/test/20261025_encoders_genre/DECISION_RULE.md` §2.3, §3.3, §4.3 (the structure to reuse for the new rule).
5. The held numbers: `docs/reports/auto/percept/2026-12-19_claim_test_stage2.md` lines 160 to 175 (first shared model:
   genre 0.4438, PercepT core 0.4230, instructed Qwen genre image 0.5062, frozen default image 0.4521; 9,949
   paintings) and `docs/reports/auto/percept/2026-12-24_recipe_backbones.md` lines 168 to 180 (the current recipe's
   held margins: genre +0.0265 over PercepT's maximum, +0.0014 over the frozen input, −0.0527 under the instructed
   image).
6. `~/.claude/references/grill.md` (protocol; it overrides the skills), `~/.claude/rules/spec-templates.md`,
   `~/.claude/templates/spec_core.md` and `spec_experiment.md` (a claim test stage: the rule reviewed by an
   independent agent on the most capable model, then committed before any code).
7. The code, when a question needs it: `src/test/20261025_encoders_genre/eg_train3.py` lines 90 to 120
   (`switched_head_loss`: the genre switches m, lam, w), `src/test/20261019_model_sheet/ms_train.py` lines 100 to 115
   (`distill_loss`: direct and relational, temperature 0.05), `src/test/20261025_encoders_genre/eg_variants.py`
   (ladders, fresh seeds, the target 0.4989162, the seed ranges), `src/test/20261025_recipe_backbones/rb_rows.py`
   (the recipes; `R6-5` is the current one).

Skills: `grilling` and `domain-modeling` at step 5.

## 2. Decided by the user, not to be reopened

- **A (2026-10-26, decide 23):** option 2 then 1, "close the genre gap first" (user, own pick). As one full loop with
  three stages: the sweep, the fresh-seed confirmation, the held genre claim test (user, after seeing the
  recommendation: flag it in the spec's brief).
- **B (2026-10-26, decide 23):** all three candidates go into the sweep (user, own pick: "cheap; I don't really know
  what the sweep affects"): 1, the ladder up (weights 16 and 32 on seed 0; weights 2 and 4 on fresh seeds 1 to 3);
  2, relational alone (the direct term removed on the genre head) and a sharper teacher temperature, both at weight 8;
  3, teacher anchors in place of the in-batch rows. Our view, given after the pick: 1 and 2 may be selected for the
  held test; whether 3, a new mechanism tested on one night, may be selected is a grill question (K3 below).
- The stage report, its page and its post stand as they are (user: "Everything on the stage report would be fine");
  the page is live on the dev-log site (deploys 16:03 and 18:07, by the design Q&A chats on the user's word). No repo
  name, link, account or author name on the page or in the post.
- Aspect-named retrieval for the 11-10 gate waits (decide 20); the write-up of the encoder claims comes after this
  loop. The CVPR abstract is due 2026-11-10.
- Agent default, not objected to: the two standing wrap-up items (a disk-ledger row with a brace-glob path the checker
  cannot resolve; the routing item of 2026-10-12) are handled at this loop's wrap-up.

## 3. Where things stand

**Held genre, full scale, 9,949 of the 12,281 held paintings, five seeds** (the first shared model's row from 12-19;
the recipe's values from 12-24's margins):

| Entry | Genre AMI |
|---|---:|
| Instructed Qwen genre image | 0.5062 |
| Current recipe (no own-caption terms, teacher 0.25) | about 0.4535 (−0.0527 under the instructed image) |
| Frozen default Qwen image | 0.4521 |
| First shared model | 0.4438 |
| PercepT core's best reading | 0.4230 (12-19); about 0.4270 as 12-24's "PercepT maximum" |

**Development, Qwen, 29,232 train-side paintings (genre on 23,682), stage iii of 12-25** (models trained on 23,080 of
them, stopped on 6,152; seed 0 unless said):

| Variant | Genre AMI | Emotion effect | Style effect | Guarded |
|---|---:|---:|---:|---|
| Current recipe (reference) | 0.4674 | | | |
| Instructed Qwen genre image (target) | 0.4989 | | | |
| Relational, weight 0.5 | 0.4935 | −0.0064 | −0.0034 | no (emotion) |
| Relational, weight 2 | 0.4948 | +0.0001 | −0.0016 | yes |
| Relational, weight 4 | 0.4938 | −0.0004 | −0.0002 | yes |
| Relational, weight 8 | 0.5018 | −0.0032 | −0.0052 | yes |
| Teacher weight x2 / x4 | 0.4594 / 0.4478 | | | yes (genre falls) |
| Regression term, weight 1 | 0.4675 (its adaptor 0.4985) | −0.0045 | −0.0093 | no (style) |
| Weight 8 on fresh seeds 1 to 3 | 0.5022 against the recipe's 0.4694, +0.0328 [+0.0235, +0.0442]; emotion +0.0004, style −0.0010 | | | |

Guards: emotion effect ≥ −0.0046, style effect ≥ −0.0082 (one seed range each; genre's range 0.0128). Picked among
seven variants on one seed. **Note the gap sizes:** on development the recipe trails the instructed image by 0.0295
and weight 8 closes it; on held the gap is 0.0527, so a lift of the development size would leave the student about
0.02 below the instructed image on held. The held test also trains at full scale (42,969 paintings) while stage iii
trained on 23,080; whether the lift grows with scale is unknown.

**Costs (12-25 run log):** five variants 613 s; the two extra rungs 1,668 s with lock waits; fresh seeds (two
variants, three seeds) 138 s; scoring and re-derivations the larger part; the whole stage iii 23:00 to 00:34.
Full-scale training about 120 s per model seed (first shared model). The new loop's compute is light (under about
3 hours); the chats are the cost.

**The term as coded** (controller's note of the literature note): per painting, KL from the teacher's row to the
student's, each row a softmax over similarities to the other paintings of the batch (same painting excluded),
temperature 0.05 on both sides, as the direct term; mean over the batch; teacher rows from the instructed Qwen genre
image, student rows from the genre head's image output; added beside the direct term (weight 0.25) on the genre head
only.

**The literature (12-26):** the term is a similarity-distribution distillation (closest: CompRess and SEED with a
queue of anchors; CLIP-KD in-batch), not RKD; no published sweep shows a monotone gain with the relational weight
(CLIP-KD flat from 0.5 to 2, lower at 10); no head-to-head of relational alone against beside direct on clustering;
SEED's teacher temperature mattered most (top-1 57.9 at 0.01 against 56.5 at 0.05); no label-free genre number on
paintings exists, the instructed Qwen image stays the bar; the "level with the teacher" margin is ours to set before
the held read.

## 4. Open points for the next step

The grill (`grill.md`): context first, at most three key questions per round, answer-first on key questions, a
scenario check after each key answer, a closing summary; ask once whether the plan is wanted as an html-plan page.

- **K1, key: the held genre bar.** Options: (a) level with the instructed Qwen genre image within a margin X
  (non-inferiority: the student's genre AMI minus the instructed image's, one-sided lower bound above −X; X candidates:
  one genre seed range, 0.0128, or half the held gap, 0.026); (b) above PercepT core's best held genre by +0.0100 with
  the interval above zero (the emotion rule's form; the recipe already meets it as a reading, so it claims little
  new); (c) two levels: GO as (a), a partial outcome when the student closes at least half the held gap with the
  interval above zero. What each lets the paper say, and the risk of (a) alone, given the gap sizes above. Hold the
  recommendation until the user answers.
- **K2, key: the selection rule for the held variant.** Draft: among all arms on seed 0, the guarded variant (emotion
  ≥ −0.0046, style ≥ −0.0082 against the recipe) with the largest unrounded genre AMI; then fresh seeds 1 to 3 against
  the recipe (gain ≥ 0.0128 with the paired interval above zero, guards met on the fresh-seed means); fallback: weight
  8, already confirmed. Also: do weights 2 and 4 on fresh seeds compete for the held slot (only if they beat weight 8
  on the same fresh seeds), and does the ladder stop at 32 (a rung 64 only when 32 is best and guarded, as D26)?
- **K3, key: the anchors arm.** A new mechanism (new code, one night): may it be selected for the held test, or does
  it run as an exploration arm whose result is a reading for the next loop? The build risk and the selection
  disclosure (seven earlier variants plus about seven new ones on the same development paintings) bear on this.
- **Minor, one batch with recommendations:** the teacher temperature values (0.02 and 0.01 at weight 8, or one of
  them); the anchors' form (the teacher's k-means centroids, K = 200, fit on the training paintings' instructed genre
  image, temperature 0.05; or 4,096 fixed teacher features); relational alone at weight 8 only; the held protocol as
  12-24's stage i (42,969 train paintings, seeds 0 to 4, k-means at 20, the genre labels of 9,949 held paintings;
  ArtELingo-28 has emotion labels only, genre not checked); diagnostics that decide nothing (the loss scale of the
  direct and relational terms at weight 1, the literature note's suggestion; Leiden; the old 5,000 paintings); the
  regression gates (the new trainer with every switch off reproduces the recipe's seed 0 bit for bit; weight 8 seed 0
  reproduces 0.5018); deliverables (auto report, user-read report); the glossary entry "relational term"; timing
  (build tonight, the run on 10-27, light GPU work, so the weekday rule and the 11:00 rule are not touched).
- **After the grill:** the spec (`spec-templates.md`; its brief flags the one-loop shape as "after seeing the
  recommendation"), the decision rule (reviewed by an independent agent on the most capable model, then committed
  before any code), the user's approval of the brief, then the cut to the build chat.

## 5. Code and pitfalls

- A new round folder, `src/test/20261027_<name>/`, with a new module prefix (the folders go on `sys.path`; `eg_`,
  `rb_`, `ab_`, `ms_` are taken). The genre switches live in `eg_train3.py` (`GenreLoss`: m, lam, w); a teacher
  temperature and an anchors mode are new switches; relational alone means the direct term's weight 0 on the genre
  head (today `weight = distill_weight * m` with m > 0 asserted).
- The anchors arm: fixed teacher features A (centroids or a sample); student row softmax(z Aᵀ / T), teacher row
  softmax(t Aᵀ / T), KL teacher first. The student head's output and the teacher live in the same 2048-d space (the
  direct term already compares them), so no projection is needed.
- The held read: one read, after the decision rule is committed; no tuning or selection on held data
  (`~/.claude/rules/held-data.md`). The development P paintings and seeds 0 to 3 are development data from here on.
- `codex-usage --room <kind>` before any Codex job (`~/.claude/references/codex-jobs.md`). No GPU work in the decide
  chat; delete nothing. Check `.scratch/pending_deletions.md` (12-25's queue listed the GT-2, GT-4 and GA-1 weights).
- Before the cut to the build chat: ask about the container restart (`loop-chats.md`, "How to cut"); the build chat
  opens with `--unattended`.
- The design Q&A chats added two framework figures to the stage report and its page (commits 2adb228, 16a349f) and
  deployed the page; nothing there to redo. Their handoffs: `2026-10-26-r10-design-qa-handoff.md`, `-qa-2-handoff.md`.

## 6. State at handoff

- Running: nothing of this chat (its preview server on port 8767 is stopped; the stage report 3 chat's servers are
  gone); no subagent, Codex job, GPU lock or cluster job.
- Uncommitted: nothing in the project. The dev-log site repo is committed by the Q&A chats (deploy commits f6ea4f7,
  48c7437); check `git -C /project/cosir-devlog status` once.
- On disk over 1 GB: unchanged (the rounds' outputs on /data/SSD2, kept; the shared feature stores). Small: four PDFs
  (0.9 to 3.2 MB each) in this session's tool-results folder, gone with the session.
