# Handoff: relaunch the VLM2Vec wait, then build the 4.6 qualitative figure (candidates, review page, user picks)

## Brief
- Done: Tables 6 (other encoders), 7 (photos, with a paintings-only row) and 8 (speed) are in the LaTeX, with underlines. The plan for the qualitative figure is decided.
- State: Table 3's VLM2Vec and PercepT (CLIP L/14) rows are still empty. r25's validation store had 2 of 3 aspects at 09:13.
- Next: relaunch the VLM2Vec wait script, tell r26 this tab's name, then design the candidate rules for the figure.
- Blocked: the user's 11:00 meeting. Ask nothing about it until the user says it is over.

> Written 2026-11-11 09:15 by the "paper sec4 tables 4" tab, cut at about 230k tokens. Outside the research loop: no
> loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/`; T3 means `/project/CoSiR/src/test/20261111_table3_rows/`; EPT means
> `/project/CoSiR/src/test/20261111_encoder_photo_tables/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the last five rows (2026-11-11): Tables 6 to 8 and the qualitative plan.
3. `src/test/20261109_r20_figures/20261109_r20_figures_log.md`: how r20 built the teaser and the cluster figure
   (candidate rules, image-rights rule, review page, who picked what, the figure scripts in `PF/figures/src/`).
4. `2026-11-11-paper-sec4-tables-4-handoff.md`: sections 2 and 5, and section 4 items 2 to 4 (the Table 3 rows and
   the 4.1 PercepT note). They stand unchanged.
5. `EPT/20261111_encoder_photo_tables_log.md`, only when Tables 6 to 8 change.
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md`.

Skills: none. Before you show the user the review page (an HTML page), read `~/.claude/references/container-ops.md`.

## 2. Decided by the user, not to be reopened

All of section 2 of the earlier handoffs stands. New, all 2026-11-11, recorded in the PF README:

- **Tables 6 and 7 now, after Table 5 in 4.5, with no prose** (user). Their final place (own subsections, 4.5 or the
  supplementary) is for the user's meeting. Rows and layout are agent defaults.
- **Bold** is the best per column over the table (user, after seeing the recommendation). **Underline** is the best
  per column on each backbone, and in Table 7 the teacher counts as Qwen3 (user).
- **Table 7 has a paintings-only row**: c* before the photo fine-tuning, Qwen3 only (user, after seeing the
  recommendation).
- **Table 8 (speed) sits under Table 7** so the co-authors see it; it goes to the supplementary later (user).
- **Qualitative figure, 4.6** (user, after seeing the recommendation):
  - one full-width figure with two panels: (a) same subject, different feeling: plain-embedding neighbours share the
    query's subject, our emotion neighbours share its emotion across subjects; (b) close in one aspect, far in
    another: pairs that are near in one of our spaces and far in another;
  - the cluster figure rebuilt at the class count (9, 27, 10), in place of the 20-cluster version;
  - text-to-image queries only if there is room;
  - a photo example and a failure case go to the supplementary;
  - built as in r20: candidates by a stated rule, a review page, the user picks, then the final figure.
- **The speed question** (user, 09:13): the speed-up is against the instructed Qwen3 teacher only. The answer given:
  the fair same-encoder number is the Qwen3 row (3.0, one pass against three). The larger ratios on smaller encoders
  mostly reflect encoder size. A teacher that encodes the image once was not measured (r19's arithmetic: about 203
  ms, so about 2.3 instead of 3.0). Whether the instruction can share the image's cached tokens in Qwen3-VL-Embedding
  was not checked. Any later caption or sentence on speed must say what the teacher's time includes.

## 3. Where things stand

- REPO (pushed): `edd0b57` Tables 6 and 7, `7486b3e` underlines, `76a7feb` paintings-only row and Table 8. PF
  (pushed): `5c79d7e` and `54bce7d` trace, `16ba620` README rows. Main repo (pushed): `609a6243`, `8e102c84`,
  `c0ad1aab` (EPT generator, frozen-row scorer, records).
- Table 6 (ArtELingo test): ours beats the plain embedding and PercepT on every encoder in every cell. Against the
  teacher, every cell of ours is at or above it except CLIP L/14 emotion mAP (0.267 against 0.268).
- Table 7: the photo model beats the teacher everywhere (Qwen3 Affection AMI 0.174 against 0.155, EmoSet 0.440
  against 0.394). The paintings-only model is above plain but below the teacher (Affection 0.123 against 0.088 and
  0.155; EmoSet 0.385 against 0.349 and 0.394).
- Table 8: ours 88.6 ms (Qwen3) to 5.2 ms (CLIP B/32) per image against 266.8 ms for the teacher.
- Table 3's VLM2Vec row: this tab's wait script was stopped at 09:15 before it scored anything. At 09:13 the
  validation store `/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val/` had emotion and style; r25's GPU job
  `r25:extract_val` was over its estimate (started 07:58).
- PercepT (CLIP L/14) rows of Tables 2 and 3: r26's full run may wait until 19:00 (weekday power rule, r26 run
  handoff). r26's handoff tells it to send its files to the highest "CoSiR paper sec4 tables N" tab.

## 4. Open points for the next step

Key:
1. **First, message the r26 tab** ("CoSiR r26 PercepT original") with this tab's name (`ListAgents` prints it). Check
   `ReadNotifications` too.
2. **VLM2Vec row of Table 3:** launch `T3/t3_wait_vlm2vec.sh` with Bash `run_in_background`, then follow item 2 of
   the previous handoff's section 4 (fill, bold, compile, look, commit, push, trace, send the PDF).
3. **PercepT (CLIP L/14) rows** and the 4.1 note: items 3 and 4 of the previous handoff's section 4.
4. **Qualitative figure:** in chat, propose the candidate rules for panels (a) and (b) and for the class-count
   cluster figure, and get the user's yes. Then a worker builds the candidates and a review page from r20's code
   (`src/test/20261109_r20_figures/`: `r20_cand.py`, `r20_figs.py`, `r20_view.py`; outputs
   `/data/SSD2/cosir_r20/20261109_r20_figures/`) in a new dated round folder. The user picks. Then the look work, an
   Opus `high` worker per `agent-routing.md`, draws the figure in `PF/figures/` and the controller copies it into
   REPO. Ideas for the rules (agent's, not yet shown): (a) queries whose plain-embedding neighbours agree with the
   query's emotion label least and whose emotion-head neighbours agree most; (b) pairs ranked high in one head and
   low in another, from the same seed and k-means state as r20.

Minor:
5. Placement of Tables 6 to 8: after the meeting, when the user raises it.
6. Not asked for and not done: frozen c* on photos for SigLIP 2, SigLIP 1 and CLIP L/14 (needs inference of r14's
   checkpoints on Affection, then a read-out pick); timing a teacher that encodes the image once (r19's parked option,
   about 1 h of local GPU).

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand (two chats share REPO and PF; check `git status`; commit only
  your own paths with `git commit -- <paths>`; push at once; no notes of ours in REPO; paper names only; no
  semicolons between clauses; "test set", never "held-out").
- **Tables 6 to 8 are generated**: `EPT/ept_make.py <REPO>/Tables` writes `encoders.tex`, `photos.tex`, `cost.tex`
  and `EPT/records/cells.json`. Edit the generator, never the `.tex` files, since a rerun overwrites them. Captions
  are raw strings: one backslash before `method`, or LaTeX prints a line break.
- `ept_make.py` reads `EPT/records/frozen_cstar.json` (gitignored type, added with `git add -f`) and asserts its gate.
- Image rights for any painting in a figure: public domain only (artist died 1955 or earlier, Wikidata), r20's rule.
- r20's figures used c* seed 0 and k-means state 0 of r17's `ns_metrics.partitions`. Keep the same for the rebuild.

## 6. State at handoff

- Running: nothing of this tab (the VLM2Vec wait was stopped before scoring). Others: `r25:extract_val` (GPU), r21's
  jobs and `r26:extract_patch` queued. Other tabs: "paper experiments 2", r25 report, r26, r21.
- Uncommitted: in REPO, `Sections/2_related_work.tex` and `main.tex` (not ours). In PF, `CBMI_IMP_CAMREADY (2).pdf`
  (the user's). In the main repo, files not ours. In EPT, `logs/` (gitignored, small).
- On disk over 1 GB: nothing. The session scratchpad holds 364 KB (page renders and a check script).
