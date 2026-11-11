# Handoff: wait for r26's PercepT (CLIP L/14) files (about 21:15), score Table 3's last row; meanwhile follow the user

## Brief
- Done: Table 3's VLM2Vec row (0.248 / 0.280 / 0.559, below CoSiR), Figures 3 and 4 in 4.6 (the user's picks), the user's own LaTeX edits committed.
- State: Table 3 has one empty row, PercepT on CLIP L/14. r26 sends its files at about 21:15.
- Next: tell the "CoSiR r26 run" tab this tab's name, then work on what the user asks. Score the PercepT row when r26's message arrives.
- Blocked: the user's 11:00 meeting. Ask nothing about it until the user says it is over.

> Written 2026-11-11 09:45 by the "paper sec4 tables 5" tab, cut at about 210k tokens. Outside the research loop: no
> loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/`; T3 means `/project/CoSiR/src/test/20261111_table3_rows/`; QF means
> `/project/CoSiR/src/test/20261111_qual_figure/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the last six rows (2026-11-11): the figure rules and picks, the user's title change, Table 8 to
   the supplementary, Figures 3 and 4 in 4.6.
3. `2026-11-11-paper-sec4-tables-4-handoff.md`, section 4 items 3 and 4 (the PercepT row and the 4.1 note) and section
   5. They stand unchanged, except the r26 tab's name (see section 4 here).
4. `T3/20261111_table3_rows_log.md`, only when you score the PercepT row.
5. `QF/20261111_qual_figure_log.md` and `PF/figures/captions.md` (last two sections), only when the figures change.
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the earlier handoffs stands. New, all 2026-11-11, recorded in the PF README and in
`QF/20261111_qual_figure_log.md`:

- **Figure rules** (agent's proposal, user yes): ArtELingo test set, c* seed 0 against the plain Qwen3 image only,
  public-domain paintings only, near copies skipped, the teaser painting left out. Panel (a) keeps queries with at
  least 4 of 5 emotion matches in our emotion head, at most 1 in plain, and at least 3 plain genre matches ("same
  subject" = same genre label, kept by the user). Panel (b): near in one head (top 5, same label), far in another
  (different label, farthest rank first). The cluster figure keeps r20's largest-gain rule at the class count.
- **Picks:** panel (a) a5, Metsu "The Sick Girl" (user, after seeing the recommendation). Panel (b) b5, b7 and b15 (b5
  user; b7 and b15 user, after seeing the recommendation). Cluster figure contentment, Ukiyo-e, still life (user).
- **Figures 3 and 4 are in 4.6 now, with short draft captions.** Captions and placement are decided later (user: "yes
  for now, we decide them later").
- **The user's own edits** (committed at the user's request, REPO `ba4161e`): title "CoSiR: Conditional Similarity
  Learning for Aspects", Table 8 (speed) commented out of 4.5 (it goes to the supplementary), a `\newpage` before the
  references.

## 3. Where things stand

- REPO (pushed): `33a9764` VLM2Vec row, `ba4161e` the user's edits, `6d586f3` Figures 3 and 4. PF (pushed) up to
  `edefc8a`. Main repo (pushed): `a0363770` (T3 record), `45f851b3` and `47f2e221` (QF code, log and picks).
- **Table 3, VLM2Vec** (image to image, VLM2Vec-V2, read-out centering + k = 10 picked on validation): 0.248 / 0.280 /
  0.559 against CoSiR 0.269 / 0.323 / 0.615. Bold unchanged. Record `T3/records/vlm2vec.json`.
- **Table 3, PercepT (CLIP L/14):** r26's full run starts at 19:00 (it draws about 260 W, so the weekday rule
  applies). r26 sends at about 21:15: (a) `OUT/test/k40/L1__s{seed}.npz` (keys `prob`, `names`, r17's held order),
  `OUT = /data/SSD2/cosir_r26/20261111_percept_original/outputs`; (b) the k40 seeds that did not collapse, with their
  count.
- **Figures:** Figure 3 (qualitative, 6.875 x 1.72 in) and Figure 4 (clusters at the class count, 6.875 x 1.2 in)
  float to page 7, after the references start. In (a), plain has 1 sad neighbour of 5, ours 4. In the cluster
  figure, ours against plain: contentment 73% / 59%, Ukiyo-e 94% / 17%, still life 91% / 36%.
- **The review page** (`QF/outputs/viewer/`, all 9 + 18 candidates and every cluster class) is not served now. To
  show it: `python3 -m http.server 8767 --bind 127.0.0.1 --directory /data/SSD2/cosir_paper/20261111_qual_figure/viewer`
  (`container-ops.md`).

## 4. Open points for the next step

Key:
1. **First, message the "CoSiR r26 run" tab** (the r26 tab's new name; "r26 PercepT original" is the old one) with
   this tab's name (`ListAgents` prints it), so that its 21:15 message comes here. Check `ReadNotifications` too.
2. **PercepT (CLIP L/14) row of Table 3**, when r26's message arrives: `t3_run.py percept-clip --seeds <kept, e.g.
   0,1,3> --test-dir <OUT>/test/k40 [--test-file-pattern 'L1__s{seed}.npz']` through `res-queue cpu --short` (the T3
   log says the test pattern is assumed the same as validation's; pass it when r26's name differs). Then fill, bold, compile, look, commit, push, trace,
   send the PDF, as for VLM2Vec (T3 log, last section).
3. **4.1 PercepT note:** the CLIP L/14 version is a mean over the seeds that did not collapse, with r26's count. Show
   the sentence to the user first.

Minor:
4. Figure captions and placement, Tables 6 to 8 placement, the MMArt TODO under 4.4: when the user raises them.
5. In Figure 3 (a), the same Goya portrait (fear) is first in both rows. Mention it if the user revisits the figure.
6. Supplementary items added today (`PF/notes/supplementary_items.md`): Table 8, a photo example and a failure case.

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand (two chats share REPO and PF; check `git status`; commit only
  your own paths with `git commit -- <paths>`; push at once; no notes of ours in REPO; paper names only; no
  semicolons between clauses; "test set", never "held-out" or "held").
- The user writes the paper (project memory `feedback_user-writes-paper`). The figure captions in REPO are drafts for
  the user to rewrite. The full draft captions and every printed value's source are in `PF/figures/captions.md`.
- In zsh, a variable that holds several paths does not split: list paths in the command itself.
- Figure scripts: `PF/figures/src/fig_qualitative.py` and `fig_clusters_classcount.py` assert the ids, the labels and
  the public-domain rule. After a rerun, copy the PDFs into `REPO/Figures/` again.
- Workers may overwrite a round log they did not start: tell them not to touch it.

## 6. State at handoff

- Running: nothing of this tab (the review page server is stopped). Others: r21 run, r26 run (GPU at 19:00).
- Uncommitted: in REPO, `Sections/1_introduction.tex` (not ours). In PF, `CBMI_IMP_CAMREADY (2).pdf` (the user's,
  left out of git). In the main repo, files not ours.
- On disk over 1 GB: nothing. QF outputs on `/data/SSD2/cosir_paper/20261111_qual_figure/` hold 7.5 MB (the smoke
  folder, 3.5 MB, can go when the figures are final). The session scratchpad holds a few MB of renders.
