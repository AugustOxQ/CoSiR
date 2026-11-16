# Handoff: paper plan 17: the Table 3 caption first (the user's cut, one gap), then the captions of Tables 4 to 7

## Brief
- Done: co-author point 5 (datasets paragraph); point 6 in part: overview redrawn left/right (170 pt, was 243), time table to the supplementary, Table 2 caption 107 to 47 words.
- State: REPO `a907b3f` pushed; 11 pages; main text ends on page 8, Figures 4 and 5 on page 9.
- Next: the user's Table 3 caption cut (agent view given, one gap), then Tables 4 to 7 by the same rule.
- Blocked: nothing; the user pulls REPO into Overleaf.

> Written 2026-11-16 02:51 by the "paper plan 16" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-15_coauthor_comments.md`: the nine points and their status (points 4, 6 and 8 open).
3. PF `README.md`, the rows dated 2026-11-16 (all decisions of this chat).
4. REPO `Tables/main_retrieval.tex` (Table 3), then the captions of `Tables/mmart.tex`, `photos.tex`, `ablation.tex`,
   `encoders.tex`, and `Sections/4_experiments.tex` (what the text already says).
5. Project memory `feedback_user-writes-paper.md` (how a part is written; resolved co-author notes are commented out).
6. `docs/superpowers/handoffs/2026-11-15-paper-plan-14-handoff.md`, section 5 (build, crops, style rules).

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, each a README row in PF:
- Point 5: the Datasets paragraph names the four datasets first, then how each is used, with the Affection and EmoSet
  splits. Sec. 4.4's repeated fine-tuning sentence and the photos caption's test sizes are gone.
- Resolved co-author notes are commented out in place (`% \nvn{...}` on its own line), never deleted. The three
  deleted ones are restored (REPO `449dde5`).
- Overview figure: training left, test time right; the three test outputs in one block with the line from its bottom.
  In the paper now ("still got some issues, but can put in paper now, we discuss later"). Script
  `PF/figures/src/fig1_overview_lr.py`, output in `PF/figures/drafts/`, copied to `REPO/Figures/fig1_overview.pdf`.
- The time-per-image table is in the supplementary ("Inference time"); Sec. 4.5 gives its two rows as text.
- Table 1: column widened so p_emo fits on two lines. Table TODOs on underlining commented out; underlines stay.
- Captions rule (user): what the text already says, or can say, leaves the caption; settings move into the text
  first. Done for Table 2 (TURTLE and BERTopic settings now in Sec. 4.1). "The same for the rest of tables' captions
  in the section."

## 3. Where things stand

- REPO `a907b3f` = origin. Builds clean, 11 pages (was 12). Log: "Text page 3 contains only floats" (page 3's left
  column holds Figure 3 in full, no blank column); the old 0.7 pt overfull box.
- Table numbers: photos 5, ablation 6, other encoders 7 (the time table left the main paper).
- Length (rough, from the layout): the text, with the empty Conclusion heading, ends on page 8 at y about 504 pt in
  the right column; Figures 4 and 5 (figure*, 124 pt and 86 pt plus captions) sit on page 9. With the abstract and
  conclusion still empty, about 1 to 1.2 columns are left to cut. Not measured.
- Paper deadline 2026-11-17 12:59 Amsterdam.

## 4. Open points for the next step

Key, first: **the Table 3 caption.** The user proposed keeping only "Retrieval with captions on the ArtELingo test
set, in the three directions that involve text (image-to-image retrieval is in Tab. 2). Each cell is the label mAP of
one aspect." plus "CoSiR uses a Qwen3 text pass if the input is text and a SigLIP image pass if the input is an
image. The best result per aspect and direction is in bold." and asked whether they miss something. Agent view (given
in chat, the user has not answered yet):
- Good cut. One gap: the instructed comparator embeds each caption under each of the three aspect instructions. Sec.
  3.1 defines the instructed caption embedding only for emotion, so no other place says it. Keep a short clause, such
  as "The instructed embedding encodes each caption once per aspect instruction."
- The parenthesis "(image-to-image retrieval is in Tab. 2)" repeats Sec. 4.2, which already says which table holds
  which direction; it can go under the user's rule.
- "Passes" is then defined nowhere; the column name and the values (1 or 3) read on their own. Mention it.

Then, Tables 4 to 7 by the same rule: list per caption what the text already says, propose the cut and any sentence
that must move into the text first, and let the user decide caption by caption ("I will go through these captions one
by one"). Then the overview caption (still the long one, with "left" and "right") and Fig. 4's caption.

Then, in the user's order: point 4 (the related-work opening sentence, the user writes it), point 8 (Affection
examples, only if the fine-tuned photo model's test embeddings exist). The overview's remaining issues: "we discuss
later".

Minor, carried: the abstract's three questions; the teaser caption; a limitations paragraph; the LLM Usage Statement
(the user writes it, agent lists our uses); the CVPR 2027 LLM policy re-read; the title; whether Fig. 4(a) stays; the
old teaser to the supplementary; "label-free" with the co-authors; the checks before upload. The status table
(PF `user_read/2026-11-13_paper_status.md`) was updated after point 5 only: bring it up to date at the next quiet
point.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Before each REPO edit: `git pull --ff-only`. Edit with an assert script, build with
  `latexmk -pdf -interaction=nonstopmode -silent main.tex`, check `main.log` for "undefined" and "contains only
  floats", render with `pdftoppm`, crop, send with SendUserFile. Commit and push REPO after the user's OK of each
  change, then add the README row in PF and push PF.
- A `\TODO` or note at the end of a caption: comment it out on its own line and keep the caption's closing brace
  (`underlined.%` newline `% \TODO{...}` newline `}`); a `%` in the middle of the line would comment out the brace.
- TURTLE: keep "with the default settings of its released code and without its hyperparameter selection" beside it
  (project memory r18-paper-rows); it is now in Sec. 4.1. Sec. 4.2 still says only "TURTLE, at its default settings".
- The supplementary is `Sections/X_supplementary.tex`, input from `main.tex` only when uncommented; its `\cref`s to
  the main paper work only in that joint build.
- The handoffs live in MAIN, a public repo: no names of co-authors there; the PF notes are private.

## 6. State at handoff

- Running: nothing of this chat. The figure worker has finished.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, two scratch copies of the paper, a few MB) goes with the
  session.
- Repo heads: REPO `a907b3f` (= origin), PF `b73ecd8`, MAIN (after this cut's commit). Memory change
  (`feedback_user-writes-paper.md`, the comment-out rule) is in `claude-config`, synced by `sync.sh`.
