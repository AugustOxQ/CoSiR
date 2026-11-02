# Handoff: paper, first stage: check the related-work and method drafts (facts, citations), humanizer pass, then the user reads

## Brief
- Done: first drafts of related work (about 3,800 words, 127 checked citations) and method (about 4,300 words, 17 equations, 7 tables, 2 figures); three checked source lists; refs.bib at 197 entries.
- State: drafts not yet checked; the method found that each aspect keeps its own trunk at test, so c* is not one trunk read once.
- Next: fact check of the method against the code, citation check of the related work, bib cleanup, humanizer pass; then the user reads.
- Blocked: nothing.

> Written 2026-11-02 18:58 by the paper chat (first stage). Outside the research loop: a paper chat beside r15 decide,
> without ARS (user's choice for these two drafts). Direction log: not touched (the paper is not a loop step).

## 1. Read in this order

1. The paper folder's `README.md` (`docs/paper/cvpr2027_aspect_spaces/`, **PF**; its own private git repo): the user's
   decisions for this paper.
2. `PF/notes/drafting_brief_v1.md`: the common brief and the outline of both sections (names table, claims,
   citation rules, style).
3. The drafts: `PF/drafts/v1/02_related_work.md` and `PF/drafts/v1/03_method.md` (each ends with a `TRACE` block);
   `PF/drafts/v1/paper_v1.md` is the joined file (`python3 PF/tools/join_draft.py PF/drafts/v1`; never edit it by hand).
4. Only as the checks need them: `PF/notes/facts_brief.md` (the method as coded, with sources),
   `PF/notes/sources_{a,b,c}_*.md` (the checked source lists; the only sources the related work may use), the figures
   `PF/figures/fig{1,2}_*.png` with `captions.md` and `src/`.

Skills: `humanizer:humanizer` (the pass over both sections, last step before the user reads).

## 2. Decided by the user, not to be reopened

All in `PF/README.md`, "Decisions" (2026-11-02): method-first framing; one general method for any frozen encoder plus
a photo subsection; avoid "label-free", state that the instructions name the aspects; Markdown with pandoc keys now,
LaTeX later; our own section writers, not ARS, for these drafts; long and detailed now, trimmed later; every citation
true, where deep-research notes and the archived verified bibliography count as checked and everything else is checked
before it enters `refs.bib`; field-standard names, approved when the user reads.

## 3. Where things stand

- **Method draft** (Opus medium writer). It settled at the code:
  - each head saves both towers' trunks at its own best epoch, and scoring loads one file per aspect; the saved image
    trunks differ by 17% to 444% of the trunk norm between aspects (the writer measured the weight files);
  - early stopping uses the training loss on the validation split's own graphs (no labels);
  - for the other encoders only the image trunk's input width changes (1152 or 768); the caption tower still reads
    Qwen captions, and the output stays 2048 to match the teacher;
  - the photo-adapted model fine-tunes both towers' trunks and emotion heads, and all five runs stopped at best epoch 1.
  New facts no report states: the style training graph leaves 16,123 of 42,969 paintings without an edge (method
  Table 3); the trunk differences; the parameter counts. The facts brief's emotion figure 0.137 against 0.118 belongs
  to the fixed-weight graph; the draft uses the per-node figures 0.132 against 0.120.
- **Figures** (Opus high, look-and-fix loop; the controller viewed both): Figure 1, overview of training and test;
  Figure 2, the content-corrected mutual-kNN graph with a toy example. Known simplification: the toy takes content
  neighbours as the top k by content similarity, while the code uses the union of the mutual-kNN graphs on the plain
  caption and plain image. The main instance's frozen encoder is Qwen3-VL-Embedding-2B under the generic instruction,
  so "no aspect instructions at test" is the accurate wording, not "no Qwen at test".
- **Related work** (Opus medium writer): 2.1 to 2.5 as outlined, 127 distinct keys, all in refs.bib. It left out
  SEED and EmoVLM-KD because list a marks them excluded; both count as checked (list b: SEED from the 12-26
  deep-research note; list c: EmoVLM-KD verified on arXiv), so they may be cited. Its PercepT Stage 2 sentence rests
  on the 12-05 deep-research note (facts brief §5).
- **Sources:** three deep-research lists (Opus high for aspect and instructed embeddings plus the closest prior;
  Sonnet high for distillation and graphs; Sonnet high for affect and art), copied verbatim into the literature note
  `docs/reports/literature/2027-01-03_paper_related_work_sources.md` (indexed). Closest prior: no work does any of our
  three points in our form; novelty wording in list a, part C.
- **Commits:** paper repo `3d0d0a8` (pushed); main repo `10ffa6c` on `r14-backbones-sweep` (pushed; the branch is
  shared with the r15 decide chat).

## 4. Open points for the next step

- **Key, user (asked in the paper chat at 18:55; carry the answer if given there):**
  a. Per-aspect trunks at test: state it as it is (the draft does), and optionally a short side job of one trunk for
     all heads stopped on the summed validation loss and re-scored (cost not checked). Our recommendation: state it
     now; the side job if the r15 chat leaves GPU time before 11-10.
  b. The supervision sentence: settings were picked on train-side development paintings (paintings) and on the
     validation split (photos). Our recommendation: "trained without aspect labels; labels are used only to select
     settings on held-out development data and to evaluate". Then update the README decision and the drafting brief.
- **The checks** (routing: text Claude wrote goes to Codex when `codex-usage --room fidelity` says room; else Claude
  Sonnet high):
  1. Method fidelity: re-derive every number in the method's TRACE block from its source, and check each method
     statement against the code functions the facts brief names (graphs, losses, masks, temperatures, stopping,
     read-out, photo model, other encoders).
  2. Related-work citation check: each cited sentence against its source-list entry (no claim beyond the entry, no
     excluded work, the caveats kept: no "frozen" for PercepT's CLIP, no conflicting dataset counts, MMArt in the
     third person).
  3. Bib cleanup: two TODO citations (Sculley 2010 mini-batch k-means in the method; Warner 2024 ModernBERT in the
     related work) need a check before they enter; incomplete entries (`yang2024clipkd` and `hao2021wnn` with
     "others", `tian2020crd` without a year, `park2019rkd` pages differ between lists: 3967-3976 against Crossref
     3962-3971, `passalis2018pkt` venue from a listing).
  4. Fix wave, then the humanizer pass over both sections; re-join `paper_v1.md`; commit and push the paper repo.
- **Minor:** the method's name stays `{Method}` until the user picks one; Figure 2 is 3.37 in tall and the Figure 1
  legend spacing is uneven (polish later); a final arXiv sweep for concurrent work the week before submission.

## 5. Code and pitfalls

- The paper repo is `docs/paper/` (private, its own git): commit paper work there. The main repo is public: never
  name it, the project, or a path in paper text.
- The main repo's working tree is shared with the r15 decide chat: commit only your own files, with explicit paths.
- Deep-research and other long hand-backs arrive as a `SubagentHandback` call; `PF/tools/handback.py <transcript>
  <out.md>` appends one to a file without passing it through the chat, and `PF/tools/merge_bib.py <sources.md>
  <refs.bib> "<section title>"` merges a list's BibTeX (last occurrence wins, existing keys skipped).
- Key aliases in list a: use `koohpayegani2020compress`, `tian2020crd`, `furlanello2018born`.
- MMArt (`wang2026mmart`, arXiv 2608.10706) lists the user as an author: cite it in the third person only.
- Report dates in `docs/reports/` are sequence numbers, not calendar dates.

## 6. State at handoff

- Running: nothing of this chat's.
- Uncommitted: nothing of this chat's (the untracked files in the main repo belong to other chats).
- On disk over 1 GB: nothing new.
