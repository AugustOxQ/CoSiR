# Handoff: paper plan 18: point 8 (Affection examples) from the Codex search, then the page count (Figures 4 and 5 into 8 pages)

## Brief
- Done: all Sec. 4 table captions (Tables 3 to 7) and the overview, Fig. 4 and Fig. 5 captions cut, settings moved into the text; overview at 90% width; co-author point 4 (related-work opening).
- State: REPO `9c832d8` pushed; 11 pages; the text ends at the top of page 8's right column, Figures 4 and 5 on page 9.
- Next: show the user the point 8 search result, then the page count.
- Blocked: nothing; the user pulls REPO into Overleaf.

> Written 2026-11-16 03:36 by the "paper plan 17" tab (a paper chat, outside the loop). Cut for context size (past 200k
> tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. MAIN `src/test/20261116_affection_examples/findings.md`: the Codex search for point 8 (section 4 below says what
   was found and what the controller checked).
3. PF `notes/2026-11-15_coauthor_comments.md`: the nine points; open now: 6 (the overview's remaining issues, "we
   discuss later") and 8.
4. PF `README.md`, the rows dated 2026-11-16 from "Table 3 caption cut" on (all decisions of this chat).
5. Project memory `feedback_user-writes-paper.md` (how a part is written; resolved co-author notes are commented out).

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, each a README row in PF:
- The captions rule: what the text says, or can say, leaves the caption; settings move into the text first. Applied
  to Tables 3 to 7 and to the overview, Fig. 4 and Fig. 5 captions (now 56, 17, 27, 23, 26, 36, 37 and 36 words).
- Moved into the text: Sec. 4.1 (the instructed caption embedding per aspect instruction; MMArt's 20 style classes;
  the MMArt read-out pick; "on the validation split of each dataset"; "unless stated otherwise"), Sec. 4.3 (the three
  methods not run on MMArt), Sec. 4.5 (no retuning and plain cosine in the ablation; the Qwen3 graphs, instructed
  embeddings and PercepT topics on every encoder).
- Rows "Plain embedding" renamed "Plain SigLIP" in Tables 4 and 6 (as in Table 2). Table 6 keeps one caption sentence
  ("Each row of the middle block removes one part ..."), since the bare row names read ambiguously.
- Overview at 90% width (85% was "a bit too small").
- Related work: its two TODO notes commented out (REPO `9c832d8`). It opens: "We focus on conditional similarity in datasets and models, and on neighbour graphs as
  supervision." (point 4).
- Point 8: a subagent searched in parallel, "though we might not have space for it".

## 3. Where things stand

- REPO `9c832d8` = origin. Builds clean, 11 pages; log has only the old 0.7 pt overfull box (page 3 is no longer
  floats only).
- Length: the Conclusion heading is at y 181 pt at the top of page 8's right column (was y 445 before this chat's
  caption cuts); the rest of that column is empty. Figures 4 and 5 (figure*, full width) fill page 9 from the top to
  about y 365 pt. The abstract and conclusion are still empty. Rough estimate, not measured: Figures 4 and 5 need about
  300 pt of full width, so with the abstract and conclusion the paper is still about half a column over 8 pages.
- The overview at 90%: page 3 gains 17 pt, but the end of the text did not move (the float layout absorbs it for now).
- Paper deadline 2026-11-17 12:59 Amsterdam.

## 4. Open points for the next step

Key, first: **point 8, the Codex search.** Codex job (gpt-6.1-terra, medium, started 03:31) with brief
`/tmp/claude-0/-project-CoSiR/03bfe35f-d3a3-4d0c-85a8-e2c44c348a1e/scratchpad/codex_p8_task.txt` (the scratchpad may
be gone; findings.md restates the task). Step 1: find the photo model's (SigLIP, fine-tuned on Affection) emotion
embeddings on the Affection test and the plain SigLIP features. Step 2, only if both exist: a contact sheet of 12
candidate queries (panel (a)'s rule: CoSiR 4 or 5 of 5 neighbours share the label, plain 0 or 1), and a CSV with each
photo's source for a licence check. No GPU. Result (job done 03:35, controller checked the export sidecar, `results.json` and the sheet):
- Both stores exist. Photo model: `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/exports/affection/photo_siglip1/test__s{0..4}.npz`
  (8,501 x 2,048, checkpoint `.../outputs/photo/siglip1/s<seed>/emotion.pt`, fine-tuned from the painting c*). Plain:
  `/data/SSD2/pre_extract/affection_local/siglip_so400m/` (85,007 x 1,152, all splits, joined by key). IDs match.
- 1,917 test photos have a tied label and are skipped; of 6,584, 186 pass the rule; the sheet shows 12 (at most two
  per emotion), seed 0 only. Sheet: `src/test/20261116_affection_examples/contact_sheet.png` (not in git, third-party
  photos in a public repo); CSV with each photo's source dataset and file name (COCO, Flickr30k, Visual Genome).
- Controller's view of the sheet: candidates 3 (pets, amusement), 5 (places, awe) and 6 (food, disgust) show no
  people. Many others show identifiable people, and one plain neighbour of candidate 7 shows an offensive slogan.
  Licences not checked: COCO images carry per-image Flickr licences in the COCO annotations, Flickr30k images are
  under Flickr's terms for research use.
The user has not seen the sheet yet. Send it, then they decide whether an Affection panel goes in at all (space), and
which candidate.

Then: **the page count.** Figures 4 and 5 must fit within 8 pages. Options to weigh with the user: the abstract and
conclusion lengths, Fig. 5 or Fig. 4(b) to the supplementary, smaller figures, text cuts.

Minor, carried: the abstract's three questions; the teaser caption; a limitations paragraph; the LLM Usage Statement
(the user writes it, agent lists our uses); the CVPR 2027 LLM policy re-read; the title; whether Fig. 4(a) stays; the
old teaser to the supplementary; "label-free" with the co-authors; the overview's remaining issues ("we discuss
later"); the checks before upload. The status table (PF `user_read/2026-11-13_paper_status.md`) was last updated after
point 5: bring it up to date at the next quiet point.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Before each REPO edit: `git pull --ff-only`. Edit with an assert script, build with
  `latexmk -pdf -interaction=nonstopmode -silent main.tex`, check `main.log` for "undefined" and "contains only
  floats", find positions with `pdftotext -bbox`, render with `pdftoppm -r 150`, crop, send with SendUserFile. Commit
  and push REPO after the user's OK, then add the README row in PF and push PF.
- A `\TODO` or note at the end of a caption: comment it out on its own line and keep the caption's closing brace.
  Resolved co-author notes: `% \nvn{...}` on its own line after the resolving sentence, never deleted.
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant". One name for one thing (match paragraph headings and row names).
- The handoffs live in MAIN, a public repo: no names of co-authors there; the PF notes are private.
- Photos in a figure: the paintings figures say "public-domain"; Affection photos need a licence check per photo
  before any goes into the paper.

## 6. State at handoff

- Running: nothing. The Codex job has ended (ledger row in `claude-config`, `cadf527`).
- Uncommitted: nothing of ours in REPO or PF. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
  MAIN `src/test/20261116_affection_examples/`: the script, findings and log are committed; the CSV and results.json
  are gitignored;
  `contact_sheet.png` and `preview.png` (4.6 MB, third-party photos) stay untracked on purpose.
- On disk over 1 GB: nothing. The scratchpad (page renders, a few MB) goes with the session.
- Repo heads: REPO `9c832d8` (= origin), PF `424b11b`, MAIN (after this cut's commit).
