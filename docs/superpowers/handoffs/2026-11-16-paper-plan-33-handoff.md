# Handoff: go on with Part A of the merged review list from A24, then put all approved fixes into the paper

## Brief
- Done: A6, A15, A19, A13, A10, A26 and A25 approved (PF `8f6ebb8`); with A1, A3, A5 they cost 0 lines, and A13 gained room.
- State: all approved rows built together: 11 pages, every float on its page, about 3.4 free lines on page 8 (was 1.4). Not yet in REPO (`a56c4c1`).
- Next: A24's small items (source lines below), then A4, A7, A8, A9, A11, A12, A14, A16 to A18, A20 to A23; then all rows into REPO.
- Blocked: the user's abstract, the Overleaf pull, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 21:05 by the "paper plan 32" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-16_fix_round.md`: every approved fix, with old and new text, source, who decided, built cost.
3. PF `notes/review/2026-11-16_panel_merged.md`: the Part A table (line 112 on), then each group when you reach it.
   Open a seat's report (`2026-11-16_panel_<seat>.md`) only for the exact text of a fix.
4. PF `notes/review/2026-11-16_user_review.md`: the user's own review. Open it only for the exact words of a U point.
5. The plan 32 handoff (`2026-11-16-paper-plan-32-handoff.md`), sections 2 and 5, and the plan 31 handoff, section 5.
6. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

Skills: none. The chat starts with `claude-paper`, so ARS is loaded if a further ARS step is needed.

## 2. Decided by the user, not to be reopened

All on 2026-11-16 evening, each "user, after seeing the recommendation", each a row in `fix_round.md` and in PF
`tools/apply_fixes.py`:
- **A6** (`4_experiments.tex:126-127`): "Trained on paintings only, \method has a lower AMI and mAP than the instructed
  Qwen3 embedding on both photo sets. On SigLIP, the fine-tuning raises its Affection AMI from 0.150 to 0.182."
- **A15** (`:56`): "For image retrieval, we pick one read-out per method, that is, the way it compares embeddings, by its
  mAP on the validation split of each dataset, and use it for all aspects."
- **A19** (`:163`): "On Qwen3, \method takes 88.6\,ms, so the single pass gives a factor of 3.0 and the smaller encoder a
  factor of 4.5." (Table S2's 88.6, not the review's 88.9.)
- **A13** (`1_introduction.tex:39-41`): item 1 scoped to "the frozen Qwen3-VL-Embedding model"; item 2 "on frozen
  encoders"; item 3 "Beyond the clustering test of PercepT [28], we test each aspect embedding on a test split, with
  style, with image and text retrieval, on MMArt [46], and for emotion on photos [3, 52]." (saves a line on page 2).
- **A10**: `4_experiments.tex:156` "There, its style and genre AMI stay within 0.01 of those of the instructed
  embedding, so the encoder also matters."; `:94` names "the SigLIP encoder (\cref{sec:ablation})" among the reasons.
- **A26** (Figure 2 caption, the user's text): "At test time (right), each aspect uses its own trunk and head, saved at
  its best epoch."
- **A25**: ModernBERT and GoEmotions cited in `4_experiments.tex:41` and `X_supplementary.tex:22` (references 58 to 60;
  most numbers shift, PercepT [28] to [29]; no figure has a baked-in reference number).
- Still in force (plan 32 handoff, section 2): A1, A3, A5; the way of working (per group: exact old and new text,
  options, a recommendation, the user picks, the row, a scratch build of all rows, commit and push in PF; REPO gets all
  rows at once after Part A); the abstract and the Figure 1 caption are the user's; the LLM Usage Statement is never
  read; Claude subagents only, never Codex.

## 3. Where things stand

- REPO `a56c4c1` = origin (fetched 2026-11-16 20:40, no new co-author commit then). PF `8f6ebb8` = origin. MAIN: this
  handoff.
- All rows built (scratch): 11 pages; floats p1 Fig. 1, p3 Figs. 2 and 3, p6 Tables 1 to 3, p7 Tables 4 to 6, p8 Fig. 4
  and Table 7; "future work." at ruler 563; about 3.4 free lines on page 8; no overfull box, no undefined reference.
- The supplementary (`supp.tex`, 3 pages) builds with 9 undefined-warning lines, the same with and without the fixes
  (not checked further; supplementary work, status row 15, due 2026-11-24).
- Room: whether page 5 can now take one added line (A13 freed a line on page 2) is not measured.

## 4. Open points for the next step

Key, in order:
1. **A24**, small items (merged list line 656). Source lines read at REPO `a56c4c1` with the approved rows:
   - Ducktor m9: `4_experiments.tex:152` "where the aspect graphs, the instructed embeddings and the topics of PercepT
     come from Qwen3": fix "instructed graphs" (its second use, `:156`, is gone with A10).
   - Ducktor m12: Table 6 rows "Instructed graphs", "Content correction", "Graph-positive loss", "Relational term" get a
     "w/o" prefix; caption "Ablation on the ArtELingo test set. Each row of the middle block removes one part of \method
     and gives the change." to "Ablation on the ArtELingo test set with plain cosine. Each middle row removes one part
     of \method and gives the change."
   - Ducktor m15(b): `2_related_work.tex:50` "The $k$-reciprocal neighbours re-rank" to "Reciprocal k-nearest neighbours
     re-rank". m15(e): `4_experiments.tex:52` "cf.~\cite{mohamed2026percept}" to "as in~\cite{...}". (m15(d) is A22.)
   - Ducktor m20 (optional): `1_introduction.tex:15` "yet convey ... yet differ" to "but"; `:19` "General-purpose" to
     "General".
   - Ducktor m8: p and q reused in Eq. 9, bold u reused; renaming math symbols tonight is a risk (not built).
   - R1: `4_experiments.tex:105` "On emotion, the instructed Qwen3 embedding keeps the higher mAP" holds for Table 3
     only (in Table 2 ANGLE's emotion mAP is 0.001 higher); fix "On emotion in \cref{tab:retrieval}," (+3 words).
   - R4: "Qwen3" never defined as the short name. `4_experiments.tex:66` "we use the frozen
     Qwen3-VL-Embedding-2B~\cite{...}" (page 6, ruler 425-426), add "(Qwen3 below)" (+2 words).
   - U: `2_related_work.tex:34` names CRL before CLAY; swap the clauses (0 words), merge wording in the list.
   - U: Table 5 caption (`Tables/photos.tex:28`) "on each backbone" to "on each encoder" and drop one "and" (-1 word);
     Table 2 caption (`main_clustering.tex:31`) "Methods without a named backbone use SigLIP.": change both or neither.
   - U: Figure 3 caption (`3_method.tex:36`) "(a) Scores of painting $i$" to "(a) Scaled scores of painting $i$" (+1
     word, last line nearly full: build).
   - U: M_k notation, no change (defined at `3_method.tex:71`). Figure 4 capitals: after the deadline (B15).
2. **The rest of Part A** in the list's order: A4, A7, A8, A9, A11, A12, A14, A16, A17, A18, A20, A21, A22, A23. A2
   stays input to the user's abstract. A7, A18 edit page 8 (now about 3.4 free lines).
3. **Into REPO:** `git -C REPO fetch` first. Then `python3 tools/apply_fixes.py REPO`, build, `layout_measure.sh`,
   `free_lines.py`, float pages, commit, push. Update status rows 14 and 25.
4. **The user's abstract**, the last build check (status row 16), then the user's Overleaf pull, OpenReview and upload
   (plan 31 handoff, section 4, points 3 to 5).

## 5. Code and pitfalls

- **The shell is zsh:** `for v in $list` does not split words. The scratch variant builder is a bash script; write a
  new one (a spec of name, old, new per block; it copies REPO, applies `apply_fixes.py`, applies the variant, builds
  each in parallel with `layout_measure.sh`, then `free_lines.py`). The old one is in the plan 32 chat's scratchpad,
  `/tmp/claude-0/-project-CoSiR/b0b99321-62dc-4361-96b1-713cf4f515a1/scratchpad/variants.sh`.
- **Get times with `TZ=Europe/Amsterdam date`**; the plan 32 chat first wrote guessed times into `fix_round.md` and had
  to correct them from the commit clock.
- `apply_fixes.py` needs each old text exactly once in its file; rows hold `\\method` and `\\,` escapes.
- Page 5 rule (no added line) was measured before A13 freed a line on page 2; measure again before relying on it.
- The rest as in the plan 32 handoff, section 5. The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 283 MB of scratch builds (none over 100 MB alone), not
  needed: every row is in `apply_fixes.py`, and the next chat rebuilds from REPO.
