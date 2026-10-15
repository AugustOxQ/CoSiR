# Handoff: CoSiR paper R1 (2): draft §2, then the abstract, §1, §7 and §8; number check, humanizer, full ARS review and one revision by 2026-10-19

## Brief
- Done: the literature gap-fill (two notes, `refs.bib` with 96 checked entries), the Figure 6 drawing rule, draft v1 of §3 to §6 with Tables 2 and 3, and a private repo for `docs/paper/`.
- State: v1 has 4 of 8 sections, 4,276 words by the counter. Every number has a TRACE line.
- Next: the §2 writer, then the writer for the abstract, §1, §7 and §8; then the number check and the ARS review.
- Blocked: nothing.

> Written 2026-10-13 16:37 by the paper R1 chat (`claude-paper`, outside the loop's phase chats). Cut at the context
> meter (about 200k tokens), at a clean point. Previous handoff: [paper R1](2026-10-13-paper-r1-handoff.md).

## 1. Read in this order

1. `docs/paper/cvpr2027_aspect_episodes/README.md`: every decision and who made it (six new rows today), the
   schedule and the layout.
2. `notes/drafting_brief_v1.md` in that folder: the common brief every section writer got. It fixes the names, the
   number tracing, citations, style and anonymity. Give it to the next two writers unchanged.
3. `outline.md` §2, §1, §7, §8, the claim versions, and "Checks before drafting" (check 3 was rewritten today).
4. For §2: `docs/reports/literature/2026-12-02_paper_icl_background.md` (in-context learning; the novelty boundary;
   §4 has ready wording) and `2026-12-03_paper_citation_gapfill.md` (corrections, and in its addendum the key
   mapping: five ICL works sit in `refs.bib` under other keys, e.g. `brown2020gpt3`, `li2025bgeenicl`).

## 2. Decided by the user, not to be reopened

All are in the paper README's decision table (2026-10-13), plus the earlier ones listed in the previous handoff §2:
- `docs/paper/` is its own private repo, `AugustOxQ/cosir-paper`. Commit and push paper work there, never to the
  public repo. A global rule now says the same for every public project (`docs-layout.md`, `git.md`).
- The method-A folder moved to `docs/paper/archive/`.
- Markdown until the Final round. LaTeX previews at the ends of R2 and R3 check length and reading only.
- One file per section in `drafts/vN/`, joined by `tools/join_draft.py` into `paper_vN.md`, which is never edited by hand.
- `[@key]` citations and `refs.bib` from R1; an entry enters `refs.bib` only after its citation is checked.
- The data stays as it is. §3.2 gets the user's sentence on the 11 dropped artists (outline check 3).

## 3. Where things stand

- **Drafted** (`drafts/v1/`): `03_task.md`, `04_protocol.md`, `05_method.md`, `06_experiments.md` (6.6 to 6.8 are
  PENDING placeholders), and `tables_figures.md` (Tables 2 and 3 built by script from `descriptive.json`; captions for
  Figures 1 to 6; the Figure 6 caption has `[n]` slots that fill when the episodes are drawn).
- **Agent defaults, told to the user and not yet confirmed:**
  - Paper names: RS for the scorer "R1", B′ for B′(A0), B′+style for B′(A1).
  - The details of the Figure 6 drawing rule (outline §6.5).
  - Drafting by our own section writers, with ARS used for the review rounds.
- **Writer findings to settle at the v1 read:**
  1. B already uses GoEmotions: its grouping term is k-means-64 on the captions' GoEmotions probabilities, CLIP
     image features and CLIP caption features (§4.2 says so). AFF is fused on B, not on B′. This also answers the
     reviewer risk "GoEmotions leaks the labels": every condition-free comparator has the same input.
  2. AFF's setting pick maximises the smaller of "hits above B" and the condition gain; the matched control picks by
     hits (`r2_fusion.py`). §5.4 states it.
  3. The fixed recipe reuses the affect grouping instead of refitting it (R9 §3.7). §5.5 says so; outline §5.5's
     wording "every part fitted again" needs the same refinement.
  4. §5.4 gives the redundancy numbers (0.35 to 0.38 against 0.62 to 0.71) as the reason for choosing affect. Keep
     them there or move them to §6.9.
  5. §6 derives the either rate as 2·R@1 − gain and adds a held reading cost of 0.59 (1.80 / 3.06) beside the fresh
     0.52. Table 2 shows intervals for all rows; its caption says which were tested.
  6. "23 styles" means the 23 of 27 styles with at least 30 labelled paintings in the development split.
  7. All our data is ArtELingo's English *training* split, split again 70/10/20 by painting. ArtELingo's own
     validation and test splits are unused, and a reviewer may ask why: §3.2 or §7 needs one line.
  8. The count of reads of the held paintings in §4.3 is PENDING r9.

## 4. Open points for the next step

- **R1 steps left, in order** (agent defaults):
  1. §2 writer: `worker`, Opus `high`, with the common brief, outline §2, the digest and both notes. It also writes
     Table 1 (property table) in `tables_figures.md`.
  2. Writer for the abstract, §1, §7 and §8: Opus `high`, after §2, reading the finished sections. Claim version V0.
  3. Join, then the number check: every TRACE line re-derived from its file. Codex if `codex-usage` shows room
     (`codex-jobs.md`), else a Sonnet `high` worker. Then one fix wave.
  4. A humanizer pass over the joined draft.
  5. The full ARS review (`academic-paper-reviewer`, full mode) into `reviews/`, our answer, one revision. Tag the
     pre-review state in the paper repo (minor, agent default: the revision stays v1).
- **Minor, ask the user:** co-authors and funding before the abstract registration on 2026-11-10.
- **Minor:** when to draw Figure 6. It is a quick CPU job: re-score the drawn episodes with B and AFF.

## 5. Code and pitfalls

- The previous handoff's §5 still holds: numbers are traced to files, B′ and B are told apart, the pair-metric
  wording is fixed, no "CoSiR" or repo, licensed wording only.
- The counter in `join_draft.py` counts math tokens and citation keys, so the prose is about 10% below its numbers.
- Literature notes go to the public repo: never write the paper's title in them.
- The r9 build chat works in the same tree on `r9-scope`: commit only your own paths.
- Rocchio 1971 is still unverified: write "a Rocchio-style update" without the citation, or verify the chapter.

## 6. State at handoff

- Running: nothing of this chat's.
- Uncommitted: nothing. The paper repo is pushed at the joined v1.
- On disk over 1 GB: nothing new. Scratch: one small script in this session's scratchpad.
