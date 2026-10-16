# Handoff: r10 decide 3: grill the user on the new line's candidate ideas (decision B, then the rest of A)

## Brief
- Done: step 3 for the new line: PercepT and conditional-similarity literature checks, PercepT code audit, premise checks, three candidate ideas.
- State: PercepT's own tables show DEC adds little (AMI 0.16 without vs 0.18 with); our PercepT code is not faithful; affect is weak without a supervised encoder (AMI 0.05 vs 0.20).
- Next: the grill, round 1: decision B (which idea), the meaning of "label-free", the PercepT baseline.
- Blocked: waiting for the user's answers.

> Written 2026-10-17 01:05 by `r10 decide` (cut at 200k context, after step 3). Loop step reached: step 3 done; step 4
> (decision B) and step 5 (grill) next; decision A still a draft. Direction log entry:
> [2026-10-15](../direction_log.md) (no new entry yet; it is written at the next wrap-up).

## 1. Read in this order

1. **The step 3 note**, [2026-12-06_new_line_candidates](../../reports/auto/percept/2026-12-06_new_line_candidates.md):
   the facts, the three candidates, the parked ideas, and the round-1 questions with our recommendation folded away.
   This is the grill's facts brief; present from it.
2. Only when a question needs detail:
   [PercepT literature](../../reports/literature/2026-12-05_percept_neighbourhood.md),
   [conditional similarity](../../reports/literature/2026-12-04_conditional_similarity_update.md),
   [code audit](../../reports/auto/percept/2026-12-04_percept_code_audit.md) (section 7: the eight changes for a
   faithful PercepT), premise checks `src/test/20261017_aspect_premise_checks/` (log, `results.json`).
3. Before the grill: `~/.claude/references/grill.md` (answer-first, at most three key questions per round, a scenario
   check after each key answer, closing summary, ask once about an html-plan page).

## 2. Decided by the user, not to be reopened

- The decisions in the [r10 decide 2 handoff](2026-10-15-r10-decide-2-handoff.md), section 2 (stop and archive the
  conditioned line; held data rules).
- **The new line's bases (user, 2026-10-17, in the user's words):** "switch back to buddy-vs-PercepT"; the core base
  papers are (1) GeneCIS, "conditional similarity as the core idea", (2) PercepT, "the main model framework", (3) the
  conditional buddy, "our own idea". The submission builds on these three.
- Still a draft from 2026-10-15 (to confirm in the grill, not to re-ask from zero): PercepT's task plus retrieval; a
  method that discovers aspects, label-free if possible; CVPR soft (works by about 2026-10-29, else try to the
  abstract deadline 2026-11-10, else no submission).

## 3. Where things stand

- **Tension to raise once:** on 2026-10-15 the line was "a condition-free model"; on 2026-10-17 the user named
  conditional similarity as the core idea. Candidates (a) and (b) reconcile the two: the model trains without a
  condition input, and the condition (an aspect) is chosen at query time. Check that the user means this.
- PercepT: preprint (2026-06-02), 0 citations, no code; Stage 1 is IDEC (uncited); headline SI 0.97 is DEC scored in
  its own space; AMI 0.18 is the harmonic mean of the emotion and genre AMIs.
- Our PercepT code: no generation is faithful (captions averaged per painting; B/32 + RoBERTa instead of L/14 +
  ModernBERT; concatenation instead of Eq. 2; a missing encoder layer; a balance term; forced K instead of Algorithm 1;
  one-hot Stage 2 targets). The 09-30 "buddy ties PercepT" was against this PercepT-like baseline.
- GeneCIS side: no work found uses aspects discovered from image-text data as retrieval conditions; expected baseline
  CRL (NeurIPS 2025) with LLM-named aspects, and an instruction embedder (Qwen3-VL-Embedding).
- The buddy graph is a known construction (UMAP intersection, Seurat WNN, TAC): claim its use, not the graph.
- Feature caches that exist: ArtELingo CLIP B/32, CSD style, VGG Gram, Qwen3-VL-Embedding-2B, SigLIP 2; GeneCIS
  COCO and VG crops (`/data/SSD2/pre_extract/genecis_*`). Not cached: CLIP L/14, ModernBERT-GoEmotions (PercepT's
  encoders); only 28-d GoEmotions classifier outputs exist, in the archive caches (paths in the premise-check log).

## 4. Open points for the next step

1. **Key, round 1** (the note's section 4): decision B; the meaning of "label-free" for affect; the PercepT baseline.
2. **Key, round 2** (after B): what counts as "something works" by 2026-10-29; the evaluation (per-aspect AMI at
   matched k, Stage 2 against human labels, aspect-conditioned retrieval; GeneCIS itself or not).
3. **Minor batch** (defaults in the note): second domain (Affection, no genre labels), backbone (CLIP L/14 plus B/32),
   retrieval sets (ArtELingo labels; COCO-4c mood).
4. After the choice: the candidates not chosen and the two parked ideas go to the direction log's "Parked ideas", one
   line each. Then the spec (step 6), in the same decide chat if context allows.
5. **Not done here:** no user-read report was written for the step 3 note; the grill presents its content directly.
   Write a half-page one at the wrap-up if the user wants it.

## 5. Code and pitfalls

- PercepT code lives on branch `experiment/percept_topic_pipeline` (worktree `/project/CoSiR-buddy_prototype_conditioning`);
  the head-to-head harness is `scripts/buddy_percept_sweep/` there (215 CPU tests pass). Do not reuse its PercepT path
  as "PercepT" without the audit's changes.
- A new round folder needs a new module prefix (`apc_` is taken by the premise checks).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours (the 7 untracked archive result files predate this chat).
- On disk over 1 GB: nothing new (the premise-check folder is 104 KB).
