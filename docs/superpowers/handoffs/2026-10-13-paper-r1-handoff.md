# Handoff: CoSiR paper R1: fill the literature gaps, then draft v1 of the stable sections from the approved outline

## Brief
- Done: R0. Framing (task and protocol first), title, four ARS rounds, the Phase 0 record and the outline, all approved by the user.
- State: the outline's numbers were checked by Codex (60 of 62 match; the one error is fixed). No draft text yet.
- Next: the R1 literature gap-fill (in-context learning, backbones, 18 unverified citations), then draft v1 and its full ARS review by 2026-10-19.
- Blocked: nothing. The r9 results (about 10-16) come later, at R2.

> Written 2026-10-13 16:01 by the paper start chat (`claude-paper`, outside the loop's phase chats). Cut at the context
> meter (about 210k tokens), after R0. Previous handoff: [paper start](2026-10-13-paper-start-handoff.md).

## 1. Read in this order

1. The paper folder `docs/paper/cvpr2027_aspect_episodes/`: `README.md` (every decision, who made it, the schedule),
   then `outline.md` (the approved outline, its evidence map, its "Checks before drafting"), then
   `phase0_paper_configuration.md`. The folder is gitignored: it exists only on disk.
2. `notes/related_work_digest.md` in that folder: what our literature notes say, by section, with each citation's
   status (verified, opened, not verified) and the gaps (its §5).
3. For drafting, the sources the outline names: the stage report `docs/reports/stage/2026-10-12_affect_steering_line.md`
   and the held-test report `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md`, read section by section as each
   draft section needs them, not whole.

## 2. Decided by the user, not to be reopened

All in the paper README's decision table (2026-10-13):
- Framing: task and protocol first; AFF is the label-free method that shows the condition can be read.
- Title: "Similar in Which Respect? Cross-Modal Similarity Conditioned on Examples".
- Four ARS rounds: R0 to 10-15 (done), R1 to 10-19, R2 to 10-30, R3 to 11-09, Final to 11-15.
- Phase 0 record as written (CVPR 2027, IEEE numeric, English-only abstract, about 6,500 words, Markdown then LaTeX).
- Outline approved with changes: related work in four headings, one of them in-context learning; analyses added:
  selected episodes (body Figure 6, drawn by a rule fixed before looking), affect groups, MLLM phrases and gate
  shares (supplement).
- From the paper-start handoff: results enter the body if their report exists by 2026-11-06; every backbone in one
  main table, pass or fail; the claim sentence in versions until the r9 reads exist.

## 3. Where things stand

- R0 is done. R1's work, in order:
  1. **Literature gap-fill** (the `deep-research` agent; Phase 0 note 2, outline "Checks before drafting" item 5):
     in-context learning in LLMs and multimodal models, visual in-context learning, benchmarks of multimodal
     in-context learning; CLIP, SigLIP 2, Qwen3-VL; few-shot background (prototype networks); Leiden (Traag et al.),
     LoRA, Tversky; the 18 unverified rows of the digest. Read the earlier literature notes first; save the note in
     `docs/reports/literature/` and index it. The TPIPS concurrent-work search waits for the week before the abstract.
  2. **Figure 6 data**: check that `src/test/20261125_artelingo_held_test/results/held_arrays.npz` keeps per-episode
     rankings per scorer; write the drawing rule into the outline before any episode is looked at.
  3. **Draft v1** of the sections that do not wait for r9: abstract and introduction with claim version V0, §2 to
     §5, §6.1 to §6.5 (CLIP), §6.9, §7 and §8; §6.6 to §6.8 stay as marked placeholders.
  4. **Full ARS review** of v1 (`academic-paper-reviewer`), then one revision, by 2026-10-19.
- The r9 build runs in its own chat; H6 and H7 report about 2026-10-16. They enter at R2.

## 4. Open points for the next step

- **How ARS drafts** (minor, agent default to state): ARS `full` mode from Phase 3 (argument blueprint, then the
  writer/evaluator pair), or our own drafting by subagents section by section with ARS used for the review rounds.
  Say which and why in one line.
- **A backup of the paper folder** (minor, ask the user): `docs/paper/` is gitignored in this public repo, so the
  drafts have no version history and no copy off this disk. Propose a private repo (as for the dev-log site).
- **Co-authors and funding** (minor): ask before the abstract registration on 2026-11-10.

## 5. Code and pitfalls

- Every number in a draft is traced to its result file (outline evidence map), never to memory or a summary. HT is
  the source for held numbers; the JSON files agree with it at the shown precision (Codex check, 2026-10-13).
- The text names "B′(A0)" and "B" apart: 18.84 is B′(A0), 18.81 is B (the error the check found).
- Pair metrics: "eight pair metrics, 12.99 to 13.21; the support prototype's fitted weight is 0, so it equals cosine".
- The paper never names the repo and avoids the name "CoSiR" (double-blind; the repo is public).
- Licensed wording only: held read H5's claim text, r9 spec §6 for the scope results, "to our knowledge" for novelty,
  and none of the digest's "claims we must not make".
- The r9 build chat works in the same working tree on `r9-scope`: commit only your own paths.

## 6. State at handoff

- Running: nothing of this chat's.
- Uncommitted: nothing tracked. The paper folder is gitignored by design (on disk only).
- On disk over 1 GB: nothing new.
