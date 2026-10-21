# Handoff: decision C on the D2 ablation, then the next step, the improvement list, the framing and item 5's grill

## Brief
- Done: the D2 ablation reports (auto report 12-22, user-read report), the literature and framing notes.
- State: the instructed graphs carry D2's gain; without the caption terms, style AMI is 0.3945 against D2's 0.3697 and the instructed Qwen image's 0.3952 (development).
- Next: your reading of the results brief (decision C), the wrap-up, then the next step and item 5's grill.
- Blocked: nothing. The container restart question is asked in this chat.

> Written 2026-10-21 07:08 by "r10 run 13". Loop step reached: step 9 (verify and report) done; step 1 (decision C)
> starts. Direction log entry: none yet for this read; the last one is
> [2026-10-21 · Claim test read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The auto report's results brief:
   [2026-12-22_d2_ablation.md](../../reports/auto/percept/2026-12-22_d2_ablation.md), "Results brief". Show it to
   the user without the `<details>` block, ask for their reading in a sentence or two, then open ours and say where
   the two differ (`research-loop.md`, decision C). The user-read report beside it:
   [docs/user_read/2026-10-21_d2_ablation.md](../../user_read/2026-10-21_d2_ablation.md) (its advice is folded the
   same way).
3. For the wrap-up: `~/.claude/templates/direction_log_entry.md`, `storage.md`, `.scratch/pending_deletions.md`.
4. For the next step and item 5: the literature note
   [2026-12-20_ways_to_improve](../../reports/literature/2026-12-20_ways_to_improve.md) (12 ranked ideas, "Row"
   column), the framing note [2026-12-21_advantages_framing](../../reports/auto/percept/2026-12-21_advantages_framing.md),
   the ArtELingo-28 facts `.scratch/r10-run-12/a28/a28_overlap_facts.md`, the f2 facts brief for backbones
   `.scratch/r10-decide-16/f2_facts.md`.
5. Before the grill: `~/.claude/references/grill.md`; before any spec: `spec-templates.md`.

Skills: grilling, domain-modeling (the grill runs as grill-with-docs).

## 2. Decided by the user, not to be reopened

- The five-part plan (D1, 2026-10-21 02:50, spec
  [2026-10-21-d2-ablation-design.md](../specs/2026-10-21-d2-ablation-design.md) §4): parts 1 to 4 are done; item 5
  (code for other backbones and ArtELingo-28) is grilled in this morning chat (D11).
- Development data only for the ablation (D5); no bar, two reports (D9).

## 3. Where things stand

- **Run** (r10 run 12): stage 0 passed bit for bit (18/18); 11 rows × 3 seeds trained; scores and diagnostics done.
- **Reports** (r10 run 13): auto report 12-22 drafted by Codex, claim review by a Sonnet worker (about 250 numbers
  checked, effect words fixed), results brief and "Our reading" rewritten by the controller; user-read report with a
  Haiku cold read and a Sonnet fidelity check, one fix wave, no re-check. The auto report is 17.5 KB, over the 15 KB
  target. The user-read report had a manual plain-English pass, not the humanizer skill.
- **Confound found in the claim review:** our model's loss is buddy + own-caption + 0.5 teacher, so the teacher has
  1/5 of the weight; without the caption terms (and without the buddy terms) it has 1/3, and image endpoints only also
  gives the image–image buddy term 2/3 against 1/10. The caption-free gains may come from either change.
- **Main facts** (image AMI, 5,000 development paintings; D2 0.0962 / 0.3697 / 0.4540):
  - one default-view graph: −0.0222 / −0.1681 / −0.0287 (clear on all three);
  - no teacher: not resolved / small / not resolved; default-view teacher: not resolved on all three;
  - image endpoints only (its caption network gets no loss): +0.0057 small / +0.0248 clear / not resolved; no own-caption terms:
    +0.0055 small / +0.0180 clear / not resolved;
  - no instruction anywhere: 0.0733 / 0.1848 / 0.4241, below PercepT core (0.0772 / 0.2286 / 0.4359);
  - instructed inputs: genre 0.4545, below their own input (0.4956).
- **Amendment A1** (agent default, night): the no-content-correction row uses the style graph of the instructed image
  view (R0), not the caption view (R0o). For the user's check.
- **Not checked:** whether a second development sample (train paintings outside the model sheet sample, with all
  stores) can be drawn for option 1 of the brief. Check it before its cost is quoted as fact.
- **The 11-10 gate** needs aspect-named retrieval: the image-endpoints-only variant has no caption network.

## 4. Open points for this chat

a. **Decision C (key):** the user's reading first, then ours (`research-loop.md`, step 1).
b. **Wrap-up:** the direction log entry and the "Now" paragraph (it still says the ablation ran on node408; it ran on
   the local 3090, D3); the storage summary with every row of `.scratch/pending_deletions.md` and the two outputs below;
   disk ledger rows dated 2026-10-21; `wrapup-check`, `disk-check`, `routing-check`, then `sync.sh`.
c. **A1 (minor):** confirm or change.
d. **Decision A (key):** the brief's options 1 to 3 (check the caption-free variants on other development paintings;
   genre; item 5), with the literature note's ideas and the framing note.
e. **Item 5's grill (key):** other backbones (f2 brief) and ArtELingo-28 (a28 facts: 1,658 paintings, 790 in our
   train paintings, 122 validation, 248 held, 498 outside our store), with Qwen's instructed graphs kept for training.
f. **Restart question** before any build or run chat opens (`loop-chats.md`, "How to cut").

## 5. Code and pitfalls

- Change no code in `src/test/20261021_d2_ablation/`: `ab_row.json` binds a code hash. Its outputs, logs and graphs
  are gitignored and live only in the main checkout.
- Report sequence names: 12-20 literature, 12-21 framing, 12-22 ablation; the next report is 12-23.
- In running text, plain names: "image endpoints only" (R7), "no own-caption terms" (R6), "no instruction anywhere"
  (R11); labels in tables only.
- Held data: a held test of a caption-free variant chosen on development paintings is fine; choosing between variants
  on held paintings is not (`held-data.md`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing after this commit (`.scratch/` is gitignored).
- On disk over 1 GB: `src/test/20261021_d2_ablation/outputs/` 4.5 GB (row exports); `src/test/20261020_claim_test/outputs/stage2/`
  4.6 GB (keep). Both need ledger rows at this wrap-up.
