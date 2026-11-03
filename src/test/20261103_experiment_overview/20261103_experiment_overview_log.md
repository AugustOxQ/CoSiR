# 2026-11-03 experiment overview (r16 report 2)

**Problem.** The user asked for one overview of what exists and what is missing, clustering first and retrieval
second, with every open to-do labelled (clustering, retrieval, both) and costed, to prioritise the next experiments.

**Steps.**
1. 17:11 to 17:27: a Codex fact-finding read (gpt-6.1-sol, medium; brief `facts_inventory_task.txt`) built
   `facts_inventory.md`: results per method, test set, encoder and aspect (39 clustering rows, 26 retrieval rows),
   coverage grids, ablations, 64 items never run (from the literature notes, specs and parked ideas) and cost facts
   from the run logs. Claude spot-checked the headline cells against the reports and the direction log.
2. Claude drafted the user-read report `docs/user_read/2026-11-03_experiment_overview.md` from the inventory.
3. Checks: a cold read (Haiku) and a fidelity check against the auto reports (Codex, gpt-6.1-sol medium): one
   must-fix (a CRL EmoSet retrieval row exists, the draft said "none") and five should-fix (the frozen c* Affection
   cell is a failed confirmation, not a claim; the read-out and wrong-head scopes; the Affection test reuse; the EmoSet
   near-copy cut; CRL's bars set after its scores were seen). One fix wave by Claude; one re-check by Codex (resume).

**Result.** The report and its 23-item to-do list. Cost figures marked "not measured" or "a guess" are ours.

**Files.** `facts_inventory.md` (the source sheet; each row cites its report), `facts_inventory_task.txt` (the brief).
- Final fixes after the re-check (Codex): the photo-model sentence (separate train photos), the two ablation parts named exactly, ArtELingo genre label count, the comparator and LEVEL/BELOW definitions.
