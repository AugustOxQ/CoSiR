# Handoff: r9 build 3: build the stage tickets from 05 on (generic chain, recipe, ArtELingo-28, runners)

## Brief
- Done: stage tickets 04 to 28 written; 04 (backbone loader, Opus-reviewed), 24 (ledgers) and 28 (licence) merged.
- State: 25 (independent re-derivation, Codex) is built on branch r9-t25: 17 tests, CLIP check 272 comparisons, 0 differences; not yet reviewed.
- Next: 25's Opus review and merge; 05 (generic chain I) and 26 (probe corpora); then 06, 07, 14, 19.
- Blocked: nothing. For the user: ask ArtELingo-28's authors for its data terms; H4's ledger cell differs from rule §7.1.

> Written 2026-10-13 17:10 by `r9 build 2` (cut at the context meter). Loop step reached: step 7, the build, tickets
> 04, 24, 28 merged. Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. The tickets: `.scratch/r9-scope/issues/00-index.md` (list, coverage line, shared interfaces), then each ticket as it
   starts. They cite the rule and the spec by section.
2. The rule `src/test/20261202_r9_scope/DECISION_RULE.md` @ a443902 (binding), the sections your tickets cite.
3. The round log `src/test/20261202_r9_scope/20261202_r9_scope_log.md` ("Stage tickets", "Ledgers", "ArtELingo-28 licence").
4. `~/.claude/references/matt-chain.md` (implement-spec with our changes) and `codex-jobs.md`. The Codex brief parts
   used in this chat: `.scratch/r9-scope/notes/codex_common.md` (append it to every Codex ticket brief).
5. The code map `.scratch/r9-scope/notes/aff_chain_map.md`, only the sections a ticket cites.

## 2. Decided by the user, not to be reopened

All of spec §4 (D1 to D34); the agent defaults of the earlier build chats are in the round log.

## 3. Where things stand

- **Merged on `r9-scope`:** 01 to 03 (preparation), 04 (`r9_backbone.py`, `r9_data.py`: the three backbones load in
  `load_artelingo()`'s order, CLIP's split for all; 50 tests; Opus review re-derived the join on 2,000 rows per store
  and the split against round 6, two test gaps fixed), 24 (held ledger Backbone column; seeds 55 to 57 with builder
  seeds 55001 to 57027 and the smoke builder seeds entered after a collision check), 28 (licence, in the log).
- **Ticket 25** (branch `r9-t25`, commit d0c31e8, `rederive/` with `rd9_` modules, 3,220 lines): built from the rules
  and round 6's `rederive/` only. Codex's notes: `rederive/phase_notes.md` (interface choices, open schema points),
  `inputs.md` (the pinned files phase 1 reads for CLIP: the recipe must write the same for Qwen and SigLIP 2; check
  07, 09, 10, 11 against it), `rd9_commands.md`. It needs an Opus `high` review (independence: no `r9_`/`r6_` import;
  the arithmetic; the A28 path on synthetic data) before the merge. Codex thread `01a135b2-ac45-73f2-b5b6-b5bc4881571b`.
- **Frontier now:** 25's review; 05 and 26 (blocked by 04 only). After 05: 06, 07, 14, 19;
  then 08, 09, 10, 11, 17; after 06 and 16: 18; and so on (index).
- **Agent default in the split** (log): the ArtELingo-28 row space is the 74,569 read rows before the empty-output
  drops; the store holds the kept rows. CUB's development species by `src/data/cub.py` `dev_species`.
- **Codex:** 04 and 25 together used about 15% of a five-hour window (reset 21:37); weekly 95% left. Job-ledger rows
  for 04 and 25 are in `/project/claude-config/codex/jobs.md`; at the cut the window showed 37% used.

## 4. Open points for the next step

- **For the user (minor):** rule §7.1 says rows H1 to H5 read "CLIP ViT-B/32"; H4 was the backbone check on CUB with
  four backbones, so its cell reads "CLIP ViT-B/32, SigLIP 2, PE-Core, Qwen3-VL-Embedding-2B (backbone check)".
  The rule's sentence needs the user's approval to change; the ledger states the fact.
- **For the user (minor, paper):** ArtELingo-28 states no data licence; ask its authors in writing before submission (D22).
- **Stage order** (rule §3.5 of the spec): 0a after 06, 0b after 07 to 12, stage 2 after 0b; stage 1 stores after 14
  and 15. The controller runs the long steps (0a about 10 min CPU, the CLIP replay about 1 h with A3 on the GPU).

## 5. Code and pitfalls

- Verdict tickets: Opus `high` review before the merge; mutation-check the key guards (04's review found a guard with
  no failing test). Codex fix rounds by `resume <thread_id>` (first line of the job's `--json` log).
- `git commit -a` in the main checkout can pick up other chats' files (the paper chat commits to `r9-scope` too):
  add paths explicitly.
- The shell for Bash tool calls is not fish: use `for ...; do ...; done`.
- Worktrees `/project/CoSiR-wt/r9-tNN` (branches `r9-tNN`) stay until the wrap-up; list them on
  `.scratch/pending_deletions.md` at the end of the build.

## 6. State at handoff

- Running: nothing (no subagents, no Codex jobs, no cluster jobs).
- Uncommitted: nothing on `r9-scope` except the untracked round 8 results trees, as before.
- On disk over 1 GB: nothing new. Small scratch: `/tmp/r9-t04-review-*` (6.6 MB, Codex's mutation copies).
