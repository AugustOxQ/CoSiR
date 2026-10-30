# r12 round log: frozen c* on Affection, CRL bars (build and run)

Spec: [2026-10-30-affection-confirmation-design.md](../../../docs/superpowers/specs/2026-10-30-affection-confirmation-design.md)
(approved 2026-10-30 21:17). Build handoff: `docs/superpowers/handoffs/2026-10-30-r12-build-handoff.md`.

## Build (2026-10-30, unattended, night mode)

Integration branch `r12-affection`; tickets in `.scratch/r12-affection/issues/` (outside git; the list is copied into
the run handoff).

### Build decisions (agent defaults; none changes a decision of spec §4)

- **Store merge without in-place writes.** FI is extracted alone into `qwen3vl_emb_2b_fi/`; `ac_store.py merge` writes
  the 85,007-image store into a new folder, checks gate 0a(i), then renames the old store to `qwen3vl_emb_2b_pre_fi/`
  and the merged one to `qwen3vl_emb_2b/`. Nothing is deleted; the old folder goes on `.scratch/pending_deletions.md`.
- **0a(ii), "the first 200 Affection validation keys including 100 FI images"**: read as the first 100 non-FI and the
  first 100 FI validation keys in sorted order (the sorted list puts FI, `you_aaai_affective`, last).

### Timeline

- 21:24 integration branch and worktree; 21:27 round skeleton (`conftest.py`, `ac_paths.py`).
- 21:30 wave 1: ticket 02 (Codex, `gpt-6.1-sol` high), 06 and 07 (Claude Sonnet `high` workers).
- 21:30 to 22:05 tickets 02, 06 and 07 built, each reviewed by an Opus `high` worker (pass, no must-fix), fixed and
  merged. Ticket 11 built by Codex and reviewed (pass after fixes); its fix round by Codex (89 tests), merged at 22:06.

### Readings settled in the build (agent defaults; reviewers agreed)

- **"Within 0.001 r of the best" (CRL version) and "more than 0.001 r" (photo wording)**: 0.001 in the score's own
  units of r. On Affection the two readings are the same condition; on ArtELingo the score mixes three r_a, so only
  this reading is defined (r11's convention).
- **r11's guard on Affection read-outs**: kept; with one aspect it cannot change a pick.
- **A seeded clustering comparator** (PercepT core, descriptive only) draws its own seeds, as r10's d_R; SC1 and SC3
  have unseeded comparators, so no verdict number depends on it.

### Checks run in the build (CPU)

- Labels: validation and test class and tie counts equal spec §3.4 exactly (ticket 02; re-derived by the reviewer).
- 0a(iii), r10 part: c*'s held emotion block through `ac_cluster` equals A_cand, A_T, d and the interval within 1e-9.
- 0a(iii), r11 stage 4 part: all seven arms' stage 4 scores within 1e-9 and the seven picks equal (23 minutes).
