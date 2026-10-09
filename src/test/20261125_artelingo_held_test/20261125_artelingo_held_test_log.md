# Round 6: the ArtELingo held-split paper test of AFF: run log

Folder date 20261125 is a sequence number. Times are Amsterdam local time (`TZ=Europe/Amsterdam date`). The binding
rule is `DECISION_RULE.md` in this folder (c394b60, SHA-256
7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724); the spec is
`docs/superpowers/specs/2026-10-09-r6-held-test-design.md`.

## Timeline

| Time | Event |
|---|---|
| 2026-10-09 04:54 | `r6 build` chat started from `docs/superpowers/handoffs/2026-10-09-r6-build-handoff.md`; rule SHA-256 checked |
| 2026-10-09 04:56 | Two read-only explorers dispatched (CPU scoring path, Opus; GPU and cluster path, Sonnet); their notes are in the local build tracker `.scratch/r6-held-test/notes/` (`cpu_path.md`, `gpu_path.md`) |
| 2026-10-09 04:56 | Build handoff amended by the decide chat (8df6a4f, git time): the run chat closes by opening `r6 read`; a parallel `r7 decide` chat stays off DAS6 and off this folder |
| 2026-10-09 04:57 | `.scratch/` gitignored (ba6d290) |
| 2026-10-09 04:58 | Integration branch `r6-held-test` from main, checked out in the worktree `/project/CoSiR-r6` (kept off `/project/CoSiR`, where the r7 chat may commit) |
| 2026-10-09 05:10 to 05:22 | Implementer brief, contracts (module APIs, file formats) and 15 tickets with the coverage line written to `.scratch/r6-held-test/` (index `issues/00-index.md`). Findings that shaped them: AFF, CF and R1 are fused on z(B), so B's frozen picks feed them; the head-fit functions return no classifiers (r6 copies needed); the csd refit was never checked against its stored arrays; held images need a new uint8 cache for LB/LoRA; the 8B probe never generated text (DTS needs `generate`, greedy forced); no round 4 or 5 code needs copying (B′(A1) is written directly) |
