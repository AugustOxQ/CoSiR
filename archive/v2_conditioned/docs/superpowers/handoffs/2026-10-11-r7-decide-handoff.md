# Handoff: decide round 7, design L from the style source (a told ceiling, a screen of style sources, then the grouping redesign): step 2 short, then step 3

## Brief
- Done: round 6 read and wrapped up (held test GO, decision C recorded, storage cleaned, DAS6 nodes cleaned).
- State: AFF +0.60 R@1 over B′(A0) on held paintings; style × genre −0.43; no margin over B′(A1) (+0.14) or R1.
- Next: `r7 decide`: confirm type and loop size, then step 3 for design L, starting from the style source.
- Blocked: nothing; the user starts this chat by hand.

> Written 2026-10-11 02:40 by the `r6 read` chat, for the user to start by hand. Loop step reached: decision A made
> (design L, from the style source); type and loop size to confirm; step 3 next. Direction log entry:
> [2026-10-11 · Round 6 held test GO; design L next, from the style source](../direction_log.md). Supersedes
> [2026-10-09-r7-decide-handoff.md](2026-10-09-r7-decide-handoff.md), whose report pointers (its §1.3) still hold.

## 1. Read in this order

1. `docs/superpowers/direction_log.md`: "Now", "Parked ideas" and the 2026-10-11 entry (decisions C and A).
2. The results brief and the per-pair table of round 6's report,
   `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` (brief at the top; per pair in "Results").
3. Why the source, not the reader, is the limit: `docs/superpowers/direction_log_raw_2026-10.md`, the round 2 lines
   (14 and 18: CSD's AMI with style 0.34 and with genre 0.33; the no-caption spike's told style × genre only +0.134).
4. The reports the 2026-10-09 handoff cites in its §1.3 (cited parts only): round 3 §7 and §8, the Leiden partition
   report around lines 418 to 434, the reader-fix-with-CSD report around lines 43, 141 and 565 to 593, round 4's
   B′(A1) floor.
5. `GLOSSARY.md`, `docs/superpowers/constitution.md`; `~/.claude/rules/research-loop.md` steps 2 to 6,
   `~/.claude/references/grill.md`, `~/.claude/rules/spec-templates.md`.

## 2. Decided by the user, not to be reopened

- Design L is next, in its own loop and chat (user, 2026-10-09).
- **It starts from the style source** (user, 2026-10-11). The question is the user's own: the style information comes
  from the grouping, which comes from the encoder, so can design L fix style × genre without touching that source? Our
  answer, no, and the order below, are our proposal, which the user accepted ("Good, so prepare everything"):
  1. a **ceiling check**: a told grouping built from ArtELingo's true style labels; if even that does not fix style ×
     genre, the problem is downstream (scoring or combining) and design L stops there;
  2. a **source screen**: each candidate source (CSD raw, CSD with the CLIP image embedding regressed out, other style
     encoders that step 3's literature check finds) grouped label-free, then scored by its match to style against
     genre (labels only to score);
  3. the **grouping redesign and a reader run** on the best source only.
- **Decision C:** "we may work beyond the ArtELingo paper, toward a bigger method paper; if the method fails, we drop
  back to the ArtELingo paper" (user, 2026-10-11, after seeing ours).
- Go/no-go on Fri 2026-10-16; CVPR abstract 2026-11-10.
- Design L may get its own pre-registered held read later (constitution C5, amendment 2). The user has now seen
  round 6's held numbers, so the 2026-10-09 handoff's question about designing L blind to them is settled: the paper
  says design L was designed after AFF's held read and its numbers were seen.

## 3. Where things stand

- Round 6 (held paintings, seeds 52 to 54, 36,864 episodes): AFF 19.44 R@1; +0.60 [+0.46, +0.73] over B′(A0), +0.63
  over B and over its matched control; condition gain +3.06. Against B′(A1) +0.14 [−0.03, +0.30], against R1 +0.02
  [−0.06, +0.11]. Per pair against B′(A0): emotion × style +0.59, emotion × genre +1.64, style × genre −0.43
  [−0.65, −0.21]; against B′(A1), style × genre is −1.24.
- Seed 42 (development): B′(A1) 18.805 against B′(A0) 18.437; a style grouping lifts the condition-free score, but no
  reader has yet used it to steer.
- A better style grouping also lifts the condition-free B′ built on it (round 2: style × genre +0.94 with CSD, its
  matched counterpart +0.60), so any gain must survive the matched control (C2).
- The 30-minute label-free check of a de-genred style grouping (parked) is source step 2's "CSD with CLIP regressed
  out"; never run.

## 4. Open points for the next step

1. **Step 2, short:** confirm the type (new idea) and the loop size (full: step 3 includes a literature check). Our
   default: full.
2. **Step 3:** parked ideas first, then our own results, then the literature check on the `deep-research` agent:
   style encoders and representations that separate style from content and genre (skim `docs/reports/literature/`
   first; say which model and depth and why). Two or three candidates; the user picks (step 4).
3. **For the grill (key, answer-first):** the ceiling check's stop threshold, set before it is read; which sources
   enter the screen; whether picking the encoder by label match on development rows is acceptable (selection,
   disclosed); what design L must show by the 2026-10-16 go/no-go.
4. **Minor, for later:** a bigger method paper also needs a second dataset or backbone (round 6's option (c)); design L
   alone does not supply it.

## 5. Code and pitfalls

- New round folder in `src/test/` with the next sequence date after `20261125` and module prefix `r7_` (round folders
  go on `sys.path`). Development on seed 42; selection seeds from 55 (52 to 54 were round 6's held seeds). No held row
  is read (C5); sample IDs per C6; every Python call with `PYTHONDONTWRITEBYTECODE=1`.
- Style and genre labels may be used to score a grouping on development rows only, never to build the method's
  grouping (the ceiling check's told grouping is a reference, not a method).
- CPU work: 8 threads, `uptime` and `free -g` first; local GPU under its lock.
- DAS6: the reservation of node401, node402 and node408 ends about 2026-10-14 02:40; all three were cleaned of round
  6's files. Use the CLI at `~/.claude/skills/cluster-run/cluster` (`cluster` on PATH is Graphviz's). A project's own
  job inputs on a node now go under a profile's `INPUTS_REMOTE`, which `cluster du` lists and `clean --path` may remove.
- Codex: the allow rule matches only a command that starts with `codex e --dangerously-bypass-approvals-and-sandbox`
  (no `cd` or `timeout` in front).
- **Chats record:** started by hand, so add this chat's row (label `r7 decide`, this session's id, predecessor the
  `r6 read` session `eb2b1504-c50b-4cf9-aef4-a63753755532`) to `docs/superpowers/handoffs/chats.tsv` and commit it.
- At this chat's end (spec and rule approved), cut to `r7 build` with `loop-next` as usual.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing.
- On disk over 1 GB: `res/cluster_jobs/` 3.1 GB (round 6's pulled outputs; keep, ledgered).
