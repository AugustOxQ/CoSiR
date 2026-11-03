# Handoff: r18 spec: write the outside-methods spec from the finished grill, settle its open points, get the brief approved

## Brief
- Done: TPIPS downloaded (32 GB) and probed: 219 ms per image per factor on the 3090, about 6.4 GPU h for r18. Grill
  done: TURTLE then TEMI; CLAY with both text sets, picked on validation.
- State: every grill answer recorded (direction log, round log). No spec yet.
- Next: write the r18 spec straight from the grill, ask only the open points below, get the brief approved, cut to build.
- Blocked: auto mode refused running cloned TURTLE code; the spec asks the user how the build runs outside code.

> Written 2026-11-03 19:25 by tab "r18 decide". Loop step reached: after the grill (decisions A and B made), at step
> 6. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 grill".

## 1. Read in this order

1. [Round log](../../../src/test/20261103_outside_methods/20261103_outside_methods_log.md): the probe, the env,
   the refused step and every grill answer with who decided.
2. [Facts brief](../../../src/test/20261103_outside_methods/om_facts_brief.md): CLAY, TEMI, TURTLE code; populations
   and class counts (§C); read-out (§D) and clustering (§E) protocols; CRL's adapter. Sources and pinned commits.
3. The previous handoff's §2, [r18 decide](2026-11-03-r18-decide-handoff.md): the user's earlier decisions (items,
   TPIPS model, CLAY link, paper framing, the paper-change line in each report).
4. The overview's plan, loop 2: [experiment overview](../../user_read/2026-11-03_experiment_overview.md), "The plan".
5. r17's spec, if written by then (`docs/superpowers/specs/2026-11-03-*`): its cluster counts per aspect and its
   scoring module's name.
6. Before the spec: `~/.claude/rules/spec-templates.md` and its templates (`spec_core.md`, `spec_experiment.md`).

Skills: none for the spec. cluster-run when checking the DAS6 node.

## 2. Decided by the user, not to be reopened

- The previous handoff's §2 holds (2026-11-03, before 18:35).
- Closing summary of the grill, confirmed by the user at 19:20:
  1. r18 runs outside methods on ArtELingo held and MMArt. TPIPS: the 8B late fusion model, factors "emotion", "style",
     "genre", on DAS6. CLAY: CLIP L/14, with CRL's 50-value lists and new 100-value lists, picked on validation.
     TURTLE, then TEMI.
  2. CLAY and TPIPS are context rows in both tasks. They use r11 to r15's read-out, picked on validation. Each gets
     a WIN, LEVEL or BELOW label against c*, with no pass or fail for the loop.
  3. TURTLE replaces k-means on the plain image (four encoders, r17's counts), fitted on the scored set with its
     paper's defaults. It also runs on c*'s heads at the class count and 20. TEMI follows; a timing run at its build
     sets its scope, and results after 11-16 go to the supplementary.
  4. TPIPS, CLAY and TURTLE build after r17's scoring module merges; TEMI comes later. Each report says what changes
     in the paper draft: related work, method, experiments.
- Who decided: TURTLE-then-TEMI is the user's own (our recommendation was TURTLE only; flag it in the brief). CLAY's
  two text sets: the user's own, after a cost table, before our recommendation (the same). Minor points i to vii of
  the round log: user, after seeing the recommendation. Scenarios for both key answers confirmed. No plan page.

## 3. Where things stand

- TPIPS: 219 ms per image at batch 1, 298 at 4, 350 at 8 (padding over 96 to 1,739 visual tokens); peak 17.0 GB;
  load 34 s. 34,817 images × 3 factors = about 6.4 GPU h on the 3090 (earlier guess 7.5 to 11 h); on node402's
  three A6000s about 2.5 h of wall time (not measured on the A6000).
- Populations (facts brief §C): ArtELingo held 12,281, validation 6,152; MMArt test 16,384, its validation 5,769
  (a subset of ArtELingo validation). 11,528 is MMArt's shared held subset, not ArtELingo held. Class counts: emotion
  9 and 9; style 27 (ArtELingo held) and 20 (MMArt); genre 10 and 10.
- DAS6 node402: job 64557, about 115 h left at 18:40 (to about 11-08 14:00); all three 48 GB GPUs free at 18:40 (the
  foreign 9.65 GB processes are gone). r19 may also plan to use it.
- Codex: 5 h window 80% left (reset 23:35), weekly 97% left.

## 4. Open points for the next step

- **Key: how the build runs CLAY's, TURTLE's and TEMI's code.** Facts: auto mode refused running TURTLE's cloned
  `run_turtle.py`; CLAY and TURTLE have no licence file, and the CoSiR repo is public, so their code must never be
  committed; TEMI is Apache-2.0. Options to put to the user: pinned clones outside git, imported by our adapters,
  with a permission rule the user adds; or our own re-implementation of the parts we use (CLAY's math is about 40
  lines, TURTLE's loop about 60) with a parity check against the clone. Ask answer-first.
- Minor: r17's cluster counts per aspect (take them from r17's spec).
- Minor: TPIPS on DAS6: the node env's transformers version, the WikiArt files on the node, the 32 GB of weights
  through the HF cache mapping (`/var/scratch/wding/cache/hub` in `cluster.conf`), the `tpips` install. None checked.
- Minor, agent default: CLAY's alignment uses the gallery's mean image, as CLAY's code does (label-free).
- Minor, agent default: Draft B's lists come from r11's generation pattern (Qwen3-VL-8B-Instruct, greedy, in the HF
  cache) with "List 100 distinct values"; a short local GPU job.
- Minor: TPIPS length-sorted batching may beat batch 1; test it in the build's smoke run.
- Glossary: add rows for TPIPS, CLAY, TURTLE and TEMI (field names, so no new names to approve).

## 5. Code and pitfalls

- TPIPS env: `/root/venvs/tpips` (venv over the CoSiR env, `tpips==0.1.2` only); the CoSiR env has no tpips. It lives
  in the container layer; the round log gives the two commands to recreate it.
- TPIPS prints a tokenizer "incorrect regex pattern" warning; expected (TPIPS sets `fix_mistral_regex=False`).
- Clones (this chat's scratchpad, may vanish): re-clone at the pinned commits in the facts brief.
- Earlier round folders stay frozen: import, never edit. Module prefix `om_`. r17's scoring module: import read-only
  after it merges.
- Local GPU: weekdays 10:00 to 19:00 short or light jobs only; overnight runs end by 11:00.
- Report rows: CRL, CLAY and TPIPS sit in a "training images from the domain" column (user, earlier).

## 6. State at handoff

- Running: nothing of r18. Beside it: the r17 and r19 decide chats, and the r18 main chat.
- Uncommitted: nothing.
- On disk over 1 GB: the TPIPS weights and backbone, 32 GB in `/data/SSD2/HF_home/hub` (keep: the TPIPS run needs
  them; the disk-ledger row is due at the wrap-up's storage summary).
