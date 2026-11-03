# Handoff: r18 build: wait for r17's scorer to merge, then build the outside-methods round (TPIPS, CLAY, TURTLE) from the approved spec

## Brief
- Done: the r18 spec is approved (labels only against c*; Opus rule review, all fixes applied). The official CLAY and
  TURTLE code made the parity references once, attended.
- State: nothing built. r17's scoring code is not merged yet; node402 and the local queue were free at 20:45.
- Next: wait for r17's merge, then tickets, build and the final review; merge to main; cut to the r18 run chat.
- Blocked: nothing (the build waits for r17's merge by design).

> Written 2026-11-03 20:55 by tab "r18 spec". Loop step reached: step 6 done (spec approved 20:45), step 7 next.
> Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The spec, §1 onward: [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md). It
   governs; its gates (§3.5), the margin-source table (§3.4) and the TEMI unit list are binding.
2. The round log's "spec chat" section: [round log](../../../src/test/20261103_outside_methods/20261103_outside_methods_log.md)
   (the parity inputs, the reference driver, the hashes, the `src` name clash).
3. The facts brief, sections A and B for CLAY's math and TURTLE's loop:
   [om_facts_brief.md](../../../src/test/20261103_outside_methods/om_facts_brief.md).
4. r17's spec §3.7 (the scorer's interface), then r17's merged scorer itself:
   [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md).
5. `~/.claude/references/matt-chain.md` (the build chain) and `~/.claude/rules/shared-resources.md` (the queue).

Skills: cluster-run (for TPIPS on node402).

## 2. Decided by the user, not to be reopened

- Everything in the spec's §4 rows marked "user" or "user, after seeing the recommendation" (2026-11-03).
- The code route (D12, 19:45): our own CLAY and TURTLE, checked against the references; TEMI's code copied with its
  Apache-2.0 notice at TEMI's own later build. **The build never runs the clones, and no CLAY or TURTLE code enters
  git** (the repo is public).
- The build starts after r17's scorer merges (D17). TEMI is not part of this build: it is built after TURTLE has run.
- Spec approved 2026-11-03 20:45; no container restart before the build (user, 20:45).

## 3. Where things stand

- Parity references (gates (i), (ii)): `/data/SSD2/cosir_r18/ext/parity/` holds `inputs.npz` (first 2,000 sorted
  ArtELingo validation paintings: unit CLIP L/14 and plain Qwen rows, r11's emotion text set A), `clay_ref.npz`
  (conditioned query and gallery, P, μ; rank 49) and `turtle_ref.npz` (weights after weight-norm removal, bias,
  partition; K 9, T 100, CPU, seed 0, 8 threads). Hashes in the round log. Our tests read these files only.
- Clones at the pinned commits: `/data/SSD2/cosir_r18/ext/{CLAY,turtle,TEMI}` (outside git; read, never run).
- TPIPS: weights and backbone in `/data/SSD2/HF_home/hub` (32 GB); local env `/root/venvs/tpips` (venv over the CoSiR
  env with `tpips==0.1.2`); 219 ms per image per factor at batch 1 on the 3090.
- node402 (2026-11-03 20:45): job 64557, 113 h left (to about 11-08 13:50), GPUs 0 to 2 free, no jobs, no waiting
  launches; the CoSiR env exists on the node (`/var/scratch/wding/conda-envs/CoSiR`), without tpips. Local queue:
  GPU idle, 4 CPU slots free.
- Glossary rows for TPIPS, CLAY, TURTLE and TEMI are in `GLOSSARY.md`.

## 4. Open points for the next step (agent defaults; the build decides and logs them)

- **Waiting for r17:** watch main for r17's build merge (its scorer module in `src/test/20261103_new_scores/`, prefix
  `ns_`). Wait with a Monitor until-loop, not a sleep. If r17's interface differs from its spec §3.7, adapt in our
  round folder only.
- **TPIPS on node402** (spec §5, not checked): the node env needs `tpips==0.1.2` (no deps) and transformers ≥ 5.0;
  the 32 GB of weights through the node's HF cache mapping; the 34,817 images through `DATA_MAP` (ArtELingo from
  `/data/PDD/wikiart_proj/wikiart/`, MMArt from `/data/SSD/mmart/mmart_index.parquet:image_path`). A brand-new
  `DATA_MAP` path must be verified with a real job. If the node cannot be readied unattended, fall back to the local
  3090 at night (about 7 h, ending by 10:00 on a weekday).
- **Smoke checks:** TPIPS length-sorted batching against batch 1 (D20, gate (iv)); one TURTLE fit per width, timed,
  with its power draw (over 200 W means heavy: outside weekdays 10:00 to 19:00).
- **Text set B:** Qwen3-VL-8B-Instruct (in the HF cache), r11's prompt with 100; a short local GPU job.

## 5. Code and pitfalls

- **Queue rule (2026-11-03 20:45):** all local GPU work `res-queue gpu --est <duration> --label r18:<step> -- <cmd>`
  (`--short` under 15 minutes); CPU-heavy steps `res-queue cpu --label r18:<step>`; DAS6 launches
  `cluster launch --queue --est <duration> --label r18:<step> -- <cmd>`. Queue each step, not a whole script; never
  put a script that takes `/tmp/gpu0.lock` itself inside `res-queue`. Subagents pass `--no-wait`.
- **The `cluster` on the PATH is the wrong program** (miniconda's graph-clustering tool). The DAS6 CLI is
  `~/.claude/skills/cluster-run/cluster`; `status` needs `--node node402`.
- The CoSiR env has its own top-level package `src`; a third-party `src` cannot be imported by name.
- TURTLE parity: compare on the CPU with 8 threads (as the reference); GPU runs with TF32 off.
- Earlier round folders are frozen: import, never edit; module prefix `om_`; round folders go on `sys.path`.
- TPIPS prints a tokenizer "incorrect regex pattern" warning: expected.
- Held data: picks and ranges on validation only (spec §3.3).
- r17 (CPU) and r19 (local GPU at night or node402) run beside r18; the queue orders the jobs.

## 6. State at handoff

- Running: nothing of r18. Beside it: the r17 build or decide 2 chat, the r19 spec chat, the main chats.
- Uncommitted: r17's direction-log bullet and other chats' lines in `briefs.md` and `chats.tsv` (not ours; left as
  they are, except the chat records that `loop-next` writes).
- On disk over 1 GB: the TPIPS weights and backbone, 32 GB in `/data/SSD2/HF_home/hub` (keep: the TPIPS run needs
  them; disk-ledger row due at the wrap-up). Small: `/data/SSD2/cosir_r18/ext/` (83 MB: clones and references).
