# Handoff: r18 build 2: wait for r17's scorer to merge, then tickets, build and the final review (same task as build 1)

## Brief
- Done: build 1 read the spec and the code map and drafted the ticket plan (§3). It cut at once: its context passed 200k.
- State: nothing built. r17's scorer is not merged (r17 build chat started 20:46). GPU check passes.
- Next: wait for r17's merge (background until-loop, §5), then write the tickets from §3, build, final review, merge.
- Blocked: nothing (the build waits for r17's merge by design, D17).

> Written 2026-11-03 20:53 by tab "r18 build". Loop step reached: step 6 done, step 7 not started.
> Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The build 1 handoff, all of it: [2026-11-03-r18-build-handoff.md](2026-11-03-r18-build-handoff.md). It still
   governs; this file only adds build 1's notes.
2. The spec from §3.2 to §6: [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md).
3. r17's saved-outputs inventory, the code map for every reused input:
   [ns_saved_outputs.md](../../../src/test/20261103_new_scores/ns_saved_outputs.md).
4. r15's ticket index and common briefs, the model for ours: `/project/CoSiR/.scratch/r15-mmart-test/issues/00-index.md`,
   `.../notes/implementer_brief.md`, `.../notes/reviewer_brief.md` (outside git).
5. Only when a ticket needs it: the facts brief's sections A and B; `~/.claude/skills/cluster-run/SKILL.md` (ticket 04).

Keep the context small: build 1 passed 200k on reads alone. Read the spec's §3 to §6, not its brief, and give
subagents paths, not pasted text.

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 20:45, no container restart).

## 3. Where things stand: the ticket plan (agent default, build 1; adjust to r17's merged interface)

Round folder `src/test/20261103_outside_methods/`, prefix `om_`; integration branch `r18-outside-methods`; tickets in
`.scratch/r18-outside-methods/issues/`. The controller writes `om_paths.py` (outputs root D18 with an `OM_OUTPUTS`
override for smoke, `OM_SMOKE`, records, encoders, sets, aspects, class counts, r17's counts, ext and parity paths,
`read_committed_record` as r15's) before the first dispatch.

| # | Ticket | Blocked by | Spec IDs | Verdict |
|---|---|---|---|---|
| 01 | CLAY math `om_clay.py`: subspace (P, μ) at rank n−1 or min(50, n−1), two Householder maps through the midpoint, log map, P, normalize; mean over the scored split; set A from r11's `crl/<aspect>_text.npy` | none | §3.4 CLAY, gate (i), D4, D6 | yes |
| 02 | TURTLE loop `om_turtle.py`: code defaults, official RNG order (weight-norm task encoder and Adam, then `init_inner`, cold start each step, `np.random.choice` draws), data on the GPU, TF32 off, no label | none | §3.4 TURTLE, gate (ii), D9, D21 | yes |
| 03 | TPIPS extractor `om_tpips.py` and gate (iv): stores per D18 (float32 unit rows, 4,096 wide, keys once each), shards, length-sorted batches against batch 1 (D20) | none | D2, D18, D20, gate (iv) | no (asserts) |
| 04 | TPIPS node route: r14's ticket 10 pattern (`git show 981fa97`, `c800ed8`): node inputs list, run script, weights by `cluster hf-fetch` (both repos, pinned snapshots), images through `DATA_MAP`, a probe job for transformers ≥ 5.0 and peft ≥ 0.19 | 03 | D3, §5 | no |
| 05 | Text set B `om_textset_b.py`: r11's `rt_crl` prompt with 100, greedy, dedup and fallback per D5, CLIP L/14 text vectors of "a painting of {value}" | none | D5 | no (asserts) |
| 06 | Loaders `om_data.py`: the four sets in r17's keys and orders; plain, c*, TPIPS (MMArt validation from the ArtELingo validation store by name), CLAY vectors; labels | r17, 01 | §3.3, C6 | yes |
| 07 | Replays, gate (iii) a, b, c | 06 | gate (iii) | yes |
| 08 | Stage 1 `om_stage1.py`: Qwen ArtELingo AMI-at-20 range; the r source table; CLAY text-set picks (D7, D8); TPIPS and CLAY read-out picks (menu clipped to rank) | 06 | §3.4, D7, D8, D14 | yes |
| 09 | TURTLE runner: the 412 fits, partitions with sidecars (loss, entropy, clusters, seconds); timing smoke per width with power draw | 02, 06 | D9, D21, §3.6 fits | no |
| 10 | Cells and labels `om_cells.py`: SC1 to SC6 through r17's paired cell (partition input), r from stage 1, voids per §3.5's table, a C13 guard as r15's `verdicts` step | 06, 08 | §6, §3.4, §3.5, D11, D13, D24 | yes |
| 11 | Stage runner, records, CPU queue script, end-to-end smoke | all | §3.5 stages, §3.7 | no |
| 12 | Diagnostics `om_diag.py` | 06, 09, 10 | §3.6, D22 | no |

Coverage (hard rule 6): SC1 06, 10 · SC2 06, 08, 10 · SC3 01, 06, 08, 10 · SC4 01, 05, 06, 08, 10 · SC5, SC6 02, 09,
10 · SC7, SC8 and D10: TEMI's later build · D2 03 · D3 04 · D4 01, 05, 08 · D5 05 · D6 01 · D7, D8 08 · D9 02, 09 ·
D11, D13, D15 10 · D12 01, 02 · D14 08, 10 · D16 11 · D18 `om_paths` · D20 03 · D21 09 · D22 12 · D23 01, 02, 03, 07 ·
D24 10, 11 · D19, D25: the run chat's report · C13: the run chat (Codex), from the saved outputs.

## 4. Open points for the next step (agent defaults)

- As build 1's §4. Build 1 found: `DATA_MAP` already maps `/var/scratch/wding/Dataset/wikiart_proj` (ArtELingo and
  MMArt images are WikiArt files) and the node HF hub (`/var/scratch/wding/cache/hub`); `DATA_MAX_GB` is 30, so the
  16.4 GB weights go by `hf-fetch` on the head node, not by sync. Launchable forms are `python <ENTRY> …` and
  `bash scripts/run_*.sh …` only.
- Codex at 20:50: five-hour window 38% left (reset 23:35), weekly 90%. `codex-usage --room` kinds: figure,
  rederivation, fidelity, mechanical, tool, review, bug, report, implementer, other.

## 5. Code and pitfalls

- As build 1's §5.
- **The wait:** run in a background Bash (one notification):
  `until git -C /project/CoSiR ls-tree -r --name-only main -- src/test/20261103_new_scores/ | grep -v review_check | grep -q 'ns_.*\.py$'; do sleep 60; done`.
  r17 moves main by `git update-ref` at its end, so new `ns_` Python files on main mean the merge.
- TURTLE reference facts: the official run saves weights after `remove_weight_norm`; its every-20-steps evaluation
  draws no random numbers; it ran at about 37 steps per second on the 2,000 rows (CPU).

## 6. State at handoff

- Running: nothing of r18 (build 1's background wait was stopped).
- Uncommitted: other chats' lines in `briefs.md` and `chats.tsv` (not ours).
- On disk over 1 GB: as build 1's §6 (TPIPS weights, 32 GB, keep).
