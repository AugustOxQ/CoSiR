# Handoff: build the recipe sweep loop's code and gates from the approved spec and rule, then cut to the run chat

## Brief
- Done: the spec and decision rule of the recipe sweep loop; the user approved the brief at 03:48; an Opus `xhigh` review passed after fixes (all applied).
- State: model G is the reference (held genre 0.5001 against the instructed Qwen image's 0.5062); no new number exists.
- Next: the build per matt-chain.md (tickets, implement, final review), with gates V0a, V1a, V3a, V4a and a node check and smoke job on node402.
- Blocked: nothing. Night mode: do not wait for the user.

> Written 2026-10-27 04:01 by "r10 decide 27". Loop step reached: step 6 done (spec approved), before step 7.
> Direction log entry: [2026-10-27](../direction_log.md) (its links name the spec and rule).

## 1. Read in this order

1. The spec, from §1 on (skip the brief): `docs/superpowers/specs/2026-10-27-recipe-sweep-design.md` (the number sheet
   k1 to k13 is in §3.2).
2. The binding rule: `src/test/20261027_recipe_sweep/DECISION_RULE.md` @ 35fcf68 (the header line was filled in a
   later commit; the content is the same). Where it and the spec differ, the rule governs.
3. `~/.claude/references/matt-chain.md` (the build chain), and `~/.claude/references/codex-jobs.md` before any Codex job.
4. The facts brief `.scratch/r10-decide-26/facts_brief.md` (outside git): G's knobs with code locations (§A), costs
   (§D), what stage 0 needs (§E), cluster readiness (§F).
5. The 12-27 round's code as the base: `src/test/20261027_relational_genre/` (`rg_train.py`, `rg_full.py`,
   `rg_loss.py`, `rg_score.py`, `rg_held.py`, `rg_paths.py`) and the stage 0 scorer to adapt,
   `src/test/20261025_recipe_backbones/rb_score.py`.

Skills: `to-tickets` and `implement-spec` (the chain); `cluster-run` for the node check, the smoke job and the new data
mapping.

## 2. Decided by the user, not to be reopened

- Decisions A, B and C and the grill: the decide 26 handoff §2 and the spec §4 (D1 to D16).
- The brief approved as written (2026-10-27 03:48, this chat). The user said to apply the review's fixes unless one
  changed a user decision or a bar; none did (spec amendment A1).
- No container restart before the build (03:48; the GPU check passed: driver 595.99.02 everywhere, container started
  2026-10-25 18:44). Run mode set to night by the user at 03:47.

## 3. Where things stand

- Nothing of this round exists yet beyond the rule. The round folder is `src/test/20261027_recipe_sweep/` (prefix
  `rs_`); outputs go to `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/`, linked as the folder's `outputs/` as in
  12-27.
- Schedule shown to the user: build until about 10:30; then the run chat: stage 1 on node402 (3 GPUs) and stage 0
  locally from about 10:30; stage 2 to about 14:30; stages 3 and 4 locally to about 16:00; the development ablation by
  day; the held ablation from 19:00 to about 23:00; reports by 2026-10-28 03:00.
- DAS6 at 03:05: only node402 reserved, 3 A6000 free, until about 2026-10-31 00:40. DustDevilVis said it uses 2 of the
  3 GPUs for vLLM for 2 to 3 hours from about 03:15.

## 4. Open points for the build (agent defaults, no user question)

a. The node's inputs (rule §4.2, spec D19): one small folder (the frozen sample, its graphs and the k11 graph
   variants, P's list and labels) under a new `DATA_MAP` entry in the shared `cluster.conf` (commit only our line;
   verify the new path with a real job), or `INPUTS_REMOTE` with a sync step if the CLI prefers it. The Qwen stores
   are mapped already.
b. Trainer switches (rule §5, "New"): k1 and k2 (the relational term on emotion and style), k12 (the buddy
   temperature apart from the direct and relational ones), k11 (`ct2_graphs._mutual` hard-codes 20 in every mutual
   top-k), k8 (training and stopping-loss batches), k9 and k10 (trunk width and dropout are hard-coded in `ms_train`:
   patch them), and the eight removal switches with unit tests.
c. The stage 0 adapter: `rb_score.py` takes K2 to K5 only; G needs its own candidate entry, and the development CLI
   allows seeds 0 to 3 only (seed 4 is needed).
d. The gates that run in the build: V0a, V1a, V3a, V4a (rule), plus N1 and N2 on node402. Gate files are kept, with
   code hashes, so the run chat's queue reuses them as 12-27 did.
e. The run chat's queue scripts: one per GPU stage locally (under `flock -o -E 75 /tmp/gpu0.lock`), and node shards
   launched one GPU each with the `cluster` CLI (it launches `scripts/run_*.sh` runners or `main_cosir.py` only).

## 5. Code and pitfalls

- A new round needs its own module prefix (`rs_`). Do not rerun any 12-27 queue; read its outputs.
- Frozen outputs are gitignored and exist only in the main checkout (`rg_paths.py`'s note): never read them through a
  worktree's copy, and never pass one as an output folder.
- `cluster sync` refuses uncommitted code under `src configs scripts main_cosir.py`: commit round code first. Cluster
  work only through the `cluster` CLI; never `--allow-concurrent`; never kill another project's job.
- The rule's earlier batch text: 512 pairs (1,024 rows) is right; the new rule says so.
- Weekday rule: no long heavy local GPU run between 10:00 and 19:00; overnight local work ends by 11:00. The build's
  gates are short.
- Night mode (`run-mode`): the build and run chats never wait for the user; delete nothing (items go on
  `.scratch/pending_deletions.md`); a blocked step is done another way or noted for the morning (`loop-chats.md`).

## 6. State at handoff

- Running: nothing (the review worker finished; no job, Codex job or GPU lock of this chat).
- Uncommitted: nothing after the cut commit.
- On disk over 1 GB: nothing new from this chat.
