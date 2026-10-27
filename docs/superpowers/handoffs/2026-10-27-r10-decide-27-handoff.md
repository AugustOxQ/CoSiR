# Handoff: write the spec and decision rule of the staged sweep loop, get the user's approval of its brief, then cut to the build

## Brief
- Done: decision C on report 12-27, the wrap-up, decision A, decision B and the grill of the next loop (closing summary confirmed by the user).
- State: model G is level with the instructed Qwen genre image on held paintings (0.5001 against 0.5062); next loop: held emotion test with G, a staged sweep on DAS6 node402, an ablation and an analysis.
- Next: step 6, the spec and its decision rule from §2; the user approves the brief; then the restart question and the cut to the build.
- Blocked: nothing.

> Written 2026-10-27 03:30 by "r10 decide 26" (cut at the context meter's 200k note, after the grill's closing
> summary). Loop step reached: after decisions C, A and B and the grill (step 5 done), before step 6. Direction log
> entry: [2026-10-27](../direction_log.md) (C, A, B and the grill answers, with who decided).

## 1. Read in this order

1. This handoff: §2 is the content of the spec; nothing in it is to be asked again.
2. The direction log's 2026-10-27 entry (`docs/superpowers/direction_log.md`): the same decisions with their wording.
3. The facts brief `.scratch/r10-decide-26/facts_brief.md` (outside git, 28 KB, a source per fact; written by Codex,
   two facts checked): G's knobs with current values and which were never varied (§A), the earlier ablations (§B)
   and analyses (§C), run costs and the development protocol (§D), what the held emotion test with G needs (§E),
   cluster readiness (§F).
4. The previous spec and rule as the format model: `docs/superpowers/specs/2026-10-26-relational-genre-design.md`,
   `src/test/20261027_relational_genre/DECISION_RULE.md` (stages, gates, outcomes), and report 12-27
   (`docs/reports/auto/percept/2026-12-27_relational_genre.md`).
5. `~/.claude/rules/spec-templates.md` (the spec, its brief, the exploration number sheet) and
   `~/.claude/references/ste-writing.md` (the brief's wording).

Skills: none for the spec. `cluster-run` only if the spec needs a live node fact (read-only `status`).

## 2. Decided by the user, not to be reopened

All on 2026-10-27 in "r10 decide 26"; recorded in the direction log entry of that date.

- **Decision C:** "I also think so, so we now have two levels and one win (emotion), and hopefully we can get all
  three as win (higher-than-noise)." (user, before ours).
- **Decision A:** one medium loop, type sweep: held emotion test with G first, then the sweep with the ablation and
  analysis beside it; the other encoders and retrieval are later loops (user; the order after seeing the
  recommendation). The sweep aims at the strongest recipe for all three aspects; "level" with the teacher is an
  accepted end state (user, after seeing the recommendation).
- **Decision B, a staged sweep (user, own pick):**
  - Stage 1: a one-knob scan around G, about 45 configurations, seed 0, development paintings.
  - Stage 2: a joint grid over the 3 or 4 knobs with the largest mean gain, about 70 configurations, seed 0.
  - Stage 3: the top 3 on fresh seeds 1 to 3.
  - Stage 4: one held test of the pick.
  - The wide random search over all knobs is a later follow-up, only with spare time and idle node GPUs, scored on
    development paintings (never held). It is not part of this spec.
- **Where (Q1):** stages 1 and 2 run on DAS6 node402. Every stage that decides or gives a reported number runs locally:
  stages 0, 3 and 4, the ablation and the analysis. G's reference arm runs on the node as well, and node numbers are
  never compared with local numbers (user chose DAS6; the split after seeing the recommendation; flag in the brief).
- **The pick (Q2):** rank by mean gain in seed-range units over emotion, style and genre (user, option d), with the
  user's own rule that every aspect is at least level with G, in this per-stage form (after seeing the
  recommendation; flag in the brief):
  - stages 1 and 2 (seed 0): an aspect may sit at most half a seed range below G;
  - stage 3 (seeds 1 to 3, local): each aspect level with G in the 12-27 sense (interval lower end above minus one
    seed range), and the mean gain's interval above zero; the highest mean gain wins; if none qualifies, G stays and
    stage 4 is skipped (G's held numbers already exist).
  - Scenario accepted by the user: emotion −0.0010 [−0.0040, +0.0020] against G with style +0.0100 and genre +0.0080
    qualifies.
- **A held win over the teacher (Q3):** the interval of the pick minus the instructed Qwen image is above zero;
  "level" as in 12-27; the size of each win is reported in seed ranges; the paper says "above the teacher (interval
  above zero)" (user, our agreement shown after).
- **Minor m1 to m7, accepted as recommended (user):**
  - m1, stage 0 follows the earlier emotion claim test: both scales, 5 seeds, +0.0100 over PercepT core with the
    interval above zero. It reuses G's full-scale weights and held exports after provenance checks, and adds sample
    seed 4 and held inference for the sample seeds. G became the recipe after its held numbers existed: disclose it.
  - m2, the stage 1 knobs, each varied alone with 2 or 3 values (exact values in the spec's number sheet):
    - the relational term on emotion and on style (never tried);
    - the genre relational weight and teacher temperature; the direct teacher weight;
    - learning rate, weight decay, batch, trunk width, dropout;
    - graph neighbours, buddy temperature, patience.
  - m3, stages 1 and 2 on seed 0; the top 3 on seeds 1 to 3; the pick at full scale, 5 seeds, held.
  - m4, k-means K stays 20.
  - m5, the ablation runs on the final model: development paintings with 3 seeds, and held paintings at full scale
    with 5 seeds for the paper's table. Removals: the relational term, the sharper teacher row, the direct teacher,
    the buddy terms, the caption endpoints, the content correction, instructed graphs to default graphs, and the
    shared trunk to separate heads.
  - m6, the real analysis runs on G now and reruns on the pick:
    - the matrix of each head against each label set;
    - per-class gains over the teacher, and cluster composition with galleries;
    - agreement between the three heads' partitions;
    - geometry (effective rank, hubness, alignment and uniformity);
    - where the student's neighbours differ from the teacher's.
  - m7, the encoders and retrieval are later loops.
- **No plan page** (user, "plain").
- **The schedule the user saw** (estimates, ours):

  | When | Where | What |
  |---|---|---|
  | about 04:30 to 10:30 | build chat | the build, with a node check and a smoke job on node402 |
  | about 10:30 to 14:30 | node402, 3 GPUs | stages 1 and 2 as parallel jobs |
  | about 10:30 to 12:00 | local | stage 0 and the analysis of G |
  | about 14:30 to 16:00 | local | stages 3 and 4 |
  | 19:00 to about 21:30 | local | the ablation, under the weekday rule for long GPU runs |

## 3. Where things stand

- Held, model G (report 12-27), against the instructed Qwen image: emotion +0.0118 [+0.0083, +0.0150], style +0.0032
  [−0.0019, +0.0085], genre −0.0060 [−0.0105, −0.0014]. Against the strongest PercepT entry: style +0.1760, genre
  +0.0732, emotion not computed with the claim test (G 0.1039 against PercepT core's own reading 0.0874).
- Seed ranges (D2, seeds 0 to 2, old 5,000 paintings): emotion 0.0046, style 0.0082, genre 0.0128.
- Costs (facts brief §D): one development run trains in 41 s locally (13,400 fit paintings, scored on 29,232); about
  3 min per configuration with scoring (stage A of 12-27: 7 variants in 23 min); one full-scale seed 114 s.
- DAS6 at 03:05 (live `cluster status`): only node402 is reserved, 3 A6000 free, 93 h 35 min left (until about
  2026-10-31 00:40). node401 and node408 have no srun window any more. DustDevilVis said it uses 2 of the 3 GPUs for
  vLLM for 2 to 3 hours from about 03:15 (user). On 2026-10-20 one A6000 was about 1.8 times slower than the 3090 on
  the ancestor trainer.
- Data for the node: the Qwen stores are 9.1 GB (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b`, 1.5 GB, and
  `.../qwen3vl_emb_2b_aspect_prompts`, 7.6 GB), mapped by `DATA_MAP`, under the 30 GB cap. The round's inputs under
  `/data/SSD2/cosir_r10` (frozen sample, graphs, G exports) are not mapped.
- Parity: on 2026-10-20 the ancestor trainer's AMI on node408 differed from local by 0.00006 to 0.0168
  (`src/test/20261020_claim_test/20261020_claim_test_run_log.md`, 16:34); the cause was never found.

## 4. Open points for the spec (minor, to its "Check these" with our recommendation)

a. The exact knob values of stage 1 and the grid size of stage 2 (the number sheet).
b. Graph neighbours means rebuilding the graphs: its cost decides whether it stays in stage 1.
c. The relational term on emotion and style needs new code (`rg_loss.py` applies it to genre only).
d. Stage 0's scorer: `rb_score.py` takes K2 to K5 only; an adapter for G and a sample seed 4 are needed (facts brief §E).
e. The node route: a `scripts/run_*.sh` runner for the sweep shards (the CLI launches only those or `main_cosir.py`);
   the round's inputs via a new `DATA_MAP` entry (shared `cluster.conf`: commit only our change, and verify a new
   path with a real job) or `INPUTS_REMOTE` with a sync step; a node check, a smoke job and a parity reading of G's
   reference arm against local (recorded, not a gate, since no comparison crosses machines).
f. From the 12-27 handoff: the rule text's batch size is 512 pairs (1,024 rows); the full-scale held guard should hash
   the frozen `ms_train` and `ct` modules.

Before the build chat opens: the restart question (`~/.claude/references/loop-chats.md`, "How to cut", step 3).

## 5. Code and pitfalls

- A new round folder needs a new module prefix (project CLAUDE.md). Do not rerun any 12-27 queue; read its outputs.
- `cluster sync` refuses uncommitted code under `src configs scripts main_cosir.py`; round code must be committed.
- Long local GPU runs: none on weekdays 10:00 to 19:00; local overnight work ends by 11:00 (`shared-resources.md`).
- The development split and seeds as in 12-27 (facts brief §D); held data chooses nothing.

## 6. State at handoff

- Running: nothing (no job, subagent, Codex job or GPU lock of this chat).
- Uncommitted: nothing in CoSiR after the cut commit. `/project/claude-config`: the CoSiR memory pointer and one Codex
  ledger row, committed by `sync.sh` at the cut.
- On disk over 1 GB: the 12-27 round's outputs, 11 GB on `/data/SSD2` (ledger rows of 2026-10-27).
