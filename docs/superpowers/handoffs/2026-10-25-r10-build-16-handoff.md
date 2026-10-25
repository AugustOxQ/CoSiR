# Handoff: build the recipe pick and the backbone claim tests (tickets, build, final review, merge), then open the run chat

## Brief
- Done: the spec and decision rule of decide 17's two-part loop, reviewed (rule: Opus `xhigh`; cold check: Sonnet `high`) and committed.
- State: part 1 picks one of six recipes on 29,232 fresh development paintings; part 2 tests SigLIP 2 and CLIP B/32 students against PercepT on held emotion.
- Next: tickets from the spec and rule, the build, the final review, merge; then open "r10 run 14".
- Blocked: nothing. Night mode; no container restart (user).

> Written 2026-10-25 03:30 by "r10 decide 19". Loop step reached: step 6 (the spec) done, step 7 (build) next.
> Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The spec, [2026-10-25-recipe-and-backbones-design.md](../specs/2026-10-25-recipe-and-backbones-design.md), from
   §1, and its binding rule, `src/test/20261025_recipe_backbones/DECISION_RULE.md` (it governs where they differ).
3. `~/.claude/references/matt-chain.md` (tickets, build, reviews, merge) and `~/.claude/references/codex-jobs.md`.
4. The code it reuses: the ablation round `src/test/20261021_d2_ablation/` (`ab_rows.py`, `ab_train.py`,
   `ab_score.py`), the claim test round `src/test/20261020_claim_test/` (`ct2_data.py`, `ct2_d2.py`,
   `ct2_percept.py`, `ct2_verdict.py`, `ct_score.py`), the model sheet `src/test/20261019_model_sheet/ms_train.py`,
   PercepT core `src/test/20261018_percept_core/`. The hard-wired widths and paths: `.scratch/r10-decide-16/f2_facts.md`
   §5.

Skills: none (`to-tickets` and `implement-spec` per `matt-chain.md`, read from their SKILL.md).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D19, D30) and the decide 19 handoff §2: the six recipes and the guarded rule; the 29,232
fresh paintings; SigLIP 2, then CLIP B/32, DINOv2 only if image endpoints only wins; Qwen captions, graphs and teacher
for every student; the bar (the strongest PercepT core entry, the matched arm included); held, the sample scale first,
the full scale unless S1-NO; ArtELingo-28 as a test set plus a descriptive culture split; local 3090. Approved through
the closing summary (2026-10-25 03:02). Night mode; no restart.

## 3. Where things stand

- Nothing of the round is built. The round folder holds only the rule. Module prefix `rb_` (D33).
- Regression targets: 12-22's seed-0 exports `src/test/20261021_d2_ablation/outputs/train/{R0,R6,R7}/*__s0.npz`
  (stage 0); the claim test's K1 and B1 seed-0 held outputs at both scales, its saved partitions and its verdict
  values, `src/test/20261020_claim_test/outputs/{held,stage2/full/held}/` and `verdict_stage{1,2}.json` (stages 2,
  3a).
- Saved claim test artifacts to reuse (frozen): Stage 1 topics `outputs/percept/core/stage1/qwen_qwen__s*` (and
  `outputs/stage2/full/percept/core/stage1/`); R-COL records `outputs/percept/rcol_stage1.json` and
  `outputs/stage2/full/percept/rcol.json`; held outputs of core P1 to P4 and the paper versions; D2 weights
  `outputs/d2/weights`, `outputs/stage2/full/d2/weights`; the 42,969-node graphs `outputs/stage2/full/d2/graphs`.
  Held list hash `a0b1139c…a876` (rule §3.2).
- Stores (`/data/SSD2/pre_extract/`): `artelingo_full/siglip2_so400m` (all 61,402 paintings); Qwen default and the
  instructed images `artelingo_full/qwen3vl_emb_2b_aspect_prompts/*{,_trainimg,_s2img,_held}`; ArtELingo-28
  `artelingo28/{qwen3vl_emb_2b,siglip2_so400m,clip_b32}` and `read_set_a28.parquet` (`row_id`, not `sample_id`). The
  new CLIP B/32 store needs an allow-list line in `scripts/extract_features.py` (about line 60). DINOv2-large weights
  are cached; no encoder code.
- Timings: D2 at the sample scale about 35 s per run; at full scale 598 s for 5 seeds; the claim test's CPU queue 43
  to 78 min; Qwen 11.5 images/s, SigLIP 2 74.7, CLIP 106.
- Genre labels on the 29,232: not counted (the WikiArt CSVs are under `/data/SSD/wikiart_genre/`); stage 0 counts
  them.

## 4. Open points for the next step (agent work; nothing for the user)

a. **Tickets** (`to-tickets`), each citing the spec's D IDs and the rule's IDs, with the coverage line (hard rule 6).
b. **Verdict code** gets ticket reviews and unit tests (`matt-chain.md`): part 1's pick (rule §3.1, §4.1), part 2's
   scorer and bootstrap (§3.2), the ArtELingo-28 labels and reading (§3.3, §4.3).
c. **The re-derivations** (VP1, V2) are separate code written from the rule alone, by an implementer who has not
   seen the main scorer (C13). The run chat runs them before any pick or outcome is stated.
d. **Smoke runs** of part 1 and part 2 on the GPU under `flock -o -w … /tmp/gpu0.lock`; outputs in the scratchpad.
e. **Final whole-branch review** (Opus `xhigh`), one fix wave, a scoped re-review; merge into main and push.
f. **The run chat's handoff**: the queue order (spec §3.5), commands, and the end estimate against 11:00. Sunday is a
   weekend day: if part 2 ends after 11:00, the run chat moves it (for example part 1 in the morning, part 2 from
   the evening) and records why (D32). Then `loop-next <handoff> --label "r10 run 14" --unattended`.

## 5. Code and pitfalls

- Change no code in `src/test/20261021_d2_ablation/` or `src/test/20261020_claim_test/`: their `*_row.json` and
  verdict files bind code hashes. Import from them or copy into the new round.
- Every D2 and PercepT core path asserts 2048-wide inputs (f2 §5). Open only the image network's first layer (and
  the matched arm's PercepT Stage 2 input); the teacher and the outputs stay 2048.
- Joins are positional (C6): IDs from extraction, never `range(n)`. ArtELingo-28 uses `row_id`, our stores
  `sample_id`.
- `held-data.md`: part 1 reads development paintings only; nothing chooses on held or ArtELingo-28 numbers.
- New stores go only under `/data/SSD2/pre_extract/<dataset>/<backbone>/`; never overwrite an existing store.
- CPU work at 8 threads or fewer (`OMP_NUM_THREADS=8`); the bootstrap AMI calls dominate the CPU time.
- A new round folder goes on `sys.path`: keep the `rb_` prefix on every module.

## 6. State at handoff

- Running: nothing of this chat. Run mode: night.
- Uncommitted: nothing after this commit (`.scratch/` is gitignored).
- On disk over 1 GB: nothing new.
