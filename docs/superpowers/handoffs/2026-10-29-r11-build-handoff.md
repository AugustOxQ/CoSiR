# Handoff: r11 build: build the approved retrieval spec on Matt Pocock's chain, unattended

## Brief
- Done: r11's grill, spec and rule review (Opus `xhigh`, pass after fixes); the user approved the spec from its brief.
- State: c* is frozen; every input exists (c*'s held and validation embeddings, all baseline stores); no retrieval code yet.
- Next: tickets, implementation and the final review of the retrieval spec; then open the run chat.
- Blocked: nothing. The Affection extraction holds the GPU from 19:00 to about 20:15.

> Written 2026-10-29 18:35 by "r11 decide 2" (cut at 200k context, after approval). Loop step reached: after step 6
> (spec approved), before step 7 (build). Direction log entry: [2026-10-29](../direction_log.md) (its Idea (B) line).

## 1. Read in this order

1. The spec: [2026-10-29-retrieval-design.md](../specs/2026-10-29-retrieval-design.md). Its decision rule is §6 with
   §3.4 and §3.5; the brief is the user's page.
2. `~/.claude/references/matt-chain.md` (the build chain) and `~/.claude/references/loop-chats.md`, "Unattended
   chats".
3. The facts brief, outside git: `.scratch/r11-retrieval/facts_brief_2026-10-29.md` (Codex, read-only, 2026-10-29).
   It gives the code paths behind every input: splits, stores, the validation files, PercepT's outputs, the r10
   inference and scorer functions.

Skills: `tdd` for the verdict-code tickets (metrics, bootstrap and outcomes, the selection rule).

## 2. Decided by the user, not to be reopened

- The whole protocol, the bar, the baselines, the switch test and the read-out-first optimization: spec §4, D1 to D8
  (grill of 2026-10-29) and D15 (the claim rests on stage 5, 2026-10-29 18:00).
- Spec approved from its brief, 2026-10-29 18:30; no container restart before the build (user).
- Earlier: c* freezes at 11-10; r11 is a medium loop; the plan to the deadlines (direction log, 2026-10-29).

## 3. Where things stand

- Inputs (all checked by the facts brief or the reviewer):
  - c* held: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40/{aspect}__s{0..4}.npz`, key `img`,
    rows in the order of `src/test/20261020_claim_test/outputs/held/held_paintings.json`.
  - c* validation (6,152, sorted names): `.../outputs/full/train/k11=40/R6-5/{emotion__Gres,style__Fres05,genre__R0o}__s{0..4}.npz`, key `img`.
  - Weights: `.../outputs/full/weights/k11=40/{aspect}__s{seed}.pt`; inference `rb_student.infer` on the plain Qwen
    image (facts brief F2). The reviewer's CPU re-inference of seed 0 matched within 6.1e-5.
  - Plain Qwen, CLIP L/14, SigLIP 2: `/data/SSD2/pre_extract/artelingo_full/{qwen3vl_emb_2b,clip_l14,siglip2_so400m}/`,
    all 61,402 paintings, join by `index.json` painting name.
  - Instructed Qwen: `.../qwen3vl_emb_2b_aspect_prompts/{emotion_focus,style_word,genre_word}_held` (held) and
    `..._s2img` (r10's P, includes all validation paintings); join by `painting_names` in `ids.npz`.
  - PercepT core P3: held `src/test/20261020_claim_test/outputs/held/percept/core__k40/P3__s*.npz` (`prob`); validation
    `src/test/20261020_claim_test/outputs/stage2/full/percept/core/stage2/k40/P3__s*.npz` (`eval_prob`).
  - Stage 0b reference values: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/stage4/verdict.json` and
    `rederive_stage4.json` beside it; the values are also in the spec's §3.5.
- CRL: the public code (github.com/XLearning-SCU/2025-NeurIPS-CRL) uses GPT-written value lists and CLIP B/32; spec
  D12 fixes our version. The archived round 6 code (`archive/v2_conditioned/.../r6_dts.py:385`) is a reference only.
- Codex had room at 18:00 (5-hour window 95% left, weekly 76%).

## 4. Open points for the next step

None for the user. Minor items that stay open from the decide chat, not for this build: a literature check on a
multi-task storyline (not decided); the FI images for Affection (the user's call).

## 5. Code and pitfalls

- New round folder with module prefix `rt_` (CoSiR CLAUDE.md, "Pitfalls"). r10's round code is frozen: import it
  read-only (stage 0b calls r10's partition and AMI functions on the new code's matrices; spec §3.5).
- Joins (C6): c*'s and PercepT's npz rows by position to the held list, or to sorted validation names; name-keyed
  stores by painting name. Validation names come from `ct2_data.split2('full')` (facts brief F1).
- Stage 0a commits the seed ranges r_a before any held retrieval number (spec §3.5).
- Unit tests against scikit-learn's AP and nDCG need tie-free inputs; scikit-learn groups ties, our rule keeps gallery
  order (spec §9).
- The GPU (CRL's lists with Qwen3-VL-8B-Instruct): under `flock -o /tmp/gpu0.lock`, not between 19:00 and about 20:15
  tonight. Everything else runs on the CPU at 8 threads.
- The round outputs are small; keep them in the round folder's outputs (gitignored) or under `/data/SSD2/cosir_r11/`.
- In this shell a bare `cluster` is Graphviz; this loop needs no DAS6.

## 6. State at handoff

- Running: `queue_affection_extract.sh` (pid 1545349, nohup, worktree `/project/CoSiR-wt-r11-impressions`): from 19:00
  it extracts `affection_local` with Qwen under the GPU lock (about 75 min); status in that worktree's
  `src/test/20261029_r11_data/queue_status.txt`. After it ends, merge branch `r11-data` into main and remove the
  worktree (the data side job; any chat may do it).
- Uncommitted: nothing in the project. `claude-config`: Codex ledger rows (one from this chat), for `sync.sh` at the
  next wrap-up.
- On disk over 1 GB: nothing new.
