# Handoff: r18 decide: start the TPIPS download, then grill and spec for the outside methods (TPIPS, CLAY, TEMI or TURTLE)

## Brief
- Done: decision A (user): six loops r17 to r22; r17, r18, r19 decided tonight, in parallel.
- State: r18 runs the outside methods: the TPIPS probe then its run, CLAY, and TEMI or TURTLE on the plain image.
- Next: start the 16.5 GB TPIPS download at once, then a short grill and spec; the probe's time goes in the spec.
- Blocked: the DAS6 node's three GPUs each hold 9.65 GB of a process that is not a cluster job (told to the user).

> Written 2026-11-03 18:35 by tab "r16 report 3". Loop step reached: after decision A, at step 3 (medium loop; the
> literature notes cover these methods). Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md),
> bullets "The user's reading of the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The overview's section "The plan (decided 2026-11-03)", near the end of
   [`docs/user_read/2026-11-03_experiment_overview.md`](../../user_read/2026-11-03_experiment_overview.md): loop 2 is
   r18. Its to-do table gives items 3, 13, 14 and 15 with costs.
2. The direction log's r16 entry: the TPIPS and CLAY bullets and the two bullets named above.
3. [`src/test/20261103_experiment_overview/facts_inventory.md`](../../../src/test/20261103_experiment_overview/facts_inventory.md),
   section D (TPIPS, CLAY, TEMI, TURTLE rows and their literature notes).
4. Before the grill: `~/.claude/references/grill.md`. Before the spec: `~/.claude/rules/spec-templates.md`.

Skills: grilling, domain-modeling (as grill-with-docs); cluster-run for the DAS6 node.

## 2. Decided by the user, not to be reopened

- r18's items (2026-11-03 18:30): 13 (TPIPS timing probe) then 14 (TPIPS as a comparator: ArtELingo held and MMArt,
  three aspects), 15 (CLAY on CLIP L/14: ArtELingo held and MMArt), 3 (TEMI or TURTLE on the plain image, four
  encoders). CLAY and TPIPS count for both tasks: the user lists them as missing in clustering and retrieval.
- TPIPS: the released 8B late fusion model, the aspect name as its factor; no 2B (needs our own training); as a
  teacher, parked (user, earlier 2026-11-03).
- CLAY: `github.com/kaist-ami/CLAY`, training-free, no license file; on the list (user, earlier 2026-11-03).
- Paper framing: CRL, CLAY and TPIPS are context rows in a "training images from the domain" column (user, earlier).
- Agent recommendation, seen by the user: the stronger clustering method replaces only the k-means step on the plain
  image, and also runs on c*'s head outputs.
- **Each loop's report states shortly how much it changes the current paper writing: related work, method,
  experiments** (user, 2026-11-03 18:33). Put it in the spec's report section; the draft is
  `docs/paper/cvpr2027_aspect_spaces/drafts/v1/`.

## 3. Where things stand

- TPIPS run time: 7.5 to 11 GPU hours on the 3090 for three aspects, a guess (8B assumed 3 to 4 times slower than
  Qwen 2B's 86 to 95 ms per image); not measured. The probe measures it.
- DAS6: node402, slurm job 64557, about 115 h left at 18:35 (to about 11-08 14:00; not checked whether a new
  reservation follows). Three 48 GB GPUs, each with 9,650 MiB used by a process that is not a cluster job, so
  `launch` picks none. The user was told; check `cluster status --node node402` again, never kill it.
- `cluster.conf` names node401; pass `--node node402` (or ask before changing the shared config).
- Local GPU (3090): free at 18:35.

## 4. Open points for the next step

- Key: CLAY's aspect prompts for emotion, style and genre (ours to write; CLAY ships Stanford40 prompts only). Show
  the user the draft prompts.
- Key: TEMI or TURTLE (one of them), and at which cluster counts (match r17's sweep).
- Key: TPIPS's factor text per aspect, and where it runs (DAS6 or a local night), given the probe's time.
- Minor: CLAY's encoder (CLIP L/14 is in our grid; CLAY's code ships CLIP B/16 and SigLIP base).
- Minor: the read-out protocol: the same as r11 to r15 (picked on validation).

## 5. Code and pitfalls

- **First action:** a separate conda env for TPIPS (it pins transformers and peft; never install it into the CoSiR
  env) and the download of `sywang/TPIPS-Embed-Qwen3VL-8B` (16.5 GB) into `/data/SSD2/HF_home` through the default
  cache link, in a background Bash. Then the grill while it downloads. The probe takes the GPU lock.
- A new round folder `src/test/20261103_<name>/` with r18's own module prefix. Earlier round folders stay frozen:
  import, never edit. Score with r17's scoring module, read-only, after r17 merges (tonight); r18's build starts
  after that merge. Adapters write embeddings into new folders; no loop overwrites another's files.
- Local GPU: on weekdays from 10:00 to 19:00, short or light jobs only; long runs go to DAS6.
- Codex: weekly limit at 5% until 2026-11-04 08:27.
- Open r18's main chat at brief approval if `--new-loop` did not: `loop-next ~/.claude/templates/main_chat_start.md
  --label "r18 main" --main --new-workspace "CoSiR r18" --model sonnet --effort high --notify`.

## 6. State at handoff

- Running: nothing of r18. Beside it: the r17 and r19 decide chats.
- Uncommitted: nothing.
- On disk over 1 GB: nothing new (the TPIPS weights will add 16.5 GB on `/data/SSD2`).
