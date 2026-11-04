# r18 outside methods: round log

Loop r18 (medium): TPIPS, CLAY, TURTLE then TEMI as comparators on ArtELingo held and MMArt. Module prefix `om_`.
Decide chat: [handoff in](../../../docs/superpowers/handoffs/2026-11-03-r18-decide-handoff.md).

## 2026-11-03, decide chat

**TPIPS download and env (18:35 to 18:42).**
- `sywang/TPIPS-Embed-Qwen3VL-8B` (snapshot `86a1493c`, `model.pt` 16.4 GB, a full 8B state) and its backbone
  `Qwen/Qwen3-VL-Embedding-8B` (snapshot `2c456551`, 16.3 GB) are in `/data/SSD2/HF_home/hub`. The loader builds the
  backbone with `from_pretrained` and then swaps in the full TPIPS state, so both are needed (32 GB, not the 16.5 GB
  first planned). Download log: `logs/tpips_download.log` (gitignored).
- Env: `/root/venvs/tpips`, a venv with `--system-site-packages` over the CoSiR env, holding only `tpips==0.1.2`
  (`--no-deps`). The CoSiR env already meets TPIPS's needs (transformers 5.6.2 ≥ 5.0, peft 0.19.1 ≥ 0.19, torch
  2.11 ≥ 2.7; no exact pins). The CoSiR env stays unchanged. The venv is in the container layer: after a container
  rebuild, recreate it with `/root/miniconda3/envs/CoSiR/bin/python -m venv --system-site-packages /root/venvs/tpips`
  and `/root/venvs/tpips/bin/pip install --no-deps tpips==0.1.2`.

**TPIPS timing probe (item 13), `om_tpips_probe.py`, local RTX 3090 under the GPU lock.**
- 48 MMArt new paintings (first by key), factor "emotion", images preloaded, after one warm-up call.
- ms per image: 219.4 at batch 1, 297.7 at batch 4, 349.6 at batch 8. Batching is slower: visual tokens vary from 96
  to 1,739 per image (mean 436; TPIPS's max_pixels is 1800×32×32), so padding costs. Length-sorted batching: not
  tested.
- Peak GPU memory 17.0 GB at batch 1 (20.4 GB at 8). Load 34 s. SDPA attention (no FlashAttention). Width 4,096.
- Factor check, image 0: cosine emotion|style 0.769, emotion|genre 0.758, style|genre 0.869.
- Estimate: 34,817 images (ArtELingo held 12,281, ArtELingo validation 6,152, MMArt test 16,384) × 3 factors =
  104,451 passes × 219 ms = about 6.4 GPU h on the 3090, image loading not counted. Earlier guess: 7.5 to 11 h.
- Expected warning: transformers says the TPIPS tokenizer has "an incorrect regex pattern"; TPIPS passes
  `fix_mistral_regex=False` on purpose (its code says it uses a Qwen3-VL tokenizer).
- Record: `records/tpips_probe.json`; log `logs/tpips_probe.log` (gitignored).

**Facts brief** (Codex `gpt-6.1-sol` medium, about 9 min): [om_facts_brief.md](om_facts_brief.md). CLAY, TEMI and
TURTLE code at pinned commits (CLAY `a3f1890`, TEMI `c6a81a6`, TURTLE `4674efa`), our populations and class counts,
r11 to r15's read-out and clustering protocol, CRL's adapter. Spot checks by Claude: CLAY's `main.py` and math in
`src/utils.py`, TURTLE's loop, CRL's lists (`/data/SSD2/cosir_r11/20261029_retrieval/outputs/crl/lists.json`), the
counts in `mt_paths.py`. No wrong fact found.

**A refused step.** A TURTLE timing run (its cloned `run_turtle.py` on 12,281 Qwen rows) was refused by the auto-mode
permission check, since it ran code cloned from GitHub. Not needed for the grill; the scratch data (193 MB) was
deleted. The build must settle how it runs CLAY's, TURTLE's and TEMI's code (handoff to the spec chat, §4).

**Grill (decision B), 19:00 to 19:20.**
- Key 1, TEMI or TURTLE: both, in order; TURTLE (the cheaper one) built and run first, TEMI built after it and run
  later (user's own; our recommendation, shown after, was TURTLE only). Added and confirmed: TEMI's scope is set at
  its build by a short timing run; TEMI results after 11-16 go to the supplementary. Scenario confirmed: if TURTLE
  closes c*'s emotion gap on the plain image and TEMI does not, all rows are reported and the claim is phrased per
  method.
- Key 2, CLAY's texts: both text sets, picked on validation (user's own, after a cost table; ours the same).
  Draft A: CRL's 50 values per aspect (r11, Qwen3-VL-8B-Instruct), template "a painting with the {aspect} of
  {value}", subspace rank 49. Draft B: 100 values per aspect from the same LLM and prompt with 100, template "a
  painting of {value}", rank 50 (CLAY's code value). Scenario confirmed: label-name matches are listed, not removed.
- Minor i to vii accepted (user, after seeing the recommendation): TPIPS factors "emotion", "style", "genre"; TPIPS
  on DAS6 node402, one third of the images per GPU (local 3090 at night as the fallback); the 34,817 images above;
  r11 to r15's read-out menu picked on validation, one pick per method across aspects; k-means on each aspect's
  CLAY and TPIPS embedding at r17's counts; TURTLE or TEMI fitted on the scored set, paper defaults, plain image of
  four encoders × two datasets × r17's counts × three starts, c*'s heads at the class count and 20 with one start
  per seed; WIN, LEVEL or BELOW against c* with no gate.
- No plan page (user). Closing summary confirmed (user): the handoff's §2.

## 2026-11-03, spec chat

**Code route (key point left by the grill), 19:30 to 19:45.** Options put answer-first: a. clones outside git under
a standing permission rule; b. re-implement all three, checked against the clones; c. re-implement CLAY and TURTLE,
TEMI later. User's first pick a (no reason given). After our recommendation (b), a cost table (CLAY about 1 h,
TURTLE 1 to 2 h, TEMI about half a day), the risks of a (the rule, torch 2.11, TURTLE's CPU batch copy, repo-only
reproduction) and the licence facts (no licence: no copy in the public repo, credit line or not; TEMI Apache-2.0:
copy allowed with its notice), final pick **f**: CLAY's math and TURTLE's loop re-implemented and checked; TEMI's code
copied with its notice; no request to the authors (user, after seeing the recommendation). Spec D12.

**Clones** copied to `/data/SSD2/cosir_r18/ext/{CLAY,turtle,TEMI}` (outside git, 15 MB), clean at the pinned commits
a3f1890, 4674efa, c6a81a6.

**Parity references (gates (i), (ii)), run once by the official code, attended, with the user's yes (19:58).**
- Inputs: `om_parity_inputs.py` (ours) writes `/data/SSD2/cosir_r18/ext/parity/inputs.npz`: the first 2,000 sorted
  ArtELingo validation paintings, unit CLIP L/14 and plain Qwen rows, r11's emotion text vectors (text set A, 50).
  sha256 7b10bf69d6fe0358b7b00d98ef974a2d486a73a74f4faa384cc75dd219e4876c.
- Script: `/data/SSD2/cosir_r18/ext/parity/ref_official.py` (outside git: it mirrors CLAY's main block), sha256
  2c3e9df9b3de20ecb89bbbbb68124260d74b385768b195b07807c5771cdc2047; `python -I`, CPU, 8 threads.
- CLAY: `make_subspace` with r 49 on the text vectors (a stub model returns them), then main.py's alignment by the
  gallery mean, log map and projection; query = gallery = the 2,000 rows. `clay_ref.npz` sha256
  3bd542a01e9898dceb4f198c99a4240fc87f81693f089f91c6d0554104c8771a. First try failed: the env's own `src` package
  hid CLAY's `src`; fixed by pointing the name at CLAY's folder.
- TURTLE: `run_turtle.run`, K 9 (through its `datasets_to_c`), T 100, CPU, seed 0, train = val = the Qwen rows, zero
  labels for its printed accuracy; weights after its weight-norm removal and the argmax partition. 9 of 9 clusters.
  `turtle_ref.npz` sha256 d73efb37269c50711356e642c0fb78fd65167d588636e2d704cb2353f327a4b1.
- The build's parity tests read these files and must use 8 CPU threads for the CPU comparison.

**Glossary:** rows for TPIPS, CLAY, TURTLE and TEMI (field names).

## 2026-11-04, build chat 2

Tickets in `.scratch/r18-outside-methods/` (outside git); the list goes into the build's handoff to the run chat.

- **r17 merged** at 8cf743b2 (00:20); its signatures match the planned ones.
- **SC5's candidate form (agent default):** `ns_clu._boot` asserts a candidate of [5, 3, n] and r17's folder is
  read-only, so ticket 10 repeats X-c* + TURTLE's one start three times per seed. The mean over states and the draws
  are unchanged.
- **Per-metric words (agent default):** `paired_cell` takes `r` for every metric or none, so cells call it with None
  and use `ns_outcome.word_of(lo, r)` per metric; `not WIN` with a missing range is this round's `inconclusive`.
- **Codex** had no room for an implementer at 00:25 (77% left, 80% needed, reset 04:35): Claude took the first wave.
- **Gate (i) reach (Opus review of ticket 01):** at rank 49 of 50, P equals the projector on span(T) ∩ μ⊥, so the
  reference cannot see centring, angle scaling or singular-vector order in `subspace`. Set B's truncated path (rank 50
  of about 99) has no official reference; it rests on the unit test and a float32 cross-check. On MMArt test and
  ArtELingo held, float32 and our float64 math differ by at most 9.15e-7, with identical k-means partitions.

## Build 3 (2026-11-04, unattended)

- **Codex** had no room at 00:33 (64% left, 80% needed, reset 04:35): Claude workers again.
- **Ticket 02 review (Opus `high`): CHANGES NEEDED, then fixed.** `fit`'s math and order were right, but a gate at 100
  steps on 2,000 rows cannot see batch draws (the batch is a permutation of all rows), and `batch_indices` was a twin
  of `fit`'s draw, not the same code. Three sabotages survived the 12 tests. Fix: one generator `_batch_draws`, an
  oracle test on real subsets (n 12,281), and a code hash in each part of `gate_ii.json`. `fit`'s numbers did not move
  (old against new difference 0.0); Sonnet `high` re-review approved. Gate (ii) CPU part: 2.54e-7 against 1e-5.
- **Text set B generated** (frozen, 00:40, 5.8 min): emotion 80 values (`short`, the reply loops on "Skepticism"),
  style 100, genre 100; all `capped`; no header lines kept. Rank 50 for each. **43 of the 100 style values are
  "gothic X" drift** from the greedy reply. D5 keeps them (first 100 after dedup; no fallback at 2 or more values), so
  set B on style is weaker than its count says. Ticket 08's dev-split pick decides between sets A and B.
- **D20 settled: batch 1.** Probe on 48 images, 3 factors, length-sorted: batch 1 46 s, batch 4 39 s, batch 8 41 s.
  Against batch 1, the lowest row cosine is 0.9981 and 15 to 23 of 48 rows fall below gate (iv)'s 0.999 per factor.
  Batches are 15% faster and fail the gate.
- **TPIPS timing is not settled.** The probe set is lighter than the real sets (mean 436 visual tokens against 723).
  The node's gate probe gives the node's time on the same 48 images.
- **Node route (agent default):** no new env. The job runs the node's CoSiR env with tpips 0.1.2 and peft 0.19.1 from
  `node_inputs/tpips_pylibs` on PYTHONPATH (pure Python; nothing installed). A second CLI profile cannot build an env:
  `load_profiles` refuses a shared `REPO_URL`. The node's versions differ from the local ones (torch 2.14 against
  2.11, transformers 5.16.1 against 5.6.2 when last read); gate (iv) compares its vectors with the local batch-1 run.
  Two `DATA_MAP` pairs added to `cluster.conf`. The dry-run line must give node paths, not local ones.

## Build 4 (2026-11-04, unattended)

**Gate (iv) on node402 fails; the stores come from the local 3090.** The route was: the probe failed offline, since
`hf-fetch --revision` writes no `refs/main` and tpips loads the backbone by repo name (fixed by a job-local hub with the ref
pinned). The gate probe then ran. All numbers are the lowest row cosine against the local batch-1 reference, in the order
emotion, style, genre; the gate needs 0.999.

| Run | Lowest row cosine | Median or mean | Reading |
|---|---|---|---|
| Local rerun, stock packages | 1.000000 all three | median 1.000000 | no noise locally |
| Node, its own packages (transformers 5.16.1, torch 2.14) | 0.997342, 0.996178, 0.989027 | median about 0.9988 | fails |
| Local, transformers 5.16.1 in place of 5.6.2 | 0.997342, 0.996178, 0.989027 | median about 0.9988 | transformers is the main cause: it reproduces the node to six digits |
| Node against local 5.16.1 | 0.998936, 0.998810, 0.998623 | median 1.000000 | 13 of 48 rows off, the rest exact |
| Node, exact copy of the local packages | 0.998909, 0.998737, 0.998698 | mean 0.9998 | fails narrowly |

The 48 images are byte-identical on the node (sha256). Versions were checked in the job. So the last gap is the hardware
(RTX A6000 and its CPU against the 3090 and a Ryzen 5950X); which part was not checked. The off rows are not simply the
large images: they overlap the exact rows in token count. The threshold stays 0.999, as the spec sets it. Two side facts for
the report: TPIPS vectors move with the transformers version (median row cosine 0.9988 between 5.6.2 and 5.16.1), and a node
rerun skips finished chunks, so two runs in this list first looked like results and were not (a run with the same six digits
as the last and no `probe shard` line).

**Local stores.** `om_tpips_local.sh`, 6 shards. Shard 0 started 01:38. The job draws 256 W at 77% GPU, so it counts as heavy:
the loop starts no shard on a weekday from 06:00 to 19:00. Rate so far: 512 of 3,073 rows in 938 s, about 1.8 s per row
(the estimate assumed 1.6). The CPU was loaded by r17's picks. Whether later rows of a shard are slower: not checked.

**TURTLE timing** (`OUT/smoke/turtle_timing.json`): 105 to 114 s per fit at K 20 on MMArt test, 12.3 h for 412 fits on one
3090 (the spec said 1 to 2 h; the estimate errs high, since held ArtELingo has fewer rows). Mean 186 W, max 291 W, so light
by the 200 W line; Qwen alone means 199.6 W.

**Tickets.** 09 merged (31 data tests). 08 merged after an Opus `high` review, one fix round, Sonnet `high` re-review APPROVE;
47 tests. 07 has an Opus `high` review, APPROVE with notes; its three should-fixes are open (`.scratch/.../notes/t07_review.md`).
