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

## Build 5 (2026-11-04, unattended)

- **Ticket 07 merged.** Fix round on the three should-fixes, then a Sonnet `high` re-review (approve with notes). The controller
  applied its two low notes: the code hash is taken before part a, and it covers `rg_held`. 41 tests.
- **Gate (iii) passes**, 45 min on the CPU, not the 2.5 h expected: a 60 items, max difference 0; b 63 items, 0; c 42 items, 5.55e-17.
  The hash covers `om_gate_iii`, `om_data`, `om_paths`, `ns_clu`, `ns_metrics`, `ns_ret`, `ns_data`, `rg_held`. Main's r17 records
  and the r19 build touch none of them, so merging main kept the hash valid (checked).
- **Ticket 10 merged.** Opus `high` review: CHANGES NEEDED, one blocker (r17's gate (vi) voided nothing; §3.5 says the same cells
  here), five should-fixes, each proved by a run, among them a stale parts file reaching `cells.json` and 13 of 25 sabotages
  surviving. Fix round: 102 tests, all 13 caught. Sonnet `high` re-review found one more: a failed unit read as a void and could
  reach `labels.json`. The controller fixed it: `labels` refuses a cell `not scored`, and a failed unit leaves no older parts file.
  104 tests. r17's gate (vi) is `ok` today, so the blocker changes no label.
- **Ticket 12 merged**, Sonnet `medium`; partial smoke on what exists. Its TURTLE and TPIPS paths have not run.
- **TURTLE route settled (agent default): node402**, ticket 09b, as `shared-resources.md` says for a reserved node. The 128 inputs
  are exported once to `OUT/turtle_pack` (14 GB) and synced through a new `DATA_MAP` pair. Probe on the node: torch 2.11, `entr`
  compiles on the GPU, one full fit (MMArt Qwen plain, K 20, start 0) 119.3 s, loss 2.9114, 20 clusters, against 105 s locally. The
  loss and AMI comparison with a local fit is open: the build's next chat does it. Gate (iv) does not apply to TURTLE, since
  nothing compares its vectors across machines.

## Build 6 (2026-11-04, unattended)

- **Ticket 11 merged** (Sonnet `medium`; Codex had no room at 03:17): `om_run.py`, `om_queue.sh`, `om_smoke_inputs.py`. Controller read the diff.
  The worker ran two CPU jobs outside `res-queue` at nice 19 with 2 threads after the queue answered busy; the rest of the smoke went through the queue.
- **End-to-end smoke passes:** stage 0, the real `ranges` step (140 keys, 4.7 min), CLAY and TPIPS picks on both sets, stage 2 on 9 units (all six
  criteria on MMArt Qwen; SC2, SC4, SC5 on ArtELingo Qwen), `assemble` (83 cells, each with numbers), `labels`, `om_diag all` for one aspect and
  encoder. Synthetic: TPIPS stores (random rows), TURTLE partitions (random labels, one short real fit), gate (iii) and (iv) stubs, the C13 file.
  Real: CLAY vectors, set B, r17's ranges, c* heads. So the labels in the smoke mean nothing; the paths are what it proves. A stale hash on gate
  (iii) part c voided its 12 cells and the run went on (D24). Stage 2 refused a changed record. Times per unit on 8 threads, smoke draws: SC1 18 min,
  SC3 15, SC6 12, SC2 5, SC4 4 to 14, SC5 1.
- **TURTLE reference fit:** the local 3090 and node402 give the same partition for MMArt Qwen plain, K 20, start 0: every row equal, loss 2.911379 on
  both, AMI against style 0.22189, genre 0.43461, emotion 0.0408 on both. 140 s locally (under queue load), 119 s on the node. So the 16 jobs stay on node402.
- **Final review.** The first Opus `xhigh` reviewer passed 200k tokens in 36 calls and gave no verdict. Two fresh ones, A (rule, cells, records) and B
  (methods, gates, diagnostics): each APPROVE with notes, no blocker. Each stopped at the 200k meter with its list of what it did not check.
  Re-derived: the 337 ids and 28 units from the spec's text by independent code; SC4 and SC2 cells on MMArt Qwen emotion by own code at the same
  draws (largest difference 6e-11; SC4 cand 0.20107, comp 0.19168, d +0.00939); CLAY vectors to 1.5e-8; gate (i) (P 2.2e-7, mu 6.0e-8, query 6.6e-7,
  tolerance 1e-5); gate (ii) hash and CPU part; gate (iii) hash from the 8 files, 36 of 36 r15 retrieval expected values found exactly in r15's
  `verdicts.json`. Sabotages: 19 on the rule, 2 missed and now tested (the SC4 text pin; the comparator read-out in `check_fresh`); gate (i), gate (ii),
  merge, gate (iv) and `check-pulled` sabotages in B. D12 clean: shared lines are only `if __name__`, `torch.no_grad`, seeds.
- **Gate reach for the report:** gate (i) cannot see the log map's arccos scale at rank 49. On set B that scale matters (up to 0.011), so set B's path rests on
  reading and on the reviewer's own restatement of the official math (largest difference 7.2e-7, 0.001 degrees; the clone was not run). Gate (ii) at
  100 steps cannot see batch draws; the oracle test carries that claim, proved by sabotage. Gate (ii)'s GPU part ran on the 3090 and the fits run on
  node402; the reference fit above closes that gap for one fit. Gate (iv) compares no store row with the reference, hence `om_tpips_check.py`.
- **Fix wave** (Codex `gpt-6.1-sol` `high`, 12 items, one session, 21% of the five-hour window): `assemble` and `labels` refuse a status that differs from the gate and
  stage 1 files; `labels` refuses a stub C13 file outside smoke and removes an older `labels.json` on refusal; `om_env` refuses OM_OUTPUTS without
  OM_SMOKE and the reverse in `om_run` and `om_cells`; r17-family r must equal r17's record at assemble (an r17 rerun could keep r 0.003668 against r17's
  0.007336 under a rebuilt pin); `stage1_records` asserts 140 keys; `sweep` keeps the spread over starts; `adds` says its pair is on rows only;
  `check-pulled` refuses a mixed host, torch version, device or settings; new `om_tpips_check.py` compares the 48 probe rows per shard with the
  reference and lists changed packages. No hashed file, `om_tpips.py`, `om_turtle.py` or `om_clay.py` changed.
- **Re-review** (Sonnet `high`): APPROVE with notes, 168 tests pass, 3 sabotages caught. Left as they are, all low: `check-pulled`'s guard has no repo test
  (plumbing; proved by 7 synthetic cases); a status refusal comes before `labels` removes an old file; `om_tpips_check.py` exits 0 while shards are
  pending, so read `complete`; `OM_SMOKE=2` reads as smoke in `om_paths` but not in `om_env`.
- **Left for later builds or the report:** parts files bind no input hash, so pass `--force` after any re-extraction or refit; gates (i) and (iv) carry no code
  hash; `sweep` has no TEMI rows; `r17_kmeans` stays empty until r17's stage 3 files exist; in `adds`, the interval's widening is not measured.

## Run (2026-11-04, unattended)

- **Checks before the fits:** the whole round's suite, run once after the fix wave through `res-queue cpu`: 314 passed in 2 min 13 s. Stage 0 (`om_run.py
  stage0`): gate (i) pass, largest difference 6.6e-07; gate (ii) CPU pass. Status: i, ii, iii a, b, c `ok`; iv `failed` until the local rerun after shard 5.
- **TURTLE:** 16 jobs on node402 at c29073c9, one GPU each, 05:50 to 10:17. All three GPUs were free: r19's 8 node jobs had ended by 03:59. Each job took 32 to
  57 min, mean 44.5, so the whole set ended about an hour before the handoff's estimate. 0 failures.
- **Pulls ran after the last job, not per job.** All 16 share `/local/wding/res/r18_turtle`, and a tagged pull's verifying dry run would find another job's new
  files. All 16 pulls verified and marked; each code copy removed.
- **`check-pulled res/r18_turtle`:** 412 of 412 fits good, 0 bad; hosts node402 412; torch 2.11.0+cu130 412. The printed rsync line ran; its dry run showed no
  overwrite. `OUT/parts/turtle` holds 412 npz and 412 json, 3.0 GB, and a repeat dry run finds nothing left.
- **TPIPS:** shard 1 ended 08:02 after 151.7 min, not the 3.9 h the handoff expected; the store check at 10:29 finds every probe row equal, cosine 1.0, and no
  package changed. The loop (pid 2887562) waits for 19:00. At 2.5 h a shard, shards 2 to 5 end about Thursday 05:00; at 3.9 h, shard 5 still ends Thursday 23:00.
- **Git:** `*.json` is gitignored at the repo root, so stage 1's three records need `git add -f`, as r17's did.
- **TPIPS stores:** shards 2 to 5 took about 1.9 to 2.5 h each and ended 2026-11-05 03:08; the loop had six full shards at 03:09, not Thursday 23:00. Store check
  `complete=True`, no package changed. Both merges passed (ArtELingo 18,433 rows, MMArt 16,384). Gate (iv) passes: lowest cosine 0.99999994 against the bar of 0.999,
  48 rows per factor; all six stores pass. All six gates `ok`.
- **Merge:** main into r18-int at b6b9975a. No hashed file moved (`git diff --stat` on `om_gate_iii`, `om_data`, `om_paths`, `ns_clu`, `ns_metrics`, `ns_ret`, `ns_data`,
  `rg_held` is empty). Status after it still `ok` for all gates.
- **Stage 1:** 03:11 to 03:19, 8 min against the 2 h estimate. 140 ranges, 0 with r null. Picks: TPIPS and CLAY `none+k10` on both sets; CLAY text set A for clustering
  and retrieval on both sets (`a_higher`). Records committed at 8ca05e1c with `git add -f`.
- **Stage 2 ran four units at a time, not one at a time as `om_queue.sh stage2 --run` does (agent default).** Each unit is its own process with its own part file, and the
  caches are in memory, so no number depends on the order. 28 units through `res-queue cpu`, 03:20 to 04:10, 3.1 to 12.8 min each, all exit 0.
- **Assemble:** 337 cells, 0 void, r17-family ranges equal r17's record. Labels stay provisional until C13 passes.

## C13 code (2026-11-05, unattended)

- **Brief:** `.scratch/r18-outside-methods/c13/task.txt`, from r17's. Codex `gpt-6.1-sol` high (5h 100% left before, 84% after). It may not read `OUT/stage1`, `OUT/stage2`,
  `OUT/clay`, `OUT/steps`, `records/stage1_*`, `records/status.json`, this log, the handoffs or the reports. Only `finish` opens the status files; only `compare` opens the round's numbers.
- **Files:** `om_c13.py`, `test_om_c13.py`; 46 tests pass (controller's run: 46 passed in 3.60 s).
- **Low-draw tree:** `/data/SSD2/cosir_r18/20261103_outside_methods/c13_round_smoke`, the round's own code at 40 and 100 draws on real inputs, six units, 63 cells, 0 void
  (`.scratch/r18-outside-methods/run/c13_round_smoke.sh`). Under `OM_SMOKE=1` the real TPIPS stores, TURTLE partitions and set B are links, and the stage 0 and stage 1 files are copies.
- **Subset smoke:** `C13 PASS items=2400 failed=0 max_abs=2.89213097915e-14 scope=subset`. No mismatch. 16.4 min with queue waits.
- **Estimate of the full derive:** 52 sections, 5.5 slot-hours. Not started: the chat cut at the 200k context meter before the derive, since a chat may not cut while its run goes on.
- **Binding:** derived files bind the text of `om_c13.py` above `# RESULT BOUNDARY`, plus `ns_c13.py`. A fix below that line keeps them valid. A fix above it means a new derive.
