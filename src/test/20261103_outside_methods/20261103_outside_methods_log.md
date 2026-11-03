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
