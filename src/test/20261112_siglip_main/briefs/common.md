# r28 common brief (read first; every r28 job brief points here)

Round r28 puts CoSiR's main result on the SigLIP 1 student. The round's plan is
`/project/CoSiR/docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md` (read sections 2, 4 and 5; your job brief says
which item of section 4 is yours). The inventory of what exists: `src/test/20261112_siglip_main/siglip_inventory.md`
(a Codex read; most lines not checked at source: check any line you rely on).

## Fixed facts
- Repo `/project/CoSiR`. Python: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python`.
- Round folder `src/test/20261112_siglip_main/`, module prefix `r28_` (the folder goes on `sys.path`; never name a
  module without the prefix). Outputs under `/data/SSD2/cosir_r28/20261112_siglip_main/<your job>/`.
- SigLIP 1 = store folder `siglip_so400m`, width 1152 (`src/test/20261102_backbones_sweep/bs_paths.py`). Encoder keys
  in r14: `siglip1`, `siglip2`, `clip_l14`, `qwen`.
- Sample ID consistency (constitution C6): join by the extraction's `sample_ids` / the stores' `index.json`, never
  `range(n)`. Assert row counts and ID alignment at every file boundary you read.
- **Validation only for every pick** (read-out, k, threshold, hyperparameter). The test split is scored, never used
  to choose. If any step of yours would read test data to choose, stop and say so.
- Reuse the code that made the existing Qwen3 rows (import it; do not rewrite metrics). If you must change an existing
  file, keep its old behaviour bit for bit and log the edit in `/project/CoSiR/.claude/20261112_log.md`
  (`# <path>`, before/after, why).
- The paper repo `docs/paper/` is read only.

## How to work
- No single command over about 4 minutes. GPU: `res-queue gpu --no-wait --short --label r28:<job> -- <cmd>` (exit 75:
  busy, report). A run longer than 4 minutes: write the script and a smoke on a small slice, prove the smoke, and
  hand back the exact full command; the controller runs it.
- Unit tests only for code that computes a number (metrics, picks, joins); `pytest -q`, only your tests.
- Delete nothing; list what you would remove. Do not commit (the controller commits).
- Write only the files your brief names (other jobs write beside you in the same folder).

## Hand-back (at most about 350 words)
1. Files written (paths) and the exact commands the controller must run, if any.
2. Numbers, each with the file it comes from (read from the file, not from memory).
3. **Decision lines**, one per result-shaping choice you made (a read-out or scoring choice, a filter, a tie rule, a
   fallback, a tolerance): choice · main alternative · where (file:line) · could it change a verdict or claim
   (yes / maybe / no).
4. What you left out or could not check.
