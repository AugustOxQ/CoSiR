# Ticket 11 build note (rs_vh.py): PROGRESS NOTE, handed over at 200k tokens

State: rs_vh.py (stage0, stage4, ablation, selfcheck modes) and test_rs_vh.py (41 tests, all pass, about 30 s) are written.
The self-check is running; not finished when this note was written.

## Definitions settled within the ticket
- T_a partitions are recomputed from the instructed-image store as float32 rows divided by their float32 norm
  (`ct_diag.target_images` does this; the style store is not exactly unit norm, and without the division the style
  partitions differ from 12-24's frozen file; with it they are equal). Why: only that reading reproduces the frozen file.
- Student outputs are read as float16, unit norm checked, not repaired (RB rule 3.1); comparator probabilities are
  divided by their float32 row norm (CT rule 3).
- Where the stage 4 and ablation verdict files keep their numbers is not fixed: `STAGE4_ASPECT_PATHS`,
  `STAGE4_MARGIN_PATHS`, `ABLATION_PATHS` list the alternatives tried; edit them when tickets 09 and 10 fix their keys.
- G's full-scale held outputs and the stage 4 candidate "G" are read from 12-27's folder (`held_dir`); every other run
  from `full_root(outputs)/held/<name>`; removal arms are named by `rs_sheet.name(config with remove)`.
- A scale whose G outputs fail V0c has no outcome (V0d); the comparison then needs the verdict to say the same.
- The self-check leaves out G's emotion margin over PercepT (never computed before; it is stage 0's held result).

## Commands
- Tests (from the worktree): `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261027_recipe_sweep/test_rs_vh.py`
- Self-check with the point values only: `python rs_vh.py selfcheck --draws 0 --outputs /tmp/claude-0/r10-rs-11/p0`
  (first run: everything equal except the style T partitions, fixed afterwards by the float32 row division).
- Full self-check (2,000 draws): `python rs_vh.py selfcheck --outputs /tmp/claude-0/r10-rs-11/full`, log
  `/tmp/claude-0/r10-rs-11/self_full.log`, report `/tmp/claude-0/r10-rs-11/full/rederive_selfcheck.json`.

## Left open
- The result of the full self-check (point values and bounds, largest difference) is not in this note yet.
- Tests for the ablation wiring (a synthetic outputs tree with one present removal and seven missing ones) and for
  compare_stage4 with ticket 09 style blocks were drafted but not yet added to test_rs_vh.py.
- Not run: `stage0`, `stage4`, `ablation` against real verdict files (they do not exist yet).
- Partitions of G's sample held outputs are compared only when `<outputs>/stage0/partitions/<scale>/G/emotion__s{s}__r20.npz`
  exists; the count of unchecked partitions is in the report.
