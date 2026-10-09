"""Round 6 run chat (r6 run, 2026-10-10): seed 42's selection episodes (4,096 per pair) for run_r6_dts.py's
--episodes and r6_gpu_inputs.py, saved with r6_episodes.save_episodes. Built with the merged code exactly as
r6_context.RowContext builds them (run_r6_held.load_rows, then r6_episodes.build_seed on the selection rows with the
development value sets); the per-pair SHA-256s are then checked against baselines_seed42.json's by
run_r6_dts.load_episodes_checked. Writes nothing else and refuses an existing file. Not an r6 module (not hashed by
r6_common.r6_module_shas); the file is CPU-internal and never shipped.

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/prep_episodes_seed42.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_episodes as E  # noqa: E402
import run_r6_dts as D  # noqa: E402
import run_r6_held as RH  # noqa: E402

OUT = R.RESULTS / "dts_inputs" / "episodes_seed42.npz"


def main() -> int:
    if OUT.exists():
        print(f"refused: {OUT} exists")
        return 4
    rows = RH.load_rows()
    eps = E.build_seed(rows.labels, rows.split.groups, rows.split.selection, rows.index, rows.value_sets,
                       R.DEV_SEED, R.N_PER_PAIR)
    E.save_episodes(OUT, eps)
    D.load_episodes_checked(R.DEV_SEED, OUT)
    print(f"written {OUT} (seed {R.DEV_SEED}, per-pair SHA-256s equal to baselines_seed42.json's); "
          f"sha256 {R.sha256_file(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
