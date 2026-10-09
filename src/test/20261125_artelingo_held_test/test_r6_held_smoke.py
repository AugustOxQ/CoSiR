"""The smoke of the held runner on real selection rows (ticket 08; rule section 6 item 7, section 8 items 1 to 3;
contracts section 7): run_r6_held.run_smoke on seeds 9001 to 9003, 64 episodes per pair, selection rows, the same
functions as the read (steps 3 to 8), into tmp_path/results/smoke/.

The runner's own setup() runs on the real data (heads refit on scorer-train rows and checked bit for bit against the
stored selection posteriors, PM fits, readers); its output is captured with the smoke's. refit_check.json is written
from that run's check and heads by run_r6_refit.record (the real record); the other seed-42 records the read needs
(picks_seed42.json, regression_seed42.json, sensitivity_seed42.json) are stand-ins in tmp_path with the current module
SHA-256s, since the real ones are written by the run chat (the smoke's numbers are not looked at). Checks: the run
completes; its stdout and stderr (written to a log file) contain no decimal number (regex \\d*\\.\\d+, the tmp path
removed); every file of contracts section 7 is written under results/smoke/ with the right keys and no verdict; the
smoke's episode hashes equal the stored per-pair hashes of AB's smoke seeds 9001 to 9003 (rule section 6 item 4). No
held row is loaded beyond the split's index arrays. About 5 to 10 minutes on CPU:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_held_smoke.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_refit as RR  # noqa: E402
import test_r6_held as TH  # noqa: E402  (stand-in records)

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import METRICS  # noqa: E402

DECIMAL = re.compile(r"\d*\.\d+")


def test_smoke_on_real_selection_rows(tmp_path, capfd, monkeypatch):
    res = tmp_path / "results"
    env = RH.setup()                                                   # real data, heads, check, PM, readers
    TH.write_records(res, coef=env.coef_sha256, smoke_names=(), value_sets=env.value_sets, data=env.data)
    (res / RH.REFIT_NAME).write_text(json.dumps(RR.record(env.head_check, env.heads, env.inputs)))
    edit_sens = json.loads((res / RH.SENS42_NAME).read_text())         # the regression record was not changed
    assert edit_sens["regression_sha256"] == R.sha256_file(res / RH.REGRESSION_NAME)
    monkeypatch.setattr(RH, "setup", lambda: env)                      # the smoke's step 4 gets this setup's env
    code = RH.run_smoke(results=res, here=HERE)
    cap = capfd.readouterr()
    out = cap.out + cap.err
    log = tmp_path / "run_r6_held_smoke.log"
    log.write_text(out)
    assert code == 0, out[-2000:]
    assert env.head_check["passed"] is True                            # the real refit reproduced the stored bits

    # no decimal number in the stdout or the log (paths of tmp_path removed)
    text = log.read_text().replace(str(tmp_path), "<tmp>")
    assert not DECIMAL.search(text), DECIMAL.findall(text)[:5]
    smoke = res / "smoke"
    lines = cap.out.strip().splitlines()
    assert lines[-1] == f"smoke held pass written {R.sha256_file(smoke / 'held_pass.json')}"

    # the files of contracts section 7 under results/smoke/, no verdict, nothing outside it but the records
    names = sorted(p.name for p in smoke.iterdir())
    assert names == sorted(["held_started.json", RH.SMOKE_COPY_DIR, "sensitivity_held.json", "held_arrays.npz",
                            "held_pass.json"] + [f"held_episodes_seed{s}.npz" for s in R.SMOKE_SEEDS])
    assert sorted(p.name for p in res.iterdir()) == sorted(
        ["smoke", RH.REFIT_NAME, RH.PICKS_NAME, RH.REGRESSION_NAME, RH.SENS42_NAME, RH.DTS_STOP_NAME,
         RH.DTS_CHOSEN_NAME, RH.VALUE_SETS_NAME])
    assert not any("verdict" in p.name or p.name.endswith(".partial") for p in tmp_path.rglob("*"))

    # the episodes are the stored smoke episodes of AB (rule section 6 item 4), on selection rows
    targets = E.identity_targets()
    att = json.loads((smoke / "held_started.json").read_text())["attempts"]
    assert len(att) == 1 and att[0]["flags"]["smoke"] is True and att[0]["coef_sha256"] == env.coef_sha256
    sel = np.zeros(R.N_ROWS, dtype=bool)
    sel[env.split.selection] = True
    for s in R.SMOKE_SEEDS:
        want = targets[s]["episodes_sha256"]
        assert att[0]["episodes_sha256"][str(s)] == want, s
        eps = E.load_episodes(smoke / f"held_episodes_seed{s}.npz")
        assert eps.sha == want and eps.n == 3 * R.N_SMOKE and sel[eps.pooled.rows()].all()

    pr = json.loads((smoke / "held_pass.json").read_text())
    assert pr["mode"] == "smoke" and pr["seeds"] == list(R.SMOKE_SEEDS) and pr["n_episodes"] == 9 * R.N_SMOKE
    assert set(pr["checks"]) == set(ST.CHECKS) and set(pr["secondary"]) == set(ST.SECONDARY)
    assert pr["module_sha256"] == R.r6_module_shas() and pr["coef_sha256"] == env.coef_sha256
    sens = json.loads((smoke / "sensitivity_held.json").read_text())
    assert sens["N"] == 9 * R.N_SMOKE and all(set(sens[c]) >= {"SE", "x95"} for c in ST.CHECKS + ST.SECONDARY)
    with np.load(smoke / "held_arrays.npz") as z:
        assert set(z.files) == ({f"{s}__{m}" for s in RH.READ_SCORERS for m in METRICS}
                                | {"cl", "pair_index", "seed_index"})
        assert z["cl"].shape == (9 * R.N_SMOKE,) and (z["aff_cf__gain"] == 0).all()
