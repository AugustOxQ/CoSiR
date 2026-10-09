"""fr_c: two rule-named guards of area C fired through their entry points (reviewer C): rule 8.3's "GPU jobs receive
no labels" (scripts/das6_sync_r6.py's job check, run with the system python as the run chat does) and rule 8.3's
"the descriptive script asserts that held_verdict.json exists and records the phase-2 agreement"."""
import json, subprocess, sys
from pathlib import Path
F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import numpy as np  # noqa: E402
OUT = Path(sys.argv[1]) / "guards"
OUT.mkdir(parents=True, exist_ok=True)
res = {}
# 1. a job folder that carries a label-bearing array and a WikiArt path
job = OUT / "job_bad"
job.mkdir(exist_ok=True)
np.savez(job / "verbalise_input.npz", seed=np.int64(42), episode_index=np.arange(2), anchor=np.arange(2))
np.savez_compressed(job / "rows_manifest.npz", rows=np.arange(2), image_name=np.array(["0" * 20 + ".jpg"] * 2),
                    caption=np.array(["Impressionism/monet_x.jpg", "a caption"]))
good = OUT / "job_good"
good.mkdir(exist_ok=True)
np.savez(good / "verbalise_input.npz", seed=np.int64(42), episode_index=np.arange(2))
np.savez_compressed(good / "rows_manifest.npz", rows=np.arange(2), image_name=np.array(["0" * 20 + ".jpg"] * 2),
                    caption=np.array(["a caption", "b caption"]))
code = ("import sys; sys.path.insert(0, %r); import das6_sync_r6 as S; from pathlib import Path; f = S.style_folders();"
        "print(len(S.forbidden_in_job(Path(%r), f)), len(S.forbidden_in_job(Path(%r), f)))"
        % (str(F.parents[2] / "scripts"), str(job), str(good)))
p = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
res["das6_job_check (bad, good) problems"] = p.stdout.strip() or p.stderr.strip()[-300:]
# 2. the descriptive pass without a verdict: refused (exit 4), nothing written
d = OUT / "desc_real"
d.mkdir(parents=True, exist_ok=True)
rc = RD.run(smoke=False, out=d)
res["descriptive_without_verdict_exit"] = rc
res["descriptive_wrote"] = sorted(p.name for p in d.iterdir())
print(json.dumps(res))
(F / "final_review" / "fr_c_guards.json").write_text(json.dumps(res, indent=1))
