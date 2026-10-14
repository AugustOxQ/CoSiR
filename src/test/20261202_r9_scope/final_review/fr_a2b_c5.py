"""Final review A2b (copy of fr_a2_c5.py, rerun on 8db7ceb); A2 item 1 (C5 order): which verdict files open r9_score.score_seed's before-verdict guards, and
does run_r9_descriptive.bind_read accept the verdict that run_r9_apply_rule really writes?

Synthetic only: test_r9_score.make_bundle (random arrays), test_r9_apply_rule's staged pass in a temp folder.
Writes fr_a2b_c5.json beside this file. No real held or ArtELingo-28 row is read.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(F))
import r9_common as R  # noqa: E402
import r9_score as S  # noqa: E402
import test_r9_score as TS  # noqa: E402
import test_r9_apply_rule as TA  # noqa: E402
import run_r9_apply_rule as AR  # noqa: E402
import run_r9_descriptive as RD  # noqa: E402
import r9_bundle as B  # noqa: E402

out = {}
tmp = Path(tempfile.mkdtemp(prefix="fr_a2_c5_", dir="/tmp/claude-0/-project-CoSiR/06992efa-c388-41ec-bd52-4b5c99c57e84/scratchpad/fr_a2b"))
readers = B.load_readers()
held = TS.make_bundle()                       # mode "held": the scorer does not know which read (H6 or H7) it is
a28 = TS.make_bundle(seed=55, mode="a28")


def try_score(bundle, vpath, **kw):
    try:
        sc = S.score_seed(bundle, TS.NESTED_PICKS, TS.lambdas_all(), readers, include_pm=True, verdict=vpath, **kw)
        return {"accepted": True, "pm_keys": sorted(k for k in sc if k in S.PM_SCORERS)[:3]}
    except AssertionError as e:
        return {"accepted": False, "error": str(e)[:120]}


cases = {}
d = tmp / "score"
d.mkdir()
# (a) H6 bundle, H7's real verdict content under H7's name: the guard only needs a file whose name fits the mode
(d / "held_verdict_H7.json").write_text(json.dumps({"read": "H7", "verdict": "NO-GO", "status": "run"}))
cases["held bundle, verdict=held_verdict_H7.json (another read's verdict)"] = try_score(held, d / "held_verdict_H7.json")
# (b) a null not-run verdict under the read's own name
(d / "held_verdict_H6.json").write_text(json.dumps({"read": "H6", "verdict": None, "status": "not run: switched off"}))
cases["held bundle, held_verdict_H6.json with verdict null (not run)"] = try_score(held, d / "held_verdict_H6.json")
# (c) an empty file
(d / "e").mkdir()
(d / "e" / "held_verdict_H6.json").write_text("")
cases["held bundle, empty held_verdict_H6.json"] = try_score(held, d / "e" / "held_verdict_H6.json")
# (d) the reserve name while the original verdict does not exist
(d / "r").mkdir()
(d / "r" / "held_verdict_H6_reserve.json").write_text("{}")
cases["held bundle, held_verdict_H6_reserve.json only"] = try_score(held, d / "r" / "held_verdict_H6_reserve.json")
# (e) an a28 bundle (H9 or H10) given H8's verdict
(d / "held_verdict_H8.json").write_text(json.dumps({"read": "H8", "verdict": "full claim", "status": "run"}))
cases["a28 bundle, verdict=held_verdict_H8.json (H9/H10 bundle)"] = try_score(a28, d / "held_verdict_H8.json")
# (f) on_scores (direction breakdown) with another read's verdict
seen = []
cases["held bundle, on_scores with held_verdict_H7.json"] = try_score(
    held, d / "held_verdict_H7.json", on_scores=lambda n, s: seen.append(n))
cases["held bundle, on_scores with held_verdict_H7.json"]["on_scores_calls"] = len(seen)
# control: the read's own verdict missing
cases["held bundle, verdict file missing (control)"] = try_score(held, d / "nope" / "held_verdict_H6.json")
out["score_seed"] = cases


# ---------------------------------------------------------------- bind_read against apply_rule's own verdict
def bind(read, d, mutate=None):
    if mutate is not None:
        p = d / f"held_verdict_{read}.json"
        v = json.loads(p.read_text())
        mutate(v)
        p.unlink()
        p.write_text(json.dumps(v))
    try:
        b = RD.bind_read(d, read, False)
        return {"bound": True, "backbone": b.backbone}
    except RD.Refused as e:
        return {"bound": False, "refused": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"bound": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


bind_cases = {}
old_dir = AR.AGREEMENT_DIR
for read in ("H6", "H8"):
    t = tmp / f"apply_{read}"
    t.mkdir()
    dd = TA.stage(t, read, TA.ALL0)
    path, verdict = TA.call(read, dd)
    AR.AGREEMENT_DIR = dd
    bind_cases[f"{read}: verdict written by run_r9_apply_rule.apply"] = {
        "verdict_backbone": verdict.get("backbone"), "verdict": verdict.get("verdict"),
        "RH.READ_BACKBONES": RD.RH.READ_BACKBONES[read], **bind(read, dd)}
    bind_cases[f"{read}: same verdict, backbone set to the code name"] = bind(
        read, dd, lambda v, r=read: v.update(backbone=RD.RH.READ_BACKBONES[r]))
AR.AGREEMENT_DIR = old_dir
out["bind_read"] = bind_cases
(HERE / "fr_a2b_c5.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps(out, indent=1, default=str))
