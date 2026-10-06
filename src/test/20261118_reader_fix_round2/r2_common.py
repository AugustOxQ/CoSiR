"""Shared constants and helpers of reader-fix round 2 (binding rule: DECISION_RULE.md in this folder, SHA-256 asserted).

Round 1's verified modules are imported from src/test/20261117_reader_fix_csd/ by path and never modified; their
functions that write files (common.save_candidate, common.write_json_once, common.res_dir, the rb_build.py stages,
run_rc.py) are not called. No module of this folder may share a name with a round-1 module (common, rc_core, rb_*,
run_ra, run_rc, apply_rule, test_*): round 1's folder is on sys.path and a same-named module would shadow it.

    from r2_common import C, K, rb, rbe, rf      # round 1's common, rc_core, rb_build, rb_eval, rb_features
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R1 = ROOT / "src/test/20261117_reader_fix_csd"
for p in (str(ROOT), str(R1)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C  # noqa: E402  round 1's common.py
import rc_core as K  # noqa: E402
import rb_build as rb  # noqa: E402
import rb_eval as rbe  # noqa: E402
import rb_features as rf  # noqa: E402

if Path(C.__file__).resolve().parent != R1:
    raise ImportError(f"'common' resolved to {C.__file__}, not round 1's")

RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c"
RES = HERE / "results"

# rule D14: round-1 inputs (read only), path relative to round 1's folder -> SHA-256
R1_INPUTS = {
    "DECISION_RULE.md": "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c",
    "common.py": "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
    "rc_core.py": "e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c",
    "rb_build.py": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "rb_eval.py": "1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14",
    "rb_features.py": "1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf",
    "results/rb_reader_A0.pkl": "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "results/rb_reader_A0.json": "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    "results/rb_reader_A0.npz": "41ef7f5d9c8c0f0c91d2bf36f2f5d79ae4ea1b7b2fbb3ba89e7febb9404cc4d0",
    "results/rb_bank_A0_half0.npz": "f0711629d18d795f357368ac080c9f816e480cec44c3ec0d057e13eac9e8a1e2",
    "results/rb_bank_A0_half0.json": "7ba7b49a0f7c6344949a92b3aa3cdcf2fe6df796c4d775beb9fff75315331347",
    "results/rb_bank_A0_half1.npz": "edd48914d8db50e492e21ad725619464a655098f77651e7226c200c86db8ca80",
    "results/rb_bank_A0_half1.json": "7d12804ae7a9835648e6a0c523fcb5495b2c83ceaedc34bd11aed9869641c50d",
    "results/rb_halves.npz": "ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da",
    "results/rb_halves.json": "cbdecbd9e33807701791ec4df20da8967edcfa6090133e056ab3392fb10676b2",
    "results/rb_heads_affect.npz": "e21e846be8b9e565a7443c5e7ecd4135a04def709ce64bef6dceae6545823324",
    "results/rb_heads_affect.json": "9492b4cbb3e07961c8d4dbc5c0c1e143ba479be261602b1436b6d844af53dda1",
    "results/rb_heads_image.npz": "3743b3fe9c8942cddf399b8ffc7ef47806d8d2a8d4a6ec8ea474473010f3c46f",
    "results/rb_heads_image.json": "8888c2904b48e2cd14f6efd14e9b172d49c1267243c537be8e9805ec28bff121",
    "results/rb_heads_caption.npz": "00782ea90b9dcccea8ad164ce21f48792d7422adba89bebf9acaf7705168cea8",
    "results/rb_heads_caption.json": "02e9745f0946083c638b954048e1c5b3de18ab762a67012d45bab482ba765c07",
    "results/rb_diag_A0.json": "7364a27804db4f902f1177bdc449dd2f2b79dde1e8584e2ac26e691d53f7a03f",
    "results/cand_Rc_Rb_expected_A0.npz": "628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981",
    "results/cand_Rc_Rb_expected_A0.json": "c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e",
    "results/rc_tau.json": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    "results/rule_application.txt": "9a074850e70db8075e85bc5475e9361905182d6f0815f0023eaf380ccdee9785",
    "results/rb_reader_A1.pkl": "4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6",
    "results/rb_reader_A1.json": "41ec3bd1ef99523430ce6aacb2576c7928446a93b65fb9ded2b3ae6a6089d649",
    "results/rb_reader_A1.npz": "f0297b92276cb0eda69f3717545c74a766203517950622f2966978b209c488cd",
    "results/rb_bank_A1_half0.npz": "a499041d94b2396d28c586d1027bfe9fbd39d733334319ca8e2fe1ebec83bac7",
    "results/rb_bank_A1_half0.json": "7a60c1a35d920af3f0c7cdb7c63599a6be8c6e76a7eb3e721911847c1313bf3d",
    "results/rb_bank_A1_half1.npz": "887c80658ea295c72e21f2724ac8b289cbf9aee6789ffa5948f892f94ade87fd",
    "results/rb_bank_A1_half1.json": "023f6b993258a7dad0c3cbc8fc6a670772afa0522c75d4281ddf7c1dce819692",
    "results/rb_heads_csd.npz": "97906e87bade9756272b2120cfd3dcabaf2dae52305833554886868bcb96d028",
    "results/rb_heads_csd.json": "9c75532eb14321e1bb7ebe28171c09487665f2462d968943684ef308c785eda4",
    "results/rb_diag_A1.json": "658503b6a50134f375d6cbd44626e28b827e6ec764b74fb43022189fadeefef0",
}
_CHECKED = {}


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def now_ams() -> str:
    """Amsterdam local time, plain (no offset)."""
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


def assert_rule():
    """Every script calls this first: the rule's SHA-256 must be the committed one."""
    got = sha_file(RULE)
    if got != RULE_SHA:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA}")


def r1_path(name) -> Path:
    if name not in R1_INPUTS:
        raise KeyError(f"{name} is not a D14 input")
    return R1 / name


def assert_inputs(names) -> dict:
    """SHA-256 of each named D14 input (path relative to round 1's folder); a mismatch stops the run."""
    out = {}
    for n in names:
        if n not in _CHECKED:
            got = sha_file(r1_path(n))
            if got != R1_INPUTS[n]:
                raise SystemExit(f"round-1 input {n}: SHA-256 {got} differs from the rule's {R1_INPUTS[n]}")
            _CHECKED[n] = got
        out[n] = _CHECKED[n]
    return out


def res_dir(smoke) -> Path:
    d = RES / "smoke" if smoke else RES
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance(smoke) -> dict:
    return {"rule_sha256": RULE_SHA, "written": now_ams(), "smoke": bool(smoke), "git_head": C.git_head()}


def refuse_existing(paths, smoke):
    """Non-smoke outputs are never overwritten."""
    if smoke:
        return
    busy = [str(p) for p in paths if Path(p).exists()]
    if busy:
        raise SystemExit(f"outputs exist ({', '.join(busy)}); refusing to overwrite")


def write_json_once(path, rec, smoke):
    """A results JSON with this round's provenance (non-smoke: never overwritten; must be finite)."""
    path = Path(path)
    refuse_existing([path], smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**C.jsonable(rec), "provenance": provenance(smoke)}
    C.assert_finite_tree(rec)
    path.write_text(json.dumps(rec, indent=1))
    return rec
