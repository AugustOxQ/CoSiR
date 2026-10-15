"""Round 3 shared constants and helpers (rule DECISION_RULE.md of this folder).

Puts round 1's folder (src/test/20261117_reader_fix_csd), round 2's folder and the repo root on sys.path and imports
round 1's verified modules and round 2's r2_fusion by name. No round-3 module may share a name with a round-1 or
round-2 module (rule section 10).
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST = ROOT / "src/test"
R1 = TEST / "20261117_reader_fix_csd"
R2 = TEST / "20261118_reader_fix_round2"
AB = TEST / "20261030_aspect_baselines"
for p in (str(ROOT), str(R1), str(R2)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C  # noqa: E402  round 1's common.py
import rc_core as K  # noqa: E402
import rb_build as rb  # noqa: E402
import rb_eval as rbe  # noqa: E402
import rb_features as rf  # noqa: E402
import r2_fusion as F  # noqa: E402  round 2's fusion pieces

if Path(C.__file__).resolve().parent != R1:
    raise ImportError(f"'common' resolved to {C.__file__}, not round 1's")
if Path(F.__file__).resolve().parent != R2:
    raise ImportError(f"'r2_fusion' resolved to {F.__file__}, not round 2's")

RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"
RES = HERE / "results"

# rule D15 (and §2, D2): read-only inputs, path relative to src/test/ -> SHA-256
INPUTS = {
    "20261117_reader_fix_csd/DECISION_RULE.md": "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c",
    "20261117_reader_fix_csd/common.py": "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
    "20261117_reader_fix_csd/rc_core.py": "e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c",
    "20261117_reader_fix_csd/rb_build.py": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "20261117_reader_fix_csd/rb_eval.py": "1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14",
    "20261117_reader_fix_csd/rb_features.py": "1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf",
    "20261117_reader_fix_csd/results/rb_reader_A0.pkl": "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "20261117_reader_fix_csd/results/rb_reader_A0.json": "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz": "628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json": "c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e",
    "20261117_reader_fix_csd/results/rc_tau.json": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    "20261117_reader_fix_csd/results/rule_application.txt": "9a074850e70db8075e85bc5475e9361905182d6f0815f0023eaf380ccdee9785",
    "20261118_reader_fix_round2/DECISION_RULE.md": "368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c",
    "20261118_reader_fix_round2/r2_fusion.py": "1406e8e71629e74c468f8b009a110e49aa05c481ed52cc1b8d99b59f87016027",
    "20261118_reader_fix_round2/results/rule_application.txt": "cf1ec3852e334b1d56920bd8ab25c2db858bf97ab46ea36a7f65ebdd0916741f",
    "20261120_r1_levers_brainstorm/bs_lib.py": "45cc5abd941ca1cd7e47f415cdf0e11d440fd9c703b99516e23eb6e41d49bf88",
    "20261120_r1_levers_brainstorm/bs_04_readers.py": "0260fe5ad61e3d4a664cbb921f7d84efed8c9329716229310651712697a372b8",
    "20261120_r1_levers_brainstorm/bs_05_aff.py": "4c947b1d42df1caa7fb7f769f245e7df9a59bce22ca2611413c6a5a7fd45d734",
    "20261120_r1_levers_brainstorm/bs_10_subsets.py": "b00d9ed0b18e65e1e2be4dd8cdc5e05e277f673dbec5c826ddce4e18669cd7b4",
    "20261120_r1_levers_brainstorm/results/bs_04_readers.json": "42bf6f204598c011bfada46e5eabf42dc7759fc6f6750721527d16239141691e",
    "20261120_r1_levers_brainstorm/results/bs_05_aff.json": "a96719ba505b72883bfa2eeaca66ab872da128314654ba8f74580998b2a72b10",
    "20261120_r1_levers_brainstorm/results/bs_10_subsets.json": "c9d64815c1c6dba9e84177f7a33fb4e917a326f801ed9b115ae770ce1a1988f1",
    "20261030_aspect_baselines/run_baselines.py": "26508dde35f77850c4a80e98d08b0ea6576638577c84d3e96c39660ef02e63c4",
    "20261030_aspect_baselines/results/per_anchor_seed42.npz": "a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d",
    "20261030_aspect_baselines/results/codes_provenance.json": "8e6a517b73610d4d21b42e2eb39f6fd4118dc4132049364c8e89eecb7716bfaf",
    "20261030_aspect_baselines/results/baselines_seed42.json": "ce42c81e8eec256496454e88fc07dc4fcfb02e2d5f1b043f2274ba6008564ce6",
    "20261030_aspect_baselines/results/baselines_seed43.json": "b250e89caadb1f5ad5937fb36b92ccf1f0f3b63a82b71056dc1ce2b9cc7e2859",
    "20261030_aspect_baselines/results/baselines_seed45.json": "ecbcf5e900a5f844af9a3e986d1154d2764bb10ba2e2123aba2aaf77917c9890",
    "20261030_aspect_baselines/results/baselines_seed47.json": "56e5c0661ad20372cd1c9976d2e48f07c995e44ce33f0649da5cc351f8652386",
    "20261030_aspect_baselines/results/baselines_seed48.json": "feed925f3eddbf2957b4bce284e1bbf5e9d85132b760f94f6d2c76cb08654286",
    # D15 text: D2's head identity
    "20261111_community_told_oracle/results/told_oracle.json": "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2",
    # rule §2 and D1, D2, D10 (also asserted by round 1's common.verify_inputs)
    "20261030_aspect_baselines/results/episodes_seed42.npz": "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    "20261111_community_told_oracle/results/per_anchor_told_oracle.npz": "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    "20261031_pseudo_partitions/results/partitions.npz": "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "20261108_new_method_quick_checks/results/n6_posteriors.npz": "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt": "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
}
_CHECKED = {}

A0 = ("affect", "image", "caption")            # rule A0; index 0 = affect (D5, D6)
AFFECT = 0
TEST_SEEDS = (49, 50, 51)                       # rule §6
SMOKE_SEEDS = (9001, 9002, 9003)                # rule §10
EARLIER_SEEDS = (42, 43, 45, 47, 48)            # rule §6.2 hash check
TAUS = (3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211)   # rule D6

# rule §5 item 2: round-1 R-c (cell numbers (t*7 + u)*8 + a; index = tune half)
RC_CELLS = {"fused": (116, 119), "cf": (58, 123), "sigma": (0.0, 0.0)}
RC_NUMBERS = {
    "fused_r1": 18.918863932291664, "cf_r1": 18.475341796875, "comparator": "counterpart",
    "bar": (0.4435221354166667, 0.21646171563312194, 0.6735669710776852),
    "gain_statistic": (2.667236328125, 2.325087836946873, 3.012361650695922),
}
# rule §5 item 3: AFF as the brainstorm recorded it (seed 42, full precision)
AFF_CELLS = {"fused": (39, 119), "cf": (149, 10), "sigma": (0.0, 0.0)}
AFF_BRAINSTORM = {
    "fused_r1": 19.136555989583336, "cf_r1": 18.39599609375, "comparator": "B_prime",
    "bar": (0.6998697916666667, 0.4598852740816973, 0.9371680126852968),
    "margin": (0.7405598958333333, 0.5196896694963071, 0.9598857494832738),
    "gain_statistic": (3.110758463541667, 2.780005709854805, 3.4559584315470384),
    "either": -1.629638671875,
    "per_pair_bar": {"emotion__style": 0.9765625, "emotion__genre": 1.45263671875, "style__genre": -0.32958984375},
    "aff_minus_r1_fused": (0.21769205729166666, 0.06425880757348419, 0.3709597330984391),
    "aff_minus_r1_bar": (0.25634765625, 0.04280778303598444, 0.46195041633015954),
    "open_tau0_counts": {"a": 9941, "b": 3627},
}
# rule D7: redundancy of z(s_h) with z(B) on seed 42
REDUNDANCY_42 = {
    "affect": {"i2t": 0.35348060377541385, "t2i": 0.3828024789253903},
    "image": {"i2t": 0.7145397990123284, "t2i": 0.7090878258485419},
    "caption": {"i2t": 0.6182295729609555, "t2i": 0.665152773464146},
}


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


def input_path(name) -> Path:
    if name not in INPUTS:
        raise KeyError(f"{name} is not a D15 input")
    return TEST / name


def assert_inputs(names) -> dict:
    """SHA-256 of each named input (path relative to src/test/); a mismatch stops the run."""
    out = {}
    for n in names:
        if n not in _CHECKED:
            got = sha_file(input_path(n))
            if got != INPUTS[n]:
                raise SystemExit(f"input {n}: SHA-256 {got} differs from the rule's {INPUTS[n]}")
            _CHECKED[n] = got
        out[n] = _CHECKED[n]
    return out


def assert_taus():
    """rule D6: tau_0..tau_3 read from round 1's rc_tau.json equal the rule's values exactly."""
    assert_inputs(["20261117_reader_fix_csd/results/rc_tau.json"])
    taus = tuple(json.loads((R1 / "results/rc_tau.json").read_text())["taus"])
    if taus != TAUS:
        raise SystemExit(f"rc_tau.json taus {taus} differ from the rule's {TAUS}")
    return TAUS


def res_dir(smoke) -> Path:
    d = RES / "smoke" if smoke else RES
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_head() -> str:
    return C.git_head()


def provenance(smoke) -> dict:
    return {"rule_sha256": RULE_SHA, "written": now_ams(), "smoke": bool(smoke), "git_head": git_head()}


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
