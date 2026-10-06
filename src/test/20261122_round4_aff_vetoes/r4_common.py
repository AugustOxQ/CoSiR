"""Round 4 shared constants and helpers (rule DECISION_RULE.md of this folder).

Imports round 3's modules by path (r3_common, r3_bundle, r3_fusion, r3_stats; round 3's folder goes on sys.path so
their own imports resolve) and checks each resolved to round 3's file. On import it sets R3.TEST_SEEDS to this round's
fresh seeds (rule section 4 item 1), which round 3's seed guard reads at call time. No round-4 module may share a name
with a module of rounds 1 to 3.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST = ROOT / "src/test"
R3_DIR = TEST / "20261121_round3_affect_gate"
for p in (str(R3_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_common as R3  # noqa: E402  (puts rounds 1 and 2 and the repo root on sys.path)
import r3_bundle as RB3  # noqa: E402
import r3_fusion as RF3  # noqa: E402
import r3_stats as RS3  # noqa: E402

for _m in (R3, RB3, RF3, RS3):
    if Path(_m.__file__).resolve().parent != R3_DIR:
        raise ImportError(f"'{_m.__name__}' resolved to {_m.__file__}, not round 3's")

C = R3.C
sha_file = R3.sha_file
now_ams = R3.now_ams
git_head = R3.git_head

R3.TEST_SEEDS = (52, 53, 54)   # rule section 4 item 1 (round 3's seed guard reads this at call time)

RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b"
RES = HERE / "results"

# rule D11: round 4's own inputs, path relative to src/test/ -> SHA-256 (round 3's D15 table stays in R3.INPUTS)
INPUTS = {
    "20261121_round3_affect_gate/DECISION_RULE.md": "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925",
    "20261121_round3_affect_gate/r3_common.py": "b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86",
    "20261121_round3_affect_gate/r3_bundle.py": "bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d",
    "20261121_round3_affect_gate/r3_fusion.py": "ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637",
    "20261121_round3_affect_gate/r3_stats.py": "846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d",
    "20261117_reader_fix_csd/results/rb_reader_A1.pkl": "4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6",
    "20261117_reader_fix_csd/results/rb_reader_A1.json": "41ec3bd1ef99523430ce6aacb2576c7928446a93b65fb9ded2b3ae6a6089d649",
    "20261117_reader_fix_csd/results/rb_reader_A1.npz": "f0297b92276cb0eda69f3717545c74a766203517950622f2966978b209c488cd",
    "20261116_grouping_step1_style/run_step1.py": "f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0",
    "20261116_grouping_step1_style/results/step1_heads_style.npz": "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    "20261118_reader_fix_round2/results/cand_R1_A1.npz": "7494f8ef3f17db4ec1ef4cc5e5948b5fbd0c793beaf700a3a8464b296d01e075",
    "20261118_reader_fix_round2/results/cand_R1_A1.json": "4d13edaf7f3ea9f25782685c2f7daa9ebafeec6a5595d8de233cda376a4b9eb2",
    "20261120_r1_levers_brainstorm/results/bs_08_csd_evidence.json": "7786d2aeaca8cc6ab67797b654a5b5ef3c4fd4df197f3b21eb8253e46c136a4b",
    "20261120_r1_levers_brainstorm/bs_08_csd_evidence.py": "c4001e27c54afc15e8fa1fe51b84f59434770cb240731495ee872ef12f48813b",
    "20261120_r1_levers_brainstorm/bs_07_detector.py": "1b287bcc725270af9b3b0a0b41b9021d9ed3ea844ef404df52ca4f4e9e47088a",
    "20261120_r1_levers_brainstorm/results/bs_07_detector.json": "e9d52462a2c128b39cb5a2fe7b78b425e5ceac982b091c334d99d5d951a485e1",
    "20261120_r1_levers_brainstorm/bs_11_visual_side.py": "cb27cd636bae5f92ba016602148ad2508ed8a50867fe4c6b3b0c7c7885704ad2",
    "20261120_r1_levers_brainstorm/results/bs_11_visual_side.json": "a1b51be306750af10a2634488fdf775ce096588e36ac143c61194630282dc6c7",
    "20261118_reader_fix_round2/run_r2_fusion.py": "fe6fb9ff870c3942bef98327ff23894782522e516fc100cb1635aeec6e7e5b65",
    "20261030_aspect_baselines/results/baselines_seed49.json": "a7e9fbce85f2d2724a0d07ab3df6d4c0b5d83e314d2efd02aac278f9be1c21b3",
    "20261030_aspect_baselines/results/baselines_seed50.json": "b0567f8c2ee4782780097d8c601e0e043a9512a99a421f8463b2aba199cdc8fa",
    "20261030_aspect_baselines/results/baselines_seed51.json": "0d1c9c87446e99e4bed73cd2b105eeb6ef62630e0d7c3200e3dffea31299f42d",
}
_CHECKED = {}

A1 = ("affect", "image", "caption", "csd")
TEST_SEEDS = (52, 53, 54)                                  # rule section 4 item 1
SMOKE_SEEDS = (9001, 9002, 9003)                           # rule section 10
EARLIER_SEEDS = (42, 43, 45, 47, 48, 49, 50, 51)           # rule section 6.2 hash check
V75 = 0.021043562795966864                                 # rule section 5 item 3
V75_KEEP_42 = 9216
BPA1_MEAN_42 = 18.804931640625
CANDIDATES = ("V4", "V2", "V24")
READS_CSD = {"V4": False, "V2": True, "V24": True}
TIE_BAND_UNITS = 24

# rule section 5 item 3: R1 x a_v (IMGABST_q75), seed 42, full precision
IMGABST_TARGETS = {
    "fused_r1": 19.059244791666664, "cf_r1": 18.49365234375, "comparator": "counterpart",
    "bar_margin": (0.5655924479166667, 0.3448683992591827, 0.79821625538382),
    "gain_statistic": (2.878824869791667, 2.554983173204304, 3.2105685950938248),
    "either": -1.7476399739583333,
    "per_pair_bar": {"emotion__style": 0.677490234375, "emotion__genre": 1.28173828125,
                     "style__genre": -0.262451171875},
    "cells": {"fused": (117, 119), "cf": (58, 67), "sigma": (0.0, 0.0)},
}

# round 3's targets, re-exported (rule section 5 items 2 and 3 of round 3)
AFF_BRAINSTORM = R3.AFF_BRAINSTORM
AFF_CELLS = R3.AFF_CELLS
RC_NUMBERS = R3.RC_NUMBERS
RC_CELLS = R3.RC_CELLS


def assert_rule():
    """Every script calls this first: this rule's and round 3's rule's SHA-256 must be the committed ones."""
    got = sha_file(RULE)
    if got != RULE_SHA:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA}")
    R3.assert_rule()


def input_path(name) -> Path:
    if name not in INPUTS and name not in R3.INPUTS:
        raise KeyError(f"{name} is not a D11 input")
    return TEST / name


def assert_inputs(names) -> dict:
    """SHA-256 of each named input (path relative to src/test/); round 3's names go through R3.assert_inputs."""
    names = list(names)
    own = [n for n in names if n in INPUTS]
    out = dict(R3.assert_inputs([n for n in names if n not in INPUTS]))
    for n in own:
        if n not in _CHECKED:
            got = sha_file(input_path(n))
            if got != INPUTS[n]:
                raise SystemExit(f"input {n}: SHA-256 {got} differs from the rule's {INPUTS[n]}")
            _CHECKED[n] = got
        out[n] = _CHECKED[n]
    return out


def res_dir(smoke) -> Path:
    d = RES / "smoke" if smoke else RES
    d.mkdir(parents=True, exist_ok=True)
    return d


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
