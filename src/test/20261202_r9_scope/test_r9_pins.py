"""Tests of r9_pins.py, of parts("clip_b32") and of the independence of the r9 modules from round 6 (ticket 06; rule 2,
3.1; ticket 11's cells schema).

Real files read: the pinned CLIP parts (hashed), round 3's seed42_arrays.npz (the AFF gates), baselines_seed42.json,
picks_seed42.json, rc_tau.json. No feature, label or episode.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_pins.py
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_backbone as BB  # noqa: E402
import r9_pins as PINS  # noqa: E402
import r9_score as S  # noqa: E402

import numpy as np  # noqa: E402

PY = "/root/miniconda3/envs/CoSiR/bin/python"
R3_ARRAYS_REL = "20261121_round3_affect_gate/results/seed42_arrays.npz"


# ---------------------------------------------------------------- parts("clip_b32")

def test_every_r6_input_that_is_a_part_resolves_to_the_file_r6_pins():
    parts = BB.parts("clip_b32")
    by_path = {}
    for role, path in parts.items():
        by_path.setdefault(path, []).append(role)
    seen = set()
    for rel, sha in R.INPUT_SHA256.items():
        path = R.TEST / rel
        if path in by_path:
            assert R.sha256_file(path) == sha, rel
            seen |= set(by_path[path])
    # the pinned roles of R6 rule 6 item 1: the heads' inputs, the baselines, the readers, the checkpoint
    for role in ("prepare", "held_codes", "partition_L", "told_oracle", "step1_group_style", "partitions",
                 "affect_km", "n6_posteriors", "step1_heads_style", "A3_seed42", "rb_reader_A0",
                 "rb_reader_A0_json", "rc_tau", "baselines_seed42", "pm_fit_rows", "episodes_seed42"):
        assert role in seen, role


def test_the_roles_of_the_picks_and_the_a3_inputs_exist():
    parts = BB.parts("clip_b32")
    for role in ("A3_seed42", "picked", "rb_halves", "rb_reader_A0", "rb_reader_A0_json", "rc_tau",
                 "baselines_seed42", "picks_seed42", "cells_seed42"):
        assert role in parts, role
    assert parts["picks_seed42"] == R.R6D / "results/picks_seed42.json"
    assert parts["picks_seed42"].is_file() and parts["picked"].is_file()


def test_the_cells_part_is_in_the_round_folder():
    assert BB.parts("clip_b32")["cells_seed42"] == R.HERE / "cells_seed42_clip.json"
    assert BB.parts("qwen3vl_emb_2b")["cells_seed42"] == R.RESULTS / "recipe/qwen3vl_emb_2b/cells_seed42.json"
    assert BB.parts("siglip2_so400m")["cells_seed42"] == R.RESULTS / "recipe/siglip2_so400m/cells_seed42.json"
    assert BB.parts("clip_b32")["cells_seed42"].is_file()
    assert "cells_seed42" in BB.RECIPE_ROLES


# ---------------------------------------------------------------- gate_share (ticket 06, from 25's fix round)

def aff_gates():
    with np.load(R.INPUT_PATHS[R3_ARRAYS_REL]) as z:
        return ({c: z[f"aff_gate__{c}"] for c in "ab"}, z["parity"], tuple(int(x) for x in z["aff_fused_cells"]))


def test_gate_share_is_the_integer_count_over_the_integer_total_in_float64():
    R.assert_input(R3_ARRAYS_REL)
    gates, parity, cells = aff_gates()
    got = PINS.gate_share(gates, parity, cells)
    assert got == {"gate_share": 11071 / 24576, "gate_open": 11071, "gate_total": 24576}
    assert got["gate_share"] == 0.4504801432291667
    assert type(got["gate_open"]) is int and type(got["gate_total"]) is int
    chosen = np.concatenate([gates[c][cells[h] // 56][parity == 1 - h] for h in (0, 1) for c in "ab"])
    assert float(chosen.mean()) == 0.4504801332950592                  # the float32 mean the chain's reader gives
    assert got["gate_share"] != float(chosen.mean())


def test_gate_share_uses_the_gates_of_the_chosen_tau_on_the_other_parity():
    n = 8
    parity = np.arange(n) % 2
    gates = {c: np.zeros((4, n), np.float32) for c in "ab"}
    cells = (0 * 56 + 3, 2 * 56 + 5)                         # tau index 0 for half 0, tau index 2 for half 1
    gates["a"][0, parity == 1] = 1                           # half 0 scores the parity-1 episodes
    gates["b"][2, parity == 0] = 1                           # half 1 scores the parity-0 episodes
    gates["a"][1] = 1                                        # another tau: ignored
    gates["b"][0, parity == 0] = 1                           # right tau, wrong parity for half 0: ignored
    got = PINS.gate_share(gates, parity, cells)
    assert (got["gate_open"], got["gate_total"]) == (4 + 4, 16)
    assert got["gate_share"] == 0.5


def test_gate_share_refuses_a_value_that_is_not_a_gate():
    gates = {c: np.zeros((4, 4), np.float32) for c in "ab"}
    gates["a"][0, 0] = 0.5
    with pytest.raises(AssertionError, match="gate"):
        PINS.gate_share(gates, np.arange(4) % 2, (0, 0))


# ---------------------------------------------------------------- cells_seed42_clip.json

def test_switched_off_means_both_halves_have_no_agreement_weight():
    off = next(c for c in range(R.F2.CELLS_PER_KAPPA) if R.F2.cell_values(c)[3] == 0.0)
    on = next(c for c in range(R.F2.CELLS_PER_KAPPA) if R.F2.cell_values(c)[3] > 0.0)
    assert PINS.switched_off((off, off)) is True
    assert PINS.switched_off((off, on)) is False
    assert PINS.switched_off((on, on)) is False
    assert PINS.switched_off((39, 119)) is False


def test_the_clip_cells_record():
    rec = PINS.clip_cells()
    assert list(rec) == ["aff", "cf", "r1_fused", "r1_counterpart", "sigma_star", "switched_off", "gate_share",
                         "gate_open", "gate_total"]
    assert rec["aff"] == [39, 119] and rec["cf"] == [149, 10]
    assert rec["r1_fused"] == [116, 119] and rec["r1_counterpart"] == [58, 123]
    assert rec["sigma_star"] == [0.0, 0.0] and rec["switched_off"] is False
    assert (rec["gate_open"], rec["gate_total"]) == (11071, 24576)
    assert rec["gate_share"] == rec["gate_open"] / rec["gate_total"] == 0.4504801432291667
    assert rec["aff"] == list(S.CELLS["aff"]["fused"]) and rec["cf"] == list(S.CELLS["aff"]["cf"])
    assert rec["r1_fused"] == list(S.CELLS["r1"]["fused"]) and rec["r1_counterpart"] == list(S.CELLS["r1"]["cf"])


def test_the_committed_cells_file_is_the_clip_record():
    path = BB.parts("clip_b32")["cells_seed42"]
    assert json.loads(path.read_text()) == PINS.clip_cells()
    assert PINS.read_cells(path) == PINS.clip_cells()


def test_read_cells_checks_the_schema(tmp_path):
    good = PINS.clip_cells()

    def read(rec):
        p = tmp_path / "cells.json"
        p.write_text(json.dumps(rec))
        return PINS.read_cells(p)

    assert read(good) == good
    for key in ("aff", "cf", "r1_fused", "r1_counterpart", "sigma_star", "switched_off", "gate_share"):
        bad = {k: v for k, v in good.items() if k != key}
        with pytest.raises(AssertionError, match="cells"):
            read(bad)
    for key, value in (("aff", [39]), ("aff", [39, 224]), ("cf", [1.5, 2]), ("sigma_star", [0.0]),
                       ("switched_off", 0), ("switched_off", True), ("gate_share", 0.45),
                       ("gate_open", 11072)):
        with pytest.raises(AssertionError, match="cells"):
            read({**good, key: value})


# ---------------------------------------------------------------- pins_clip.json

def test_pins_clip_json_is_the_files_it_names():
    pins = json.loads(R.pins_path("clip_b32").read_text())
    assert R.pins_path("clip_b32") == R.HERE / "pins_clip.json"
    assert pins == PINS.build_pins("clip_b32")
    assert pins["backbone"] == "clip_b32"
    assert pins["input_sha256"] == R.INPUT_SHA256 and len(pins["input_sha256"]) == 51
    parts = BB.parts("clip_b32")
    assert set(pins["parts"]) == set(parts)
    for role, rec in pins["parts"].items():
        path = parts[role]
        assert rec["path"] == PINS.rel_path(path) and rec["path"].startswith("src/test/"), role
        assert (R.CHECKOUT if path.is_relative_to(R.HERE) else R.MAIN) / rec["path"] == path, role
        assert rec["sha256"] == R.sha256_file(path), role


def test_pins_clip_json_holds_the_picks():
    pins = json.loads(R.pins_path("clip_b32").read_text())
    picks = json.loads((R.R6D / "results/picks_seed42.json").read_text())
    base = json.loads(R.INPUT_PATHS["20261030_aspect_baselines/results/baselines_seed42.json"].read_text())
    for name in ("B", "B0", "B1"):
        assert pins["picks"][name] == picks[name]
    assert pins["picks"]["mean_r1"] == picks["mean_r1"]
    for name in ("rca", "diag", "diag_relu", "bilinear", "kissme", "xing", "wang", "probe", "tip", "value_prototype"):
        assert pins["picks"]["lambda"][name] == base["scorers"][name]["lambda_picks"]
    assert pins["picks"]["cells"] == PINS.clip_cells()
    assert pins["picks"]["taus"] == list(R.R3.TAUS)


def test_pins_clip_json_names_the_gate_references():
    pins = json.loads(R.pins_path("clip_b32").read_text())
    import run_r9_stage0a as RS
    assert pins["gate_references"] == {name: sha for name, (_, sha) in RS.REFERENCES.items()}


def test_write_pins_refuses_to_overwrite(tmp_path):
    out = tmp_path / "pins.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        PINS.write_pins("clip_b32", out)
    assert out.read_text() == "{}"


def test_write_pins_writes_the_built_record(tmp_path):
    out = tmp_path / "pins.json"
    PINS.write_pins("clip_b32", out)
    assert json.loads(out.read_text()) == PINS.build_pins("clip_b32")


def test_build_pins_names_a_missing_part(tmp_path):
    with pytest.raises(AssertionError, match="rc_tau"):
        PINS.build_pins("qwen3vl_emb_2b")                    # no recipe output exists yet


def test_write_cells_clip_refuses_to_overwrite(tmp_path):
    out = tmp_path / "cells.json"
    PINS.write_cells_clip(out)
    assert json.loads(out.read_text()) == PINS.clip_cells()
    with pytest.raises(FileExistsError):
        PINS.write_cells_clip(out)


# ---------------------------------------------------------------- no r9 module imports round 6

def r6_names(path):
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "import_module", "__import__", "spec_from_file_location"):
            names |= {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    return {n for n in names if n.split(".")[0].startswith(("r6_", "run_r6_")) or "artelingo_held_test" in n}


def test_no_r9_module_imports_an_r6_module():
    files = sorted(HERE.glob("r9_*.py")) + sorted(HERE.glob("run_r9_*.py"))
    assert {f.name for f in files} >= {"r9_context.py", "r9_bundle.py", "r9_picks.py", "r9_pins.py", "r9_score.py",
                                       "r9_common.py", "run_r9_stage0a.py"}
    # Ticket 13 explicitly calls the original R6 statistical producers. The
    # generic chain remains independent; this driver is its sole exception.
    found = {f.name: r6_names(f) for f in files if f.name != "run_r9_stage2.py" and r6_names(f)}
    assert not found, found
    assert r6_names(HERE / "run_r9_stage2.py") == {"run_r6_held", "run_r6_sensitivity", "r6_stats"}


def test_importing_the_r9_modules_loads_no_r6_module():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import r9_common, r9_backbone, r9_data, r9_heads, r9_episodes, r9_stats, r9_context, r9_bundle, "
        "r9_picks, r9_score, r9_pins, run_r9_stage0a\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('r6_', 'run_r6_')))\n"
        "path = [p for p in sys.path if p.endswith('20261125_artelingo_held_test')]\n"
        "print(bad, path)\n"
    ) % str(HERE)
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=str(HERE.parents[2]),
                         env={"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
                              "MKL_NUM_THREADS": "8", "PATH": "/usr/bin:/bin",
                              "HOME": os.environ.get("HOME", "/root")})  # git reads safe.directory from HOME
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[] []", out.stdout


def test_r9_common_does_not_put_round_6_on_sys_path():
    src = (HERE / "r9_common.py").read_text()
    assert "sys.path.append(str(R6D))" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("append", "insert"):
            assert "R6D" not in ast.unparse(node), ast.unparse(node)


# ---------------------------------------------------------------- module SHAs

def test_module_shas_cover_the_new_modules():
    shas = R.r9_module_shas()
    for name in ("r9_context.py", "r9_bundle.py", "r9_picks.py", "r9_pins.py", "run_r9_stage0a.py"):
        key = f"src/test/20261202_r9_scope/{name}"
        assert shas[key] == R.sha256_file(HERE / name), key
