"""Tests of r9_recipe_cells.py (ticket 11; rule section 3.2 item 3 and 3.4; R3 rule D4 to D9): the 224-cell picks, sigma*,
the switched-off flag, the cells record. Synthetic data only; no feature, label or file of the earlier rounds is read
(the pinned CLIP check runs inside the script, not here).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_recipe_cells.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_picks as P  # noqa: E402
import r9_recipe_cells as RCELLS  # noqa: E402

import numpy as np  # noqa: E402

F2, RF = R.F2, R.RF
N = 8
PARITY = np.arange(N) % 2                      # tune half 0 = episodes 0, 2, 4, 6; half 1 = 1, 3, 5, 7
H0, H1 = np.flatnonzero(PARITY == 0), np.flatnonzero(PARITY == 1)
CTRL = {0: (0.0, 4), 1: (0.0, 4)}              # (sigma*, rho_ctrl) per tune half


def table():
    z = lambda: np.zeros((224, N), np.int8)    # noqa: E731
    return z(), z(), z()


def put(arr, cell, half_idx, values):
    arr[cell, half_idx] = values


# ---------------------------------------------------------------- the min-margin pick (R3 rule D8 item 5)

def test_min_margin_pick_is_neither_max_rho_nor_max_gamma():
    fri, fgi, cri = table()
    # tune half 0: criterion min(rho - rho_ctrl, gamma), rho_ctrl = 4
    put(fri, 5, H0, [4, 4, 4, 4]), put(fgi, 5, H0, [0, 0, 0, 0])        # rho 16, gamma 0   -> min(12, 0) = 0  (max rho)
    put(fri, 9, H0, [3, 3, 2, 2]), put(fgi, 9, H0, [2, 2, 2, 1])        # rho 10, gamma 7   -> min(6, 7)  = 6  (the pick)
    put(fri, 100, H0, [2, 2, 2, 2]), put(fgi, 100, H0, [2, 2, 2, 2])    # rho 8,  gamma 8   -> min(4, 8)  = 4  (max gamma)
    # tune half 1 (its own values only; half 0's tune values of the cells above are all on other episodes)
    put(fri, 200, H1, [3, 3, 3, 3]), put(fgi, 200, H1, [3, 3, 3, 3])    # rho 12, gamma 12  -> 8
    put(fri, 41, H1, [4, 4, 4, 4]), put(fgi, 41, H1, [4, 4, 4, 4])      # rho 16, gamma 16  -> 12
    put(fri, 17, H1, [4, 4, 4, 4]), put(fgi, 17, H1, [4, 4, 4, 4])      # the same as cell 41: the lower number wins
    picks = RCELLS.pick_cells(fri, fgi, cri, CTRL, PARITY)
    assert picks["fpick"] == {0: 9, 1: 17}


def test_min_margin_arithmetic_with_a_control():
    fri, fgi, cri = table()
    put(fri, 3, H0, [2, 2, 2, 2]), put(fgi, 3, H0, [4, 4, 4, 4])        # rho 8,  gamma 16
    put(fri, 4, H0, [3, 3, 3, 3]), put(fgi, 4, H0, [1, 1, 1, 1])        # rho 12, gamma 4
    zero = {0: (0.0, 0), 1: (0.0, 0)}
    assert RCELLS.pick_cells(fri, fgi, cri, zero, PARITY)["fpick"][0] == 3      # min(8, 16) = 8 vs min(12, 4) = 4
    ctrl = {0: (0.0, 6), 1: (0.0, 0)}                                   # cell 3: min(8 - 6, 16) = 2; cell 4: min(12 - 6, 4) = 4
    assert RCELLS.pick_cells(fri, fgi, cri, ctrl, PARITY)["fpick"][0] == 4
    swapped = {0: (0.0, 0), 1: (0.0, 6)}                                # half 1's control does not touch half 0's pick
    assert RCELLS.pick_cells(fri, fgi, cri, swapped, PARITY)["fpick"][0] == 3


# ---------------------------------------------------------------- ties (R3 rule D8 item 5, D9)

def test_tie_goes_to_the_lowest_cell_number_fused_and_counterpart():
    fri, fgi, cri = table()
    for cell in (70, 33, 150):                                         # three cells with the same criterion
        put(fri, cell, H0, [3, 3, 3, 3]), put(fgi, cell, H0, [2, 2, 2, 2])
        put(cri, cell, H0, [3, 3, 3, 3])
    for cell in (210, 120):
        put(fri, cell, H1, [3, 3, 3, 3]), put(fgi, cell, H1, [3, 3, 3, 3])
        put(cri, cell, H1, [2, 2, 2, 2])
    picks = RCELLS.pick_cells(fri, fgi, cri, CTRL, PARITY)
    assert picks["fpick"] == {0: 33, 1: 120}
    assert picks["cpick"] == {0: 33, 1: 120}


def test_all_equal_table_picks_cell_zero():
    fri, fgi, cri = table()
    picks = RCELLS.pick_cells(fri, fgi, cri, CTRL, PARITY)
    assert picks["fpick"] == {0: 0, 1: 0} and picks["cpick"] == {0: 0, 1: 0}


def test_counterpart_takes_the_largest_rho_and_ignores_gain():
    fri, fgi, cri = table()
    put(cri, 60, H0, [1, 1, 1, 1]), put(cri, 61, H0, [4, 4, 4, 3]), put(cri, 62, H0, [4, 4, 4, 4])
    put(fgi, 60, H0, [4, 4, 4, 4])                                      # gain on the fused table does not enter
    put(cri, 62, H1, [4, 4, 4, 4]), put(cri, 63, H1, [3, 3, 3, 3])
    picks = RCELLS.pick_cells(fri, fgi, cri, CTRL, PARITY)
    assert picks["cpick"] == {0: 62, 1: 62}


def test_pick_cells_rejects_a_table_that_is_not_224_cells_or_has_one_half_only():
    fri, fgi, cri = table()
    with pytest.raises(AssertionError):
        RCELLS.pick_cells(fri[:223], fgi[:223], cri[:223], CTRL, PARITY)
    with pytest.raises(AssertionError):
        RCELLS.pick_cells(fri, fgi, cri, CTRL, np.zeros(N, int))
    with pytest.raises(AssertionError):
        RCELLS.pick_cells(fri, fgi[:, :N - 1], cri, CTRL, PARITY)


# ---------------------------------------------------------------- the switched-off flag (R3 rule D4; rule 3.4)

@pytest.mark.parametrize("cells,want", [
    ((8, 16), True),            # (t*7+u)*8 + a with a = 0 in both halves
    ((0, 0), True),
    ((8, 39), False),           # 39 = lambda_a 16
    ((39, 119), False),         # CLIP's AFF cells
    ((1, 0), False),            # lambda_a = 0.25 on half 0
    ((0, 7), False),
])
def test_switched_off_is_true_only_when_both_halves_have_lambda_a_zero(cells, want):
    assert RCELLS.switched_off(cells) is want
    assert all((F2.cell_values(c)[3] == 0) == (c % 8 == 0) for c in cells)    # cell number c: a index = c % 8


def test_cells_record_flag_follows_the_aff_cells_and_passes_the_schema(tmp_path):
    gates = {c: np.ones((4, N), np.float32) for c in "ab"}
    gates["b"][0, :4] = 0
    base = dict(aff=[8, 16], cf=[149, 10], r1_fused=[116, 119], r1_counterpart=[58, 123], sigma=[0.0, 0.0])
    rec = RCELLS.cells_record(base["aff"], base["cf"], base["r1_fused"], base["r1_counterpart"], base["sigma"],
                              gates, PARITY)
    assert rec["switched_off"] is True and rec["sigma_star"] == [0.0, 0.0]
    path = tmp_path / "cells.json"
    path.write_text(json.dumps(rec))
    assert P.read_cells(path) == rec
    rec2 = RCELLS.cells_record([8, 39], base["cf"], base["r1_fused"], base["r1_counterpart"], base["sigma"], gates,
                               PARITY)
    assert rec2["switched_off"] is False
    # gate share: half 0 uses tau of cell 8 (index 0) on parity 1, half 1 tau of cell 16 (index 0) on parity 0
    assert rec["gate_total"] == 2 * 2 * N // 2 and 0 < rec["gate_open"] < rec["gate_total"]
    assert rec["gate_share"] == rec["gate_open"] / rec["gate_total"]


# ---------------------------------------------------------------- the family on synthetic scores

def synth_bundle(n=128, seed=0):
    """Scores with a signal: candidate 0 (condition a) and 1 (condition b) lead in B and more in T."""
    rng = np.random.default_rng(seed)
    parity = np.arange(n) % 2
    B = {c: {} for c in "ab"}
    T = {c: {} for c in "ab"}
    for d in ("i2t", "t2i"):
        b = rng.normal(size=(n, 13)).astype(np.float32)
        b[:, :2] += 0.8
        for c, col in (("a", 0), ("b", 1)):
            B[c][d] = b.copy()
            t = rng.normal(size=(n, 13)).astype(np.float32)
            t[:, col] += rng.uniform(0, 3, size=n).astype(np.float32)
            T[c][d] = t
    m = {c: rng.uniform(size=n) for c in "ab"}
    gates = [{c: (m[c] >= q).astype(np.float32) for c in "ab"} for q in (0.0, 0.25, 0.5, 0.75)]
    return SimpleNamespace(B=B, parity=parity, n=n), T, gates


def test_family_picks_equal_round_3_run_family():
    """R3's run_family is the rule's code (D8, D9); this module's picks on the same scores must equal it exactly."""
    for seed in (0, 1, 2):
        bundle, T, gates = synth_bundle(seed=seed)
        ours = RCELLS.family_picks(bundle.B, T, gates, bundle.parity)
        theirs = RF.run_family(bundle, T, gates)
        assert ours["fpick"] == theirs["fpick"] and ours["cpick"] == theirs["cpick"]
        assert ours["sigma"] == theirs["sigma"]
        assert ours["details"] == theirs["details"]


def test_family_picks_equal_the_brainstorm_family():
    """bs_05_aff.py (CLIP's cells were picked there) runs L.Family(...).crossfit() on the tied family: the same cells,
    in (tau index, lambda_u, lambda_a) form, on the same z-scored terms (float64 there, float32 here)."""
    sys.path.insert(0, str(R.TEST / "20261120_r1_levers_brainstorm"))
    import bs_lib as L
    for seed in (0, 1):
        bundle, T, gates = synth_bundle(seed=seed)
        ours = RCELLS.family_picks(bundle.B, T, gates, bundle.parity)
        zB = {c: {d: L.zrows(bundle.B[c][d]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
        data = SimpleNamespace(zB=zB, E=bundle.n, parity=bundle.parity, ctrl=F2.control_choice(zB, bundle.parity))
        zT = L.zterm(T)
        fam = L.Family(data, [L.MD(zT, g) for g in gates], tied=True).stats()
        fp, cp = fam.crossfit()
        assert [F2.cell_values(ours["fpick"][h])[1:] for h in (0, 1)] == \
               [(fam.cells[fp[h]][0], fam.cells[fp[h]][1], fam.cells[fp[h]][2]) for h in (0, 1)]
        assert [F2.cell_values(ours["cpick"][h])[1:] for h in (0, 1)] == \
               [tuple(fam.cf_cells[cp[h]]) for h in (0, 1)]


def test_aff_and_r1_sigma_must_agree():
    """sigma* depends on B and the parity only (D8 item 5): the record takes it from the AFF family and asserts R1's."""
    bundle, T, gates = synth_bundle()
    fam = RCELLS.family_picks(bundle.B, T, gates, bundle.parity)
    other = dict(fam, sigma={0: 0.25, 1: 0.0})
    with pytest.raises(AssertionError, match="sigma"):
        RCELLS.sigma_star(fam, other)
    assert RCELLS.sigma_star(fam, fam) == [fam["sigma"][0], fam["sigma"][1]]


# ---------------------------------------------------------------- the CLIP check (inside the script)

def good_clip_record():
    from r9_backbone import parts
    return P.read_cells(parts("clip_b32")["cells_seed42"])


def test_check_clip_accepts_the_pinned_record_and_names_a_changed_cell():
    rec = good_clip_record()
    RCELLS.check_clip(rec)
    bad = dict(rec, aff=[39, 118])
    with pytest.raises(AssertionError, match="aff"):
        RCELLS.check_clip(bad)
    bad = dict(rec, sigma_star=[0.0, 0.25])
    with pytest.raises(AssertionError, match="sigma_star"):
        RCELLS.check_clip(bad)
    bad = dict(rec, switched_off=True)
    with pytest.raises(AssertionError, match="switched_off"):
        RCELLS.check_clip(bad)


# ---------------------------------------------------------------- the files

def test_record_files_refuse_to_overwrite(tmp_path):
    rec = good_clip_record()
    path = RCELLS.write_cells(tmp_path, rec, backbone="clip_b32", input_sha256={"x": "0" * 64}, upstream_sha256={},
                              extra={})
    assert path.name == "cells_seed42.json" and (tmp_path / "cells_seed42.json.record.json").is_file()
    assert P.read_cells(path) == rec
    meta = json.loads((tmp_path / "cells_seed42.json.record.json").read_text())
    assert set(meta["keys_made"]) == set(rec) == set(P.CELLS_KEYS)
    assert meta["keys_copied"] == [] and meta["script_sha256"] == R.sha256_file(RCELLS.__file__)
    with pytest.raises(FileExistsError):
        RCELLS.write_cells(tmp_path, rec, backbone="clip_b32", input_sha256={}, upstream_sha256={}, extra={})
