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
import r9_bundle as RB  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_recipe_cells as RCELLS  # noqa: E402

import numpy as np  # noqa: E402

F2, RF = R.F2, R.RF
N = 8
AFFECT = 0                                     # the grouping index of affect in A0 (rule D1)
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


def test_family_picks_equal_the_brainstorm_family():
    """family_picks is r3_fusion.run_family (L2). bs_05_aff.py (CLIP's cells were picked there) runs L.Family(...).crossfit() on the tied family: the same cells,
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


# ---------------------------------------------------------------- compute_cells wiring (L3, M1, M2, L4)

BACKBONE = "qwen3vl_emb_2b"                    # any backbone but CLIP: the CLIP constants are not in play


class World:
    """A fixed {T, m, pick} reader, a fake cross-fit and a picks_seed42.json, wired into compute_cells."""

    def __init__(self, monkeypatch, tmp_path, seed, n=128, b_mean_shift=0.0, bpick_in_file=None):
        bundle, self.T, _ = synth_bundle(n=n, seed=seed)
        self.B, self.parity, self.n = bundle.B, bundle.parity, n
        rng = np.random.default_rng(100 + seed)
        self.m = {c: rng.uniform(size=n) for c in "ab"}
        self.pick = {c: rng.integers(0, 3, size=n) for c in "ab"}
        self.taus = [float(x) for x in RCELLS.RC.thresholds(self.m)[0]]
        self.bpick = {0: [0.5, 0.25], 1: [0.0, 0.0]}
        self.mean = 100 * float(np.mean(RCELLS.per_anchor(self.B)["r1"])) + b_mean_shift
        file_pick = bpick_in_file or self.bpick
        # B0 and B1 differ from B (and from each other), so that reading another entry of the file is a failure (F1)
        other = {"B0": {0: [0.5, 0.5], 1: [0.0, 0.25]}, "B1": {0: [0.0, 0.5], 1: [0.5, 0.0]}}
        rec = {"B": {str(h): list(file_pick[h]) for h in (0, 1)},
               **{name: {str(h): list(pick[h]) for h in (0, 1)} for name, pick in other.items()}}
        rec.update({"mean_r1": {"B": self.mean, "B0": 1.5, "B1": 2.5}, "passed": True})
        self.picks_file = tmp_path / "picks_seed42.json"
        self.picks_file.write_text(json.dumps(rec))
        monkeypatch.setattr(RF, "reader", lambda bundle, readers: {"T": self.T, "m": self.m, "pick": self.pick})
        monkeypatch.setattr(RCELLS, "crossfit_condition_free", lambda cos, tu, t6, parity: (self.B, self.bpick))
        monkeypatch.setattr(RCELLS, "parts", lambda backbone: {"picks_seed42": self.picks_file})
        self.bundle = SimpleNamespace(parity=self.parity, cos=None, t_n1u=None, t6u_B=None)

    def run(self, taus=None):
        return RCELLS.compute_cells(self.bundle, None, self.taus if taus is None else taus, BACKBONE)

    def family(self, gates):
        return RF.run_family(SimpleNamespace(B=self.B, parity=self.parity), self.T, gates)


@pytest.fixture(params=[0, 2, 3, 4, 5])
def world(request, monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path, request.param)


def test_compute_cells_wiring_aff_on_aff_gates_r1_on_r1_gates_and_the_gate_share_at_aff_cells(world):
    rec, details = world.run()
    aff = world.family(RF.gates_aff(world.m, world.pick, world.taus))
    r1 = world.family(RF.gates_r1(world.m, world.taus))
    assert rec["aff"] == [aff["fpick"][h] for h in (0, 1)] and rec["cf"] == [aff["cpick"][h] for h in (0, 1)]
    assert rec["r1_fused"] == [r1["fpick"][h] for h in (0, 1)]
    assert rec["r1_counterpart"] == [r1["cpick"][h] for h in (0, 1)]
    assert rec["sigma_star"] == [aff["sigma"][0], aff["sigma"][1]]
    # the hand count: AFF's gate of the tau of AFF's cell of tune half h, on the episodes of parity 1 - h, conditions a, b
    want = 0
    for h in (0, 1):
        tau = world.taus[rec["aff"][h] // 56]
        for c in "ab":
            want += int(((world.m[c] >= tau) & (world.pick[c] == AFFECT))[world.parity == 1 - h].sum())
    assert rec["gate_open"] == want and rec["gate_total"] == 2 * world.n
    assert rec["switched_off"] is RCELLS.switched_off(rec["aff"])
    assert details["taus"] == world.taus and details["b_mean_r1"] == world.mean


def test_the_wiring_test_can_tell_the_mutants_apart(world):
    """The data of this fixture must make the four wirings differ, or the test above would not notice a swap."""
    rec, _ = world.run()
    cf_cells = rec["cf"]
    hand = lambda cells: sum(int(((world.m[c] >= world.taus[cells[h] // 56]) & (world.pick[c] == AFFECT))
                                 [world.parity == 1 - h].sum()) for h in (0, 1) for c in "ab")   # noqa: E731
    r1_on_aff = world.family(RF.gates_aff(world.m, world.pick, world.taus))
    aff_on_r1 = world.family(RF.gates_r1(world.m, world.taus))
    differs = [rec["aff"] != [aff_on_r1["fpick"][h] for h in (0, 1)] or rec["cf"] != [aff_on_r1["cpick"][h] for h in (0, 1)],
               rec["r1_fused"] != [r1_on_aff["fpick"][h] for h in (0, 1)]
               or rec["r1_counterpart"] != [r1_on_aff["cpick"][h] for h in (0, 1)],
               hand(rec["aff"]) != hand(cf_cells)]
    assert all(differs), differs


def test_compute_cells_rejects_taus_that_are_not_the_bundles_own_margins(world):
    """M1: the gates are cut at the thresholds of the bundle's R1 margins (rc_core.thresholds), whatever file they came from."""
    world.run()                                                                 # the right taus pass
    with pytest.raises(AssertionError, match="tau"):
        world.run(taus=world.taus[::-1])
    with pytest.raises(AssertionError, match="tau"):
        world.run(taus=[world.taus[0], world.taus[1], world.taus[2], world.taus[3] + 1e-9])


def test_load_taus_keeps_the_files_order_and_feeds_compute_cells(world, monkeypatch, tmp_path):
    """The thresholds are read in the file's order (tau index 0 is the lowest); a reversed list is not the bundle's own."""
    f = tmp_path / "rc_tau.json"
    f.write_text(json.dumps({"taus": world.taus}))
    monkeypatch.setattr(RCELLS, "parts", lambda backbone: {"rc_tau": f, "picks_seed42": world.picks_file})
    taus = RCELLS.load_taus(BACKBONE)
    assert taus == world.taus and taus == sorted(taus)
    world.run(taus=taus)
    f.write_text(json.dumps({"taus": world.taus[:3]}))
    with pytest.raises(AssertionError):
        RCELLS.load_taus(BACKBONE)


def test_compute_cells_rejects_a_b_that_is_not_the_one_of_picks_seed42(monkeypatch, tmp_path):
    """M2: B cross-fitted here must be the B that the held read will rebuild from picks_seed42.json."""
    World(monkeypatch, tmp_path, 0).run()
    with pytest.raises(AssertionError, match="B"):
        World(monkeypatch, tmp_path, 0, bpick_in_file={0: [0.5, 0.25], 1: [0.5, 0.0]}).run()
    with pytest.raises(AssertionError, match="mean R@1"):
        World(monkeypatch, tmp_path, 0, b_mean_shift=1e-9).run()


def test_check_b_picks_reads_the_backbones_file(monkeypatch, tmp_path):
    w = World(monkeypatch, tmp_path, 0)
    RCELLS.check_b_picks(w.bpick, w.mean, BACKBONE)
    with pytest.raises(AssertionError, match="B"):
        RCELLS.check_b_picks({0: [0.5, 0.25], 1: [0.5, 0.0]}, w.mean, BACKBONE)
    with pytest.raises(AssertionError, match="mean R@1"):
        RCELLS.check_b_picks(w.bpick, w.mean + 1e-6, BACKBONE)


def text_mutant(tmp_path, old, new, count=1):
    """An import of r9_recipe_cells.py with ``old`` replaced by ``new`` (``count`` places), for a mutation test."""
    import importlib.util
    src = (HERE / "r9_recipe_cells.py").read_text()
    assert src.count(old) == count, (old, src.count(old))
    path = tmp_path / f"cells_mut_{abs(hash(old + new))}.py"
    path.write_text(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_fixture_gives_b_b0_and_b1_different_picks_and_means(world):
    picks = P.load_picks(world.picks_file, BACKBONE)
    assert picks["B"] != picks["B0"] and picks["B"] != picks["B1"] and picks["B0"] != picks["B1"]
    means = json.loads(world.picks_file.read_text())["mean_r1"]
    assert len({means["B"], means["B0"], means["B1"]}) == 3


@pytest.mark.parametrize("old,new", [('P.load_picks(path, backbone)["B"]', 'P.load_picks(path, backbone)["B0"]'),
                                     ('P.load_picks(path, backbone)["B"]', 'P.load_picks(path, backbone)["B1"]'),
                                     ('["mean_r1"]["B"]', '["mean_r1"]["B0"]')])
def test_check_b_picks_must_read_the_b_entry_of_the_file(monkeypatch, tmp_path, old, new):
    """F1: a version that reads the B0 or B1 entry (the picks or the mean R@1) is refused on a file whose three differ."""
    w = World(monkeypatch, tmp_path, 0)
    RCELLS.check_b_picks(w.bpick, w.mean, BACKBONE)
    mut = text_mutant(tmp_path, old, new)
    monkeypatch.setattr(mut, "parts", lambda backbone: {"picks_seed42": w.picks_file})
    with pytest.raises(AssertionError, match="B"):
        mut.check_b_picks(w.bpick, w.mean, BACKBONE)


def force_other_float64_picks(monkeypatch):
    """bs_lib's tied family (the float64 check) answers other picks than the float32 family does."""
    sys.path.insert(0, str(R.TEST / "20261120_r1_levers_brainstorm"))
    import bs_lib as L
    real = L.Family.crossfit
    monkeypatch.setattr(L.Family, "crossfit", lambda self: tuple({0: 223, 1: 223} for _ in real(self)))


def test_float64_check_leaves_its_run_untouched(world, monkeypatch):
    """F3 (unit): `float64_tied_check` reads the family's picks and writes none of them."""
    import copy
    gates = RF.gates_r1(world.m, world.taus)
    run = world.family(gates)
    before = copy.deepcopy(run)
    force_other_float64_picks(monkeypatch)
    out = RCELLS.float64_tied_check(world.B, world.T, gates, world.parity, run)
    assert out["fpick_equal"] is False and out["cpick_equal"] is False       # the forced answer is seen ...
    assert set(run) == set(before)
    for key in ("fpick", "cpick", "sigma", "ctrl"):
        assert run[key] == before[key], key                                  # ... and changes nothing in the run


def test_a_different_float64_answer_changes_no_pick_and_no_output(world, monkeypatch):
    """F3: the float64 table is recorded, never used: another float64 answer gives the same record and details."""
    rec, details = world.run()
    force_other_float64_picks(monkeypatch)
    rec2, details2 = world.run()
    assert details2["float64_tied_family"]["equal"] is False        # the other answer is seen and recorded ...
    assert rec2 == rec
    assert {k: v for k, v in details2.items() if k != "float64_tied_family"} == \
        {k: v for k, v in details.items() if k != "float64_tied_family"}


def test_the_float64_tied_family_check_is_recorded_and_decides_nothing(world):
    """L4: details say whether bs_lib's float64 tied family gives the same integer tables and picks."""
    _, details = world.run()
    chk = details["float64_tied_family"]
    assert set(chk) == {"equal", "aff", "r1"} and isinstance(chk["equal"], bool)
    for fam in ("aff", "r1"):
        assert set(chk[fam]) == {"tables_equal", "fused_cells_differ", "cf_cells_differ", "fpick_equal", "cpick_equal"}
    assert chk["equal"] == all(chk[f]["tables_equal"] and chk[f]["fpick_equal"] and chk[f]["cpick_equal"]
                               for f in ("aff", "r1"))


def test_differing_cells_names_the_rows_that_differ():
    x = np.zeros((224, 6), np.int8)
    y = x.copy()
    y[77, 3] = 1
    y[5, 0] = -1
    assert RCELLS.differing_cells(x, x) == [] and RCELLS.differing_cells(x, y) == [5, 77]
    with pytest.raises(AssertionError):
        RCELLS.differing_cells(x, y[:, :5])


# ---------------------------------------------------------------- the CLIP checks fail on a changed cell (L3)

def test_check_brainstorm_names_a_changed_cell():
    rec = good_clip_record()
    RCELLS.check_brainstorm(rec)
    for key in ("aff", "cf", "r1_fused", "r1_counterpart"):
        bad = dict(rec, **{key: [rec[key][0], (rec[key][1] + 8) % 224]})
        with pytest.raises(AssertionError, match=key):
            RCELLS.check_brainstorm(bad)


# ---------------------------------------------------------------- the episodes and the stack checks (L6, L1)

def test_check_episode_sha_compares_with_the_frozen_hashes(monkeypatch):
    want = {pair: f"{i:064x}" for i, pair in enumerate(R.PAIR_NAMES)}
    monkeypatch.setattr(E, "identity_targets", lambda: {R.DEV_SEED: {"n_per_pair": 4096, "episodes_sha256": dict(want)}})
    RCELLS.check_episode_sha(dict(want))
    bad = dict(want, **{R.PAIR_NAMES[0]: "f" * 64})
    with pytest.raises(AssertionError, match=R.PAIR_NAMES[0]):
        RCELLS.check_episode_sha(bad)


def stack_f(n=6, seed=0):
    rng = np.random.default_rng(seed)
    stack = {d: rng.normal(size=(n, len(RB.A0), RB.K_CAND)).astype(np.float32) for d in RB.DIRECTIONS}
    fa = rng.normal(size=(n, RB.N_FEATURES))
    fb = rng.normal(size=(n, RB.N_FEATURES))
    for j in range(len(RB.A0)):
        fb[:, 6 * j + 2] = -fa[:, 6 * j + 2]
    return SimpleNamespace(n=n, stack=stack, F={"a": fa, "b": fb})


def test_check_stack_f_is_the_stack_f_and_delta_part_of_validate():
    b = stack_f()
    RB.check_stack_f(b)
    bad = stack_f()
    bad.stack["i2t"][2, 1, 3] = np.nan
    with pytest.raises(AssertionError, match="stack"):
        RB.check_stack_f(bad)
    bad = stack_f()
    bad.F["a"][1, 4] = np.inf
    with pytest.raises(AssertionError, match="F/a"):
        RB.check_stack_f(bad)
    bad = stack_f()
    bad.F["b"][0, 2] += 1.0
    with pytest.raises(AssertionError, match="Delta"):
        RB.check_stack_f(bad)
    bad = stack_f()
    bad.F["a"] = bad.F["a"].astype(np.float32)
    with pytest.raises(AssertionError, match="F/a"):
        RB.check_stack_f(bad)


def scores_n(n, seed=0):
    """Finite float32 scores {c: {d: (n, 13)}}, condition-free (a and b equal)."""
    rng = np.random.default_rng(seed)
    one = {d: rng.normal(size=(n, RB.K_CAND)).astype(np.float32) for d in RB.DIRECTIONS}
    return {c: {d: x.copy() for d, x in one.items()} for c in RB.CONDITIONS}


def test_validate_runs_check_stack_f():
    """F2: a NaN in `stack` and a broken Delta are refused by `validate` itself, not only by `check_stack_f`."""
    n = 6
    sf = stack_f(n)
    ep = np.arange(n)
    b = SimpleNamespace(n=n, cl=ep.copy(), parity=ep % 2, pair_index=ep // 2, anchor=ep.copy(),
                        cos=scores_n(n, 1), t_n1u=scores_n(n, 2), t6u_B=scores_n(n, 3), t6u_B0=scores_n(n, 4),
                        t6u_B1=scores_n(n, 5), post={h: None for h in RB.A0}, stack=sf.stack, F=sf.F,
                        pm_terms={k: scores_n(n, 7 + i) for i, k in enumerate(RB.PM_TERM_NAMES)},
                        episodes_sha256={pair: "0" * 64 for pair in R.PAIR_NAMES}, module_sha256={})
    assert RB.validate(b)["delta_b_equals_minus_delta_a"] is True
    nan = SimpleNamespace(**vars(b))
    nan.stack = {d: x.copy() for d, x in b.stack.items()}
    nan.stack["t2i"][3, 0, 5] = np.nan
    with pytest.raises(AssertionError, match="stack"):
        RB.validate(nan)
    bad_delta = SimpleNamespace(**vars(b))
    bad_delta.F = {c: x.copy() for c, x in b.F.items()}
    bad_delta.F["b"][0, 2] += 1.0
    with pytest.raises(AssertionError, match="Delta"):
        RB.validate(bad_delta)


def test_cells_bundle_runs_check_stack_f(monkeypatch):
    """F2: `cells_bundle` refuses a bundle whose `stack` has a NaN (its other steps are replaced by stand-ins)."""
    n = 6
    sf = stack_f(n)
    state = {"stack": sf.stack}
    ctx = SimpleNamespace(pooled=SimpleNamespace(), n=n, post={h: {"img": np.ones((n, 2)), "txt": np.ones((n, 2))}
                                                               for h in (*RB.A0, "affect_km")}, rows=np.arange(n), parity=np.arange(n) % 2,
                          pair_index=np.arange(n) // 2, cos=scores_n(n, 1), mode="selection", seed=42, smoke=False)
    monkeypatch.setattr(RB, "check_context", lambda c: {})
    monkeypatch.setattr(RB, "model_inputs", lambda c, backbone: None)
    monkeypatch.setattr(RCELLS, "centered_term", lambda inp, ep, uniform: scores_n(n, 2))
    monkeypatch.setattr(RCELLS, "uniform_probe_scores", lambda post, ep, parts_: scores_n(n, 3))
    monkeypatch.setattr(RCELLS.C, "grouping_stack", lambda post, ep, groupings: state["stack"])
    monkeypatch.setattr(RCELLS.rbe, "seed42_features", lambda c, groupings: (sf.F, {}))
    reached = []
    monkeypatch.setattr(RCELLS.P, "check_seed42", lambda b: reached.append(b))
    assert RCELLS.cells_bundle(ctx, BACKBONE).stack is sf.stack and len(reached) == 1       # the good one passes
    bad = {d: x.copy() for d, x in sf.stack.items()}
    bad["i2t"][2, 1, 3] = np.nan
    state["stack"] = bad
    with pytest.raises(AssertionError, match="stack"):
        RCELLS.cells_bundle(ctx, BACKBONE)
    assert len(reached) == 1                                                                 # refused before check_seed42


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
