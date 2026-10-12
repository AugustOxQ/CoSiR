"""Ticket 14: known frozen numbers and synthetic rule boundaries (§9, G1–G4)."""
import numpy as np
import pytest

import r7_common as R
from rederive import rd7_bundle as BU, rd7_core as K, rd7_family as F


@pytest.fixture(scope="module")
def legacy():
    return BU.build_legacy(42)


def test_aff_equals_round3_per_anchor(legacy):
    reader = BU.read(legacy, K.A0, BU.legacy_reader("A0"))
    gates = F.gates_aff(reader["m"], reader["pick"], K.RULE_TAUS)
    rec, fused, cf, _ = F.run_family(legacy.B, reader["T"], gates, K.RULE_TAUS, legacy.parity)
    ref = BU.checked_npz(R.R3D / "results/seed42_arrays.npz")
    for metric in K.METRICS:
        np.testing.assert_array_equal(fused[metric], ref[f"aff_fused__{metric}"])
        np.testing.assert_array_equal(cf[metric], ref[f"aff_cf__{metric}"])
    assert [rec["fused_cells"][h]["cell"] for h in "01"] == [39, 119]
    assert [rec["counterpart_cells"][h]["cell"] for h in "01"] == [149, 10]
    assert [rec["control"][h]["sigma_star"] for h in "01"] == [0, 0]


def test_condition_free_baselines_equal_frozen_numbers(legacy):
    got = BU.baselines(legacy)
    ref = BU.checked_npz(R.R3D / "rederive/out/rd3_bundle_seed42.npz")
    for name, prefix, point in (("B", "B", 18.341064453125), ("B0", "Bp", 18.436686197916664)):
        expected = K.metrics({c: {d: ref[f"{prefix}__{c}__{d}"] for d in K.DIRS} for c in K.COND})
        for metric in K.METRICS:
            np.testing.assert_array_equal(got[name][metric], expected[metric])
        assert 100 * got[name]["r1"].mean() == point
    assert 100 * got["B1"]["r1"].mean() == 18.804931640625
    aff = BU.checked_npz(R.R3D / "results/seed42_arrays.npz")
    r6 = BU.checked_npz(R.TEST / "20261125_artelingo_held_test/results/seed42_per_episode.npz")
    np.testing.assert_array_equal(aff["aff_fused__r1"] - got["B1"]["r1"], r6["diff__S1"])


def test_r1_a1_equals_round1_and_xaff_all_positions(legacy):
    parts = (*K.A0, "csd")
    reader = BU.read(legacy, parts, BU.legacy_reader("A1"))
    taus = K.thresholds(reader["m"]["a"], reader["m"]["b"])
    got = BU.score(legacy, reader, taus, "R1")
    ref = BU.checked_npz(R.TEST / "20261119_no_caption_csd_spike/results/sp_cand_R1_A1_224.npz")
    np.testing.assert_array_equal(got["fused"]["r1"], ref["fused_r1"])
    np.testing.assert_array_equal(got["fused"]["r1"] - got["cf"]["r1"], ref["margin_r1"])
    xaff = BU.score(legacy, reader, taus, "XAFF", allowed=(0, 1, 2, 3))
    for who in ("fused", "cf"):
        for metric in K.METRICS:
            np.testing.assert_array_equal(xaff[who][metric], got[who][metric])


def test_told_a1_equals_frozen_spike(legacy):
    got = BU.told(legacy, "csd")
    ref = BU.checked_npz(R.TEST / "20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz")
    np.testing.assert_array_equal(got["fused"]["r1"], ref["fused_r1"])
    np.testing.assert_array_equal(got["fused"]["r1"] - got["cf"]["r1"], ref["margin_r1"])
    assert got["gate_a"].shape == (4, 12288) and got["gate_a"].all() and got["gate_b"].all()


def test_gate_threshold_equalities_and_allowed_positions():
    m = {"a": np.array([0., .25, .5, .75]), "b": np.array([.75, .5, .25, 0.])}
    pick = {"a": np.array([0, 1, 3, 2]), "b": np.array([3, 2, 1, 0])}
    got = F.gates_x_aff(m, pick, (0., .25, .5, .75))
    assert np.stack([g["a"] for g in got]).tolist() == [
        [True, False, True, False], [False, False, True, False],
        [False, False, True, False], [False, False, False, False]]
    assert np.stack([g["b"] for g in got]).tolist() == [
        [True, False, False, True], [True, False, False, False],
        [True, False, False, False], [True, False, False, False]]
    pi, margin = K.picks_margins(np.array([[.4, .4, .1, .1], [.1, .1, .4, .4]]))
    assert pi.tolist() == [0, 2] and margin.tolist() == [0., 0.]


def test_224_cells_tie_to_first_and_metrics_are_strict():
    scores = {c: {d: np.zeros((4, 13), np.float32) for d in K.DIRS} for c in K.COND}
    gates = [{c: np.ones(4, bool) for c in K.COND} for _ in range(4)]
    rec, pn, pc, extra = F.run_family(scores, scores, gates, (0., .25, .5, .75), np.array([0, 1, 0, 1]))
    assert extra["rho_f"].shape == (224, 2) and extra["rho_f"].dtype == np.int64
    assert not extra["rho_f"].any() and not extra["gam_f"].any()
    for h in "01":
        assert rec["fused_cells"][h]["cell"] == rec["counterpart_cells"][h]["cell"] == 0
        assert rec["control"][h]["sigma_star"] == 0
    assert all(not p[m].any() for p in (pn, pc) for m in K.METRICS)
    assert K.cell_decode(223) == (3, 6, 7) and K.cell_id(3, 6, 7) == 223


def test_phases_write_shared_names_and_reject_wrong_seed_roles():
    from rederive import rd7_arrays as A
    score = {"record": {"fused_cells": {h: {"cell": 39} for h in "01"},
                        "counterpart_cells": {h: {"cell": 149} for h in "01"},
                        "control": {h: {"sigma_star": 0.} for h in "01"}},
             "fused": {m: np.zeros(2, np.float64) for m in K.METRICS},
             "cf": {m: np.zeros(2, np.float64) for m in K.METRICS},
             "gate_a": np.ones((4, 2), np.float32), "gate_b": np.ones((4, 2), np.float32)}
    out = {}
    A.add_family(out, "XAFF_V1", score)
    assert set(out) == {f"{name}__{m}" for name in ("XAFF_V1", "XAFF_V1_cf") for m in K.METRICS} | {
        f"XAFF_V1__{m}" for m in ("fused_cells", "cf_cells", "sigma", "gate_a", "gate_b")}
    assert out["XAFF_V1__fused_cells"].dtype == np.int64
    assert A.phase_path("A", 9001).name == "phaseA_seed9001.npz"
    assert A.phase_path("K", 9001).name == "phaseK.npz"
    assert A.phase_path("B", 9002).name == "phaseB_seed9002.npz"
    for phase, seed, chain in (("A", 55, False), ("K", 9002, True), ("B", 42, False), ("A", 52, False)):
        with pytest.raises(AssertionError, match="phase seed role"):
            A.check_phase_seed(phase, seed, chain)


def test_reader_uses_checked_stored_probabilities_and_rejects_mutation(legacy, tmp_path):
    pk = BU.legacy_reader("A0")
    expected = BU.read(legacy, K.A0, pk)
    stored = {f"P__{c}": expected["P"][c].copy() for c in K.COND}
    # Within the specified probability tolerance; gates must use stored P.
    p = stored["P__a"][0]
    p[p.argmax()] += 2e-13
    p[p.argmin()] -= 2e-13
    for c in K.COND:
        pi, margin = K.picks_margins(stored[f"P__{c}"])
        stored[f"m__{c}"], stored[f"pick__{c}"] = margin, pi
    path = tmp_path / "reader.npz"
    np.savez(path, **stored)
    checked = BU.read(legacy, K.A0, pk, path)
    np.testing.assert_array_equal(checked["P"]["a"], stored["P__a"])
    np.testing.assert_array_equal(checked["m"]["a"], stored["m__a"])
    stored["P__a"][0, 0] += 1e-4
    np.savez(path, **stored)
    with pytest.raises(AssertionError, match="stored reader P"):
        BU.read(legacy, K.A0, pk, path)


@pytest.fixture
def synthetic_frozen(tmp_path):
    """Known rankings: R0 reinforces a negative; X can steer to either target.

    Reader examples pick all four positions, so closing image/caption gates
    changes hits. The fitted margins exceed AFF's taus and define L12's taus.
    """
    import json
    import pickle
    from types import SimpleNamespace
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = 12
    # Distinct, sparse global row IDs for queries, candidates and both banks.
    rows = np.arange(2, 2 + 30 * n * 2, 2, dtype=np.int64)
    blocks = rows.reshape(n, 30)
    ep = SimpleNamespace(anchor=blocks[:, 0], candidates=blocks[:, 1:14],
                         pairs_a_img=blocks[:, 14:18], pairs_a_txt=blocks[:, 18:22],
                         pairs_b_img=blocks[:, 22:26], pairs_b_txt=blocks[:, 26:30])
    inside = np.zeros(rows[-1] + 2, bool)
    inside[rows] = True
    def condition_free_scores(negative=False):
        values = {d: np.zeros((n, 13), np.float32) for d in K.DIRS}
        if negative:
            for v in values.values():
                v[:, 2] = 1
        return {c: values for c in K.COND}
    ctx = SimpleNamespace(n=n, seed=9001, selection=rows, groups=np.zeros(len(inside), np.int64),
                          in_sel=inside, pooled=ep, cos=condition_free_scores(True))
    b = SimpleNamespace(ctx=ctx, B=condition_free_scores(True), t_n1u=condition_free_scores(),
                        parity=np.arange(n) % 2, cl=np.arange(n) // 2,
                        pair_index=np.repeat(np.arange(3), 4), post={})
    (tmp_path / "heads").mkdir()
    (tmp_path / "readers").mkdir()
    seed_dir = tmp_path / "seed9001"
    seed_dir.mkdir()
    dev_dir = tmp_path / "seed42"
    dev_dir.mkdir()
    ha, hb = np.arange(n) % 4, (np.arange(n) + 1) % 4
    for name in (*K.A0, "R0", "T", *(f"V{k}" for k in range(1, 7))):
        position = K.A0.index(name) if name in K.A0 else 3
        post = {}
        for m in ("img", "txt"):
            full = np.full((len(inside), 3), np.nan, np.float32)
            full[rows] = [0, 1, 0]
            full[ep.anchor] = [1, 0, 0]
            for i in range(n):
                for col, h, strength in ((0, ha[i], .9), (1, hb[i], .7)):
                    if position == h and name != "R0":
                        full[ep.candidates[i, col]] = [strength, 1 - strength, 0]
                if name == "R0":
                    full[ep.candidates[i, 2]] = [.95, .05, 0]
                for bank, h in (("a", ha[i]), ("b", hb[i])):
                    full[getattr(ep, f"pairs_{bank}_img")[i]] = [1, 0, 0]
                    strength = .55 + .15 * (i // 4)
                    full[getattr(ep, f"pairs_{bank}_txt")[i]] = [strength, 1 - strength, 0] if position == h else [0, 1, 0]
            post[m] = full
        b.post[name] = post
        np.savez(tmp_path / f"heads/post_{name}.npz", rows=rows, **post)
    feats = K.features_both(b.post, (*K.A0, "V1"), ep)
    train = np.concatenate([feats[c] for c in K.COND])
    scaler = StandardScaler().fit(train)
    model = LogisticRegression(C=100, max_iter=300).fit(scaler.transform(train), np.concatenate([ha, hb]))
    taus = {}
    for name in ("A0", *(f"V{k}" for k in range(1, 7))):
        parts = K.A0 if name == "A0" else (*K.A0, name)
        pk = BU.legacy_reader("A0") if name == "A0" else {
            "groupings": parts, "feature_names": K.feature_names(parts),
            "halves": [{"model": model, "scaler": scaler} for _ in range(2)]}
        if name != "A0":
            with (tmp_path / f"readers/reader_{name}.pkl").open("wb") as f:
                pickle.dump(pk, f)
        reader = BU.read(b, parts, pk)
        arrays = {f"{field}__{c}": reader[field][c] for field in ("P", "m", "pick") for c in K.COND}
        np.savez(seed_dir / f"reader_{name}.npz", **arrays)
        if name != "A0":
            # A synthetic stored development snapshot; no seed-42 scoring.
            np.savez(dev_dir / f"reader_{name}.npz", **arrays)
            taus[name] = K.thresholds(reader["m"]["a"], reader["m"]["b"])
    (tmp_path / "taus.json").write_text(json.dumps(taus))
    return b, tmp_path


def test_three_phases_cover_candidates_and_picks_on_smoke(synthetic_frozen):
    import json
    from rederive import rd7_arrays as A
    b, root = synthetic_frozen
    k = A.derive("K", b, root)
    assert k["told_T__gate_a"].all() and k["told_T__gate_b"].all()
    a = A.derive("A", b, root)
    for idx in range(1, 7):
        for kind in ("R1", "XAFF"):
            assert a[f"{kind}_V{idx}__r1"].shape == (12,)
            assert a[f"{kind}_V{idx}__cf_cells"].shape == (2,)
        assert a[f"BX_V{idx}__r1"].shape == (12,)
    # Same frozen objects, another seed's parity; no seed-42 cells are reused.
    b.ctx.seed = 9002
    b.parity = 1 - b.parity
    (root / "seed9001").rename(root / "seed9002")
    (root / "gate_verdict.json").write_text(json.dumps({
        "picks": {"csd": "R1_V1", "generic": "XAFF_V5"}, "n": 2}))
    test = A.derive("B", b, root)
    for name in ("B", "B0", "B1", "AFF", "AFF_cf", "BX_V1", "BX_V5", "R1_V1", "R1_V1_cf", "XAFF_V5", "XAFF_V5_cf"):
        assert test[f"{name}__r1"].shape == (12,)
    assert "R1_V2__r1" not in test
    np.testing.assert_array_equal(test["parity"], b.parity)
    path = root / A.phase_path("B", 9002)
    A.write_once(path, test)
    saved = BU.checked_npz(path)
    assert set(saved) == set(test)
    for key in test:
        np.testing.assert_array_equal(saved[key], test[key])
    original = path.read_bytes()
    fixed = A.write_once(path, test)
    assert fixed.name == "phaseB_seed9002_fix1.npz"
    assert path.read_bytes() == original
    for key, value in BU.checked_npz(fixed).items():
        np.testing.assert_array_equal(value, test[key])
    # Deliberate wiring mutation must fail the named posterior-row assertion.
    z = BU.checked_npz(root / "heads/post_T.npz")
    z["rows"] = z["rows"][::-1]
    np.savez(root / "heads/post_T.npz", **z)
    with pytest.raises(AssertionError, match="head selection rows: T"):
        A.derive("K", b, root)


def test_threshold_table_accepts_write_once_metadata_and_checks_rule(tmp_path):
    import json
    from rederive import rd7_arrays as A
    path = tmp_path / "taus.json"
    dev = tmp_path / "seed42"
    dev.mkdir()
    margins = np.array([0., 1 / 3, 2 / 3, 1.])
    np.savez(dev / "reader_V1.npz", m__a=margins, m__b=margins)
    path.write_text(json.dumps({"V1": [0., .25, .5, .75], "rule_sha256": R.RULE_SHA256, "written": "2026-10-12 03:00"}))
    got = A.thresholds(tmp_path)
    assert set(got) == {"V1"} and got["V1"].tolist() == [0., .25, .5, .75]
    path.write_text(json.dumps({"V1": [0., .25, .5, .75], "rule_sha256": "wrong"}))
    with pytest.raises(AssertionError, match="threshold rule SHA"):
        A.thresholds(tmp_path)


def test_real_build_keeps_r3_a0_posteriors_and_arrays(legacy, tmp_path):
    # No RESULTS A0 heads exist here: the real path must use R3 D2's recipe.
    actual = BU.build(42, tmp_path)
    for name in K.A0:
        for modality in ("img", "txt"):
            got, want = actual.post[name][modality], legacy.post[name][modality]
            assert got.shape == want.shape and got.dtype == want.dtype
            assert np.array_equal(got[actual.ctx.selection], want[legacy.ctx.selection])
            assert np.isnan(got[~actual.ctx.in_sel]).all()
    for name, got, want in (
        ("B", K.metrics(actual.B), K.metrics(legacy.B)),
        ("B0", BU.condition_free(actual, K.A0), BU.condition_free(legacy, K.A0)),
    ):
        for metric in K.METRICS:
            np.testing.assert_array_equal(got[metric], want[metric], err_msg=f"{name}/{metric}")
    pk = BU.legacy_reader("A0")
    got = BU.score(actual, BU.read(actual, K.A0, pk), K.RULE_TAUS, "AFF")
    want = BU.score(legacy, BU.read(legacy, K.A0, pk), K.RULE_TAUS, "AFF")
    for who in ("fused", "cf"):
        for metric in K.METRICS:
            np.testing.assert_array_equal(got[who][metric], want[who][metric])
    assert got["record"] == want["record"]


def test_phase_a_b1_uses_r0_values(synthetic_frozen):
    from rederive import rd7_arrays as A
    b, root = synthetic_frozen
    actual = A.derive("A", b, root)
    correct = BU.condition_free(b, (*K.A0, "R0"))
    wrong = BU.condition_free(b, (*K.A0, "V1"))
    assert not correct["r1"].any()
    assert np.all(wrong["r1"] == .5)
    for metric in K.METRICS:
        np.testing.assert_array_equal(actual[f"B1__{metric}"], correct[metric])


def test_phase_a_xaff_uses_only_affect_and_x_values(synthetic_frozen):
    import json
    from rederive import rd7_arrays as A
    b, root = synthetic_frozen
    actual = A.derive("A", b, root)
    tau = json.loads((root / "taus.json").read_text())["V1"]
    reader = BU.read(b, (*K.A0, "V1"), BU.fitted_reader(root, "V1"), root / "seed9001/reader_V1.npz")
    gates = F.gates_x_aff(reader["m"], reader["pick"], tau, (0, 3))
    rec, fused, cf, _ = F.run_family(b.B, reader["T"], gates, tau, b.parity)
    for c in K.COND:
        np.testing.assert_array_equal(actual[f"XAFF_V1__gate_{c}"], np.stack([g[c] for g in gates]))
        excluded = np.isin(reader["pick"][c], (1, 2)) & (reader["m"][c] >= tau[0])
        assert excluded.any()
        assert actual[f"R1_V1__gate_{c}"][0, excluded].all()
        assert not actual[f"XAFF_V1__gate_{c}"][0, excluded].any()
    for name, expected in (("XAFF_V1", fused), ("XAFF_V1_cf", cf)):
        for metric in K.METRICS:
            np.testing.assert_array_equal(actual[f"{name}__{metric}"], expected[metric])
    np.testing.assert_array_equal(actual["XAFF_V1__fused_cells"], [rec["fused_cells"][h]["cell"] for h in "01"])
    assert not np.array_equal(actual["R1_V1__r1"], actual["XAFF_V1__r1"])


def test_phase_a_each_candidate_uses_its_frozen_taus(synthetic_frozen):
    import json
    from rederive import rd7_arrays as A
    b, root = synthetic_frozen
    actual = A.derive("A", b, root)
    table = json.loads((root / "taus.json").read_text())
    for x, tau in table.items():
        reader = BU.read(b, (*K.A0, x), BU.fitted_reader(root, x), root / f"seed9001/reader_{x}.npz")
        gates = F.gates_r1(reader["m"], tau)
        aff_gates = F.gates_r1(reader["m"], BU.aff_taus())
        assert any(not np.array_equal(g["a"], wrong["a"]) for g, wrong in zip(gates, aff_gates))
        for kind, selected in (("R1", gates), ("XAFF", F.gates_x_aff(reader["m"], reader["pick"], tau, (0, 3)))):
            for c in K.COND:
                np.testing.assert_array_equal(actual[f"{kind}_{x}__gate_{c}"], np.stack([g[c] for g in selected]))


def test_thresholds_rederived_from_frozen_development_margins(tmp_path):
    import json
    from rederive import rd7_arrays as A
    dev = tmp_path / "seed42"
    dev.mkdir()
    margins = np.array([0., 1 / 3, 2 / 3, 1.])
    table = {x: [0., .25, .5, .75] for x in ("V1", "Z6", "R0", "N")}
    for x in table:
        np.savez(dev / f"reader_{x}.npz", m__a=margins, m__b=margins)
    path = tmp_path / "taus.json"
    path.write_text(json.dumps(table))
    assert set(A.thresholds(tmp_path)) == set(table)
    # A different, still sorted threshold cannot pass merely a range check.
    table["V1"][1] = .2
    path.write_text(json.dumps(table))
    with pytest.raises(AssertionError, match="frozen development thresholds: V1"):
        A.thresholds(tmp_path)


@pytest.mark.parametrize("chain,seed", [(False, 9001), (False, 9002), (False, 9003), (True, 9002)])
def test_smoke_reads_frozen_objects_and_writes_its_own_outputs(synthetic_frozen, monkeypatch, chain, seed):
    import json
    import shutil
    from rederive import rd7_arrays as A
    b, results = synthetic_frozen
    rule_root, chain_root = results / "smoke/rule", results / "smoke/chain"
    monkeypatch.setattr(R, "RESULTS", results)
    monkeypatch.setattr(R, "SMOKE_RULE", rule_root)
    monkeypatch.setattr(R, "SMOKE_CHAIN", chain_root)
    objects, output = (chain_root, chain_root) if chain else (results, rule_root)
    output.mkdir(parents=True)
    if chain:
        for folder in ("heads", "readers"):
            shutil.copytree(results / folder, objects / folder)
        shutil.copyfile(results / "taus.json", objects / "taus.json")
        shutil.copytree(results / "seed9001", objects / "seed9001")
    runtime = output / f"seed{seed}"
    if not runtime.exists():
        shutil.copytree(results / "seed9001", runtime)
    (objects / "gate_verdict.json").write_text(json.dumps({
        "picks": {"csd": "XAFF_V1", "generic": None}, "n": 1}))
    b.ctx.seed = seed
    # Replace only the expensive input boundary; run the real derivation.
    def built(got_seed, root, smoke):
        assert got_seed == seed and root == objects and smoke
        return b
    monkeypatch.setattr(BU, "build", built)
    path = A.run("B", seed, chain)
    assert path == output / "rederive" / f"phaseB_seed{seed}.npz"
    actual = BU.checked_npz(path)
    assert "XAFF_V1__r1" in actual and "R1_V1__r1" not in actual
    fixed = A.run("B", seed, chain)
    assert fixed == output / "rederive" / f"phaseB_seed{seed}_fix1.npz"
    for key, value in BU.checked_npz(fixed).items():
        np.testing.assert_array_equal(value, actual[key])
