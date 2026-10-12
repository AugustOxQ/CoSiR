"""Round 7 descriptive pass: DECISION_RULE §5, §6, §7.9; spec §3.6/D13.

Each part requires its finalized verdict before reading scores or labels. Labels
are used only for diagnostics, on scorer-train rows; chain smoke permutes them
with default_rng(0). Reused modules are read-only through r7_common.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

import r7_common as R
import r7_bundle as B
import r7_fusion as F
import r7_gate as S

METRICS = ("r1", "gain", "other", "swap", "strict")
IDS = S.VARIANTS + tuple(f"Z{k}" for k in range(1, 7)) + ("T", "R0", "N")


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def record(path, smoke=None):
    path = Path(path)
    assert path.is_file(), f"{path}: required file missing"
    rec = json.loads(path.read_text())
    assert isinstance(rec, dict), f"{path}: expected object"
    assert rec.get("rule_sha256") == R.rule_sha(), f"{path}: rule SHA differs"
    if smoke is not None:
        assert rec.get("smoke") is smoke, f"{path}: smoke role differs"
    return rec


def npz(path):
    assert path.is_file(), f"{path}: required file missing"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def arrays(path, smoke):
    a = npz(path)
    assert {"cl", "pair_index", "parity"} <= a.keys(), f"{path}: missing episode IDs"
    n = len(a["cl"])
    assert n == (192 if smoke else 12288), f"{path}: episode count differs"
    for k in ("cl", "pair_index", "parity"):
        assert a[k].dtype == np.int64 and a[k].shape == (n,), f"{path}/{k}: ID shape/dtype"
    assert (a["cl"] >= 0).all(), f"{path}: invalid anchor paintings"
    assert np.array_equal(a["parity"], np.arange(n) % 2), f"{path}: parity alignment differs"
    assert np.array_equal(np.bincount(a["pair_index"], minlength=3), [n // 3] * 3), f"{path}: pair counts differ"
    for k, x in a.items():
        if k.endswith(tuple(f"__{m}" for m in METRICS)):
            assert x.dtype == np.float64 and x.shape == (n,) and np.isfinite(x).all(), f"{path}/{k}: metric shape/dtype"
            S.sum4(x)
        elif k.endswith(("__gate_a", "__gate_b")):
            assert x.dtype == np.float32 and x.shape == (4, n) and np.isin(x, (0, 1)).all(), f"{path}/{k}: gates"
        elif k.endswith(("__fused_cells", "__cf_cells")):
            assert x.dtype == np.int64 and x.shape == (2,) and ((x >= 0) & (x < 224)).all(), f"{path}/{k}: cells"
        elif k.endswith("__sigma"):
            assert x.dtype == np.float64 and x.shape == (2,) and np.isfinite(x).all(), f"{path}/{k}: sigma"
        else:
            assert k in ("cl", "pair_index", "parity"), f"{path}/{k}: unknown array"
    return a


def metrics(a, name):
    assert all(f"{name}__{m}" in a for m in METRICS), f"{name}: missing metrics"
    return {m: a[f"{name}__{m}"] for m in METRICS}


def reader(path, n, parts):
    a = npz(path)
    assert {f"{k}__{c}" for k in ("P", "m", "pick") for c in ("a", "b")} <= a.keys(), f"{path}: reader keys"
    out = {k: {c: a[f"{k}__{c}"] for c in ("a", "b")} for k in ("P", "m", "pick")}
    for c in ("a", "b"):
        p, m, pick = (out[k][c] for k in ("P", "m", "pick"))
        assert p.dtype == np.float64 and p.shape == (n, len(parts)) and np.isfinite(p).all(), f"{path}/{c}: probabilities"
        assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12), f"{path}/{c}: probability sums"
        assert m.dtype == np.float64 and m.shape == (n,) and np.isfinite(m).all(), f"{path}/{c}: margins"
        assert pick.dtype == np.int64 and pick.shape == (n,), f"{path}/{c}: picks"
        assert np.array_equal(pick, p.argmax(axis=1)), f"{path}/{c}: picks disagree with P"
        assert np.array_equal(m, F.C.top_two_margin(p)), f"{path}/{c}: margins disagree with P"
    return out


def taus(root, x):
    rec = record(root / "taus.json")
    t = np.asarray(rec.get(x), dtype=np.float64)
    assert t.shape == (4,) and np.isfinite(t).all() and (np.diff(t) >= 0).all(), f"{x}: frozen taus"
    return t.tolist()


def summary(pa, cf, a, comparators=None):
    """Points and paired cluster intervals; no diagnostic participates in a pick."""
    for who in (pa, cf):
        assert set(who) == set(METRICS), "summary: metric keys differ"
        for v in who.values():
            assert v.dtype == np.float64 and v.shape == a["cl"].shape and np.isfinite(v).all(), "summary: metric shape/dtype"

    def scope(mask):
        cl = a["cl"][mask]
        stat = lambda v: F.C.point_ci(np.asarray(v, dtype=np.float64)[mask], cl)
        out = {"n": int(mask.sum()), "r1": stat(pa["r1"]), "cf_r1": stat(cf["r1"]),
               "margin": stat(pa["r1"] - cf["r1"]), "gain": stat(pa["gain"]),
               "gain_statistic": stat(pa["gain"] - cf["gain"])}
        if comparators:
            out["vs_comparators"] = {k: stat(pa["r1"] - v) for k, v in comparators.items()}
        return out
    return {"pooled": scope(np.ones(len(a["cl"]), dtype=bool)),
            "per_pair": {p: scope(a["pair_index"] == i) for i, p in enumerate(S.PAIRS)}}


def alignment(bundle, a):
    for k in ("cl", "pair_index", "parity"):
        assert np.array_equal(getattr(bundle, k), a[k]), f"bundle/{k}: stored episode alignment differs"


def read_groups(root):
    labels, records = {}, {}
    for stem in ("r7_groupings", "r7_grouping_N"):
        path = root / "groupings" / stem
        rec, a = record(path.with_suffix(".json")), npz(path.with_suffix(".npz"))
        assert "groupings" in rec and {"scorer_train", "g_st"} <= a.keys(), f"{path}: grouping keys"
        st, g = a["scorer_train"], a["g_st"]
        assert st.dtype == g.dtype == np.int64 and st.ndim == 1 and g.shape == st.shape, f"{path}: row shape/dtype"
        assert len(st) and (st >= 0).all() and (np.diff(st) > 0).all() and (g >= 0).all(), f"{path}: row IDs"
        for k, v in (("scorer_train_sha256", st), ("g_st_sha256", g)):
            if k in rec:
                assert rec[k] == sha_array(v), f"{path}/{k}: alignment SHA differs"
        if stem == "r7_groupings":
            rows, paintings = st, g
        else:
            assert np.array_equal(st, rows) and np.array_equal(g, paintings), "N: row alignment differs"
        for x, item in rec["groupings"].items():
            assert x in IDS and type(item.get("dropped")) is bool, f"{path}/{x}: record differs"
            if x in a:
                local = a[x]
                assert local.dtype == np.int64 and local.shape == st.shape, f"{path}/{x}: labels"
                _, first = np.unique(g, return_index=True)
                assert np.array_equal(np.unique(g), np.arange(len(first))), f"{path}: painting codes"
                assert np.array_equal(local[first][g], local), f"{x}: labels vary within painting"
                assert np.array_equal(np.unique(local), np.arange(len(np.unique(local)))), f"{x}: noncompact labels"
                if "partition_sha256" in item:
                    assert item["partition_sha256"] == sha_array(local), f"{x}: partition SHA differs"
                labels[x] = local
            else:
                assert item["dropped"], f"{path}/{x}: labels missing"
        records.update(rec["groupings"])
    assert set(records) == set(IDS), "grouping record IDs differ"
    return rows, paintings, labels, records


def label_diagnostics(role):
    # This guard precedes even loading scorer-train labels (rule §6).
    record(role.root / "gate_verdict.json", role.smoke)
    st, g, groupings, records = read_groups(role.root)
    rb = R.load("rb_build")
    data, sp, expected, groups = rb.splits()
    assert np.array_equal(st, expected), "label scorer-train rows differ"
    assert np.array_equal(g, np.unique(groups[st], return_inverse=True)[1]), "label painting alignment differs"
    lab = R.load("run_told_oracle").artelingo_aspect_labels(data)
    labels = {k: np.asarray(lab[k])[st] for k in ("style", "genre")}
    del data, lab
    key = "20261031_pseudo_partitions/results/partitions.npz"
    path = R.path_of(key)
    assert R.sha_file(path) == R.INPUTS[key], "content grouping SHA differs"
    a = npz(path)
    assert {"image", "local_groups"} <= a.keys(), "content grouping keys"
    assert np.array_equal(a["local_groups"], g), "content grouping row alignment differs"
    labels["content"] = a["image"]
    for k, v in labels.items():
        assert v.dtype == np.int64 and v.shape == st.shape, f"{k}: label shape/dtype"
    if role.smoke:
        perm = np.random.default_rng(0).permutation(len(st))
        labels = {k: v[perm] for k, v in labels.items()}
    out = {}
    _, first = np.unique(g, return_index=True)
    for x, rec in records.items():
        if x not in groupings:
            out[x] = {"dropped": True, "reason": rec.get("reason")}
            continue
        v = groupings[x]
        info = {"dropped": rec["dropped"], "n_groups": len(np.unique(v)),
                "sizes_rows": np.bincount(v).tolist(), "sizes_paintings": np.bincount(v[first]).tolist(),
                "stability_ami": rec.get("stability_ami"), "ami": {}, "head_accuracy": None}
        for k, lab in labels.items():
            mask = lab >= 0
            assert mask.any(), f"{k}: no labelled scorer-train rows"
            info["ami"][k] = {"n_rows": int(mask.sum()), "value": float(ami(v[mask], lab[mask]))}
        head = role.root / "heads" / f"head_{x}.json"
        if head.exists():
            h = record(head, role.smoke)
            assert h.get("grouping") == x and h.get("n_groups") == info["n_groups"], f"{head}: grouping/classes"
            if "label_sha256" in h:
                assert h["label_sha256"] == sha_array(v), f"{head}: label alignment SHA differs"
            assert type(h.get("dropped")) is bool, f"{head}: drop status"
            info["head_dropped"] = h["dropped"]
            if not h["dropped"]:
                acc = h.get("standard", {}).get("heldout_accuracy")
                assert isinstance(acc, dict) and set(acc) == {"img", "txt"}, f"{head}: accuracy keys"
                assert all(np.isfinite(v) and 0 <= v <= 100 for v in acc.values()), f"{head}: accuracies"
                info["head_accuracy"] = acc
        else:
            assert rec["dropped"], f"{head}: missing kept grouping head"
        out[x] = info
    if role.smoke:
        token = {"status": "chain-smoke fixture; no token extraction"}
    else:
        path = Path("/data/SSD2/pre_extract/artelingo/clip_b32_tokstats/meta.json")
        meta = record(path)
        shares = np.asarray(meta.get("dropped_share_per_block"))
        assert meta.get("blocks") == [4, 6, 8] and meta.get("dropped_share_n_images") == 61402, "token coverage"
        assert shares.shape == (3,) and np.isfinite(shares).all() and ((shares >= 0) & (shares <= 1)).all(), "token shares"
        token = {"blocks": [4, 6, 8], "shares": shares.tolist(), "n_images": 61402}
    for x in ("V5", "V6"):
        out[x]["dropped_token_shares"] = token
    return out


def scorer(role, a, name, folder):
    x = "A0" if name == "AFF" else name.split("_", 1)[1]
    parts = B.A0 if x == "A0" else B.A0 + (x,)
    rd = reader(folder / f"reader_{x}.npz", len(a["cl"]), parts)
    t = F.TAUS if x == "A0" else taus(role.root, x)
    if name.startswith(("XAFF_", "AFF")):
        gates = F.gates_x_aff(rd["m"], t, rd["pick"], {0} if x == "A0" else {0, 3})
    else:
        gates = F.gates_r1(rd["m"], t)
    for c in ("a", "b"):
        assert f"{name}__gate_{c}" in a, f"{name}: stored gates missing"
        assert np.array_equal(np.stack([gt[c] for gt in gates]), a[f"{name}__gate_{c}"]), f"{name}: stored gate wiring differs"
    for k in ("fused_cells", "cf_cells", "sigma"):
        assert f"{name}__{k}" in a, f"{name}: missing {k}"
    comps = {k: a[f"{k}__r1"] for k in ("B", "B0", "B1", "AFF")}
    if x != "A0":
        comps["BX"] = metrics(a, f"BX_{x}")["r1"]
    pa, cf = metrics(a, name), metrics(a, f"{name}_cf")
    if x != "A0":
        bar = S.bar_comparator({"B1": comps["B1"], "BX": comps["BX"], "CF": cf["r1"],
                                "B0": comps["B0"], "B": comps["B"]})
        comps["bar"] = cf["r1"] if bar == "CF" else comps[bar]
    out = summary(pa, cf, a, comps)
    if x != "A0":
        out["bar_comparator"] = bar
    told = F.C.TOLD["A0"] if x == "A0" else {"emotion": "affect", "style": x, "genre": "image"}
    out["pick_accuracy"], out["pick_shares"] = F.C.pick_statistics(rd["pick"], parts, told, a["pair_index"], a["cl"])
    out["gate_open_shares"] = R.load("r3_fusion").open_shares(gates, a["pair_index"])
    out["cells"] = {k: [F.describe(int(c), t) for c in a[f"{name}__{k}"]] for k in ("fused_cells", "cf_cells")}
    out["sigma"] = a[f"{name}__sigma"].tolist()
    return out, rd, gates, parts


def ceiling(role):
    verdict = record(role.root / "ceiling_verdict.json", role.smoke)
    assert verdict.get("zone") in S.ZONES, "ceiling verdict zone differs"
    a = arrays(role.root / f"seed{role.dev}" / "arrays_ceiling.npz", role.smoke)
    bundle = B.build_bundle(role.dev, role.smoke, extra=("T",), root=role.root)
    alignment(bundle, a)
    b1 = B.condition_free(bundle, B.A1)[0]
    bt = B.condition_free(bundle, B.A0 + ("T",))[0]
    pa, cf = metrics(a, "told_T"), metrics(a, "told_T_cf")
    out = {"told_T": summary(pa, cf, a, {"B1": b1["r1"], "BT": bt["r1"]})}
    head = record(role.root / "heads/head_T.json", role.smoke)
    assert head.get("grouping") == "T" and head.get("dropped") is False, "T head record differs"
    out["head_accuracy"] = head["standard"]["heldout_accuracy"]
    assert set(out["head_accuracy"]) == {"img", "txt"}, "T head accuracy keys differ"
    assert all(np.isfinite(v) and 0 <= v <= 100 for v in out["head_accuracy"].values()), "T head accuracies differ"
    lab = R.load("run_told_oracle").artelingo_aspect_labels(bundle.ctx.data)["style"]
    rows = bundle.ctx.selection
    assert lab.dtype == np.int64 and lab.shape == (len(bundle.ctx.groups),), "T* style labels shape/dtype"
    codes = lab[rows]
    assert (codes >= 0).all() and (codes < bundle.post["T"]["img"].shape[1]).all(), "T* missing/out-of-range style code"
    if role.smoke:
        codes = codes[np.random.default_rng(0).permutation(len(codes))]
    onehot = np.full_like(bundle.post["T"]["img"], np.nan)
    onehot[rows] = np.eye(onehot.shape[1], dtype=np.float32)[codes]
    star = F.told(bundle, B.A0 + ("T",), {"emotion": "affect", "style": "T", "genre": "image"},
                  posteriors_override={"T": {"img": onehot, "txt": onehot.copy()}})
    out["Tstar"] = summary(star["fused"], star["cf"], a)
    return out


def gate(role):
    verdict = record(role.root / "gate_verdict.json", role.smoke)
    assert {"candidates", "references", "picks"} <= verdict.keys(), "gate verdict keys"
    refs = {f"{kind}_{x}" for x in ("R0", "N") + tuple(f"Z{k}" for k in range(1, 7)) for kind in S.KINDS}
    assert set(verdict["candidates"]) == set(S.CANDIDATES) and set(verdict["references"]) == refs, "gate scorer IDs differ"
    folder = role.root / f"seed{role.dev}"
    a = arrays(folder / "arrays.npz", role.smoke)
    out = {"groupings": label_diagnostics(role), "scorers": {}}
    for name, rec in (verdict["candidates"] | verdict["references"]).items():
        assert type(rec.get("dropped")) is bool, f"{name}: verdict drop status"
        out["scorers"][name] = {"dropped": True} if rec["dropped"] else scorer(role, a, name, folder)[0]
    out["scorers"]["AFF"] = scorer(role, a, "AFF", folder)[0]
    out["random_controls"] = {x: {"grouping": f"Z{x[1:]}",
        "diagnostics": out["groupings"][f"Z{x[1:]}"],
        "scorers": {kind: out["scorers"][f"{kind}_Z{x[1:]}"] for kind in S.KINDS}} for x in S.VARIANTS}
    return out


def test(role):
    verdict = record(role.root / "test_verdict.json", role.smoke)
    gate_verdict = record(role.root / "gate_verdict.json", role.smoke)
    assert {"chains", "FV"} <= verdict.keys() and "picks" in gate_verdict, "test/gate verdict keys"
    assert set(gate_verdict["picks"]) == set(S.FAMILIES), "gate pick families differ"
    picks = {family: name for family, name in gate_verdict["picks"].items() if name is not None}
    assert set(verdict["chains"]) == set(picks), "test chains differ from gate picks"
    for family, name in picks.items():
        assert verdict["chains"][family].get("pick") == name and name in S.CANDIDATES, "test pick differs"
    if not picks:
        return {"seeds": {}, "picks": {}, "status": "no picked chains"}
    dev = arrays(role.root / f"seed{role.dev}/arrays.npz", role.smoke)
    groups = label_diagnostics(role)
    out = {"seeds": {}, "picks": {f: {"scorer": s, "ami": groups[s.split("_", 1)[1]]["ami"]} for f, s in picks.items()}}
    for seed in role.tests:
        folder = role.root / f"seed{seed}"
        a = arrays(folder / "arrays.npz", role.smoke)
        xs = tuple(dict.fromkeys(name.split("_", 1)[1] for name in picks.values()))
        bundle = B.build_bundle(seed, role.smoke, extra=xs, root=role.root)
        alignment(bundle, a)
        per_seed = {}
        for name in dict.fromkeys((*picks.values(), "AFF")):
            info, rd, gates, parts = scorer(role, a, name, folder)
            stack, _ = B.stack_and_features(bundle, parts)
            term = F.C.expected_term(stack, rd["P"])
            assert all(f"{name}__{k}" in dev for k in ("fused_cells", "cf_cells")), f"{name}: dev cells missing"
            frozen = F.score_frozen(bundle, term, gates, dev[f"{name}__fused_cells"], dev[f"{name}__cf_cells"])
            info["frozen_cells"] = {k: dev[f"{name}__{k}"].tolist() for k in ("fused_cells", "cf_cells")}
            info["frozen"] = summary(frozen["fused"], frozen["cf"], a)
            per_seed[name] = info
        out["seeds"][str(seed)] = per_seed
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("ceiling", "gate", "test"))
    p.add_argument("--chain-smoke", action="store_true")
    args = p.parse_args()
    R.assert_rule()
    R.assert_threads()
    role = R.roles("chain" if args.chain_smoke else "real")
    # Refuse before any score or label read, including when an output already exists.
    record(role.root / f"{args.part}_verdict.json", role.smoke)
    path = role.root / f"descriptive_{args.part}.json"
    assert not path.exists() or role.smoke, f"{path}: refusing to overwrite descriptive output"
    out = {"smoke": role.smoke, "part": args.part, "units": "R@1/gain/accuracy in points; AMI unscaled",
           **{"ceiling": ceiling, "gate": gate, "test": test}[args.part](role)}
    R.write_json_once(path, out)
    print(f"descriptive {args.part}: {path.name}; file boundary assertions passed")


if __name__ == "__main__":
    main()
