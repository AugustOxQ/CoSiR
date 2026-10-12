"""Round 7 runner: DECISION_RULE L10–L16, §§5, 6, 9 (prep, pending, independent finalize).

Stages remain separate functions for ticket 13. Chain smoke uses synthetic paintings,
features and episodes, the public head/reader/fusion paths and the index's smoke sizes.
It never writes to reused folders. Re-derivation is read from files, never imported.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import r7_common as R
import r7_groupings as G
import r7_nrkmeans as N
import r7_heads as H
import r7_reader as Q
import r7_bundle as B
import r7_fusion as F
import r7_gate as S

GROUPINGS = Q.GROUPINGS + ("T",)
REFERENCES = tuple(f"{kind}_{x}" for x in ("R0", "N") + tuple(f"Z{k}" for k in range(1, 7))
                   for kind in S.KINDS)
METRICS = ("r1", "gain", "other", "swap", "strict")
CORE = {"ceiling": ("sum4d", "n", "point", "ci95", "zone"),
        "gate": ("candidates", "references", "picks", "n", "alpha", "promising", "random_control_flag")}


def read_record(path, smoke=None):
    rec = json.loads(Path(path).read_text())
    assert isinstance(rec, dict), f"{path}: expected object"
    assert rec.get("rule_sha256") == R.rule_sha(), f"{path}: rule SHA differs"
    if smoke is not None:
        assert rec.get("smoke") == smoke, f"{path}: smoke role differs"
    return rec


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    assert {"cl", "pair_index", "parity"} <= arr.keys(), f"{path}: missing episode IDs"
    n = len(arr["cl"])
    for k in ("cl", "pair_index", "parity"):
        assert arr[k].dtype == np.int64 and arr[k].shape == (n,), f"{path}/{k}: IDs differ"
    assert n > 0 and np.isin(arr["pair_index"], (0, 1, 2)).all(), f"{path}: invalid pairs"
    assert np.isin(arr["parity"], (0, 1)).all(), f"{path}: invalid parity"
    for k, a in arr.items():
        if k.endswith(tuple(f"__{m}" for m in METRICS)):
            assert a.shape == (n,) and a.dtype == np.float64 and np.isfinite(a).all(), f"{path}/{k}: metrics differ"
        elif k.endswith(("__gate_a", "__gate_b")):
            assert a.shape == (4, n) and a.dtype == np.float32 and np.isin(a, (0, 1)).all(), f"{path}/{k}: gates differ"
        elif k.endswith(("__fused_cells", "__cf_cells")):
            assert a.shape == (2,) and a.dtype == np.int64 and np.all((a >= 0) & (a < 224)), f"{path}/{k}: cells differ"
        elif k.endswith("__sigma"):
            assert a.shape == (2,) and a.dtype == np.float64 and np.isfinite(a).all(), f"{path}/{k}: sigma differs"
    return arr


def _group_records(root):
    out = {}
    for stem in ("r7_groupings", "r7_grouping_N"):
        rec = read_record(root / "groupings" / f"{stem}.json")
        assert "groupings" in rec, f"{stem}: missing records"
        with np.load(root / "groupings" / f"{stem}.npz", allow_pickle=False) as z:
            assert {"scorer_train", "g_st"} <= set(z.files), f"{stem}: missing alignment"
            st, g = z["scorer_train"], z["g_st"]
            assert st.dtype == g.dtype == np.int64 and st.ndim == 1 and g.shape == st.shape, f"{stem}: row shape/dtype"
            assert len(st) > 0 and np.all(np.diff(st) > 0) and (g >= 0).all(), f"{stem}: row order/codes"
            if stem == "r7_groupings":
                alignment = (st, g)
            else:
                assert all(np.array_equal(a, b) for a, b in zip(alignment, (st, g))), "N: grouping alignment differs"
            for name, item in rec["groupings"].items():
                assert type(item["dropped"]) is bool, f"{name}: missing drop status"
                assert item["dropped"] or name in z.files, f"{name}: missing labels"
                if not item["dropped"]:
                    labels = z[name]
                    assert labels.dtype == np.int64 and labels.shape == st.shape, f"{name}: label shape/dtype"
                    assert np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))), f"{name}: noncompact labels"
        out.update(rec["groupings"])
    assert set(GROUPINGS) <= out.keys(), "prep: missing grouping records"
    return out


def prep(role):
    root = role.root
    if role.smoke:
        return smoke_prep(role)
    for stem, build in (("r7_groupings", G.build), ("r7_grouping_N", N.build)):
        if not (root / "groupings" / f"{stem}.json").exists():
            build(out=root / "groupings")
    records = _group_records(root)
    assert not any(records[x]["dropped"] for x in ("R0", "T")), "required R0/T grouping dropped"
    rb = R.load("rb_build")
    data, sp, st, groups = rb.splits()
    halves = H.load_halves(st, groups)
    dropped = [x for x, rec in records.items() if rec["dropped"]]
    for name in Q.BASE + GROUPINGS:
        if name in dropped:
            continue
        path = root / "heads" / f"head_{name}.json"
        if path.exists():
            rec = read_record(path, False)
            assert rec["grouping"] == name, f"{path}: grouping differs"
        else:
            labels = H.load_groupings((name,), st, halves["g_st"], root)[name]
            rec = H.build_heads(name, data, sp, labels, halves, root)
        if rec["dropped"]:
            assert name not in Q.BASE + ("R0", "T"), f"required head {name} dropped"
            dropped.append(name)
        else:
            ctx = SimpleNamespace(groups=groups, selection=sp.selection)
            if name in Q.BASE:
                with np.load(root / "heads" / f"post_{name}.npz", allow_pickle=False) as z:
                    assert {"rows", "img", "txt"} <= set(z.files), f"{name}: posterior keys differ"
                    assert np.array_equal(z["rows"], sp.selection), f"{name}: selection differs"
                    B._posterior_ok({m: z[m] for m in ("img", "txt")}, ctx, name)
            else:
                B.load_posteriors(ctx, name, root)
            assert (root / "heads" / f"head_{name}.pkl").is_file(), f"{name}: incomplete head"
            if name != "T":
                labels = H.load_groupings((name,), st, halves["g_st"], root)[name]
                Q.load_crossfit(name, labels, halves, root)
    tasks = []
    for name in Q.GROUPINGS:
        if name in dropped:
            continue
        path = root / "readers" / f"reader_{name}.json"
        if path.exists():
            rec = read_record(path, False)
            assert rec["grouping"] == name, f"{path}: grouping differs"
            if rec["dropped"]:
                dropped.append(name)
            else:
                assert R.sha_file(path.with_suffix(".pkl")) == rec["reader_sha256"], f"{name}: reader SHA differs"
        else:
            tasks.append(name)
    # The public reader entry point runs three fits at once, eight threads each.
    if tasks:
        subprocess.run([sys.executable, str(Path(__file__).with_name("r7_reader.py")),
                        "--grouping", *tasks, "--root", str(root), "--jobs", "3"], check=True)
        for name in tasks:
            rec = read_record(root / "readers" / f"reader_{name}.json", False)
            if rec["dropped"]:
                dropped.append(name)
    assert "R0" not in dropped, "G3: R0 reader dropped"
    path = root / "prep.json"
    if not path.exists():
        R.write_json_once(path, {"smoke": False, "dropped": sorted(set(dropped))})
    else:
        assert read_record(path, False)["dropped"] == sorted(set(dropped)), "prep drop record changed"
    print(f"prep: {len(GROUPINGS)} grouping records checked")


def _ready(role, stage):
    if stage == "ceiling":
        rec = read_record(role.root / "stage0_regression.json")
        assert rec.get("passed") is True, "stage 0 regression did not pass"
    else:
        if role.smoke:
            # Synthetic plumbing must reach gate even when random smoke data has
            # a K1 STOP. This authorization cannot produce a real-stage verdict.
            rec = read_record(role.root / "ceiling_pending.json", True)
            assert rec.get("synthetic") is True, "smoke gate requires synthetic fixtures"
        else:
            rec = read_record(role.root / "ceiling_verdict.json", False)
            assert rec["zone"] in ("WEAK", "CLEAR"), "K1 STOP: gate forbidden"
    return read_record(role.root / "prep.json", role.smoke)


def episode_arrays(bundle):
    return {k: getattr(bundle, k) for k in ("cl", "pair_index", "parity")}


def add_metrics(arr, name, pa):
    assert set(pa) == set(METRICS), f"{name}: missing metrics"
    arr.update({f"{name}__{k}": v for k, v in pa.items()})


def save_reader(role, bundle, name, rd):
    path = role.root / f"seed{role.dev}" / f"reader_{name}.npz"
    arrays = {f"{k}__{c}": rd[k][c] for k in ("P", "m", "pick") for c in ("a", "b")}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert set(arrays) <= set(z.files), f"{path}: missing reader keys"
            assert all(np.array_equal(v, z[k]) for k, v in arrays.items()), f"{path}: reader changed"
    else:
        R.write_npz_once(path, **arrays)


def ceiling(role):
    _ready(role, "ceiling")
    path = role.root / f"seed{role.dev}" / "arrays_ceiling.npz"
    if path.exists():
        arr = read_arrays(path)
    else:
        bundle = make_bundle(role, ("T",))
        scored = F.told(bundle, B.A0 + ("T",), {"emotion": "affect", "style": "T", "genre": "image"})
        save_reader(role, bundle, "T", scored["reader"])
        arr = episode_arrays(bundle) | scored["arrays"]
        R.write_npz_once(path, **arr)
    assert len(arr["cl"]) == (192 if role.smoke else 12288), "ceiling episode count differs"
    mask = arr["pair_index"] == 2
    assert mask.sum() == (64 if role.smoke else 4096), "K1 must get exactly style x genre episodes"
    pending = S.ceiling_stats(arr["told_T__r1"][mask], arr["told_T_cf__r1"][mask], arr["cl"][mask])
    write_pending(role, "ceiling", pending)
    print(f"ceiling: {path.name}, {int(mask.sum())} style x genre episodes checked")


def gate(role):
    prep_rec = _ready(role, "gate")
    path = role.root / f"seed{role.dev}" / "arrays.npz"
    dropped = prep_rec["dropped"]
    active = tuple(x for x in Q.GROUPINGS if x not in dropped)
    if path.exists():
        arr = read_arrays(path)
    else:
        bundle = make_bundle(role, active)
        arr = episode_arrays(bundle)
        add_metrics(arr, "B", bundle.pB)
        add_metrics(arr, "B0", bundle.pBp)
        add_metrics(arr, "B1", B.condition_free(bundle, B.A1)[0])
        arr.update(F.score_aff(bundle)["arrays"])
        save_reader(role, bundle, "A0", F.reader(bundle, B.A0))
        tau_path = role.root / "taus.json"
        taus = {}
        for x in active:
            parts = B.A0 + (x,)
            rd = F.reader(bundle, parts, role.root / "readers" / f"reader_{x}.pkl")
            save_reader(role, bundle, x, rd)
            taus[x], n = F.thresholds(rd["m"])
            assert n == 2 * bundle.n == (384 if role.smoke else 24576), "L12 margin count differs"
        if tau_path.exists():
            stored = read_record(tau_path)
            assert all(stored.get(x) == tau for x, tau in taus.items()), "frozen taus changed"
        else:
            R.write_json_once(tau_path, taus)
        for x in active:
            parts, reader = B.A0 + (x,), role.root / "readers" / f"reader_{x}.pkl"
            add_metrics(arr, f"BX_{x}", B.condition_free(bundle, parts)[0])
            arr.update(F.score_r1(bundle, parts, reader, taus[x])["arrays"])
            arr.update(F.score_x_aff(bundle, parts, reader, taus[x])["arrays"])
        R.write_npz_once(path, **arr)
    assert len(arr["cl"]) == (192 if role.smoke else 12288), "gate episode count differs"
    frozen = read_record(role.root / "taus.json")
    for x in active:
        tau = np.asarray(frozen.get(x), dtype=np.float64)
        assert tau.shape == (4,) and np.isfinite(tau).all() and (np.diff(tau) >= 0).all(), f"{x}: frozen taus differ"
    pending = S.gate_stats(arr, references=REFERENCES, dropped=dropped)
    write_pending(role, "gate", pending)
    print(f"gate: {path.name}, {len(arr)} arrays checked")


def write_pending(role, stage, rec):
    path = role.root / f"{stage}_pending.json"
    if path.exists():
        old = read_record(path, role.smoke)
        assert all(old[k] == rec[k] for k in CORE[stage]), f"{stage}: pending changed"
    else:
        R.write_json_once(path, {**rec, "smoke": role.smoke, "synthetic": role.smoke})


def differences(a, b, path=""):
    """Compare only the pipeline's core schema; reporting extras are ignored recursively."""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [path]
        return [p for k, v in a.items() for p in
                ([f"{path}.{k}"] if k not in b else differences(v, b[k], f"{path}.{k}"))]
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return [path]
        return [p for i, v in enumerate(a) for p in differences(v, b[i], f"{path}[{i}]")]
    if type(a) is float:
        ok = type(b) in (int, float) and np.isfinite(b) and abs(a - b) <= (1e-9 if 'ci' in path else 0)
    else:
        ok = type(a) is type(b) and a == b
    return [] if ok else [path]


def finalize(role, stage):
    _ready(role, stage)
    verdict = role.root / f"{stage}_verdict.json"
    assert not verdict.exists(), f"{verdict}: refusing to overwrite verdict"
    pending = read_record(role.root / f"{stage}_pending.json", role.smoke)
    stats_path = role.root / "rederive" / f"stats_{stage}.json"
    stats = json.loads(stats_path.read_text())
    assert isinstance(stats, dict), f"{stats_path}: expected object"
    diff = differences({k: pending[k] for k in CORE[stage]}, stats, stage)
    filename = "arrays_ceiling.npz" if stage == "ceiling" else "arrays.npz"
    arr = read_arrays(role.root / f"seed{role.dev}" / filename)
    phase = "phaseK.npz" if stage == "ceiling" else f"phaseA_seed{role.dev}.npz"
    required = {"told_T__r1", "told_T_cf__r1"}
    if stage == "gate":
        required = set()
        for name in S.CANDIDATES:
            if pending["candidates"][name]["dropped"]:
                continue
            x = name.split("_", 1)[1]
            required.update(f"{prefix}__{m}" for prefix in (name, f"{name}_cf", f"BX_{x}") for m in METRICS)
            required.update(f"{name}__{k}" for k in ("fused_cells", "cf_cells", "sigma", "gate_a", "gate_b"))
    diff.extend(f"pipeline.{k}:missing" for k in sorted(required - set(arr)))
    with np.load(role.root / "rederive" / phase, allow_pickle=False) as z:
        diff.extend(f"{phase}.{k}:missing" for k in sorted(required - set(z.files)))
        for k in sorted(set(z.files) & set(arr)):
            if z[k].dtype != arr[k].dtype or not np.array_equal(z[k], arr[k]):
                diff.append(f"{phase}.{k}")
    if diff:
        path = R.fix_path(role.root / f"{stage}_disagreement.json")
        R.write_json_once(path, {"stage": stage, "differing_keys": diff, "smoke": role.smoke})
        raise SystemExit(f"re-derivation disagreement: {len(diff)} keys; {path}")
    R.write_json_once(verdict, {k: v for k, v in pending.items() if k not in ("rule_sha256", "written")})
    print(f"{stage}: finalize comparisons passed; {verdict.name}")
    if stage == "ceiling" and not role.smoke:
        if pending["zone"] == "STOP":
            print("STOP: design L ends; FV NO-GO; no stage 2 or fresh-seed build.")
        elif pending["zone"] == "WEAK":
            print("WEAK: ceiling is weak; continue with the report's first line flagged and a run-chat phone note.")


def smoke_inputs():
    """Synthetic row IDs are local fixture IDs; no real extraction joins are made."""
    rng = np.random.default_rng(9001)
    groups = np.repeat(np.arange(6000, dtype=np.int64), 4)
    st = np.flatnonzero(groups < 4000).astype(np.int64)
    selection = np.flatnonzero(groups >= 4000).astype(np.int64)
    features = {m: rng.normal(size=(len(groups), 16)).astype(np.float32) for m in ("img", "txt")}
    data = SimpleNamespace(img_features=features["img"], txt_features=features["txt"])
    sp = SimpleNamespace(groups=groups, scorer_train=st, selection=selection)
    membership = (groups[st] >= 2000).astype(np.int8)
    halves = {"scorer_train": st, "g_st": groups[st], "painting_of_local_row": groups[st],
              "half_of_local_row": membership}
    for h in (0, 1):
        local = np.flatnonzero(membership == h).astype(np.int64)
        halves[f"local_rows_half{h}"] = local
        halves[f"global_rows_half{h}"] = st[local]
    labels = {}
    for name in Q.BASE + GROUPINGS:
        nclass = 17 if name == "R0" else 27 if name == "T" else 8
        labels[name] = rng.integers(0, nclass, size=6000, dtype=np.int64)[groups]
    return data, sp, halves, labels


def smoke_prep(role):
    root = role.root
    data, sp, halves, full = smoke_inputs()
    st = sp.scorer_train
    for stem, names in (("r7_groupings", G.IDS), ("r7_grouping_N", ("N",))):
        if (root / "groupings" / f"{stem}.json").exists():
            continue
        arrays = {"scorer_train": st, "g_st": halves["g_st"]}
        arrays.update({x: full[x][st] for x in names})
        R.write_npz_once(root / "groupings" / f"{stem}.npz", **arrays)
        R.write_json_once(root / "groupings" / f"{stem}.json",
                          {"synthetic": True, "smoke": True, "groupings": {x: {"dropped": False} for x in names}})
    _group_records(root)
    for name in Q.BASE + GROUPINGS:
        if (root / "heads" / f"head_{name}.json").exists():
            rec = read_record(root / "heads" / f"head_{name}.json", True)
            assert rec["synthetic"] and not rec["dropped"], f"{name}: smoke head differs"
            if name != "T":
                Q.load_crossfit(name, full[name][st], halves, root)
            continue
        local = full[name][st]
        # R0's real build_heads asserts frozen real CSD labels; use the same public
        # fit functions for this synthetic fixture, with the fixed smoke sizes.
        clfs, post, prov = H.fit_standard(data, local, st, sp.selection, 5000, 1000)
        cf = None
        if name != "T":
            cf, _ = H.fit_crossfit(data, local, st, halves, 5000, 1000)
        R.write_pickle_once(root / "heads" / f"head_{name}.pkl", clfs)
        R.write_npz_once(root / "heads" / f"post_{name}.npz", **post, rows=sp.selection)
        if cf is not None:
            R.write_npz_once(root / "heads" / f"xheads_{name}.npz", **cf)
        R.write_json_once(root / "heads" / f"head_{name}.json",
                          {"grouping": name, "smoke": True, "synthetic": True, "dropped": False,
                           "n_groups": len(np.unique(local)), "label_sha256": G.sha_array(local), "standard": prov})
    for name in Q.GROUPINGS:
        path = root / "readers" / f"reader_{name}.json"
        if path.exists():
            rec = read_record(path, True)
            assert not rec["dropped"], f"synthetic reader {name} dropped"
            assert R.sha_file(path.with_suffix(".pkl")) == rec["reader_sha256"], f"{name}: reader SHA differs"
            continue
        labels = {h: full[h][st] for h in Q.BASE + (name,)}
        rec = Q.build_reader(name, labels, halves, root, True)
        assert not rec["dropped"], f"synthetic reader {name} dropped"
    R.write_json_once(root / "prep.json", {"smoke": True, "synthetic": True, "dropped": []})
    R.write_json_once(root / "stage0_regression.json", {"smoke": True, "synthetic": True, "passed": True})
    print(f"prep smoke: {len(GROUPINGS)} groupings; head draws 5000/1000; banks 256/block")


def make_bundle(role, extra):
    if not role.smoke:
        return B.build_bundle(role.dev, False, extra=extra, root=role.root)
    R.check_seed(role.dev, True)
    _, sp, _, labels = smoke_inputs()
    builder = R.load("src.train.pseudo_partitions")
    ep = builder.build_episode_bank({h: labels[h] for h in Q.BASE}, sp.groups, sp.selection,
                                    n_per_pair=64, seed=role.dev, min_paintings=30)
    n = len(ep.anchor)
    assert n == 192, "synthetic episode count differs"
    rng = np.random.default_rng(role.dev)
    cosine = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in ("i2t", "t2i")}
    cos = {c: cosine for c in ("a", "b")}
    term = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in ("i2t", "t2i")}
    t_n1u = {c: term for c in ("a", "b")}
    ctx = SimpleNamespace(groups=sp.groups, selection=sp.selection, pooled=ep,
                          pair_index=np.repeat(np.arange(3, dtype=np.int64), 64))
    bundle = SimpleNamespace(seed=role.dev, smoke=True, ctx=ctx, n=n, cl=sp.groups[ep.anchor],
                             pair_index=ctx.pair_index, parity=np.arange(n, dtype=np.int64) % 2,
                             cos=cos, t_n1u=t_n1u, post={})
    for name in dict.fromkeys(Q.BASE + ("R0",) + tuple(extra)):
        if name in Q.BASE:
            with np.load(role.root / "heads" / f"post_{name}.npz", allow_pickle=False) as z:
                assert np.array_equal(z["rows"], sp.selection), f"{name}: smoke selection differs"
                post = {m: z[m] for m in ("img", "txt")}
            B._posterior_ok(post, ctx, name)
            bundle.post[name] = post
        else:
            bundle.post[name] = B.load_posteriors(ctx, name, role.root)
    bundle.stack, bundle.F = B.stack_and_features(bundle, B.A0)
    bundle.readers = R.load("rb_build").load_readers("A0", False)[0]
    bundle.pBp = B.condition_free(bundle, B.A0)[0]
    bundle.pB = bundle.pBp  # Synthetic E2 comparator; never represents a real result.
    quick = R.load("src.eval.aspect_quick_checks")
    bundle.B = quick.crossfit_condition_free(cos, t_n1u, quick.uniform_probe_scores(bundle.post, ep, B.A0), bundle.parity)[0]
    return bundle


def chain_smoke(role):
    assert role.smoke and role.root == R.SMOKE_CHAIN, "chain smoke root differs"
    prep(role)
    ceiling(role)
    gate(role)
    print("chain smoke: prep -> ceiling -> gate assertions passed; no metrics printed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("prep", "ceiling", "gate"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--chain-smoke", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="use chain-smoke fixtures and dev seed 9001")
    args = parser.parse_args()
    R.assert_rule()
    R.assert_threads()
    role = R.roles("chain" if args.smoke or args.chain_smoke else "real")
    if args.chain_smoke:
        assert args.stage is None and not args.finalize, "chain smoke takes no stage or finalize"
        chain_smoke(role)
    else:
        assert args.stage is not None, "a stage is required"
        if args.finalize:
            assert args.stage in CORE, "prep has no verdict"
            finalize(role, args.stage)
        else:
            {"prep": prep, "ceiling": ceiling, "gate": gate}[args.stage](role)


if __name__ == "__main__":
    main()
