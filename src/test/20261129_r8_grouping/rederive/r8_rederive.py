"""Separate C13 implementation of M11--M15 and decision-rule sections 6--7.

Only the frozen R7 phase-A library pieces are used. No r8 scorer, grouping,
placement, head or gate implementation is read or imported. See r8_rederive.md
for the explicit stored-file invocation and pending-record contract.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r8_common as R

MAIN_GROUPS = ["V100", "V67", "V33", "NB100", "NB67", "NB33"] + [f"FC{i}" for i in range(1, 7)] + ["NBC"]
SCREEN_GROUPS = ["AL", "DI", "SG"]
CONTROL_INDEX = {g: i + 1 for i, g in enumerate(MAIN_GROUPS + SCREEN_GROUPS)}
KINDS = ("R1", "XAFF")
METRICS = ("r1", "gain", "other", "strict", "swap")
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
A0 = ("affect", "image", "caption")


def placements(phi, labels, halves, selection_phi):
    """M11/M12 in painting order, with float32 unit centres and float64 dots."""
    phi, selection_phi = np.asarray(phi), np.asarray(selection_phi)
    labels, halves = np.asarray(labels), np.asarray(halves)
    assert phi.ndim == 2 and selection_phi.ndim == 2 and phi.shape[1] == selection_phi.shape[1]
    assert phi.dtype == selection_phi.dtype == np.float32 and np.isfinite(phi).all() and np.isfinite(selection_phi).all()
    assert labels.shape == halves.shape == (len(phi),) and labels.dtype.kind in "iu"
    assert set(np.unique(halves)) == {0, 1}
    k = int(labels.max()) + 1
    assert k >= 2 and np.array_equal(np.unique(labels), np.arange(k)), "compact labels"

    def centres(mask):
        mus = []
        for g in range(k):
            members = phi[mask & (labels == g)]
            assert len(members), f"empty half/group {g}"
            mu = members.mean(axis=0, dtype=np.float32)
            norm = np.linalg.norm(mu)
            assert norm > 0 and np.isfinite(norm), "zero centre"
            mus.append(mu / norm)
        return np.asarray(mus, np.float32)

    def assign(x, mu):
        return np.eye(k, dtype=np.float32)[np.argmax(x.astype(np.float64) @ mu.astype(np.float64).T, axis=1)]

    own = assign(selection_phi, centres(np.ones(len(phi), bool)))
    bank = np.empty((len(phi), k), np.float32)
    for h in (0, 1):
        bank[halves == h] = assign(phi[halves == h], centres(halves != h))
    return {"selection": own, "bank": bank}


def library(name):
    """R7 section 9 phase A permits frozen objects, zscore and CF/CI routines."""
    return R.load_r7("r7_common").load(name)


def counts(scores):
    n = len(scores["a"]["i2t"])
    hit, other, swap, strict = (np.zeros(n, np.int64) for _ in range(4))

    def first(s, target):
        assert s.shape == (n, 13) and np.isfinite(s).all(), "score boundary"
        win = s < s[:, target, None]
        win[:, target] = True
        return win.all(axis=1)

    for d in DIRS:
        a, b = scores["a"][d], scores["b"][d]
        aa, bb, ab, ba = first(a, 0), first(b, 1), first(a, 1), first(b, 0)
        hit += aa.astype(np.int64) + bb
        other += ab.astype(np.int64) + ba
        swap += (a[:, 0] > a[:, 1]) & (b[:, 1] > b[:, 0])
        strict += aa & bb
    return np.stack([hit, other, swap, strict])


def metrics(cnt):
    hit, other, swap, strict = cnt.astype(np.int64)
    return {"r1": hit / 4., "gain": (hit - other) / 4., "other": other / 4.,
            "swap": swap / 2., "strict": strict / 2.}


def reader_values(P):
    out = {}
    for c in CONDS:
        p = np.asarray(P[c])
        assert p.dtype == np.float64 and p.ndim == 2 and p.shape[1] in (3, 4)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.allclose(p.sum(1), 1, atol=1e-12, rtol=0), "reader probability mass"
        s = np.sort(p, axis=1)
        out[c] = (p.argmax(axis=1).astype(np.int64), s[:, -1] - s[:, -2])
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


_ACTIVE_LEDGER = None
_IN_AUDIT = False


def _audit_open(event, args):
    # Includes data and episodes opened inside the permitted frozen libraries.
    global _IN_AUDIT
    if event != "open" or _ACTIVE_LEDGER is None or _IN_AUDIT or not isinstance(args[0], (str, bytes)):
        return
    mode = args[1]
    if mode is not None and any(c in mode for c in "wax+"):
        return
    p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).absolute()
    if not any(p.is_relative_to(base) for base in (R.MAIN, Path("/data"), Path(__file__).resolve().parents[4])):
        return
    if p.is_file():
        _IN_AUDIT = True
        try:
            _ACTIVE_LEDGER.touch(p)
        finally:
            _IN_AUDIT = False


sys.addaudithook(_audit_open)


class Files:
    """SHA before use, explicit reads plus audited reads of frozen dependencies."""
    def __init__(self):
        self.hashes = {}
        self.disagreements = []
        # The governing table, rather than a duplicate drifting list of hashes.
        self.known = {}
        for line in R.RULE.read_text().splitlines():
            if not line.startswith("| "):
                continue
            columns = line.split("|")
            paths = re.findall(r"`([^`]+)`", columns[1])
            hashes = re.findall(r"\b[0-9a-f]{64}\b", columns[2])
            if len(paths) != len(hashes):
                continue
            for name, digest in zip(paths, hashes):
                if name.startswith("…/"):
                    name = "20261126_design_l_style_source/" + name[2:]
                p = Path(name)
                if not p.is_absolute():
                    p = R.MAIN / "src/test" / (name if "/" in name else "20261126_design_l_style_source/" + name)
                self.known[str(p)] = digest
        self.touch(R.RULE)

    def touch(self, path, expected=None):
        p = Path(path).absolute()
        key = str(p)
        if key not in self.hashes:
            self.hashes[key] = sha(p)
            if key in self.known:
                assert self.hashes[key] == self.known[key], f"rule section 10 SHA-256 differs: {p}"
        if expected is not None and self.hashes[key] != expected:
            self.disagreements.append({"where": key, "reason": "input SHA-256 differs", "expected": expected, "actual": self.hashes[key]})
        return p

    def npz(self, path):
        with np.load(self.touch(path), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def json(self, path):
        return json.loads(self.touch(path).read_text())

    def compare(self, where, got, want, atol=0):
        ga, wa = np.asarray(got), np.asarray(want)
        equal = ga.shape == wa.shape and (np.array_equal(ga, wa, equal_nan=True) if ga.dtype.kind in "fiub" and wa.dtype.kind in "fiub"
                                         else np.array_equal(ga, wa))
        if not equal and atol and ga.shape == wa.shape and ga.dtype.kind in "fiu" and wa.dtype.kind in "fiu":
            equal = np.allclose(ga, wa, rtol=0, atol=atol, equal_nan=True)
        if not equal:
            rec = {"where": where, "reason": "values differ", "derived_shape": list(ga.shape), "stored_shape": list(wa.shape)}
            if ga.shape == wa.shape and ga.size and ga.dtype.kind in "fiub" and wa.dtype.kind in "fiub":
                changed = ~(np.isclose(ga, wa, atol=atol, rtol=0, equal_nan=True))
                rec["changed"] = int(changed.sum())
                rec["first_index"] = np.argwhere(changed)[0].tolist()
            self.disagreements.append(rec)

    def finish(self):
        for p, before in list(self.hashes.items()):
            after = sha(p)
            if after != before:
                self.disagreements.append({"where": p, "reason": "input changed during re-derivation", "before": before, "after": after})


def stack_post(post, ep, parts):
    return {d: np.stack([np.einsum("nc,nkc->nk", post[g][qm][ep.anchor], post[g][cm][ep.candidates]) for g in parts], axis=1)
            for d, qm, cm in (("i2t", "img", "txt"), ("t2i", "txt", "img"))}


def condition_free(cos, tu, post, ep, parts, parity):
    stack = stack_post(post, ep, parts)
    term = {c: {d: stack[d].mean(axis=1) for d in DIRS} for c in CONDS}
    return library("src.eval.aspect_quick_checks").crossfit_condition_free(cos, tu, term, parity)[0]


def read_post(files, path, c):
    z = files.npz(path)
    assert {"img", "txt", "rows"} <= set(z), f"head keys {path}"
    assert z["rows"].dtype == np.int64 and np.array_equal(z["rows"], c.selection), f"head selection alignment {path}"
    mask = np.zeros(len(c.groups), bool)
    mask[c.selection] = True
    for m in ("img", "txt"):
        p = z[m]
        assert p.dtype == np.float32 and p.ndim == 2 and p.shape[0] == len(mask) and p.shape[1] >= 2
        assert np.isfinite(p[mask]).all() and np.isnan(p[~mask]).all(), f"head mask {path}"
        assert ((p[mask] >= 0) & (p[mask] <= 1)).all() and np.allclose(p[mask].sum(1), 1, rtol=0, atol=1e-6)
    assert z["img"].shape == z["txt"].shape
    return {m: z[m] for m in ("img", "txt")}


def inputs(files, fixture, smoke):
    """Common supplies all real joins; the explicit smoke fixture supplies tiny ones."""
    if fixture:
        assert smoke, "synthetic inputs are smoke only"
        z = files.npz(fixture)
        keys = ("scorer_train", "g_st", "halves", "selection", "groups", "painting_image", "row_to_image", "image_index")
        c = SimpleNamespace(**{k: z[k] for k in keys})
        ep = SimpleNamespace(anchor=z["anchor"], candidates=z["candidates"])
        cos = {cond: {d: z[f"cos__{cond}__{d}"] for d in DIRS} for cond in CONDS}
        tu = {cond: {d: z[f"tu__{cond}__{d}"] for d in DIRS} for cond in CONDS}
        e2 = {g: {m: z[f"e2__{g}__{m}"] for m in ("img", "txt")} for g in A0}
        ids = {k: z[k] for k in ("cl", "pair_index", "parity")}
    else:
        R.assert_rule()
        common = R.ctx()
        c = SimpleNamespace(scorer_train=common.scorer_train, g_st=common.g_st, halves=common.halves,
                            selection=common.selection, groups=common.groups, painting_image=R.painting_image(),
                            row_to_image=R.row_to_image(), image_index=R.nonheld_images())
        context = library("run_gonogo").EvalContext(9001 if smoke else 42, smoke)
        assert np.array_equal(context.selection, c.selection) and np.array_equal(context.groups, c.groups)
        ep, cos = context.pooled, context.cos
        # Loading/encoding the frozen fitted object is permitted by R7 phase A;
        # the N1u term below is our arithmetic, with no scorer/helper import.
        checkpoint = R.MAIN / "src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt"
        files.touch(checkpoint, "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2")
        ic, tc = context.encode(checkpoint)
        sides = {}
        for d, qcodes, ccodes, ex in (("i2t", ic, tc, np.concatenate([ep.pairs_a_img, ep.pairs_b_img], axis=1)),
                                    ("t2i", tc, ic, np.concatenate([ep.pairs_a_txt, ep.pairs_b_txt], axis=1))):
            q, cand, examples = qcodes[ep.anchor].astype(np.float64), ccodes[ep.candidates].astype(np.float64), qcodes[ex].astype(np.float64)
            w = np.full(q.shape, 1. / q.shape[1])
            sides[d] = np.einsum("nf,nf,nkf->nk", w, q - examples.mean(1), cand - cand.mean(1, keepdims=True)).astype(np.float32)
        tu = {cond: sides for cond in CONDS}
        path = R.MAIN / "src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz"
        files.touch(path, "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0")
        z = files.npz(path)
        assert np.array_equal(z["selection"], c.selection)
        e2 = {}
        for g in A0:
            e2[g] = {}
            for m in ("img", "txt"):
                full = np.full((len(c.groups), z[f"{g}__{m}"].shape[1]), np.nan, np.float32)
                full[c.selection] = z[f"{g}__{m}"]
                e2[g][m] = full
        ids = {"cl": context.anchor_group.astype(np.int64), "pair_index": context.pair_index.astype(np.int64), "parity": context.parity.astype(np.int64)}
        if not smoke:
            assert len(ep.anchor) == 12288
    n = len(ep.anchor)
    assert ep.candidates.shape == (n, 13)
    assert np.isin(ep.anchor, c.selection).all() and np.isin(ep.candidates, c.selection).all()
    assert all(v.dtype == np.int64 and v.shape == (n,) for v in ids.values())
    assert np.array_equal(ids["parity"], np.arange(n) % 2)
    return c, ep, cos, tu, e2, ids


def random_inputs(parent, selection_codes, index):
    k, n = int(parent.max()) + 1, len(parent)
    sizes = np.bincount(parent, minlength=k)
    shares = sizes.astype(np.float64) / n
    perm = np.random.default_rng(7100 + index).permutation(n)
    labels = np.empty(n, np.int64)
    labels[perm] = np.repeat(np.arange(k), sizes)
    bank = np.random.default_rng(7200 + index).choice(k, size=n, p=shares)
    _, inverse = np.unique(selection_codes, return_inverse=True)
    draw = np.random.default_rng(7300 + index).choice(k, size=int(inverse.max()) + 1, p=shares)
    onehot = np.eye(k, dtype=np.float32)
    return labels, {"bank": onehot[bank], "selection": onehot[draw[inverse]]}


def derive_files(root, *, fixture=None, smoke=False, groupings=None, files=None, aff_reader=None, tp_map=None, keep_counts=False):
    """Read stored φ, labels, halves, posteriors, reader probabilities and taus."""
    global _ACTIVE_LEDGER
    files, root = files or Files(), Path(root)
    previous, _ACTIVE_LEDGER = _ACTIVE_LEDGER, files
    try:
        files.touch(R.RULE)
        c, ep, cos, tu, e2, arrays = inputs(files, fixture, smoke)
        grouping_paths = groupings or [root / "groupings/r8_groupings.json"]
        labels, records, dropped = {}, {}, {}
        for path in grouping_paths:
            rec = files.json(path)
            assert {"groupings", "dropped"} <= set(rec)
            assert all(isinstance(v, str) and v for v in rec["dropped"].values()), "drop reasons"
            for g, value in rec["groupings"].items():
                assert g not in records or records[g] == value, f"duplicate grouping record {g}"
                records[g] = value
            dropped.update(rec["dropped"])
            z = files.npz(Path(path).with_suffix(".npz"))
            assert np.array_equal(z["scorer_train"], c.scorer_train) and np.array_equal(z["g_st"], c.g_st), "grouping alignment"
            for g in rec["groupings"]:
                if g in dropped:
                    continue
                assert g in z and z[g].shape == c.scorer_train.shape and z[g].dtype == np.int64
                first = np.unique(c.g_st, return_index=True)[1]
                lab = z[g][first]
                assert np.array_equal(lab[c.g_st], z[g]), "labels constant within painting"
                assert len(np.unique(lab)) >= 2 and np.array_equal(np.unique(lab), np.arange(lab.max() + 1)), "grouping classes"
                if g in labels:
                    files.compare(f"duplicate grouping {g}", labels[g], lab)
                labels[g] = lab
        assert not set(labels) & set(dropped), "evaluated and dropped grouping"
        assert set(labels) <= set(MAIN_GROUPS + SCREEN_GROUPS + ["R0own", "NRK", "TP"] + ["Z" + g for g in CONTROL_INDEX]), "grouping ID"
        if not smoke:
            accounted = set(records) | set(dropped)
            assert set(MAIN_GROUPS[:-1] + SCREEN_GROUPS + ["R0own", "NRK", "TP"]) <= accounted, "unrecorded candidate/reference drop"
            assert {"Z" + g for g in labels if g in CONTROL_INDEX} <= accounted, "unrecorded control drop"
        posts = {g: read_post(files, root / f"heads/post_{g}.npz", c) for g in A0}
        legacy = root / "heads/post_R0.npz" if fixture else R.MAIN / "src/test/20261126_design_l_style_source/results/heads/post_R0.npz"
        posts["legacy_R0"] = read_post(files, legacy, c)
        B = condition_free(cos, tu, e2, ep, A0, arrays["parity"])
        for name, score in (("B", B), ("B0", condition_free(cos, tu, posts, ep, A0, arrays["parity"])),
                            ("B1", condition_free(cos, tu, posts, ep, (*A0, "legacy_R0"), arrays["parity"]))):
            arrays.update({f"{name}__{m}": v for m, v in metrics(counts(score)).items()})
        tau_table = files.json(root / "taus.json")
        # rule gap: the shared reader contract does not name AFF's A0 reader;
        # default to reader_A0.npz and accept --aff-reader for its stored path.
        az = files.npz(aff_reader or root / "seed42/reader_A0.npz")
        AP = {cond: az[f"P__{cond}"] for cond in CONDS}
        aff_tau = [3.8684538364530674e-05, .21702129490553143, .47973989883399526, .7502585816077211]
        out, detail = family(B, stack_post(posts, ep, A0), AP, aff_tau, arrays["parity"], "AFF")
        add_family(arrays, "AFF", out, detail, keep_counts)
        for g in labels:
            posts[g] = read_post(files, root / f"heads/post_{g}.npz", c)
            k = int(labels[g].max()) + 1
            assert posts[g]["img"].shape[1] == k, f"posterior K {g}"
            if g.startswith("Z"):
                parent = g[1:]
                assert parent in labels and parent in CONTROL_INDEX, "control without evaluated parent"
                rl, placement = random_inputs(labels[parent], c.groups[c.selection], CONTROL_INDEX[parent])
                files.compare(f"M13 labels {g}", rl, labels[g])
            elif g == "TP":
                # rule gap: M9's stored text-vector and merge-map schema is
                # unspecified. --tp-map supplies text_vectors and merge_map.
                assert tp_map, "TP requires --tp-map (text_vectors, merge_map)"
                tp = files.npz(tp_map)
                assert {"text_vectors", "merge_map"} <= set(tp)
                xp = R.x_paintings()
                xs = R.x_rows(c.selection)
                vec, merge = tp["text_vectors"], tp["merge_map"]
                assert vec.dtype == np.float32 and vec.ndim == 2 and merge.shape == (len(vec),) and merge.dtype.kind in "iu"
                assert np.isfinite(vec).all() and merge.min() >= 0 and merge.max() < k
                onehot = np.eye(k, dtype=np.float32)
                placement = {"bank": onehot[merge[(xp @ vec.T).argmax(1)]], "selection": onehot[merge[(xs @ vec.T).argmax(1)]]}
                files.compare("TP painting labels", placement["bank"].argmax(1), labels[g])
            else:
                space = files.npz(root / f"spaces/{g}.npz")
                assert {"phi", "image_index"} <= set(space)
                phi, images = space["phi"], space["image_index"]
                assert np.array_equal(images, c.image_index), f"space image alignment {g}"
                assert phi.dtype == np.float32 and phi.ndim == 2 and len(phi) == len(images) and np.isfinite(phi).all()
                assert np.allclose(np.linalg.norm(phi, axis=1), 1, rtol=0, atol=1e-5), "unit space"
                pp, sp = np.searchsorted(images, c.painting_image), np.searchsorted(images, c.row_to_image[c.selection])
                assert np.array_equal(images[pp], c.painting_image) and np.array_equal(images[sp], c.row_to_image[c.selection])
                placement = placements(phi[pp], labels[g], c.halves, phi[sp])
            files.compare(f"M11 post_{g} img", placement["selection"], posts[g]["img"][c.selection])
            # The independently recovered own map drives our scoring too.
            posts[g]["img"] = posts[g]["img"].copy()
            posts[g]["img"][c.selection] = placement["selection"]
            bank = files.npz(root / f"heads/xheads_{g}.npz")
            assert {"rows", "img", "txt", "classes_half0", "classes_half1", "filled_by_half"} <= set(bank)
            assert np.array_equal(bank["rows"], c.scorer_train), "bank row alignment"
            assert bank["img"].shape == bank["txt"].shape == (len(c.scorer_train), k)
            assert bank["img"].dtype == bank["txt"].dtype == np.float32 and np.isfinite(bank["txt"]).all()
            assert np.allclose(bank["txt"].sum(1), 1, atol=1e-6, rtol=0)
            files.compare(f"M12 xheads_{g} img", placement["bank"][c.g_st], bank["img"])
            files.compare(f"M12 filled_by_half {g}", 1 - c.halves[c.g_st], bank["filled_by_half"])
            for h in (0, 1):
                files.compare(f"M12 classes_half{h} {g}", np.arange(k), bank[f"classes_half{h}"])
            parts = (*A0, g)
            reader = files.npz(root / f"seed42/reader_{g}.npz")
            assert {f"{key}__{cond}" for cond in CONDS for key in ("P", "m", "pick")} <= set(reader)
            P = {cond: reader[f"P__{cond}"] for cond in CONDS}
            pm = reader_values(P)
            assert all(P[cond].shape == (len(ep.anchor), 4) for cond in CONDS)
            for cond in CONDS:
                files.compare(f"reader_{g} pick {cond}", pm[cond][0], reader[f"pick__{cond}"])
                files.compare(f"reader_{g} margin {cond}", pm[cond][1], reader[f"m__{cond}"])
            tau = np.percentile(np.concatenate([pm[cond][1] for cond in CONDS]), [0, 25, 50, 75])
            files.compare(f"tau {g}", tau, tau_table[g])
            bx = condition_free(cos, tu, posts, ep, parts, arrays["parity"])
            arrays.update({f"BX_{g}__{m}": v for m, v in metrics(counts(bx)).items()})
            for kind in KINDS:
                out, detail = family(B, stack_post(posts, ep, parts), P, tau_table[g], arrays["parity"], kind)
                add_family(arrays, f"{kind}_{g}", out, detail, keep_counts)
        stats = statistics(arrays, set(labels), dropped)
        stats["groupings"] = records
        return {"arrays": arrays, "statistics": stats, "input_sha256": files.hashes, "disagreements": files.disagreements}
    finally:
        _ACTIVE_LEDGER = previous


def add_family(arrays, name, values, detail, keep_counts=False):
    for key, value in values.items():
        arrays[(name + "_cf__" + key[4:]) if key.startswith("cf__") else (name + "__" + key)] = value
    # Pipeline batches may store any of these 224-cell integer checks too.
    for key, value in detail.items():
        if key not in ("f_cnt", "c_cnt") or keep_counts:
            arrays[name + "__" + key] = value


def compare_json(files, path, got, want):
    """Exact discrete fields; only intervals have the C13 1e-9 tolerance."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            where = f"{path}.{key}"
            if key not in got or key not in want:
                files.disagreements.append({"where": where, "reason": "missing record key"})
            else:
                compare_json(files, where, got[key], want[key])
    elif isinstance(got, bool) or isinstance(want, bool):
        if type(got) is not type(want) or got != want:
            files.disagreements.append({"where": path, "reason": "boolean differs"})
    else:
        files.compare(path, got, want, atol=1e-9 if path.endswith("ci95") else 0)


def read_declared_inputs(files, pending, base):
    """Cover the pending record's explicit input manifest, including unscored files."""
    # rule gap: index specifies coverage but no pending-file input schema.
    # Accept inputs/input_sha256 as path->SHA, path records or a list of paths.
    for field in ("inputs", "input_sha256"):
        values = pending.get(field, {})
        if isinstance(values, dict):
            values = [(k, v) for k, v in values.items()]
        else:
            assert isinstance(values, list), "pending input manifest"
            values = [(v, None) if isinstance(v, str) else (v["path"], v.get("sha256")) for v in values]
        for key, value in values:
            if isinstance(value, dict):
                path, expected = value.get("path", key), value.get("sha256")
            else:
                path, expected = key, value
            p = Path(path)
            files.touch(p if p.is_absolute() else base / p, expected)


def stage_checks(files, path, root, smoke):
    """Rule 7(d): only recomputable ESC1 b--d and SC2 finite/count checks."""
    if path is None:
        assert smoke, "real C13 requires --stage-evidence (ESC1 b--d and SC2 stored evidence)"
        return {}
    evidence = files.json(path)
    out = {}
    # rule gap: no file/key contract is given for raw ESC1/SC2 evidence or
    # stage-record subkeys. This explicit manifest maps stored paths and keys;
    # its expected flags are read from the pipeline stage record, never guessed.
    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else Path(path).parent / p

    def field(rec, key):
        for part in key.split("."):
            rec = rec[part]
        return rec

    for name, item in evidence.get("encoders", {}).items():
        assert name in SCREEN_GROUPS
        record = files.json(resolve(item["record"]))
        if item.get("not_produced"):
            assert record["pass"] is False and isinstance(item["not_produced"], str)
            out[name] = {"status": "not produced", "reason": item["not_produced"]}
            continue
        assert "fixture" not in item or smoke, "synthetic stage joins are smoke only"
        emb = np.load(files.touch(resolve(item["embeddings"])), mmap_mode="r", allow_pickle=False)
        ix = np.load(files.touch(resolve(item["image_index"])), allow_pickle=False)
        expected = (files.npz(resolve(item["fixture"]))["image_index"] if "fixture" in item else R.nonheld_images()).astype(np.int64)
        dims = {"AL": 256, "DI": 1024, "SG": 1152}
        n = len(expected) if smoke else 49121
        flags = {"shape": bool(emb.shape == (n, dims[name])), "finite": bool(np.isfinite(emb).all()),
                 "index": bool(ix.shape == expected.shape and np.array_equal(ix, expected)
                               and hashlib.sha256(np.ascontiguousarray(ix, dtype=np.int64).tobytes()).hexdigest() == item["index_sha256"])}
        assert set(item["keys"]) == set(flags), "ESC1 evidence keys"
        for check, value in flags.items():
            compare_json(files, f"ESC1 {name}.{check}", value, field(record, item["keys"][check]))
        out[name] = flags
    item = evidence.get("fcmi")
    if item:
        record = files.json(resolve(item["record"]))
        if item.get("not_produced"):
            assert record["pass"] is False and isinstance(item["not_produced"], str)
            out["FCMI"] = {"status": "not produced", "reason": item["not_produced"]}
        else:
            labels = np.load(files.touch(resolve(item["labels"])), allow_pickle=False)
            losses = np.load(files.touch(resolve(item["losses"])), allow_pickle=False)
            assert labels.ndim == 1 and labels.dtype.kind in "iu" and losses.ndim == 1 and len(losses)
            if not smoke:
                assert len(labels) == 36518
            flags = {"label_count": bool(len(np.unique(labels)) >= 2), "finite_loss": bool(np.isfinite(losses).all())}
            assert set(item["keys"]) == set(flags), "SC2 evidence keys"
            for check, value in flags.items():
                compare_json(files, f"SC2.{check}", value, field(record, item["keys"][check]))
            out["FCMI"] = {**flags, "n_labels": int(len(np.unique(labels)))}
    if not smoke:
        assert set(evidence["encoders"]) == set(SCREEN_GROUPS) and item, "all stages need evidence or a recorded non-production reason"
    return out


def check_files(root, pending, step, *, indexes=None, fixture=None, smoke=False,
                groupings=None, stage_evidence=None, aff_reader=None, tp_map=None):
    """Public stored-file seam. Disagreements prevent the CLI's successful exit."""
    assert step in ("trigger", "gate")
    files, root = Files(), Path(root).absolute()
    pending = Path(pending).absolute()
    rec = files.json(pending)
    assert isinstance(rec, dict)
    read_declared_inputs(files, rec, pending.parent)
    if "smoke" in rec:
        assert rec["smoke"] is smoke, "pending smoke role"
    if "rule_sha256" in rec:
        files.compare("pending rule SHA", sha(R.RULE), rec["rule_sha256"])
    stored = {}
    for index_path in indexes or [root / "seed42/index.json"]:
        index = files.json(index_path)
        assert isinstance(index["batches"], list) and index["batches"], "seed42 index batches"
        for batch in index["batches"]:
            assert {"file", "sha256"} <= set(batch)
            path = Path(batch["file"])
            path = path if path.is_absolute() else Path(index_path).parent / path
            files.touch(path, batch["sha256"])
            arrays = files.npz(path)
            for k in ("cl", "pair_index", "parity", "AFF__r1", "B__r1", "B0__r1", "B1__r1"):
                assert k in arrays, f"batch lacks {k}"
            assert all(arrays[k].dtype == np.int64 and arrays[k].ndim == 1 for k in ("cl", "pair_index", "parity")), "batch ID dtypes/shapes"
            assert all(arrays[k].shape == arrays["cl"].shape for k in ("pair_index", "parity")), "batch ID row counts"
            for k, a in arrays.items():
                if k in stored:
                    files.compare(f"duplicate batch key {k}", stored[k], a)
                else:
                    stored[k] = a
    derived = derive_files(root, fixture=fixture, smoke=smoke, groupings=groupings, files=files,
                           aff_reader=aff_reader, tp_map=tp_map,
                           keep_counts=any(k.endswith(("__f_cnt", "__c_cnt")) for k in stored))
    full = derived["arrays"]
    if not smoke:
        available = set(derived["statistics"]["groupings"])
        if step == "trigger":
            assert "NBC" not in available and "ZNBC" not in available, "NBC ran before its trigger C13"
        elif derived["statistics"]["trigger"]["triggered"]:
            assert "NBC" in available or "NBC" in derived["statistics"]["main"]["dropped"], "triggered NBC has neither a result nor a recorded drop"
        else:
            assert "NBC" not in available and "ZNBC" not in available, "NBC evaluated without trigger"
    optional = {name + "__" + detail for name in ["AFF"] + [f"{kind}_{g}" for g in derived["statistics"]["groupings"] for kind in KINDS]
                for detail in ("rho_f", "gam_f", "rho_c", "crit_f", "ctrl_rho", "f_cnt", "c_cnt")}
    for k in sorted(set(full) - optional):
        if k not in stored:
            files.disagreements.append({"where": k, "reason": "required pipeline array missing"})
        else:
            files.compare(k, full[k], stored[k])
    for k in sorted(set(stored) - set(full)):
        files.disagreements.append({"where": k, "reason": "pipeline array was not independently derived"})
    for k in sorted(optional & set(stored)):
        files.compare(k, full[k], stored[k])
    computed = derived["statistics"]["trigger"] if step == "trigger" else derived["statistics"]
    # rule gap: the index specifies the C13 result but no pending verdict key
    # layout. The documented canonical layout is `expected`, or the same
    # fields at top level with inputs/rule/time metadata removed.
    expected = rec.get("expected", {k: v for k, v in rec.items() if k not in ("inputs", "input_sha256", "smoke", "rule_sha256", "written", "step")})
    compare_json(files, "pending", computed, expected)
    stages = stage_checks(files, stage_evidence, root, smoke)
    files.finish()
    return {"status": "agree" if not files.disagreements else "disagree", "step": step, "smoke": bool(smoke),
            "disagreements": files.disagreements, "input_sha256": files.hashes, "rederived": computed, "stage_checks": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("trigger", "gate"), required=True)
    parser.add_argument("--root", type=Path, default=R.RESULTS)
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--index", type=Path, action="append")
    parser.add_argument("--groupings", type=Path, action="append")
    parser.add_argument("--aff-reader", type=Path)
    parser.add_argument("--tp-map", type=Path)
    parser.add_argument("--stage-evidence", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.root.resolve().is_relative_to(R.SMOKE.resolve()) if args.smoke else not args.root.resolve().is_relative_to(R.SMOKE.resolve()), "output smoke role (set R8_RESULTS for worktree tests)"
    record = check_files(args.root, args.pending, args.step, indexes=args.index, fixture=args.fixture, smoke=args.smoke,
                         groupings=args.groupings, stage_evidence=args.stage_evidence, aff_reader=args.aff_reader, tp_map=args.tp_map)
    path = args.root / f"rederive/{args.step}_rederive.json"
    R.write_json_once(path, record)
    print(f"{args.step}: {record['status']}; {len(record['disagreements'])} disagreements; {path}")
    return 0 if record["status"] == "agree" else 1


def family(B, stack, P, taus, parity, kind):
    """224 cells, integer tune statistics, independent opposite-parity assembly."""
    import torch
    zscore = library("src.model.aspect_rule").zscore_rows
    tau, parity = np.asarray(taus, np.float64), np.asarray(parity)
    assert tau.shape == (4,) and np.isfinite(tau).all() and (np.diff(tau) >= 0).all()
    assert ((tau >= 0) & (tau <= 1)).all() and set(np.unique(parity)) == {0, 1}
    assert kind in (*KINDS, "AFF")
    pm = reader_values(P)
    n = len(parity)
    z = lambda a: zscore(torch.as_tensor(a, dtype=torch.float32)).numpy()
    zb, zt = {}, {}
    for d in DIRS:
        assert np.array_equal(B["a"][d], B["b"][d]), "condition-free B"
        assert B["a"][d].shape == (n, 13) and stack[d].shape == (n, P["a"].shape[1], 13)
        assert np.isfinite(stack[d]).all()
        zb[d] = z(B["a"][d])
    for c in CONDS:
        zt[c] = {d: z(np.einsum("nh,nhk->nk", P[c], stack[d].astype(np.float64)).astype(np.float32)) for d in DIRS}
    gate = {}
    for c in CONDS:
        pick, margin = pm[c]
        allowed = np.ones(n, bool) if kind == "R1" else np.isin(pick, (0,) if kind == "AFF" else (0, 3))
        gate[c] = (margin[None, :] >= tau[:, None]) & allowed[None, :]
    terms = [{c: {d: gate[c][t, :, None].astype(np.float32) * zt[c][d] for d in DIRS} for c in CONDS} for t in range(4)]
    cf_terms = [{d: (.5 * (terms[t]["a"][d].astype(np.float64) + terms[t]["b"][d].astype(np.float64))).astype(np.float32)
                 for d in DIRS} for t in range(4)]
    nu, na = (0., .5, 1., 2., 4., 8., 16.), (0., .25, .5, 1., 2., 4., 8., 16.)
    sigmas = sorted({u + a for u in nu for a in na})

    def combine(base, term, u, a):
        sc = base
        if u > 0:
            sc = sc + np.float32(u) * base
        if a > 0:
            sc = sc + np.float32(a) * term
        return sc

    ctrl = np.empty((len(sigmas), 2), np.int64)
    for j, sigma in enumerate(sigmas):
        cnt = counts({c: {d: combine(zb[d], None, sigma, 0) for d in DIRS} for c in CONDS})
        ctrl[j] = [cnt[0, parity == h].sum() for h in (0, 1)]
    si = ctrl.argmax(axis=0)
    rho_ctrl = ctrl[si, np.arange(2)]
    fc, cc = np.empty((224, 4, n), np.int8), np.empty((224, 4, n), np.int8)

    def scores(cid, counterpart):
        t, u, a = cid // 56, (cid // 8) % 7, cid % 8
        return {c: {d: combine(zb[d], cf_terms[t][d] if counterpart else terms[t][c][d], nu[u], na[a])
                    for d in DIRS} for c in CONDS}

    for cid in range(224):
        fc[cid], cc[cid] = counts(scores(cid, False)), counts(scores(cid, True))
    rf = np.stack([fc[:, 0, parity == h].sum(axis=1, dtype=np.int64) for h in (0, 1)], axis=1)
    rc = np.stack([cc[:, 0, parity == h].sum(axis=1, dtype=np.int64) for h in (0, 1)], axis=1)
    gf = np.stack([(fc[:, 0, parity == h].astype(np.int64) - fc[:, 1, parity == h]).sum(axis=1) for h in (0, 1)], axis=1)
    criterion = np.minimum(rf - rho_ctrl, gf)
    chosen_f, chosen_c = criterion.argmax(axis=0), rc.argmax(axis=0)
    out = {"fused_cells": chosen_f.astype(np.int64), "cf_cells": chosen_c.astype(np.int64),
           "sigma": np.asarray(sigmas, np.float64)[si], **{f"gate_{c}": gate[c].astype(np.float32) for c in CONDS}}
    for prefix, cnt, chosen, counterpart in (("", fc, chosen_f, False), ("cf__", cc, chosen_c, True)):
        assembled = np.empty((4, n), np.int64)
        literal = {c: {d: np.empty((n, 13), np.float32) for d in DIRS} for c in CONDS}
        for h in (0, 1):
            apply = parity != h
            assembled[:, apply] = cnt[chosen[h]][:, apply]
            sc = scores(chosen[h], counterpart)
            for c in CONDS:
                for d in DIRS:
                    literal[c][d][apply] = sc[c][d][apply]
        assert np.array_equal(counts(literal), assembled), "literal cross-fit assembly"
        out.update({prefix + m: a for m, a in metrics(assembled).items()})
    assert not out["cf__gain"].any(), "condition-free counterpart"
    detail = {"rho_f": rf, "gam_f": gf, "rho_c": rc, "crit_f": criterion, "ctrl_rho": ctrl,
              "f_cnt": fc, "c_cnt": cc}
    return out, detail


def q4(a):
    a = 4 * np.asarray(a, np.float64)
    assert np.isfinite(a).all() and np.allclose(a, np.rint(a), rtol=0, atol=1e-9), "quarter integer values"
    return np.rint(a).astype(np.int64)


def interval(a, cl):
    """Own C9 call; never consumes the pipeline's interval or resamples."""
    lo, hi = library("src.eval.aspect_metrics").cluster_bootstrap(a, cl, n_boot=5000, seed=42, chunk=250)["ci95"]
    return [100 * lo, 100 * hi]


def statistics(a, present, dropped):
    """Rule 6.1--6.5 (and the deterministic descriptive H target of 6.6)."""
    cl, pair = a["cl"], a["pair_index"]
    n = len(cl)
    assert n > 0 and pair.shape == (n,) and np.array_equal(np.bincount(pair, minlength=3), [n // 3] * 3)
    assert len(np.unique(cl)) >= 2
    all_stats = {}
    for g in sorted(present):
        for kind in KINDS:
            name = f"{kind}_{g}"
            s, cf = a[name + "__r1"], a[name + "_cf__r1"]
            assert s.shape == cf.shape == (n,)
            dg = a[name + "__gain"] - a[name + "_cf__gain"]
            assert not a[name + "_cf__gain"].any(), "CF gain"
            d = s - a["AFF__r1"]
            sd = int(q4(d).sum())
            ci = interval(d, cl)
            sc3 = {"sum4d": sd, "n": n, "point": 25 * sd / n, "ci95": ci, "pass": ci[0] > 0}
            sc4 = {}
            for p, label in enumerate(("emotion__style", "emotion__genre")):
                sel = pair == p
                total = int(q4(d[sel]).sum())
                sc4[label] = {"sum4d": total, "n": int(sel.sum()), "point": 25 * total / int(sel.sum()), "pass": total >= -49}
            sc4["pass"] = all(sc4[label]["pass"] for label in ("emotion__style", "emotion__genre"))
            comps = [("B1", a["B1__r1"]), ("BX", a[f"BX_{g}__r1"]), ("CF", cf), ("B0", a["B0__r1"]), ("B", a["B__r1"])]
            bar_name, bar = max(comps, key=lambda kv: int(q4(kv[1]).sum()))
            db, sg = s - bar, pair == 2
            sb, gg = int(q4(db).sum()), int(q4(dg).sum())
            bci, gci = interval(db, cl), interval(dg, cl)
            s1 = {"sum4d": sb, "point": 25 * sb / n, "ci95": bci,
                  "gain_sum4d": gg, "gain_point": 25 * gg / n, "gain_ci95": gci,
                  "pass": bool(sb >= 246 and bci[0] > 0 and gci[0] > 0)}
            s2sum = int(q4((s - cf)[sg]).sum())
            s2 = {"sum4d": s2sum, "pass": s2sum >= 0}
            all_stats[name] = {"SC3": sc3, "SC4": sc4, "S1": s1, "S2": s2, "S3": sc4,
                               "bar_comparator": bar_name, "pass": bool(sc3["pass"] and sc4["pass"]), "dropped": False}

    def ordered(groups, passing=False):
        rows = []
        for i, g in enumerate(groups):
            for j, k in enumerate(KINDS):
                nm = f"{k}_{g}"
                if nm in all_stats and (not passing or all_stats[nm]["pass"]):
                    rec = all_stats[nm]
                    rows.append((-rec["SC3"]["sum4d"], -rec["S1"]["gain_sum4d"], i, j, nm))
        return [r[-1] for r in sorted(rows)]

    nb = ordered(["NB100", "NB67", "NB33"])
    triggered = bool(nb) and not any(all_stats[nm]["pass"] for nm in nb)
    # rule gap: 6.3 does not define the no-evaluated-NB record; ticket 11
    # explicitly fixes this reason and null strength (matching ticket 10).
    reason = "no NB scorer was evaluated" if not nb else ("no NB scorer passed" if triggered else "an NB scorer passed")
    trigger = {"triggered": triggered, "strength": {"NB100": 1., "NB67": .67, "NB33": .33}[nb[0].split("_", 1)[1]] if triggered else None,
               "reason": reason, "NB": {nm: all_stats[nm] for nm in sorted(nb)}}
    out = {"trigger": trigger}
    for track, groups in (("main", MAIN_GROUPS), ("screen", SCREEN_GROUPS)):
        passes = ordered(groups, passing=True)
        pick = passes[0] if passes else None
        flag = None
        if pick:
            kind, g = pick.split("_", 1)
            ctl = f"{kind}_Z{g}"
            flag = bool(all_stats[ctl]["pass"]) if ctl in all_stats else "not evaluable"
        cands = {nm: rec for nm, rec in all_stats.items() if nm.split("_", 1)[1] in groups}
        out[track] = {"candidates": cands, "pick": pick, "random_control_flag": flag,
                      "dropped": {g: reason for g, reason in dropped.items() if g in groups}}
        if track == "main":
            out[track]["promising"] = not pick and any(rec["S1"]["pass"] and rec["S2"]["pass"] and rec["S3"]["pass"] for rec in cands.values())
            ranks = ordered(groups)
            out[track]["H_target"] = pick or (ranks[0] if ranks else None)
    out["references"] = {nm: rec for nm, rec in all_stats.items() if nm.split("_", 1)[1] in ("R0own", "NRK", "TP")}
    out["controls"] = {nm: rec for nm, rec in all_stats.items() if nm.split("_", 1)[1].startswith("Z")}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
