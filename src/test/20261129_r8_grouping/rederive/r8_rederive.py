"""Separate C13 implementation of M11--M15 and decision-rule sections 6--7.

Only the frozen R7 phase-A library pieces are used. No r8 scorer, grouping,
placement, head or gate implementation is read or imported. See r8_rederive.md
for the explicit stored-file invocation and pending-record contract.
"""
import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r8_common as R

MAIN_GROUPS = ["V100", "V67", "V33", "NB100", "NB67", "NB33"] + [f"FC{i}" for i in range(1, 7)] + ["NBC"]
SCREEN_GROUPS = ["AL", "DI", "SG"]
REFERENCES = ("R0own", "NRK", "TP")
NB_GROUPS = ("NB100", "NB67", "NB33")
STRENGTH = {"NB100": 1.0, "NB67": 0.67, "NB33": 0.33}
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
BAR_ORDER = ("B1", "BX", "CF", "B0", "B")
STAGE_DIM = {"AL": 256, "DI": 1024, "SG": 1152}
R7_RESULTS = R.MAIN / "src/test/20261126_design_l_style_source/results"
CONTROL_INDEX = {g: i + 1 for i, g in enumerate(MAIN_GROUPS + SCREEN_GROUPS)}
KINDS = ("R1", "XAFF")
METRICS = ("r1", "gain", "other", "strict", "swap")
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
A0 = ("affect", "image", "caption")


def placements(phi, labels, halves, selection_phi):
    """M11/M12 in painting order. Centres follow the index convention: float64 mean of the float32 phi, float64 unit
    vector, float64 dot, argmax ties to the lowest group."""
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
            mu = members.astype(np.float64).mean(axis=0)
            norm = np.linalg.norm(mu)
            assert norm > 0 and np.isfinite(norm), "zero centre"
            mus.append(mu / norm)
        return np.asarray(mus, np.float64)

    def assign(x, mu):
        return np.eye(k, dtype=np.float32)[np.argmax(x.astype(np.float64) @ mu.T, axis=1)]

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
    p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
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
                self.known[str(p.resolve())] = digest
        self.touch(R.RULE)

    def touch(self, path, expected=None):
        p = Path(path).resolve()
        key = str(p)
        if key not in self.hashes:
            self.hashes[key] = sha(p)
            if key in self.known:
                assert self.hashes[key] == self.known[key], f"rule section 10 SHA-256 differs: {p}"
        if expected is not None and self.hashes[key] != expected:
            self.disagreements.append({"where": key, "reason": "input SHA-256 differs", "expected": expected, "actual": self.hashes[key]})
        return p

    def npz(self, path, expected=None):
        with np.load(self.touch(path, expected), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def json(self, path, expected=None):
        return json.loads(self.touch(path, expected).read_text())

    def declared(self, declared, path):
        """Touch a file the pending record must list; its SHA-256 must be the listed one."""
        key = str(Path(path).resolve())
        if key not in declared:
            self.disagreements.append({"where": key, "reason": "input is not listed in the pending record's input_sha256"})
        return self.touch(path, declared.get(key))

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


def merge_groupings(files, paths):
    """All grouping JSONs first (as ticket 10 does): the records, the drops, and which files carry groupings.
    A drop overrides a grouping; a drops-only file has no NPZ."""
    records, dropped, with_groupings = {}, {}, []
    for path in paths:
        rec = files.json(path)
        assert isinstance(rec.get("groupings"), dict) and isinstance(rec.get("dropped"), dict), f"grouping file shape {path}"
        assert all(isinstance(v, str) and v.strip() for v in rec["dropped"].values()), "drop reasons"
        for target, source in ((records, rec["groupings"]), (dropped, rec["dropped"])):
            for g, value in source.items():
                assert g not in target or target[g] == value, f"conflicting grouping record {g}"
                target[g] = value
        if rec["groupings"]:
            with_groupings.append((Path(path), list(rec["groupings"])))
    return records, dropped, with_groupings


def read_labels(files, c, with_groupings, dropped):
    """Painting labels of every grouping that is not dropped. An NPZ is read only for a file that has groupings."""
    labels = {}
    first = np.unique(c.g_st, return_index=True)[1]
    for path, ids in with_groupings:
        z = files.npz(path.with_suffix(".npz"))
        assert np.array_equal(z["scorer_train"], c.scorer_train) and np.array_equal(z["g_st"], c.g_st), "grouping alignment"
        for g in ids:
            parent = g[1:] if g.startswith("Z") and g[1:] in CONTROL_INDEX else None
            if g in dropped or (parent in dropped and g not in dropped):
                continue    # a drop overrides a grouping; Z<X> is skipped when X is dropped (Z<X> not recorded dropped)
            assert g in z and z[g].shape == c.scorer_train.shape and z[g].dtype == np.int64, f"labels {g}"
            lab = z[g][first]
            assert np.array_equal(lab[c.g_st], z[g]), "labels constant within painting"
            assert len(np.unique(lab)) >= 2 and np.array_equal(np.unique(lab), np.arange(lab.max() + 1)), "grouping classes"
            if g in labels:
                files.compare(f"duplicate grouping {g}", labels[g], lab)
            labels[g] = lab
    return labels


def derive_files(root, *, fixture=None, smoke=False, groupings=None, files=None, aff_reader=None, sources=None,
                 keep_counts=False):
    """Read stored φ, labels, halves, posteriors, reader probabilities and taus.
    `sources` maps each scored grouping to its tau file and reader file (from the seed-42 index entries)."""
    global _ACTIVE_LEDGER
    files, root = files or Files(), Path(root).resolve()
    previous, _ACTIVE_LEDGER = _ACTIVE_LEDGER, files
    try:
        files.touch(R.RULE)
        c, ep, cos, tu, e2, arrays = inputs(files, fixture, smoke)
        grouping_paths = groupings or [root / "groupings/r8_groupings.json"]
        records, dropped, with_groupings = merge_groupings(files, grouping_paths)
        labels = read_labels(files, c, with_groupings, dropped)
        assert not set(labels) & set(dropped), "evaluated and dropped grouping"
        assert set(labels) <= set(MAIN_GROUPS + SCREEN_GROUPS + list(REFERENCES) + ["Z" + g for g in CONTROL_INDEX]), "grouping ID"
        if not smoke:
            accounted = set(records) | set(dropped)
            assert set(MAIN_GROUPS[:-1] + SCREEN_GROUPS + list(REFERENCES)) <= accounted, "unrecorded candidate/reference drop"
            assert {"Z" + g for g in labels if g in CONTROL_INDEX} <= accounted, "unrecorded control drop"
        # A0 heads and the legacy R0 head are R7's frozen files (fixture: beside the stored files).
        a0_dir = root / "heads" if fixture else R7_RESULTS / "heads"
        posts = {g: read_post(files, a0_dir / f"post_{g}.npz", c) for g in A0}
        posts["legacy_R0"] = read_post(files, a0_dir / "post_R0.npz", c)
        B = condition_free(cos, tu, e2, ep, A0, arrays["parity"])
        for name, score in (("B", B), ("B0", condition_free(cos, tu, posts, ep, A0, arrays["parity"])),
                            ("B1", condition_free(cos, tu, posts, ep, (*A0, "legacy_R0"), arrays["parity"]))):
            arrays.update({f"{name}__{m}": v for m, v in metrics(counts(score)).items()})
        # AFF's A0 reader is R7's frozen file (fixture: beside the stored files).
        az = files.npz(aff_reader or (root / "seed42/reader_A0.npz" if fixture else R7_RESULTS / "seed42/reader_A0.npz"))
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
            else:
                # Every other grouping, TP included (its space file is the one-hot of its final map), goes through M11/M12.
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
            # R7's format: no `rows` key; the rows are the scorer-train rows, in order.
            assert {"img", "txt", "classes_half0", "classes_half1", "filled_by_half"} <= set(bank), f"xheads keys {g}"
            assert len(bank["img"]) == len(c.scorer_train), f"xheads row count {g}"
            assert bank["img"].shape == bank["txt"].shape == (len(c.scorer_train), k)
            assert bank["img"].dtype == bank["txt"].dtype == np.float32 and np.isfinite(bank["txt"]).all()
            assert np.allclose(bank["txt"].sum(1), 1, atol=1e-6, rtol=0)
            files.compare(f"M12 xheads_{g} img", placement["bank"][c.g_st], bank["img"])
            files.compare(f"M12 filled_by_half {g}", 1 - c.halves[c.g_st], bank["filled_by_half"])
            for h in (0, 1):
                files.compare(f"M12 classes_half{h} {g}", np.arange(k), bank[f"classes_half{h}"])
            parts = (*A0, g)
            assert sources and g in sources, f"no seed-42 index entry lists {g}"
            src = sources[g]
            tau_table = files.json(src["taus"], src["taus_sha256"])["taus"]
            reader = files.npz(src["reader"], src["reader_sha256"])
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
        return {"arrays": arrays, "evaluated": set(labels), "records": records, "dropped": dropped,
                "image_index": c.image_index, "input_sha256": files.hashes, "disagreements": files.disagreements}
    finally:
        _ACTIVE_LEDGER = previous


def add_family(arrays, name, values, detail, keep_counts=False):
    for key, value in values.items():
        arrays[(name + "_cf__" + key[4:]) if key.startswith("cf__") else (name + "__" + key)] = value
    # Pipeline batches may store any of these 224-cell integer checks too.
    for key, value in detail.items():
        if key not in ("f_cnt", "c_cnt") or keep_counts:
            arrays[name + "__" + key] = value


def compare_json(files, path, got, want, tol=0.0):
    """Exact discrete fields; only intervals (keys ending ci95) have the C13 1e-9 tolerance.
    `got` is what RD derived, `want` what the pipeline stored. A key on one side only is a disagreement."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            where = f"{path}.{key}"
            if key not in want:
                files.disagreements.append({"where": where, "reason": "missing record key"})
            elif key not in got:
                files.disagreements.append({"where": where, "reason": "extra record key"})
            else:
                compare_json(files, where, got[key], want[key], 1e-9 if key.endswith("ci95") else tol)
    elif isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            files.disagreements.append({"where": path, "reason": "list length differs", "derived": len(got), "stored": len(want)})
        else:
            for i, (g, w) in enumerate(zip(got, want)):
                compare_json(files, f"{path}[{i}]", g, w, tol)
    elif isinstance(got, bool) or isinstance(want, bool):
        if type(got) is not type(want) or got != want:
            files.disagreements.append({"where": path, "reason": "boolean differs", "derived": got, "stored": want})
    elif got is None or want is None or isinstance(got, (str, dict, list)) or isinstance(want, (str, dict, list)):
        if type(got) is not type(want) or got != want:
            files.disagreements.append({"where": path, "reason": "values differ", "derived": got, "stored": want})
    else:
        same = abs(got - want) <= tol if tol else got == want
        if not same:
            files.disagreements.append({"where": path, "reason": "values differ", "derived": got, "stored": want})


def stage_checks(files, root, declared, expected_index, smoke):
    """Rule 7(d): the recomputable ESC1 b to d and SC2 checks, from the evidence the stage records list
    (stageE_<X>.json: evidence.embeddings, evidence.image_index; stage1_fcmi.json: loss_log, labels_before_merge).
    Returns (checks, {stage record name: SHA-256})."""
    out, stage_sha = {}, {}

    def record(name):
        path = files.declared(declared, root / name)
        rec = files.json(path)
        stage_sha[name] = files.hashes[str(Path(path).resolve())]
        assert isinstance(rec.get("pass"), bool) and rec.get("smoke") is smoke, f"{name}: pass/smoke"
        return rec

    rec0 = record("stage0_regression.json")
    assert rec0["pass"] is True or (smoke and rec0.get("wiring_pass") is True), "SC1 did not pass"
    for name in SCREEN_GROUPS:
        rec = record(f"stageE_{name}.json")
        evidence = rec.get("evidence") or {}
        if not {"embeddings", "image_index"} <= set(evidence):
            assert rec["pass"] is False, f"stageE_{name}: no evidence files but pass is true"
            out[name] = {"status": "not produced", "reason": rec.get("reason")}
            continue
        emb = np.load(files.touch(evidence["embeddings"]["path"], evidence["embeddings"]["sha256"]), mmap_mode="r", allow_pickle=False)
        ix = np.load(files.touch(evidence["image_index"]["path"], evidence["image_index"]["sha256"]), allow_pickle=False)
        n = int(rec["n"]) if smoke else len(expected_index)
        assert n <= len(expected_index)
        want_index = np.ascontiguousarray(expected_index[:n], dtype=np.int64)
        flags = {"b_shape": bool(emb.shape == (n, STAGE_DIM[name])), "c_finite": bool(np.isfinite(emb).all()),
                 "d_index_hash": bool(ix.dtype == np.int64 and ix.shape == want_index.shape and
                                      hashlib.sha256(np.ascontiguousarray(ix).tobytes()).hexdigest()
                                      == hashlib.sha256(want_index.tobytes()).hexdigest())}
        for check, value in flags.items():
            compare_json(files, f"ESC1 {name}.{check}", value, rec["esc1"][check]["ok"])
        out[name] = flags
    rec = record("stage1_fcmi.json")
    log = Path(rec["loss_log"])
    if not log.is_file():
        assert rec["pass"] is False, "stage1_fcmi: no loss log but pass is true"
        out["FCMI"] = {"status": "not produced", "reason": rec.get("error")}
        return out, stage_sha
    entries = []
    for line in files.touch(log, rec["loss_log_sha256"]).read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            break   # an abrupt death can leave an incomplete last line
    # "finite" is recomputed from the loss values, never read from the stored flag.
    finite = all(e["total_loss"] is not None and np.isfinite(e["total_loss"]) for e in entries)
    if rec["labels_before_merge"] is None:
        assert rec["labels_sha256"] is None
        clusters = 0
    else:
        z = files.npz(rec["labels_before_merge"], rec["labels_sha256"])
        lab = z["labels"]
        assert lab.ndim == 1 and lab.dtype.kind in "iu"
        if not smoke:
            assert len(lab) == 36518
        clusters = int(len(np.unique(lab)))
    compare_json(files, "SC2.finite_loss", finite, rec["tests"]["finite_loss"])
    compare_json(files, "SC2.clusters", clusters >= 2, rec["tests"]["clusters"])
    compare_json(files, "SC2.clusters_before_merge", clusters, rec["clusters_before_merge"])
    compare_json(files, "SC2.steps", len(entries), rec["steps"])
    out["FCMI"] = {"finite_loss": finite, "clusters": clusters >= 2, "clusters_before_merge": clusters}
    return out, stage_sha


def read_indexes(files, declared, index_paths):
    """The seed-42 indexes: every batch's arrays (hash checked, duplicates must agree) and, per scored grouping, the
    tau file and reader file its index entry names (09's layout: taus_file, readers_sha256, groupings)."""
    stored, sources = {}, {}
    for index_path in index_paths:
        index_path = Path(index_path).resolve()
        index = files.json(files.declared(declared, index_path))
        assert isinstance(index["batches"], list) and index["batches"], "seed42 index batches"
        base = index_path.parent
        for entry in index["batches"]:
            assert {"file", "sha256", "groupings", "taus_file", "taus_sha256", "readers_sha256"} <= set(entry), "index entry keys"
            path = (base / entry["file"]).resolve()
            assert path.parent == base, "batch file outside the index folder"
            files.touch(files.declared(declared, path), entry["sha256"])
            arrays = files.npz(path)
            for k in ("cl", "pair_index", "parity", "AFF__r1", "B__r1", "B0__r1", "B1__r1"):
                assert k in arrays, f"batch lacks {k}"
            assert all(arrays[k].dtype == np.int64 and arrays[k].ndim == 1 for k in ("cl", "pair_index", "parity")), "batch ID dtypes/shapes"
            assert all(arrays[k].shape == arrays["cl"].shape for k in ("pair_index", "parity")), "batch ID row counts"
            for x in entry["groupings"]:
                assert x not in sources, f"{x} is listed by two index entries"
                assert x in entry["readers_sha256"], f"index entry lacks the reader SHA-256 of {x}"
                sources[x] = {"taus": base / entry["taus_file"], "taus_sha256": entry["taus_sha256"],
                              "reader": base / f"reader_{x}.npz", "reader_sha256": entry["readers_sha256"][x]}
            for k, a in arrays.items():
                if k in stored:
                    files.compare(f"duplicate batch key {k}", stored[k], a)
                else:
                    stored[k] = a
    return stored, sources


PENDING_COMPARED = ("main", "screen", "trigger", "groupings", "dropped", "fallback")
PENDING_METADATA = ("step", "smoke", "rule_sha256", "written", "stage_sha256", "probe_trigger_sha256", "input_sha256",
                    "report_flags")
FALLBACK_CONSTANT_STRINGS = ("main_failure_after_held", "held_start_cutoff")


def compare_pending(files, derived, rec):
    """Ticket 10's pending layout: stats at the top level (main, screen, trigger, fallback) plus groupings and dropped.
    Skipped: report_flags, the two constant fallback strings (only checked to be strings) and written."""
    for key in PENDING_COMPARED:
        if key not in rec:
            files.disagreements.append({"where": f"pending.{key}", "reason": "missing pending key"})
    for key in sorted(set(rec) - set(PENDING_COMPARED) - set(PENDING_METADATA)):
        files.disagreements.append({"where": f"pending.{key}", "reason": "extra pending key"})
    for key in PENDING_COMPARED:
        if key not in rec:
            continue
        stored = rec[key]
        if key == "fallback":
            assert isinstance(stored, dict) and all(isinstance(stored.get(k), str) for k in FALLBACK_CONSTANT_STRINGS), "fallback strings"
            stored = {k: v for k, v in stored.items() if k not in FALLBACK_CONSTANT_STRINGS}
        compare_json(files, f"pending.{key}", derived[key], stored)


def check_files(root, pending, step, *, indexes=None, fixture=None, smoke=False, groupings=None, aff_reader=None):
    """Public stored-file seam. Disagreements prevent the CLI's successful exit."""
    assert step in ("trigger", "gate")
    files, root = Files(), Path(root).resolve()
    pending = Path(pending).resolve()
    rec = files.json(pending)
    assert isinstance(rec, dict)
    assert rec.get("step") == step and rec.get("smoke") is smoke, "pending step/smoke role"
    assert rec.get("rule_sha256") == files.hashes[str(R.RULE.resolve())], "pending rule SHA-256"
    declared = rec["input_sha256"]
    assert isinstance(declared, dict) and declared, "pending input_sha256"
    for path, digest in declared.items():     # every declared input is read and hashed, used or not
        assert Path(path).is_absolute(), "pending input paths are absolute"
        files.touch(path, digest)
    stored, sources = read_indexes(files, declared, indexes or [root / "seed42/index.json"])
    grouping_paths = [Path(p).resolve() for p in (groupings or [root / "groupings/r8_groupings.json"])]
    for path in grouping_paths:
        files.declared(declared, path)
    derived = derive_files(root, fixture=fixture, smoke=smoke, groupings=grouping_paths, files=files, aff_reader=aff_reader,
                           sources=sources, keep_counts=any(k.endswith(("__f_cnt", "__c_cnt")) for k in stored))
    full = derived["arrays"]
    stats = statistics(full, derived["evaluated"], derived["dropped"], nbc_allowed=step == "gate", strict=not smoke)
    optional = {name + "__" + detail for name in ["AFF"] + [f"{kind}_{g}" for g in derived["evaluated"] for kind in KINDS]
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
    payload = {**stats, "groupings": derived["records"], "dropped": derived["dropped"]}
    compare_pending(files, payload, rec)
    stages, stage_sha = stage_checks(files, root, declared, derived["image_index"], smoke)
    compare_json(files, "pending.stage_sha256", stage_sha, rec["stage_sha256"])
    probe = root / "probe_trigger.json"
    if step == "gate":
        probe_rec = files.json(files.declared(declared, probe))
        compare_json(files, "pending.probe_trigger_sha256", files.hashes[str(probe.resolve())], rec["probe_trigger_sha256"])
        compare_json(files, "probe_trigger.json", stats["trigger"], {k: probe_rec.get(k, "<absent>") for k in stats["trigger"]})
    else:
        compare_json(files, "pending.probe_trigger_sha256", None, rec["probe_trigger_sha256"])
    files.finish()
    return {"status": "agree" if not files.disagreements else "disagree", "step": step, "smoke": bool(smoke),
            "disagreements": files.disagreements, "input_sha256": files.hashes, "rederived": stats, "stage_checks": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("trigger", "gate"), required=True)
    parser.add_argument("--root", type=Path, help="results root (default results/, or results/smoke/gate with --smoke)")
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--index", type=Path, nargs="+", help="one or more immutable seed42 indices (as ticket 10)")
    parser.add_argument("--groupings", type=Path, nargs="+", help="one or more grouping/drop files (as ticket 10)")
    parser.add_argument("--aff-reader", type=Path, help="AFF's A0 reader (default: R7's results/seed42/reader_A0.npz)")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root or (R.SMOKE / "gate" if args.smoke else R.RESULTS)
    assert root.resolve().is_relative_to(R.SMOKE.resolve()) if args.smoke else not root.resolve().is_relative_to(R.SMOKE.resolve()), "output smoke role (set R8_RESULTS for worktree tests)"
    record = check_files(root, args.pending, args.step, indexes=args.index, fixture=args.fixture, smoke=args.smoke,
                         groupings=args.groupings, aff_reader=args.aff_reader)
    path = root / f"rederive/{args.step}_rederive.json"
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


def need(a, n):
    """L16's integer bar ceil(a * n / 25), exact (a is its decimal text): -0.3 -> -49, 0.5 -> 246, 0 -> 0 at n = 12,288."""
    f = Fraction(a) * n / 25
    return -((-f.numerator) // f.denominator)


def interval(a, cl):
    """Own C9 call (point, [lo, hi] in R@1 points); never consumes the pipeline's interval or resamples."""
    r = library("src.eval.aspect_metrics").cluster_bootstrap(a, cl, n_boot=5000, seed=42, chunk=250)
    lo, hi = r["ci95"]
    return 100 * float(r["point"]), [100 * float(lo), 100 * float(hi)]


def dropped_record(reason):
    return {"dropped": True, "reason": reason, "pass": False, "r7_pass": False, "SC3": None, "SC4": None, "S1": None,
            "S2": None, "S3": None, "gain_sum4d": None, "bar_comparator": None}


def scorer_record(a, name, cl, masks):
    """Rule 6.1 for one scorer: SC3, SC4 and the recorded R7 S1 to S3, in ticket 10's record shape."""
    kind, x = name.split("_", 1)
    n, n_pair = len(cl), len(cl) // 3
    for key in (f"{name}__r1", f"{name}__gain", f"{name}_cf__r1", f"{name}_cf__gain", "AFF__r1", "B1__r1", "B0__r1",
                "B__r1", f"BX_{x}__r1"):
        assert key in a, f"missing array {key}"
        assert a[key].shape == (n,) and np.isfinite(a[key]).all(), f"{key}: shape or nonfinite values"
        q4(a[key])
        assert ((a[key] >= 0) & (a[key] <= 1)).all() if key.endswith("__r1") else ((a[key] >= -1) & (a[key] <= 1)).all(), key
    s, gain = a[f"{name}__r1"], a[f"{name}__gain"]
    cf, cf_gain = a[f"{name}_cf__r1"], a[f"{name}_cf__gain"]
    assert not cf_gain.any(), f"{name}: the counterpart's condition gain must be 0 everywhere"
    aff = a["AFF__r1"]
    comps = [("B1", a["B1__r1"]), ("BX", a[f"BX_{x}__r1"]), ("CF", cf), ("B0", a["B0__r1"]), ("B", a["B__r1"])]
    bar_name, bar = max(comps, key=lambda kv: int(q4(kv[1]).sum()))    # ties to the earliest in BAR_ORDER
    d_aff, d_bar, d_gain = s - aff, s - bar, gain - cf_gain
    sd, sb, sg = int(q4(d_aff).sum()), int(q4(d_bar).sum()), int(q4(d_gain).sum())
    point, ci = interval(d_aff, cl)
    assert abs(point - 25 * sd / n) < 1e-9, "bootstrap point differs from the integer point"
    sc3 = {"sum4d": sd, "n": n, "point": 25 * sd / n, "ci95": ci, "pass": bool(ci[0] > 0)}
    s3, sc4 = {}, {}
    for p in (0, 1):
        scoped = d_aff[masks[p]]
        total = int(q4(scoped).sum())
        ok = bool(total >= need("-0.3", n_pair))
        s3[PAIRS[p]] = {"sum4d": total, "pass": ok}
        sc4[PAIRS[p]] = {"sum4d": total, "pass": ok, "n": n_pair, "point": 25 * total / n_pair,
                         "ci95": interval(scoped, cl[masks[p]])[1]}
    s3["pass"] = sc4["pass"] = bool(s3[PAIRS[0]]["pass"] and s3[PAIRS[1]]["pass"])
    bar_ci, gain_ci = interval(d_bar, cl)[1], interval(d_gain, cl)[1]
    s1 = {"sum4d": sb, "ci95": bar_ci, "gain_sum4d": sg, "gain_ci95": gain_ci, "n": n, "point": 25 * sb / n,
          "gain_point": 25 * sg / n, "pass": bool(sb >= need("0.5", n) and bar_ci[0] > 0 and gain_ci[0] > 0)}
    scope = masks[2]    # S2 is the style x genre pair (pair_index 2) only
    s2_sum = int(q4((s - cf)[scope]).sum())
    s2 = {"sum4d": s2_sum, "pass": bool(s2_sum >= need("0", n_pair)), "n": n_pair, "point": 25 * s2_sum / n_pair,
          "ci95": interval((s - cf)[scope], cl[scope])[1]}
    r7_pass = bool(s1["pass"] and s2["pass"] and s3["pass"])
    return {"bar_comparator": bar_name, "S1": s1, "S2": s2, "S3": s3, "SC3": sc3, "SC4": sc4, "dropped": False,
            "gain_sum4d": sg, "r7_pass": r7_pass, "pass": bool(sc3["pass"] and sc4["pass"])}


def select(records, order, passing_only):
    """Rule 6.4: largest SC3 sum, then largest gain sum, then the rule's list order, then R1 before X-AFF."""
    keys = [k for k, r in records.items() if not r["dropped"] and (r["pass"] or not passing_only)]
    if not keys:
        return None
    return min(keys, key=lambda k: (-records[k]["SC3"]["sum4d"], -records[k]["gain_sum4d"],
                                    order.index(k.split("_", 1)[1]), KINDS.index(k.split("_", 1)[0])))


def statistics(a, evaluated, dropped, *, nbc_allowed, strict=True):
    """Rules 6.1 to 6.5 (and the deterministic H target of 6.6) in ticket 10's pending layout.
    `evaluated`: groupings with derived arrays; `dropped`: {grouping: reason}. NBC is part of the roster only when
    `nbc_allowed` (the gate step) and the trigger fires. `strict` (real runs): every roster scorer is evaluated or dropped."""
    cl, pair = a["cl"], a["pair_index"]
    n = len(cl)
    assert n > 0 and n % 3 == 0 and pair.shape == (n,)
    masks = [pair == p for p in range(3)]
    assert all(int(m.sum()) == n // 3 for m in masks), "three equal pairs"
    assert len(np.unique(cl)) >= 2
    cache = {}

    def of(x):
        """Both kinds' records of grouping x (none when x has neither arrays nor a recorded drop, smoke only)."""
        if x not in cache:
            cache[x] = {}
            for kind in KINDS:
                if x in dropped:
                    cache[x][f"{kind}_{x}"] = dropped_record(dropped[x])
                elif x in evaluated:
                    cache[x][f"{kind}_{x}"] = scorer_record(a, f"{kind}_{x}", cl, masks)
                else:
                    assert not strict, f"{x}: neither arrays nor a recorded drop"
        return cache[x]

    nb = {k: v for x in NB_GROUPS for k, v in of(x).items()}
    assert not strict or len(nb) == 6, "all six NB scorers must be evaluated or dropped"
    available = {k: v for k, v in nb.items() if not v["dropped"]}
    source, reason = None, "an NB scorer passes"
    if not any(v["pass"] for v in nb.values()):
        if available:
            source, reason = select(available, MAIN_GROUPS, False), "none of the six NB scorers passes"
        else:
            reason = "no NB scorer was evaluated; NBC does not run"    # rule gap: as ticket 10 (6.3 literal would trigger)
    trigger = {"nb_records": nb, "triggered": source is not None, "source_scorer": source,
               "strength": STRENGTH[source.split("_", 1)[1]] if source else None, "reason": reason}
    include_nbc = nbc_allowed and trigger["triggered"]
    main = MAIN_GROUPS if include_nbc else MAIN_GROUPS[:-1]
    controls = [f"Z{x}" for x in (*main, *SCREEN_GROUPS) if x not in dropped or f"Z{x}" in dropped]
    roster = {*main, *SCREEN_GROUPS, *REFERENCES, *controls}
    assert not set(evaluated) - roster, f"evaluated outside the roster: {sorted(set(evaluated) - roster)}"
    if include_nbc:
        assert "NBC" in evaluated or "NBC" in dropped, "triggered NBC has neither a result nor a recorded drop"
    out = {}
    for track, xs in (("main", main), ("screen", SCREEN_GROUPS)):
        cands = {k: v for x in xs for k, v in of(x).items()}
        refs = {k: v for x in (REFERENCES if track == "main" else ("R0own",)) for k, v in of(x).items()}
        ctls = {k: v for x in xs if f"Z{x}" in controls for k, v in of(f"Z{x}").items()}
        pick = select(cands, MAIN_GROUPS if track == "main" else SCREEN_GROUPS, True)
        flag = None
        if pick:
            kind, x = pick.split("_", 1)
            assert f"{kind}_Z{x}" in ctls, f"missing random control for {pick}"
            flag = "not evaluable" if ctls[f"{kind}_Z{x}"]["dropped"] else bool(ctls[f"{kind}_Z{x}"]["pass"])
        out[track] = {"candidates": cands, "references": refs, "controls": ctls, "pick": pick, "random_control_flag": flag}
        if track == "main":
            promising = pick is None and any(not r["dropped"] and all(r[s]["pass"] for s in ("S1", "S2", "S3")) for r in cands.values())
            target = pick or select(cands, MAIN_GROUPS, False)
            head = None    # rule gap: no H target when every main candidate is dropped
            if target:
                kind, x = target.split("_", 1)
                head = {"grouping": x, "scorer_kind": kind, "scorer": f"{kind}_H{x}", "source_scorer": target}
            out[track].update(promising=bool(promising), head_target=head, main_line_fails_at_gate=pick is None)
    out["trigger"] = trigger
    out["fallback"] = {"main_has_pick": out["main"]["pick"] is not None,
                       "main_line_fails_at_gate": out["main"]["pick"] is None,
                       "screen_has_pick": out["screen"]["pick"] is not None,
                       "screen_read_if_main_fails": out["screen"]["pick"] is not None,
                       "screen_eligible_at_gate": out["main"]["pick"] is None and out["screen"]["pick"] is not None,
                       "held_rule_required": True}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
