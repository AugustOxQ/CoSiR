"""Backbone-selection check (throwaway, diagnostic). Never reads val/held rows.

  run_backbone_check.py measure_clip_repro        # CLIP cache reproduction check only (no GPU)
  run_backbone_check.py extract <model> [artelingo|cub ...]   # GPU; features -> /data/SSD2/pre_extract/backbone_check
  run_backbone_check.py measure <model> ...       # CPU; writes results/backbone_check.json
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path("/project/CoSiR"); sys.path.insert(0, str(ROOT))
HERE = ROOT / "src/test/20261025_backbone_check"
OUT = Path("/data/SSD2/pre_extract/backbone_check")
RES = HERE / "results/backbone_check.json"
TIM = HERE / "results/timings.json"
WIKIART = Path("/data/PDD/wikiart_proj/wikiart")
CUB = Path("/data/SSD/cub")


def nrm(x): x = np.asarray(x, np.float32); return x / np.linalg.norm(x, axis=1, keepdims=True)
r1 = lambda S, col: float(np.mean((np.delete(S, col, axis=1) < S[:, [col]]).all(axis=1)) * 100)


def load_json(p): return json.loads(p.read_text()) if p.exists() else {}


def dump(p, d): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(d, indent=1))


# ---------------------------------------------------------------- ArtELingo rows
_AR = {}
def ar_state():
    if _AR: return _AR
    spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
    ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
    from src.data.artelingo import load_artelingo, join_captions, ANNOTATIONS_PATH
    data = load_artelingo(); cache, *_ = ra.grid.load_grid()
    st, sl = cache["scorer_train"], cache["selection"]
    rng = np.random.default_rng(0); tr = rng.choice(st, 60000, replace=False)
    ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
    rows = np.union1d(tr, sl)  # only scorer-train sample + selection; val/held never touched
    _AR.update(data=data, st=st, sl=sl, tr=tr, rows=rows, ann=ann, join_captions=join_captions)
    return _AR


def ar_inputs():
    s = ar_state(); d, rows = s["data"], s["rows"]
    pid = np.asarray(d.sample_ids)[rows]
    caps = s["join_captions"](d.sample_ids[rows], s["ann"])
    imgpath = [str(WIKIART / s["ann"][int(i)]["image"]) for i in pid]
    paint = np.asarray(d.paintings)[rows]
    up, first, inv = np.unique(paint, return_index=True, return_inverse=True)
    return dict(rows=rows, caps=list(caps), paths=[imgpath[i] for i in first], inv=inv)


# ---------------------------------------------------------------- CUB
def cub_data():
    img_rel = [l.split()[1] for l in (CUB / "CUB_200_2011/images.txt").read_text().splitlines()]
    caps = []
    for r in img_rel:
        cls, f = r.split("/"); p = CUB / "captions/extracted/text_c10" / cls / (Path(f).stem + ".txt")
        lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
        assert len(lines) >= 10, (p, len(lines)); caps.append(lines[:10])
    return [str(CUB / "CUB_200_2011/images" / r) for r in img_rel], caps


def extract(model, sets):
    import torch, extract as ex
    t0 = time.time(); m = ex.build(model); tim = load_json(TIM); tl = time.time() - t0
    tim.setdefault(model, {})["load_s"] = tl
    for ds in sets:
        out = OUT / ds / model; out.mkdir(parents=True, exist_ok=True)
        if ds == "artelingo":
            if model == "clip": print("clip artelingo comes from the cache"); continue
            a = ar_inputs(); np.save(out / "rows.npy", a["rows"]); np.save(out / "row_to_painting.npy", a["inv"])
            t = time.time(); I = m.images(a["paths"]); ti = time.time() - t
            t = time.time(); T = m.texts(a["caps"]); tt = time.time() - t
            np.save(out / "img.npy", I.astype(np.float16)); np.save(out / "txt.npy", T.astype(np.float16))
            tim[model][ds] = dict(n_img=len(I), n_txt=len(T), img_s=ti, txt_s=tt)
        else:
            paths, caps = cub_data(); (out / "index.json").write_text(json.dumps({"image_paths": paths}))
            t = time.time(); I = m.images(paths); ti = time.time() - t
            t = time.time(); T = m.texts([c for cs in caps for c in cs]).reshape(len(paths), 10, -1); tt = time.time() - t
            np.save(out / "img.npy", I.astype(np.float16)); np.save(out / "txt.npy", T.astype(np.float16))
            tim[model][ds] = dict(n_img=len(I), n_txt=int(T.shape[0] * 10), img_s=ti, txt_s=tt)
        print(model, ds, tim[model][ds], flush=True); dump(TIM, tim)
    del m; torch.cuda.empty_cache()


# ---------------------------------------------------------------- ArtELingo measures
def ar_features(model):
    """returns img_of(rows), txt_of(rows) (L2-normalised float32) for the rows in the pool"""
    s = ar_state()
    if model == "clip":
        d = s["data"]; return (lambda idx: nrm(d.img_features[idx])), (lambda idx: nrm(d.txt_features[idx]))
    out = OUT / "artelingo" / model
    rows, inv = np.load(out / "rows.npy"), np.load(out / "row_to_painting.npy")
    I, T = np.load(out / "img.npy"), np.load(out / "txt.npy")
    pos = np.full(len(s["data"].emotions), -1, np.int64); pos[rows] = np.arange(len(rows))
    def at(idx): p = pos[np.asarray(idx)]; assert (p >= 0).all(); return p
    return (lambda idx: nrm(I[inv[at(idx)]])), (lambda idx: nrm(T[at(idx)]))


def artelingo_measure(model):
    s = ar_state(); d = s["data"]; sl, tr = s["sl"], s["tr"]
    ep = np.load(HERE.parent / "20261023_aspect_episode_spike/results/aspect_episodes.npz")
    A, C = ep["anchor"], ep["cands"]; assert np.isin(np.unique(np.concatenate([A, C.ravel()])), sl).all()
    emo, sty = np.asarray(d.emotions), np.asarray(d.art_styles)
    imgf, txtf = ar_features(model)
    res = {}
    qi, qt = imgf(A), txtf(A); ci = imgf(C.ravel()).reshape(*C.shape, -1); ct = txtf(C.ravel()).reshape(*C.shape, -1)
    cos_i2t, cos_t2i = np.einsum("nd,nkd->nk", qi, ct), np.einsum("nd,nkd->nk", qt, ci)
    for name, col in (("emotion", 0), ("style", 1)):
        res[f"r1_{name}_i2t"], res[f"r1_{name}_t2i"] = r1(cos_i2t, col), r1(cos_t2i, col)
    res["r1_pooled"] = float(np.mean([res[f"r1_{n}_{k}"] for n in ("emotion", "style") for k in ("i2t", "t2i")]))
    Ftr = {"img": imgf(tr), "txt": txtf(tr)}; Fsl = {"img": imgf(sl), "txt": txtf(sl)}
    probes = {}
    for ln, lab in (("emotion", emo), ("style", sty)):
        for mod in ("img", "txt"):
            clf = LogisticRegression(C=1.0, max_iter=300).fit(Ftr[mod], lab[tr]); probes[(ln, mod)] = clf
            res[f"probe_{ln}_{mod}"] = 100 * clf.score(Fsl[mod], lab[sl]); print(model, ln, mod, res[f"probe_{ln}_{mod}"], flush=True)
    for name, col in (("emotion", 0), ("style", 1)):
        P = {m: probes[(name, m)].predict_proba((imgf if m == "img" else txtf)(C.ravel())).reshape(*C.shape, -1) for m in ("img", "txt")}
        Q = {m: probes[(name, m)].predict_proba((imgf if m == "img" else txtf)(A)) for m in ("img", "txt")}
        e = lambda a, b: np.einsum("nc,nkc->nk", a, b)
        res[f"ceil_{name}_cross_i2t"], res[f"ceil_{name}_cross_t2i"] = r1(e(Q["img"], P["txt"]), col), r1(e(Q["txt"], P["img"]), col)
        res[f"ceil_{name}_same_img"], res[f"ceil_{name}_same_txt"] = r1(e(Q["img"], P["img"]), col), r1(e(Q["txt"], P["txt"]), col)
        res[f"ceil_{name}_cross_mean"] = 0.5 * (res[f"ceil_{name}_cross_i2t"] + res[f"ceil_{name}_cross_t2i"])
    return res


EXPECT = {"r1_emotion_i2t": 9.42, "r1_emotion_t2i": 10.86, "r1_style_i2t": 10.77, "r1_style_t2i": 13.45,
          "probe_emotion_img": 35.2, "probe_emotion_txt": 56.9, "probe_style_img": 60.8, "probe_style_txt": 25.4,
          "ceil_emotion_cross_i2t": 23.58, "ceil_emotion_cross_t2i": 24.29, "ceil_emotion_same_img": 17.85, "ceil_emotion_same_txt": 36.62,
          "ceil_style_cross_i2t": 21.41, "ceil_style_cross_t2i": 23.07, "ceil_style_same_img": 49.80, "ceil_style_same_txt": 13.11}


def repro_check(res):
    bad = {k: (res[k], v) for k, v in EXPECT.items() if abs(res[k] - v) > (0.006 if k.startswith("r1_") else 0.3)}
    return bad


# ---------------------------------------------------------------- CUB measures
def cub_labels():
    names = {int(l.split()[0]): l.split()[1] for l in (CUB / "attributes.txt").read_text().splitlines()}
    import pandas as pd
    df = pd.read_csv(CUB / "CUB_200_2011/attributes/image_attribute_labels.txt", sep=r"\s+", header=None,
                     usecols=[0, 1, 2, 3], names=["img", "att", "pres", "cert"], engine="python", on_bad_lines="skip")
    df = df[(df.pres == 1) & (df.cert >= 3)]
    out = {}
    for g in ("has_primary_color", "has_bill_shape", "has_size"):
        ids = {a for a, n in names.items() if n.startswith(g + "::")}
        sub = df[df.att.isin(ids)]; cnt = sub.groupby("img").size(); one = cnt[cnt == 1].index
        lab = np.full(11788, -1); sub = sub[sub.img.isin(one)]
        order = sorted(ids); lab[sub.img.values - 1] = [order.index(a) for a in sub.att.values]
        out[g] = lab
    return out


def cub_measure(model):
    out = OUT / "cub" / model; I, T = nrm(np.load(out / "img.npy")), np.load(out / "txt.npy").astype(np.float32)
    split = np.array([int(l.split()[1]) for l in (CUB / "CUB_200_2011/train_test_split.txt").read_text().splitlines()])
    te, trn = np.where(split == 0)[0], np.where(split == 1)[0]; assert len(te) == 5794
    res = {}
    S = I[te] @ nrm(T[te, 0]).T          # image i vs first caption j
    res["ret_i2t_r1"] = 100 * float((S.argmax(1) == np.arange(len(te))).mean())
    res["ret_t2i_r1"] = 100 * float((S.argmax(0) == np.arange(len(te))).mean())
    Tm = nrm(nrm(T.reshape(-1, T.shape[-1])).reshape(T.shape).mean(1))   # mean of the 10 L2-normalised caption features
    for g, lab in cub_labels().items():
        a, b = trn[lab[trn] >= 0], te[lab[te] >= 0]
        r = dict(n_labelled=int((lab >= 0).sum()), n_train=len(a), n_test=len(b), n_classes=int(len(np.unique(lab[lab >= 0]))),
                 majority_rate=100 * float(np.bincount(lab[b]).max() / len(b)))
        for mod, F in (("img", I), ("txt", Tm)):
            r[f"probe_{mod}"] = 100 * LogisticRegression(C=1.0, max_iter=300).fit(F[a], lab[a]).score(F[b], lab[b])
        res[g] = r; print(model, g, r, flush=True)
    return res


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "extract":
        extract(args[0], args[1:] or ["artelingo", "cub"])
    elif cmd == "measure":
        model, sets = args[0], args[1:] or ["artelingo", "cub"]
        all_ = load_json(RES); cur = all_.setdefault(model, {})
        if "artelingo" in sets:
            t = time.time(); cur["artelingo"] = artelingo_measure(model); cur["artelingo"]["measure_s"] = time.time() - t
            if model == "clip":
                bad = repro_check(cur["artelingo"]); cur["artelingo"]["repro_mismatch"] = bad
                dump(RES, all_)
                if bad: print("REPRODUCTION FAILED", bad); sys.exit(3)
                print("CLIP reproduction OK")
        if "cub" in sets:
            cur["cub"] = cub_measure(model)
        dump(RES, all_)


if __name__ == "__main__":
    main()
