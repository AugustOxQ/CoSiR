"""Round 8 descriptive diagnostics and the CLIP-head arm (ticket 12; A 3.6, D28; B 3.6, ED10; rule 6.6, 7 "after it").

Measured only; nothing here decides anything and nothing here is read by the gate.

  labelfree  group sizes, AMI with the content grouping, between settings, with R0own, between encoders, own-map
             agreement on held-out halves, FCMI's content-share spread, (ii)'s final loss, M3's (r, K) records.
             May run before the verdict. Writes descriptive_labelfree.json (write-once).
  harm       the H<X> arm (rule 6.6): runs ticket 08's CLIP image heads and ticket 09's driver (subprocess) in their own
             arrays batch "hx". Refuses before gate_verdict.json. Writes harm_run.json.
  full       everything: label-free parts again, plus the label-based and score-based parts (AMI with style and genre,
             caption-head accuracy, own-map accuracy against labels, per-scorer numbers, pick accuracy under the told map,
             placement, the H<X> scorers). Refuses before gate_verdict.json. Writes descriptive.json (write-once).

The refusal is a guard (C5-like ordering): the verdict file is checked before any label, score or head file is read.
--smoke works on <results>/smoke/gate (override with --root, under results/smoke/) and writes there only; labels are
permuted with default_rng(0), as R7's chain smoke did.

  python run_r8_descriptive.py labelfree [--smoke]
  python run_r8_descriptive.py harm [--smoke]
  python run_r8_descriptive.py full [--smoke] [--no-harm]
"""
import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

import r8_common as R
import r8_gate as G

HERE = Path(__file__).resolve().parent
A0 = ("affect", "image", "caption")
SETTINGS = {"V": ("V100", "V67", "V33"), "NB": ("NB100", "NB67", "NB33", "NBC"),
            "FC": tuple(f"FC{i}" for i in range(1, 7))}
ENCODERS = ("AL", "DI", "SG")
LOSS_IDS = ("V100", "V67", "V33", "NB100", "NB67", "NB33", "NBC")
HARM_BATCH = "hx"


# ------------------------------------------------------------------------------------------- roots and the guard

def results_root(smoke, root=None):
    if smoke:
        out = Path(root) if root else R.SMOKE / "gate"
        assert out.resolve().is_relative_to(R.SMOKE.resolve()), f"smoke outputs must stay under {R.SMOKE}"
    else:
        assert not root, "--root is for smoke only; real results go to results/"
        out = R.RESULTS
    return out.resolve()


def require_verdict(root, smoke):
    """Refuse before any label, score or head file is read: gate_verdict.json must exist (rule 7, "after it, not before")."""
    path = Path(root) / "gate_verdict.json"
    assert path.is_file(), f"refusing to run: {path} does not exist (label-based diagnostics and H<X> come after the verdict)"
    fixes = sorted(path.parent.glob("gate_verdict_fix*.json"))
    assert not fixes, f"{path}: a corrected verdict exists ({[f.name for f in fixes]}); the user decides which one to describe"
    rec = json.loads(path.read_text())
    assert isinstance(rec, dict) and rec.get("smoke") is smoke, f"{path}: smoke role differs"
    assert rec.get("rule_sha256") == R.sha_file(R.RULE), f"{path}: rule SHA-256 differs"
    assert isinstance(rec.get("main"), dict) and isinstance(rec.get("screen"), dict), f"{path}: keys main and screen"
    return rec


# ------------------------------------------------------------------------------------------- small helpers

def ami(a, b):
    return float(adjusted_mutual_info_score(a, b))


def read_json(path):
    path = Path(path)
    assert path.is_file(), f"{path}: required file missing"
    rec = json.loads(path.read_text())
    assert isinstance(rec, dict), f"{path}: expected an object"
    return rec


def frame_from_ctx():
    """The arrays this module needs, from r8_common.ctx() (the one place that builds them)."""
    c = R.ctx()
    return SimpleNamespace(scorer_train=c.scorer_train, g_st=c.g_st, first_rows=c.first_rows, content=c.content,
                           halves=c.halves, n_paintings=c.n_paintings, data=c.data)


def read_groupings(root, frame):
    """(labels {ID: int64 per scorer-train row}, records {ID: M3 record}, dropped {ID: reason}); the main file plus
    ticket 08's optional NBC file (r8_groupings_nbc)."""
    labels, records, dropped = {}, {}, {}
    for stem in ("r8_groupings", "r8_groupings_nbc"):
        npz, js = root / "groupings" / f"{stem}.npz", root / "groupings" / f"{stem}.json"
        if stem != "r8_groupings" and not js.is_file():
            continue
        rec = read_json(js)
        assert isinstance(rec.get("groupings"), dict) and isinstance(rec.get("dropped"), dict), f"{js}: shape"
        with np.load(npz, allow_pickle=False) as z:
            assert {"scorer_train", "g_st"} <= set(z.files), f"{npz}: keys"
            assert np.array_equal(z["scorer_train"], frame.scorer_train), f"{npz}: scorer-train rows differ"
            assert np.array_equal(z["g_st"], frame.g_st), f"{npz}: painting codes differ"
            for x in rec["groupings"]:
                assert x in z.files, f"{npz}: labels for {x} missing"
                lab = np.asarray(z[x])
                assert lab.dtype == np.int64 and lab.shape == frame.g_st.shape, f"{x}: label shape/dtype"
                assert np.array_equal(lab[frame.first_rows][frame.g_st], lab), f"{x}: labels vary within a painting"
                assert np.array_equal(np.unique(lab), np.arange(lab.max() + 1)), f"{x}: labels are not compact"
                labels[x] = lab
        records.update(rec["groupings"])
        dropped.update({k: str(v) for k, v in rec["dropped"].items()})
    # Late drops (index "Late drops": 08's M12/M16 and 09's reader drops), one drops-only file per batch.
    for js in sorted((root / "groupings").glob("r8_drops_*.json")):
        rec = read_json(js)
        assert rec.get("groupings") == {} and isinstance(rec.get("dropped"), dict), f"{js}: shape"
        dropped.update({k: str(v) for k, v in rec["dropped"].items()})
    return labels, records, dropped


# ------------------------------------------------------------------------------------------- label-free parts

def share_spread(lab_p, content_p, n_content=64):
    """FCMI's spread of content-group shares per cluster (A 3.6): per cluster the shares of the 64 content groups.
    Reported per cluster: size, largest share, normalised entropy (1 = as spread as 64 equal groups), the standard
    deviation of the 64 shares; the same three for all paintings together as the reference; mean over clusters."""
    # rule gap: A 3.6 names "spread of content-group shares per cluster" but no statistic; these three are chosen.
    k = int(lab_p.max()) + 1
    tab = np.zeros((k, n_content))
    np.add.at(tab, (lab_p, content_p), 1)

    def stats(counts):
        s = counts / counts.sum()
        nz = s[s > 0]
        return {"max_share": float(s.max()), "entropy_normalised": float(-(nz * np.log(nz)).sum() / np.log(n_content)),
                "share_std": float(s.std())}
    clusters = [{"size": int(tab[i].sum()), **stats(tab[i])} for i in range(k)]
    mean = {key: float(np.mean([c[key] for c in clusters])) for key in ("max_share", "entropy_normalised", "share_std")}
    return {"clusters": clusters, "mean": mean, "reference_all_paintings": stats(tab.sum(axis=0))}


def own_map_agreement(img, local, halves_row):
    """Share of scorer-train rows whose cross-fitted own-map group (argmax of the xheads one-hot, centres from the
    other half, M12) equals the grouping's label; overall, per painting half, and per painting."""
    assert img.shape[0] == local.shape[0] == halves_row.shape[0]
    hit = img.argmax(axis=1) == local
    return {"rows": float(hit.mean()), "half0": float(hit[halves_row == 0].mean()),
            "half1": float(hit[halves_row == 1].mean())}


def m3_record(rec):
    keep = ("evaluated", "chosen_r", "K", "stage", "target_K", "exact_17", "merge_min_rows", "merge_moves", "seed",
            "stability_ami", "stability_seeds", "graph")
    return {k: rec[k] for k in keep if k in rec}


def label_free(root, frame, smoke):
    labels, records, dropped = read_groupings(root, frame)
    first = frame.first_rows
    lab_p = {x: v[first] for x, v in labels.items()}
    content = frame.content
    halves_row = frame.halves[frame.g_st]
    out = {"groupings": {}, "dropped": dropped, "between_settings": {}, "between_encoders": {}, "with_R0own": {}}
    for x in sorted(labels, key=lambda s: list(records).index(s)):
        p = lab_p[x]
        info = {"n_groups": int(p.max()) + 1, "sizes_paintings": np.bincount(p).tolist(),
                "sizes_rows": np.bincount(labels[x]).tolist(), "ami_content_paintings": ami(content, p),
                "m3": m3_record(records[x])}
        if x in LOSS_IDS:
            sp = read_json(root / "spaces" / f"{x}.json")
            info["head_final_loss"] = sp["training"]["last_epoch_mean_loss"]
        if x.startswith("FC") and x[2:].isdigit():
            info["content_share_spread"] = share_spread(p, content)
        if not x.startswith("Z"):               # a random control's image side is a draw: no own map to compare
            xh = root / "heads" / f"xheads_{x}.npz"
            if xh.is_file():
                with np.load(xh, allow_pickle=False) as z:
                    info["own_map_agreement_heldout"] = own_map_agreement(z["img"], labels[x], halves_row)
            else:
                info["own_map_agreement_heldout"] = None
        out["groupings"][x] = info
    for fam, ids in SETTINGS.items():
        have = [i for i in ids if i in lab_p]
        out["between_settings"][fam] = {f"{a}__{b}": ami(lab_p[a], lab_p[b]) for a, b in combinations(have, 2)}
    have = [e for e in ENCODERS if e in lab_p]
    out["between_encoders"] = {f"{a}__{b}": ami(lab_p[a], lab_p[b]) for a, b in combinations(have, 2)}
    # rule gap: "with R0" is read as AMI with R0own (same grouping and recipe as R7's R0, SC1e), painting level.
    if "R0own" in lab_p:
        out["with_R0own"] = {x: ami(lab_p["R0own"], p) for x, p in lab_p.items() if x != "R0own"}
    out["groupings_unit"] = "AMI unscaled, painting level; own-map agreement a share of scorer-train rows"
    return out


# ------------------------------------------------------------------------------------------- label-based parts

def style_genre(frame, smoke):
    """Style and genre labels of the scorer-train rows, as R7 reads them (artelingo_aspect_labels); chain smoke permutes."""
    lab = R.load_prior("run_told_oracle").artelingo_aspect_labels(frame.data)
    out = {k: np.asarray(lab[k])[frame.scorer_train] for k in ("style", "genre")}
    for k, v in out.items():
        assert v.dtype == np.int64 and v.shape == frame.scorer_train.shape, f"{k}: label shape/dtype"
    if smoke:
        perm = np.random.default_rng(0).permutation(len(frame.scorer_train))
        out = {k: v[perm] for k, v in out.items()}
    return out


def majority_accuracy(group, label, half_row, k):
    """Share of labelled rows whose label equals the majority label of their group, the majority taken on the other
    painting half (no row votes for itself); and the single-majority-class share as the baseline."""
    ok = label >= 0
    hit = total = 0
    for h in (0, 1):
        fit, test = ok & (half_row != h), ok & (half_row == h)
        tab = np.zeros((k, int(label.max()) + 1), dtype=np.int64)
        np.add.at(tab, (group[fit], label[fit]), 1)
        guess = tab.argmax(axis=1)
        hit += int((guess[group[test]] == label[test]).sum())
        total += int(test.sum())
    base = np.bincount(label[ok]).max() / ok.sum()
    return {"accuracy": hit / total, "n_rows": total, "majority_class_share": float(base)}


def label_diagnostics(root, frame, smoke):
    labels, records, dropped = read_groupings(root, frame)
    lab = style_genre(frame, smoke)
    halves_row = frame.halves[frame.g_st]
    out = {}
    for x, v in labels.items():
        info = {"ami": {}, "own_map_ami": {}, "own_map_majority_accuracy": {}}
        for k, y in lab.items():
            m = y >= 0
            assert m.any(), f"{k}: no labelled scorer-train rows"
            info["ami"][k] = {"n_rows": int(m.sum()), "value": ami(v[m], y[m])}
        hp = root / "heads" / f"head_{x}.json"
        if x in dropped:    # dropped after its grouping was built: labels exist, heads may not
            info["dropped"] = dropped[x]
            out[x] = info
            continue
        head = read_json(hp)
        assert head.get("grouping") == x and head.get("dropped") is False, f"{hp}: grouping/drop status"
        text = head["text"]
        if "standard" in text:
            acc = text["standard"]["heldout_accuracy"]
            info["caption_head_accuracy"] = {"txt": acc["txt"], "img_clip_head": acc.get("img"),
                                             "majority_share": text["standard"].get("check_majority_share"),
                                             "uniform": text["standard"].get("uniform")}
        else:   # R0own: R7's own post_R0 text side; its accuracy is R7's head_R0.json
            r7 = read_json(R.MAIN / "src/test/20261126_design_l_style_source/results/heads/head_R0.json")
            acc = r7["standard"]["heldout_accuracy"]
            info["caption_head_accuracy"] = {"txt": acc["txt"], "img_clip_head": acc.get("img"),
                                             "source": "R7 head_R0.json"}
        if not x.startswith("Z") and (root / "heads" / f"xheads_{x}.npz").is_file():
            with np.load(root / "heads" / f"xheads_{x}.npz", allow_pickle=False) as z:
                own = z["img"].argmax(axis=1).astype(np.int64)
            k_groups = int(v.max()) + 1
            for k, y in lab.items():
                m = y >= 0
                info["own_map_ami"][k] = ami(own[m], y[m])
                # rule gap: "own-map accuracy against labels" is read as the held-half majority-label accuracy of the
                # cross-fitted own-map group, beside the same measure for the grouping's own labels.
                info["own_map_majority_accuracy"][k] = {
                    "own_map": majority_accuracy(own, y, halves_row, k_groups),
                    "grouping": majority_accuracy(v, y, halves_row, k_groups)}
        out[x] = info
    return out


# ------------------------------------------------------------------------------------------- scores

def load_seed42(root, smoke, allow_hx):
    """Merge every indexed seed-42 batch (SHA-256 asserted). The H<X> batch is read only after the verdict (callers)."""
    seed42 = root / "seed42"
    indices = sorted(seed42.glob("index*.json"))
    assert indices, f"{seed42}: no index"
    batches, shas = [], {}
    for ip in indices:
        for e in read_json(ip)["batches"]:
            p = seed42 / e["file"]
            assert R.sha_file(p) == e["sha256"], f"{p}: SHA-256 differs from {ip.name}"
            with np.load(p, allow_pickle=False) as z:
                b = {k: z[k] for k in z.files}
            if not allow_hx:
                b = {k: v for k, v in b.items() if not k.startswith(("R1_H", "XAFF_H", "BX_H"))}
            batches.append(b)
            shas[str(p)] = e["sha256"]
    return G.merge_batches(batches), shas


def open_shares(arr, name, pair_index):
    ga, gb = arr[f"{name}__gate_a"], arr[f"{name}__gate_b"]
    out = {}
    for t in range(ga.shape[0]):
        out[f"tau_{t}"] = {"pooled": float(100.0 * (ga[t].sum() + gb[t].sum()) / (2 * ga.shape[1])),
                           **{G.PAIRS[i]: float(100.0 * (ga[t][pair_index == i].sum() + gb[t][pair_index == i].sum())
                                                / (2 * (pair_index == i).sum())) for i in range(3)}}
    return out


def pick_accuracy(root, name, arr, smoke):
    kind, x = G.split_name(name)
    path = root / "seed42" / f"reader_{x}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        pick = {c: z[f"pick__{c}"] for c in ("a", "b")}
    parts = A0 + (x,)
    told = {"emotion": "affect", "style": x, "genre": "image"}
    acc, shares = R.load_prior("common").pick_statistics(pick, parts, told, np.asarray(arr["pair_index"]),
                                                          np.asarray(arr["cl"]))
    return {"told_map": told, "accuracy": acc, "shares": shares}


def scorer_numbers(arr, name, boot=None, with_intervals=True):
    """Per-scorer descriptives (A 3.6): per-pair R@1 and condition gain; margins over B'(A1), B'(X), CF and AFF; gate-open
    shares. Points are R@1 points (25 * sum(4d) / n); intervals pooled only."""
    kind, x = G.split_name(name)
    cl, pi = np.asarray(arr["cl"]), np.asarray(arr["pair_index"])
    n = len(cl)
    s, gain = np.asarray(arr[f"{name}__r1"], dtype=np.float64), np.asarray(arr[f"{name}__gain"], dtype=np.float64)
    cf, cf_gain = np.asarray(arr[f"{name}_cf__r1"], dtype=np.float64), np.asarray(arr[f"{name}_cf__gain"], dtype=np.float64)
    comps = {"B1": arr["B1__r1"], "BX": arr[f"BX_{x}__r1"], "CF": cf, "AFF": arr["AFF__r1"]}
    masks = {"pooled": np.ones(n, dtype=bool), **{G.PAIRS[i]: pi == i for i in range(3)}}
    out = {"per_pair": {}, "margins": {}}
    for scope, m in masks.items():
        out["per_pair"][scope] = {"n": int(m.sum()), "r1": float(100 * s[m].mean()), "cf_r1": float(100 * cf[m].mean()),
                                  "gain": float(100 * gain[m].mean()),
                                  "gain_statistic": float(100 * (gain - cf_gain)[m].mean())}
    for cname, c in comps.items():
        d = s - np.asarray(c, dtype=np.float64)
        rec = {}
        for scope, m in masks.items():
            rec[scope] = {"sum4d": G.sum4(d[m]), "point": G.point_of(G.sum4(d[m]), int(m.sum()))}
        if with_intervals:
            rec["pooled"]["ci95"] = G.interval(d, cl, boot)[1]
        out["margins"][f"s_minus_{cname}"] = rec
    out["gate_open_shares"] = open_shares(arr, name, pi)
    return out


def paired(arr, a, b, boot=None):
    """a - b pooled and per pair on R@1 (sum of 4d, points); interval pooled."""
    cl, pi = np.asarray(arr["cl"]), np.asarray(arr["pair_index"])
    d = np.asarray(arr[a], dtype=np.float64) - np.asarray(arr[b], dtype=np.float64)
    out = {"pooled": {"sum4d": G.sum4(d), "point": G.point_of(G.sum4(d), len(d)), "ci95": G.interval(d, cl, boot)[1]}}
    for i, p in enumerate(G.PAIRS):
        out[p] = {"sum4d": G.sum4(d[pi == i]), "point": G.point_of(G.sum4(d[pi == i]), int((pi == i).sum()))}
    return out


def placement(arr, smoke, pick_names, h_names, boot=None):
    """R0 with own assignment against round 7's R0 (A 3.6), and the pick against X-H (the CLIP-head arm)."""
    out = {"R0own_vs_R7_R0": {}, "pick_vs_H": {}}
    old_path = R.MAIN / "src/test/20261126_design_l_style_source/results/seed42/arrays.npz"
    if not smoke:
        with np.load(old_path, allow_pickle=False) as z:
            old = {k: z[k] for k in z.files}
        assert np.array_equal(old["cl"], arr["cl"]) and np.array_equal(old["pair_index"], arr["pair_index"]), \
            "R7 seed-42 anchors differ from this round's"
        for kind in G.KINDS:
            if f"{kind}_R0own__r1" in arr:
                merged = {"new": arr[f"{kind}_R0own__r1"], "old": old[f"{kind}_R0__r1"], "cl": arr["cl"],
                          "pair_index": arr["pair_index"]}
                out["R0own_vs_R7_R0"][kind] = paired(merged, "new", "old", boot)
    else:
        out["R0own_vs_R7_R0"] = {"smoke": "R7's real seed-42 arrays have 12,288 episodes; not compared in smoke"}
    for pick, h in zip(pick_names, h_names):
        if pick is not None and f"{h}__r1" in arr:
            out["pick_vs_H"][pick] = {"H_scorer": h, **paired(arr, f"{pick}__r1", f"{h}__r1", boot)}
    return out


def harm_scorers(arr, verdict, boot=None):
    """H<X> with §6.1's quantities (rule 6.6), both kinds, the target's kind first."""
    target = verdict["main"].get("head_target")
    if target is None:
        return {"target": None, "note": "no main-line candidate was evaluated; the verdict has no H<X> target"}
    x, kind = target["grouping"], target["scorer_kind"]
    out = {"target": target, "scorers": {}}
    for k in (kind, *[k for k in G.KINDS if k != kind]):
        name = f"{k}_H{x}"
        assert f"{name}__r1" in arr, f"{name}: not in the seed-42 batches; run `harm` first"
        out["scorers"][name] = {"gate_quantities": G.evaluate_scorer(arr, name, boot),
                                "numbers": scorer_numbers(arr, name, boot)}
    out["same_kind_source_scorer"] = target["source_scorer"]
    return out


def full(root, smoke, need_harm=True):
    verdict = require_verdict(root, smoke)              # the guard comes first
    frame = frame_from_ctx()
    out = {"smoke": smoke, "units": "R@1, gain and accuracy in points; AMI unscaled",
           "label_free": label_free(root, frame, smoke)}
    out["label_based"] = label_diagnostics(root, frame, smoke)
    arr, shas = load_seed42(root, smoke, allow_hx=need_harm)
    out["seed42_batches"] = shas
    scorers = {}
    for track in ("main", "screen"):
        for role in ("candidates", "references", "controls"):
            for name, rec in verdict[track].get(role, {}).items():
                if rec["dropped"]:
                    scorers[name] = {"dropped": True, "reason": rec.get("reason")}
                else:
                    scorers[name] = {"role": f"{track}_{role}", "numbers": scorer_numbers(arr, name),
                                     "r7_S1": rec["S1"], "r7_S2": rec["S2"],
                                     "pick_accuracy_told_map": pick_accuracy(root, name, arr, smoke)}
    out["scorers"] = scorers
    out["random_control_beside_pick"] = {
        track: ({"pick": v["pick"], "control": f"{G.split_name(v['pick'])[0]}_Z{G.split_name(v['pick'])[1]}",
                 "flag": v.get("random_control_flag")} if v.get("pick") else None)
        for track, v in ((t, verdict[t]) for t in ("main", "screen"))}
    h_names = []
    picks = []
    if need_harm:
        out["H_arm"] = harm_scorers(arr, verdict)
        target = out["H_arm"].get("target")
        if target:
            picks.append(verdict["main"]["pick"])
            h_names.append(f"{G.split_name(verdict['main']['pick'])[0]}_H{target['grouping']}"
                           if verdict["main"]["pick"] else None)
    else:
        out["H_arm"] = "not included (--no-harm)"
    out["placement"] = placement(arr, smoke, picks, h_names)
    return out


# ------------------------------------------------------------------------------------------- the H<X> arm

def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    log.append({"cmd": cmd, "returncode": r.returncode, "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]})
    assert r.returncode == 0, f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr[-500:]}"


def harm(root, smoke):
    """Build the H<X> arm after the verdict: placement with CLIP image heads, then readers, taus, score in batch hx."""
    verdict = require_verdict(root, smoke)              # the guard comes first
    target = verdict["main"].get("head_target")
    if target is None:
        return {"target": None, "note": "the verdict has no H<X> target (no main-line candidate evaluated)", "steps": []}
    x = target["grouping"]
    hx = f"H{x}"
    py = sys.executable
    flag = ["--smoke"] if smoke else []
    log = []
    # interface: in smoke, r8_place.py writes under results/smoke/ (R.SMOKE), so the H<X> arm needs that root too.
    assert not smoke or root == R.SMOKE.resolve(), f"smoke harm needs --root {R.SMOKE} (where r8_place.py writes)"
    if not (root / "heads" / f"post_{hx}.npz").is_file():
        # interface: ticket 08 `r8_place.py --grouping X --h-only [--smoke [N]]` (X is placed already) writes only
        # heads/post_H<X>, xheads_H<X>, head_H<X>; it refuses when X's files are missing or X is dropped.
        # smoke: 08 asserts `--smoke N` equals the painting count of the groupings file (default N is 8,192), so pass that
        # count, read from the file (queue smoke groupings hold every painting). Real mode passes no N.
        # NBC lives in its own grouping file (07's NBC batch); 08 finds it only by --groupings-name.
        gname = "r8_groupings_nbc" if x == "NBC" else "r8_groupings"
        n_flag = []
        if smoke:
            gjson = root / "groupings" / f"{gname}.json"
            assert gjson.is_file(), f"smoke harm: {gjson} missing (08 reads it)"
            n_flag = [str(int(json.loads(gjson.read_text())["n_paintings"]))]
        run([py, "r8_place.py", "--grouping", x, "--h-only", *flag, *n_flag, "--groupings-name", gname], log)
    seed42 = root / "seed42"
    root_flag = ["--root", str(root)] if smoke else []
    if not (seed42 / f"reader_{hx}.npz").is_file():
        # interface: ticket 09 `run_r8.py readers --groupings H<X> [--smoke --root R]`.
        run([py, "run_r8.py", "readers", "--groupings", hx, *flag, *root_flag], log)
    if not (seed42 / f"index_{HARM_BATCH}.json").is_file():
        # interface: ticket 09 `run_r8.py taus --batch hx --groupings H<X>` and `score --batch hx --groupings H<X>`
        # (index written as seed42/index_hx.json, arrays as seed42/arrays_hx.npz).
        run([py, "run_r8.py", "taus", "--batch", HARM_BATCH, "--groupings", hx, *flag, *root_flag], log)
        run([py, "run_r8.py", "score", "--batch", HARM_BATCH, "--groupings", hx, *flag, *root_flag], log)
    return {"target": target, "steps": log}


# ------------------------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=("labelfree", "harm", "full"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--root", help="smoke only: results root (default results/smoke/gate)")
    p.add_argument("--no-harm", action="store_true", help="full: leave the H<X> arm out")
    args = p.parse_args(argv)
    R.assert_rule()
    R.assert_threads()
    root = results_root(args.smoke, args.root)
    if args.part == "labelfree":
        rec = {"smoke": args.smoke, **label_free(root, frame_from_ctx(), args.smoke)}
        path = root / "descriptive_labelfree.json"
    elif args.part == "harm":
        rec = {"smoke": args.smoke, **harm(root, args.smoke)}
        path = root / "harm_run.json"
    else:
        rec = full(root, args.smoke, need_harm=not args.no_harm)
        path = root / "descriptive.json"
    R.write_json_once(path, rec)
    print(f"{args.part}: {path}")


if __name__ == "__main__":
    main()
