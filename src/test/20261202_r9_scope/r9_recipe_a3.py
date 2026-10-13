"""Recipe II (ticket 08; rule §3.2 item 2, §3.4): A3 trained from one backbone's features, with the identity gates.

    python r9_recipe_a3.py --backbone B --run {L1,L2,C1,main} [--out DIR] [--inputs DIR] [--compare-to CKPT] [--dry-run]

What it runs: `run_gonogo.train` for A3 (seed 42, 2000 steps), re-done here only because `run_gonogo.train` is
hard-wired to CLIP's `load_artelingo()` and to fixed folders. Everything else is run_gonogo's own code, imported from
the main checkout: `run_config` (R3_CONFIG + C0 cell + A3's RUNS entry), `check_base_config`, `FIELDS`, `sha_file`,
`sha_array`, `code_stats`. `train_factors.py` is unchanged. The call is the same as in `train`:
`train_factors(img[st], txt[st], graph, config, device, group_ids=local_groups, aspect_bank=bank, history=history,
log_every=50)`, then `save_factor_checkpoint`.

Inputs: `partitions.npz` (`local_groups`), `graph.npz`, `bank_AIC.npz` and `build_record.json`, from the backbone's
parts (`r9_backbone.parts`; CLIP: round 6's pinned files, whose `build_record.json` sits beside `partitions.npz`;
a new backbone: `results/recipe/<B>/`, where ticket 07 writes it) or from `--inputs DIR` (all four files in DIR, e.g.
`results/replay/clip_b32/`). The three SHA checks are `run_gonogo.train`'s (local_groups as `sha_array` of the int64
array, graph and bank as file SHAs, against `build_record.json["sha256"]`).

Identity gates (rule §3.2 item 2), written to `identity_gates.json`:
- `config`: the run's config equals the stored one (`history_A3_seed42.json` and the pinned checkpoint's config) with
  `aspect_tau_fixed: False` added. Every backbone; checked before training.
- `inputs` (CLIP only): local_groups, graph and bank SHA-256 equal the pinned ones; the feature-store files and the
  scorer-train feature arrays are hashed and recorded. Checked before training.
- `step1_loss` (CLIP only): `history["loss"][0]` within 1e-5 relative of 20.68084716796875. Needs training, so
  `--dry-run` does not run it.
After training, on CLIP, the checkpoint is compared with the pinned one by `state_equal` (stage 0b case 1); `--compare-to`
compares with another checkpoint (L2 against L1; case 2).

Outputs in `--out` (default `results/recipe/<B>/` for `main`, `results/replay/<B>/a3_<run>/` for L1, L2, C1):
`A3_seed42.pt`, `history_A3_seed42.json`, `picked.json` (run, checkpoint, checkpoint SHA-256), `identity_gates.json`,
each with `<file>.record.json` (SHA-256 of the feature-store files and every upstream part, of this script and the
modules it imports; library versions; device; keys made and copied). Nothing existing is overwritten; a crash leaves
no checkpoint, and the rerun uses the same seed. `--dry-run` writes nothing and does not touch the GPU.
Training needs the GPU (about 10 min on CLIP, peak 3.9 GiB); the script refuses to train on the CPU.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
from r9_backbone import BACKBONES, STORE_PATHS, parts  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402
from r9_recipe_groups import store_shas  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.train.train_factors import load_factor_checkpoint, save_factor_checkpoint, train_factors  # noqa: E402

RUNS = ("L1", "L2", "C1", "main")
RUN, SEED, STEPS, BANK = "A3", 42, 2000, "AIC"      # run_gonogo: RUNS["A3"], FULL_STEPS
GG = R.TEST / "20261101_aspect_factor_gonogo"
PINNED_HISTORY = GG / "results/history_A3_seed42.json"
PINNED_CKPT = GG / "checkpoints/A3_seed42.pt"
PINNED = {                                            # CLIP's stored values (build_record.json, history_A3_seed42.json)
    "local_groups": "10d4559a9f026512ea2c57541321ce0100267e460de0ec1cd0f4c7c917b9b8d4",
    "graph.npz": "485cb7fe12b44bd70bf45ae9c131fc347306c76fc7c6cf75704b95edbb393de8",
    "bank_AIC.npz": "b6c662ef4b79684bc86ee422056782b8cfec278c8c6c164044b9cffbcd59fc41",
    "checkpoint_sha256": "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    "step1_loss": 20.68084716796875,
}
STEP1_RTOL = 1e-5
OUTPUTS = ("A3_seed42.pt", "history_A3_seed42.json", "picked.json", "identity_gates.json")
CODE_FILES = {"run_gonogo": GG / "run_gonogo.py", "train_factors": R.MAIN / "src/train/train_factors.py",
              "r9_data": HERE / "r9_data.py", "r9_common": HERE / "r9_common.py",
              "r9_recipe_groups": HERE / "r9_recipe_groups.py"}
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- state_equal (stage 0b case 1)

def _state(x) -> dict:
    """A state dict, a checkpoint payload or a checkpoint path -> the state dict."""
    if isinstance(x, (str, Path)):
        x = torch.load(x, map_location="cpu", weights_only=True)
    return x["state_dict"] if "state_dict" in x else x


def state_diff(a, b) -> list:
    """What differs between two state dicts: missing keys, and per key dtype, shape or any element (`torch.equal`)."""
    a, b = _state(a), _state(b)
    out = [f"key only in a: {k}" for k in a if k not in b] + [f"key only in b: {k}" for k in b if k not in a]
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if x.dtype != y.dtype:
            out.append(f"{k}: dtype {x.dtype} != {y.dtype}")
        elif x.shape != y.shape:
            out.append(f"{k}: shape {tuple(x.shape)} != {tuple(y.shape)}")
        elif not torch.equal(x.cpu(), y.cpu()):
            out.append(f"{k}: values differ ({int((x.cpu() != y.cpu()).sum())} of {x.numel()} elements)")
    return out


def state_equal(a, b) -> bool:
    """Every state-dict tensor equal (`torch.equal`, same dtype and shape, same keys)."""
    return not state_diff(a, b)


# ---------------------------------------------------------------- gates

def stored_config() -> dict:
    """The stored A3 config (the pinned history's) with `aspect_tau_fixed: False` added (rule §3.2 item 2)."""
    stored = json.loads(PINNED_HISTORY.read_text())["config"]
    assert "aspect_tau_fixed" not in stored, "the stored config already has aspect_tau_fixed"
    return {**stored, "aspect_tau_fixed": False}


def config_gate(config) -> dict:
    mine, want = dataclasses.asdict(config), stored_config()
    diff = {k: (mine.get(k), want.get(k)) for k in sorted(set(mine) | set(want)) if mine.get(k) != want.get(k)}
    assert not diff, f"config gate: differs from the stored one plus aspect_tau_fixed False: {diff}"
    payload = torch.load(PINNED_CKPT, map_location="cpu", weights_only=True)
    ckpt_diff = {k: (mine.get(k), v) for k, v in {**payload["config"], "aspect_tau_fixed": False}.items()
                 if mine.get(k) != v}
    assert not ckpt_diff, f"config gate: differs from the pinned checkpoint's config: {ckpt_diff}"
    return {"passed": True, "added": {"aspect_tau_fixed": False}, "config": mine}


def step1_gate(history) -> dict:
    got = float(history["loss"][0])
    rel = abs(got - PINNED["step1_loss"]) / abs(PINNED["step1_loss"])
    assert rel <= STEP1_RTOL, f"step-1 loss {got!r} differs from {PINNED['step1_loss']!r} by {rel:.3g} (rtol {STEP1_RTOL})"
    return {"passed": True, "step1_loss": got, "pinned": PINNED["step1_loss"], "relative_difference": rel,
            "rtol": STEP1_RTOL}


# ---------------------------------------------------------------- inputs

def input_paths(backbone, inputs) -> dict:
    if inputs is not None:
        d = Path(inputs)
        return {"partitions": d / "partitions.npz", "graph": d / "graph.npz", "bank": d / "bank_AIC.npz",
                "build_record": d / "build_record.json"}
    p = parts(backbone)
    return {"partitions": p["partitions"], "graph": p["graph"], "bank": p["bank_AIC"],
            "build_record": p["partitions"].parent / "build_record.json"}


def load_inputs(backbone, inputs):
    """Everything A3 trains on, with `run_gonogo.train`'s alignment and SHA checks. Returns a dict."""
    rg = R.rg
    paths = input_paths(backbone, inputs)
    for name, path in paths.items():
        assert path.is_file(), f"missing input {name}: {path}"
    record = json.loads(paths["build_record"].read_text())
    assert {"local_groups", "graph.npz", "bank_AIC.npz"} <= set(record["sha256"]), "build_record.json keys"
    data = R.load_data(backbone)
    split = R.load_split(data)
    st = np.asarray(split.scorer_train)
    n = len(st)
    assert n == 183_694, f"scorer_train has {n} rows"
    with np.load(paths["partitions"], allow_pickle=False) as z:
        assert "local_groups" in z.files, "partitions.npz columns"
        local_groups = z["local_groups"].astype(np.int64)
    assert np.array_equal(local_groups, np.unique(split.groups[st], return_inverse=True)[1]), \
        "partitions.npz local_groups != np.unique(groups[scorer_train], return_inverse=True)[1]"
    sha = {"local_groups": rg.sha_array(local_groups), "graph.npz": rg.sha_file(paths["graph"]),
           "bank_AIC.npz": rg.sha_file(paths["bank"])}
    for key, want in record["sha256"].items():
        if key in sha:
            assert sha[key] == want, f"{key}: SHA-256 {sha[key]} differs from {paths['build_record']}"
    graph = load_npz(paths["graph"]).tocsr()
    assert graph.shape == (n, n), f"graph shape {graph.shape} != ({n}, {n})"
    with np.load(paths["bank"], allow_pickle=False) as z:
        bank = AspectEpisodes(str(z["aspect_a"]), str(z["aspect_b"]), *(z[f].astype(np.int64) for f in rg.FIELDS))
    rows = bank.rows()
    assert 0 <= rows.min() and rows.max() < n, f"bank rows leave 0..{n - 1}: [{rows.min()}, {rows.max()}]"
    img, txt = data.img_features[st], data.txt_features[st]
    assert img.shape == txt.shape == (n, data.img_features.shape[1]) and img.dtype == txt.dtype == np.float32
    return {"paths": paths, "sha": sha, "record_sha": R.sha256_file(paths["build_record"]), "img": img, "txt": txt,
            "graph": graph, "bank": bank, "local_groups": local_groups, "n": n}


def inputs_gate(backbone, loaded) -> dict:
    """CLIP: the inputs equal the pinned ones. Other backbones: the SHA checks of `load_inputs` stand."""
    feats = {"img_scorer_train": R.rg.sha_array(loaded["img"]), "txt_scorer_train": R.rg.sha_array(loaded["txt"])}
    if backbone != "clip_b32":
        return {"passed": True, "pinned": False, "sha256": loaded["sha"], "features": feats}
    for key in ("local_groups", "graph.npz", "bank_AIC.npz"):
        assert loaded["sha"][key] == PINNED[key], f"inputs gate: {key} {loaded['sha'][key]} != pinned {PINNED[key]}"
    return {"passed": True, "pinned": True, "sha256": loaded["sha"], "features": feats,
            "feature_store_files": store_shas(backbone)}


# ---------------------------------------------------------------- records, refusal

def refusal(out, why) -> FileExistsError:
    present = [str(Path(out) / n) for f in OUTPUTS for n in (f, f + ".record.json") if (Path(out) / n).exists()]
    return FileExistsError(f"refusing to overwrite {why}. A completed run is never replaced (rule §3.4); a crashed "
                           f"run leaves no checkpoint. To rerun in {out}, remove these first: {' '.join(present)}")


def preflight(out):
    for f in OUTPUTS:
        for target in (Path(out) / f, Path(out) / (f + ".record.json")):
            if target.exists():
                raise refusal(out, target)


def code_record() -> dict:
    return {"files": {k: {"file": str(p), "sha256": R.sha256_file(p)} for k, p in CODE_FILES.items()},
            "versions": {"numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}}


def write_file(out, name, content, *, input_sha256, made, copied, extra=None):
    """One output (bytes via a callable taking the path, or JSON-able content) plus its record; exclusive creation."""
    out = Path(out)
    path, rec_path = out / name, out / (name + ".record.json")
    for target in (path, rec_path):
        if target.exists():
            raise refusal(out, target)
    out.mkdir(parents=True, exist_ok=True)
    if callable(content):
        content(path)
    else:
        with path.open("x") as file:
            json.dump(content, file, indent=1)
            file.write("\n")
    rec = {"file_sha256": R.sha256_file(path), "input_sha256": input_sha256, "script": str(Path(__file__)),
           "script_sha256": R.sha256_file(__file__), "code": code_record(), "keys_made": list(made),
           "keys_copied": list(copied), **(extra or {})}
    with rec_path.open("x") as file:
        json.dump(rec, file, indent=2)
        file.write("\n")
    return rec


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--run", choices=RUNS, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inputs", type=Path, default=None, help="folder with partitions.npz, graph.npz, bank_AIC.npz, "
                    "build_record.json (default: the backbone's parts)")
    ap.add_argument("--compare-to", type=Path, default=None, help="a checkpoint to compare with (L2 against L1)")
    ap.add_argument("--dry-run", action="store_true", help="CPU: config, inputs and the gates that need no training")
    args = ap.parse_args(argv)
    if args.run in ("L1", "L2", "C1"):
        assert args.backbone == "clip_b32", "L1, L2 and C1 are the stage 0b replay on CLIP"
    out = args.out or (R.RESULTS / "recipe" / args.backbone if args.run == "main"
                       else R.RESULTS / "replay" / args.backbone / f"a3_{args.run}")
    preflight(out)
    rg = R.rg
    config = rg.run_config(RUN, SEED, STEPS)
    rg.check_base_config(SEED, STEPS)
    gates = {"backbone": args.backbone, "run": args.run, "config": config_gate(config)}
    log(f"{args.backbone} {args.run}: config gate passed")
    loaded = load_inputs(args.backbone, args.inputs)
    gates["inputs"] = inputs_gate(args.backbone, loaded)
    log(f"inputs loaded ({loaded['n']} scorer-train rows, graph {loaded['graph'].shape}, "
        f"bank {len(loaded['bank'].anchor)} episodes); inputs gate passed (pinned: {gates['inputs']['pinned']})")
    if args.dry_run:
        gates["step1_loss"] = {"passed": None, "note": "needs training; not run in --dry-run"}
        log("dry run: gates that need no training passed; nothing written")
        print(json.dumps({"config": gates["config"]["passed"], "inputs": gates["inputs"]["passed"],
                          "inputs_pinned": gates["inputs"]["pinned"], "step1_loss": "not run"}))
        return 0
    assert torch.cuda.is_available(), "A3 trains on the GPU (run under the gpu0 lock); the script refuses the CPU"
    history: dict = {}
    torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    model, img_codes, txt_codes = train_factors(
        loaded["img"], loaded["txt"], loaded["graph"], config, device="cuda", group_ids=loaded["local_groups"],
        aspect_bank=loaded["bank"], history=history, log_every=50)
    train_s = perf_counter() - t0
    assert np.isfinite(img_codes).all() and np.isfinite(txt_codes).all(), "non-finite codes; nothing written"
    peak = torch.cuda.max_memory_allocated() / 2**30
    if args.backbone == "clip_b32":
        gates["step1_loss"] = step1_gate(history)
    else:
        gates["step1_loss"] = {"passed": None, "note": "pinned for CLIP only"}
    ckpt = out / OUTPUTS[0]
    ups = {"build_record.json": loaded["record_sha"], **{k: v for k, v in loaded["sha"].items()}}
    feat = store_shas(args.backbone) if args.backbone != "clip_b32" else gates["inputs"]["feature_store_files"]
    upstream = {"parts": {k: str(v) for k, v in loaded["paths"].items()}, "part_sha256": ups, "feature_store": feat}
    dev = {"device": torch.cuda.get_device_name(0), "train_s": train_s, "peak_gpu_gib": peak}
    write_file(out, OUTPUTS[0], lambda p: save_factor_checkpoint(model, config, p), input_sha256=upstream,
               made=["state_dict", "config", "feature_dim"], copied=[], extra=dev)
    ck_sha = R.sha256_file(ckpt)
    cmp = {}
    if args.backbone == "clip_b32":
        d = state_diff(ckpt, PINNED_CKPT)
        cmp["pinned"] = {"state_equal": not d, "differences": d[:20], "n_differences": len(d),
                         "checkpoint_sha256_equal": ck_sha == PINNED["checkpoint_sha256"]}
    if args.compare_to is not None:
        d = state_diff(ckpt, args.compare_to)
        cmp[str(args.compare_to)] = {"state_equal": not d, "differences": d[:20], "n_differences": len(d)}
    gates["state_equal"] = cmp
    hist = {"run": RUN, "seed": SEED, "steps": STEPS, "bank": BANK, "backbone": args.backbone,
            "config": dataclasses.asdict(config), "checkpoint_sha256": ck_sha, "input_sha256": ups, **dev,
            "code_stats_scorer_train": rg.code_stats(img_codes, txt_codes), "history": history}
    write_file(out, OUTPUTS[1], hist, input_sha256=upstream, made=list(hist), copied=[])
    picked = {"run": RUN, "backbone": args.backbone, "seed": SEED, "checkpoint": str(ckpt), "checkpoint_sha256": ck_sha}
    write_file(out, OUTPUTS[2], picked, input_sha256={OUTPUTS[0]: ck_sha}, made=list(picked), copied=[])
    write_file(out, OUTPUTS[3], gates, input_sha256=upstream, made=list(gates), copied=[])
    log(f"A3 {args.backbone} {args.run}: {STEPS} steps in {train_s:.1f}s, peak {peak:.2f} GiB, "
        f"state_equal {cmp} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
