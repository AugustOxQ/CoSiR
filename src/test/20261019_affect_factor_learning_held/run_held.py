"""CoSiR v2 Candidate A affect factor learning: the held test (affect spec §7, plan Task 5).

Spec: docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md (§7 is the criterion).
The picked cell (SE, from run_affect.py --evaluate) is confirmed once against the matched control C0 on fresh held
label episodes. Run from the repository root, one phase per call, in this order:

    python src/test/20261019_affect_factor_learning_held/run_held.py --power   # n per label (spec §7); writes
                                                                                # results/power.json, reads no held row
    python src/test/20261019_affect_factor_learning_held/run_held.py --smoke   # the --run code path on SELECTION rows
                                                                                # (seed 43, the chosen n); discarded
    python src/test/20261019_affect_factor_learning_held/run_held.py --run     # the held test: ONCE
    python src/test/20261019_affect_factor_learning_held/run_held.py --tables  # reprint held_results.json
                                                                                # (--tables --smoke: the smoke's)

Hard rules (asserted in the code):
- Held rows are read only by --run, once. --run refuses to start if results/held_results.json exists, or if
  results/held_started.json exists (a started run that did not finish) unless --after-crash is given (the attempt is
  then recorded). It also refuses unless power.json and a passing smoke of THIS script's exact bytes exist.
- --power writes results/power.json before any held read (its timestamp is asserted to precede the run's start).
- Factors only ENCODE held rows (checkpoints trained on scorer-train rows; original R3 on the train part); codes and
  CLIP features are NaN outside the evaluated rows. Affect vectors are never computed: the GoEmotions functions are
  replaced by a raising stub before anything runs, and the affect vectors (cache/affect_prepare.npz) are never
  loaded (only affect_prepare.json's recorded C0 SHA-256 is read, as a cross-check).
- The split is recomputed (grouped_split(leakage_groups(...), seed=42)) and asserted equal to stage (d)'s cache before
  split.held is used; every held-episode row is asserted to be a held row; the episode SHA-256s are recorded.
- The criterion (spec §7) is fixed: confirmed iff the D_emo,held lower bound > 0 AND the D_style,held lower bound
  > -1.5 (picked cell seed 42 vs C0 seed 42, naive rule at beta 0.3, mean of directions, paired bootstrap 5,000
  resamples seed 42). Nothing else decides the verdict; everything else is reported context.
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_AFFECT_PATH = ROOT / "src/test/20261018_affect_factor_learning/run_affect.py"
_aspec = importlib.util.spec_from_file_location("run_affect", _AFFECT_PATH)
affect = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(affect)
grid, sel, probe, fin = affect.grid, affect.grid.sel, affect.grid.probe, affect.grid.fin

import src.data.affect as affect_src  # noqa: E402
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, grouped_subsplit, leakage_groups  # noqa: E402
from src.eval.condition_eval import paired_bootstrap  # noqa: E402
from src.eval.label_episodes import (LabelEpisodes, label_episode_weights,  # noqa: E402
                                     label_episodes_sha256, standard_label_episodes)
from src.train.train_factors import R3_CONFIG, encode_rows, load_factor_checkpoint  # noqa: E402


def _no_affect(*_args, **_kwargs):
    raise AssertionError("affect vectors are never computed in the held test (affect spec §7)")


for _module in (affect_src, affect):                     # structural guard: the GoEmotions path cannot run here
    for _name in ("goemotions_probabilities", "load_goemotions"):
        setattr(_module, _name, _no_affect)

SEED = 42                                  # split, bootstrap, oracle folds; the verdict models' seed
EPISODE_SEED = 43                          # fresh held episodes (spec §7)
REPLICATION_SEEDS = (43, 44)
LABELS, SCOPES, DIRECTIONS = affect.LABELS, affect.SCOPES, affect.DIRECTIONS
BETAS, BETA_FIXED, key = affect.BETAS, affect.BETA_FIXED, affect.key
ORACLE_STEPS, ORACLE_BETAS, NULL_BETA = affect.ORACLE_STEPS, affect.ORACLE_BETAS, affect.NULL_BETA
STYLE_MARGIN = affect.STYLE_MARGIN         # -1.5
Z = 1.959964
R3_SHA256 = "1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f"
N_BOOT = 5000
RESULTS = HERE / "results"
POWER_JSON = RESULTS / "power.json"
SMOKE_JSON, SMOKE_NPZ = RESULTS / "smoke_held.json", RESULTS / "smoke_held_ranks.npz"
HELD_JSON, HELD_NPZ = RESULTS / "held_results.json", RESULTS / "held_ranks.npz"
HELD_STARTED = RESULTS / "held_started.json"
STAGE_D_HELD_CODES = fin.HELD_NPZ           # stage (d)'s held R3 codes (same checkpoint): encoding consistency check
CODE_TOLERANCE = 1e-5
log = affect.log

HELD_ROW_HISTORY = (
    "Held rows were read in two earlier final tests: the repair plan's (Task 7, 2026-10-12 report) and stage (d)'s "
    "(2026-10-14), both on seed-42 held label episodes (1,024 per label). The factor-learning 2x2 never read them. "
    "These seed-43 episodes are new; the rows and paintings are not.")


# ----------------------------------------------------------------------------- Step 1: power (spec §7)

N_CHOICES, SHRINK, TARGET_POWER, N_SELECTION = (2048, 4096, 8192), 0.75, 0.8, 4096


def held_episode_count(d_point: float, d_ci95: list[float]) -> tuple[int, dict]:
    """Affect spec §7: SE from the selection D_emo CI (4,096 episodes), effect shrunk by 0.75, smallest n per label
    with power >= 0.8, else 8,192."""
    se = (d_ci95[1] - d_ci95[0]) / (2 * 1.959964)
    effect = SHRINK * d_point
    table = {n: float(norm.cdf(effect / (se * (N_SELECTION / n) ** 0.5) - 1.959964)) for n in N_CHOICES}
    chosen = next((n for n in N_CHOICES if table[n] >= TARGET_POWER), N_CHOICES[-1])
    return chosen, table


def _check_power() -> dict:
    """The brief's hand check: d 2.0, CI [1.0, 3.0] -> SE 0.5102, effect 1.5, power .547 @ 2,048, .836 @ 4,096."""
    chosen, table = held_episode_count(2.0, [1.0, 3.0])
    se = 2.0 / (2 * 1.959964)
    assert abs(se - 0.5102) < 5e-5, se
    assert abs(table[2048] - 0.547) < 5e-4 and abs(table[4096] - 0.836) < 5e-4, table
    assert chosen == 4096, chosen
    return {"d_point": 2.0, "d_ci95": [1.0, 3.0], "se": se, "effect": 1.5, "table": table, "chosen": chosen}


def _power_record(selection: dict) -> dict:
    picked = selection["rule"]["picked"]
    if picked not in affect.CELLS:
        raise AssertionError(f"selection picked no cell: {selection['rule']}")
    d_emo, d_style = selection["d_emo"][picked], selection["d_style"][picked]
    chosen, table = held_episode_count(d_emo["point"], d_emo["ci95"])
    se_sel = (d_emo["ci95"][1] - d_emo["ci95"][0]) / (2 * Z)
    # Style guard at the chosen n (context, stated before the run): SE scaled like D_emo's.
    se_style_sel = (d_style["ci95"][1] - d_style["ci95"][0]) / (2 * Z)
    se_style_n = se_style_sel * (N_SELECTION / chosen) ** 0.5
    threshold = STYLE_MARGIN + Z * se_style_n
    guard = {"se_style_selection": se_style_sel, "se_style_at_n": se_style_n, "pass_threshold_point": threshold,
             "p_pass_by_true_style_effect": {f"{mu:+.1f}": float(1 - norm.cdf((threshold - mu) / se_style_n))
                                             for mu in (0.0, -0.5, -1.0)}}
    return {"picked": picked, "d_emo_selection": d_emo, "d_style_selection": d_style,
            "se_selection": se_sel, "assumed_effect": SHRINK * d_emo["point"],
            "power_table": {str(n): p for n, p in table.items()},
            "se_at_n": {str(n): se_sel * (N_SELECTION / n) ** 0.5 for n in N_CHOICES},
            "chosen_n_per_label": chosen, "chosen_power": table[chosen], "reached_target": table[chosen] >= TARGET_POWER,
            "settings": {"n_choices": list(N_CHOICES), "shrink": SHRINK, "target_power": TARGET_POWER,
                         "n_selection": N_SELECTION, "z": Z},
            "style_guard_at_n": guard}


def power() -> dict:
    RESULTS.mkdir(exist_ok=True)
    if HELD_JSON.exists() or HELD_STARTED.exists():
        raise RuntimeError("the held run has started or finished: power.json must predate it and is not rewritten")
    hand = _check_power()
    selection = json.loads(affect.SELECTION_JSON.read_text())
    record = {**_power_record(selection), "hand_check": hand,
              "selection_json_sha256": grid.sha256_file(affect.SELECTION_JSON),
              "written_at": datetime.now(timezone.utc).isoformat(),
              "note": "computed from selection_results.json only; no held row read"}
    POWER_JSON.write_text(json.dumps(grid._jsonable(record), indent=2))
    log(f"Power (spec §7): SE_sel {record['se_selection']:.4f}, effect {record['assumed_effect']:.4f}, table "
        f"{record['power_table']} -> n = {record['chosen_n_per_label']} per label "
        f"(power {record['chosen_power']:.3f}); style guard at n: {record['style_guard_at_n']}")
    return record


def _load_power() -> dict:
    if not POWER_JSON.exists():
        raise RuntimeError("run --power first")
    stored = json.loads(POWER_JSON.read_text())
    recomputed = _power_record(json.loads(affect.SELECTION_JSON.read_text()))
    if stored["chosen_n_per_label"] != recomputed["chosen_n_per_label"] or stored["picked"] != recomputed["picked"]:
        raise AssertionError("power.json differs from a recomputation from selection_results.json")
    return stored


# ----------------------------------------------------------------------------- the verdict (spec §7)

def held_verdict(d_emo: dict, d_style: dict) -> dict:
    """Confirmed iff the D_emo,held lower bound > 0 and the D_style,held lower bound > -1.5 (spec §7)."""
    emo_ok, style_ok = d_emo["ci95"][0] > 0.0, d_style["ci95"][0] > STYLE_MARGIN
    return {"confirmed": bool(emo_ok and style_ok), "d_emo_lower_bound_above_0": bool(emo_ok),
            "d_style_lower_bound_above_margin": bool(style_ok), "style_margin": STYLE_MARGIN,
            "rule": "confirmed iff D_emo,held ci95[0] > 0 and D_style,held ci95[0] > -1.5 (picked seed 42 vs C0 "
                    "seed 42, naive beta 0.3, mean of directions, paired bootstrap 5,000 resamples seed 42)"}


def _check_verdict() -> None:
    blk = lambda p, lo: {"point": p, "ci95": [lo, p + 1.0]}                        # noqa: E731
    assert held_verdict(blk(1.0, 0.1), blk(0.0, -1.4))["confirmed"]
    assert not held_verdict(blk(1.0, 0.0), blk(0.0, -1.4))["confirmed"]         # lower bound must be > 0
    assert not held_verdict(blk(1.0, 0.1), blk(-0.5, -1.5))["confirmed"]        # style lower bound must be > -1.5
    assert not held_verdict(blk(-1.0, -2.0), blk(2.0, 1.0))["confirmed"]


# ----------------------------------------------------------------------------- models

def model_table(picked: str) -> dict:
    """name -> checkpoint path, expected SHA-256 (from the selection / replication records) and config."""
    selection = json.loads(affect.SELECTION_JSON.read_text())
    replication = json.loads(affect.REPLICATION_JSON.read_text())
    if replication["picked"] != picked:
        raise AssertionError("replication.json is for another cell")
    table = {}
    for seed in (SEED, *REPLICATION_SEEDS):
        for cell in (picked, "C0"):
            name = f"{cell}_seed{seed}"
            expected = (selection["models"][cell]["sha256"] if seed == SEED
                        else replication["per_seed"][str(seed)]["models"][cell]["model"]["sha256"])
            config_cell = "C0" if cell == "C0" else "S"
            table[name] = {"path": affect.model_path(cell, seed), "expected_sha256": expected, "seed": seed,
                           "cell": cell, "config": dataclasses.asdict(grid.cell_config(config_cell, seed)),
                           "config_name": f"grid.cell_config({config_cell!r}, {seed})", "device": grid.DEVICE}
    prep = json.loads((affect.CACHE / "affect_prepare.json").read_text())
    if table[f"C0_seed{SEED}"]["expected_sha256"] != prep["c0_ref_sha256"]:
        raise AssertionError("C0 seed 42 SHA-256 differs between selection_results.json and affect_prepare.json")
    if sel.R3_SHA256 != R3_SHA256 or Path(sel.R3_PATH) != ROOT / "src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt":
        raise AssertionError("stage (d)'s R3 path / SHA-256 differ from the plan's")
    table["R3"] = {"path": Path(sel.R3_PATH), "expected_sha256": R3_SHA256, "seed": SEED, "cell": "R3",
                   "config": dataclasses.asdict(R3_CONFIG), "config_name": "R3_CONFIG (original R3)",
                   "device": "cpu"}             # CPU, as stage (d) encoded it (bit-identical codes, checked below)
    return table


def load_models(table: dict) -> tuple[dict, dict]:
    """Load every checkpoint; assert its SHA-256 and stored config. Reads no data rows."""
    models, info = {}, {}
    for name, spec in table.items():
        sha = grid.sha256_file(spec["path"])
        if sha != spec["expected_sha256"]:
            raise AssertionError(f"{name}: SHA-256 {sha} differs from the recorded {spec['expected_sha256']}")
        model, config = load_factor_checkpoint(spec["path"], device=spec["device"])
        if dataclasses.asdict(config) != spec["config"]:
            raise AssertionError(f"{name}: stored config differs from {spec['config_name']}")
        models[name] = model
        info[name] = {"checkpoint": str(Path(spec["path"]).relative_to(ROOT)), "sha256": sha,
                      "config": spec["config_name"], "encoded_on": spec["device"]}
    log(f"Checkpoints: SHA-256 and config asserted for {list(models)}")
    return models, info


def encode_only(model, data, rows: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-length codes, finite on ``rows`` only (only these rows are passed to the encoder)."""
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=rows)
    return probe.scatter_rows(ic, rows, n_rows), probe.scatter_rows(tc, rows, n_rows)


# ----------------------------------------------------------------------------- rows and episodes

def recompute_split(data, cache: dict) -> dict:
    """grouped_split(leakage_groups(...), seed=42) and its selection sub-split, asserted equal to stage (d)'s cache."""
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, seed=SEED)
    sizes = (len(split.train), len(split.val), len(split.held))
    if sizes != tuple(sel.EXPECTED_SPLIT):
        raise AssertionError(f"unexpected split sizes {sizes}")
    scorer_train, selection = grouped_subsplit(groups, split.train, sel.SELECTION_FRACTION, seed=SEED)
    for name, mine, cached in (("groups", groups, cache["groups"]), ("split.train", split.train, cache["split_train"]),
                               ("scorer_train", scorer_train, cache["scorer_train"]),
                               ("selection", selection, cache["selection"])):
        if not np.array_equal(mine, cached):
            raise AssertionError(f"recomputed {name} differs from stage (d)'s cache")
    held, val = np.asarray(split.held, dtype=np.int64), np.asarray(split.val, dtype=np.int64)
    if np.intersect1d(held, split.train).size or np.intersect1d(held, val).size:
        raise AssertionError("held rows overlap train or val rows")
    if np.intersect1d(groups[held], groups[split.train]).size or np.intersect1d(groups[held], groups[val]).size:
        raise AssertionError("a leakage group spans held and another split")
    return {"groups": groups, "train": np.asarray(split.train), "held": held, "sizes": sizes,
            "scorer_train": scorer_train, "selection": selection}


def build_episodes(data, groups: np.ndarray, rows: np.ndarray, n: int, seed: int) -> tuple[dict, dict, dict]:
    in_rows = probe.row_mask(len(groups), rows)
    episodes, meta = {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, groups, rows, label, n, seed=seed)
        used = np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])
        if not in_rows[used].all():
            raise AssertionError(f"{label} episodes use rows outside the evaluated rows")
        if eps.distractors.shape[1] + 1 != 13 or len(eps.anchor) != n:
            raise AssertionError("expected n episodes with 13 candidates each")
        prefix = LabelEpisodes(**{f.name: getattr(eps, f.name)[:1024] for f in dataclasses.fields(eps)})
        episodes[label] = eps
        meta[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))),
                       "target_counts": {str(t): int(c) for t, c in zip(*np.unique(eps.labels, return_counts=True))},
                       "sha256": label_episodes_sha256(eps), "prefix_1024_sha256": label_episodes_sha256(prefix),
                       "seed": seed, "all_rows_in_evaluated_rows": True,
                       "distinct_rows_used": int(len(np.unique(used))),
                       "distinct_paintings_used": int(len(np.unique(groups[np.unique(used)])))}
    if LABELS != ("emotion", "art_style"):
        raise AssertionError("null targets are drawn in (emotion, art_style) order")
    rng = np.random.default_rng(seed)
    nulls = {label: rng.integers(1, 13, len(episodes[label].anchor)) for label in LABELS}
    return episodes, nulls, meta


# ----------------------------------------------------------------------------- scoring

def per_target(ranks: dict, episodes: dict, a: str, b: str) -> dict:
    """Per target label: a - b in R@1 (naive at beta 0.3 and 0, oracle at 0), mean of directions, paired bootstrap."""
    out = {}
    for label in LABELS:
        targets = episodes[label].labels
        out[label] = {}
        for col, scorer, beta in (("naive@0.3", "naive", 0.3), ("naive@0", "naive", 0.0), ("oracle@0", "oracle", 0.0)):
            ha = {d: probe.hits(ranks[key(scorer, a, beta)][label][d]) for d in DIRECTIONS}
            hb = {d: probe.hits(ranks[key(scorer, b, beta)][label][d]) for d in DIRECTIONS}
            rows = {}
            for target in np.unique(targets):
                mask = targets == target
                diff = {d: ha[d][mask] - hb[d][mask] for d in DIRECTIONS}
                rows[str(target)] = {
                    "n": int(mask.sum()), b: 100 * float(0.5 * (hb["i2t"][mask] + hb["t2i"][mask]).mean()),
                    a: 100 * float(0.5 * (ha["i2t"][mask] + ha["t2i"][mask]).mean()),
                    "diff": probe.pct_block(paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"]))),
                    **{f"diff_{d}": 100 * float(diff[d].mean()) for d in DIRECTIONS}}
            total = sum(v["n"] * v["diff"]["point"] for v in rows.values())
            for v in rows.values():
                v["share_of_total"] = (v["n"] * v["diff"]["point"] / total) if total else float("nan")
            out[label][col] = rows
    return out


def score_rows(data, groups: np.ndarray, rows: np.ndarray, n: int, episode_seed: int, codes: dict, picked: str,
               full: bool = True) -> dict:
    """Every number of the held phase on ``rows``: naive ranks at beta 0.3 for all models (and the beta grid for the
    picked cell, C0 and R3 at seed 42), CLIP-only, the label oracle and its null (picked, C0, R3), the criterion and
    the reported comparisons. ``full=False`` scores only the naive rule at beta 0.3 (the smoke's reproduction check)."""
    n_rows = len(groups)
    in_rows = probe.row_mask(n_rows, rows)
    img, txt = sel.masked(data.img_features, rows), sel.masked(data.txt_features, rows)
    for name, arr in (("img features", img), ("txt features", txt)):
        probe.assert_row_scope(name, arr, in_rows)
    episodes, nulls, eps_meta = build_episodes(data, groups, rows, n, episode_seed)
    log(f"Episodes (seed {episode_seed}, n {n}): " + "; ".join(
        f"{lab} targets {m['targets']} sha {m['sha256'][:12]}" for lab, m in eps_meta.items()))
    a, b = f"{picked}_seed{SEED}", f"C0_seed{SEED}"
    context = (a, b, "R3")
    ranks, ties, weights_info, spread, scale, timings = {}, {}, {}, {}, {}, {}
    for name, (ic, tc) in codes.items():
        t0 = perf_counter()
        for side, arr in (("img", ic), ("txt", tc)):
            probe.assert_row_scope(f"{name} {side} codes", arr, in_rows)
        weights = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
        betas = BETAS if (full and name in context) else ((0.0, BETA_FIXED) if full else (BETA_FIXED,))
        for beta in betas:
            ranks[key("naive", name, beta)], ties[key("naive", name, beta)] = probe.fixed_weight_ranks(
                img, txt, ic, tc, episodes, weights, beta)
        if full and name in context:
            weights_info[name] = grid.weight_summary(weights)
            spread[name] = grid.term_spread(img, txt, ic, tc, episodes, weights)
            scale[name] = {"mean_rms_evaluated_rows": probe.mean_rms(ic, tc, rows)}
            for beta in ORACLE_BETAS:
                ranks[key("oracle", name, beta)] = probe.oracle_ranks(img, txt, ic, tc, episodes, beta,
                                                                      ORACLE_STEPS, grid.DEVICE)
            ranks[key("oracle_null", name, NULL_BETA)] = probe.oracle_ranks(img, txt, ic, tc, episodes, NULL_BETA,
                                                                            ORACLE_STEPS, grid.DEVICE, nulls)
        if name == b:                                   # CLIP-only: zero weights, beta 0.3 (codes irrelevant)
            zero = {label: torch.zeros(len(episodes[label].anchor), ic.shape[1]) for label in LABELS}
            ranks[key("clip_only", "-", BETA_FIXED)], _ = probe.fixed_weight_ranks(img, txt, ic, tc, episodes, zero,
                                                                                   BETA_FIXED)
        timings[f"model:{name}"] = perf_counter() - t0
    for k, v in ranks.items():
        for label in LABELS:
            for d in DIRECTIONS:
                if not np.isfinite(np.asarray(v[label][d], dtype=np.float64)).all():
                    raise AssertionError(f"{k} {label} {d}: non-finite ranks")

    naive = {name: ranks[key("naive", name, BETA_FIXED)] for name in codes}
    criterion = probe.r1_diff(naive[a], naive[b])
    d_emo, d_style, d_pooled = (criterion[s]["mean"] for s in ("emotion", "art_style", "pooled"))
    out = {"episodes": eps_meta, "d_emo": d_emo, "d_style": d_style, "d_pooled": d_pooled, "criterion_diff": criterion,
           "verdict": held_verdict(d_emo, d_style), "ranks": ranks}
    if not full:
        return out
    t0 = perf_counter()
    seeds = {str(s): probe.r1_diff(naive[f"{picked}_seed{s}"], naive[f"C0_seed{s}"]) for s in (SEED, *REPLICATION_SEEDS)}
    clip = ranks[key("clip_only", "-", BETA_FIXED)]
    out.update({
        "seeds_vs_c0": {s: {"d_emo": v["emotion"]["mean"], "d_style": v["art_style"]["mean"],
                            "d_pooled": v["pooled"]["mean"], "diff": v} for s, v in seeds.items()},
        "seeds_vs_c0_beta0": {str(s): probe.r1_diff(ranks[key("naive", f"{picked}_seed{s}", 0.0)],
                                                    ranks[key("naive", f"C0_seed{s}", 0.0)])
                              for s in (SEED, *REPLICATION_SEEDS)},
        "naive_r1_beta0": {name: probe.r1_points(ranks[key("naive", name, 0.0)]) for name in codes},
        "naive_r1": {name: probe.r1_with_ci(naive[name]) for name in codes},
        "clip_only_r1": probe.r1_with_ci(clip),
        "vs_r3": {name: probe.r1_diff(naive[name], naive["R3"]) for name in codes if name != "R3"},
        "vs_clip_only": {name: probe.r1_diff(naive[name], clip) for name in context},
        "beta_grid_r1": {name: {f"{bt:g}": probe.r1_points(ranks[key("naive", name, bt)]) for bt in BETAS}
                         for name in context},
        "vs_c0_beta_grid": {f"{bt:g}": probe.r1_diff(ranks[key("naive", a, bt)], ranks[key("naive", b, bt)])
                            for bt in BETAS},
        "vs_r3_beta0": {name: probe.r1_diff(ranks[key("naive", name, 0.0)], ranks[key("naive", "R3", 0.0)])
                        for name in (a, b)},
        "oracle_r1": {name: {**{f"oracle_{bt:g}": probe.r1_with_ci(ranks[key("oracle", name, bt)])
                                for bt in ORACLE_BETAS},
                             f"oracle_null_{NULL_BETA:g}": probe.r1_with_ci(ranks[key("oracle_null", name, NULL_BETA)])}
                      for name in context},
        "oracle_minus_c0_oracle": {f"{bt:g}": probe.r1_diff(ranks[key("oracle", a, bt)], ranks[key("oracle", b, bt)])
                                   for bt in ORACLE_BETAS},
        "oracle_minus_r3_oracle": {name: probe.r1_diff(ranks[key("oracle", name, 0.0)], ranks[key("oracle", "R3", 0.0)])
                                   for name in (a, b)},
        "oracle_minus_own_naive": {variant: {name: probe.r1_diff(ranks[key("oracle", name, ob)],
                                                                 ranks[key("naive", name, nb)]) for name in context}
                                   for variant, ob, nb in (("same_beta_0.3", 0.3, 0.3), ("same_beta_0", 0.0, 0.0))},
        "per_target": per_target(ranks, episodes, a, b),
        "naive_weights": weights_info, "term_spread_beta0.3": spread, "code_scale": scale,
        "naive_tied_episodes": ties, "r1": {k: probe.r1_points(v) for k, v in ranks.items()},
        "chance_r1": probe.CHANCE_R1})
    timings["comparisons"] = perf_counter() - t0
    out["timings_seconds"] = timings
    return out


# ----------------------------------------------------------------------------- the phase (smoke or held)

def _script_sha256() -> str:
    return grid.sha256_file(Path(__file__).resolve())


def _check_smoke_before_run(n: int) -> dict:
    if not SMOKE_JSON.exists():
        raise RuntimeError("run --smoke first: the held run needs a passing smoke of this exact script")
    smoke = json.loads(SMOKE_JSON.read_text())
    meta = smoke["meta"]
    if not meta.get("passed"):
        raise RuntimeError("the last smoke did not pass")
    if meta["n_per_label"] != n or meta["episode_seed"] != EPISODE_SEED:
        raise RuntimeError("the smoke ran with another n or episode seed")
    if meta["script_sha256"] != _script_sha256():
        raise RuntimeError("run_held.py changed after the smoke: rerun --smoke before --run")
    return {"smoke_json_sha256": grid.sha256_file(SMOKE_JSON), "smoke_finished_at": meta["finished_at"],
            "smoke_script_sha256": meta["script_sha256"]}


def held_phase(smoke: bool, after_crash: bool = False) -> dict:
    _check_verdict()
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    out_json, out_npz = (SMOKE_JSON, SMOKE_NPZ) if smoke else (HELD_JSON, HELD_NPZ)
    pw = _load_power()
    n, picked = pw["chosen_n_per_label"], pw["picked"]
    if pw["written_at"] >= started_at:
        raise AssertionError("power.json must predate this phase")
    attempts = []
    if smoke:
        if HELD_JSON.exists() or HELD_STARTED.exists():
            raise RuntimeError("the held run has started or finished: no smoke afterwards")
        preconditions = {}
    else:
        if HELD_JSON.exists():
            raise RuntimeError(f"{HELD_JSON} exists: the held test is single-use; use --tables to reprint it")
        if HELD_STARTED.exists():
            previous = json.loads(HELD_STARTED.read_text())
            if not after_crash:
                raise RuntimeError(f"a held run started at {previous['attempts'][-1]['started_at']} and did not "
                                   f"finish; held rows may have been read. Rerun only with --after-crash (recorded).")
            attempts = previous["attempts"]
        preconditions = _check_smoke_before_run(n)
    table = model_table(picked)
    models, model_info = load_models(table)         # no data rows read yet
    if not smoke:
        attempts = attempts + [{"started_at": started_at, "power_written_at": pw["written_at"], **preconditions}]
        HELD_STARTED.write_text(json.dumps({"attempts": attempts}, indent=2))
    log(f"{'SMOKE (selection rows in place of held rows; numbers discarded)' if smoke else 'HELD TEST'}: "
        f"picked {picked}, n {n} per label, episode seed {EPISODE_SEED}")

    t0 = perf_counter()
    data = load_artelingo()
    cache, stage_d_meta = sel.load_prepared()
    split = recompute_split(data, cache)
    groups, n_rows = split["groups"], len(split["groups"])
    rows = cache["selection"] if smoke else split["held"]
    if np.intersect1d(rows, cache["scorer_train"]).size:
        raise AssertionError("evaluated rows overlap scorer-train rows (the new models' training rows)")
    if not smoke and np.intersect1d(rows, cache["split_train"]).size:
        raise AssertionError("held rows overlap the train part (original R3's training rows)")
    load_seconds = perf_counter() - t0
    log(f"Split recomputed and equal to stage (d)'s cache {split['sizes']}; evaluating {len(rows):,} "
        f"{'selection' if smoke else 'held'} rows; everything else NaN")

    # Encode ONLY the evaluated rows.
    t0 = perf_counter()
    codes = {name: encode_only(models[name], data, rows, n_rows) for name in table}
    if smoke:
        ref_img, ref_txt = cache["img_codes"][rows], cache["txt_codes"][rows]
        ref_source = "stage (d) cache img_codes / txt_codes (selection rows)"
    else:
        ref = np.load(STAGE_D_HELD_CODES)
        if not np.array_equal(ref["held_rows"], rows):
            raise AssertionError("stage (d)'s held-code rows differ from the recomputed held rows")
        ref_img, ref_txt, ref_source = ref["img_codes"], ref["txt_codes"], "stage (d) final cache/held_codes.npz"
    r3_diff = max(float(np.abs(codes["R3"][0][rows] - ref_img).max()), float(np.abs(codes["R3"][1][rows] - ref_txt).max()))
    if r3_diff > CODE_TOLERANCE:
        raise AssertionError(f"R3 codes differ from {ref_source} by {r3_diff}")
    r3_check = {"reference": ref_source, "max_abs_diff": r3_diff,
                "bit_identical": bool(np.array_equal(codes["R3"][0][rows], ref_img)
                                      and np.array_equal(codes["R3"][1][rows], ref_txt))}
    encode_seconds = perf_counter() - t0
    log(f"Encoded {len(table)} models on {len(rows):,} rows only ({encode_seconds:.1f} s); R3 vs {ref_source}: "
        f"max |diff| {r3_diff:.3g}")

    t0 = perf_counter()
    scored = score_rows(data, groups, rows, n, EPISODE_SEED, codes, picked, full=True)
    score_seconds = perf_counter() - t0
    ranks = scored.pop("ranks")
    if not smoke:
        for label, m in scored["episodes"].items():
            if m["sha256"] == fin.HELD_SHA256[label] or m["prefix_1024_sha256"] == fin.HELD_SHA256[label]:
                raise AssertionError(f"{label}: these held episodes equal the earlier final tests' episodes")
    finite = bool(all(np.isfinite(v["point"]) and np.isfinite(v["ci95"]).all()
                      for v in (scored["d_emo"], scored["d_style"], scored["d_pooled"])))
    if not finite:
        raise AssertionError("non-finite criterion values")

    reproduction = None
    if smoke:                  # the same code on the selection episodes (seed 42, 4,096) must give the selection D
        t0 = perf_counter()
        sub = {name: codes[name] for name in (f"{picked}_seed{SEED}", f"C0_seed{SEED}")}
        rep = score_rows(data, groups, rows, N_SELECTION, SEED, sub, picked, full=False)
        stored = json.loads(affect.SELECTION_JSON.read_text())
        stored_ranks = np.load(affect.SELECTION_NPZ)
        share = {name: min(float(np.mean(np.asarray(rep["ranks"][key("naive", name, BETA_FIXED)][label][d])
                                         == stored_ranks[f"naive__{cell}__0.3__{label}__{d}"]))
                           for label in LABELS for d in DIRECTIONS)
                 for name, cell in ((f"{picked}_seed{SEED}", picked), (f"C0_seed{SEED}", "C0"))}
        reproduction = {"identical_rank_share": share,
                        "episode_sha256_equal": {lab: rep["episodes"][lab]["sha256"] == stored["episodes"][lab]["sha256"]
                                                 for lab in LABELS},
                        "d_emo": rep["d_emo"], "d_style": rep["d_style"],
                        "d_emo_selection": stored["d_emo"][picked], "d_style_selection": stored["d_style"][picked],
                        "d_point_abs_diff": max(abs(rep["d_emo"]["point"] - stored["d_emo"][picked]["point"]),
                                                abs(rep["d_style"]["point"] - stored["d_style"][picked]["point"])),
                        "seconds": perf_counter() - t0}
        if not all(reproduction["episode_sha256_equal"].values()):
            raise AssertionError("the held code's selection episodes differ from run_affect's")
        if min(share.values()) < grid.IDENTICAL_SHARE_MIN or reproduction["d_point_abs_diff"] > 0.05:
            raise AssertionError(f"the held code does not reproduce the selection criterion: {reproduction}")
        log(f"Reproduction on the selection episodes (seed 42, 4,096): D_emo {affect._ci(rep['d_emo'])} vs stored "
            f"{affect._ci(stored['d_emo'][picked])}; D_style {affect._ci(rep['d_style'])} vs stored "
            f"{affect._ci(stored['d_style'][picked])}; identical-rank share {share}")

    finished_at = datetime.now(timezone.utc).isoformat()
    results = {
        "label": ("SMOKE: the held code path on SELECTION rows (seed-43 episodes); numbers DISCARDED" if smoke else
                  "affect held test (spec §7), run once; picked cell seed 42 vs C0 seed 42 decides; all else context"),
        "picked": picked, **scored,
        "models": model_info,
        "power": {k: pw[k] for k in ("chosen_n_per_label", "chosen_power", "power_table", "se_at_n", "se_selection",
                                     "assumed_effect", "written_at", "style_guard_at_n")},
        "meta": {"smoke": smoke, "rows_evaluated": "selection (smoke)" if smoke else "held",
                 "n_rows_evaluated": int(len(rows)), "n_per_label": n, "episode_seed": EPISODE_SEED,
                 "split_sizes": split["sizes"], "split_equals_stage_d_cache": True,
                 "held_rows_disjoint_from_train_and_val": True, "rows_disjoint_from_scorer_train": True,
                 "codes_encoded_on_evaluated_rows_only": True, "affect_vectors_computed": False,
                 "r3_code_check": r3_check, "reproduction_on_selection_episodes": reproduction,
                 "criterion_finite": finite, "passed": True,
                 "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "unit": "episode (paired)"},
                 "oracle": {**probe.ORACLE_SETTINGS, "steps": ORACLE_STEPS, "betas": list(ORACLE_BETAS),
                            "null_beta": NULL_BETA, "null_targets": f"default_rng({EPISODE_SEED}).integers(1, 13, n)"},
                 "held_row_history": HELD_ROW_HISTORY,
                 "earlier_held_episode_sha256": fin.HELD_SHA256, "attempts": attempts if not smoke else None,
                 "preconditions": preconditions, "script_sha256": _script_sha256(),
                 "started_at": started_at, "finished_at": finished_at, "device": grid.DEVICE,
                 "seconds": {"load_and_split": load_seconds, "encode": encode_seconds, "score": score_seconds,
                             "total": perf_counter() - started}}}
    out_json.write_text(json.dumps(grid._jsonable(results), indent=2))
    np.savez(out_npz, **{f"{k.replace('|', '__')}__{label}__{d}": np.asarray(v[label][d])
                         for k, v in ranks.items() for label in LABELS for d in DIRECTIONS})
    v = scored["verdict"]
    log(f"{'SMOKE (discarded)' if smoke else 'HELD'}: D_emo {affect._ci(scored['d_emo'])}, D_style "
        f"{affect._ci(scored['d_style'])}, pooled {affect._ci(scored['d_pooled'])} -> "
        f"{'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}{' (smoke numbers discarded)' if smoke else ''}")
    log(f"Phase done in {results['meta']['seconds']['total']:.1f} s -> {out_json}")
    return results


# ----------------------------------------------------------------------------- tables

def tables(path: Path = HELD_JSON) -> None:
    res = json.loads(path.read_text())
    picked = res["picked"]
    a, b = f"{picked}_seed{SEED}", f"C0_seed{SEED}"
    _ci, _r1ci = affect._ci, affect._r1ci
    m = res["meta"]
    out = [f"All numbers: {res['label']}.",
           f"Rows: {m['n_rows_evaluated']:,} {m['rows_evaluated']}; {m['n_per_label']} episodes per label, seed "
           f"{m['episode_seed']}; " + "; ".join(f"{lab} targets {e['targets']} sha {e['sha256']} (first 1,024 "
                                                  f"{e['prefix_1024_sha256'][:12]})" for lab, e in res["episodes"].items()),
           f"Started {m['started_at']}, finished {m['finished_at']}; power written {res['power']['written_at']}.\n"]
    pw = res["power"]
    out.append("### Power (spec §7, stated before the run)\n")
    out.append("| n per label | SE (points) | power |")
    out.append("|---:|---:|---:|")
    out += [f"| {n} | {pw['se_at_n'][n]:.3f} | {100 * p:.1f}% |" for n, p in pw["power_table"].items()]
    out.append(f"\nSE_sel {pw['se_selection']:.4f}; assumed effect {pw['assumed_effect']:.4f}; chosen n "
               f"{pw['chosen_n_per_label']} (power {100 * pw['chosen_power']:.1f}%). Style guard at n: "
               f"{pw['style_guard_at_n']}\n")

    v = res["verdict"]
    out.append("### Criterion (spec §7): " + f"{picked} seed 42 - C0 seed 42 (naive, beta 0.3; paired R@1 points)\n")
    out.append("| Scope | mean | i2t | t2i | bar |")
    out.append("|---|---:|---:|---:|---|")
    for scope, bar in (("emotion", "lower bound > 0"), ("art_style", "lower bound > -1.5"), ("pooled", "context")):
        c = res["criterion_diff"][scope]
        out.append(f"| {scope} | {_ci(c['mean'])} | {_ci(c['i2t'])} | {_ci(c['t2i'])} | {bar} |")
    out.append(f"\n**{'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}** (D_emo lower bound > 0: "
               f"{v['d_emo_lower_bound_above_0']}; D_style lower bound > -1.5: {v['d_style_lower_bound_above_margin']})\n")
    if m.get("reproduction_on_selection_episodes"):
        rp = m["reproduction_on_selection_episodes"]
        out.append(f"Smoke reproduction (selection episodes, seed 42, 4,096): D_emo {_ci(rp['d_emo'])} vs stored "
                   f"{_ci(rp['d_emo_selection'])}; D_style {_ci(rp['d_style'])} vs stored "
                   f"{_ci(rp['d_style_selection'])}; identical-rank share {rp['identical_rank_share']}\n")

    out.append("### Naive R@1 (%) at beta 0.3, 95% CI\n")
    out.append("| Model | " + " | ".join(f"{s} {d}" for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append("|---|" + "---:|" * (3 * len(SCOPES)))
    for name, h in (*res["naive_r1"].items(), ("clip_only", res["clip_only_r1"])):
        out.append(f"| {name} | " + " | ".join(_r1ci(h[s][d]) for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append(f"\nChance {res['chance_r1']:.2f}.\n")

    out.append(f"### {picked} - C0 per seed (same-seed C0; naive beta 0.3; reported, not gating)\n")
    out.append("| Seed | D_emo | D_emo i2t | D_emo t2i | D_style | D_style i2t | D_style t2i | pooled |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s, blk in res["seeds_vs_c0"].items():
        d = blk["diff"]
        out.append(f"| {s}{' (criterion)' if int(s) == SEED else ''} | {_ci(blk['d_emo'])} | {_ci(d['emotion']['i2t'])} | "
                   f"{_ci(d['emotion']['t2i'])} | {_ci(blk['d_style'])} | {_ci(d['art_style']['i2t'])} | "
                   f"{_ci(d['art_style']['t2i'])} | {_ci(blk['d_pooled'])} |")
    out.append("\n| Seed | D_emo at beta 0 | D_style at beta 0 | pooled at beta 0 |")
    out.append("|---|---:|---:|---:|")
    for s, d in res["seeds_vs_c0_beta0"].items():
        out.append(f"| {s} | {_ci(d['emotion']['mean'])} | {_ci(d['art_style']['mean'])} | {_ci(d['pooled']['mean'])} |")
    out.append("\n| Model | naive beta 0: pooled / emotion / style |")
    out.append("|---|---|")
    for name, r in res["naive_r1_beta0"].items():
        out.append(f"| {name} | {r['pooled']['mean']:.2f} / {r['emotion']['mean']:.2f} / {r['art_style']['mean']:.2f} |")
    out.append("")

    out.append("### Context: naive(model, 0.3) - naive(original R3, 0.3) and - CLIP-only (paired R@1 points)\n")
    out.append("| Model | vs R3 pooled | vs R3 emotion | vs R3 style | vs CLIP-only pooled | vs CLIP-only emotion | "
               "vs CLIP-only style |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in (a, b, "R3"):
        r3 = res["vs_r3"].get(name)
        cl = res["vs_clip_only"][name]
        r3s = [_ci(r3[s]["mean"]) for s in SCOPES] if r3 else ["(reference)"] * 3
        out.append(f"| {name} | " + " | ".join(r3s) + " | " + " | ".join(_ci(cl[s]["mean"]) for s in SCOPES) + " |")
    out.append("")

    out.append(f"### {picked} - C0 (seed 42) on the beta grid (naive; paired R@1 points; context)\n")
    out.append("| beta | pooled | emotion | emotion i2t | emotion t2i | art style | style i2t | style t2i |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bt, d in res["vs_c0_beta_grid"].items():
        out.append(f"| {bt} | {_ci(d['pooled']['mean'])} | {_ci(d['emotion']['mean'])} | {_ci(d['emotion']['i2t'])} | "
                   f"{_ci(d['emotion']['t2i'])} | {_ci(d['art_style']['mean'])} | {_ci(d['art_style']['i2t'])} | "
                   f"{_ci(d['art_style']['t2i'])} |")
    out.append("\n| Model | " + " | ".join(f"beta {bt}" for bt in res["beta_grid_r1"][a]) + " |  (pooled / emotion / style)")
    out.append("|---|" + "---|" * len(res["beta_grid_r1"][a]))
    for name, per in res["beta_grid_r1"].items():
        out.append(f"| {name} | " + " | ".join(f"{r['pooled']['mean']:.2f} / {r['emotion']['mean']:.2f} / "
                                               f"{r['art_style']['mean']:.2f}" for r in per.values()) + " |")
    out.append("\n| Model vs R3 at beta 0 | pooled | emotion | style |")
    out.append("|---|---:|---:|---:|")
    for name, d in res["vs_r3_beta0"].items():
        out.append(f"| {name} | {_ci(d['pooled']['mean'])} | {_ci(d['emotion']['mean'])} | {_ci(d['art_style']['mean'])} |")
    out.append("")

    out.append("### Code scale and term spread at beta 0.3 (evaluated rows)\n")
    out.append("| Model | mean RMS | spread cos | spread factor | factor / (0.3 cos) | all-zero weights | mean non-zero |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in res["term_spread_beta0.3"]:
        s, t, w = res["code_scale"][name], res["term_spread_beta0.3"][name], res["naive_weights"][name]
        out.append(f"| {name} | {s['mean_rms_evaluated_rows']:.4f} | {t['cos_spread']:.4f} | {t['factor_spread']:.4f} | "
                   f"{t['factor_to_clip']:.2f} | {100 * w['all_zero_share']:.2f}% | {w['mean_nonzero_factors']:.1f} |")
    out.append("")

    out.append("### Label oracle (%, mean of directions, 95% CI) and its null\n")
    out.append("| Model | naive 0.3 pooled | oracle 0.3 pooled | oracle 0 pooled | oracle 0 emotion | oracle 0 style | "
               "null 0 pooled | oracle - own naive, same 0.3 | same 0 (pooled) | same 0 (emotion) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, o in res["oracle_r1"].items():
        on = res["oracle_minus_own_naive"]
        out.append(f"| {name} | {_r1ci(res['naive_r1'][name]['pooled']['mean'])} | {_r1ci(o['oracle_0.3']['pooled']['mean'])} | "
                   f"{_r1ci(o['oracle_0']['pooled']['mean'])} | {_r1ci(o['oracle_0']['emotion']['mean'])} | "
                   f"{_r1ci(o['oracle_0']['art_style']['mean'])} | {_r1ci(o['oracle_null_0']['pooled']['mean'])} | "
                   f"{_ci(on['same_beta_0.3'][name]['pooled']['mean'])} | {_ci(on['same_beta_0'][name]['pooled']['mean'])} | "
                   f"{_ci(on['same_beta_0'][name]['emotion']['mean'])} |")
    out.append(f"\n| {picked} - C0, oracle | pooled | emotion | emotion i2t | emotion t2i | art style |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for bt, d in res["oracle_minus_c0_oracle"].items():
        out.append(f"| beta {bt} | {_ci(d['pooled']['mean'])} | {_ci(d['emotion']['mean'])} | {_ci(d['emotion']['i2t'])} | "
                   f"{_ci(d['emotion']['t2i'])} | {_ci(d['art_style']['mean'])} |")
    for name, d in res["oracle_minus_r3_oracle"].items():
        out.append(f"| {name} - R3, oracle beta 0 | {_ci(d['pooled']['mean'])} | {_ci(d['emotion']['mean'])} | "
                   f"{_ci(d['emotion']['i2t'])} | {_ci(d['emotion']['t2i'])} | {_ci(d['art_style']['mean'])} |")
    out.append("")

    for label, per in res["per_target"].items():
        cols = list(per)
        out.append(f"### Per-target {picked} - C0, {label} (mean of directions)\n")
        out.append("| Target | n | " + " | ".join(f"{c}: C0 / {picked} / diff [CI] / i2t, t2i" for c in cols) + " |")
        out.append("|---|---:|" + "---|" * len(cols))
        for t in sorted(per[cols[0]], key=lambda t: per["naive@0.3"][t]["diff"]["point"]):
            cells = [f"{per[c][t][b]:.1f} / {per[c][t][a]:.1f} / {_ci(per[c][t]['diff'])} / "
                     f"{per[c][t]['diff_i2t']:+.1f}, {per[c][t]['diff_t2i']:+.1f}" for c in cols]
            out.append(f"| {t} | {per[cols[0]][t]['n']} | " + " | ".join(cells) + " |")
        out.append("")
    tie = res["naive_tied_episodes"]
    out.append("Episodes with any tie at beta 0.3 (emotion i2t/t2i, style i2t/t2i): " + "; ".join(
        f"{name} {tie[key('naive', name, BETA_FIXED)]['emotion']['i2t']}/{tie[key('naive', name, BETA_FIXED)]['emotion']['t2i']}, "
        f"{tie[key('naive', name, BETA_FIXED)]['art_style']['i2t']}/{tie[key('naive', name, BETA_FIXED)]['art_style']['t2i']}"
        for name in res["naive_r1"]))
    out.append(f"\nModels: " + "; ".join(f"{k} {v['sha256'][:12]} ({v['config']}, {v['encoded_on']})"
                                        for k, v in res["models"].items()))
    out.append(f"R3 code check: {m['r3_code_check']}")
    out.append(f"Timings (s): {m['seconds']}; per model {res['timings_seconds']}")
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--after-crash", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if sum((args.power, args.run, args.tables)) + (args.smoke and not args.tables) != 1:
        raise SystemExit("choose exactly one of --power, --smoke, --run, --tables [--smoke]")
    if args.power:
        power()
    elif args.run:
        held_phase(smoke=False, after_crash=args.after_crash)
        tables(HELD_JSON)
    elif args.tables:
        tables(SMOKE_JSON if args.smoke else HELD_JSON)
    else:
        held_phase(smoke=True)
        tables(SMOKE_JSON)


if __name__ == "__main__":
    main()
