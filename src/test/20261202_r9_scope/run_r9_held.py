"""Held runner for r9 H6 (Qwen) and H7 (SigLIP 2), DECISION_RULE sections 4.1, 7 and 10.

Ledger first, pinned inputs, local head refits, started file and its folder copy, H5's episodes,
sensitivity, primary arrays and pooled P1 to P7. Writes no verdict and prints no metric.
The run chat commits the started copy. Smoke uses selection rows, seeds 9001 to 9003 and
64 episodes per pair under results/smoke; ticket 22 writes the smoke record.

    python run_r9_held.py --backbone clip_b32 --read H6 --smoke
"""
import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # before modules that import src: earlier rounds come from MAIN
import r9_backbone as BB
import r9_bundle as B
import r9_context as X
import r9_episodes as E
import r9_heads as H
import r9_picks as P
import r9_pins as PINS
import r9_score as S
import r9_stats as ST
import numpy as np
from src.eval.aspect_metrics import CONDITIONS, METRICS

LEDGER = R.MAIN / "docs/superpowers/held_ledger.md"
LEDGER_NAMES = BB.LEDGER_NAMES
LEDGER_CELLS = ("#", "Date", "Dataset and split", "Backbone", "Purpose",
                "Episode / data SHA-256", "Script SHA-256", "Report")
BACKBONE_CELL, EPISODE_CELL, SCRIPT_CELL, REPORT_CELL = 3, 5, 6, 7
READ_BACKBONES = {"H6": "qwen3vl_emb_2b", "H7": "siglip2_so400m"}
PENDING = "(pending)"
MAX_ATTEMPTS = 2
READ_DEADLINE = date(2026, 10, 18)
EXIT_REFUSE = 4
GIT = "git"
HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
SUBDIR_RE = re.compile(r"[A-Za-z0-9_]+")
READ_SCORERS = S.PRIMARY_SCORERS
READ_KEYS = READ_SCORERS + S.EXTRA_KEYS
SIGMA_KEYS = ("sigma_a2", "sigma_eps2")
SMOKE_RECORDS = {k: "smoke_record" + ("" if k == "smoke" else "_" + k) + ".json"
                 for k in ("smoke", "fix1", "crash1", "reserve")}
VALUE_SETS_PATH = R.R6D / "results/value_sets.json"
VALUE_SETS_SHA = "4ea6d33853e5516adc57a868194a03ab20963bcfe6f646df355a7e72d1970369"
SIGMA_CLIP_SHA = "bc4b0a92d376b2fdf47da2aa08303a5bb2266a29daf85621ca97ec753bef5661"


class Refused(Exception):
    """The run refuses to start (exit 4): a precondition of the rule is not met."""


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _refuse_unless(cond, msg):
    if not cond:
        raise Refused(msg)


def _read_json(path):
    return json.loads(Path(path).read_text())


def runner_sha256() -> str:
    return R.sha256_file(Path(__file__).resolve())


def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def _plain(x):
    """The value as JSON reads it back (keys strings, tuples lists)."""
    return json.loads(json.dumps(_norm(x)))


def amsterdam_today() -> date:
    """Today's date in Europe/Amsterdam: the time-box's clock (a module-level function, so that a test sets it)."""
    return datetime.now(ZoneInfo("Europe/Amsterdam")).date()


def read_attempts(path) -> list:
    """The attempts of a started file (this rule's), each with its flags and its per-seed episode hashes."""
    path = Path(path)
    try:
        rec = _read_json(path)
    except (OSError, ValueError) as e:
        raise Refused(f"{path.name} cannot be read ({type(e).__name__}): the user decides")
    rec = rec if isinstance(rec, dict) else {}
    att = rec.get("attempts")
    _refuse_unless(rec.get("rule_sha256") == R.RULE_SHA256 and isinstance(att, list) and att
                   and all(isinstance(a, dict) and isinstance(a.get("flags"), dict)
                           and isinstance(a.get("episodes_sha256"), dict) for a in att),
                   f"{path.name} is not a started file of this rule: the user decides")
    return att


def attempt_hashes(attempts) -> dict:
    """{seed: {pair: sha}} recorded over the attempts (each seed once per attempt); two attempts that disagree on a
    seed refuse (an --after-crash rerun asserted them equal)."""
    out = {}
    for a in attempts:
        for s, h in a["episodes_sha256"].items():
            h = {p: str(h[p]) for p in R.PAIR_NAMES} if isinstance(h, dict) and set(h) == set(R.PAIR_NAMES) else None
            _refuse_unless(h is not None, f"attempt {a.get('attempt')}: seed {s} lacks a pair's episode hash")
            _refuse_unless(out.setdefault(int(s), h) == h, f"the attempts disagree on seed {s}'s episode hashes")
    return out


def _complete(hashes) -> bool:
    return set(hashes) == set(R.HELD_SEEDS)


def git_provenance(here=None) -> dict:
    """Record the checkout commit and src status; a git error is provenance, never permission to reread."""
    root = Path(here or HERE).resolve().parents[2]
    out = {}
    for key, args in (("head", ("rev-parse", "HEAD")), ("status_src", ("status", "--porcelain", "--", "src"))):
        try:
            p = subprocess.run([GIT, "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True,
                               timeout=10)
            out[key] = (p.stdout.rstrip("\n") if p.returncode == 0
                        else f"error: exit {p.returncode}: {p.stderr.strip()}")
        except (OSError, subprocess.SubprocessError) as e:
            out[key] = f"error: {type(e).__name__}: {e}"
    return out


def _write_bytes(path, data: bytes):
    """Atomic: a .partial file beside ``path``, then os.replace, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _json_bytes(rec) -> bytes:
    return json.dumps(rec, indent=1).encode()


def _npz_bytes(arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def write_started(path, copy_to, rec):
    """The started file and its copy (this folder's held_started_<Hn>.json, committed by the run chat), same bytes."""
    data = _json_bytes(rec)
    _write_bytes(path, data)
    if copy_to is not None:
        _write_bytes(copy_to, data)


def start_attempt(path, copy_to, attempts, flags, record, overwrite=False) -> int:
    """Step 5, the read's first write: the earlier attempts (re-read, asserted unchanged since step 1) plus this one
    (Amsterdam time, flags, SHA-256s, coefficient SHA-256s, an empty per-seed episode record). -> its number. The
    smoke (``overwrite``) starts a new file whatever an earlier smoke left."""
    path = Path(path)
    if not overwrite:
        now = read_attempts(path) if path.exists() else []
        _require(_plain(now) == _plain(attempts), f"{path.name} changed since the refusal check")
    k = len(attempts) + 1
    att = {"attempt": k, "time": R.amsterdam_now(), "flags": dict(flags), **record, "episodes_sha256": {}}
    write_started(path, copy_to, {"rule_sha256": R.RULE_SHA256, "attempts": [*attempts, att]})
    return k


def append_hashes(path, copy_to, attempt, seed, hashes):
    """One seed's per-pair episode SHA-256s appended to this attempt in the started file (and its copy)."""
    rec = _read_json(path)
    att = rec["attempts"][-1]
    _require(att["attempt"] == attempt and str(seed) not in att["episodes_sha256"],
             f"{Path(path).name}: attempt {attempt} is not the last, or seed {seed} is recorded twice")
    att["episodes_sha256"][str(seed)] = dict(hashes)
    write_started(path, copy_to, rec)


def write_new(path, data: bytes, overwrite=False):
    """A results file of the read: never overwritten (rule section 7.3), except by the smoke."""
    _require(overwrite or not Path(path).exists(), f"{path} exists: a results file is never overwritten (rule section "
                                                   f"7.3)")  # guard:never_overwrite
    _write_bytes(path, data)


def keep_or_write_json(path, rec, same, overwrite=False) -> str:
    """Write ``rec``; if an earlier attempt wrote the file (and this is not the smoke), its ``same`` keys must equal
    this attempt's instead (rule section 7.3: never overwritten). -> "written" or "kept"."""
    path = Path(path)
    if path.exists() and not overwrite:
        old = _read_json(path)
        diff = [k for k in same if old.get(k) != _plain(rec[k])]
        _require(not diff, f"{path.name}, written by an earlier attempt, differs from this attempt's in {diff}: the "
                           f"user decides")  # guard:kept_json
        return "kept"
    _write_bytes(path, _json_bytes(rec))
    return "written"


def keep_or_write_npz(path, arrays, overwrite=False) -> str:
    """As keep_or_write_json for an npz: an earlier attempt's file must hold the same keys and equal arrays (shape,
    dtype, values)."""
    path = Path(path)
    if path.exists() and not overwrite:
        with np.load(path) as z:
            same = set(z.files) == set(arrays) and all(array_item(arrays[k], z[k])["equal"] for k in arrays)
        _require(same, f"{path.name}, written by an earlier attempt, differs from this attempt's arrays: the user "
                       f"decides")  # guard:kept_npz
        return "kept"
    _write_bytes(path, _npz_bytes(arrays))
    return "written"


def array_item(got, want) -> dict:
    """An exact comparison of arrays: shape, dtype and every value."""
    g, w = np.asarray(got), np.asarray(want)
    out = {"equal": bool(g.shape == w.shape and g.dtype == w.dtype and np.array_equal(g, w)),
           "got": f"array {g.shape} {g.dtype}", "want": f"array {w.shape} {w.dtype}"}
    return out


def read_seed_bundle(env, mode, seed, n_per_pair, backbone, on_episodes=None):
    """seed_bundle with the row scope of the read checked: every A0 posterior of the bundle is finite on exactly the
    split's rows of ``mode`` (held, or selection) and NaN elsewhere, so r9_heads.predict ran in its own call on
    exactly that row set, never on a union (ticket 03's review)."""
    rows = np.asarray(env.split.held if mode == "held" else env.split.selection)
    in_rows = np.zeros(R.N_ROWS, dtype=bool)
    in_rows[rows] = True
    bundle = seed_bundle(env, mode, seed, n_per_pair, backbone, on_episodes=on_episodes)
    _require(bundle.mode == mode and int(bundle.seed) == int(seed), f"a bundle of {bundle.mode} seed {bundle.seed}, "
                                                                    f"not {mode} seed {seed}")
    for h, sides in bundle.post.items():
        for side, p in sides.items():
            _require(np.array_equal(np.isfinite(np.asarray(p)).all(axis=1), in_rows),
                     f"posteriors {h}/{side} are not finite on exactly split.{mode} (rule section 5 item "
                     f"5)")  # guard:post_rows
    return bundle


class EpisodeRecorder:
    """RowContext's on_episodes for one read (rule sections 4.1 and 7.4). Per seed, right after its
    episodes exist and before cosine or any posterior, in this order: the per-pair SHA-256s appended to this attempt
    in the started file; compared with the hashes recorded before (Refused on a difference); assert_distinct over
    every seed of the read so far against the non-smoke hashes read in step 3; the episode file saved, or compared
    when an earlier attempt saved it."""

    def __init__(self, started, copy_to, attempt, want, recorded, out, overwrite=False, *, read, h5=None):
        self.started, self.copy_to, self.attempt = Path(started), copy_to, attempt
        self.want, self.recorded, self.out, self.overwrite = dict(want), recorded, Path(out), overwrite
        self.read, self.h5 = read, h5
        self.hashes, self.expected = {}, None

    def expect(self, seed):
        self.expected = int(seed)

    def __call__(self, eps):
        s = int(eps.seed)
        _require(s == self.expected and s not in self.hashes, f"episodes of seed {s}, expected seed {self.expected}")
        h = {p: str(eps.sha[p]) for p in R.PAIR_NAMES}
        append_hashes(self.started, self.copy_to, self.attempt, s, h)
        if s in self.want:
            _refuse_unless(h == self.want[s], f"seed {s}: the episode SHA-256s differ from those recorded before "
                                              f"(rule section 7.3): the user decides")  # guard:rerun_hashes
        if self.h5 is not None:
            _refuse_unless(h == self.h5.get(s), f"seed {s}: rebuilt episodes differ from ledger H5")  # guard:h5_hashes
        self.hashes[s] = h
        E.assert_distinct(dict(self.hashes), recorded=self.recorded)
        path = self.out / episodes_name(self.read, s)
        if path.exists() and not self.overwrite:
            got = E.load_episodes(path)
            _require(got.seed == s and got.sha == h, f"{path.name}, saved by an earlier attempt, holds other "
                                                     f"episodes: the user decides")  # guard:episode_file
        else:
            E.save_episodes(path, eps)


def sensitivity_held(bundles, seeds, sigma, n_per_pair) -> dict:
    """R6's sensitivity writer restricted to P1 to P7 (rule 7.4).
    ST.detectable uses ST.Z_P and ST.Z_95; no second copy of either z value lives here.
    """
    cl = np.concatenate([np.asarray(b.cl) for b in bundles])
    n = len(seeds) * len(R.PAIRS) * int(n_per_pair)
    _require(len(bundles) == len(seeds) and cl.shape == (n,), f"{cl.shape} pooled anchors, not ({n},)")
    _, m_p = np.unique(cl, return_counts=True)
    rec = {}
    for c in ST.CHECKS:
        sig = {k: float(sigma[c][k]) for k in SIGMA_KEYS}
        rec[c] = {"quantity": ST.QUANTITIES[c][2], **sig,
                  **ST.detectable(sig, m_p, n, "P")}
    rec.update({"N": n, "n_paintings": int(m_p.size), "seeds": [int(s) for s in seeds]})
    return rec


def read_arrays(scored) -> dict:
    """Primary per-anchor arrays, cluster/pair/seed indices and raw gates/reader inputs.
    Seeds are concatenated in pass order. No aggregate rate, B1, R1 or PM is computed.
    """
    _require(len(scored) >= 1, "no seed scored")
    out = {}
    for name in READ_SCORERS:
        metrics = tuple(scored[0][name])
        _require(metrics == tuple(METRICS) and all(tuple(sc[name]) == metrics for sc in scored),
                 f"{name}: metrics {metrics}, not score_seed's {tuple(METRICS)} in every seed")
        for m in metrics:
            parts = [np.asarray(sc[name][m]) for sc in scored]
            _require(all(p.dtype == np.float64 and p.shape == (len(sc["cl"]),) for p, sc in zip(parts, scored)),
                     f"{name}__{m}: per-anchor values must be float64 of the seed's episode count")
            out[f"{name}__{m}"] = np.concatenate(parts)
    for k in ("cl", "pair_index"):
        parts = [np.asarray(sc[k]) for sc in scored]
        _require(all(p.dtype.kind in "iu" and p.ndim == 1 for p in parts), f"{k}: a 1-d integer array per seed")
        out[k] = np.concatenate(parts).astype(np.int64)
    out["seed_index"] = np.concatenate([np.full(len(sc["cl"]), i, dtype=np.int64) for i, sc in enumerate(scored)])
    # Raw gates and reader choices are saved without computing rates (rule 7.4, 8 item 2).
    _require(all(set(sc['gates']) == {'aff'} for sc in scored), 'only AFF gates before verdict')
    for who in scored[0]["gates"]:
        for condition in CONDITIONS:
            out[f"{who}_gate__{condition}"] = np.concatenate(
                [sc["gates"][who][condition] for sc in scored], axis=1)
    # Same raw reader key names as stage 0a; score_seed's values are dicts by condition.
    for key, prefix in (("P", "P"), ("m", "margin"), ("pick", "pick")):
        for condition in CONDITIONS:
            out[f"{prefix}__{condition}"] = np.concatenate([sc["reader"][key][condition] for sc in scored], axis=0)
    return out


def read_names(read, variant="held") -> SimpleNamespace:
    """Single source of read paths. Fix reuses the first started, episodes, sensitivity and verdict.
    Reserve gets its own started/sensitivity/verdict and reuses the first read's episodes, as R6.
    """
    _require(read in READ_BACKBONES and variant in ("held", "fix1", "reserve", "smoke"),
             f"unknown read or variant: {read}/{variant}")
    suffix = "_reserve" if variant == "reserve" else ""
    pass_suffix = "_fix1" if variant == "fix1" else suffix
    return SimpleNamespace(read=read, kind=variant, started=f"held_started_{read}{suffix}.json",
                           verdict=f"held_verdict_{read}{suffix}.json", pass_=f"held_pass_{read}{pass_suffix}.json",
                           arrays=f"held_arrays_{read}{pass_suffix}.npz", sensitivity=f"sensitivity_{read}{suffix}.json",
                           row=read + ("-R" if variant == "reserve" else ""),
                           smoke_record=SMOKE_RECORDS.get(variant if variant in ("fix1", "reserve") else "smoke"))



def episodes_name(read, seed):
    return f"held_episodes_{read}_seed{int(seed)}.npz"



def ledger_row(path, row_id):
    path = Path(path)
    _refuse_unless(path.is_file(), f"the held ledger {path} is missing")
    rows = [[c.strip() for c in line.strip()[1:-1].split("|")]
            for line in path.read_text().splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")]
    rows = [c for c in rows if c[0] == row_id]
    _refuse_unless(bool(rows), f"ledger row {row_id} is missing")  # guard:ledger_row
    _refuse_unless(len(rows) <= 1, f"ledger row {row_id} appears twice")
    if not rows:
        return None
    cells = rows[0]
    _refuse_unless(len(cells) == len(LEDGER_CELLS), f"ledger row {row_id} needs eight cells")
    return {"row": row_id, "cells": cells, "episodes": HEX64.findall(cells[EPISODE_CELL]),
            "scripts": HEX64.findall(cells[SCRIPT_CELL]), "report": cells[REPORT_CELL]}



def ledger_guard(path, row_id, backbone, runner_sha, pins_sha=None):
    row = ledger_row(path, row_id)
    cells = row['cells']
    _refuse_unless(not any(s in cell for cell in cells for s in ("not run: switched off", "dropped under the rule")),
                   f"ledger row {row_id}: not run under the rule")  # guard:ledger_not_run
    _refuse_unless(cells[BACKBONE_CELL] == LEDGER_NAMES[backbone],
                   f"ledger row {row_id}: another backbone")  # guard:ledger_backbone
    # Labelled entries are required; a bare hash or another script's hash cannot authorise the read.
    runner = re.findall(r"(?<![\w.])run_r9_held\.py\s+([0-9a-f]{64})(?![0-9a-fA-F])", cells[SCRIPT_CELL])
    pins_path = R.pins_path(backbone)
    pins_name = pins_path.name
    if pins_sha is None:
        _refuse_unless(pins_path.is_file(), f"{pins_path}: pins file is missing")
        pins_sha = R.sha256_file(pins_path)
    pins = re.findall(r"(?<![\w.])" + re.escape(pins_name) + r"\s+([0-9a-f]{64})(?![0-9a-fA-F])", cells[SCRIPT_CELL])
    _refuse_unless(bool(runner) and runner[-1] == runner_sha,
                   f"ledger row {row_id}: runner SHA is not this runner's")  # guard:ledger_runner_sha
    _refuse_unless(bool(pins) and pins[-1] == pins_sha,
                   f"ledger row {row_id}: pins SHA is not {pins_name}'s")  # guard:ledger_pins_sha
    _refuse_unless(row['report'] == PENDING, f"ledger row {row_id}: report cell is not (pending)")  # guard:ledger_report
    return row



def h5_hashes(path=LEDGER):
    """Parse H5's labelled seed and pair hashes, never infer their order from unlabelled hex strings."""
    cell = ledger_row(path, "H5")['cells'][EPISODE_CELL]
    records = re.findall(r"seed\s+(\d+):\s*([^;]+)", cell)
    out = {}
    for seed, body in records:
        seed = int(seed)
        _require(seed not in out, f"ledger H5 repeats seed {seed}")
        pairs = {}
        for pair in R.PAIR_NAMES:
            a, b = pair.split("__")
            found = re.findall(re.escape(a) + r"\s*×\s*" + re.escape(b) + r"\s+([0-9a-f]{64})(?![0-9a-fA-F])", body)
            _require(len(found) == 1, f"ledger H5 seed {seed} needs one hash for {pair}")
            pairs[pair] = found[0]
        out[seed] = pairs
    _require(set(out) == set(R.HELD_SEEDS) and len(HEX64.findall(cell)) == 9, "ledger H5 needs nine episode hashes")
    E.assert_distinct(out, recorded={})
    return out



def read_guard(read, backbone, cells, smoke=False):
    _refuse_unless(smoke or READ_BACKBONES[read] == backbone,
                   f"{read} requires {READ_BACKBONES[read]}, not {backbone}")  # guard:read_backbone
    _refuse_unless(cells['switched_off'] is False, f"{backbone}: switched off, no read")  # guard:switched_off



def refuse_or_go(results, ledger, read, backbone, variant="held", after_crash=False, folder=None):
    """C5: filenames, started copies and ledger only; no held data or score is read here."""
    n, base = read_names(read, variant), read_names(read, "held")
    res = Path(results)
    copy = Path(folder or HERE) / n.started
    started = res / n.started
    first = variant == "reserve" or (variant == "held" and not after_crash)
    # Ledger before any read inputs. The pins file itself is hashed, not a pick or a cell.
    row = ledger_guard(ledger, n.row, backbone, runner_sha256())
    if variant == "held" and not after_crash:
        today = amsterdam_today()
        _refuse_unless(today <= READ_DEADLINE, f"Amsterdam date {today}: not started by {READ_DEADLINE}, no read starts")  # guard:time_box
    _refuse_unless(not (res / n.verdict).exists(), f"{n.verdict} exists: the read is over")  # guard:verdict_exists
    _refuse_unless(not (res / n.pass_).exists(), f"{n.pass_} exists: use --fix 1 only once")  # guard:pass_exists
    if variant == "reserve":
        _refuse_unless(not started.exists(), f"{n.started} exists: one reserve only")  # guard:reserve_started
        _refuse_unless((res / base.verdict).is_file(), "--reserve corrects a read whose verdict exists")  # guard:reserve_needs_verdict
    elif variant == "fix1":
        _refuse_unless((res / base.pass_).is_file(), f"--fix 1 corrects {base.pass_}, which is missing")  # guard:fix_needs_pass
        _refuse_unless((res / n.smoke_record).is_file(), "--fix 1 needs its own smoke")  # guard:fix_smoke
        _refuse_unless(started.is_file(), f"--fix 1: {n.started} is missing")
    elif started.exists():
        _refuse_unless(after_crash, f"{n.started}: read has started, use --after-crash")  # guard:started_exists
    else:
        _refuse_unless(not after_crash, "--after-crash: no attempt to rerun")  # guard:after_crash_needs_start
    if first:
        _refuse_unless(not copy.exists(), f"{copy}: committed copy exists, read already started")  # guard:copy_exists
    elif started.exists():
        _refuse_unless(copy.is_file() and copy.read_bytes() == started.read_bytes(),
                       f"{copy} missing or differs from {started}")  # guard:copy_matches
    attempts = read_attempts(started) if started.exists() else []
    if after_crash:
        _refuse_unless(len(attempts) < MAX_ATTEMPTS, f"{n.started}: already 2 attempts, the user decides")  # guard:two_attempts
    if variant == "fix1":
        _refuse_unless(not any(a['flags'].get('fix') for a in attempts),
                       f"{n.started}: already a --fix 1 attempt")  # guard:fix_once
    if first:
        _refuse_unless(row['cells'][EPISODE_CELL] == PENDING,
                       f"ledger row {n.row}: episode cell is not (pending)")  # guard:episodes_pending
    want = attempt_hashes(attempts)
    if variant == "fix1":
        _refuse_unless(_complete(want) and set(row['episodes']) == {h for hs in want.values() for h in hs.values()},
                       f"--fix 1: ledger row {n.row} needs exactly the nine episode hashes")  # guard:fix_episodes
    elif variant == "reserve":
        first_started = res / base.started
        want = attempt_hashes(read_attempts(first_started)) if first_started.is_file() else {}
        _refuse_unless(_complete(want), "--reserve: first read needs nine held episode hashes")  # guard:reserve_hashes
    return SimpleNamespace(names=n, attempts=attempts, want=want, ledger=row)



def latest_smoke_record(results, variant):
    """Same precedence as run_r9_apply_rule; fix1 specifically requires its own smoke."""
    results = Path(results)
    if variant in ("fix1", "reserve"):
        return results / SMOKE_RECORDS[variant]
    for kind in ("fix1", "crash1"):
        if (results / SMOKE_RECORDS[kind]).is_file():
            return results / SMOKE_RECORDS[kind]
    return results / SMOKE_RECORDS['smoke']



def smoke_guard(results, variant, here=None):
    path = latest_smoke_record(results, variant)
    _refuse_unless(path.is_file(), f"smoke {path.name} missing")  # guard:smoke_missing
    rec = _read_json(path)
    _refuse_unless(isinstance(rec, dict) and rec.get('passed') is True, f"smoke {path.name} did not pass")  # guard:smoke_passed
    _refuse_unless(rec.get('module_sha256') == R.r9_module_shas(here),
                   f"{path.name}: module bytes differ, new smoke needed")  # guard:smoke_stale
    return {"name": path.name, "sha256": R.sha256_file(path)}



def check_pins(backbone):
    path = R.pins_path(backbone)
    _refuse_unless(path.is_file(), f"{path} is missing")
    pins = _read_json(path)
    _refuse_unless(pins == PINS.build_pins(backbone), f"{path.name} differs from its named files")  # guard:pins
    return pins



def sigma_guard(backbone, results):
    """Rule 6 item 4's producer interface. New-backbone sigmas are not recipe cells' sigma_star.
    The source bytes are recorded in started and pass files. CLIP uses the frozen R6 file.
    """
    path = (R.R6D / 'results/sensitivity_seed42.json' if backbone == 'clip_b32'
            else Path(results) / f'sensitivity_seed42_{backbone}.json')
    raw = path.read_bytes() if path.is_file() else b''
    try:
        rec = json.loads(raw)
    except ValueError:
        rec = {}
    rec = rec if isinstance(rec, dict) else {}
    sigma = {c: {k: (rec.get(c) or {}).get(k) for k in SIGMA_KEYS} for c in ST.CHECKS}
    valid = bool(raw) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                              for sig in sigma.values() for v in sig.values())
    _refuse_unless(valid and (backbone != 'clip_b32' or R.sha256_bytes(raw) == SIGMA_CLIP_SHA),
                   f"{path}: missing, changed or invalid finite non-negative sigma parts for P1 to P7")  # guard:sigma
    return {"sigma": sigma, "file": str(path), "sha256": R.sha256_bytes(raw)}



def value_sets_file_guard():
    path = VALUE_SETS_PATH
    raw = path.read_bytes() if path.is_file() else b''
    _refuse_unless(R.sha256_bytes(raw) == VALUE_SETS_SHA,
                   f"{path}: not the frozen R6 value_sets.json")  # guard:value_sets_file
    return {"file": str(path), "sha256": R.sha256_bytes(raw)}



def value_sets_guard(env, checked):
    path = Path(checked['file'])
    raw = path.read_bytes()
    _refuse_unless(R.sha256_bytes(raw) == checked['sha256'] and json.loads(raw) == _plain(R.value_sets_record(env.value_sets, env.data)),
                   f"{path}: not the development value sets recomputed now")  # guard:value_sets_equal
    return checked



def refit_reference(backbone):
    # Selection posteriors are checked on every backbone; R6's extra arm-L coefficient identity is CLIP only.
    if backbone != 'clip_b32':
        return None
    path = R.R6D / 'results/refit_check.json'
    _require(R.sha256_file(path) == 'aebc840fa26266754370e01ae690d120d483cc86a4090b6130feb39be679736d',
             'R6 refit_check.json changed')
    return _read_json(path)



def head_guard(env, refit):
    _refuse_unless(isinstance(env.head_check, dict) and env.head_check.get('passed') is True,
                   'refit selection posteriors differ from the stored ones')  # guard:head_check
    if refit is not None:
        want = refit.get('coef_sha256')
        _refuse_unless(isinstance(want, dict) and bool(want) and _norm(env.coef_sha256) == _norm(want),
                       'head coefficient SHA-256 differs from refit_check.json')  # guard:coef_sha
    return _norm(env.coef_sha256)



def setup(backbone):
    H.require_threads()
    rows = R.load_rows(backbone)
    heads = H.fit_heads_r9(rows.data, rows.split.groups, rows.split.scorer_train, backbone=backbone)
    check = H.check_selection(heads, rows.data, rows.split.selection, backbone)
    pm = B.fit_pm(rows.data, rows.split.scorer_train, backbone)
    readers = B.load_readers(backbone)
    return SimpleNamespace(**vars(rows), heads=heads, head_check=check,
                           coef_sha256=H.coef_sha256(heads), pm=pm, readers=readers)



def seed_bundle(env, mode, seed, n_per_pair, backbone, on_episodes=None):
    ctx = X.RowContext(mode, seed, env.data, env.split, env.labels, env.heads, env.value_sets,
                       n_per_pair, on_episodes=on_episodes)
    return B.build_bundle_r6(ctx, env.readers, env.pm, backbone)



def started_guard(started, copy):
    _require(started.is_file() and copy.is_file() and started.read_bytes() == copy.read_bytes(),
             'started file and its folder copy must be the first write, before episodes')  # guard:started_first



def sensitivity_guard(path):
    _require(path.is_file(), 'sensitivity must exist before scores')  # guard:sensitivity_first



def run_read(spec, results, here=None, env=None):
    """Every attempt checks pins before picks/cells, then follows rule 7.4's computation order.
    Per-anchor inputs are saved; only pass_record computes aggregate checks, pooling all seeds/pairs.
    """
    t0 = time.time()
    out, n, backbone = Path(spec.out), spec.names, spec.backbone
    inputs = R.assert_inputs(backbone)
    check_pins(backbone)
    paths = BB.parts(backbone)
    picks = P.load_picks(paths['picks_seed42'], backbone)
    lambdas = P.frozen_lambdas(backbone)
    cell_record = P.read_cells(paths['cells_seed42'])
    read_guard(spec.read, backbone, cell_record, smoke=spec.mode == 'smoke')
    cells, taus = S.cells_from_record(cell_record), S.taus_of(backbone)
    sigma = sigma_guard(backbone, results)
    vs_file = value_sets_file_guard()
    recorded = E.recorded_hashes()
    env = env or setup(backbone)
    coef = head_guard(env, refit_reference(backbone))
    values = value_sets_guard(env, vs_file)
    started = out / n.started
    record = {"mode": spec.mode, "read": spec.read, "backbone": backbone,
              "runner_sha256": runner_sha256(), "module_sha256": R.r9_module_shas(here),
              "input_sha256": dict(inputs), "pins_sha256": R.sha256_file(R.pins_path(backbone)),
              "sigma_source": {k: sigma[k] for k in ('file', 'sha256')}, "coef_sha256": coef,
              "value_sets": values, "git": git_provenance(here), **spec.record}
    k = start_attempt(started, spec.copy_to, spec.attempts, spec.flags, record, spec.overwrite)
    started_guard(started, spec.copy_to)
    rec = EpisodeRecorder(started, spec.copy_to, k, spec.want, recorded, out, spec.overwrite,
                          read=spec.read, h5=spec.h5)
    bundles = []
    for seed in spec.seeds:
        rec.expect(seed)
        bundle = read_seed_bundle(env, spec.ctx_mode, seed, spec.n_per_pair, backbone, on_episodes=rec)
        _require(seed in rec.hashes and dict(bundle.episodes_sha256) == rec.hashes[seed],
                 f'seed {seed}: episodes not reported to recorder')  # guard:episodes_reported
        bundles.append(bundle)
    rows = env.split.held if spec.ctx_mode == 'held' else env.split.selection
    R.assert_held_eligible(env.labels, env.split.groups, rows, env.value_sets)
    sens = sensitivity_held(bundles, spec.seeds, sigma['sigma'], spec.n_per_pair)
    same = list(sens)
    sens.update({"episodes_sha256": {str(s): rec.hashes[s] for s in spec.seeds},
                 "sigma_source": {k: sigma[k] for k in ('file', 'sha256')},
                 "module_sha256": R.r9_module_shas(here), "time": R.amsterdam_now()})
    keep_or_write_json(out / n.sensitivity, sens, same + ['episodes_sha256', 'sigma_source'], spec.overwrite)
    sensitivity_guard(out / n.sensitivity)
    scored = []
    for bundle in bundles:
        sc = S.score_seed(bundle, picks, lambdas, env.readers, include_pm=False, secondary=False,
                          cells=cells, taus=taus, verdict=out / n.verdict)
        _require(tuple(sc) == READ_KEYS, f'scores {tuple(sc)}: only {READ_KEYS} before verdict')  # guard:allowed_keys
        scored.append(sc)
    keep_or_write_npz(out / n.arrays, read_arrays(scored), spec.overwrite)
    extra = {"episodes_sha256": {s: rec.hashes[s] for s in spec.seeds}, "runner_sha256": runner_sha256(),
             "module_sha256": R.r9_module_shas(here)}
    pr = ST.pass_record(scored, spec.mode, list(spec.seeds), extra, secondary=False)
    pr.update({"read": spec.read, "backbone": backbone, "kind": n.kind, "attempt": k,
               "input_sha256": dict(inputs), "coef_sha256": coef, "pins_sha256": record['pins_sha256'],
               "sigma_source": record['sigma_source'], "value_sets": values,
               "outputs": {p.name: R.sha256_file(p) for p in (out / n.arrays, out / n.sensitivity)},
               "runtime_s": round(time.time() - t0)})
    path = out / n.pass_
    write_new(path, _json_bytes(pr), spec.overwrite)
    print(f'held pass written {path.name} {R.sha256_file(path)}')
    return 0



def run_held(backbone, read, after_crash=False, fix=None, reserve=False, results=None, ledger=None,
             here=None, folder=None, env=None):
    results = Path(results or R.RESULTS)
    try:
        _refuse_unless(fix in (None, 1) and sum((bool(after_crash), fix is not None, bool(reserve))) <= 1,
                       'at most one of --after-crash, --fix 1, --reserve')
        variant = 'reserve' if reserve else 'fix1' if fix else 'held'
        go = refuse_or_go(results, ledger or LEDGER, read, backbone, variant, after_crash, folder)
        smoke = smoke_guard(results, variant, here)
        spec = SimpleNamespace(backbone=backbone, read=read, mode='held', names=go.names, ctx_mode='held',
                               seeds=R.HELD_SEEDS, n_per_pair=R.N_PER_PAIR, out=results,
                               copy_to=Path(folder or HERE) / go.names.started, overwrite=False,
                               flags={"after_crash": bool(after_crash), "fix": fix, "reserve": bool(reserve), "smoke": False},
                               attempts=go.attempts, want=go.want, h5=h5_hashes(ledger or LEDGER),
                               record={"smoke_record": smoke, "ledger": go.ledger})
        return run_read(spec, results, here, env)
    except Refused as e:
        print(f'refused: {e}')
        return EXIT_REFUSE



def run_smoke(backbone, read, results=None, subdir=None, here=None, env=None):
    results = Path(results or R.RESULTS)
    try:
        _refuse_unless(subdir is None or SUBDIR_RE.fullmatch(str(subdir)),
                       '--smoke-subdir must contain only letters, digits or underscore')
        out = results / 'smoke' / (subdir or '')
        names = read_names(read, 'smoke')
        _refuse_unless(not (out / names.verdict).exists(), 'verdict exists: repeated smoke needs --smoke-subdir')  # guard:smoke_verdict
        spec = SimpleNamespace(backbone=backbone, read=read, mode='smoke', names=names, ctx_mode='selection',
                               seeds=R.SMOKE_SEEDS, n_per_pair=R.N_SMOKE, out=out,
                               copy_to=out / 'folder' / names.started, overwrite=True,
                               flags={"after_crash": False, "fix": None, "reserve": False, "smoke": True},
                               attempts=[], want={}, h5=None, record={"smoke_subdir": subdir})
        return run_read(spec, results, here, env)
    except Refused as e:
        print(f'refused: {e}')
        return EXIT_REFUSE



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--backbone', required=True, choices=BB.BACKBONES)
    ap.add_argument('--read', required=True, choices=tuple(READ_BACKBONES))
    once = ap.add_mutually_exclusive_group()
    once.add_argument('--after-crash', action='store_true')
    once.add_argument('--fix', type=int, choices=(1,))
    once.add_argument('--reserve', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-subdir')
    args = ap.parse_args(argv)
    if args.smoke and (args.after_crash or args.fix is not None or args.reserve):
        ap.error('--smoke does not accept --after-crash, --fix or --reserve')
    if args.smoke_subdir is not None and not args.smoke:
        ap.error('--smoke-subdir requires --smoke')
    if args.smoke:
        return run_smoke(args.backbone, args.read, subdir=args.smoke_subdir)
    return run_held(args.backbone, args.read, args.after_crash, args.fix, args.reserve)


if __name__ == '__main__':
    sys.exit(main())
