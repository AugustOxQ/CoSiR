"""Round 6: the verdict step (DECISION_RULE.md §4, §8.3, §8.5, §9), run only after the phase-2 agreement of §8.4.

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_apply_rule.py [--smoke | --reserve]

Reads, in results/ (results/smoke/ with --smoke), all parsed as JSON:
  - rederive_agreement.json (--reserve: rederive_agreement_reserve.json): the re-derivation's phase-2 record
    (contracts §8). It must say phase 2, all_agree true, no disagreements, and carry the smoke flag of this mode.
  - the pass file it names: held_pass.json or held_pass_fix1.json (--smoke: held_pass.json; --reserve:
    held_pass_reserve.json), bound to the record by its SHA-256 (the bytes hashed are the bytes parsed). Its
    rule_sha256 must be this rule's, its mode "held" ("smoke" with --smoke), its seeds 52, 53, 54 (9001 to 9003).
  - sensitivity_held<suffix>.json, <suffix> the pass file's ("", "_fix1", "_reserve"), else sensitivity_held.json:
    x and x95 of P1 to P7, x2 and x95 of S1 and S2 (rule §8.2), used only to read a failed check.
Writes held_verdict.json (--reserve: held_verdict_reserve.json) once; an existing verdict file is never replaced.

The rule, applied here from the pass file's integer counts n_j:
  - Holm across P1 to P7 (rule §3) is recomputed from the counts; the pass file's flags, ranks, levels, order and
    boundary flags must equal the recomputation, else the step stops (exit 5) without a verdict.
  - GO iff all seven pass. NO-GO: each failed check is read by C11 (point > 0: inconclusive at the detectable margin
    x; point <= 0: "AFF did not beat <name> on new paintings"); a check that fails only because an earlier one in the
    Holm order failed is "not reached"; the NO-GO is "inconclusive" if every failed check reads inconclusive, else
    "did not beat". Passed checks in a NO-GO are listed as passed and license nothing (claim null).
  - After a GO only: Holm across S1, S2 (m = 2, ties S1 first); a pass licenses "AFF also beats B′(A1)" or "AFF also
    beats R1", appended to the rule's claim. After a NO-GO, S1 and S2 are reported (point, 95%), not tested, not read.
  - Boundary report (§8.5): every check (and S1, S2 after a GO) within one count of n*_k; the integer rule decides.

Exit codes: 0 verdict written; 4 refused (a precondition failed; nothing written); 5 the pass file contradicts the
rule's recomputation, or this script's fixed texts are not the rule's (stop and report; nothing written).
Smoke mode prints no value, only the path written.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def _local_common() -> SimpleNamespace:
    """Stand-in for the few r6_common names (contracts §1) this step uses, until ticket 01 is merged."""
    here = Path(__file__).resolve().parent
    results = here / "results"

    def sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def sha256_file(path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def amsterdam_now() -> str:
        return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")

    def r6_module_shas() -> dict:
        root = here.parents[2]
        files = sorted(set(here.glob("r6_*.py")) | set(here.glob("run_r6_*.py")))
        files += [p for p in [here / "dts_settings.json"] if p.is_file()]
        files += sorted((root / "scripts").glob("run_r6_*.sh"))
        files += [p for p in [root / "scripts" / "das6_sync_r6.py"] if p.is_file()]
        return {str(p.relative_to(root)): sha256_file(p) for p in files}

    return SimpleNamespace(
        HERE=here, RESULTS=results, SMOKE=results / "smoke",
        RULE_SHA256="7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724",
        HELD_SEEDS=(52, 53, 54), SMOKE_SEEDS=(9001, 9002, 9003),
        sha256_bytes=sha256_bytes, sha256_file=sha256_file, amsterdam_now=amsterdam_now,
        r6_module_shas=r6_module_shas)


CM = _local_common()  # once ticket 01 is merged, this one line becomes: import r6_common as CM

RULE_PATH = CM.HERE / "DECISION_RULE.md"
RESULTS = CM.RESULTS
SMOKE = CM.SMOKE

N_BOOT = 5000
P_CHECKS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
S_CHECKS = ("S1", "S2")
# Rule §4's <name> for "AFF did not beat <name> on new paintings".
NAMES = {"P1": "cosine", "P2": "RCA", "P3": "B", "P4": "B′(A0)", "P5": "its matched control",
         "P6": "the condition-free scorers on condition gain", "P7": "RCA on condition gain",
         "S1": "B′(A1)", "S2": "R1"}
# Plain names of the checks, for the "not reached" reading.
LABELS = {"P1": "R@1 against cosine", "P2": "R@1 against RCA", "P3": "R@1 against B", "P4": "R@1 against B′(A0)",
          "P5": "R@1 against the matched control", "P6": "condition gain against the condition-free scorers",
          "P7": "condition gain against RCA", "S1": "R@1 against B′(A1)", "S2": "R@1 against R1"}
LICENSE = {"S1": "AFF also beats B′(A1)", "S2": "AFF also beats R1"}
# Rule §4, "Claim licensed by a GO", verbatim (asserted against the rule's text on every run).
CLAIM_GO = ("AFF beats COS, RCA, B, B′(A0) and its matched control on aspect R@1, and its condition gain exceeds "
            "theirs and RCA's, pooled over three aspect pairs, on paintings never used to fit, select or tune it, "
            "with Holm across the seven checks. No per-pair margin, no other dataset or backbone. Disclosures of "
            "§10.4 accompany every AFF number.")
# Fixed phrases of rule §4 this step writes; each must appear in the rule's text.
RULE_PHRASES = (CLAIM_GO, '"AFF also beats B′(A1)"', '"AFF also beats R1"', '"AFF did not beat <name> on new paintings"',
                '"inconclusive at a detectable margin of x"', "the condition-free scorers on condition gain",
                '"RCA on condition gain"', '"not reached: the Holm procedure stopped at <earlier check>"',
                "cosine, RCA, B, B′(A0), its matched control")
PASS_FILES = {"real": ("held_pass.json", "held_pass_fix1.json"), "smoke": ("held_pass.json",),
              "reserve": ("held_pass_reserve.json",)}
SHA_RE = re.compile(r"[0-9a-f]{64}")


class Stop(Exception):
    """A refusal (code 4) or a contradiction (code 5); nothing is written."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def refuse(msg):
    raise Stop(4, msg)


def contradiction(msg):
    raise Stop(5, msg)


# ---------------------------------------------------------------- the rule's integer Holm (private copies)
# Ticket 04 owns the shared versions (r6_stats.holm, r6_stats.boundary); these copies decide nothing the pass file
# has not already recorded: they recompute it, and any difference stops the step.

def n_star(k: int, m: int) -> int:
    """Rule §8.5: the largest passing count at Holm rank k of m, floor(5001 / (40 (m + 1 - k))) - 1."""
    return (N_BOOT + 1) // (40 * (m + 1 - k)) - 1


def holm(counts: dict, names: tuple, m: int) -> list:
    """Rule §3 (m = 7) and §4 (m = 2): order by n ascending, ties in the given order; the k-th passes iff it and
    every earlier one pass and 40 (n + 1) (m + 1 - k) <= 5001."""
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    out, ok = [], True
    for k, nm in enumerate(order, start=1):
        own = 40 * (counts[nm] + 1) * (m + 1 - k) <= N_BOOT + 1
        ok = ok and own
        out.append({"name": nm, "k": k, "n": counts[nm], "passes": ok, "own_count_passes": own,
                    "level_two_sided": 1.0 - 0.05 / (m + 1 - k), "n_star": n_star(k, m),
                    "near_boundary": abs(counts[nm] - n_star(k, m)) <= 1})
    return out


# ---------------------------------------------------------------- inputs

def _parse(raw: bytes, what: str):
    def no_constant(c):
        raise ValueError(f"non-finite constant {c}")
    try:
        rec = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    except (ValueError, UnicodeDecodeError) as e:
        refuse(f"{what} is not valid JSON ({type(e).__name__})")
    if not isinstance(rec, dict):
        refuse(f"{what} is not a JSON object")
    return rec


def _is_int(x) -> bool:
    return type(x) is int


def _is_num(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def _is_interval(x) -> bool:
    return isinstance(x, list) and len(x) == 2 and all(_is_num(v) for v in x) and x[0] <= x[1]


def assert_rule() -> None:
    """The rule file is the committed one, and this script's fixed texts are its own."""
    raw = RULE_PATH.read_bytes()
    if CM.sha256_bytes(raw) != CM.RULE_SHA256:
        refuse(f"{RULE_PATH.name}: its SHA-256 is not the committed rule's")
    text = " ".join(raw.decode("utf-8").split())
    missing = [p for p in RULE_PHRASES if p not in text]
    if missing:
        contradiction(f"{len(missing)} fixed phrase(s) of this script are not in rule §4 verbatim")


def load_agreement(path: Path, smoke: bool, allowed: tuple) -> tuple:
    """Rule §8.3, §6.7: the phase-2 agreement record, or refuse."""
    if not path.is_file():  # GUARD: agreement-missing
        refuse(f"{path.name} is missing: the verdict is written only after the phase-2 agreement (rule §8.3)")
    raw = path.read_bytes()
    agr = _parse(raw, path.name)
    if not (_is_int(agr.get("phase")) and agr["phase"] == 2):  # GUARD: phase-2
        refuse(f"{path.name}: not a phase-2 record (rule §8.4)")
    if type(agr.get("smoke")) is not bool:
        refuse(f"{path.name}: its smoke flag is not a boolean")
    if agr["smoke"] is not smoke:  # GUARD: smoke-flag
        refuse(f"{path.name}: a {'real' if agr['smoke'] is False else 'smoke'} agreement record is refused in "
               f"{'smoke' if smoke else 'real'} mode (rule §6.7)")
    if agr.get("all_agree") is not True:  # GUARD: all-agree
        refuse(f"{path.name}: all_agree is not true; trace the disagreement first (rule §8.4)")
    if agr.get("disagreements") != []:
        refuse(f"{path.name}: its disagreements list is not empty")
    if not (_is_int(agr.get("n_quantities")) and agr["n_quantities"] >= 1):
        refuse(f"{path.name}: n_quantities is not a positive integer")
    if agr.get("pass_file") not in allowed:
        refuse(f"{path.name}: pass_file is not one of {', '.join(allowed)}")
    if not (isinstance(agr.get("held_pass_sha256"), str) and SHA_RE.fullmatch(agr["held_pass_sha256"])):
        refuse(f"{path.name}: held_pass_sha256 is not a SHA-256")
    return agr, CM.sha256_bytes(raw)


def _check_entry(rec: dict, name: str, where: str, with_flags: bool) -> None:
    c = rec.get(name)
    if not isinstance(c, dict):
        refuse(f"{where}: {name} is missing")
    ok = (isinstance(c.get("quantity"), str) and _is_int(c.get("n")) and 0 <= c["n"] <= N_BOOT
          and _is_num(c.get("point")) and _is_interval(c.get("ci95")) and _is_interval(c.get("ci_holm"))
          and _is_int(c.get("holm_k")) and _is_num(c.get("level_two_sided"))
          and type(c.get("near_boundary")) is bool)
    if with_flags:
        ok = ok and type(c.get("passes")) is bool and type(c.get("own_count_passes")) is bool
    if not ok:
        refuse(f"{where}: {name} does not have the fields and types of contracts §6")


def load_pass(raw: bytes, name: str, smoke: bool) -> dict:
    """The pass file (contracts §6): this rule, this mode, these seeds, every field present."""
    rec = _parse(raw, name)
    if rec.get("rule_sha256") != CM.RULE_SHA256:  # GUARD: pass-rule-sha
        refuse(f"{name}: its rule_sha256 is not this rule's")
    mode = "smoke" if smoke else "held"
    if rec.get("mode") != mode:  # GUARD: pass-mode
        refuse(f"{name}: its mode is not {mode!r}")
    seeds = list(CM.SMOKE_SEEDS if smoke else CM.HELD_SEEDS)
    if rec.get("seeds") != seeds:
        refuse(f"{name}: its seeds are not {seeds}")
    checks, sec = rec.get("checks"), rec.get("secondary")
    if not isinstance(checks, dict) or set(checks) != set(P_CHECKS):
        refuse(f"{name}: checks are not exactly P1 to P7")
    if not isinstance(sec, dict) or not set(S_CHECKS) <= set(sec):
        refuse(f"{name}: secondary lacks S1 or S2")
    for nm in P_CHECKS:
        _check_entry(checks, nm, name, with_flags=True)
    for nm in S_CHECKS:
        _check_entry(sec, nm, name, with_flags=False)
    order = rec.get("holm_order")
    if not (isinstance(order, list) and all(isinstance(o, str) for o in order) and sorted(order) == list(P_CHECKS)):
        refuse(f"{name}: holm_order is not an ordering of P1 to P7")
    return rec


def load_sensitivity(out: Path, suffix: str) -> tuple:
    """Rule §8.2: x, x95 (P) and x2, x95 (S), finite and non-negative."""
    path = out / f"sensitivity_held{suffix}.json"
    if not path.is_file():
        path = out / "sensitivity_held.json"
    if not path.is_file():
        refuse(f"{path.name} is missing (rule §8.2)")
    raw = path.read_bytes()
    rec = _parse(raw, path.name)
    for nm in P_CHECKS + S_CHECKS:
        xkey = "x" if nm in P_CHECKS else "x2"
        c = rec.get(nm)
        if not (isinstance(c, dict) and all(_is_num(c.get(k)) and c[k] >= 0 for k in (xkey, "x95"))):
            refuse(f"{path.name}: {nm} lacks a finite non-negative {xkey} and x95")
    return rec, path, CM.sha256_bytes(raw)


# ---------------------------------------------------------------- the rule (pure)

def assert_consistent(family: dict, recomputed: list, what: str, with_flags: bool) -> None:
    """The pass file's Holm fields must equal the recomputation from its counts."""
    for e in recomputed:
        c = family[e["name"]]
        diffs = [f for f, ok in (("holm_k", c["holm_k"] == e["k"]),
                                 ("level_two_sided", abs(c["level_two_sided"] - e["level_two_sided"]) <= 1e-12),
                                 ("near_boundary", c["near_boundary"] is e["near_boundary"])) if not ok]
        if with_flags:
            diffs += [f for f in ("passes", "own_count_passes") if c[f] is not e[f]]
        if diffs:
            contradiction(f"{what} {e['name']}: recorded {', '.join(diffs)} differ from the integer Holm recomputed "
                          f"from the counts")


def c11(name: str, c: dict, sens_c: dict, xkey: str) -> dict:
    """Rule §4 (C11) for one failed check: point > 0 inconclusive at x, else did not beat <name>."""
    if c["point"] > 0:
        x, x95 = float(sens_c[xkey]), float(sens_c["x95"])
        hw95 = (c["ci95"][1] - c["ci95"][0]) / 2.0
        hwh = (c["ci_holm"][1] - c["ci_holm"][0]) / 2.0
        text = (f"inconclusive at a detectable margin of {'x' if xkey == 'x' else 'x₂'} = {x:.3f} points (x95 = "
                f"{x95:.3f}; realised half-width {hw95:.3f} at 95%, {hwh:.3f} at its Holm level)")
        return {"reading_kind": "inconclusive", "c11_reading": text, xkey: x, "x95": x95,
                "half_width_95": hw95, "half_width_holm": hwh}
    return {"reading_kind": "did not beat", "c11_reading": f"AFF did not beat {NAMES[name]} on new paintings"}


def read_family(family: dict, names: tuple, recomputed: list, sens: dict, xkey: str) -> dict:
    """Each check's pass, reading and "not reached" mark, in the family's canonical order."""
    stop = next((e["name"] for e in recomputed if not e["own_count_passes"]), None)
    out = {}
    for e in recomputed:
        nm, c = e["name"], family[e["name"]]
        r = {"passes": e["passes"], "own_count_passes": e["own_count_passes"], "holm_k": e["k"], "n": e["n"],
             "n_star": e["n_star"], "near_boundary": e["near_boundary"], "quantity": c["quantity"],
             "label": LABELS[nm], "point": c["point"], "ci95": c["ci95"], "ci_holm": c["ci_holm"],
             "level_two_sided": c["level_two_sided"], "not_reached_after": None}
        if e["passes"]:
            r.update(reading_kind="passed", reading="passed")
        else:
            r.update(c11(nm, c, sens[nm], xkey))
            if e["own_count_passes"]:
                assert stop is not None and stop != nm
                r["not_reached_after"] = stop
                r["reading"] = (f"not reached: the Holm procedure stopped at {stop} ({LABELS[stop]}); its own reading: "
                                f"{r['c11_reading']}")
            else:
                r["reading"] = r["c11_reading"]
        out[nm] = r
    return {nm: out[nm] for nm in names}


def decide(rec: dict, sens: dict) -> dict:
    """Rule §4 and §8.5 on a validated pass record and sensitivity record."""
    checks, sec = rec["checks"], rec["secondary"]
    hp = holm({nm: checks[nm]["n"] for nm in P_CHECKS}, P_CHECKS, 7)
    assert_consistent(checks, hp, "check", with_flags=True)
    if rec["holm_order"] != [e["name"] for e in hp]:
        contradiction("holm_order differs from the order recomputed from the counts (ties P1 to P7)")
    hs = holm({nm: sec[nm]["n"] for nm in S_CHECKS}, S_CHECKS, 2)
    assert_consistent(sec, hs, "secondary", with_flags=False)

    go = all(e["passes"] for e in hp)
    read = read_family(checks, P_CHECKS, hp, sens, "x")
    failed = [nm for nm in P_CHECKS if not read[nm]["passes"]]
    assert go == (not failed)
    kind = None
    if not go:
        kind = "inconclusive" if all(read[nm]["reading_kind"] == "inconclusive" for nm in failed) else "did not beat"

    boundary = [{"check": e["name"], "family": "pass", "holm_k": e["k"], "n": e["n"], "n_star": e["n_star"],
                 "passes": e["passes"], "rederivation_n": e["n"]} for e in hp if e["near_boundary"]]
    if go:  # GUARD: gatekeeping
        sread = read_family(sec, S_CHECKS, hs, sens, "x2")
        for nm in S_CHECKS:
            sread[nm]["license"] = LICENSE[nm] if sread[nm]["passes"] else None
        secondary = {"tested": True, "holm_order": [e["name"] for e in hs], **sread}
        boundary += [{"check": e["name"], "family": "secondary", "holm_k": e["k"], "n": e["n"], "n_star": e["n_star"],
                      "passes": e["passes"], "rederivation_n": e["n"]} for e in hs if e["near_boundary"]]
        claim = CLAIM_GO + "".join(f" {LICENSE[nm]}." for nm in S_CHECKS if sread[nm]["passes"])
    else:
        secondary = {"tested": False, **{nm: {"quantity": sec[nm]["quantity"], "label": LABELS[nm],
                                              "point": sec[nm]["point"], "ci95": sec[nm]["ci95"],
                                              "passes": None, "reading": None} for nm in S_CHECKS}}
        claim = None
    for b in boundary:
        b["note"] = ("rule §8.5: reported to the user; the integer rule decides; the re-derivation's count equals the "
                     "runner's by the phase-2 agreement")
    return {"verdict": "GO" if go else "NO-GO", "kind": kind, "checks": read, "holm_order": [e["name"] for e in hp],
            "failed": failed, "secondary": secondary, "boundary_report": boundary, "claim": claim}


# ---------------------------------------------------------------- the step

def write_once(path: Path, obj: dict) -> None:
    """Write the complete file under its name only if no file of that name exists (atomic, never replaces)."""
    data = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(data, encoding="utf-8")
    try:
        os.link(tmp, path)
    except FileExistsError:
        refuse(f"{path.name} exists; a verdict is never overwritten")
    finally:
        tmp.unlink()


def apply(smoke: bool = False, reserve: bool = False) -> tuple:
    """Every refusal, then the verdict record written once. -> (path, record)."""
    smoke, reserve = bool(smoke), bool(reserve)
    if smoke and reserve:
        refuse("--smoke and --reserve exclude each other")
    out = SMOKE if smoke else RESULTS
    sfx = "_reserve" if reserve else ""
    vpath = out / f"held_verdict{sfx}.json"
    if vpath.exists():  # GUARD: verdict-exists
        refuse(f"{vpath.name} exists; a verdict is never overwritten (rule §8.1, §9)")
    if reserve and not (out / "held_verdict.json").is_file():
        refuse("a reserve verdict follows the original held_verdict.json, which is missing (rule §9)")
    assert_rule()
    apath = out / f"rederive_agreement{sfx}.json"
    agr, agr_sha = load_agreement(apath, smoke, PASS_FILES["smoke" if smoke else "reserve" if reserve else "real"])
    ppath = out / agr["pass_file"]
    if not ppath.is_file():
        refuse(f"{ppath.name}, named by {apath.name}, is missing")
    raw = ppath.read_bytes()
    pass_sha = CM.sha256_bytes(raw)
    if agr["held_pass_sha256"] != pass_sha:  # GUARD: sha-binding
        refuse(f"{apath.name} checked another pass: its held_pass_sha256 is not the SHA-256 of {ppath.name}")
    if ppath.name == "held_pass_fix1.json" and not (out / "held_pass.json").is_file():
        refuse("held_pass_fix1.json is named, but the original held_pass.json it corrects is missing (rule §8.1)")
    rec = load_pass(raw, ppath.name, smoke)
    suffix = ppath.name[len("held_pass"):-len(".json")]
    sens, spath, sens_sha = load_sensitivity(out, suffix)

    v = decide(rec, sens)
    v.update({
        "held_pass_sha256": pass_sha, "agreement_sha256": agr_sha, "rule_sha256": CM.RULE_SHA256,
        "module_sha256": CM.r6_module_shas(), "time": CM.amsterdam_now(),
        "mode": "smoke" if smoke else "held", "reserve": reserve, "seeds": rec["seeds"],
        "pass_file": ppath.name, "agreement_file": apath.name,
        "agreement": {k: agr.get(k) for k in ("phase", "smoke", "all_agree", "pass_file", "held_pass_sha256",
                                              "n_quantities", "time")},
        "sensitivity_file": spath.name, "sensitivity_sha256": sens_sha,
    })
    write_once(vpath, v)
    return vpath, v


def say(msg: str) -> None:
    print(f"[run_r6_apply_rule] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="the smoke chain (results/smoke/, smoke agreement only)")
    g.add_argument("--reserve", action="store_true", help="the reserve read of rule §9 (_reserve files)")
    args = ap.parse_args(argv)
    try:
        path, v = apply(smoke=args.smoke, reserve=args.reserve)
    except Stop as e:
        say(("REFUSED: " if e.code == 4 else "STOP, report to the user: ") + e.msg)
        raise SystemExit(e.code)
    if not args.smoke:
        say(f"verdict: {v['verdict']}" + (f" ({v['kind']})" if v["kind"] else ""))
        for nm, c in v["checks"].items():
            say(f"{nm}: " + ("passed" if c["passes"] else f"failed, {c['reading_kind']}"
                             + (f", not reached after {c['not_reached_after']}" if c["not_reached_after"] else "")))
        if v["secondary"]["tested"]:
            for nm in S_CHECKS:
                say(f"{nm}: " + ("passed" if v["secondary"][nm]["passes"] else "failed"))
        else:
            say("S1, S2: not tested (NO-GO)")
        for b in v["boundary_report"]:
            say(f"boundary (rule §8.5, the integer rule decides): {b['check']} count {b['n']}, "
                f"largest passing count {b['n_star']}")
    say(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
