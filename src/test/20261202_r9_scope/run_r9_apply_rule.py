"""Round 9: the verdict step for H6 to H10 (DECISION_RULE.md 4.1, 4.2, 7.3, 7.4, 7.6, 10), from R6's run_r6_apply_rule.py.

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_apply_rule.py --read H8 \
        [--variant reserve] [--smoke] [--not-run switched_off|dropped]

Reads results/held_pass_<Hn>.json and results/sensitivity_<Hn>.json, writes results/held_verdict_<Hn>.json once (an
existing file is never replaced). Before any verdict (R6 rule 8.1, 8.3; this rule 4.1 "6 passed", 6 item 5, 7.4):
  - the latest smoke record (results/smoke_record_fix1.json if it exists, else smoke_record_crash1.json if it exists,
    else smoke_record.json; --variant reserve: smoke_record_reserve.json) says "passed": true and its module_sha256
    equals r9_common.r9_module_shas() key for key (a changed file needs a new smoke); not in --smoke mode;
  - the phase-2 agreement of the pass file exists, agreement true, bound to the pass file's SHA-256 (see AGREEMENT below).
One pass file per verdict (R6 rule 8.1): when held_pass_<Hn>_fix1.json exists it is the pass (its agreement, and the
original held_pass_<Hn>.json must exist) and the original is not read; the verdict is the unsuffixed
held_verdict_<Hn>.json either way. There is no --variant fix1. --variant reserve reads the _reserve names (pass,
sensitivity, agreement, smoke record), writes held_verdict_<Hn>_reserve.json and needs held_verdict_<Hn>.json.
With --smoke the files are read from and written to results/smoke/ (agreement too) and must be smoke files (seeds
9001 to 9003, mode "smoke", agreement smoke true). Exit codes: 0 verdict written; 4 refused (a precondition failed;
nothing written); 5 the files contradict the rule's recomputation (stop and report; nothing written).

The rule applied, from the integer counts n_j of the pass file (Holm in R6 rule 3's integer form,
40 * (n + 1) * (m + 1 - k) <= 5,001, order by n ascending, ties in the order of the names):
  - H6, H7 (design "seven"): Holm over P1 to P7; GO iff all seven pass, else NO-GO.
  - H8 to H10 (design "two_level"): level 1 = (P4, P6) with m = 2; level 2 = (P1, P2, P3, P5, P7) with m = 5, tested
    only if both level-1 checks pass. "full claim" = both levels pass; "small claim" = level 1 passes, level 2 does
    not; "NO-GO" = level 1 fails (the level-2 checks are then reported with point and 95% interval, not tested, not
    read). Both claims start "AFF on <backbone's ledger name>" and carry the set-aside and flagged languages and the
    near-duplicate counts of rule 3.3.
  - A failed tested check is read as R6 rule 4 (C11): point > 0 "inconclusive at a detectable margin of x", point <= 0
    "AFF did not beat <name> on new paintings"; a check that fails only because an earlier one in its Holm order
    failed is "not reached". A NO-GO is "inconclusive" if every failed check reads inconclusive, else "did not beat".
    x = z * SE with z = 3.532 (design seven), 3.083 (level 1), 3.417 (level 2); x95 = 2.80 * SE.
  - 7.6: a tested check whose count is within one of its n* goes to boundary_report with its count, n* and the
    re-derivation's count (rederivation_n, from the agreement file); the integer rule decides.
  - --not-run switched_off | dropped (D25, no pass file and no held_started_<Hn>.json may exist, no variant) writes a
    verdict file whose status reads "not run: switched off" or "dropped under the rule" and whose verdict is null. The
    ledger row <Hn> must read the same wording (not in --smoke mode). switched_off is refused for H8 (CLIP cannot be
    switched off) and, for H9 and H10, unless held_verdict_H6.json / H7.json reads "not run: switched off" (D30).

INPUT SCHEMAS (what runners 20 and 21 and the re-derivation (ticket 25) write). All numbers are R@1 points; every JSON
is finite (no NaN).
  held_pass_<Hn>.json, H6 and H7: exactly r9_stats.pass_record's output (R6's schema): rule_sha256 (= r9_common.
    RULE_SHA256), mode "held", seeds [52, 53, 54], n_episodes 36864, n_clusters, checks {P1..P7: {quantity, n (0..5000),
    point, ci95 [lo, hi], holm_k, ci_holm [lo, hi], level_two_sided, passes (bool), own_count_passes (bool),
    near_boundary (bool)}}, holm_order [7 names], episodes_sha256, runner_sha256, module_sha256, time. A `secondary`
    key, if present, is ignored (B1, R1 and PM are not tested and not read here, rule 4.1).
  held_pass_<Hn>.json, H8 to H10: the same, with these changes only:
    seeds [55, 56, 57]; "design": "two_level";
    each checks entry also has "level" (1 for P4, P6; 2 for P1, P2, P3, P5, P7) and its holm_k, ci_holm,
      level_two_sided, passes, own_count_passes, near_boundary are those of its own level's Holm (m = 2 for level 1,
      m = 5 for level 2; ranks 1 to m; computed for the level-2 checks even when level 1 fails, the verdict step
      decides whether they are tested);
    holm_order is {"level1": [P4, P6 in count order], "level2": [five names in count order]};
    no `secondary`;
    "set_aside": [{"language": one of the 27 names of rule 5.1 item 1, "pair": a pair name, "seed": one of the seeds},
      ...] ([] if none; no duplicate entry; at most two per (pair, seed): else exit 5, rule 5.1 item 6);
    "flagged_languages": [str, ...] (rule 3.3; [] if none);
    "near_duplicates": {"cos_099": int, "cos_098": int} (rule 3.3: read paintings whose zip image has a train-side
      image at CLIP cosine >= 0.99 and >= 0.98; 0 <= cos_099 <= cos_098 <= 746). NEW: the runner (ticket 21) fills it.
  sensitivity_<Hn>.json (all seven, also the untested): {P1..P7: {"SE", "x", "x95"}} with x = z * SE for the check's
    read kind (see detectable_margins), x95 = 2.80 * SE; other top-level keys are ignored. The step recomputes x and x95
    from SE and stops (exit 5) on a difference.
  AGREEMENT: rd9_<Hn>_agreement[_fix1|_reserve].json, one per pass file, in rederive/rd9_out/ (results/smoke/ with
    --smoke), written by rederive/rd9_compare.py (ticket 25), in ticket 25's names. Fields read: phase (2),
    smoke (bool, = --smoke; the comparator takes it from the pass file's own "mode"), agreement (true), compared (int >= 1,
    = len(items)), failed (0, = the number of items that are not equal), items (a list of {key, equal (bool), ...}; none
    may have equal false), read (<Hn>), pass_file (the pass file's name), held_pass_sha256 (= the SHA-256 of that pass
    file's bytes), rederived_counts {P1..P7: int} (the independent pass's n_j, equal to the pass file's). Other keys
    (independent_identity, backbone) are not read. Earlier drafts of this step expected other names (all_agree,
    disagreements, n_quantities); they are not accepted.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import r9_common as CM  # noqa: E402  first, before anything that imports src
import r9_stats as ST  # noqa: E402  the integer Holm, n* (ticket 05)

RULE_PATH = CM.HERE / "DECISION_RULE.md"
RESULTS = CM.RESULTS
SMOKE = CM.SMOKE

P_CHECKS = tuple(ST.CHECKS)
L1 = ("P4", "P6")                       # level 1, ties P4 first (rule 4.2)
L2 = ("P1", "P2", "P3", "P5", "P7")     # level 2, ties in this order
assert P_CHECKS == ("P1", "P2", "P3", "P4", "P5", "P6", "P7") and sorted(L1 + L2) == sorted(P_CHECKS)
N_BOOT = ST.N_BOOT
N_LANGUAGES = 27
Z = {"seven": 3.532, "level1": 3.083, "level2": 3.417}      # rule 7.4: z at the strictest Holm level plus z at 0.8
Z95 = ST.Z_95
assert (Z["seven"], Z["level1"], Z95) == (ST.Z_P, ST.Z_S, 2.80)

READS = {
    "H6": {"design": "seven", "backbone": "Qwen3-VL-Embedding-2B", "short": "Qwen"},
    "H7": {"design": "seven", "backbone": "SigLIP 2 so400m/14-384", "short": "SigLIP 2"},
    "H8": {"design": "two_level", "backbone": "CLIP ViT-B/32", "short": "CLIP"},
    "H9": {"design": "two_level", "backbone": "Qwen3-VL-Embedding-2B", "short": "Qwen"},
    "H10": {"design": "two_level", "backbone": "SigLIP 2 so400m/14-384", "short": "SigLIP 2"},
}
SEEDS = {"seven": (52, 53, 54), "two_level": (55, 56, 57)}
SMOKE_SEEDS = tuple(CM.SMOKE_SEEDS)
VARIANTS = {None: "", "reserve": "_reserve"}      # a fix pass has no variant: held_pass_<Hn>_fix1.json is used when it exists
SMOKE_RECORDS = {"smoke": "smoke_record.json", "crash1": "smoke_record_crash1.json",
                 "fix1": "smoke_record_fix1.json", "reserve": "smoke_record_reserve.json"}      # as R6
SHA_HERE = None                 # the folder whose r9 files count (None: this one; a test points it at a tmp copy)
AGREEMENT_DIR = CM.HERE / "rederive" / "rd9_out"      # ticket 25's output folder (real mode)
LEDGER = CM.CHECKOUT / "docs/superpowers/held_ledger.md"
SIBLING = {"H9": "H6", "H10": "H7"}      # D30: the backbone's ArtELingo row decides its ArtELingo-28 row
# Rule 5.1 item 1: the 27 CSV names in Python sort order.
LANGUAGES = ("Burmese", "Darija", "Emakhuwa", "Hausa", "Hindi", "Igbo", "Indonesian", "IsiNdebele", "IsiXhosa", "IsiZulu",
             "Kazakh", "Kinyarwanda", "Korean", "Kyrgyz", "Malay", "Setswana", "Swahili", "Tagalog", "Tamil", "Thai",
             "Turkish", "Urdu", "Uzbek", "Vietnamese", "Yoruba", "arabic", "chinese")
assert len(LANGUAGES) == N_LANGUAGES == 27 and list(LANGUAGES) == sorted(LANGUAGES)
MAX_SET_ASIDE = 2               # per pair and seed (rule 5.1 item 6)
N_READ_PAINTINGS = 746
NOT_RUN = {"switched_off": "not run: switched off", "dropped": "dropped under the rule"}      # D25

# Rule 4: <name> of "AFF did not beat <name> on new paintings".
NAMES = {"P1": "cosine", "P2": "RCA", "P3": "B", "P4": "B′(A0)", "P5": "its matched control",
         "P6": "the condition-free scorers on condition gain", "P7": "RCA on condition gain"}
LABELS = {"P1": "R@1 against cosine", "P2": "R@1 against RCA", "P3": "R@1 against B", "P4": "R@1 against B′(A0)",
          "P5": "R@1 against the matched control", "P6": "condition gain against the condition-free scorers",
          "P7": "condition gain against RCA"}
# Rule 4.3, verbatim pieces (asserted against the rule's text on every run).
CLAIM7_A = "on ArtELingo's held paintings, AFF re-fitted by its fixed recipe on frozen "
CLAIM7_B = (" features beats cosine, RCA, B, B′(A0) and its matched control on R@1, with a positive condition gain "
            "over its control and RCA, pooled over three aspect pairs")
CLAIM7_NOT = "Not each pair, not another dataset, not a margin size."
CLAIM_FULL_A = "with captions from the {N} languages that entered the read"
CLAIM_FULL_B = ("machine-translated to English, AFF trained on English ArtELingo keeps the same sentence on that "
                "backbone.")
CLAIM_FULL = CLAIM_FULL_A + ", " + CLAIM_FULL_B
CLAIM_SMALL = "beats B′(A0) on R@1 with a positive condition gain over its matched control"
CLAIM_A28_NOT = "Not native-language text, not any single language."
LANGUAGE_LIST = ", ".join(LANGUAGES)      # rule 5.1 item 1, verbatim
RULE_PHRASES = (CLAIM7_A + "Qwen (or SigLIP 2)" + CLAIM7_B, CLAIM7_NOT, CLAIM_FULL_A.replace("{N}", "N"), CLAIM_FULL_B,
                CLAIM_SMALL, CLAIM_A28_NOT, NOT_RUN["switched_off"], NOT_RUN["dropped"], LANGUAGE_LIST)      # in this folder's rule
# Fixed phrases of R6 rule 4 (C11), which this rule cites for every reading.
R6_PHRASES = ('"AFF did not beat <name> on new paintings"', '"inconclusive at a detectable margin of x"',
              "the condition-free scorers on condition gain", '"RCA on condition gain"',
              '"not reached: the Holm procedure stopped at <earlier check>"',
              "cosine, RCA, B, B′(A0), its matched control")
HEX64 = re.compile(r"[0-9a-f]{64}")
SHA_RE = HEX64


class Stop(Exception):
    """A refusal (code 4) or a contradiction (code 5); nothing is written."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def refuse(msg):
    raise Stop(4, msg)


def contradiction(msg):
    raise Stop(5, msg)


# ---------------------------------------------------------------- Holm and the detectable margins (pure)

def families(read: str) -> list:
    """[(label, names, m, kind)]: one family for H6, H7; two for H8 to H10."""
    if READS[read]["design"] == "seven":
        return [("all", P_CHECKS, 7, "seven")]
    return [("level1", L1, 2, "level1"), ("level2", L2, 5, "level2")]


def kind_of(read: str, name: str) -> str:
    """The read kind (which z) of a check: seven, level1 or level2."""
    return next(kind for _, names, _, kind in families(read) if name in names)


def holm_ranks(counts: dict, names: tuple, m: int) -> list:
    """r9_stats.holm with n*_k and the 7.6 boundary flag of each rank."""
    out = ST.holm(counts, names, m)
    for e in out:
        e["n_star"] = ST.boundary(e["k"], m)
        e["near_boundary"] = ST.near_boundary(e["n"], e["k"], m)
    return out


def margins_from_se(se: float, kind: str) -> dict:
    return {"SE": se, "x": Z[kind] * se, "x95": Z95 * se}


def detectable_margins(sig: dict, M_p, N: int, kind: str) -> dict:
    """Rule 7.4: SE² = (σ_a²·Σ_p M_p² + σ_ε²·N) / N² (r9_stats.detectable), x = z(kind)·SE, x95 = 2.80·SE."""
    return margins_from_se(ST.detectable(sig, M_p, N, "P")["SE"], kind)


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
    r6 = CM.TEST / CM.RULE_REL
    if CM.sha256_file(r6) != CM.R6_RULE_SHA256:
        refuse(f"{r6.name} of round 6: its SHA-256 is not the committed rule's")
    r6_text = " ".join(r6.read_text(encoding="utf-8").split())
    missing += [p for p in R6_PHRASES if p not in r6_text]
    if missing:
        contradiction(f"{len(missing)} fixed phrase(s) of this script are not in the rule verbatim")


def _check_entry(rec: dict, name: str, where: str, level) -> None:
    c = rec.get(name)
    if not isinstance(c, dict):
        refuse(f"{where}: {name} is missing")
    ok = (isinstance(c.get("quantity"), str) and _is_int(c.get("n")) and 0 <= c["n"] <= N_BOOT
          and _is_num(c.get("point")) and _is_interval(c.get("ci95")) and _is_interval(c.get("ci_holm"))
          and _is_int(c.get("holm_k")) and _is_num(c.get("level_two_sided"))
          and type(c.get("near_boundary")) is bool and type(c.get("passes")) is bool
          and type(c.get("own_count_passes")) is bool)
    if level is not None:
        ok = ok and _is_int(c.get("level")) and c["level"] == level
    if not ok:
        refuse(f"{where}: {name} does not have the fields and types of the pass schema")


def load_pass(raw: bytes, name: str, read: str, smoke: bool) -> dict:
    """The pass file: this rule, this mode, this read's seeds and design, every field present."""
    design = READS[read]["design"]
    rec = _parse(raw, name)
    if rec.get("rule_sha256") != CM.RULE_SHA256:
        refuse(f"{name}: its rule_sha256 is not this rule's")
    mode = "smoke" if smoke else "held"
    if rec.get("mode") != mode:
        refuse(f"{name}: its mode is not {mode!r}")
    seeds = list(SMOKE_SEEDS if smoke else SEEDS[design])
    if rec.get("seeds") != seeds:
        refuse(f"{name}: its seeds are not {seeds}")
    if rec.get("n_episodes") != 9 * (CM.N_SMOKE if smoke else CM.N_PER_PAIR):
        refuse(f"{name}: n_episodes is not 9 x {CM.N_SMOKE if smoke else CM.N_PER_PAIR}")
    if (rec.get("design") == "two_level") is not (design == "two_level"):
        refuse(f"{name}: its design key does not fit {read}")
    checks = rec.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(P_CHECKS):
        refuse(f"{name}: checks are not exactly P1 to P7")
    for nm in P_CHECKS:
        _check_entry(checks, nm, name, None if design == "seven" else (1 if nm in L1 else 2))
    order = rec.get("holm_order")
    if design == "seven":
        if not (isinstance(order, list) and sorted(order) == list(P_CHECKS)):
            refuse(f"{name}: holm_order is not an ordering of P1 to P7")
    else:
        if not (isinstance(order, dict) and set(order) == {"level1", "level2"} and sorted(order["level1"]) == list(L1)
                and sorted(order["level2"]) == sorted(L2)):
            refuse(f"{name}: holm_order is not {{level1: L1 order, level2: L2 order}}")
        sa, fl = rec.get("set_aside"), rec.get("flagged_languages")
        if not (isinstance(sa, list) and all(
                isinstance(s, dict) and set(s) == {"language", "pair", "seed"} and isinstance(s["language"], str)
                and s["pair"] in CM.PAIR_NAMES and s["seed"] in seeds for s in sa)):
            refuse(f"{name}: set_aside is not a list of {{language, pair, seed}}")
        if not (isinstance(fl, list) and all(isinstance(s, str) for s in fl)):
            refuse(f"{name}: flagged_languages is not a list of strings")
        nd = rec.get("near_duplicates")
        if not (isinstance(nd, dict) and set(nd) == {"cos_099", "cos_098"} and _is_int(nd["cos_099"])
                and _is_int(nd["cos_098"]) and 0 <= nd["cos_099"] <= nd["cos_098"] <= N_READ_PAINTINGS):
            refuse(f"{name}: near_duplicates is not {{cos_099, cos_098}} with 0 <= cos_099 <= cos_098 <= "
                   f"{N_READ_PAINTINGS} (rule 3.3)")
        check_set_aside(sa, name)
    return rec


def check_set_aside(sa: list, name: str) -> None:
    """Rule 5.1 item 6: names among the 27, no duplicate entry, at most two languages per (pair, seed); else exit 5."""
    keys = [(s["language"], s["pair"], s["seed"]) for s in sa]
    outside = sorted({s["language"] for s in sa} - set(LANGUAGES))
    if outside:
        contradiction(f"{name}: set_aside names {outside}, which are not among the 27 languages of rule 5.1 item 1")
    if len(set(keys)) != len(keys):
        contradiction(f"{name}: set_aside has a duplicate (language, pair, seed) entry")
    per = {}
    for _, pair, seed in keys:
        per[(pair, seed)] = per.get((pair, seed), 0) + 1
    over = sorted(k for k, v in per.items() if v > MAX_SET_ASIDE)
    if over:
        contradiction(f"{name}: more than {MAX_SET_ASIDE} languages set aside for {over} (rule 5.1 item 6: the read "
                      f"stops before any score)")


def load_sensitivity(path: Path, read: str) -> tuple:
    """Rule 7.4: SE, x, x95 of P1 to P7, finite and non-negative; x and x95 equal z(kind)·SE and 2.80·SE."""
    raw = path.read_bytes()
    rec = _parse(raw, path.name)
    for nm in P_CHECKS:
        c = rec.get(nm)
        if not (isinstance(c, dict) and all(_is_num(c.get(k)) and c[k] >= 0 for k in ("SE", "x", "x95"))):
            refuse(f"{path.name}: {nm} lacks a finite non-negative SE, x, x95")
        want = margins_from_se(float(c["SE"]), kind_of(read, nm))
        for k in ("x", "x95"):
            if not math.isclose(c[k], want[k], rel_tol=1e-9, abs_tol=1e-12):
                contradiction(f"{path.name}: {nm} {k} is not {'z·SE for its read kind' if k == 'x' else '2.80·SE'}")
    return rec, CM.sha256_bytes(raw)


# ---------------------------------------------------------------- the rule (pure)

def assert_consistent(checks: dict, recomputed: list, what: str) -> None:
    """The pass file's Holm fields must equal the recomputation from its counts."""
    for e in recomputed:
        c = checks[e["name"]]
        diffs = [f for f, ok in (("holm_k", c["holm_k"] == e["k"]),
                                 ("level_two_sided", abs(c["level_two_sided"] - e["level_two_sided"]) <= 1e-12),
                                 ("near_boundary", c["near_boundary"] is e["near_boundary"]),
                                 ("passes", c["passes"] is e["passes"]),
                                 ("own_count_passes", c["own_count_passes"] is e["own_count_passes"])) if not ok]
        if diffs:
            contradiction(f"{what} {e['name']}: recorded {', '.join(diffs)} differ from the integer Holm recomputed "
                          f"from the counts")


def c11(name: str, c: dict, sens_c: dict) -> dict:
    """Rule 4 (C11) for one failed check: point > 0 inconclusive at x, else did not beat <name>."""
    if c["point"] > 0:
        x, x95 = float(sens_c["x"]), float(sens_c["x95"])
        hw95 = (c["ci95"][1] - c["ci95"][0]) / 2.0
        hwh = (c["ci_holm"][1] - c["ci_holm"][0]) / 2.0
        text = (f"inconclusive at a detectable margin of x = {x:.3f} points (x95 = {x95:.3f}; realised half-width "
                f"{hw95:.3f} at 95%, {hwh:.3f} at its Holm level)")
        return {"reading_kind": "inconclusive", "c11_reading": text, "x": x, "x95": x95,
                "half_width_95": hw95, "half_width_holm": hwh}
    return {"reading_kind": "did not beat", "c11_reading": f"AFF did not beat {NAMES[name]} on new paintings"}


def read_family(checks: dict, names: tuple, recomputed: list, sens: dict, level) -> dict:
    """Each tested check's pass, reading and "not reached" mark, in the family's canonical order."""
    stop = next((e["name"] for e in recomputed if not e["own_count_passes"]), None)
    out = {}
    for e in recomputed:
        nm, c = e["name"], checks[e["name"]]
        r = {"passes": e["passes"], "own_count_passes": e["own_count_passes"], "holm_k": e["k"], "n": e["n"],
             "n_star": e["n_star"], "near_boundary": e["near_boundary"], "quantity": c["quantity"],
             "label": LABELS[nm], "point": c["point"], "ci95": c["ci95"], "ci_holm": c["ci_holm"],
             "level_two_sided": c["level_two_sided"], "not_reached_after": None}
        if level is not None:
            r["level"] = level
        if e["passes"]:
            r.update(reading_kind="passed", reading="passed")
        else:
            r.update(c11(nm, c, sens[nm]))
            if e["own_count_passes"]:
                assert stop is not None and stop != nm
                r["not_reached_after"] = stop
                r["reading"] = (f"not reached: the Holm procedure stopped at {stop} ({LABELS[stop]}); its own reading: "
                                f"{r['c11_reading']}")
            else:
                r["reading"] = r["c11_reading"]
        out[nm] = r
    return {nm: out[nm] for nm in names}


def untested(checks: dict, names: tuple, level: int) -> dict:
    """Level-2 checks after a level-1 failure: point and 95% interval only, not tested, not read (rule 4.2)."""
    return {nm: {"quantity": checks[nm]["quantity"], "label": LABELS[nm], "level": level, "point": checks[nm]["point"],
                 "ci95": checks[nm]["ci95"], "passes": None, "reading_kind": None, "reading": None}
            for nm in names}


def _kind_of_failures(read_checks: dict, failed: list) -> str:
    return "inconclusive" if all(read_checks[nm]["reading_kind"] == "inconclusive" for nm in failed) else "did not beat"


def claim_a28(rec: dict, read: str, full: bool) -> tuple:
    """-> (claim text, N): rule 4.3 with N, the set-aside languages (with pair and seed), the flagged ones and the
    near-duplicate counts (rule 3.3). Both claims start "AFF on <backbone's ledger name>"."""
    sa = rec["set_aside"]
    n = N_LANGUAGES - len({s["language"] for s in sa})
    named = "; ".join(f"{s['language']} ({s['pair']}, seed {s['seed']})" for s in sa) or "none"
    flagged = ", ".join(rec["flagged_languages"]) or "none"
    nd = rec["near_duplicates"]
    bb = READS[read]["backbone"]
    head = f"AFF on {bb}: " + CLAIM_FULL.replace("{N}", str(n)) if full else (
        f"AFF on {bb} {CLAIM_SMALL} (captions from the {n} languages that entered the read, machine-translated to "
        f"English).")
    return (f"{head} Set-aside languages: {named}. Flagged languages: {flagged}. Near-duplicates (read paintings whose "
            f"image has a train-side image at CLIP cosine >= 0.99 / >= 0.98): {nd['cos_099']} / {nd['cos_098']} of "
            f"{N_READ_PAINTINGS}, none excluded. {CLAIM_A28_NOT}"), n


def decide(rec: dict, sens: dict, read: str, rederived: dict = None) -> dict:
    """Rule 4.1, 4.2 and 7.6 on a validated pass record and sensitivity record; `rederived` = the agreement's counts."""
    checks, short = rec["checks"], READS[read]["short"]
    boundary = []

    def tested_boundary(recomputed):
        boundary.extend({"check": e["name"], "holm_k": e["k"], "n": e["n"], "n_star": e["n_star"],
                         "passes": e["passes"], "rederivation_n": None if rederived is None else rederived[e["name"]]}
                        for e in recomputed if e["near_boundary"])

    if READS[read]["design"] == "seven":
        hp = holm_ranks({nm: checks[nm]["n"] for nm in P_CHECKS}, P_CHECKS, 7)
        assert_consistent(checks, hp, "check")
        if rec["holm_order"] != [e["name"] for e in hp]:
            contradiction("holm_order differs from the order recomputed from the counts (ties P1 to P7)")
        tested_boundary(hp)
        out = read_family(checks, P_CHECKS, hp, sens, None)
        failed = [nm for nm in P_CHECKS if not out[nm]["passes"]]
        go = not failed
        v = {"verdict": "GO" if go else "NO-GO", "kind": None if go else _kind_of_failures(out, failed),
             "checks": out, "holm_order": [e["name"] for e in hp], "failed": failed,
             "claim": (CLAIM7_A + short + CLAIM7_B + ". " + CLAIM7_NOT) if go else None}
    else:
        h1 = holm_ranks({nm: checks[nm]["n"] for nm in L1}, L1, 2)
        h2 = holm_ranks({nm: checks[nm]["n"] for nm in L2}, L2, 5)
        assert_consistent(checks, h1, "level-1 check")
        assert_consistent(checks, h2, "level-2 check")
        if rec["holm_order"] != {"level1": [e["name"] for e in h1], "level2": [e["name"] for e in h2]}:
            contradiction("holm_order differs from the orders recomputed from the counts (ties in the rule's order)")
        out = read_family(checks, L1, h1, sens, 1)
        tested_boundary(h1)
        l1_pass = all(e["passes"] for e in h1)
        l2_pass = None
        if l1_pass:                                       # serial gatekeeping: level 2 only after level 1
            out.update(read_family(checks, L2, h2, sens, 2))
            tested_boundary(h2)
            l2_pass = all(e["passes"] for e in h2)
        else:
            out.update(untested(checks, L2, 2))
        out = {nm: out[nm] for nm in P_CHECKS}
        failed = [nm for nm in P_CHECKS if out[nm]["passes"] is False]
        verdict = "full claim" if l2_pass else ("small claim" if l1_pass else "NO-GO")
        claim, n_lang = (claim_a28(rec, read, full=l2_pass) if l1_pass else (None, None))
        if n_lang is None:
            n_lang = N_LANGUAGES - len({s["language"] for s in rec["set_aside"]})
        v = {"verdict": verdict, "kind": _kind_of_failures(out, failed) if verdict == "NO-GO" else None,
             "checks": out, "holm_order": rec["holm_order"], "failed": failed,
             "level1": {"names": list(L1), "holm_order": rec["holm_order"]["level1"], "passes": l1_pass},
             "level2": {"names": list(L2), "holm_order": rec["holm_order"]["level2"], "tested": l1_pass,
                        "passes": l2_pass},
             "claim": claim, "n_languages": n_lang, "set_aside": rec["set_aside"],
             "flagged_languages": rec["flagged_languages"], "near_duplicates": rec["near_duplicates"]}
    for b in boundary:
        b["note"] = ("rule 7.6: reported to the user with the count and its n*; the integer rule decides")
    v["boundary_report"] = boundary
    return v


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


def load_agreement(path: Path, smoke: bool, pass_name: str, read: str) -> tuple:
    """Rule 7.4 ("as R6 rule 8.3"), 7.5: the phase-2 agreement record of the pass file `pass_name`, or refuse.
    -> (record, SHA-256 of the record file)."""
    if not path.is_file():  # GUARD: agreement-missing
        refuse(f"{path.name} is missing: the verdict is written only after the phase-2 agreement (rule 7.4)")
    raw = path.read_bytes()
    agr = _parse(raw, path.name)
    if not (_is_int(agr.get("phase")) and agr["phase"] == 2):  # GUARD: phase-2
        refuse(f"{path.name}: not a phase-2 record (rule 7.5)")
    if type(agr.get("smoke")) is not bool:
        refuse(f"{path.name}: its smoke flag is not a boolean")
    if agr["smoke"] is not smoke:  # GUARD: smoke-flag
        refuse(f"{path.name}: a {'real' if agr['smoke'] is False else 'smoke'} agreement record is refused in "
               f"{'smoke' if smoke else 'real'} mode")
    if agr.get("agreement") is not True:  # GUARD: agreement-true
        refuse(f"{path.name}: agreement is not true; trace the disagreement first (rule 7.5)")
    items, compared, failed = agr.get("items"), agr.get("compared"), agr.get("failed")
    if not (isinstance(items, list) and all(isinstance(x, dict) and type(x.get("equal")) is bool for x in items)):
        refuse(f"{path.name}: items is not a list of entries with a boolean equal")
    if [x for x in items if not x["equal"]]:  # GUARD: failing-items
        refuse(f"{path.name}: its items list has entries that are not equal")
    if not (_is_int(failed) and failed == 0):  # GUARD: failed-zero
        refuse(f"{path.name}: failed is not 0")
    if not (_is_int(compared) and compared >= 1 and compared == len(items)):  # GUARD: compared-count
        refuse(f"{path.name}: compared is not a positive integer equal to the length of items")
    if agr.get("read") != read:  # GUARD: read-name
        refuse(f"{path.name}: its read is not {read}")
    if agr.get("pass_file") != pass_name:  # GUARD: pass-file-name
        refuse(f"{path.name}: its pass_file is not {pass_name} (one agreement file per pass)")
    if not (isinstance(agr.get("held_pass_sha256"), str) and SHA_RE.fullmatch(agr["held_pass_sha256"])):
        refuse(f"{path.name}: held_pass_sha256 is not a SHA-256")
    rc = agr.get("rederived_counts")
    if not (isinstance(rc, dict) and set(rc) == set(P_CHECKS)
            and all(_is_int(rc[nm]) and 0 <= rc[nm] <= N_BOOT for nm in P_CHECKS)):  # GUARD: rederived-counts
        refuse(f"{path.name}: rederived_counts is not {{P1..P7: count}}")
    return agr, CM.sha256_bytes(raw)


def pair_suffix(out: Path, read: str, variant) -> str:
    """One pass file per verdict (R6 rule 8.1): "_reserve" for the reserve read; "_fix1" whenever
    held_pass_<Hn>_fix1.json exists (the fix pass replaces the original, which is then not read); "" otherwise."""
    if variant == "reserve":
        return "_reserve"
    if (out / f"held_pass_{read}_fix1.json").exists():  # GUARD: fix1-pair
        return "_fix1"
    return ""


def latest_smoke_record(results: Path, variant) -> Path:
    """The smoke record covering the code run now (as R6): reserve: smoke_record_reserve.json; otherwise
    smoke_record_fix1.json if it exists, else smoke_record_crash1.json if it exists, else smoke_record.json."""
    if variant == "reserve":
        return results / SMOKE_RECORDS["reserve"]
    for kind in ("fix1", "crash1"):
        if (results / SMOKE_RECORDS[kind]).is_file():
            return results / SMOKE_RECORDS[kind]
    return results / SMOKE_RECORDS["smoke"]


def smoke_guard(results: Path, variant) -> dict:
    """Real mode (rule 4.1 "6 passed", 6 item 5): the latest smoke record passed and its module_sha256 equals
    r9_module_shas() key for key, or refuse. -> {"name", "sha256"} of the record."""
    path = latest_smoke_record(results, variant)
    if not path.is_file():  # GUARD: smoke-record
        refuse(f"{path.name} is missing: the step runs only on code that passed the smoke (rule 6 item 5)")
    raw = path.read_bytes()
    rec = _parse(raw, path.name)
    if rec.get("passed") is not True:  # GUARD: smoke-passed
        refuse(f"{path.name} records a smoke that did not pass (rule 6 item 5)")
    mods = rec.get("module_sha256") if isinstance(rec.get("module_sha256"), dict) else {}
    cur = CM.r9_module_shas(SHA_HERE)
    diff = sorted(k for k in set(mods) | set(cur) if mods.get(k) != cur.get(k))
    if not mods or diff:  # GUARD: smoke-shas
        refuse(f"{path.name}: the SHA-256s of {diff or 'every module'} differ from the smoke's: a new smoke is needed "
               f"(rule 6 item 5)")
    return {"name": path.name, "sha256": CM.sha256_bytes(raw)}


def ledger_row(ledger: Path, read: str) -> str:
    """The text of ledger row `read` (all its cells), or refuse."""
    if not ledger.is_file():
        refuse(f"{ledger.name} is missing")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and cells[0] == read:
            return " | ".join(cells)
    refuse(f"row {read} is not in {ledger.name}: a not-run verdict is written only after its ledger row (rule 7.1)")


def apply_not_run(read: str, not_run: str, out: Path, smoke: bool, ledger: Path, base: dict, vpath: Path) -> tuple:
    """D25, D30, rule 7.1, 7.3: a not-run verdict only with its evidence."""
    ppath = out / f"held_pass_{read}.json"
    if ppath.exists() or (out / f"held_pass_{read}_fix1.json").exists():
        refuse(f"{ppath.name} exists: the read was run, it is not '{NOT_RUN[not_run]}'")
    if (out / f"held_started_{read}.json").exists():
        refuse(f"held_started_{read}.json exists: the read was started, it is not '{NOT_RUN[not_run]}'")
    if not_run == "switched_off":
        if read == "H8":
            refuse("H8 reads CLIP, which cannot be switched off (D30)")
        if read in SIBLING:
            sib = out / f"held_verdict_{SIBLING[read]}.json"
            sv = _parse(sib.read_bytes(), sib.name) if sib.is_file() else {}
            if sv.get("status") != NOT_RUN["switched_off"]:
                refuse(f"{read} is switched off only if {SIBLING[read]} was (D30); {sib.name} does not read "
                       f"'{NOT_RUN['switched_off']}'")
    if not smoke:
        row = ledger_row(ledger, read)
        other = NOT_RUN["dropped" if not_run == "switched_off" else "switched_off"]
        if NOT_RUN[not_run] not in row or other in row:
            refuse(f"ledger row {read} does not read exactly '{NOT_RUN[not_run]}'")
    v = {**base, "verdict": None, "kind": None, "status": NOT_RUN[not_run], "claim": None}
    write_once(vpath, v)
    return vpath, v


def apply(read: str, smoke: bool = False, variant=None, not_run=None, results=None, agreement_dir=None,
          ledger=None) -> tuple:
    """Every refusal, then the verdict record written once. -> (path, record)."""
    if read not in READS:
        refuse(f"unknown read {read!r}; expected one of {sorted(READS)}")
    if variant not in VARIANTS:
        refuse(f"unknown variant {variant!r}; a fix pass needs no variant (held_pass_<Hn>_fix1.json is used when it "
               f"exists)")
    if not_run is not None and (not_run not in NOT_RUN or variant is not None):
        refuse(f"--not-run takes {sorted(NOT_RUN)} and no variant")
    out = Path(results) if results is not None else (SMOKE if smoke else RESULTS)
    vpath = out / f"held_verdict_{read}{VARIANTS[variant]}.json"
    if vpath.exists():  # GUARD: verdict-exists
        refuse(f"{vpath.name} exists; a verdict is never overwritten (rule 7.2)")
    if variant == "reserve" and not (out / f"held_verdict_{read}.json").is_file():  # GUARD: reserve-needs-original
        refuse(f"a reserve verdict follows the original held_verdict_{read}.json, which is missing")
    base = {"read": read, "backbone": READS[read]["backbone"], "design": READS[read]["design"], "variant": variant,
            "mode": "smoke" if smoke else "held", "rule_sha256": CM.RULE_SHA256, "module_sha256": CM.r9_module_shas(),
            "time": CM.amsterdam_now()}
    if not_run is not None:
        assert_rule()
        return apply_not_run(read, not_run, out, smoke, Path(ledger) if ledger is not None else LEDGER, base, vpath)
    smoke_record = None if smoke else smoke_guard(out, variant)
    assert_rule()
    sfx = pair_suffix(out, read, variant)
    ppath = out / f"held_pass_{read}{sfx}.json"
    adir = Path(agreement_dir) if agreement_dir is not None else (out if smoke else AGREEMENT_DIR)
    apath = adir / f"rd9_{read}_agreement{sfx}.json"
    agr, agr_sha = load_agreement(apath, smoke, ppath.name, read)
    if not ppath.is_file():
        refuse(f"{ppath.name}, named by {apath.name}, is missing")
    raw = ppath.read_bytes()
    pass_sha = CM.sha256_bytes(raw)
    if agr["held_pass_sha256"] != pass_sha:  # GUARD: sha-binding
        refuse(f"{apath.name} checked another pass: its held_pass_sha256 is not the SHA-256 of {ppath.name}")
    if sfx == "_fix1" and not (out / f"held_pass_{read}.json").is_file():  # GUARD: fix1-original
        refuse(f"{ppath.name} exists, but the original held_pass_{read}.json it corrects is missing (rule 7.3)")
    rec = load_pass(raw, ppath.name, read, smoke)
    rd = agr["rederived_counts"]
    for nm in P_CHECKS:
        if rd[nm] != rec["checks"][nm]["n"]:  # GUARD: counts-equal
            contradiction(f"{apath.name}: agreement is true, but the re-derived count of {nm} ({rd[nm]}) is not the pass "
                          f"file's ({rec['checks'][nm]['n']})")
    spath = out / f"sensitivity_{read}{sfx}.json"
    if sfx == "_fix1" and not spath.is_file():             # a fix pass reuses the original sensitivity (as R6)
        spath = out / f"sensitivity_{read}.json"
    if not spath.is_file():
        refuse(f"{spath.name} is missing (rule 7.4)")
    sens, sens_sha = load_sensitivity(spath, read)
    v = {**base, **decide(rec, sens, read, rd), "status": "run", "seeds": rec["seeds"],
         "held_pass_sha256": pass_sha, "pass_file": ppath.name, "fix_pass": sfx == "_fix1",
         "agreement_file": apath.name, "agreement_sha256": agr_sha,
         "agreement": {k: agr.get(k) for k in ("phase", "smoke", "agreement", "read", "pass_file", "held_pass_sha256",
                                               "compared")},
         "sensitivity_file": spath.name, "sensitivity_sha256": sens_sha, "smoke_record": smoke_record}
    write_once(vpath, v)
    return vpath, v


def say(msg: str) -> None:
    print(f"[run_r9_apply_rule] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True, choices=sorted(READS, key=lambda r: int(r[1:])))
    ap.add_argument("--variant", choices=("reserve",), default=None, help="the reserve read (_reserve names)")
    ap.add_argument("--smoke", action="store_true", help="read and write results/smoke/ (smoke files only)")
    ap.add_argument("--not-run", choices=sorted(NOT_RUN), default=None, dest="not_run")
    args = ap.parse_args(argv)
    try:
        path, v = apply(args.read, smoke=args.smoke, variant=args.variant, not_run=args.not_run,
                        results=SMOKE if args.smoke else RESULTS)
    except Stop as e:
        say(("REFUSED: " if e.code == 4 else "STOP, report to the user: ") + e.msg)
        raise SystemExit(e.code)
    if v["status"] != "run":
        say(f"{args.read}: {v['status']}")
    elif not args.smoke:
        say(f"{args.read} verdict: {v['verdict']}" + (f" ({v['kind']})" if v["kind"] else ""))
        for nm, c in v["checks"].items():
            say(f"{nm}: " + ("not tested" if c["passes"] is None else "passed" if c["passes"] else
                             f"failed, {c['reading_kind']}"
                             + (f", not reached after {c['not_reached_after']}" if c["not_reached_after"] else "")))
        for b in v["boundary_report"]:
            say(f"boundary (rule 7.6, the integer rule decides): {b['check']} count {b['n']}, "
                f"largest passing count {b['n_star']}")
    say(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
