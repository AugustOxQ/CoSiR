"""Round 6: the descriptive pass, core (DECISION_RULE.md of this folder: section 10 item 5, section 8 item 3, section 5
item 5; ticket 13). Runs only after held_verdict.json records the phase-2 agreement; decides nothing.

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_descriptive.py \
        [--smoke [--smoke-subdir NAME] | --reserve] > <log> 2>&1

In this order (results/, or results/smoke/ with --smoke, results/smoke/NAME/ with --smoke-subdir NAME):
  1. Refusals (exit 4; nothing loaded, nothing written):
       - held_verdict.json (--reserve: held_verdict_reserve.json) is missing; or written under another rule; or its
         mode is not this run's ("held" in real mode, so a smoke verdict is refused, and the reverse); or its verdict
         is not GO or NO-GO;
       - its agreement file (the verdict's "agreement_file": rederive_agreement[_fix1|_reserve].json) is missing, or
         its SHA-256 is not the verdict's agreement_sha256, or it is not a phase-2 record with all_agree true and no
         disagreements, or its smoke flag is not this run's, or it names another pass file;
       - the pass file (the verdict's "pass_file") is missing, or its SHA-256 differs from the verdict's and the
         agreement's held_pass_sha256, or it is of another rule, mode or seed list;
       - descriptive.json (descriptive_reserve.json) exists: it is written once.
       - the head-coefficient records (rule section 5 item 5) are missing or unusable: refit_check.json (always
         results/, the stage-1a record) has not passed or holds no coef_sha256; in real and reserve mode also
         held_started.json (held_started_reserve.json) of this folder holds no attempt of this rule that produced the
         pass (the last attempt whose "fix" flag matches the pass file: set for held_pass_fix1.json, unset otherwise)
         with its coef_sha256.
       - in real and reserve mode, external_sources.json of the results folder is missing or lacks one of its four
         entries dts, ft_lp, ft, mllm (a job not run is written {"missing": "<reason>"}; r6_external.require_sources).
       - in real and reserve mode, the latest smoke record of the results folder did not pass or its module_sha256
         is not r6_common.r6_module_shas() now (rule section 6 item 7 smokes the descriptive pass too; ticket 15):
         smoke_record_reserve.json with --reserve; otherwise smoke_record_fix1.json when it exists, else
         smoke_record_crash1.json when it exists, else smoke_record.json.
  2. Inputs, bound to the pass the verdict rests on (exit 5 on a contradiction; nothing written):
       - held_arrays<sfx>.npz (sfx of the pass file: "", "_fix1", "_reserve"; contracts section 7) reproduces the
         pass file's n_j, point and 95% interval of P1 to P7, S1, S2 exactly (r6_descriptive.check_pass);
       - held_episodes_seed<s>.npz (the held runner's episode files): each pair's SHA-256 equals the pass file's
         episodes_sha256;
       - picks_seed42.json (always results/, a passed stage-1a record) and the frozen lambdas (baselines_seed42.json).
  3. Setup: run_r6_held.setup() (inputs asserted, data, split, heads refit and checked bit for bit, PM fits, readers);
     a head check that did not pass is a contradiction (exit 5). The heads' coefficient SHA-256s must equal
     refit_check.json's and, in real and reserve mode, the producing attempt's in held_started.json (rule section 5
     item 5); otherwise refused (exit 4) before any bundle is built. All three are recorded in descriptive.json.
  4. Per seed: the seed's context and bundle (held: held_bundle(env, seed); smoke: selection mode, 64 per pair; both
     by seed_ctx_bundle, run_r6_held.seed_bundle's two calls keeping the RowContext), then
     r6_descriptive.seed_inputs: score_seed(..., include_pm=True) (the nine PM scorers and R1's counterpart are
     scored only here, after the verdict, rule section 8 item 3), the bundle's episodes equal to the file's, its
     eight core scorers equal to held_arrays.npz bit for bit; then ticket 14's per-seed hook external_seed_rows(env,
     seed, ctx, bundle, episodes, out, smoke) while ctx and bundle exist: the external rows DTS, DTS-CF, DTS-N,
     FT-LP, FT-LB, FT-LoRA and (first seed only) MLLM, from the GPU outputs named by external_sources.json in the
     results folder, joined to the episodes by their keys only here (r6_external.py: sources, joins, "missing" rows).
     Both are dropped before the next seed.
  5. r6_descriptive.describe (with the hook's rows, through external_rows) -> descriptive.json, written once, with
     the verdict's and the inputs' SHA-256s, the coefficient SHA-256s, module_sha256, the frozen picks, the time, and
     the development figure of plan section 10's item-reuse rate (seed 42, selection rows) beside the held one.
Prints only pass words and file paths (no metric, no decimal number), in both modes; in smoke mode a refusal or stop
message has every decimal number replaced by "#".

Exit codes: 0 written; 4 refused; 5 the held arrays, the episode files or a rebuilt bundle contradict the pass the
verdict rests on, or another assertion failed (stop and report to the user).

Guards carry a `# guard:<name>` marker; test_r6_descriptive.py deletes each on a copy and shows that its scenario then
goes through.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_descriptive as D  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_external as XT  # noqa: E402
import r6_picks as P  # noqa: E402
import r6_score as S  # noqa: E402
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402

EXIT_REFUSE = 4
EXIT_CONTRADICTION = 5
VERDICTS = ("GO", "NO-GO")
AGREEMENT_RE = re.compile(r"rederive_agreement(_fix1|_reserve)?\.json")
PASS_RE = re.compile(r"held_pass(_fix1|_reserve)?\.json")
SUBDIR_RE = re.compile(r"[A-Za-z0-9_]+")
DECIMAL = re.compile(r"\d*\.\d+")
REFIT_NAME = RH.REFIT_NAME                          # refit_check.json (stage 1a, results/)
SMOKE_RECORDS = R.SMOKE_RECORDS                     # run_r6_smoke.py's records (ticket 15)
SHA_HERE = None                     # the folder whose r6 files count (None: this one; a test points it at a tmp copy)


class Refused(Exception):
    """A precondition of rule section 8 item 3 is not met (exit 4)."""


class Contradiction(Exception):
    """The inputs contradict the pass the verdict rests on (exit 5)."""


def refuse(msg):
    raise Refused(msg)


def say(msg):
    print(f"[run_r6_descriptive] {msg}", flush=True)


def numberless(msg) -> str:
    """A message with every decimal number replaced by "#" (smoke paths print no decimal number)."""
    return DECIMAL.sub("#", str(msg))


def _plain(x):
    """The value as JSON reads it back (keys strings, tuples lists, numpy scalars plain)."""
    return json.loads(json.dumps(R.C.jsonable(x)))


def runner_sha256() -> str:
    return R.sha256_file(Path(__file__).resolve())


def _parse(raw: bytes, what: str) -> dict:
    def no_constant(c):
        raise ValueError(f"non-finite constant {c}")
    try:
        rec = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    except (ValueError, UnicodeDecodeError) as e:
        refuse(f"{what} is not valid JSON ({type(e).__name__})")
    if not isinstance(rec, dict):
        refuse(f"{what} is not a JSON object")
    return rec


def out_dir(smoke: bool, subdir=None, base=None) -> Path:
    """results/, results/smoke/ or results/smoke/<subdir>/ (a repeated smoke, as run_r6_apply_rule)."""
    if base is not None:
        return Path(base)
    if subdir is None:
        return R.SMOKE if smoke else R.RESULTS
    if not smoke:
        refuse("--smoke-subdir is for the smoke chain only")
    if not SUBDIR_RE.fullmatch(str(subdir)):
        refuse("--smoke-subdir must be a plain folder name (letters, digits, underscore)")
    return R.SMOKE / subdir


def mode_of(smoke: bool) -> SimpleNamespace:
    return SimpleNamespace(name="smoke" if smoke else "held",
                           seeds=list(R.SMOKE_SEEDS if smoke else R.HELD_SEEDS),
                           n_per_pair=R.N_SMOKE if smoke else R.N_PER_PAIR)


# ---------------------------------------------------------------- 1. refusals

def check_verdict(out, smoke: bool, reserve: bool) -> SimpleNamespace:
    """Rule section 8 item 3: the verdict file exists and records the phase-2 agreement of its own pass. Raises
    Refused otherwise. -> SimpleNamespace(verdict, sha256, path, agreement, pass_rec, pass_path, suffix, output)."""
    out, md = Path(out), mode_of(smoke)
    vpath = out / f"held_verdict{'_reserve' if reserve else ''}.json"
    if not vpath.is_file():
        refuse(f"{vpath.name} is missing: the descriptive pass runs only after the verdict (rule section 8 item "
               f"3)")  # guard:verdict_missing
    vraw = vpath.read_bytes()
    v = _parse(vraw, vpath.name)
    if v.get("rule_sha256") != R.RULE_SHA256:
        refuse(f"{vpath.name} was written under another rule")  # guard:verdict_rule
    if v.get("mode") != md.name:
        refuse(f"{vpath.name} is a {v.get('mode')!r} verdict; {'smoke' if smoke else 'real'} mode reads only a "
               f"{md.name!r} verdict")  # guard:verdict_mode
    if v.get("verdict") not in VERDICTS or bool(v.get("reserve")) is not reserve:
        refuse(f"{vpath.name} holds no GO or NO-GO verdict of this kind")  # guard:verdict_value
    aname, pname = v.get("agreement_file"), v.get("pass_file")
    am = AGREEMENT_RE.fullmatch(aname) if isinstance(aname, str) else None
    pm = PASS_RE.fullmatch(pname) if isinstance(pname, str) else None
    if not (am and pm and am.group(1) == pm.group(1) and (pm.group(1) == "_reserve") is reserve):
        refuse(f"{vpath.name} names no valid agreement and pass file pair")  # guard:file_names
    suffix = pm.group(1) or ""
    apath, ppath = out / aname, out / pname
    if not apath.is_file():
        refuse(f"{aname} is missing: the verdict's phase-2 agreement cannot be checked")  # guard:agreement_missing
    araw = apath.read_bytes()
    if R.sha256_bytes(araw) != v.get("agreement_sha256"):
        refuse(f"{aname}: its SHA-256 is not the verdict's agreement_sha256")  # guard:agreement_sha
    agr = _parse(araw, aname)
    if not (type(agr.get("phase")) is int and agr["phase"] == 2 and agr.get("all_agree") is True
            and agr.get("disagreements") == []):
        refuse(f"{aname} does not record a phase-2 agreement (rule section 8 item 4)")  # guard:agreement_all_agree
    if agr.get("smoke") is not smoke:
        refuse(f"{aname}: a {'smoke' if agr.get('smoke') else 'real'} agreement record is refused in "
               f"{'smoke' if smoke else 'real'} mode")  # guard:agreement_smoke
    if agr.get("pass_file") != pname:
        refuse(f"{aname} names another pass file than the verdict's")  # guard:agreement_pass_file
    if not ppath.is_file():
        refuse(f"{pname}, named by the verdict, is missing")  # guard:pass_missing
    praw = ppath.read_bytes()
    psha = R.sha256_bytes(praw)
    if not (psha == v.get("held_pass_sha256") == agr.get("held_pass_sha256")):
        refuse(f"{pname}: its SHA-256 is not the one the verdict and the agreement checked")  # guard:pass_sha
    prec = _parse(praw, pname)
    if not (prec.get("rule_sha256") == R.RULE_SHA256 and prec.get("mode") == md.name
            and prec.get("seeds") == md.seeds):
        refuse(f"{pname} is not a {md.name} pass of this rule on seeds {md.seeds}")  # guard:pass_mode
    output = out / f"descriptive{'_reserve' if reserve else ''}.json"
    if output.exists():
        refuse(f"{output.name} exists; it is written once")  # guard:output_exists
    return SimpleNamespace(verdict=v, sha256=R.sha256_bytes(vraw), path=vpath, agreement=agr, agreement_path=apath,
                           agreement_sha256=v["agreement_sha256"], pass_rec=prec, pass_path=ppath, pass_sha256=psha,
                           suffix=suffix, output=output, reserve=reserve)


def coef_records(out, smoke: bool, vr, records) -> SimpleNamespace:
    """Step 1's last refusals: the records the heads' coefficient SHA-256s are checked against (rule section 5 item 5).
    refit_check.json (in ``records``, results/ by default) passed and holds coef_sha256; in real and reserve mode the
    started file of this folder holds this rule's attempt that produced the pass, with its coef_sha256. Raises
    Refused. -> SimpleNamespace(refit, refit_path, refit_sha256, attempt, started_path, started_sha256)."""
    rpath = Path(records) / REFIT_NAME
    if not rpath.is_file():
        refuse(f"{REFIT_NAME} is missing: the heads' coefficient SHA-256s cannot be checked (rule section 5 item "
               f"5)")  # guard:refit_missing
    rraw = rpath.read_bytes()
    refit = _parse(rraw, REFIT_NAME)
    if not (refit.get("passed") is True and isinstance(refit.get("coef_sha256"), dict) and refit["coef_sha256"]):
        refuse(f"{REFIT_NAME} did not pass or holds no coef_sha256 (rule section 6 item 2)")  # guard:refit_passed
    res = SimpleNamespace(refit=refit, refit_path=rpath, refit_sha256=R.sha256_bytes(rraw), attempt=None,
                          started_path=None, started_sha256=None)
    if smoke:
        return res
    spath = Path(out) / f"held_started{'_reserve' if vr.reserve else ''}.json"
    if not spath.is_file():
        refuse(f"{spath.name} is missing: the read's coefficient SHA-256s cannot be checked")  # guard:started_missing
    sraw = spath.read_bytes()
    st = _parse(sraw, spath.name)
    fix = vr.suffix == "_fix1"
    atts = st.get("attempts") if isinstance(st.get("attempts"), list) else []
    mine = [a for a in atts if isinstance(a, dict) and isinstance(a.get("flags"), dict)
            and bool(a["flags"].get("fix")) is fix]
    if not (st.get("rule_sha256") == R.RULE_SHA256 and mine and isinstance(mine[-1].get("coef_sha256"), dict)
            and mine[-1]["coef_sha256"]):
        refuse(f"{spath.name} holds no attempt of this rule that produced {vr.pass_path.name}, with its "
               f"coef_sha256")  # guard:started_attempt
    res.attempt, res.started_path, res.started_sha256 = mine[-1], spath, R.sha256_bytes(sraw)
    return res


def coef_guard(env, coef) -> dict:
    """Step 3 (rule section 5 item 5): the heads' coefficient SHA-256s equal refit_check.json's and, in real and
    reserve mode, the producing attempt's. Raises Refused (before any bundle). -> the record for descriptive.json."""
    got = _plain(getattr(env, "coef_sha256", None))
    if got != _plain(coef.refit["coef_sha256"]):
        refuse(f"the heads' coefficient SHA-256s differ from {REFIT_NAME}'s (rule section 5 item 5)")  # guard:coef_refit
    if coef.attempt is not None and got != _plain(coef.attempt["coef_sha256"]):
        refuse(f"the heads' coefficient SHA-256s differ from those of attempt {coef.attempt.get('attempt')} in "
               f"{coef.started_path.name} (rule section 5 item 5)")  # guard:coef_started
    return {"heads": got, "refit_check": {"file": REFIT_NAME, "sha256": coef.refit_sha256, "equal": True},
            "held_started": (None if coef.attempt is None else
                             {"file": coef.started_path.name, "sha256": coef.started_sha256,
                              "attempt": coef.attempt.get("attempt"), "equal": True})}


def latest_smoke_record(out, reserve: bool) -> Path:
    """r6_common.latest_smoke_record (shared with the held runner and the apply step), in the results folder:
    --reserve: smoke_record_reserve.json; otherwise smoke_record_fix1.json when it exists, else
    smoke_record_crash1.json when it exists, else smoke_record.json."""
    return R.latest_smoke_record(out, reserve=reserve)


def smoke_guard(out, smoke: bool, reserve: bool, here=None):
    """Step 1 in real and reserve mode (rule section 6 item 7; contracts section 8 amendment 12:20, ticket 15): the
    latest smoke record passed and its module_sha256 equals r6_module_shas(), key for key. Raises Refused; -> {"file",
    "sha256"} of the record (None in smoke mode)."""
    if smoke:
        return None
    path = latest_smoke_record(out, reserve)
    if not path.is_file():
        refuse(f"{path.name} is missing: the descriptive pass runs only on code that passed the smoke (rule section 6 "
               f"item 7)")  # guard:smoke_record
    raw = path.read_bytes()
    rec = _parse(raw, path.name)
    if rec.get("passed") is not True:
        refuse(f"{path.name} records a smoke that did not pass (rule section 6 item 7)")  # guard:smoke_record
    mods = rec.get("module_sha256") if isinstance(rec.get("module_sha256"), dict) else {}
    cur = R.r6_module_shas(here or SHA_HERE)
    diff = sorted(k for k in set(mods) | set(cur) if mods.get(k) != cur.get(k))
    if not mods or diff:
        refuse(f"{path.name}: the SHA-256s of {diff or 'every module'} differ from the smoke's: a new smoke is needed "
               f"(rule section 6 item 7)")  # guard:smoke_shas
    return {"file": path.name, "sha256": R.sha256_bytes(raw)}


# ---------------------------------------------------------------- 2. inputs bound to the pass

def load_inputs(out, smoke: bool, vr) -> SimpleNamespace:
    """held_arrays<sfx>.npz (checked against the pass file) and the episode files (their hashes against the pass
    file's). Raises Contradiction on any difference."""
    out, md = Path(out), mode_of(smoke)
    apath = out / f"held_arrays{vr.suffix}.npz"
    if not apath.is_file():
        raise Contradiction(f"{apath.name} is missing: the pass the verdict rests on has no per-anchor arrays")
    try:
        core = D.load_core(apath, md.seeds, md.n_per_pair)
        agree = D.check_pass(core, vr.pass_rec)
    except AssertionError as e:
        raise Contradiction(f"{apath.name}: {e}") from None
    eps, sha = {}, {apath.name: R.sha256_file(apath)}
    for s in md.seeds:
        path = out / f"held_episodes_seed{s}.npz"
        if not path.is_file():
            raise Contradiction(f"{path.name} is missing")
        try:
            ep = E.load_episodes(path)
        except AssertionError as e:
            raise Contradiction(f"{path.name}: {e}") from None
        want = vr.pass_rec.get("episodes_sha256", {}).get(str(s))
        if not (ep.seed == s and ep.n == len(R.PAIRS) * md.n_per_pair and dict(ep.sha) == want):
            raise Contradiction(f"{path.name}: seed, size or per-pair hashes differ from the pass "
                                f"file's")  # guard:episodes_pass
        eps[s] = ep
        sha[path.name] = R.sha256_file(path)
    return SimpleNamespace(core=core, episodes=eps, sha256=sha, arrays_reproduce_pass=agree)


# ---------------------------------------------------------------- 4. contexts, bundles and ticket 14's hooks

def seed_ctx_bundle(env, mode, seed, n_per_pair):
    """run_r6_held.seed_bundle's two calls (the same RowContext on env's data, split, labels, heads and value sets,
    then build_bundle_r6 with env's readers and PM fits; a test pins the calls to seed_bundle's source), keeping the
    RowContext for ticket 14's per-seed hook (DTS needs its masked img, txt and its cos). -> (ctx, bundle)."""
    ctx = X.RowContext(mode, seed, env.data, env.split, env.labels, env.heads, env.value_sets, n_per_pair)
    return ctx, B.build_bundle_r6(ctx, env.readers, env.pm)


def held_bundle(env, seed):
    """One held seed's RowContext and bundle, rebuilt after the verdict on the same held rows (split.held, 4,096 per
    pair), heads, readers and PM fits as the held pass, as run_r6_held builds the read's bundles (seed_bundle(env,
    "held", seed, 4096, on_episodes=...)); no episode recorder here. The descriptive pass never calls held mode's entry
    of run_r6_held (no started file, no ledger, no pass). -> (ctx, bundle)."""
    return seed_ctx_bundle(env, "held", seed, R.N_PER_PAIR)


def bundle_factory(env, smoke: bool):
    """seed -> (ctx, bundle): held seeds in real mode, selection rows at 64 per pair in smoke mode."""
    if smoke:
        return lambda seed: seed_ctx_bundle(env, "selection", seed, R.N_SMOKE)
    return lambda seed: held_bundle(env, seed)


def sources_check(out, smoke: bool) -> None:
    """Step 1's last refusal (controller, ticket 14's review): in real and reserve mode external_sources.json and its
    four entries must exist (r6_external.require_sources); refused (exit 4) before any input is loaded."""
    try:
        XT.require_sources(out, smoke)
    except XT.SourcesRefused as e:
        refuse(str(e))


def external_seed_rows(env, seed, ctx, bundle, episodes, out, smoke: bool) -> dict:
    """Ticket 14's per-seed hook, called inside the seed loop after seed_inputs, while the seed's RowContext ``ctx``
    (masked img and txt, cos, pooled, parity, pair_index: what r6_dts.held_dts_scores takes) and its bundle exist;
    ``episodes`` is the seed's episode file (r6_episodes.load_episodes), ``out`` the results folder (its
    external_sources.json names the GPU outputs to join). -> {name: {"pa": per_anchor dict of this seed (or None: no
    row this seed), "label": str, "info": dict}} for DTS, DTS-CF, DTS-N, FT-LP, FT-LB, FT-LoRA and, on the mode's
    first seed only, MLLM (r6_external.seed_rows). Prints the names only."""
    try:
        rows = XT.seed_rows(env, seed, ctx, episodes, out, smoke)
    except XT.SourcesRefused as e:
        refuse(str(e))
    have = [n for n, x in rows.items() if x.get("pa") is not None]
    say(f"seed {seed}: external rows {', '.join(have) or 'none'}; missing "
        f"{', '.join(n for n in rows if n not in have) or 'none'}")
    return rows


def collect_external(collected, seed, rows) -> dict:
    """Adds one seed's hook output to describe's external argument {name: {"pa": {seed: per_anchor}, "label", "info":
    {"per_seed": {seed: info}}}}."""
    for name, x in (rows or {}).items():
        e = collected.setdefault(name, {"pa": {}, "label": x.get("label", name), "info": {"per_seed": {}}})
        if x.get("pa") is not None:
            e["pa"][int(seed)] = x["pa"]
        if x.get("info"):
            e["info"]["per_seed"][str(seed)] = dict(x["info"])
    return collected


def external_rows(env, per_seed, out, smoke: bool, collected) -> dict:
    """Ticket 14's final hook: describe's external argument from the per-seed rows ``collected``
    (r6_external.finish): the seven external rows in order, a row not computed on every seed it is reported on marked
    "missing" with its reason and no number, and each row's sources and inputs (totals of parsing failures, the
    phrases' most frequent wordings, checkpoints, files and their SHA-256s)."""
    return XT.finish(env, out, smoke, collected)


def development_reuse(groups) -> dict:
    """Plan section 10's item-reuse rate "per split", the development split's figure: seed 42's selection episodes
    (r6_descriptive.development_episodes: AB's episodes_seed42.npz, SHA-256 and per-pair hashes asserted)."""
    ep = D.development_episodes()
    return {"split": "selection rows, seed 42 (development)",
            **D.item_reuse(ep.anchor, ep.candidates, D.member_rows(ep), groups)}


# ---------------------------------------------------------------- the run

def _lambdas_json(lambdas) -> dict:
    return {name: {str(h): ("inf" if np.isinf(v) else float(v)) for h, v in lam.items()}
            for name, lam in lambdas.items()}


def write_once(path: Path, rec: dict) -> None:
    """Write the complete file under its name only if no file of that name exists (atomic, never replaces)."""
    data = json.dumps(rec, indent=1, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(data, encoding="utf-8")
    try:
        os.link(tmp, path)
    except FileExistsError:
        refuse(f"{path.name} exists; it is written once")
    finally:
        tmp.unlink()


def after_verdict(out, smoke, vr, env=None, bundle_fn=None, picks=None, lambdas=None, records=None) -> Path:
    """Steps 1 (coefficient records) to 5 (module docstring). records: the folder of refit_check.json and
    picks_seed42.json (results/ by default). -> the written path."""
    t0 = time.time()
    md = mode_of(smoke)
    records = Path(records or R.RESULTS)
    coef = coef_records(out, smoke, vr, records)
    sources_check(out, smoke)
    inp = load_inputs(out, smoke, vr)
    picks_sha = None
    if picks is None:
        picks_path = records / P.PICKS_NAME
        picks = P.load_picks(picks_path)
        picks_sha = R.sha256_file(picks_path)
    lambdas = P.frozen_lambdas() if lambdas is None else lambdas
    env = env or RH.setup()
    if not (isinstance(getattr(env, "head_check", None), dict) and env.head_check.get("passed") is True):
        raise Contradiction("the refit heads' selection posteriors differ from the stored ones (rule section 6 item "
                            "2): the bundles cannot be the read's")  # guard:head_check
    coef_rec = coef_guard(env, coef)
    bundle_fn = bundle_fn or bundle_factory(env, smoke)
    per_seed, collected = [], {}
    for s in md.seeds:
        ctx, bundle = bundle_fn(s)
        try:
            d = D.seed_inputs(bundle, picks, lambdas, env.readers, inp.core[s], inp.episodes[s], env.split.groups)
        except AssertionError as e:
            raise Contradiction(f"seed {s}: {e}") from None
        per_seed.append(d)
        collect_external(collected, s, external_seed_rows(env, s, ctx, bundle, inp.episodes[s], out, smoke))
        del ctx, bundle
        gc.collect()
        say(f"seed {s}: rebuilt bundle reproduces the held arrays; descriptive inputs done")
    rec = {"mode": md.name,
           **D.describe(per_seed, external=external_rows(env, per_seed, out, smoke, collected))}
    rec["item_reuse"]["development_seed42"] = development_reuse(env.split.groups)
    rec.update({
        "n_per_pair": md.n_per_pair,
        "verdict": {"file": vr.path.name, "sha256": vr.sha256, "verdict": vr.verdict["verdict"],
                    "kind": vr.verdict.get("kind"), "pass_file": vr.pass_path.name,
                    "held_pass_sha256": vr.pass_sha256, "agreement_file": vr.agreement_path.name,
                    "agreement_sha256": vr.agreement_sha256},
        "coef_sha256": coef_rec,
        "consistency": {"arrays_reproduce_pass": inp.arrays_reproduce_pass,
                        "bundle_reproduces_arrays": {str(d.seed): d.checks["bundle_reproduces_arrays"]
                                                     for d in per_seed},
                        "episodes_match_files": {str(d.seed): d.checks["episodes_match_file"] for d in per_seed}},
        "frozen": {"cells": {who: {part: list(c) for part, c in v.items()} for who, v in S.CELLS.items()},
                   "picks": {name: {str(h): [float(x) for x in picks[name][h]] for h in (0, 1)} for name in P.NESTED},
                   "lambdas": _lambdas_json(lambdas)},
        "input_sha256": {**dict(getattr(env, "inputs", {}) or {}), **inp.sha256,
                         **({P.PICKS_NAME: picks_sha} if picks_sha else {}),
                         D.DEV_EPISODES_REL: R.assert_input(D.DEV_EPISODES_REL)},
        "smoke_record": getattr(vr, "smoke_record", None),
        "rule_sha256": R.RULE_SHA256, "module_sha256": R.r6_module_shas(), "runner_sha256": runner_sha256(),
        "time": R.amsterdam_now(), "runtime_s": round(time.time() - t0, 1),
    })
    write_once(vr.output, rec)
    return vr.output


def run(smoke=False, reserve=False, subdir=None, out=None, env=None, bundle_fn=None, picks=None, lambdas=None,
        records=None) -> int:
    """The whole pass. out, env, bundle_fn (seed -> (ctx, bundle)), picks, lambdas and records (the folder of
    refit_check.json and picks_seed42.json) may be given by a test. -> exit code. In smoke mode a refusal or stop
    message prints no decimal number."""
    smoke, reserve = bool(smoke), bool(reserve)
    shown = numberless if smoke else str
    try:
        if smoke and reserve:
            refuse("--smoke and --reserve exclude each other")
        out = out_dir(smoke, subdir, out)
        vr = check_verdict(out, smoke, reserve)
        vr.smoke_record = smoke_guard(out, smoke, reserve)
        path = after_verdict(out, smoke, vr, env, bundle_fn, picks, lambdas, records)
    except Refused as e:
        say(f"REFUSED: {shown(e)}")
        return EXIT_REFUSE
    except Contradiction as e:
        say(f"STOP, report to the user: {shown(e)}")
        return EXIT_CONTRADICTION
    except AssertionError as e:
        say(f"STOP, report to the user: an assertion failed: {shown(e)}")
        return EXIT_CONTRADICTION
    say(f"written: {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="the smoke chain (results/smoke/, smoke verdict only)")
    g.add_argument("--reserve", action="store_true", help="after the reserve verdict (held_verdict_reserve.json)")
    ap.add_argument("--smoke-subdir", default=None, metavar="NAME", help="with --smoke: results/smoke/NAME/")
    args = ap.parse_args(argv)
    return run(smoke=args.smoke, reserve=args.reserve, subdir=args.smoke_subdir)


if __name__ == "__main__":
    sys.exit(main())
