"""Final review A2, item 4: delete each guard (and each guard call site) of my scope in place, run the tests that
cover the file, revert at once (bytes and SHA-256 checked). Resumable in time-boxed batches.

  python final_review/fr_a2_mut.py list                 # enumerate mutants -> fr_a2_mut_plan.json
  python final_review/fr_a2_mut.py run SECONDS [IDS..]  # run pending mutants until SECONDS have passed
Results: fr_a2_mut_results.json (one entry per mutant: caught, first failing test or 'survived').
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
PY = "/root/miniconda3/envs/CoSiR/bin/python"
PLAN = HERE / "fr_a2_mut_plan.json"
RESULTS = HERE / "fr_a2_mut_results.json"
TESTS = {
    "run_r9_held.py": ["test_r9_held.py", "test_r9_held_a28.py", "test_r9_held_shared.py", "test_r9_descriptive.py"],
    "r9_score.py": ["test_r9_held_shared.py", "test_r9_score.py"],
    "r9_a28_episodes.py": ["test_r9_a28_episodes.py", "test_r9_held_a28.py", "test_r9_a28_bundle.py"],
    "r9_a28_context.py": ["test_r9_a28_bundle.py", "test_r9_held_a28.py"],
    "r9_episodes.py": ["test_r9_a28_episodes.py", "test_r9_held.py", "test_r9_episodes.py"],
    "r9_bundle.py": ["test_r9_a28_bundle.py", "test_r9_held_a28.py"],
    "run_r9_descriptive.py": ["test_r9_descriptive.py"],
    "r9_descriptive.py": ["test_r9_descriptive.py"],
}
FAST = {
    "run_r9_held.py": ["test_r9_held.py", "test_r9_held_a28.py"],
    "r9_score.py": ["test_r9_held_shared.py"],
    "r9_a28_episodes.py": ["test_r9_a28_episodes.py", "test_r9_held_a28.py"],
    "r9_a28_context.py": ["test_r9_a28_bundle.py"],
    "r9_episodes.py": ["test_r9_a28_episodes.py", "test_r9_held.py"],
    "r9_bundle.py": ["test_r9_a28_bundle.py"],
    "run_r9_descriptive.py": ["test_r9_descriptive.py"],
    "r9_descriptive.py": ["test_r9_descriptive.py"],
}
SET = FAST if os.environ.get("FR_FULL") != "1" else TESTS
# call sites of guard functions (Expr statements only): name patterns
CALL_SITE = re.compile(r"^\s*(?:[A-Z]{1,2}\.)?(?:\w+_guard|check_pins|assert_seeds_free|assert_distinct|"
                       r"assert_held_eligible|validate_aspect_episodes|check_readers|check_seed42|_pinned)\(")
BOUNDARY = re.compile(r"^\s*(?:D\.)?(?:refuse_unless|require)\(")      # the descriptive runner has no markers
SIMPLE = (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Raise, ast.Return, ast.Assert)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    out = []
    for fname in TESTS:
        src = (F / fname).read_text()
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.stmt):
                continue
            if isinstance(node, SIMPLE):
                end = lines[node.end_lineno - 1]
                m = re.search(r"# guard:(\w+)", end)
                kind = None
                if m:
                    kind = "marker"
                elif isinstance(node, ast.Expr) and (
                        (fname == "run_r9_held.py" and CALL_SITE.match(lines[node.lineno - 1]))
                        or (fname in ("run_r9_descriptive.py", "r9_descriptive.py")
                            and (CALL_SITE.match(lines[node.lineno - 1]) or BOUNDARY.match(lines[node.lineno - 1])))):
                    kind = "call_site"
                if kind is None:
                    continue
                if isinstance(node, ast.Assign) and kind == "marker":
                    kind = "marker_assign"
                name = m.group(1) if m else lines[node.lineno - 1].strip()[:60]
                out.append({"file": fname, "guard": name, "kind": kind, "lineno": node.lineno,
                            "end": node.end_lineno})
            elif isinstance(node, (ast.If, ast.While)):
                m = re.search(r"# guard:(\w+)", lines[node.lineno - 1])
                if m:
                    out.append({"file": fname, "guard": m.group(1), "kind": "if_false", "lineno": node.lineno,
                                "end": node.lineno})
                    if isinstance(node.body[0], ast.Continue):
                        out.append({"file": fname, "guard": m.group(1), "kind": "if_true", "lineno": node.lineno,
                                    "end": node.lineno})
    for i, x in enumerate(out):
        x["id"] = i
    PLAN.write_text(json.dumps(out, indent=1) + "\n")
    print(len(out), "mutants;", {k: sum(1 for x in out if x["kind"] == k) for k in {x["kind"] for x in out}})


def mutate(src, x):
    lines = src.splitlines(keepends=True)
    first = lines[x["lineno"] - 1]
    ind = first[:len(first) - len(first.lstrip())]
    marker = f"  # guard:{x['guard']}" if x["kind"] != "call_site" else ""
    if x["kind"] in ("marker", "call_site"):
        lines[x["lineno"] - 1] = ind + "None" + marker + "\n"
        for i in range(x["lineno"], x["end"]):
            lines[i] = "\n"
    elif x["kind"] == "marker_assign":
        tgt = first.strip().split("=")[0].strip()
        lines[x["lineno"] - 1] = ind + f"{tgt} = {{'passed': True, 'n_seeds': 81}}" + marker + "\n"
        for i in range(x["lineno"], x["end"]):
            lines[i] = "\n"
    elif x["kind"] in ("if_false", "if_true"):
        word = "if" if first.lstrip().startswith("if") else "while"
        rest = first.split("# guard:")[1]
        lines[x["lineno"] - 1] = ind + f"{word} {x['kind'] == 'if_true'}:  # guard:" + rest
    new = "".join(lines)
    ast.parse(new)
    return new


def run(budget, ids):
    todo = json.loads(PLAN.read_text())
    done = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    for x in todo:
        prev = done.get(str(x["id"]))
        full = os.environ.get("FR_FULL") == "1"
        if (prev is not None and not prev.get("timeout") and (prev["caught"] or not full or prev.get("full"))) \
                or (ids and x["id"] not in ids):
            continue
        if time.time() - t0 > budget:
            break
        path = F / x["file"]
        orig = path.read_bytes()
        h0 = sha(path)
        try:
            path.write_text(mutate(orig.decode(), x))
            t1 = time.time()
            try:
                p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *SET[x["file"]]],
                                   cwd=F, env=env, capture_output=True, text=True, timeout=130)
            except subprocess.TimeoutExpired:
                p = subprocess.CompletedProcess([], -9, "TIMEOUT after 130 s", "")
            tail = (p.stdout or "").strip().splitlines()
            failed = [ln for ln in tail if ln.startswith(("FAILED", "ERROR"))]
            res = {**x, "rc": p.returncode, "caught": p.returncode not in (0, -9), "timeout": p.returncode == -9,
                   "first_failure": failed[0][:200] if failed else None, "summary": tail[-1][:160] if tail else "",
                   "seconds": round(time.time() - t1), "full": SET is TESTS}
        finally:
            path.write_bytes(orig)
            assert sha(path) == h0, f"revert failed for {path}"
        done[str(x["id"])] = res
        RESULTS.write_text(json.dumps(done, indent=1) + "\n")
        print(x["id"], x["file"], x["guard"], x["kind"], "caught" if res["caught"] else "SURVIVED", res["seconds"],
              flush=True)
    left = sum(1 for x in todo if (str(x["id"]) not in done or done[str(x["id"])].get("timeout"))
               and (not ids or x["id"] in ids))
    print("left", left)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        plan()
    else:
        run(float(sys.argv[2]), [int(i) for i in sys.argv[3:]])
