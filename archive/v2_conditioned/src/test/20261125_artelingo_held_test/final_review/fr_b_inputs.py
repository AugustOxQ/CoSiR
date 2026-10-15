"""Final review B, check 1 (rule section 6 item 1): every input SHA-256 the rule names, derived from the rules' own text
and the files on disk, against what r6_common asserts (INPUT_SHA256) and where each is read (INPUT_PATHS).

Own code: the SHA-256 tables are parsed from the r6 rule and the R3 rule text (every `path` | 64-hex pair and every
"`path` (64-hex)" pair), the files are hashed here; r6_common is imported only to read its constants.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

SNAP = Path("/project/CoSiR-r6-fr")
MAIN = Path("/project/CoSiR")
F = SNAP / "src/test/20261125_artelingo_held_test"
TEST = MAIN / "src/test"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def table_pairs(text):
    """(path, sha) from table rows `| \\`path\\` | sha |` and from "`path` (sha)" / "path`, SHA-256 sha" mentions."""
    out = {}
    for m in re.finditer(r"\|\s*`([^`]+)`\s*\|\s*([0-9a-f]{64})\s*\|", text):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(r"`([^`]+\.(?:npz|json|md|pt|py))`\s*\(\s*([0-9a-f]{64})\s*\)", text):
        out[m.group(1)] = m.group(2)
    return out


def main():
    sys.path.insert(0, str(F))
    import r6_common as R
    rule = (F / "DECISION_RULE.md").read_text()
    r3_rule_path = TEST / "20261121_round3_affect_gate/DECISION_RULE.md"
    r3_rule = r3_rule_path.read_text()
    res = {}
    # the rule file itself and the R3 rule (named in the precedence paragraph with its SHA)
    res["rule_file_sha"] = sha(F / "DECISION_RULE.md")
    res["rule_sha_const_ok"] = res["rule_file_sha"] == R.RULE_SHA256
    m = re.search(r"SHA-256\s*\n?\s*([0-9a-f]{64})", rule)
    res["r3_rule_sha_in_rule"] = m.group(1)
    res["r3_rule_file_sha_ok"] = sha(r3_rule_path) == m.group(1) == R.R3_RULE_SHA256
    # R3 rule D15 table (+ told_oracle.json named in D15's text)
    d15 = table_pairs(r3_rule[r3_rule.index("**D15. Inputs.**"):])
    told = re.search(r"told_oracle\.json` \(SHA-256\s*\n?\s*([0-9a-f]{64})", r3_rule)
    rows = {}
    for p, s in d15.items():
        rows[p] = s
    rows["20261111_community_told_oracle/results/told_oracle.json"] = told.group(1)
    # D15's table continues to the end of the D15 block; restrict to paths under src/test (no slash-less names)
    rows = {p: s for p, s in rows.items() if "/" in p}
    res["d15_rows_parsed"] = len(rows)
    # the r6 rule section 6 item 1 extras, 5.1's prepare.npz, 6.4's seed42_arrays.npz
    sec61 = rule[rule.index("1. **Inputs.**"):rule.index("2. **Head refits.**")]
    extras = {}
    for m in re.finditer(r"`([^`]+)` \(([0-9a-f]{64})\)", sec61):
        extras[m.group(1)] = m.group(2)
    # resolve the two bare file names of section 6.1 against the step-1 folder
    extras = {(k if k.startswith("2026") else "20261116_grouping_step1_style/results/" + k): v for k, v in extras.items()}
    prep = re.search(r"`(src/test/20261013_stage_d_selection/cache/prepare\.npz)`,\s*\n?\s*SHA-256\s*\n?\s*([0-9a-f]{64})", rule)
    extras[prep.group(1).removeprefix("src/test/")] = prep.group(2)
    arr = re.search(r"seed42_arrays\.npz` \(SHA-256\s*\n?\s*([0-9a-f]{64})\)", rule)
    extras["20261121_round3_affect_gate/results/seed42_arrays.npz"] = arr.group(1)
    res["extras_parsed"] = extras
    want = {**rows, **extras}
    bad, missing_in_code, wrong_path = [], [], []
    for rel, s in sorted(want.items()):
        disk = sha(TEST / rel)
        code = R.INPUT_SHA256.get(rel)
        if code is None:
            missing_in_code.append(rel)
        elif code != s:
            bad.append((rel, "code", code, s))
        if disk != s:
            bad.append((rel, "disk", disk, s))
        if code is not None and Path(R.INPUT_PATHS[rel]).resolve() != (TEST / rel).resolve():
            wrong_path.append((rel, str(R.INPUT_PATHS[rel])))
    res["n_rule_inputs"] = len(want)
    res["mismatches"] = bad
    res["missing_in_code"] = missing_in_code
    res["wrong_path"] = wrong_path
    # anything the code asserts beyond the rule: hash on disk equals the code's value
    beyond = {}
    for rel, s in R.INPUT_SHA256.items():
        if rel in want or rel in (R.RULE_REL, R.R3_RULE_REL):
            continue
        beyond[rel] = sha(R.INPUT_PATHS[rel]) == s
    res["beyond_rule_disk_ok"] = beyond
    res["rule_paths"] = {R.RULE_REL: str(R.INPUT_PATHS[R.RULE_REL]), R.R3_RULE_REL: str(R.INPUT_PATHS[R.R3_RULE_REL])}
    out = Path(sys.argv[1])
    out.write_text(json.dumps(res, indent=1))
    ok = (res["rule_sha_const_ok"] and res["r3_rule_file_sha_ok"] and not bad and not missing_in_code
          and not wrong_path and all(beyond.values()))
    print(f"inputs: {len(want)} rule inputs (D15 {len(rows)}), mismatches {len(bad)}, missing {len(missing_in_code)}, "
          f"wrong paths {len(wrong_path)}, beyond-rule {sum(beyond.values())}/{len(beyond)} ok -> "
          f"{'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
