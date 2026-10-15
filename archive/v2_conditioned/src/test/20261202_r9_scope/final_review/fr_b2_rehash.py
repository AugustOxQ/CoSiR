"""Final review B2, check 2: re-hash every file pins_clip.json lists (own code, no r9_* import).
input_sha256 and parts paths are relative to <main>/src/test/ (parts) or src/test/ (inputs); the cells file lives
in the snapshot's round folder; gate_references are round 6's results."""
import hashlib, json, sys
from pathlib import Path
MAIN = Path("/project/CoSiR/src/test")
SNAP = Path("/project/CoSiR-wt/r9-fr-b2/src/test")
pins = json.loads((SNAP / "20261202_r9_scope/pins_clip.json").read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
def where(rel):
    rel = rel.removeprefix("src/test/")
    for base in (SNAP, MAIN):            # round 9's own files: the snapshot; earlier rounds: the main checkout
        if rel.startswith("20261202_r9_scope/") and base is SNAP and (base / rel).is_file():
            return base / rel
        if (base / rel).is_file() and not rel.startswith("20261202_r9_scope/"):
            return MAIN / rel
    return None
out = {"input_sha256": {}, "parts": {}, "gate_references": {}}
bad = []
for rel, want in pins["input_sha256"].items():
    p = where(rel); got = sha(p) if p else None
    out["input_sha256"][rel] = got == want
    if got != want: bad.append(("input", rel, str(p), got, want))
for role, rec in pins["parts"].items():
    p = where(rec["path"]); got = sha(p) if p else None
    out["parts"][role] = got == rec["sha256"]
    if got != rec["sha256"]: bad.append(("part", role, str(p), got, rec["sha256"]))
for name, want in pins["gate_references"].items():
    p = MAIN / "20261125_artelingo_held_test/results" / name; got = sha(p) if p.is_file() else None
    out["gate_references"][name] = got == want
    if got != want: bad.append(("ref", name, str(p), got, want))
# the parts' SHA-256s against the input set where both list the same file
cross = [(r, rec["path"]) for r, rec in pins["parts"].items()
         if rec["path"].removeprefix("src/test/") in pins["input_sha256"]
         and pins["input_sha256"][rec["path"].removeprefix("src/test/")] != rec["sha256"]]
summary = {k: f"{sum(v.values())}/{len(v)}" for k, v in out.items()}
res = {"summary": summary, "mismatches": bad, "parts_vs_inputs_disagree": cross,
       "parts_not_in_inputs": sorted(r for r, rec in pins["parts"].items()
                                     if rec["path"].removeprefix("src/test/") not in pins["input_sha256"])}
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1)[:3000])
