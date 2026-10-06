"""Read-only: compare every SHA-256 the draft rule states with the file's actual SHA-256."""
import hashlib, re, sys
from pathlib import Path
ROOT = Path("/project/CoSiR")
rule = (ROOT / "src/test/20261118_reader_fix_round2/DECISION_RULE.md").read_text()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
R1 = ROOT / "src/test/20261117_reader_fix_csd"
bad = 0
# D14 table rows: | `path` | sha | ...
for m in re.finditer(r"^\| `([^`]+)` \| ([0-9a-f]{64}) \|", rule, re.M):
    p, s = m.group(1), m.group(2)
    got = sha(R1 / p)
    ok = got == s
    bad += not ok
    print(("OK  " if ok else "BAD ") + p, "" if ok else f"rule {s} actual {got}")
# inline SHA-256 mentions: `path` (SHA-256 xxx) and 'path ... SHA-256\n xxx'
inline = re.findall(r"`([^`]+\.(?:npz|pt|md))`[^`]{0,40}?SHA-256\s+([0-9a-f]{64})", rule, re.S)
for p, s in inline:
    q = ROOT / p if (ROOT / p).exists() else None
    if q is None:
        print("??  ", p); continue
    got = sha(q); ok = got == s; bad += not ok
    print(("OK  " if ok else "BAD ") + p, "" if ok else f"rule {s} actual {got}")
# spec and round-1 rule in the header
for p, s in (("docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md", "ceecc515beac5402156b1c4ee613e55308c4862fb1d8da6963f07c64301e11c8"),
             ("src/test/20261117_reader_fix_csd/DECISION_RULE.md", "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c")):
    got = sha(ROOT / p); ok = got == s; bad += not ok
    print(("OK  " if ok else "BAD ") + p + " (header)")
print("all hashes in the rule:", len(set(re.findall(r"[0-9a-f]{64}", rule))))
print("mismatches:", bad)
