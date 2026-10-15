"""Final review: apply_rule.py's item 3-5 logic on synthetic candidate files (scratch copy; originals untouched)."""
import importlib.util, json, shutil, sys, tempfile
from pathlib import Path
RD = Path(__file__).resolve().parent.parent
def run(rows):
    d = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent / "out"))
    shutil.copy(RD / "apply_rule.py", d / "apply_rule.py"); shutil.copy(RD / "DECISION_RULE.md", d / "DECISION_RULE.md")
    (d / "results").mkdir()
    for n, (bar, lo, glo) in rows.items():
        (d / "results" / f"cand_{n}.json").write_text(json.dumps({"bar": {"r1": {"point": bar, "ci95": [lo, 1.0]}, "comparator": "B_prime"},
                                                                  "gain_statistic": {"point": 1.0, "ci95": [glo, 2.0]}}))
    spec = importlib.util.spec_from_file_location("ar", d / "apply_rule.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        m.main()
    r = json.loads((d / "results" / "rule_application.json").read_text()); shutil.rmtree(d); return r["carried"], r["eligible"]
base = {n: (0.1, -0.1, 0.2) for n in ("Ra_A1", "Rb_argmax_A1", "Rb_expected_A1", "Ra_A0", "Rb_argmax_A0", "Rb_expected_A0")}
cases = {
 "A1 priority over a larger A0": ({**base, "Rb_argmax_A1": (0.51, 0.01, 0.2), "Rc_Rb_expected_A0": (0.9, 0.3, 0.2)}, "Rb_argmax_A1"),
 "A0 pool, tie within 0.05 to the earlier reader": ({**base, "Ra_A0": (0.55, 0.1, 0.2), "Rb_expected_A0": (0.59, 0.1, 0.2), "Rc_Rb_expected_A0": (0.62, 0.1, 0.2)}, "Rb_expected_A0"),
 "A1 tie at exactly M-0.05": ({**base, "Ra_A1": (0.55, 0.1, 0.2), "Rb_expected_A1": (0.60, 0.1, 0.2), "Rc_Rb_expected_A1": (0.4, 0.1, 0.2)}, "Ra_A1"),
 "clause 3 fails": ({**base, "Ra_A1": (0.7, 0.1, 0.0), "Rc_Ra_A1": (0.4, 0.1, 0.2)}, None),
 "clause 1 at exactly 0.5 clears": ({**base, "Rb_argmax_A0": (0.5, 0.01, 0.01), "Rc_Rb_argmax_A0": (0.4, 0.1, 0.2)}, "Rb_argmax_A0"),
}
for k, (rows, want) in cases.items():
    got = run(rows); print(f"{k:50s} carried {got[0]!s:16s} expected {want!s:16s} {'OK' if got[0] == want else 'MISMATCH'}")
