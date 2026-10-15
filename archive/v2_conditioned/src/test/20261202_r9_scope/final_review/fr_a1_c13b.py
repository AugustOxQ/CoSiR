"""Final review A1, item 3b: the pass-file mutations the comparer does not see, and a sensitivity mutation, through
the apply step (agreement remade by ticket 25's comparer after each mutation). -> fr_a1_c13b.json"""
import copy
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_a1_apply as A  # noqa: E402
import fr_a1_lib as L  # noqa: E402

out = {}
for design, read in (("seven", "H6"), ("two_level", "H9")):
    D = A.Data(design)
    specs = {**{c: 0 for c in L.CHECKS}, "P3": 300} if design == "seven" else {**{c: 0 for c in L.CHECKS}, "P2": 300}
    rec, own, sens = D.build(specs)
    muts = {"mode": lambda r, s: r.__setitem__("mode", "regression"),
            "rule_sha256": lambda r, s: r.__setitem__("rule_sha256", "0" * 64),
            "quantity": lambda r, s: r["checks"]["P3"].__setitem__("quantity", "R@1, AFF - B0"),
            "SE x2 (x, x95 consistent)": lambda r, s: s.__setitem__(
                "P3" if design == "seven" else "P2",
                {k: 2 * v for k, v in s["P3" if design == "seven" else "P2"].items()})}
    if design != "seven":
        muts.update({"level swap": lambda r, s: r["checks"]["P4"].__setitem__("level", 2),
                     "design drop": lambda r, s: r.pop("design")})
    for name, m in muts.items():
        d = A.SCR / f"c13b_{design}_{name.split()[0]}"
        if d.exists():
            shutil.rmtree(d)
        r, s = copy.deepcopy(rec), copy.deepcopy(sens)
        m(r, s)
        items = None
        if design != "seven":
            items = {"set_aside": ([], A.RC.reported_set_aside(r["set_aside"])), "flagged_languages": ([], [])}
        agr = A.stage(d, read, r, own, s, items)
        try:
            _, v = A.AR.apply(read, results=d, agreement_dir=d, ledger=d / "ledger.md")
            res = {"written": True, "verdict": v["verdict"],
                   "reading": v["checks"]["P3" if design == "seven" else "P2"].get("reading")}
        except A.AR.Stop as e:
            res = {"written": False, "stop": [e.code, e.msg]}
        out[f"{design}: {name}"] = {"agreement": agr["agreement"], **res}
(HERE / "fr_a1_c13b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out.items():
    print(k, v)
