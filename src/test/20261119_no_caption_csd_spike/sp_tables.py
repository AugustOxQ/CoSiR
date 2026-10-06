import json, sys
from pathlib import Path
R = json.loads((Path(__file__).parent / "results/sp_results.json").read_text())
P = ["emotion__style", "emotion__genre", "style__genre"]
f = lambda x: f"{x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
out = []
w = out.append
w("checks: " + json.dumps(R["checks"]))
w(f"B {R['B_r1_mean']:.3f}; B' {R['bprime_r1_means']}")
for k, s in R["summaries"].items():
    w(f"\n## {k}  comparator={s['bar']['comparator']} means={ {a: round(b,3) for a,b in s['bar']['comparator_mean_r1'].items()} }")
    w(f"pooled: fused {s['r1_means']['fused']:.3f} cf {s['r1_means']['counterpart']:.3f} B {s['r1_means']['B']:.3f} B' {s['r1_means']['B_prime']:.3f}")
    w(f"  margin r1 {f(s['margin']['r1'])} gain {f(s['margin']['gain'])} either {f(s['margin']['either'])}")
    w(f"  bar {f(s['bar']['r1'])} gainstat {f(s['gain_statistic'])} clears {s['clears_bar']['clears_bar']}")
    for p in P:
        pp = s['per_pair'][p]; m = s['pair_r1_means'][p]
        w(f"  {p}: fused {m['fused']:.2f} cf {m['counterpart']:.2f} B {m['B']:.2f} B' {m['B_prime']:.2f} | margin {f(pp['margin']['r1'])} gain {f(pp['margin']['gain'])} either {f(pp['margin']['either'])} | bar {f(s['bar']['per_pair_r1'][p])}")
    pa = s['pick_accuracy']
    w(f"  pick acc {f(pa['correct_share'])} both {pa['both_correct_share']:.1f}; " + "; ".join(f"{p} {v['a']:.1f}/{v['b']:.1f}" for p, v in pa['per_pair_condition'].items()))
    w("  pick share overall " + ", ".join(f"{h} {v:.1f}" for h, v in s['pick_share']['overall'].items()))
    for p in P:
        w(f"    {p}: " + " | ".join(f"{c}: " + ",".join(f"{h} {v:.0f}" for h, v in d.items()) for c, d in s['pick_share']['per_pair_condition'][p].items()))
    c = s['cells']
    w("  cells fused " + "; ".join(f"h{h}: k{v['k_top']} tau{v['tau_index']} lu{v['lambda_u']} la{v['lambda_a']}" for h, v in c['fused'].items()))
    gs = s['gate_open_share']
    for h, v in c['fused'].items():
        t = f"tau_{v['tau_index']}"
        w(f"  gate open (half {h} chosen {t}): overall {gs[t]['overall']:.1f} " + " ".join(f"{p} {gs[t]['per_pair'][p]:.1f}" for p in P))
for k, d in R["diffs"].items():
    w(f"\n## {k}")
    for m, v in d.items():
        w(f"  {m}: " + "; ".join(f"{p} {f(x)}" for p, x in v.items()))
Path(__file__).parent.joinpath("results/sp_tables.txt").write_text("\n".join(out))
print("\n".join(out))
