import json
from pathlib import Path
H = Path(__file__).parent
r = {"check1": {"cluster": json.load(open(H / "results_check1_cluster.json")), "probe": json.load(open(H / "results_check1_probe.json"))},
     "check2": json.load(open(H / "results_check2.json")), "check3": json.load(open(H / "results_check3.json")),
     "check4": json.load(open(H / "results_check4.json"))}
json.dump(r, open(H / "results.json", "w"), indent=1)
c = r["check1"]["cluster"]
print({k: c[k] for k in c if k != "cluster"})
print("%-16s %3s  %-14s %-14s %-14s %-14s" % ("feat", "k", "cap_emo", "paint_emo", "genre", "style"))
for key, v in c["cluster"].items():
    n, k = key.rsplit("_k", 1)
    print("%-16s %3s  " % (n, k) + " ".join("%.3f+-%.3f " % tuple(v[m]) for m in ("caption_AMI_emotion", "painting_AMI_emotion_majority", "painting_AMI_genre", "painting_AMI_style")))
for k, v in r["check4"]["cluster"].items(): print("goemo", k, v)
