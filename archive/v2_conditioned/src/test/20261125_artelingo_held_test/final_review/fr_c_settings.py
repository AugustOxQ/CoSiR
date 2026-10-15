"""fr_c: rule section 7 text vs dts_settings.json, character by character (reviewer C)."""
import json, re, sys
from pathlib import Path
F = Path(__file__).resolve().parents[1]
rule = (F / "DECISION_RULE.md").read_text()
s = json.loads((F / "dts_settings.json").read_text())
out = {}
# wordings from the rule's table: | W1 | "..." |
for m in re.finditer(r'^\s*\| (W[1-4]) \| "(.*)" \|$', rule, re.M):
    wid, text = m.group(1), m.group(2)
    out[wid] = (text == s["verbaliser"]["wordings"][wid])
# listing prompt
m = re.search(r'is asked: "(List K distinct values.*?)" \(K written', rule, re.S)
lp = " ".join(m.group(1).split())
want = lp.replace("List K", "List {K}").replace("<phrase>", "{phrase}")
out["listing_prompt"] = (want == s["listing"]["prompt"])
m = re.search(r'list marker \(`(.*?)`\)', rule)
out["marker_regex"] = (m.group(1) == s["listing"]["marker_regex"])
g = s["generation"]
out["tokens_32_128"] = (g["verbaliser"]["max_new_tokens"] == 32 and g["listing"]["max_new_tokens"] == 128)
out["K"] = s["listing"]["K"] == [8, 16]
out["model"] = s["model"]["id"] == "Qwen/Qwen3-VL-8B-Instruct"
out["dtsn_names"] = s["controls"]["DTS-N"]["phrase_of_target_aspect"] == {"emotion": "emotion", "style": "style", "genre": "genre"}
out["aff_hits"] = s["stop"]["aff_hits_seed42"] == 9406 and s["stop"]["n_rankings_seed42"] == 49152
out["order"] = s["seed42_order"]["tuning_settings"] == [f"W{w} K{k}" for w in range(1, 5) for k in (8, 16)]
out["first_1024"] = s["seed42_order"]["tuning_subset_first_per_pair"] == 1024
out["budget"] = s["budget"]["hours"] == 24
out["max_pixels"] = s["model"]["processor_kwargs"]["max_pixels"] == 256 * 28 * 28
print(json.dumps(out))
sys.exit(0 if all(out.values()) else 1)
