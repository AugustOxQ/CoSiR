"""Compare two render_check JSONs: per field, do all prompts match, and the first difference.
Exit 0 iff everything except the version strings matches.  Usage: compare_render.py a.json b.json"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
FIELDS = ("images_rel", "input_ids_sha256", "seq_len", "n_image_tokens", "image_grid_thw",
          "pixel_values_bf16_sha256", "pixel_values_shape", "letter_prev_token_ids")
TOP = ("model", "max_pixels", "processor_image_settings", "image_token_id", "letter_token_ids",
       "letter_token_strings", "episodes_sha256", "n_prompts")
ok = True
print(f"transformers: {a['transformers_version']} vs {b['transformers_version']} (not compared)")
for k in TOP:
    same = a[k] == b[k]
    ok &= same
    print(f"{k}: {'match' if same else 'DIFFER  a=%r b=%r' % (a[k], b[k])}")
ka, kb = set(a["prompts"]), set(b["prompts"])
if ka != kb:
    ok = False
    print(f"prompt keys differ: only a {sorted(ka - kb)[:3]}, only b {sorted(kb - ka)[:3]}")
for f in FIELDS:
    bad = [k for k in sorted(ka & kb) if a["prompts"][k][f] != b["prompts"][k][f]]
    ok &= not bad
    if bad:
        k = bad[0]
        print(f"{f}: {len(bad)}/{len(ka & kb)} DIFFER; first {k}: a={a['prompts'][k][f]!r} b={b['prompts'][k][f]!r}")
    else:
        print(f"{f}: all {len(ka & kb)} match")
print("RESULT:", "IDENTICAL" if ok else "DIFFERENT")
sys.exit(0 if ok else 1)
