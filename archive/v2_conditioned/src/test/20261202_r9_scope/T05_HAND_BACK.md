**Historical initial hand-back.** The review fixes and current interfaces are recorded in [T05_REVIEW_FIXES.md](T05_REVIEW_FIXES.md).

Ticket 05: Generic chain I: common, heads, and the backbone-free modules.

Built `r9_common.py`, `r9_heads.py`, `r9_stats.py`, `r9_episodes.py`, `r9_score.py`, `r9_check_heads.py` and the five corresponding `test_r9_*.py` files. Completed `r9_backbone.parts()` and extended the round's `.gitignore` while preserving its existing entry. No earlier-round file was edited.

Tests were copied and run before the modules existed (five collection failures), then the backbone/pin/label/writer checks were added before implementation. The later pre-stage-2 producer-pin test also failed before its implementation. Copied fixtures were adjusted for R9's rule hash, its reference to the pinned R6 check table, local temporary files and source-file mutation paths; the uncarried R6 refit runner and DTS configuration coverage were omitted.

Choices not fixed by the ticket:

- The arithmetic copies import `r9_common`; `r9_score` continues to import R6's bundle/context/picks helpers from MAIN until Ticket 06 supplies those modules. Their tests still exercise the upstream helpers. All arithmetic and existing guard markers remain unchanged.
- `input_pins(backbone)` returns `{R6 input key: (Path, SHA-256)}`. For new backbones, `results/pins_<backbone>.json` contains `{"input_sha256": {R6 input key: SHA-256}}`; recipe entries are required, frozen entries retain R6 hashes. CLIP's 51 path/hash entries are exactly R6's. The active R9 rule hash is asserted separately. This manifest schema is an integration choice for Tickets 06/13.
- `rb_reader_A0` names the pickle; `rb_reader_A0_json` names its JSON companion. `pm_fit_rows` aliases CLIP's `baselines_seed42.json` (the draw/hash record). `affect_km` aliases CLIP's partitions. Ticket 06's future CLIP `cells_seed42.json` is assigned `results/recipe/clip_b32/cells_seed42.json`, since no R6 cells file exists.
- The writer makes the eight carried posterior arrays and copies selection plus the six unused style arrays from CLIP, retaining the complete CLIP schemas. Records list keys made/copied explicitly. Those unused arrays are schema-preserving copies, not new-backbone refits.
- `write_posteriors(out, ..., backbone=...)` refits when fitted heads are not supplied; recipe callers may instead supply fitted heads/data/selection and input hashes. `recipe_head_pins` snapshots upstream parts before stage-2 manifests exist, and explicit pins are rechecked before use. Read-time refits/checks require declared pins. Records hash feature-store files, upstream parts and the caller script (`script_path`, defaulting to `r9_heads.py`). Output and record names are all checked before work, then created exclusively.
- Arm-L accuracy/provenance identity is enforced for CLIP only; new-backbone `check_selection` reports `affect_identity: null`. CPU fit settings and draws remain R6's.

228 unique tests passed (includes Ticket 04's 50 data tests because the role resolver was changed). The final frozen-role alias check was also rerun: 2 passed, 39 deselected in 5.38s.

105 passed in 171.79s

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --tb=short --basetemp=src/test/20261202_r9_scope/results/t05-final-arithmetic src/test/20261202_r9_scope/test_r9_stats.py src/test/20261202_r9_scope/test_r9_episodes.py src/test/20261202_r9_scope/test_r9_score.py
```

91 passed in 47.82s

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --tb=short --basetemp=src/test/20261202_r9_scope/results/t05-final-common src/test/20261202_r9_scope/test_r9_common.py src/test/20261202_r9_scope/test_r9_data.py
```

32 passed in 37.08s; 10 upstream SciPy deprecation warnings

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --tb=short --basetemp=src/test/20261202_r9_scope/results/t05-final-heads src/test/20261202_r9_scope/test_r9_heads.py
```

CLIP replay, run once (exit 0):

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_check_heads.py --backbone clip_b32
```

```text
[17:29:37] head affect done
[17:29:52] head image done
[17:30:11] head caption done
n6_posteriors/keys: pass
n6_posteriors/selection: pass
n6_posteriors/affect__img: pass
n6_posteriors/affect__txt: pass
n6_posteriors/image__img: pass
n6_posteriors/image__txt: pass
n6_posteriors/caption__img: pass
n6_posteriors/caption__txt: pass
step1_heads_style/keys: pass
step1_heads_style/selection: pass
step1_heads_style/style_csd__img: pass
step1_heads_style/style_csd__txt: pass
step1_heads_style/style_csd__img_src: pass
step1_heads_style/style_gram__img: pass
step1_heads_style/style_gram__txt: pass
step1_heads_style/style_gram__img_src: pass
step1_heads_style/style_rand__img: pass
step1_heads_style/style_rand__txt: pass
affect_identity: pass
clip_b32: 16 arrays, PASS (117.5s)
```

Arithmetic diff output (zero context; labels suppress checkout timestamps):

```diff
--- r6_stats.py
+++ r9_stats.py
@@ -21 +21 @@
-    import r6_common
+    import r9_common as r6_common
--- r6_episodes.py
+++ r9_episodes.py
@@ -33 +33 @@
-import r6_common as R  # noqa: E402  (before anything that imports src: it puts MAIN's src first and checks it)
+import r9_common as R  # noqa: E402  (before anything that imports src: it puts MAIN's src first and checks it)
--- r6_score.py
+++ r9_score.py
@@ -35 +35 @@
-import r6_common as R  # noqa: E402  (first, before anything that imports src)
+import r9_common as R  # noqa: E402  (first, before anything that imports src)
```

Controller commands longer than four minutes: none. All executed commands were bounded below four minutes. No GPU was used, no push was made, and no unresolved issue remains within this ticket. New-backbone real fits wait for the recipe parts and stage-2 pins from later tickets.
