# Brief: make PercepT pilot scripts' paths cluster-portable

This is a precise, mechanical refactor across many files — every change
must follow the exact pattern below, no creative variation. **Do NOT run
anything** — this is a code-editing task only, no GPU execution, and no
`git commit`.

## Context: why this is needed

Every PercepT pilot script hardcodes local-machine-only absolute paths
(`/data/SSD2/pre_extract/...` for feature caches, `/data/PDD/artelingo/...`
for raw JSON annotation files) and writes its output report next to itself
in the git tree (`OUT_DIR = os.path.dirname(__file__)`, then `REPORT_PATH =
os.path.join(OUT_DIR, "....md")`). None of this exists on the DAS6 cluster
node this branch will also run on — the user confirmed these are genuinely
local-only paths with no shared mount. The fix is three environment
variables, each defaulting to today's exact hardcoded behavior so **local
execution must be byte-for-byte unchanged** when the variables are unset:

1. `PERCEPT_FEATURE_ROOT` (default `/data/SSD2/pre_extract`) — for the
   pre-extracted CLIP feature cache and the patch-token cache.
2. `PERCEPT_RAW_JSON_ROOT` (default `/data/PDD/artelingo`) — for the raw
   annotation JSON files this pipeline actually reads (train/val_test/
   genre — NOT the release CSV, confirmed unused by any PercepT script;
   leave `MAIN_CSV` in `run_pipeline.py` completely untouched).
3. `PERCEPT_OUTPUT_ROOT` (default: unset, meaning "keep current behavior,
   write next to the script") — for where `.md` reports get written.

## Part 1 — `PERCEPT_FEATURE_ROOT`

In each file below, add (near the top, with the other path constants)
`PERCEPT_FEATURE_ROOT = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")`
(import `os` if not already imported), then rewrite the listed constant(s)
to be built from it via an f-string, preserving the exact existing
subdirectory structure:

- `src/test/20260923_artelingo_buddy_analysis/run_pipeline.py`:
  `STORAGE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo/features"`
- `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`:
  `HELDOUT_STORAGE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo_heldout/features"`
- `src/test/20260923_artelingo_buddy_analysis/run_cca_audit_pilot.py`:
  same `HELDOUT_STORAGE_DIR` rewrite (this file independently redefines the
  same constant; both must be updated identically)
- `src/test/20260922_percept_topic_pipeline/run_percept_stage2_pilot.py`:
  `PATCH_FEATURE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo_percept_patch_features"`
- `src/test/20260922_percept_topic_pipeline/run_percept_stage2_sweep_pilot.py`:
  same `PATCH_FEATURE_DIR` rewrite (independently redefined here too)
- `src/test/20260922_percept_topic_pipeline/run_percept_patch_feature_extraction.py`:
  `CACHE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo_percept_patch_features"`
  (same target directory as `PATCH_FEATURE_DIR` above, just a differently-
  named constant in this one file — keep its existing name `CACHE_DIR`, do
  not rename it)

## Part 2 — `PERCEPT_RAW_JSON_ROOT`

Same pattern: add
`PERCEPT_RAW_JSON_ROOT = os.environ.get("PERCEPT_RAW_JSON_ROOT", "/data/PDD/artelingo")`
near the top of each file, then rewrite:

- `src/test/20260923_artelingo_buddy_analysis/run_pipeline.py`:
  `TRAIN_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_train.json"` and
  `GENRE_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_genre_emotion_eng.json"`.
  **Leave `MAIN_CSV` exactly as it is** — do not touch it, it is unused by
  every PercepT script and this refactor should not risk breaking whatever
  else in this repo's history still uses it.
- `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`:
  `HELDOUT_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_val_test.json"`
- `src/test/20260923_artelingo_buddy_analysis/run_cca_audit_pilot.py`:
  same `HELDOUT_JSON` rewrite (independently redefined here too)
- `src/test/20260922_percept_topic_pipeline/run_percept_patch_feature_extraction.py`:
  both `TRAIN_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_train.json"` and
  `HELDOUT_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_val_test.json"`.
  **Leave `IMAGE_ROOT` untouched** — raw WikiArt images are not being
  synced to the node (the patch-token cache this script produces already
  covers what the node needs; re-running this specific extraction script
  on the node is out of scope for this refactor).

## Part 3 — `PERCEPT_OUTPUT_ROOT`

This is the trickiest part — read carefully, a naive find-replace will
break sibling-script loading.

Every pilot script currently has, near the top:
```python
OUT_DIR = os.path.dirname(__file__)
```
`OUT_DIR` is used for TWO different purposes across these files: (a)
locating sibling scripts to load via `load_module()`/`load_sibling_module()`
(e.g. `BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")`),
and (b) building `REPORT_PATH`. **Only purpose (b) should be redirected.**
Redirecting `OUT_DIR` itself would break every sibling-script lookup on the
node (they would be searched for inside the output directory instead of
next to the actual script).

The fix: leave every existing `OUT_DIR = os.path.dirname(__file__)` line
completely unchanged. Immediately after it, add one new line:
```python
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
```
Then change every `REPORT_PATH = os.path.join(OUT_DIR, "....md")`
assignment (there is exactly one per file, though it may be split across
multiple lines) to use `REPORT_OUT_DIR` instead of `OUT_DIR` — do not touch
any OTHER use of `OUT_DIR` in that same file (e.g. `BASE_PILOT_PATH`,
`SWEEP_PILOT_PATH`, `TRAIN_PATCH_FEATURE_PATH`, `PATCH_FEATURE_DIR` if it
happens to be built from `OUT_DIR` anywhere — those must keep pointing at
the actual script directory). Also add, immediately before the `with
open(REPORT_PATH, "w") as report_file:` line(s) in each file's `write_report()`
function (there may be more than one such `open(REPORT_PATH, "w")` call per
file, e.g. an early-return reproducibility-failure branch and the main
success branch — fix every one of them, not just the first):
```python
os.makedirs(REPORT_OUT_DIR, exist_ok=True)
```
(so a node run doesn't crash if `/local/wding/res/percept` doesn't exist
yet — local behavior is unaffected since `os.makedirs` on an
already-existing directory with `exist_ok=True` is a no-op).

Apply this `REPORT_OUT_DIR` fix to every file in
`src/test/20260922_percept_topic_pipeline/` that defines its own
`REPORT_PATH` from `OUT_DIR` (there are about 15 — grep for
`REPORT_PATH = os.path.join(OUT_DIR` and `REPORT_PATH = os.path.join(\n`
across every `.py` file in that directory to find all of them; do not rely
on a fixed list, some file may have been missed above). **One exception:**
`run_percept_stage1_cluster_count_sweep_v2_pilot.py` (and any other script
using the `template.REPORT_PATH = REPORT_PATH` monkey-patch pattern to
reuse another script's `main()`) still needs its OWN local `REPORT_PATH`
fixed the same way — the monkey-patch will correctly propagate the fixed
value, do not additionally touch the template script's own `write_report`
except via this same Part-3 fix applied to the template script itself
(since the template script's `write_report` needs the
`os.makedirs(REPORT_OUT_DIR, ...)` call too, for when the template's own
`REPORT_PATH`/`REPORT_OUT_DIR` names are referenced inside its
`write_report` function body).

`run_percept_patch_feature_extraction.py` does not write a markdown report
(it only writes the `.pt` cache files already covered by Part 1) — skip it
for Part 3.

## Verification before you finish

After editing, run (read-only, no execution of pilot logic):
```
python -m py_compile <every edited file>
```
for every file you touched, and grep to confirm no `REPORT_PATH =
os.path.join(OUT_DIR` occurrences remain anywhere in
`src/test/20260922_percept_topic_pipeline/*.py` (they should all now say
`REPORT_OUT_DIR`) — paste the grep output showing zero matches as your
final confirmation before stopping.

Do not modify any file's actual algorithmic logic, training loop,
evaluation code, or report content/wording — this task is exclusively
about where files are read from and written to.
