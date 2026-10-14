"""One-line mutations of r9_recipe_groups.py / r9_heads.py (ticket 07 fix round): each must make a test fail.

    CUDA_VISIBLE_DEVICES= python mutate_t07.py [first last]     (prints KILLED / SURVIVED per mutation; restores files)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RG, HD = HERE / "r9_recipe_groups.py", HERE / "r9_heads.py"
M = [  # (file, old, new)
    (RG, "K, SEED = 64, 42", "K, SEED = 63, 42"),
    (RG, "K, SEED = 64, 42", "K, SEED = 64, 41"),
    (RG, "BANK_SIZE, MIN_PAINTINGS = 65_536, 30", "BANK_SIZE, MIN_PAINTINGS = 65_535, 30"),
    (RG, "BANK_SIZE, MIN_PAINTINGS = 65_536, 30", "BANK_SIZE, MIN_PAINTINGS = 65_536, 29"),
    (RG, "N_SCORER = 183_694", "N_SCORER = 183_693"),
    (RG, 'AIC = ("affect", "image", "caption")', 'AIC = ("affect", "caption", "image")'),
    (RG, "min_group_rows=200) -> dict", "min_group_rows=100) -> dict"),
    (RG, '    assert (np.diff(st) > 0).all(), "scorer_train must be sorted and unique"\n', ""),
    (RG, 'assert np.array_equal(local_groups, np.unique(np.asarray(groups)[st], return_inverse=True)[1]), \\\n        "local_groups misaligned with scorer_train"',
         'assert True'),
    (RG, 'assert affect.shape == local_groups.shape == (n,), "copied affect/local_groups do not match scorer_train"', "assert True"),
    (RG, 'image = source.labels_by_view["image"][st].astype(np.int64)', 'image = source.labels_by_view["image"][:n].astype(np.int64)'),
    (RG, "kmeans_partition(np.asarray(txt)[st], np.arange(n)", "kmeans_partition(np.asarray(txt)[:n], np.arange(n)"),
    (RG, "kmeans_partition(np.asarray(txt)[st]", "kmeans_partition(np.asarray(img)[st]"),
    (RG, "ClipClusterSource(img, txt, st,", "ClipClusterSource(txt, img, st,"),
    (RG, "assert labels.min() >= 0, ", "assert labels.min() >= -1, "),
    (RG, "assert len(np.unique(labels)) == k, ", "assert len(np.unique(labels)) >= k - 1, "),
    (RG, "build_content_graph(np.asarray(img)[st], np.asarray(txt)[st]", "build_content_graph(np.asarray(img)[:len(st)], np.asarray(txt)[:len(st)]"),
    (RG, 'assert graph.shape == (len(st), len(st)), f"graph shape {graph.shape}"', "assert True"),
    (RG, "if not cuda and not allow_cpu:", "if not cuda and False:"),
    (RG, "if not cuda and not allow_cpu:", "if not cuda and allow_cpu:"),
    (RG, '"use_half": cuda,', '"use_half": True,'),
    (RG, '"cpu_allowed": bool(allow_cpu)', '"cpu_allowed": True'),
    (RG, "graph_device_record(allow_cpu)\n    st = np.asarray", "None\n    st = np.asarray"),
    (RG, "bank = build_episode_bank(sub, local_groups, rows, n_per_pair=per_pair, seed=SEED,", "bank = build_episode_bank(sub, local_groups, rows, n_per_pair=per_pair, seed=SEED + 1,"),
    (RG, "sizes[-1] -= sum(sizes) - total", "sizes[0] -= sum(sizes) - total"),
    (RG, 'BANK_CONSTANT_KEYS = ("aspect_a", "aspect_b", "block_sizes", "block_pairs")', 'BANK_CONSTANT_KEYS = ("aspect_b", "block_sizes", "block_pairs")'),
    (RG, "    made = [k for k in bank if k not in BANK_CONSTANT_KEYS]", "    made = [k for k in bank]"),
    (RG, "pins = dict(H.recipe_head_pins(backbone, out))", "pins = dict(H.recipe_head_pins(backbone))"),
    (RG, 'assert Path(pins[H.E2_PARTITIONS_REL][0]) == mine, "partitions pin is not this run\'s file"', "assert True"),
    (RG, "    if backbone != \"clip_b32\":\n        shas.update(store_shas(\"clip_b32\"))", "    pass"),
    (RG, "    shas[str(ANNOTATIONS_PATH)] = R.sha256_file(ANNOTATIONS_PATH)\n", ""),
    (RG, '"numpy": np.__version__', '"numpy": scipy.__version__'),
    (RG, '"sklearn": sklearn.__version__', '"sklearn": scipy.__version__'),
    (RG, 'CODE_MODULES = ("src.model.graph", "src.conditional_buddy.buddy_graph",', 'CODE_MODULES = ("src.model.graph",'),
    (RG, '"sha256": R.sha256_file(mod.__file__)}', '"sha256": R.sha256_file(__file__)}'),
    (RG, "            made = list(z.files)", '            made = ["graph"]'),
    (RG, '"script_sha256": R.sha256_file(__file__), "code": code_record(),', '"script_sha256": R.sha256_file(__file__),'),
    (RG, "**(extra or {})}", "}"),
    (RG, '"graph.npz": R.sha256_file(out / "graph.npz"), "bank_AIC.npz": R.sha256_file(out / "bank_AIC.npz")}}', '"graph.npz": R.sha256_file(out / "bank_AIC.npz"), "bank_AIC.npz": R.sha256_file(out / "graph.npz")}}'),
    (RG, 'sha_array(np.asarray(local_groups, dtype=np.int64))', 'sha_array(np.asarray(local_groups, dtype=np.int32))'),
    (RG, "fit_partitions(data.img_features, data.txt_features, split.groups, st, frozen)", "fit_partitions(data.txt_features, data.img_features, split.groups, st, frozen)"),
    (RG, "fit_graph(data.img_features, data.txt_features, st, allow_cpu=allow_cpu)", "fit_graph(data.txt_features, data.img_features, st, allow_cpu=allow_cpu)"),
    (RG, 'assert len(st) == N_SCORER, ', "assert True or "),
    (RG, "graph_device_record(allow_cpu) if last >= 1 else None", "None"),
    (RG, "input_sha256={**base, **frozen_sha}", "input_sha256={**base}"),
    (RG, "input_sha256=base,\n                   made=None", "input_sha256={},\n                   made=None"),
    (RG, "        write_build_record(out, new_partitions[\"local_groups\"])\n", ""),
    (RG, "input_sha256={**reads, str(mine)", "input_sha256={str(mine)"),
    (RG, "    if STEPS.index(stop_after) >= STEPS.index(\"bank\"):\n        targets.append(Path(out) / BUILD_RECORD)", "    pass"),
    (RG, "present = [str(Path(out) / n) for n in names if (Path(out) / n).exists()]", "present = []"),
    (RG, "extra={\"graph_build\": graph_build})", "extra={})"),
    (RG, "str(R.INPUT_PATHS[R.PREPARE_REL]): R.assert_input(R.PREPARE_REL),", ""),
    (RG, "fit_heads_step(out, args.backbone, data, split)", "fit_heads_step(out, args.backbone, data, split) if False else None"),
    (HD, 'if root is not None and role == "partitions":', 'if False and role == "partitions":'),
    (HD, "path = Path(root) / path.name", 'path = Path(root) / "x.npz"'),
    (HD, 'def recipe_head_pins(backbone, root=None):', 'def recipe_head_pins(backbone, root="."):'),
]


def main():
    lo, hi = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (0, len(M))
    killed = 0
    for i in range(lo, hi):
        path, old, new = M[i]
        orig = path.read_text()
        assert orig.count(old) == 1, (i, old, orig.count(old))
        path.write_text(orig.replace(old, new))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(HERE / "test_r9_recipe_groups.py")], capture_output=True, text=True)
        finally:
            path.write_text(orig)
        dead = r.returncode != 0
        killed += dead
        print(f"{i:2d} {'KILLED  ' if dead else 'SURVIVED'} {path.name}: {old.strip()[:70]!r} -> {new.strip()[:40]!r}", flush=True)
    print(f"killed {killed}/{hi - lo}")


if __name__ == "__main__":
    main()
