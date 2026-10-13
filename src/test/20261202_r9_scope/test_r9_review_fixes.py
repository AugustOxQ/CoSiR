"""Review regressions and one-line mutation checks at the new-backbone boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_common as R
import r9_heads as H
import r9_backbone as B
ORIGINAL_R = R
ORIGINAL_INPUT_PINS = R.input_pins


@pytest.fixture(params=B.BACKBONES[1:])
def recipe(tmp_path, monkeypatch, request):
    b = request.param
    st, sel = np.arange(4, dtype=np.int64), np.array([4, 5], dtype=np.int64)
    groups = np.arange(6, dtype=np.int64)
    paths = {}
    def save(rel, **arrays):
        path = tmp_path / Path(rel).name
        np.savez(path, **arrays)
        paths[rel] = path
        monkeypatch.setitem(R.INPUT_PATHS, rel, path)
        monkeypatch.setitem(R.INPUT_SHA256, rel, R.sha256_file(path))
        return path
    save(H.PARTITION_L_REL, partition_L=np.arange(4, dtype=np.int64))
    save(H.CSD_GROUP_REL, scorer_train=st, style_csd=np.arange(4, dtype=np.int64))
    save(H.E2_PARTITIONS_REL, local_groups=st, affect=st, image=st, caption=st)
    told = tmp_path / "told_oracle.json"
    told.write_text(json.dumps({"arms": {"L": {"head": {"draw": "fixed"}}}}))
    paths[H.TOLD_REL] = told
    monkeypatch.setitem(R.INPUT_PATHS, H.TOLD_REL, told)
    monkeypatch.setitem(R.INPUT_SHA256, H.TOLD_REL, R.sha256_file(told))
    save(H.CSD_HEADS_REL, selection=sel,
         **{key: np.full((2, 17), 0.25, np.float32) for key in (
             "style_csd__img", "style_csd__txt", "style_csd__img_src", "style_rand__img",
             "style_rand__txt", "style_gram__img", "style_gram__txt", "style_gram__img_src")})
    save(H.N6_REL, selection=sel,
         **{key: np.full((2, 64), 0.25, np.float32) for key in H.TARGETS[H.N6_REL]})
    recipe_part = tmp_path / "recipe_partitions.npz"
    np.savez(recipe_part, local_groups=st, affect=st, image=(st + 1) % 4, caption=(st + 2) % 4)
    pins = {rel: (path, R.INPUT_SHA256[rel]) for rel, path in paths.items()}
    pins[H.E2_PARTITIONS_REL] = (recipe_part, R.sha256_file(recipe_part))
    pins["affect_km"] = (paths[H.E2_PARTITIONS_REL], R.INPUT_SHA256[H.E2_PARTITIONS_REL])
    clip_pins = {rel: (path, R.INPUT_SHA256[rel]) for rel, path in paths.items()}
    clip_pins["affect_km"] = pins["affect_km"]
    monkeypatch.setattr(R, "input_pins", lambda backbone="clip_b32": clip_pins if backbone == "clip_b32" else pins)
    roles = {"partition_L": paths[H.PARTITION_L_REL], "step1_group_style": paths[H.CSD_GROUP_REL],
             "told_oracle": told, "partitions": recipe_part, "affect_km": paths[H.E2_PARTITIONS_REL]}
    original_parts = B.parts
    monkeypatch.setattr(B, "parts", lambda backbone: {**original_parts(backbone), **roles,
        "partitions": paths[H.E2_PARTITIONS_REL] if backbone == "clip_b32" else recipe_part})
    data = SimpleNamespace(img_features=np.ones((6, B.DIMENSIONS[b]), np.float32),
                           txt_features=np.ones((6, B.DIMENSIONS[b]), np.float32))
    store = tmp_path / "store"
    store.mkdir()
    for m in ("img", "txt"):
        np.save(store / f"{m}.npy", getattr(data, f"{m}_features"))
    (store / "index.json").write_text("{}")
    monkeypatch.setitem(B.STORE_PATHS, b, store)
    for rel in (R.PREPARE_REL, R.HELD_CODES_REL):
        path = tmp_path / Path(rel).name
        path.write_bytes(b"indices")
        monkeypatch.setitem(R.INPUT_PATHS, rel, path)
        monkeypatch.setitem(R.INPUT_SHA256, rel, R.sha256_file(path))
        pins[rel] = clip_pins[rel] = (path, R.INPUT_SHA256[rel])
    def fit_all(*args, **kwargs):
        return {name: {**{m: SimpleNamespace(
            classes_=np.arange(H.N_CLASSES[name]), coef_=np.ones((H.N_CLASSES[name], B.DIMENSIONS[b]), np.float64),
            intercept_=np.zeros(H.N_CLASSES[name], np.float64),
            predict_proba=lambda x, k=H.N_CLASSES[name]: np.full((len(x), k), 1 / k))
            for m in H.MODALITIES}, "prov": {"draw": "fixed"}} for name in H.HEADS}
    monkeypatch.setattr(H, "fit_all", fit_all)
    heads = H.fit_heads_r9(data, groups, st, backbone=b, input_pins=pins)
    return SimpleNamespace(b=b, st=st, sel=sel, groups=groups, data=data, pins=pins,
                           paths=paths, clip_pins=clip_pins, heads=heads, out=tmp_path / "out", fit_all=fit_all)


def test_new_backbone_unused_heads_are_nan_and_csd_source_is_copied(recipe):
    x = recipe
    H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)
    with np.load(x.out / "step1_heads_style.npz") as got, np.load(x.paths[H.CSD_HEADS_REL]) as old:
        for key in old.files:
            if key.startswith(("style_rand__", "style_gram__")):
                assert got[key].shape == old[key].shape and got[key].dtype == old[key].dtype
                assert np.isnan(got[key]).all(), key
        assert np.array_equal(got["style_csd__img_src"], old["style_csd__img_src"])


def test_recipe_affect_must_equal_the_frozen_partition(recipe):
    x = recipe
    path = x.pins[H.E2_PARTITIONS_REL][0]
    np.savez(path, local_groups=x.st, affect=(x.st + 1) % 4, image=x.st, caption=x.st)
    x.pins[H.E2_PARTITIONS_REL] = (path, R.sha256_file(path))
    with pytest.raises(AssertionError, match="frozen affect"):
        H.head_labels(x.groups, x.st, x.b, input_pins=x.pins)


@pytest.mark.parametrize("rel", [H.PARTITION_L_REL, H.CSD_GROUP_REL, H.TOLD_REL])
def test_explicit_pins_cannot_replace_frozen_inputs(recipe, tmp_path, rel):
    x = recipe
    source = R.INPUT_PATHS[rel]
    other = tmp_path / ("replacement_" + source.name)
    other.write_bytes(source.read_bytes())
    pins = {**x.pins, rel: (other, R.sha256_file(other))}
    with pytest.raises(AssertionError, match="frozen pin"):
        H._head_input(rel, x.b, pins)


def test_explicit_frozen_partition_pin_rejected(recipe, tmp_path):
    test_explicit_pins_cannot_replace_frozen_inputs(recipe, tmp_path, H.PARTITION_L_REL)


def test_writer_rejects_caller_hashes_unrelated_to_fit(recipe):
    x = recipe
    with pytest.raises(AssertionError, match="fit inputs"):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b, input_sha256={"invented": "a" * 64})


def test_writer_rejects_upstream_changed_since_fit(recipe):
    x = recipe
    with x.pins[H.E2_PARTITIONS_REL][0].open("ab") as file:
        file.write(b"changed after fit")
    with pytest.raises(SystemExit, match="SHA-256"):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)


def test_writer_rejects_wrong_feature_width(recipe):
    x = recipe
    data = SimpleNamespace(img_features=x.data.img_features[:, :512], txt_features=x.data.txt_features[:, :512])
    with pytest.raises(AssertionError, match="backbone width"):
        H.write_posteriors(x.out, x.heads, data, x.sel, backbone=x.b)


def test_writer_rejects_wrong_coefficient_width(recipe):
    x = recipe
    x.heads["affect"]["img"].coef_ = x.heads["affect"]["img"].coef_[:, :512]
    with pytest.raises(AssertionError, match="coefficient width"):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)


def test_n6_record_pins_all_coefficients_and_affect_provenance(recipe):
    x = recipe
    records = H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)
    rec = records["n6_posteriors.npz"]
    assert rec["coef_sha256"] == H.coef_sha256(x.heads)
    assert rec["affect_prov"] == H.rto.roundtrip(x.heads["affect"]["prov"])
    assert rec["input_pins"][H.E2_PARTITIONS_REL] == [str(x.pins[H.E2_PARTITIONS_REL][0]), x.pins[H.E2_PARTITIONS_REL][1]]


def selection_parts(x):
    H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)
    for rel in H.TARGETS:
        path = x.out / Path(rel).name
        x.pins[rel] = (path, R.sha256_file(path))
    record = x.out / "n6_posteriors.npz.record.json"
    x.pins[H.N6_REL + ".record.json"] = (record, R.sha256_file(record))
    return record


def test_new_backbone_selection_uses_its_own_arrays(recipe):
    x = recipe
    selection_parts(x)
    assert H.check_selection(x.heads, x.data, x.sel, x.b)["passed"]


def test_new_backbone_selection_checks_affect_coefficients(recipe):
    x = recipe
    selection_parts(x)
    x.heads["affect"]["img"].coef_[0, 0] += 1
    checked = H.check_selection(x.heads, x.data, x.sel, x.b)
    assert not checked["passed"] and not checked["coef_identity"]


def test_new_backbone_selection_checks_affect_provenance(recipe):
    x = recipe
    selection_parts(x)
    x.heads["affect"]["prov"] = {"draw": "another"}
    checked = H.check_selection(x.heads, x.data, x.sel, x.b)
    assert not checked["passed"] and not checked["affect_identity"]


def test_fit_heads_requires_full_draw(recipe):
    x = recipe
    with pytest.raises(AssertionError, match="60000"):
        H.fit_heads_r9(x.data, x.groups, x.st, n_rows=3, backbone=x.b, input_pins=x.pins)


def test_pins_files_live_in_round_folder():
    for b in B.BACKBONES:
        assert R.pins_path(b).parent == R.HERE


def test_load_data_routes_the_requested_backbone(recipe, monkeypatch):
    import r9_data as D
    x = recipe
    monkeypatch.setattr(D, "load_data", lambda b: x.data if b == x.b else None)
    assert R.load_data(x.b) is x.data


def test_load_rows_routes_the_requested_backbone(recipe, monkeypatch):
    from src.data import artelingo_splits as S
    x = recipe
    other = SimpleNamespace(**vars(x.data))
    monkeypatch.setattr(R, "N_ROWS", 6)
    monkeypatch.setattr(R, "load_data", lambda b: x.data if b == x.b else other)
    monkeypatch.setattr(R, "load_split", lambda data: SimpleNamespace(groups=x.groups, selection=x.sel))
    monkeypatch.setattr(S, "artelingo_aspect_labels", lambda data: {key: x.groups for key in R.ASPECTS})
    monkeypatch.setattr(R, "development_value_sets", lambda *a: {})
    assert R.load_rows(x.b).data is x.data


def test_writer_refit_routes_the_requested_backbone(recipe, monkeypatch):
    x = recipe
    called = []
    def load(b="clip_b32"):
        called.append(b)
        return SimpleNamespace(data=x.data, split=SimpleNamespace(selection=x.sel, scorer_train=x.st, groups=x.groups))
    monkeypatch.setattr(R, "load_rows", load)
    records = H.write_posteriors(x.out, backbone=x.b)
    assert called == [x.b] and records["n6_posteriors.npz"]["backbone"] == x.b


def test_head_labels_without_explicit_pins_use_recipe(recipe):
    x = recipe
    labels = H.head_labels(x.groups, x.st, x.b)
    assert np.array_equal(labels["image"][x.st], (x.st + 1) % 4)


def test_recipe_pins_keep_partitions_distinct_from_frozen_affect(recipe):
    x = recipe
    pins = H.recipe_head_pins(x.b)
    assert pins[H.E2_PARTITIONS_REL] == x.pins[H.E2_PARTITIONS_REL]
    assert pins[H.AFFECT_KM_REL] == x.pins[H.AFFECT_KM_REL]
    assert pins[H.E2_PARTITIONS_REL] != pins[H.AFFECT_KM_REL]


def test_recipe_pins_reject_replaced_frozen_path(recipe, tmp_path, monkeypatch):
    x = recipe
    original = B.parts
    other = tmp_path / "other_partition_L.npz"
    other.write_bytes(x.paths[H.PARTITION_L_REL].read_bytes())
    monkeypatch.setattr(B, "parts", lambda b: {**original(b), "partition_L": other})
    with pytest.raises(AssertionError, match="frozen recipe path"):
        H.recipe_head_pins(x.b)


def test_recipe_pins_reject_changed_frozen_bytes(recipe):
    x = recipe
    with x.paths[H.TOLD_REL].open("ab") as file:
        file.write(b" ")
    with pytest.raises(SystemExit, match="SHA-256"):
        H.recipe_head_pins(x.b)


def test_told_hash_is_checked_before_loading_labels(recipe, monkeypatch):
    x = recipe
    seen = []
    original = H.head_labels
    def labels(*a, **kw):
        seen.append(True)
        return original(*a, **kw)
    monkeypatch.setattr(H, "head_labels", labels)
    with x.paths[H.TOLD_REL].open("ab") as file:
        file.write(b" ")
    with pytest.raises(SystemExit, match="SHA-256"):
        H.fit_heads_r9(x.data, x.groups, x.st, backbone=x.b, input_pins=x.pins)
    assert not seen


def test_r9_rule_hash_cannot_be_bypassed(tmp_path, monkeypatch):
    (tmp_path / "DECISION_RULE.md").write_text("different rule")
    monkeypatch.setattr(R, "HERE", tmp_path)
    monkeypatch.setattr(R, "input_pins", lambda b: {})
    with pytest.raises(SystemExit, match="R9 DECISION_RULE"):
        R.assert_inputs("qwen3vl_emb_2b")


def test_manifest_cannot_change_a_frozen_hash(tmp_path, monkeypatch):
    if R is ORIGINAL_R:
        monkeypatch.setattr(R, "input_pins", ORIGINAL_INPUT_PINS)
    rel = R.PREPARE_REL
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"input_sha256": {rel: "a" * 64}}))
    monkeypatch.setattr(R, "pins_path", lambda b: path)
    monkeypatch.setattr(R, "input_paths", lambda b: {rel: R.TEST / rel})
    with pytest.raises(AssertionError, match="frozen pin"):
        R.input_pins("qwen3vl_emb_2b")


def test_copied_style_selection_is_checked(recipe):
    x = recipe
    path = x.paths[H.CSD_HEADS_REL]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["selection"] = np.array([0, 1], dtype=np.int64)
    np.savez(path, **arrays)
    sha = R.sha256_file(path)
    R.INPUT_SHA256[H.CSD_HEADS_REL] = sha
    x.clip_pins[H.CSD_HEADS_REL] = (path, sha)
    x.heads["affect"]["fit_inputs"]["input_sha256"][str(path)] = sha
    with pytest.raises(AssertionError, match="copied style arrays have another selection"):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)


def test_writer_exclusive_open_handles_a_racing_file(recipe, monkeypatch):
    x = recipe
    target = x.out / "n6_posteriors.npz"
    original_open = Path.open
    raced = []
    def opening(path, mode="r", *args, **kwargs):
        if path == target and mode in ("xb", "wb") and not raced:
            raced.append(True)
            path.write_bytes(b"racing owner")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(FileExistsError):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)
    assert target.read_bytes() == b"racing owner"


def test_record_names_the_actual_backbone(recipe):
    x = recipe
    records = H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)
    assert all(rec["backbone"] == x.b for rec in records.values())


def test_module_sha_coverage(tmp_path):
    folder = tmp_path / "src/test/20261202_r9_scope"
    folder.mkdir(parents=True)
    for name in ("r9_a.py", "run_r9_b.py", "test_r9_c.py", "dts_settings.json", "other.json"):
        (folder / name).write_text(name)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_r9_a.sh", "das6_sync_r9.py", "run_r6_a.sh", "other.sh"):
        (scripts / name).write_text(name)
    expected = {"src/test/20261202_r9_scope/r9_a.py", "src/test/20261202_r9_scope/run_r9_b.py",
                "scripts/run_r9_a.sh", "scripts/das6_sync_r9.py"}
    assert set(R.r9_module_shas(folder)) == expected
    assert all(sha == R.sha256_file(tmp_path / rel) for rel, sha in R.r9_module_shas(folder).items())


def test_check_script_reports_copies_and_unrefitted_arrays(recipe, monkeypatch, capsys):
    import r9_check_heads as C
    x = recipe
    selection_parts(x)
    monkeypatch.setattr(C, "parts", lambda b: {"n6_posteriors": x.out / "n6_posteriors.npz",
                                               "step1_heads_style": x.out / "step1_heads_style.npz"})
    monkeypatch.setattr(R, "load_rows", lambda b: SimpleNamespace(data=x.data,
        split=SimpleNamespace(groups=x.groups, scorer_train=x.st, selection=x.sel)))
    assert C.main(["--backbone", x.b]) == 0
    output = capsys.readouterr().out
    assert "style_csd__img_src: copied" in output
    assert "style_rand__img: not refit" in output
    assert "8 arrays, PASS" in output


def test_writer_checks_feature_file_hash_from_fit(recipe):
    x = recipe
    with (B.STORE_PATHS[x.b] / "img.npy").open("ab") as file:
        file.write(b"changed feature store")
    with pytest.raises(SystemExit, match="SHA-256 differs from fit inputs"):
        H.write_posteriors(x.out, x.heads, x.data, x.sel, backbone=x.b)


def test_selection_record_hash_is_pinned(recipe):
    x = recipe
    path = selection_parts(x)
    with path.open("a") as file:
        file.write(" ")
    with pytest.raises(SystemExit, match="SHA-256"):
        H.check_selection(x.heads, x.data, x.sel, x.b)


def test_new_manifest_requires_n6_record_hash(tmp_path, monkeypatch):
    if R is ORIGINAL_R:
        monkeypatch.setattr(R, "input_pins", ORIGINAL_INPUT_PINS)
    b = "qwen3vl_emb_2b"
    shas = {rel: R.INPUT_SHA256.get(rel, "a" * 64) for rel in R.input_paths(b)}
    record_rel = H.N6_REL + ".record.json"
    assert record_rel in R.input_paths(b)
    del shas[record_rel]
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"input_sha256": shas}))
    monkeypatch.setattr(R, "pins_path", lambda backbone: path)
    with pytest.raises(AssertionError, match="missing recipe input.*record.json"):
        R.input_pins(b)


def mutation(tmp_path, module, old, new):
    source = (HERE / module).read_text()
    assert source.count(old) == 1, old
    source = source.replace(old, new)
    if module == "r9_common.py":
        source = source.replace('HERE = Path(__file__).resolve().parent', f'HERE = Path({str(HERE)!r})')
    path = tmp_path / (Path(module).stem + "_review_mutant.py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    saved = sys.path[:]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    return mod


MUTATIONS = [
    ("unused_heads", "r9_heads.py", 'backbone != "clip_b32" and key != "style_csd__img_src"',
     'False', "test_new_backbone_unused_heads_are_nan_and_csd_source_is_copied"),
    ("recipe_affect", "r9_heads.py", '_require(np.array_equal(z["affect"], frozen_affect), "recipe affect differs from frozen affect")',
     'pass', "test_recipe_affect_must_equal_the_frozen_partition"),
    ("explicit_frozen_pin", "r9_heads.py", '_require(input_pins[rel] == frozen, f"{rel}: explicit frozen pin differs from R6")',
     'pass', "test_explicit_frozen_partition_pin_rejected"),

    ("caller_fit_hashes", "r9_heads.py", '_require(input_sha256 == fit_inputs["input_sha256"], "caller hashes differ from fit inputs")', 'pass', "test_writer_rejects_caller_hashes_unrelated_to_fit"),
    ("feature_width", "r9_heads.py", '_require(data.img_features.shape[1] == data.txt_features.shape[1] == DIMENSIONS[backbone],\n             "data differs from backbone width")', 'pass', "test_writer_rejects_wrong_feature_width"),
    ("coefficient_width", "r9_heads.py", '_require(all(head[m].coef_.shape[1] == DIMENSIONS[backbone] for head in heads.values() for m in MODALITIES),\n             "head coefficient width differs from backbone")', 'pass', "test_writer_rejects_wrong_coefficient_width"),
    ("fit_file_hash", "r9_heads.py", '_require(R.sha256_file(path) == sha, f"{path}: SHA-256 differs from fit inputs", SystemExit)', 'pass', "test_writer_checks_feature_file_hash_from_fit"),
    ("record_coefficients", "r9_heads.py", 'coef_sha256=coef_sha256(heads)', 'coef_sha256={}', "test_n6_record_pins_all_coefficients_and_affect_provenance"),
    ("record_affect_provenance", "r9_heads.py", 'affect_prov=rto.roundtrip(heads["affect"]["prov"])', 'affect_prov={}', "test_n6_record_pins_all_coefficients_and_affect_provenance"),
    ("check_coefficients", "r9_heads.py", 'coef_identity = stored["coef_sha256"] == coef_sha256(heads)', 'coef_identity = True', "test_new_backbone_selection_checks_affect_coefficients"),
    ("check_affect_provenance", "r9_heads.py", 'identity = stored["affect_prov"] == rto.roundtrip(heads["affect"]["prov"])', 'identity = True', "test_new_backbone_selection_checks_affect_provenance"),
    ("load_data_backbone", "r9_common.py", 'return backbone_data(backbone)', 'return backbone_data("clip_b32")', "test_load_data_routes_the_requested_backbone"),
    ("load_rows_backbone", "r9_common.py", 'data = load_data(backbone)', 'data = load_data("clip_b32")', "test_load_rows_routes_the_requested_backbone"),
    ("writer_refit_rows", "r9_heads.py", 'rows = R.load_rows(backbone)', 'rows = R.load_rows("clip_b32")', "test_writer_refit_routes_the_requested_backbone"),
    ("writer_refit_backbone", "r9_heads.py", 'heads = fit_heads_r9(data, rows.split.groups, rows.split.scorer_train, backbone=backbone, input_pins=upstream)', 'heads = fit_heads_r9(data, rows.split.groups, rows.split.scorer_train, backbone="clip_b32", input_pins=upstream)', "test_writer_refit_routes_the_requested_backbone"),
    ("implicit_recipe_partitions", "r9_heads.py", 'return R.input_pins(backbone)[rel][0]', 'return R.input_pins()[rel][0]', "test_head_labels_without_explicit_pins_use_recipe"),
    ("recipe_partition_role", "r9_heads.py", 'E2_PARTITIONS_REL: "partitions", AFFECT_KM_REL:', 'E2_PARTITIONS_REL: "affect_km", AFFECT_KM_REL:', "test_recipe_pins_keep_partitions_distinct_from_frozen_affect"),
    ("recipe_frozen_path", "r9_heads.py", '_require(path == R.INPUT_PATHS[canonical], f"{rel}: frozen recipe path differs from R6")', 'pass', "test_recipe_pins_reject_replaced_frozen_path"),
    ("recipe_frozen_hash", "r9_heads.py", '_require(R.sha256_file(path) == sha, f"{rel}: frozen recipe SHA-256 differs from R6", SystemExit)', 'pass', "test_recipe_pins_reject_changed_frozen_bytes"),
    ("selection_recipe_arrays", "r9_heads.py", 'with np.load(R.input_pins(backbone)[rel][0]) as z:', 'with np.load(R.input_pins()[rel][0]) as z:', "test_new_backbone_selection_uses_its_own_arrays"),
    ("full_draw", "r9_heads.py", '_require(n_rows == HEAD_ROWS, "head draw must contain 60000 rows")', 'pass', "test_fit_heads_requires_full_draw"),
    ("r9_rule_hash", "r9_common.py", '_require(sha256_file(HERE / "DECISION_RULE.md") == RULE_SHA256,\n             "R9 DECISION_RULE.md differs from its committed pin", SystemExit)', 'pass', "test_r9_rule_hash_cannot_be_bypassed"),
    ("manifest_frozen_hash", "r9_common.py", '_require(declared.get(rel, sha) == sha, f"{rel}: frozen pin differs from R6")', 'pass', "test_manifest_cannot_change_a_frozen_hash"),
    ("told_before_labels", "r9_heads.py", '_head_input(TOLD_REL, backbone, pins)', 'pass', "test_told_hash_is_checked_before_loading_labels"),
    ("copied_selection", "r9_heads.py", '_require(np.array_equal(z["selection"], selection), "copied style arrays have another selection row set")', 'pass', "test_copied_style_selection_is_checked"),
    ("exclusive_output", "r9_heads.py", '(out / name).open("xb")', '(out / name).open("wb")', "test_writer_exclusive_open_handles_a_racing_file"),
    ("record_backbone", "r9_heads.py", 'records[name] = {"backbone": backbone,', 'records[name] = {"backbone": "clip_b32",', "test_record_names_the_actual_backbone"),
    ("pins_folder", "r9_common.py", 'return HERE / ("pins_clip.json"', 'return RESULTS / ("pins_clip.json"', "test_pins_files_live_in_round_folder"),

    ("record_sha", "r9_heads.py", 'rel = N6_REL + ".record.json"\n        R.assert_input(rel, backbone)', 'rel = N6_REL + ".record.json"\n        pass', "test_selection_record_hash_is_pinned"),
    ("manifest_record_pin", "r9_common.py", 'paths[rel + ".record.json"] = Path(str(paths[rel]) + ".record.json")', 'pass', "test_new_manifest_requires_n6_record_hash"),
    ("script_copies", "r9_check_heads.py", 'args.backbone != "clip_b32" and key in records[path.name]["keys_copied"]', 'False', "test_check_script_reports_copies_and_unrefitted_arrays"),
    ("script_unrefit", "r9_check_heads.py", 'args.backbone != "clip_b32" and key in records[path.name]["keys_unrefit"]', 'False', "test_check_script_reports_copies_and_unrefitted_arrays"),

    ("frozen_affect_source", "r9_heads.py", '_head_input(AFFECT_KM_REL, backbone, input_pins)', '_head_input(E2_PARTITIONS_REL, backbone, input_pins)', "test_recipe_affect_must_equal_the_frozen_partition"),
    ("module_sha_coverage", "r9_common.py", 'files = sorted(here.glob("r9_*.py")) + sorted(here.glob("run_r9_*.py"))', 'files = sorted(here.glob("r9_*.py"))', "test_module_sha_coverage"),
]


@pytest.mark.parametrize("case", MUTATIONS, ids=lambda c: c[0])
def test_review_mutation_is_killed(case, recipe, tmp_path, monkeypatch, capsys):
    import inspect
    name, module, old, new, scenario = case
    mod = mutation(tmp_path, module, old, new)
    if module == "r9_heads.py":
        mod.fit_all = recipe.fit_all
    with monkeypatch.context() as m:
        if module == "r9_check_heads.py":
            m.setitem(sys.modules, "r9_check_heads", mod)
        else:
            m.setitem(globals(), "H" if module == "r9_heads.py" else "R", mod)
        fn = globals()[scenario]
        args = {"recipe": recipe, "tmp_path": tmp_path, "monkeypatch": m, "capsys": capsys}
        with pytest.raises((AssertionError, pytest.fail.Exception)):
            fn(**{key: args[key] for key in inspect.signature(fn).parameters})
