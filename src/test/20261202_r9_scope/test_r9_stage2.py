"""Ticket 13: provenance refusals and the final pins gate; CPU only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r9_stage2 as D


def test_refitted_array_equal_to_clip_fails():
    a = np.arange(8, dtype=np.float32)
    with pytest.raises(AssertionError, match="made.*equals CLIP"):
        D.assert_array_provenance({"post": a}, {"post": a.copy()}, ["post"], [])


def test_copied_array_changed_fails():
    with pytest.raises(AssertionError, match="copied.*differs"):
        D.assert_array_provenance({"rows": np.array([1, 3])}, {"rows": np.array([1, 2])}, [], ["rows"])


def test_frozen_array_changed_fails(tmp_path):
    a, b = tmp_path / "new.npz", tmp_path / "clip.npz"
    np.savez(a, groups=np.array([0, 2]))
    np.savez(b, groups=np.array([0, 1]))
    with pytest.raises(AssertionError, match="copied.*differs"):
        D.assert_frozen("prepare", a, b)


def test_changed_script_sha_fails(tmp_path):
    script = tmp_path / "r9_recipe_groups.py"
    script.write_text("old\n")
    sha = D.R.sha256_file(script)
    stage0b = {"passed": True, "item4": {"passed": True, "scripts": {script.name: {"sha256": sha}}}}
    script.write_text("changed\n")
    with pytest.raises(AssertionError, match="script SHA-256"):
        D.assert_script({"script": str(script), "script_sha256": sha}, stage0b, tmp_path)


@pytest.mark.parametrize("failure", ["missing_record", "failed_stage2", "failed_picks"])
def test_pins_refuses_unrecorded_part_or_failed_stage(tmp_path, failure):
    part = tmp_path / "picks_seed42.json"
    part.write_text(json.dumps({"passed": failure != "failed_picks"}))
    if failure != "missing_record":
        Path(str(part) + ".record.json").write_text('{}')
    stage = {"passed": failure != "failed_stage2", "backbone": "qwen3vl_emb_2b"}
    with pytest.raises(AssertionError, match={"missing_record": "record", "failed_stage2": "stage 2", "failed_picks": "picks"}[failure]):
        D.require_pinnable("qwen3vl_emb_2b", {"picks_seed42": part}, stage, recipe_roles={"picks_seed42"})


@pytest.mark.parametrize('failure', ['missing_record', 'failed_stage2', 'failed_picks'])
def test_pins_step_refuses_before_writing(tmp_path, monkeypatch, failure):
    backbone = 'qwen3vl_emb_2b'
    paths = {r: tmp_path / (r + '.json') for r in D.BB.CLIP_PARTS}
    for role, path in paths.items():
        path.write_text(json.dumps({'passed': failure != 'failed_picks'}))
        if role in D.BB.RECIPE_ROLES and not (role == 'n6_posteriors' and failure == 'missing_record'):
            Path(str(path) + '.record.json').write_text('{}')
    D.write_new(tmp_path / f'stage2_{backbone}.json',
                {'passed': failure != 'failed_stage2', 'backbone': backbone})
    target = tmp_path / 'final_pins.json'
    monkeypatch.setattr(D.R, 'pins_path', lambda b: target)
    monkeypatch.setattr(D.BB, 'parts', lambda b: paths)
    with pytest.raises(AssertionError, match={'missing_record': 'record', 'failed_stage2': 'stage 2',
                                           'failed_picks': 'picks'}[failure]):
        D.run_pins(backbone, paths, tmp_path, tmp_path / 'stage0b.json')
    assert not target.exists()


def test_clip_sensitivity_uses_round6_function_exactly():
    RH, RS, ST = D.round6_statistics()
    with np.load(D.R.R6D / 'results/seed42_per_episode.npz') as z:
        got = RS.sensitivity_record(z)
    stored = D.read_json(D.R.R6D / 'results/sensitivity_seed42.json')
    assert got == {k: stored[k] for k in got}


def test_made_differs_copied_equal_passes():
    D.assert_array_provenance({"coef": np.array([2.]), "rows": np.array([1])},
                              {"coef": np.array([1.]), "rows": np.array([1])}, ["coef"], ["rows"])
