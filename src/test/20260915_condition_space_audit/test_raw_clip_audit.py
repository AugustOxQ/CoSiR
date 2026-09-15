import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import raw_clip_audit as rca


def test_evaluate_direction_perfect_separation():
    features = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    labels = np.array([1, 1, 0, 0])
    direction = np.array([1.0, 0.0])
    auc = rca.evaluate_direction(features, labels, direction)
    assert np.isclose(auc, 1.0)


def test_fold_auc_symmetric():
    assert np.isclose(rca.fold_auc(0.2), 0.8)
    assert np.isclose(rca.fold_auc(0.8), 0.8)
    assert np.isclose(rca.fold_auc(0.5), 0.5)


def test_decision_rule_positive():
    assert rca.decision_rule(real_auc_folded=0.75, z=3.0) == "positive"


def test_decision_rule_partial():
    assert rca.decision_rule(real_auc_folded=0.55, z=2.5) == "partial"


def test_decision_rule_null():
    assert rca.decision_rule(real_auc_folded=0.52, z=0.8) == "null"


def test_audit_axis_reports_content_control_rank_and_same_image_na(monkeypatch):
    real_direction = np.array([1.0, 0.0])
    content_control_direction = np.array([0.0, 1.0])
    control_directions = [np.array([0.0, 1.0]), np.array([0.0, -1.0])]
    monkeypatch.setattr(
        rca.ta,
        "build_direction",
        lambda *args, **kwargs: real_direction if args[2] == ["real a"] else content_control_direction,
    )
    monkeypatch.setattr(rca.ta, "build_control_directions", lambda *args, **kwargs: control_directions)

    features = np.array([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    labels = np.array([1, 1, 0, 0])
    records = [{"ImgId": "shared"} for _ in labels]
    result = rca.audit_axis(
        {"img": features, "txt": features},
        np.arange(len(labels)),
        labels,
        ["real a"],
        ["real b"],
        None,
        None,
        n_control=2,
        records=records,
        content_control_prompts_a=["content a"],
        content_control_prompts_b=["content b"],
    )

    assert result["img"]["verdict"] == "n/a_same_images"
    assert result["txt"]["verdict"] != "n/a_same_images"
    for modality in ("img", "txt"):
        assert result[modality]["content_control_auc"] == 0.5
        assert result[modality]["content_control_auc_folded"] == 0.5
        assert result[modality]["beats_content_control"] is True
        assert result[modality]["control_rank"] == "2 of 2 controls exceeded"


def test_audit_axis_skips_same_image_check_without_imgids(monkeypatch):
    monkeypatch.setattr(rca.ta, "build_direction", lambda *args, **kwargs: np.array([1.0, 0.0]))
    monkeypatch.setattr(rca.ta, "build_control_directions", lambda *args, **kwargs: [np.array([0.0, 1.0])] * 2)
    features = np.array([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    labels = np.array([1, 1, 0, 0])
    result = rca.audit_axis(
        {"img": features}, np.arange(len(labels)), labels, ["a"], ["b"], None, None,
        n_control=2, records=[{} for _ in labels],
    )
    assert result["img"]["verdict"] != "n/a_same_images"
    assert result["img"]["content_control_auc"] is None
    assert result["img"]["content_control_auc_folded"] is None
    assert result["img"]["beats_content_control"] is None
