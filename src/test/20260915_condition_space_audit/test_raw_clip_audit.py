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
