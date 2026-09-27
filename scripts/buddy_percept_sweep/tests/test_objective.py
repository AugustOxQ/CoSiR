from scripts.buddy_percept_sweep.objective import compute_objective


def test_both_clear_returns_auc():
    assert compute_objective(0.20, 0.30, 0.87) == 0.87


def test_emotion_fails_returns_sentinel():
    assert compute_objective(0.10, 0.30, 0.99) == -1.0


def test_genre_fails_returns_sentinel():
    assert compute_objective(0.20, 0.10, 0.99) == -1.0


def test_both_fail_returns_sentinel():
    assert compute_objective(0.01, 0.01, 0.99) == -1.0


def test_boundary_exactly_at_bar_does_not_clear():
    # Strictly greater-than, not greater-or-equal.
    assert compute_objective(0.1236, 0.30, 0.99) == -1.0
    assert compute_objective(0.20, 0.1954, 0.99) == -1.0
