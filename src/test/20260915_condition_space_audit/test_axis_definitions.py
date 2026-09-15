import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import axis_definitions as ax


def test_redcaps_warmth_labels():
    records = [
        {"image": "redcaps/images2020/cats/a.jpg"},
        {"image": "redcaps/images2020/mildlyinteresting/b.jpg"},
        {"image": "redcaps/images2020/gardening/c.jpg"},  # neither pole — excluded
        {"image": "redcaps/images2020/rarepuppers/d.jpg"},
    ]
    keep, labels = ax.redcaps_binary_labels(records, "warmth")
    assert list(keep) == [0, 1, 3]
    assert list(labels) == [1, 0, 1]


def test_impressions_aesthetic_vs_description_labels():
    records = [
        {"caption_type": "aesthetic"},
        {"caption_type": "description"},
        {"caption_type": "impression"},  # neither pole — excluded
        {"caption_type": "aesthetic"},
    ]
    keep, labels = ax.impressions_binary_labels(records, "aesthetic_vs_description")
    assert list(keep) == [0, 1, 3]
    assert list(labels) == [1, 0, 1]


def test_all_axes_have_prompts():
    for axes in (ax.REDCAPS_AXES, ax.IMPRESSIONS_AXES):
        for name, spec in axes.items():
            assert spec["prompts_a"], name
            assert spec["prompts_b"], name


def test_redcaps_axes_have_content_matched_controls_only():
    for name, spec in ax.REDCAPS_AXES.items():
        assert spec["content_control_prompts_a"], name
        assert spec["content_control_prompts_b"], name
    for spec in ax.IMPRESSIONS_AXES.values():
        assert "content_control_prompts_a" not in spec
        assert "content_control_prompts_b" not in spec
