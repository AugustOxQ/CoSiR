"""Concrete two-pole label-group and prompt-pair definitions for the Exp. 17.1
condition-space audit. Subreddit/caption_type pools were chosen from labels
verified as well-populated in redcaps_150k / impressions_train during spec
research (2026-09-15) — see docs/reports/2026-09-15_condition_space_audit.md.
"""
import numpy as np

REDCAPS_AXES = {
    "warmth": {
        "description": (
            "companion-animal/cute subreddits vs. curiosity-framed subreddits "
            "— proxy for positive emotional valence"
        ),
        "pole_a": ["cats", "rarepuppers", "blackcats", "dogpictures", "pitbulls", "guineapigs", "eyebleach"],
        "pole_b": ["mildlyinteresting", "interestingasfuck", "natureisfuckinglit"],
        "prompts_a": [
            "a photo of a cute, happy animal",
            "an adorable pet photo",
            "a heartwarming picture of a beloved animal",
        ],
        "prompts_b": [
            "a photo of a surprising, curious object",
            "an unusual and interesting scene",
            "a strange or unexpected sight",
        ],
        "content_control_prompts_a": [
            "a photo of an animal",
            "an image of an animal",
            "a picture of an animal",
        ],
        "content_control_prompts_b": [
            "a photo of an object or scene",
            "an image of an object or scene",
            "a picture of an object or scene",
        ],
    },
    "register": {
        "description": (
            "'porn'-tagged aesthetic-photography subreddits vs. casual snapshot "
            "subreddits — proxy for formal/descriptive vs. casual register"
        ),
        "pole_a": ["earthporn", "foodporn", "carporn"],
        "pole_b": ["mildlyinteresting", "itookapicture"],
        "prompts_a": [
            "a professionally composed, high-quality photograph",
            "a formal, polished piece of photography",
            "an artfully composed image",
        ],
        "prompts_b": [
            "a casual snapshot photo",
            "an informal, quick picture",
            "a plain everyday photo",
        ],
        "content_control_prompts_a": [
            "a photo of nature, food, or a vehicle",
            "an image of nature, food, or a vehicle",
            "a picture of nature, food, or a vehicle",
        ],
        "content_control_prompts_b": [
            "a photo of an everyday subject",
            "an image of an everyday subject",
            "a picture of an everyday subject",
        ],
    },
}

IMPRESSIONS_AXES = {
    "aesthetic_vs_description": {
        "description": (
            "subjective aesthetic captions vs. factual descriptive captions, "
            "same images — proxy for framing register"
        ),
        "pole_a_caption_type": "aesthetic",
        "pole_b_caption_type": "description",
        "prompts_a": [
            "a subjective, evocative description of an image's aesthetic qualities",
            "an artful, impressionistic caption",
        ],
        "prompts_b": [
            "a plain, factual description of an image's contents",
            "an objective, literal caption",
        ],
    },
    "impression_vs_caption": {
        "description": (
            "interpretive clinical-impression captions vs. plain captions, same "
            "images — proxy for interpretive vs. literal framing"
        ),
        "pole_a_caption_type": "impression",
        "pole_b_caption_type": "caption",
        "prompts_a": [
            "an interpretive clinical impression of an image",
            "a diagnostic-style summary judgment",
        ],
        "prompts_b": [
            "a plain caption naming what is shown",
            "a simple literal label for an image",
        ],
    },
}


def _subreddit_of(record):
    parts = record["image"].split("/")
    return parts[2] if len(parts) > 2 else "?"


def redcaps_binary_labels(records, axis_name):
    axis = REDCAPS_AXES[axis_name]
    keep, labels = [], []
    for i, r in enumerate(records):
        sub = _subreddit_of(r)
        if sub in axis["pole_a"]:
            keep.append(i)
            labels.append(1)
        elif sub in axis["pole_b"]:
            keep.append(i)
            labels.append(0)
    return np.array(keep, dtype=np.int64), np.array(labels, dtype=np.int64)


def impressions_binary_labels(records, axis_name):
    axis = IMPRESSIONS_AXES[axis_name]
    keep, labels = [], []
    for i, r in enumerate(records):
        ct = r.get("caption_type")
        if ct == axis["pole_a_caption_type"]:
            keep.append(i)
            labels.append(1)
        elif ct == axis["pole_b_caption_type"]:
            keep.append(i)
            labels.append(0)
    return np.array(keep, dtype=np.int64), np.array(labels, dtype=np.int64)
