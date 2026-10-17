"""Fixed, label-independent caption masks and word statistics (no model loading)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PAV = HERE.parent / "20261017_prompted_aspect_views"
sys.path.insert(0, str(PAV))
from pav_sample import load_frozen

KEYS = ("default", "emotion_focus")
EMOTION_NAMES = ("amusement", "anger", "awe", "contentment", "disgust", "excitement",
                 "fear", "sadness", "something else")
LABEL_FORMS = {
    "amusement": "amusement amused amusing amuse amuses".split(),
    "awe": "awe awed awestruck awe-inspiring awesome".split(),
    "contentment": "contentment content contented".split(),
    "excitement": "excitement excited exciting excite excites".split(),
    "anger": "anger angry angered angrily".split(),
    "disgust": "disgust disgusted disgusting disgusts".split(),
    "fear": ("fear fears feared fearful afraid scared scary scare scares frightened frightening "
             "terrified terrifying").split(),
    "sadness": "sadness sad saddened saddening sadly".split(),
}
WORD_LABEL = {word: EMOTION_NAMES.index(label) for label, words in LABEL_FORMS.items() for word in words}
BROAD_EXTRA = (
    "happy happiness happily joy joyful joyous glad cheerful delight delighted delightful calm calming calmness "
    "peaceful peace serene serenity relaxed relaxing pleasant pleased love loving lovely lonely loneliness "
    "depressed depressing depression gloomy melancholy melancholic upset upsetting worried worry anxious "
    "anxiety nervous uneasy horror horrified horrible horrifying creepy eerie ominous disturbing disturbed "
    "gross funny fun laugh laughing nostalgic nostalgia hopeful hope hopeless grief sorrow sorrowful mournful "
    "tragic cozy comforting comfortable inspired inspiring curious curiosity intrigued intriguing bored boring "
    "annoyed annoying irritated irritating uncomfortable unsettling emotional feel feeling feelings feels felt"
).split()
WORDS = {"label": tuple(sorted(WORD_LABEL)), "broad": tuple(sorted(set(WORD_LABEL) | set(BROAD_EXTRA)))}
PATTERNS = {name: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w)))
                            + r")\b", re.IGNORECASE) for name, words in WORDS.items()}


def json_hash(value):
    """SHA256 of UTF-8 JSON, ensure_ascii=False and compact separators; order preserved."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


MASK_SPEC_SHA256 = json_hash(dict(words=WORDS, word_label=WORD_LABEL, token="[MASK]",
                                  pattern="whole-word Unicode boundaries; IGNORECASE; longest first"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + ".pending" + path.suffix)
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def mask_text(text, mask):
    return PATTERNS[mask].sub("[MASK]", text)


def load_captions(s):
    from scripts.extract_features import adapter_artelingo_full
    spec = adapter_artelingo_full(sample_ids=s["sample_ids"], expected=None)
    assert np.array_equal([m["sample_id"] for m in spec.text_meta], s["sample_ids"])
    assert np.array_equal([m["painting"] for m in spec.text_meta], s["painting_ids"])
    assert len(spec.texts) == len(s["sample_ids"]) == 99999
    return list(spec.texts)


def assert_emotion_map(s):
    """Scoring-only check of the apc_data.load_sample codebook against raw annotation IDs."""
    from scripts.extract_features import ARTELINGO_ANN
    from src.data.artelingo_splits import encode_labels
    annotations = json.loads(ARTELINGO_ANN.read_text())
    codes, names = encode_labels([row["emotion"] for row in annotations])
    assert tuple(names) == EMOTION_NAMES, names
    assert np.array_equal(codes[s["sample_ids"]], s["emo"]), "frozen emotion codes differ"
    assert np.array_equal(np.unique(s["emo"]), np.arange(9))
    return names


def build_masks(texts):
    masked, counters = {}, {}
    counts = np.zeros((len(texts), 9), dtype=np.uint32)
    flags = {}
    for name, pattern in PATTERNS.items():
        out, counter = [], Counter()
        flag = np.zeros(len(texts), dtype=bool)
        for i, text in enumerate(texts):
            matches = [m.group(0).lower() for m in pattern.finditer(text)]
            flag[i] = bool(matches)
            counter.update(matches)
            if name == "label":
                for word in matches:
                    counts[i, WORD_LABEL[word]] += 1
            out.append(pattern.sub("[MASK]", text))
        masked[name], counters[name], flags[name] = out, counter, flag
    assert np.all(~flags["label"] | flags["broad"])
    return dict(masked=masked, counters=counters, has_label_word=flags["label"],
                has_broad_word=flags["broad"], label_word_counts=counts,
                source_sha256=json_hash(texts),
                masked_sha256={name: json_hash(out) for name, out in masked.items()})


def save_masks(bundle, s, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, out in bundle["masked"].items():
        atomic_json(directory / f"masked_captions_{name}.json", out)
    pending = directory / "ewm_masks.pending.npz"
    np.savez_compressed(pending, sample_ids=s["sample_ids"],
                        has_label_word=bundle["has_label_word"], has_broad_word=bundle["has_broad_word"],
                        label_word_counts=bundle["label_word_counts"], emotion_names=np.asarray(EMOTION_NAMES),
                        source_sha256=bundle["source_sha256"], mask_spec_sha256=MASK_SPEC_SHA256,
                        **{f"masked_{name}_sha256": value for name, value in bundle["masked_sha256"].items()})
    pending.replace(directory / "ewm_masks.npz")


def distribution(y):
    counts = np.bincount(y, minlength=9)
    return dict(n=int(len(y)), counts=counts.tolist(),
                shares=(counts / len(y)).tolist() if len(y) else [None] * 9)


def word_statistics(bundle, y):
    own = bundle["label_word_counts"][np.arange(len(y)), y] > 0
    matching = bundle["has_label_word"]
    out = dict(n=len(y), emotion_names=list(EMOTION_NAMES), full_distribution=distribution(y),
               label_agreement_definition="Among captions with label matches, at least one matched word maps to the own label.",
               label_agreement_n=int((own & matching).sum()),
               label_agreement_share=float(own[matching].mean()) if matching.any() else None,
               masks={})
    for name in WORDS:
        flag = bundle[f"has_{name}_word"]
        per_label = {}
        for i, label in enumerate(EMOTION_NAMES):
            rows = y == i
            hit = rows & flag
            per_label[label] = dict(n=int(rows.sum()), matched_n=int(hit.sum()), share=float(flag[rows].mean()))
            if name == "label":
                per_label[label]["own_label_agreement_share"] = float(own[hit].mean()) if hit.any() else None
        out["masks"][name] = dict(matched_n=int(flag.sum()), share=float(flag.mean()), per_emotion=per_label,
            top30=[dict(word=w, count=int(c)) for w, c in bundle["counters"][name].most_common(30)])
    out["subgroup_distributions"] = {f"no_{name}_word": distribution(y[~bundle[f"has_{name}_word"]]) for name in WORDS}
    return out


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE)
    args = p.parse_args()
    s = load_frozen()
    assert_emotion_map(s)
    bundle = build_masks(load_captions(s))
    save_masks(bundle, s, args.output_dir)
    stats = word_statistics(bundle, s["emo"])
    atomic_json(args.output_dir / "ewm_word_statistics.json", stats)
    for name, v in stats["masks"].items():
        print(f"{name}: {v['matched_n']}/{len(s['emo'])} = {v['share']:.6f}; top10={v['top30'][:10]}", flush=True)
    print(f'Finished word statistics in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == "__main__":
    main()
