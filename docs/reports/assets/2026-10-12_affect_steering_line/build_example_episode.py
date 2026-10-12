"""One real aspect episode from ArtELingo (emotion x genre, development seed 42) for the stage report.

    cd /project/CoSiR && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
    docs/reports/assets/2026-10-12_affect_steering_line/build_example_episode.py

Reads (writes nothing there): src/test/20261030_aspect_baselines/results/episodes_seed42.npz (seed 42, selection
rows = development paintings, never held rows), the ArtELingo annotations, the CLIP feature store's sample_ids and the
WikiArt images. Rows are feature-store rows; row r is annotations[sample_ids[r]] (the positional join, constitution
C6). Verifies the join and every episode rule before it draws; stops on any failure. Writes example_episode.json and
example_episode.png next to this script.

The episode was chosen by a scan of the 4,096 emotion x genre episodes of seed 42: the 190 whose 21 shown captions
(13 candidates, 4 + 4 example captions) all have at most 20 words; episode 3233 is one of them (longest caption 17
words), picked for a recognisable ukiyo-e portrait query and eight distinct, readable example pairs.
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre  # noqa: E402
from src.eval.aspect_episodes import (AspectEpisodes, PaintingValueIndex,  # noqa: E402
                                      validate_aspect_episodes)

SEED, PAIR, EPISODE = 42, "emotion__genre", 3233
EPISODES = ROOT / "src/test/20261030_aspect_baselines/results/episodes_seed42.npz"
ANNOTATIONS = Path("/data/PDD/artelingo/artelingo_train.json")
IMAGES = Path("/data/PDD/wikiart_proj/wikiart")          # scripts/extract_features.py ARTELINGO_IMAGES
TEAL, ORANGE, GREY, INK, INK2, GRID = "#1baf7a", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
GENRE_PRETTY = {g: g.replace("_painting", "").replace("_", " ") for g in GENRE_NAMES}
GENRE_PRETTY["genre_painting"] = "genre scene"


def main():
    # ------------------------------------------------------------ data, positional join
    data = load_artelingo()
    sid = np.asarray(data.sample_ids, dtype=np.int64)
    ann = json.loads(ANNOTATIONS.read_text())
    assert len(np.unique(sid)) == len(sid) == len(ann) and sid.min() >= 0 and sid.max() < len(ann)
    assert not (sid == np.arange(len(sid))).all(), "sample_ids are expected to be a permutation, not the identity"
    sp = artelingo_splits(data)
    labels = artelingo_aspect_labels(data)
    index = PaintingValueIndex(labels, sp.groups)
    selection = set(np.asarray(sp.selection).tolist())
    held = set(np.asarray(sp.held).tolist())

    emo_names = sorted({a["emotion"] for a in ann} - {"something else"})
    emo_of = lambda code: emo_names[code]                                   # noqa: E731  (sorted-name codes)

    with np.load(EPISODES) as z:
        g = lambda k: np.asarray(z[f"{PAIR}__{k}"][EPISODE:EPISODE + 1], dtype=np.int64)    # noqa: E731
        ep = AspectEpisodes("emotion", "genre", g("anchor"), g("candidates"), g("pairs_a_img"), g("pairs_a_txt"),
                            g("pairs_b_img"), g("pairs_b_txt"))
    q = int(ep.anchor[0])
    cand = ep.candidates[0]
    sa_i, sa_t = ep.pairs_a_img[0], ep.pairs_a_txt[0]          # supports: emotion
    sb_i, sb_t = ep.pairs_b_img[0], ep.pairs_b_txt[0]          # contrasts: genre
    every = np.concatenate([[q], cand, sa_i, sa_t, sb_i, sb_t])

    def rec(r):
        a = ann[int(sid[r])]
        return {"row": int(r), "annotation_index": int(sid[r]), "painting": a["painting"], "image_file": a["image"],
                "caption": a["caption"], "emotion": a["emotion"], "style": a["art_style"],
                "genre": GENRE_PRETTY[GENRE_NAMES[labels["genre"][r]]], "genre_code": int(labels["genre"][r])}

    # ------------------------------------------------------------ verification (every check printed to the json)
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            print("FAILED:", name, detail)

    emo, gen = labels["emotion"], labels["genre"]
    ann_emotion = lambda r: ann[int(sid[r])]["emotion"]                      # noqa: E731
    ann_genre = load_wikiart_genre(np.asarray([ann[int(sid[r])]["painting"] for r in every], dtype=object))
    gen_from_ann = dict(zip(every.tolist(), ann_genre.tolist()))
    check("labels_match_annotations_via_sample_ids",
          all(emo_of(emo[r]) == ann_emotion(r) and gen[r] == gen_from_ann[int(r)] and
              sp.groups.shape[0] == len(sid) for r in every),
          "emotion code -> name equals annotations[sample_ids[row]]['emotion']; genre equals the WikiArt join of "
          "annotations[sample_ids[row]]['painting']")
    check("all_rows_are_development_selection_rows",
          all(int(r) in selection and int(r) not in held for r in every), "selection split, none in held")
    check("all_21+_rows_from_distinct_paintings", len(np.unique(sp.groups[every])) == len(every) and
          len(set(ann[int(sid[r])]["painting"] for r in every)) == len(every), f"{len(every)} rows")
    try:
        validate_aspect_episodes(ep, labels, sp.groups, index)
        check("project_validate_aspect_episodes", True, "spec 5.1 rules, painting-level 'lacks' included")
    except AssertionError as e:
        check("project_validate_aspect_episodes", False, str(e))

    qa, qb = emo[q], gen[q]
    check("query_values", qa >= 0 and qb >= 0, f"emotion={emo_of(qa)}, genre={GENRE_PRETTY[GENRE_NAMES[qb]]}")
    for tag, imgs, txts, lab, other, qv in (("supports_emotion", sa_i, sa_t, emo, gen, qa),
                                            ("contrasts_genre", sb_i, sb_t, gen, emo, qb)):
        shared = [(int(lab[x]), int(lab[y])) for x, y in zip(imgs, txts)]
        check(f"{tag}: image and caption of each pair share the value", all(a == b for a, b in shared), str(shared))
        vals = [a for a, _ in shared]
        check(f"{tag}: four different values", len(set(vals)) == 4, str(vals))
        check(f"{tag}: none equals the query's value", qv not in vals, f"query {qv}")
        check(f"{tag}: image and caption differ on the other aspect",
              all(other[x] != other[y] for x, y in zip(imgs, txts)))
        check(f"{tag}: pairs avoid the query's value of the other aspect (painting level)",
              all(index.lacks("genre" if tag.startswith("supports") else "emotion",
                              qb if tag.startswith("supports") else qa)[r] for r in np.concatenate([imgs, txts])))
    p_a, p_b, negs = cand[0], cand[1], cand[2:]
    check("p_A shares the query's emotion, not its genre", emo[p_a] == qa and gen[p_a] != qb)
    check("p_B shares the query's genre, not its emotion", gen[p_b] == qb and emo[p_b] != qa)
    check("11 negatives share neither (row level)", len(negs) == 11 and (emo[negs] != qa).all() and
          (gen[negs] != qb).all())
    check("11 negatives share neither (painting level)", index.lacks("emotion", qa)[negs].all() and
          index.lacks("genre", qb)[negs].all())
    short = np.concatenate([cand, sa_t, sb_t])      # captions shown (image rows show only the painting)
    words = {int(r): len(ann[int(sid[r])]["caption"].split()) for r in short}
    check("captions shown have at most 20 words", max(words.values()) <= 20, f"max {max(words.values())}")
    all_pass = all(c["pass"] for c in checks.values())
    print(f"{sum(c['pass'] for c in checks.values())}/{len(checks)} checks pass")
    if not all_pass:
        sys.exit("join verification failed; no figure drawn")

    # ------------------------------------------------------------ shown content
    perm = np.random.default_rng(0).permutation(13)          # candidate display order: shuffled, so the position tells nothing
    roles = np.array(["p_A"] + ["p_B"] + ["negative"] * 11)
    cand_rows = cand[perm]
    out = {
        "episode": {"seed": SEED, "pair": PAIR, "episode_index": EPISODE, "condition": "a (supports show emotion)",
                    "query_direction": "image query ranks captions", "source": str(EPISODES.relative_to(ROOT)),
                    "rows_are": "FeatureManager feature rows; annotation = annotations[sample_ids[row]]",
                    "candidate_display_order": "shuffled with np.random.default_rng(0).permutation(13)",
                    "scores": "none added: seed 42 has only per-anchor means over both conditions and directions; "
                              "per-ranking AFF scores are not cached for this episode"},
        "query": rec(q),
        "supports_emotion": [{"emotion_shared": emo_of(emo[x]), "image": rec(x), "caption_of": rec(y)}
                             for x, y in zip(sa_i, sa_t)],
        "contrasts_genre": [{"genre_shared": GENRE_PRETTY[GENRE_NAMES[gen[x]]], "image": rec(x), "caption_of": rec(y)}
                            for x, y in zip(sb_i, sb_t)],
        "candidates": [{"role": roles[k], **rec(cand[k])} for k in perm],
        "checks": checks,
    }
    (HERE / "example_episode.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=int))

    # ------------------------------------------------------------ figure
    W, H = 16.0, 9.0
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def thumb(r, x, y, size):
        """Draw painting r in a size x size box with top-left (x, y), aspect kept."""
        im = Image.open(IMAGES / ann[int(sid[r])]["image"]).convert("RGB")
        im.thumbnail((600, 600))
        w, h = im.size
        dw, dh = (size, size * h / w) if w >= h else (size * w / h, size)
        ax.imshow(np.asarray(im), extent=(x + (size - dw) / 2, x + (size + dw) / 2, y + (size + dh) / 2,
                                          y + (size - dh) / 2), aspect="auto", zorder=2)

    def wrap(text, width):
        return "\n".join(textwrap.wrap(text, width))

    ax.text(0.35, 0.38, "A real aspect episode from ArtELingo (emotion × genre, development paintings)",
            fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(0.35, 0.74, "Condition a: the examples show emotion. An image query ranks 13 captions; the target is the "
            "caption with the query's emotion.", fontsize=10.5, color=INK2, va="center")

    # left: query
    qx, qy, qs = 0.35, 1.45, 2.75
    ax.text(qx, 1.15, "QUERY (a painting)", fontsize=10.5, fontweight="bold", color=INK, va="center")
    thumb(q, qx, qy, qs)
    ax.add_patch(Rectangle((qx - 0.03, qy - 0.03), qs + 0.06, qs + 0.06, fill=False, ec=INK, lw=1.4, zorder=3))
    qr = out["query"]
    ax.text(qx, qy + qs + 0.3, "The query's two values", fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(qx, qy + qs + 0.7, f"emotion: {qr['emotion']}", fontsize=11, color=TEAL, fontweight="bold", va="center")
    ax.text(qx, qy + qs + 1.05, f"genre: {qr['genre']}", fontsize=11, color=ORANGE, fontweight="bold", va="center")
    ax.text(qx, qy + qs + 1.4, f"(style: {qr['style'].replace('_', ' ')})", fontsize=8.5, color=INK2, va="center")
    ax.text(qx, qy + qs + 1.85, wrap("Each example pair is an image of one painting with a caption of another "
                                     "painting.", 30), fontsize=8.5, color=INK2, va="top")

    # middle: two blocks of four pairs
    mx0, ts, rowh = 3.75, 0.66, 0.74
    mid_w = 6.15

    def block(y0, title, colour, pairs, tagkey, tagname):
        ax.text(mx0, y0 - 0.2, title, fontsize=10.5, fontweight="bold", color=colour, va="center")
        for i, p in enumerate(pairs):
            y = y0 + i * rowh
            ax.add_patch(FancyBboxPatch((mx0, y), mid_w, rowh - 0.07, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc="white", ec=colour, lw=1.8, zorder=1))
            thumb(p["image"]["row"], mx0 + 0.06, y + 0.0 + (rowh - 0.07 - ts) / 2, ts)
            ax.text(mx0 + 0.06 + ts + 0.1, y + 0.1, f"{tagname}: {p[tagkey]}", fontsize=8.5, fontweight="bold",
                    color=colour, va="top", zorder=3)
            ax.text(mx0 + 0.06 + ts + 0.1, y + 0.31, wrap("“" + p["caption_of"]["caption"] + "”", 62),
                    fontsize=8.5, color=INK, va="top", linespacing=1.15, zorder=3)

    block(1.75, "SUPPORTS: 4 pairs, each pair agrees on emotion", TEAL, out["supports_emotion"], "emotion_shared",
          "emotion")
    block(5.3, "CONTRASTS: 4 pairs, each pair agrees on genre", ORANGE, out["contrasts_genre"], "genre_shared",
          "genre")

    # right: 13 candidates
    cx0, cw = 10.2, 5.45
    ax.text(cx0, 1.15, "13 CANDIDATE CAPTIONS (the query ranks them)", fontsize=10.5, fontweight="bold", color=INK,
            va="center")
    crow = 0.5
    y0c = 1.45
    for k, c in enumerate(out["candidates"]):
        y = y0c + k * crow
        role = c["role"]
        fc, ec, lw = {"p_A": ("#d9f3e8", TEAL, 2.0), "p_B": ("#fbe1d6", ORANGE, 2.0)}.get(role, ("#f3f2ef", GRID, 1.0))
        ax.add_patch(FancyBboxPatch((cx0, y), cw, crow - 0.06, boxstyle="round,pad=0,rounding_size=0.05", fc=fc,
                                    ec=ec, lw=lw, zorder=1))
        ax.text(cx0 + 0.08, y + (crow - 0.06) / 2, wrap(c["caption"], 68 if role == "negative" else 46), fontsize=8,
                color=INK, va="center", linespacing=1.1, zorder=3)
        if role != "negative":
            lab = ("shares the query's emotion" if role == "p_A" else "shares the query's genre")
            col = TEAL if role == "p_A" else ORANGE
            ax.text(cx0 + cw - 0.07, y + (crow - 0.06) / 2, f"{role}: " + wrap(lab, 20), fontsize=7.2,
                    color=col, fontweight="bold", ha="right", va="center", zorder=3, linespacing=1.05)
    ly, cx0 = 6.95, 0.35
    ax.text(cx0, ly - 0.3, "Candidate colours", fontsize=10.5, fontweight="bold", color=INK, va="center")
    for k, (fc, ec, txt) in enumerate([("#d9f3e8", TEAL, f"p_A (target): shares the query's emotion ({qr['emotion']})"),
                                      ("#fbe1d6", ORANGE, f"p_B: shares the query's genre ({qr['genre']})"),
                                      ("#f3f2ef", GRID, "11 negatives: share neither")]):
        ax.add_patch(Rectangle((cx0, ly + k * 0.42), 0.25, 0.17, fc=fc, ec=ec, lw=1.5))
        ax.text(cx0 + 0.35, ly + k * 0.42 + 0.085, wrap(txt, 38), fontsize=8, color=INK, va="center")

    # footer
    ax.add_patch(Rectangle((0, 8.58), W, 0.01, fc=GRID, ec="none"))
    ax.text(W / 2, 8.8, "The aspect is never named. The examples never show the query's own value. Swapping the two "
            "blocks makes the genre candidate the target.", fontsize=11, color=INK, ha="center", va="center")
    ax.text(0.35, 8.4, f"seed {SEED}, episode {EPISODE} (emotion × genre); development selection split", fontsize=7,
            color=GREY, ha="left", va="center")
    fig.savefig(HERE / "example_episode.png", dpi=200)
    print("wrote", HERE / "example_episode.png")


if __name__ == "__main__":
    main()
