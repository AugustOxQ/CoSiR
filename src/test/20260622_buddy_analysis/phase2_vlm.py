"""
Phase 2 — VLM pairwise judgement: are cross-photo buddies "same content"?

For each anchor image i, we ask a VLM (Qwen2.5-VL via vLLM) whether each candidate
caption is a GOOD match for image_i, where candidates are:
  * BUDDY captions   — captions of i's CROSS-PHOTO buddies (different source image)
  * RANDOM captions  — type-matched captions from random cross-photo samples

Reuses the exact good/bad prompt from src/test/automatic_annotator/qwenannotator.py.
Type-matching the random set isolates "same visual content" from "same caption style".
If buddies are meaningful, BUDDY captions should be judged GOOD far more often than
type-matched RANDOM captions for the same image.

Usage:
  python phase2_vlm.py --graph B --n_anchors 150 --dry_run     # no server needed
  bash ../automatic_annotator/launch_vllm.sh                   # start vLLM (separate)
  python phase2_vlm.py --graph B --n_anchors 150               # real run
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "automatic_annotator")))
import buddy_analysis as ba
from qwenannotator import SYSTEM_PROMPT, _build_user_prompt, _parse_response, _encode_image_base64

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "docs", "reports", "assets", "buddy_analysis"))


def cross_photo_neighbors(data, edges):
    """adjacency: node -> set of cross-photo buddy nodes (different source image)."""
    adj = {i: [] for i in range(data.n)}
    for a, b in edges:
        if data.source_id[a] != data.source_id[b]:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def build_candidates(data, adj, n_anchors, max_buddies, seed):
    """Per anchor: (buddy_idx list, type-matched random_idx list). Returns list of dicts."""
    rng = np.random.default_rng(seed)
    # caption indices grouped by type, for type-matched random draws
    by_type = {t: np.where(data.types == t)[0] for t in range(len(ba.TYPE_NAMES))}
    eligible = [i for i in range(data.n) if adj[i]]
    rng.shuffle(eligible)
    anchors = eligible[:n_anchors]
    out = []
    for i in anchors:
        buddies = list(dict.fromkeys(adj[i]))           # dedup, keep order
        if len(buddies) > max_buddies:
            buddies = list(rng.choice(buddies, max_buddies, replace=False))
        buddy_set = set(adj[i]) | {i}
        randoms = []
        for b in buddies:                                # one type-matched random per buddy
            t = data.types[b]
            for _ in range(20):
                r = int(rng.choice(by_type[t]))
                if data.source_id[r] != data.source_id[i] and r not in buddy_set:
                    randoms.append(r); break
        out.append({"anchor": int(i), "buddies": [int(b) for b in buddies],
                    "randoms": [int(r) for r in randoms]})
    return out


def run(args):
    data = ba.load_data()
    G = ba.build_graphs(data, K=30)
    edges = ba.edges(G[args.graph])
    adj = cross_photo_neighbors(data, edges)
    cand = build_candidates(data, adj, args.n_anchors, args.max_buddies, args.seed)
    n_pairs = sum(len(c["buddies"]) for c in cand)
    print(f"graph={args.graph}  anchors={len(cand)}  buddy candidates={n_pairs}  "
          f"(+ {sum(len(c['randoms']) for c in cand)} type-matched random)")

    if args.dry_run:
        c = cand[0]
        ai = c["anchor"]
        print("\n--- example anchor ---")
        print(f"anchor img: {data.records[ai]['image']}  type={ba.TYPE_NAMES[data.types[ai]]}")
        print(f"anchor caption: {data.records[ai]['caption'][:120]}")
        for b in c["buddies"][:3]:
            print(f"  BUDDY  ({ba.TYPE_NAMES[data.types[b]]}, photo {data.source_id[b]}): "
                  f"{data.records[b]['caption'][:110]}")
        for r in c["randoms"][:3]:
            print(f"  RANDOM ({ba.TYPE_NAMES[data.types[r]]}, photo {data.source_id[r]}): "
                  f"{data.records[r]['caption'][:110]}")
        print("\n[dry-run] no server calls made.")
        return

    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    def judge(img_path, captions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_image_base64(img_path)}},
                {"type": "text", "text": _build_user_prompt(captions)}]}]
        for _ in range(args.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model_name, messages=messages, max_tokens=512,
                    temperature=0.0, extra_body={"top_k": 1})
                r = _parse_response(resp.choices[0].message.content or "", len(captions))
                if r is not None:
                    return r
            except Exception as e:
                print("  api error:", e)
        return None

    buddy_good, rand_good = [], []
    per_anchor = []
    for k, c in enumerate(cand):
        ai = c["anchor"]
        img_path = os.path.join(ba.IMG_ROOT, data.records[ai]["image"])
        items = [("buddy", b) for b in c["buddies"]] + [("rand", r) for r in c["randoms"]]
        if not items:
            continue
        order = np.random.default_rng(args.seed + k).permutation(len(items))
        items = [items[o] for o in order]
        caps = [data.records[idx]["caption"] for _, idx in items]
        res = judge(img_path, caps)
        if res is None:
            continue
        bg = [v for (lab, _), v in zip(items, res) if lab == "buddy" and v in (0, 1)]
        rg = [v for (lab, _), v in zip(items, res) if lab == "rand" and v in (0, 1)]
        buddy_good += bg; rand_good += rg
        if bg and rg:
            per_anchor.append(float(np.mean(bg)) - float(np.mean(rg)))
        if k % 10 == 0:
            print(f"  [{k}/{len(cand)}] buddy_good={np.mean(buddy_good or [0]):.3f} "
                  f"rand_good={np.mean(rand_good or [0]):.3f}")

    summary = {
        "graph": args.graph, "model": args.model_name,
        "n_anchors_judged": len(per_anchor),
        "n_buddy_pairs": len(buddy_good), "n_random_pairs": len(rand_good),
        "buddy_good_rate": float(np.mean(buddy_good)) if buddy_good else None,
        "random_good_rate": float(np.mean(rand_good)) if rand_good else None,
        "paired_diff_mean": float(np.mean(per_anchor)) if per_anchor else None,
        "paired_diff_ci95": [float(np.percentile(_boot(per_anchor), 2.5)),
                             float(np.percentile(_boot(per_anchor), 97.5))] if per_anchor else None,
    }
    out = os.path.join(ASSETS, f"phase2_vlm_{args.graph}.json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\n", json.dumps(summary, indent=2))
    print("wrote", out)


def _boot(x, n=2000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    return np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=["B", "E"], default="B")
    p.add_argument("--n_anchors", type=int, default=150)
    p.add_argument("--max_buddies", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max_retries", type=int, default=3)
    run(p.parse_args())
