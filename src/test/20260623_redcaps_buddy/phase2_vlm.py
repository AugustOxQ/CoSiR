"""
Phase 2 (RedCaps) — VLM pairwise judgement: do buddies share specific content?

For each anchor image i, ask a VLM (Qwen2.5-VL via vLLM) whether each candidate
caption is a GOOD match for image_i, where candidates are:
  * BUDDY captions          — captions of i's buddies (every RedCaps edge is
                              already cross-content; 1 image : 1 caption).
  * SUBREDDIT-RANDOM captions — captions from the ANCHOR's subreddit but a different
                              image (hard negative: same topic, different content).
  * PLAIN-RANDOM captions   — captions from any subreddit (easy floor).

Reuses the good/bad prompt from src/test/automatic_annotator/qwenannotator.py
(already written for Reddit data). If buddies capture specific content, BUDDY should
be judged GOOD far more than SUBREDDIT-RANDOM, which in turn beats PLAIN-RANDOM.

Usage:
  python phase2_vlm.py --graph B --n_anchors 150 --dry_run    # no server needed
  bash ../automatic_annotator/launch_vllm.sh                  # start vLLM (separate)
  python phase2_vlm.py --graph B --n_anchors 150              # real run
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "automatic_annotator")))
import redcaps_buddy as rb
from qwenannotator import SYSTEM_PROMPT, _build_user_prompt, _parse_response, _encode_image_base64

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "docs", "reports", "assets", "redcaps_buddy"))


def neighbors(data, edges):
    """node -> list of buddy nodes (all edges; RedCaps is 1:1 so all cross-content)."""
    adj = {i: [] for i in range(data.n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def build_candidates(data, adj, n_anchors, max_buddies, seed):
    """Per anchor: buddy idx + subreddit-matched random + plain random idx lists."""
    rng = np.random.default_rng(seed)
    by_sub = {s: np.where(data.sub_id == s)[0] for s in range(len(data.sub_names))}
    eligible = [i for i in range(data.n) if adj[i]]
    rng.shuffle(eligible)
    anchors = eligible[:n_anchors]
    out = []
    for i in anchors:
        buddies = list(dict.fromkeys(adj[i]))            # dedup, keep order
        if len(buddies) > max_buddies:
            buddies = list(rng.choice(buddies, max_buddies, replace=False))
        buddy_set = set(adj[i]) | {i}
        pool_sub = by_sub[data.sub_id[i]]                # anchor's subreddit
        sub_rand, plain_rand = [], []
        for _ in buddies:                                # one of each negative per buddy
            for _ in range(30):
                r = int(rng.choice(pool_sub))
                if r != i and r not in buddy_set:
                    sub_rand.append(r); break
            for _ in range(30):
                r = int(rng.integers(0, data.n))
                if r != i and r not in buddy_set:
                    plain_rand.append(r); break
        out.append({"anchor": int(i), "buddies": [int(b) for b in buddies],
                    "sub_randoms": [int(r) for r in sub_rand],
                    "plain_randoms": [int(r) for r in plain_rand]})
    return out


def _boot(x, n=2000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    return np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])


def _sub(data, idx):
    return data.sub_names[data.sub_id[idx]]


def run(args):
    data = rb.load_data()
    G = rb.build_graphs(data, K=30)                      # raw B/E, no bridge edges
    edges = rb.edges(G[args.graph])
    adj = neighbors(data, edges)
    cand = build_candidates(data, adj, args.n_anchors, args.max_buddies, args.seed)
    n_buddy = sum(len(c["buddies"]) for c in cand)
    print(f"graph={args.graph}  anchors={len(cand)}  buddy candidates={n_buddy}  "
          f"(+ {sum(len(c['sub_randoms']) for c in cand)} subreddit-random, "
          f"{sum(len(c['plain_randoms']) for c in cand)} plain-random)")

    if args.dry_run:
        c = cand[0]; ai = c["anchor"]
        print("\n--- example anchor ---")
        print(f"anchor img: {data.records[ai]['image']}  r/{_sub(data, ai)}")
        print(f"anchor caption: {data.records[ai]['caption'][:120]}")
        for b in c["buddies"][:3]:
            print(f"  BUDDY      (r/{_sub(data, b)}): {data.records[b]['caption'][:100]}")
        for r in c["sub_randoms"][:3]:
            print(f"  SUB-RANDOM (r/{_sub(data, r)}): {data.records[r]['caption'][:100]}")
        for r in c["plain_randoms"][:3]:
            print(f"  RANDOM     (r/{_sub(data, r)}): {data.records[r]['caption'][:100]}")
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

    good = {"buddy": [], "sub": [], "plain": []}
    per_anchor = {"sub": [], "plain": []}                # paired (buddy - negative) per anchor
    for k, c in enumerate(cand):
        ai = c["anchor"]
        img_path = os.path.join(rb.IMG_ROOT, data.records[ai]["image"])
        items = ([("buddy", b) for b in c["buddies"]]
                 + [("sub", r) for r in c["sub_randoms"]]
                 + [("plain", r) for r in c["plain_randoms"]])
        if not items:
            continue
        order = np.random.default_rng(args.seed + k).permutation(len(items))
        items = [items[o] for o in order]
        caps = [data.records[idx]["caption"] for _, idx in items]
        res = judge(img_path, caps)
        if res is None:
            continue
        by_lab = {"buddy": [], "sub": [], "plain": []}
        for (lab, _), v in zip(items, res):
            if v in (0, 1):
                by_lab[lab].append(v); good[lab].append(v)
        if by_lab["buddy"]:
            for neg in ("sub", "plain"):
                if by_lab[neg]:
                    per_anchor[neg].append(float(np.mean(by_lab["buddy"]))
                                           - float(np.mean(by_lab[neg])))
        if k % 10 == 0:
            print(f"  [{k}/{len(cand)}] buddy={np.mean(good['buddy'] or [0]):.3f} "
                  f"sub={np.mean(good['sub'] or [0]):.3f} "
                  f"plain={np.mean(good['plain'] or [0]):.3f}")

    summary = {
        "graph": args.graph, "model": args.model_name, "n_anchors_judged": len(cand),
        "n_buddy": len(good["buddy"]), "n_sub": len(good["sub"]), "n_plain": len(good["plain"]),
        "buddy_good_rate": float(np.mean(good["buddy"])) if good["buddy"] else None,
        "subreddit_random_good_rate": float(np.mean(good["sub"])) if good["sub"] else None,
        "plain_random_good_rate": float(np.mean(good["plain"])) if good["plain"] else None,
    }
    for neg in ("sub", "plain"):
        pa = per_anchor[neg]
        summary[f"paired_diff_vs_{neg}"] = float(np.mean(pa)) if pa else None
        summary[f"paired_diff_vs_{neg}_ci95"] = (
            [float(np.percentile(_boot(pa), 2.5)), float(np.percentile(_boot(pa), 97.5))]
            if pa else None)
    out = os.path.join(ASSETS, f"phase2_vlm_{args.graph}.json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\n", json.dumps(summary, indent=2), "\nwrote", out)


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
