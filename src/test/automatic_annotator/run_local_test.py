#!/usr/bin/env python3
"""
Quick local test: 10 images × 50 captions = 10 API calls.
Use this to verify the vLLM server is working and tune batch_size
before launching the full 5k×5k run.

Usage:
    # With vLLM server running on port 8000:
    python run_local_test.py --image_root /data/PDD [--batch_size 50]
"""

import json
import argparse
import numpy as np
from pathlib import Path
from qwenannotator import QwenAnnotator, _encode_image_base64, _build_user_prompt, _parse_response, SYSTEM_PROMPT
from openai import OpenAI
import time


def run_test(image_root: str, annotation_path: str, batch_size: int, port: int, n_images: int = 10, model_name: str = "Qwen/Qwen3.6-35B-A3B-FP8"):
    with open(annotation_path) as f:
        data = json.load(f)
    samples = data[:max(n_images, batch_size)]  # need at least batch_size captions

    captions = [s["caption"] for s in samples]
    image_rel_paths = [s["image"] for s in samples]
    image_root_p = Path(image_root)

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    captions_batch = captions[:batch_size]

    print(f"\nTest configuration:")
    print(f"  Images to test:  {n_images}")
    print(f"  Batch size:      {batch_size} captions/call")
    print(f"  Total API calls: {n_images}")
    print(f"  vLLM port:       {port}\n")

    results_table = np.full((n_images, batch_size), -1, dtype=np.int8)
    timings = []

    for i in range(n_images):
        img_path = image_root_p / image_rel_paths[i]
        if not img_path.exists():
            print(f"[{i}] Image not found: {img_path} — skipping")
            continue

        print(f"[{i}] {image_rel_paths[i]}")
        t0 = time.time()

        data_url = _encode_image_base64(str(img_path))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _build_user_prompt(captions_batch)},
                ],
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            elapsed = time.time() - t0
            timings.append(elapsed)
            content = resp.choices[0].message.content or ""
            results = _parse_response(content, batch_size)

            if results is not None:
                results_table[i, :] = results
                good = sum(results)
                print(f"  OK in {elapsed:.1f}s | good pairs: {good}/{batch_size} | gt pair: {'GOOD' if results[i] == 1 else 'BAD'}")
            else:
                print(f"  PARSE FAILED in {elapsed:.1f}s | raw: {content[:300]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n--- Test Summary ---")
    print(f"Completed {len(timings)}/{n_images} calls")
    if timings:
        avg = sum(timings) / len(timings)
        print(f"Avg time per call: {avg:.1f}s")
        calls_per_5k = 5000 * (5000 // batch_size + 1)
        est_hours = (calls_per_5k * avg) / 3600
        print(f"Estimated full 5k×5k runtime: {est_hours:.0f}h ({est_hours/24:.1f} days)")

    # Show a small slice of the result
    print(f"\nAnnotation matrix (first {n_images} images × first {batch_size} captions):")
    print(f"  Diagonal (GT pairs): {[results_table[i, i] for i in range(min(n_images, batch_size))]}")
    print(f"  GT correct: {sum(results_table[i,i] == 1 for i in range(min(n_images, batch_size)))}/{min(n_images, batch_size)}")

    out = Path("test_output")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "test_matrix.npz",
        matrix=results_table,
        captions=np.array(captions[:batch_size]),
        image_rel_paths=np.array(image_rel_paths[:n_images]),
    )
    print(f"\nSaved test matrix to {out / 'test_matrix.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path",
        default="/data/PDD/redcaps/redcaps_plus/redcaps_test.json",
    )
    parser.add_argument("--image_root", default="/data/PDD")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument("--model_name", default="Qwen/Qwen3.6-35B-A3B-FP8")
    args = parser.parse_args()

    run_test(
        image_root=args.image_root,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size,
        port=args.port,
        n_images=args.n_images,
        model_name=args.model_name,
    )
