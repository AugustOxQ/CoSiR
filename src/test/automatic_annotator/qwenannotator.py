#!/usr/bin/env python3
"""
Automatic annotation of image-text pairs using Qwen3.6-35B-A3B-FP8 via vLLM.

Produces an N×N int8 matrix where matrix[i, j] indicates whether
(image_i, caption_j) is a good pair: 1=good, 0=bad, -1=unannotated.

Usage:
    # Start vLLM server first (see launch_vllm.sh), then:
    python qwenannotator.py \
        --annotation_path /data/PDD/redcaps/redcaps_plus/redcaps_test.json \
        --image_root /data/PDD \
        --output_path /data/SSD2/annotations/redcaps_5k5k \
        --n_samples 5000 \
        --batch_size 50
"""

import json
import base64
import time
import argparse
import re
import numpy as np
from pathlib import Path
from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an image-caption alignment annotator for a large-scale web image dataset (Reddit).

TASK: Given ONE image and a numbered list of captions, decide for each caption whether it is a GOOD match for the image.

GOOD (1) — Use this when the caption meaningfully relates to the image by ANY of:
  • Literal: describes visible objects, people, actions, scenes, colors, or composition
  • Partial: describes a salient part of the image, even if not the whole picture
  • Abstract / emotional: metaphorical or mood-based, but the connection to the image is recognizable
  • Contextual: references the place, event, occasion, or activity clearly depicted
  • Web-style: hashtags, informal slang, abbreviations, or shorthand that map to image content
  • Food / product / brand names that match what is visually present

BAD (0) — Use this ONLY when the caption has NO plausible connection to anything visible in the image.

Key guidelines:
  - Web captions are often terse, informal, or abstract. Be inclusive — lean toward GOOD (1) when genuinely uncertain.
  - A caption describing only part of the image is still GOOD.
  - A caption that is somewhat related but imprecise can still be GOOD.
  - Reserve BAD (0) strictly for captions with zero relationship to the image.

OUTPUT FORMAT — respond ONLY with a single JSON object, nothing else:
{"results": [0, 1, 0, ...]}
Each integer corresponds to the caption at that 0-indexed position. No explanation, no reasoning, no extra text."""


def _build_user_prompt(captions: list[str]) -> str:
    lines = [f"Evaluate these {len(captions)} captions against the image:\n"]
    for i, cap in enumerate(captions):
        lines.append(f"{i + 1}. {cap}")
    lines.append(
        f'\nRespond with JSON only: {{"results": [...]}} containing exactly {len(captions)} values (0 or 1).'
    )
    return "\n".join(lines)


def _encode_image_base64(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
    mime = mime_map.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{data}"


def _parse_response(content: str, expected_n: int) -> list[int] | None:
    """Parse model response into a list of 0/1 values. Returns None on failure.

    Accepts responses with >= expected_n values and truncates to expected_n,
    since smaller models sometimes output a few extra values.
    """
    def _extract(vals_raw) -> list[int] | None:
        try:
            vals = [int(v) for v in vals_raw]
            if len(vals) >= expected_n:
                return vals[:expected_n]  # truncate if model over-generates
        except (ValueError, TypeError):
            pass
        return None

    # Attempt 1: direct JSON parse
    try:
        obj = json.loads(content.strip())
        result = _extract(obj["results"])
        if result is not None:
            return result
    except Exception:
        pass

    # Attempt 2: regex extraction (model may add surrounding text or truncated JSON)
    try:
        m = re.search(r'"results"\s*:\s*\[([^\]]*)', content)
        if m:
            # Works even if the closing ] is missing (truncated response)
            result = _extract(
                x.strip() for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()
            )
            if result is not None:
                return result
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class QwenAnnotator:
    def __init__(
        self,
        annotation_path: str,
        image_root: str,
        output_path: str,
        n_samples: int = 5000,
        batch_size: int = 50,
        port: int = 8000,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.annotation_path = annotation_path
        self.image_root = Path(image_root)
        self.output_path = Path(output_path)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = model_name

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.matrix_path = self.output_path / "annotation_matrix.npz"
        self.checkpoint_path = self.output_path / "checkpoint.json"
        self.failed_log_path = self.output_path / "failed_batches.jsonl"

        self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

        self._load_data()
        self._load_or_init_state()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_data(self):
        with open(self.annotation_path, "r") as f:
            data = json.load(f)
        self.samples = data[: self.n_samples]
        self.image_ids = [s["image_id"] for s in self.samples]
        self.captions = [s["caption"] for s in self.samples]
        self.image_rel_paths = [s["image"] for s in self.samples]
        print(f"Loaded {len(self.samples)} samples from {self.annotation_path}")

    def _load_or_init_state(self):
        if self.checkpoint_path.exists() and self.matrix_path.exists():
            with open(self.checkpoint_path) as f:
                ckpt = json.load(f)
            self.completed_rows: set[int] = set(ckpt["completed_rows"])
            loaded = np.load(self.matrix_path, allow_pickle=True)
            self.matrix = loaded["matrix"]
            print(f"Resumed checkpoint: {len(self.completed_rows)}/{self.n_samples} rows done")
        else:
            self.completed_rows = set()
            # -1 = unannotated, 0 = bad pair, 1 = good pair
            self.matrix = np.full(
                (self.n_samples, self.n_samples), -1, dtype=np.int8
            )
            print("Starting fresh annotation run")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        np.savez_compressed(
            self.matrix_path,
            matrix=self.matrix,
            image_ids=np.array(self.image_ids),
            captions=np.array(self.captions),
            image_rel_paths=np.array(self.image_rel_paths),
        )
        with open(self.checkpoint_path, "w") as f:
            json.dump({"completed_rows": sorted(self.completed_rows)}, f)

    def _log_failed_batch(self, img_idx: int, batch_start: int, raw_response: str):
        with open(self.failed_log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "img_idx": img_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_start + self.batch_size,
                        "raw": raw_response[:500],
                    }
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _annotate_batch(
        self, data_url: str, captions_batch: list[str], img_idx: int, batch_start: int
    ) -> list[int]:
        """Send one image + one caption batch to the model. Returns list of 0/1."""
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

        last_raw = ""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.8,
                    extra_body={"top_k": 20},
                )
                last_raw = resp.choices[0].message.content or ""
                results = _parse_response(last_raw, len(captions_batch))
                if results is not None:
                    return results
                print(
                    f"    [img {img_idx}, batch {batch_start}] "
                    f"Parse failed (attempt {attempt + 1}), raw: {last_raw[:200]}"
                )
            except Exception as e:
                print(
                    f"    [img {img_idx}, batch {batch_start}] "
                    f"API error (attempt {attempt + 1}): {e}"
                )
            time.sleep(self.retry_delay)

        self._log_failed_batch(img_idx, batch_start, last_raw)
        print(f"    [img {img_idx}, batch {batch_start}] All retries failed — leaving as -1")
        return [-1] * len(captions_batch)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        n = self.n_samples
        n_batches_per_image = (n + self.batch_size - 1) // self.batch_size
        total_images = n - len(self.completed_rows)
        t0 = time.time()

        print(
            f"\nAnnotation plan: {n} images × {n} captions "
            f"({n_batches_per_image} batches of {self.batch_size} per image)\n"
            f"Remaining: {total_images} images\n"
        )

        for img_idx in range(n):
            if img_idx in self.completed_rows:
                continue

            img_rel = self.image_rel_paths[img_idx]
            img_path = self.image_root / img_rel

            if not img_path.exists():
                print(f"[{img_idx}/{n}] WARNING: image not found: {img_path} — skipping row")
                self.matrix[img_idx, :] = -1
                self.completed_rows.add(img_idx)
                self._save_checkpoint()
                continue

            print(f"[{img_idx}/{n}] {img_rel}")

            # Encode image once per row (reused across all caption batches)
            data_url = _encode_image_base64(str(img_path))

            # Process all caption batches for this image
            for batch_start in range(0, n, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n)
                captions_batch = self.captions[batch_start:batch_end]

                results = self._annotate_batch(data_url, captions_batch, img_idx, batch_start)

                for j, val in enumerate(results):
                    self.matrix[img_idx, batch_start + j] = val

            self.completed_rows.add(img_idx)
            self._save_checkpoint()

            # Progress + ETA
            done = len(self.completed_rows)
            elapsed = time.time() - t0
            rate = done / elapsed  # images/sec
            remaining = n - done
            eta_h = (remaining / rate) / 3600 if rate > 0 else float("inf")
            good_count = int((self.matrix[img_idx] == 1).sum())
            print(
                f"  → row done | good pairs: {good_count}/{n} "
                f"| progress: {done}/{n} images "
                f"| ETA: {eta_h:.1f}h"
            )

        print(f"\nAnnotation complete. Matrix saved to {self.matrix_path}")
        self._print_summary()

    def _print_summary(self):
        total = self.n_samples ** 2
        good = int((self.matrix == 1).sum())
        bad = int((self.matrix == 0).sum())
        unannotated = int((self.matrix == -1).sum())
        print(
            f"\nSummary:\n"
            f"  Total pairs:    {total:,}\n"
            f"  Good (1):       {good:,}  ({100*good/total:.1f}%)\n"
            f"  Bad (0):        {bad:,}  ({100*bad/total:.1f}%)\n"
            f"  Unannotated:    {unannotated:,}\n"
            f"  Diagonal (GT):  {int((self.matrix.diagonal() == 1).sum())}/{ self.n_samples} correct GT pairs\n"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate image-text pairs with Qwen3.6-35B VLM"
    )
    parser.add_argument("--annotation_path", required=True, help="Path to redcaps_test.json")
    parser.add_argument("--image_root", required=True, help="Root directory for images (e.g. /data/PDD)")
    parser.add_argument("--output_path", required=True, help="Directory to save matrix + checkpoints")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to use (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=50, help="Captions per API call (default: 50)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    args = parser.parse_args()

    annotator = QwenAnnotator(
        annotation_path=args.annotation_path,
        image_root=args.image_root,
        output_path=args.output_path,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        port=args.port,
        model_name=args.model_name,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    annotator.run()


if __name__ == "__main__":
    main()
