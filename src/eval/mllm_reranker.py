"""In-context MLLM reranker for aspect episodes (CVPR plan spec §6 early probe, §8 tier 2). The model sees the
example and counter-example pairs, the query and 13 lettered candidates; its next-token logits over the letters
are the scores. The aspect is never named."""

import numpy as np
import torch

LETTERS = "ABCDEFGHIJKLM"
INSTRUCTION = ("Each example pair shows an image and a caption of two different artworks that are alike in one "
               "respect. The counter-example pairs are alike in a different respect. Pick the candidate that is "
               "alike to the query in the same respect as the example pairs. Answer with one letter.")


def _pair_block(title, pairs):
    parts = [{"type": "text", "text": title}]
    for i, (image, caption) in enumerate(pairs, 1):
        parts += [{"type": "text", "text": f"Pair {i}: image"}, {"type": "image", "image": image},
                  {"type": "text", "text": f"caption: {caption}"}]
    return parts


def build_messages(query, candidates, supports, contrasts, direction):
    """query = (image_path, None) for i2t or (None, caption) for t2i; candidates are captions (i2t) or image paths
    (t2i); supports / contrasts are lists of (image_path, caption)."""
    content = [{"type": "text", "text": INSTRUCTION}]
    content += _pair_block("Example pairs:", supports)
    content += _pair_block("Counter-example pairs:", contrasts)
    content.append({"type": "text", "text": "Query:"})
    content.append({"type": "image", "image": query[0]} if direction == "i2t"
                   else {"type": "text", "text": query[1]})
    content.append({"type": "text", "text": "Candidates:"})
    for letter, cand in zip(LETTERS, candidates):
        if direction == "i2t":
            content.append({"type": "text", "text": f"{letter}. {cand}"})
        else:
            content += [{"type": "text", "text": f"{letter}."}, {"type": "image", "image": cand}]
    content.append({"type": "text", "text": "Answer:"})
    return [{"role": "user", "content": content}]


class QwenReranker:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda",
                 max_pixels: int = 256 * 28 * 28):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.device = device

    @torch.no_grad()
    def score(self, messages) -> np.ndarray:
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1]
        return logits[self.letter_ids].float().cpu().numpy()
