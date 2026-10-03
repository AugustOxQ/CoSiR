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
    parts = [{"type": "text", "text": title + "\n"}]
    for i, (image, caption) in enumerate(pairs, 1):
        parts += [{"type": "text", "text": f"Pair {i}: image"}, {"type": "image", "image": image},
                  {"type": "text", "text": f"\ncaption: {caption}\n"}]
    return parts


def build_messages(query, candidates, supports, contrasts, direction):
    """query = (image_path, None) for i2t or (None, caption) for t2i; candidates are captions (i2t) or image paths
    (t2i); supports / contrasts are lists of (image_path, caption). The chat template joins text parts with no
    separator, so every line break is explicit."""
    content = [{"type": "text", "text": INSTRUCTION + "\n\n"}]
    content += _pair_block("Example pairs:", supports)
    content += _pair_block("Counter-example pairs:", contrasts)
    content.append({"type": "text", "text": "Query:\n"})
    if direction == "i2t":
        content += [{"type": "image", "image": query[0]}, {"type": "text", "text": "\n"}]
    else:
        content.append({"type": "text", "text": f"{query[1]}\n"})
    content.append({"type": "text", "text": "Candidates:\n"})
    for letter, cand in zip(LETTERS, candidates):
        if direction == "i2t":
            content.append({"type": "text", "text": f"{letter}. {cand}\n"})
        else:
            content += [{"type": "text", "text": f"{letter}. "}, {"type": "image", "image": cand},
                        {"type": "text", "text": "\n"}]
    content.append({"type": "text", "text": "Answer:"})
    return [{"role": "user", "content": content}]


def unpermute(letter_scores, perm):
    """Letter j showed candidate column perm[j]; return the scores in column order."""
    out = np.empty(len(perm), dtype=np.asarray(letter_scores).dtype)
    out[np.asarray(perm)] = letter_scores
    return out


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
        self.head = self.model.lm_head.weight

    @torch.no_grad()
    def score(self, messages) -> np.ndarray:
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        h = self.model.model(**inputs).last_hidden_state[0, -1]       # post-norm; no full-vocabulary logits
        return (self.head[self.letter_ids].float() @ h.float()).cpu().numpy()      # fp32: no bf16 ties

    @torch.no_grad()
    def score_bf16(self, messages) -> np.ndarray:
        """Reference path (full bf16 logits); used only to check the fp32 ranking."""
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        return self.model(**inputs).logits[0, -1][self.letter_ids].float().cpu().numpy()
