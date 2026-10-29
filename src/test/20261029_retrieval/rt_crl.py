"""CRL arm (our re-implementation, spec D12): LLM-written value lists, CLIP L/14 text embeddings.

  python rt_crl.py lists [--outputs DIR]   GPU; Qwen3-VL-8B-Instruct, greedy, one text-only call per aspect
  python rt_crl.py texts [--outputs DIR]   CLIP L/14 text features of "a painting with the {aspect} of {value}"
`rt_data.raw('crl', ...)` reads the text arrays through `text_array`.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

import rt_paths

LLM_ID, LLM_REVISION = 'Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
CLIP_ID, CLIP_REVISION = 'openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41'
PROMPT = ('List 50 distinct values of the following criterion for describing paintings: {aspect}. '
          'One value per line, no numbering.')
TEMPLATE = 'a painting with the {aspect} of {value}'
GENERATION = dict(do_sample=False, max_new_tokens=1024)
KEEP, MIN_KEEP, DIM = 50, 40, 768


def parse(text):
    """Strip numbering and bullets, lowercase, drop empty lines and repeats, keep the first 50.
    Returns (values, note); note is '' unless fewer than 40 values are left."""
    values = []
    for line in text.splitlines():
        v = re.sub(r'^\s*(?:[-*•]+|\d+\s*[.)]|\d+\s*:)\s*', '', line)
        v = v.strip().lower()
        if v and v not in values:
            values.append(v)
    values = values[:KEEP]
    note = f'only {len(values)} values parsed (fewer than {MIN_KEEP}); used as they are' if len(values) < MIN_KEEP else ''
    return values, note


def fold(name):
    return name.lower().replace('_', ' ').strip()


def label_names():
    """Label names per aspect, from the sources rt_data.labels uses."""
    import rs_vh
    root = str(rt_paths.TEST.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.data.wikiart_genre import GENRE_NAMES
    rows = json.loads(Path(rs_vh.ANNOTATIONS).read_text())
    styles = sorted({r['art_style'] for r in rows})
    assert len(rs_vh.EMOTIONS) == 9 and len(styles) == 27 and len(GENRE_NAMES) == 10
    return dict(emotion=list(rs_vh.EMOTIONS), style=styles, genre=list(GENRE_NAMES))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prompt(aspect):
    return PROMPT.format(aspect=aspect)


def build_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(LLM_ID, revision=LLM_REVISION, local_files_only=True)


def render(processor, aspect):
    conv = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt(aspect)}]}]
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    assert isinstance(text, str) and prompt(aspect) in text
    return text


def run_lists(outputs):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    path = Path(outputs) / 'crl' / 'lists.json'
    assert torch.cuda.is_available(), 'lists needs the GPU'
    names = label_names()
    processor = build_processor()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LLM_ID, revision=LLM_REVISION, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    doc = dict(model=LLM_ID, revision=LLM_REVISION, prompt=PROMPT, generation=GENERATION, aspects={})
    for aspect in rt_paths.ASPECTS:
        enc = processor.tokenizer(render(processor, aspect), return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.no_grad():
            out = model.generate(**enc, **GENERATION)
        raw = processor.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        values, note = parse(raw)
        labels = {fold(n) for n in names[aspect]}
        doc['aspects'][aspect] = dict(raw=raw, values=values, count=len(values), note=note,
                                      label_name_matches=[v for v in values if fold(v) in labels])
    write_lists(path, doc)


def write_lists(path, doc):
    path = Path(path)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + '\n'
    if path.exists():
        assert path.read_text() == text, f'{path} exists with different content'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_lists(outputs):
    path = Path(outputs) / 'crl' / 'lists.json'
    doc = json.loads(path.read_text())
    assert set(doc['aspects']) == set(rt_paths.ASPECTS)
    for a in doc['aspects'].values():
        assert a['count'] == len(a['values']) > 0 and len(set(a['values'])) == len(a['values'])
    return doc, file_sha(path)


def run_texts(outputs):
    import torch
    from transformers import AutoModel, AutoTokenizer
    doc, lists_sha = load_lists(outputs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(CLIP_ID, revision=CLIP_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION, local_files_only=True,
                                      dtype=torch.float32).to(device).eval()
    assert model.config.projection_dim == DIM
    for aspect in rt_paths.ASPECTS:
        texts = [TEMPLATE.format(aspect=aspect, value=v) for v in doc['aspects'][aspect]['values']]
        t = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            f = model.get_text_features(**t)
        f = getattr(f, 'pooler_output', f).float()
        x = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
        assert x.shape == (len(texts), DIM) and np.isfinite(x).all()
        base = Path(outputs) / 'crl' / f'{aspect}_text'
        np.save(base.with_suffix('.npy'), x)
        side = dict(model=CLIP_ID, revision=CLIP_REVISION, template=TEMPLATE, texts=texts, shape=list(x.shape),
                    dtype='float32', sha256=array_sha(x), lists_sha256=lists_sha, device=device)
        base.with_suffix('.json').write_text(json.dumps(side, indent=1) + '\n')


def text_array(aspect, outputs=None):
    """The checked float32 [K, 768] text array of one aspect."""
    base = Path(outputs or rt_paths.OUTPUTS) / 'crl' / f'{aspect}_text'
    side = json.loads(base.with_suffix('.json').read_text())
    x = np.load(base.with_suffix('.npy'), allow_pickle=False)
    assert x.ndim == 2 and x.shape[1] == DIM and x.dtype == np.float32 and np.isfinite(x).all(), (x.shape, x.dtype)
    assert side['shape'] == list(x.shape) and side['dtype'] == 'float32' and len(side['texts']) == x.shape[0]
    assert side['sha256'] == array_sha(x), 'CRL text array hash'
    assert side['model'] == CLIP_ID and side['revision'] == CLIP_REVISION, 'CRL text model'
    return x


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('lists', 'texts'))
    ap.add_argument('--outputs', default=str(rt_paths.OUTPUTS))
    a = ap.parse_args()
    {'lists': run_lists, 'texts': run_texts}[a.step](a.outputs)
