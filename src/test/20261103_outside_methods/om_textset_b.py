"""CLAY text set B (spec D5, section 3.2): LLM-written value lists, 100 per aspect, CLIP L/14 text vectors.

  python om_textset_b.py [dry|lists|texts|all]   default all; lists needs the GPU (Qwen3-VL-8B-Instruct, greedy)

Same recipe as r11's CRL lists (rt_crl.py, imported, not edited) with three changes: the prompt asks for 100 values,
max_new_tokens is 2048 (r11's 1024 cut the style reply), and the parser keeps 100. Output in om_paths.CLAY_B_DIR:
lists.json, <aspect>_text.npy (float32 unit rows [n, 768]), <aspect>_text.json. An aspect with fewer than 2 values or a
failed generation gets `fallback: true` and no .npy (ticket 08 then uses set A with no pick). Frozen once generated
(spec 3.7): refuses to overwrite unless OM_SMOKE.
"""
import argparse
import json
import re

import numpy as np

import om_paths as P
import rt_crl                               # r11's CRL code: constants and CLIP settings, read-only

N_VALUES = P.CLAY_B_N                       # 100
PROMPT = rt_crl.PROMPT.replace('50', '{n}')
assert PROMPT.format(n=50, aspect='{aspect}') == rt_crl.PROMPT, 'r11 prompt changed'
GENERATION = dict(do_sample=False, max_new_tokens=2048)
MIN_VALUES = 2
DIM = rt_crl.DIM


def prompt(aspect):
    return PROMPT.format(n=N_VALUES, aspect=aspect)


def parse(raw, keep=N_VALUES, capped=False):
    """r11's parse (strip numbering and bullets, lowercase, trim), repeats dropped, first `keep` kept.
    With `capped` (generation hit max_new_tokens) the last line is dropped: it may be cut mid-word.
    Returns (values, n_repeats_dropped)."""
    lines = raw.splitlines()
    if capped and lines:
        lines = lines[:-1]
    values, dropped = [], 0
    for line in lines:
        v = re.sub(r'^\s*(?:[-*•]+|\d+\s*[.)]|\d+\s*:)\s*', '', line).strip().lower()
        if not v:
            continue
        if v in values:
            dropped += 1
        else:
            values.append(v)
    return values[:keep], dropped


def entry(raw, capped, error=None):
    values, dropped = parse(raw, capped=capped)
    fb = error is not None or len(values) < MIN_VALUES
    return dict(raw=raw, values=values, count=len(values), dedup_dropped=dropped, capped=bool(capped),
                short=len(values) < N_VALUES, error=error, fallback=fb,
                rank=None if fb else P.clay_rank(len(values)))


def check_free():
    if P.SMOKE:
        return
    d = P.CLAY_B_DIR
    assert not (d / 'lists.json').exists() and not list(d.glob('*_text.*')), \
        f'{d} holds a set B already: frozen once generated (spec 3.7)'


def run_lists():
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    check_free()
    assert torch.cuda.is_available(), 'lists needs the GPU'
    assert P.CLAY_B_MODEL == rt_crl.LLM_ID
    processor = rt_crl.build_processor()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        rt_crl.LLM_ID, revision=rt_crl.LLM_REVISION, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    doc = dict(model=rt_crl.LLM_ID, revision=rt_crl.LLM_REVISION, prompt=PROMPT, n_values=N_VALUES,
               generation=GENERATION, aspects={})
    for aspect in P.ASPECTS:
        try:
            conv = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt(aspect)}]}]
            text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            assert prompt(aspect) in text
            enc = processor.tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
            with torch.no_grad():
                out = model.generate(**enc, **GENERATION)
            new = out[0, enc['input_ids'].shape[1]:]
            capped = len(new) >= GENERATION['max_new_tokens']
            raw = processor.tokenizer.decode(new, skip_special_tokens=True)
            doc['aspects'][aspect] = entry(raw, capped)
        except Exception as e:                          # D5: a failed generation is a fallback, not a crash
            doc['aspects'][aspect] = entry('', False, error=repr(e))
    path = P.CLAY_B_DIR / 'lists.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    return doc


def run_texts():
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = P.CLAY_B_DIR / 'lists.json'
    doc = json.loads(path.read_text())
    assert set(doc['aspects']) == set(P.ASPECTS)
    if not P.SMOKE:
        assert not list(P.CLAY_B_DIR.glob('*_text.*')), 'text vectors exist already (frozen)'
    lists_sha = rt_crl.file_sha(path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(rt_crl.CLIP_ID, revision=rt_crl.CLIP_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(rt_crl.CLIP_ID, revision=rt_crl.CLIP_REVISION, local_files_only=True,
                                      dtype=torch.float32).to(device).eval()
    assert model.config.projection_dim == DIM
    for aspect in P.ASPECTS:
        a = doc['aspects'][aspect]
        assert a['count'] == len(a['values']) == len(set(a['values']))
        assert a['fallback'] == (a['error'] is not None or a['count'] < MIN_VALUES)
        if a['fallback']:
            continue
        assert a['rank'] == P.clay_rank(a['count'])
        texts = [P.CLAY_B_PROMPT.format(value=v) for v in a['values']]
        t = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            f = model.get_text_features(**t)
        f = getattr(f, 'pooler_output', f).float()
        x = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
        assert x.shape == (len(texts), DIM) and np.isfinite(x).all()
        assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
        base = P.CLAY_B_DIR / f'{aspect}_text'
        np.save(base.with_suffix('.npy'), x)
        side = dict(model=rt_crl.CLIP_ID, revision=rt_crl.CLIP_REVISION, template=P.CLAY_B_PROMPT, texts=texts,
                    shape=list(x.shape), dtype='float32', sha256=rt_crl.array_sha(x), lists_sha256=lists_sha,
                    device=device)
        base.with_suffix('.json').write_text(json.dumps(side, indent=1) + '\n')


def dry_run():
    """CPU check of the parser on r11's saved replies: keep=50 must reproduce r11's lists."""
    doc = json.loads((P.R11_OUT / 'crl' / 'lists.json').read_text())
    for aspect, a in doc['aspects'].items():
        values, _ = parse(a['raw'], keep=rt_crl.KEEP)
        assert values == a['values'], aspect
        v100, d100 = parse(a['raw'])
        print(f'{aspect}: keep 50 matches r11 ({len(values)}); keep 100 gives {len(v100)} values, {d100} repeats dropped')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', nargs='?', default='all', choices=('lists', 'texts', 'all', 'dry'))
    s = ap.parse_args().step
    if s == 'dry':
        dry_run()
    if s in ('lists', 'all'):
        run_lists()
    if s in ('texts', 'all'):
        run_texts()
