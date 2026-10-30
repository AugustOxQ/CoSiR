"""CRL texts and vectors for r12 (spec 2026-10-30 §3.2 CRL rows, §3.5 0a(vii), D5).

  python ac_crl.py texts [--device cuda]   write OUTPUTS/crl/{encoder}_{wording}_{aspect}.npy and .json (6 arrays)
  python ac_crl.py check [--device cuda]   gate 0a(vii); writes OUTPUTS/stage0a/crl.json
Values are r11's frozen lists (never regenerated). `vectors` is r11's CRL vector form (rt_data.raw / embed).
CLIP painting texts are r11's own arrays, copied after a check against r11's sidecar (so gate (a) compares like with
like, bit for bit); `check` also re-encodes them and reports the cosine. Qwen texts go through the same encoder class
and default instruction as scripts/extract_features.py's text pass for the Affection store.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import ac_paths
import rt_crl
import rs_vh

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = 'a {wording} with the {{aspect}} of {{value}}'
WORDINGS = ('painting', 'photo')
ENCODERS = ('clip', 'qwen')
COS_MIN = 0.999
N_STORE_ROWS = 200
# (encoder, wording, aspect) written by `texts`: Affection emotion x 4 versions, ArtELingo Qwen painting x style, genre
# (ArtELingo Qwen emotion is the same file as Affection's qwen_painting_emotion: same values, same wording).
JOBS = ([(e, w, 'emotion') for e in ENCODERS for w in WORDINGS]
        + [('qwen', 'painting', a) for a in ('style', 'genre')])


def template(wording):
    assert wording in WORDINGS, wording
    return TEMPLATE.format(wording=wording)


def render(wording, aspect, values):
    """The value texts, in the frozen list order."""
    t = template(wording)
    return [t.format(aspect=aspect, value=v) for v in values]


def vectors(img_unit, text_unit):
    """r11's CRL form: float64 product of unit rows, L2-normalize in float64, one cast to float32."""
    img, text = np.asarray(img_unit), np.asarray(text_unit)
    assert img.ndim == 2 and text.ndim == 2 and img.shape[1] == text.shape[1], (img.shape, text.shape)
    x = img.astype(np.float64) @ text.astype(np.float64).T
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x.astype(np.float32)


def frozen_values(aspect):
    doc = json.loads(Path(ac_paths.R11_CRL_LISTS_FILE).read_text())
    entry = doc['aspects'][aspect]
    assert entry['count'] == len(entry['values']) > 0
    return entry['values']


def _src_path():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def encode_clip(texts, device='cpu'):
    """r11's CLIP L/14 text path (rt_crl.run_texts), float32, unit rows."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rt_crl.CLIP_ID, revision=rt_crl.CLIP_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(rt_crl.CLIP_ID, revision=rt_crl.CLIP_REVISION, local_files_only=True,
                                      dtype=torch.float32).to(device).eval()
    assert model.config.projection_dim == rt_crl.DIM
    t = tok(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        f = model.get_text_features(**t)
    f = getattr(f, 'pooler_output', f).float()
    x = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
    assert x.shape == (len(texts), rt_crl.DIM) and np.isfinite(x).all()
    return x


def load_qwen(device='cuda'):
    """The Affection store's text encoder: Qwen3VLEmb with the default instruction, at the pinned revision."""
    _src_path()
    from src.data.feature_extract import QWEN_INSTRUCTION_DEFAULT, Qwen3VLEmb
    import ct2_data
    enc = Qwen3VLEmb(device, QWEN_INSTRUCTION_DEFAULT)
    assert enc.m.config._commit_hash == ct2_data.QWEN_REVISION, 'Qwen revision is not the pinned one'
    return enc


def encode_qwen(enc, texts):
    """Same call as extract_features' text pass (encode_texts; batch 128, longest texts last)."""
    x = enc.encode_texts(list(texts), 128)
    assert x.shape == (len(texts), 2048) and x.dtype == np.float32 and np.isfinite(x).all()
    return x


def _write(base, x, side, overwrite):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    assert overwrite or not base.with_suffix('.npy').exists(), f'{base}.npy exists (use --overwrite)'
    assert x.dtype == np.float32 and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-3)
    np.save(base.with_suffix('.npy'), x)
    side = dict(side, shape=list(x.shape), dtype='float32', sha256=rs_vh.array_sha(x),
                lists_sha256=rs_vh.file_sha(ac_paths.R11_CRL_LISTS_FILE))
    base.with_suffix('.json').write_text(json.dumps(side, indent=1) + '\n')


def write_texts(device='cuda', overwrite=False, outputs=None):
    _src_path()
    from src.data.feature_extract import QWEN_INSTRUCTION_DEFAULT, QWEN_REPO
    import ct2_data
    out = Path(outputs or ac_paths.OUTPUTS) / 'crl'
    qwen = None
    for encoder, wording, aspect in JOBS:
        texts = render(wording, aspect, frozen_values(aspect))
        base = out / f'{encoder}_{wording}_{aspect}'
        if encoder == 'clip':
            side = dict(texts=texts, encoder='clip', model_id=rt_crl.CLIP_ID, model_revision=rt_crl.CLIP_REVISION,
                        instruction=None, template=template(wording))
            if wording == 'painting':
                # r11's own array, byte for byte, after a check against r11's sidecar.
                x = rt_crl.text_array(aspect, ac_paths.R11_OUT)
                r11_side = json.loads((ac_paths.R11_OUT / f'crl/{aspect}_text.json').read_text())
                assert r11_side['texts'] == texts and r11_side['lists_sha256'] == rs_vh.file_sha(
                    ac_paths.R11_CRL_LISTS_FILE)
                side['source'] = f'copied from r11 outputs/crl/{aspect}_text.npy (sha256 checked)'
            else:
                x = encode_clip(texts, device)
        else:
            qwen = qwen or load_qwen(device)
            x = encode_qwen(qwen, texts)
            side = dict(texts=texts, encoder='qwen', model_id=QWEN_REPO, model_revision=ct2_data.QWEN_REVISION,
                        instruction=QWEN_INSTRUCTION_DEFAULT, template=template(wording))
        _write(base, x, side, overwrite)
        print('wrote', base, x.shape, flush=True)


def check(device='cuda', outputs=None, n_rows=N_STORE_ROWS):
    """Gate 0a(vii). (a) CLIP inputs: vectors == r11's validation CRL vectors (sha256), per aspect; the CLIP
    painting texts equal r11's (sha) and a fresh encoding agrees (cosine, reported). (b) Qwen text path against the
    first n_rows rows of the Affection store's txt.npy, cosine >= 0.999."""
    import pandas as pd
    import rt_data
    out = Path(outputs or ac_paths.OUTPUTS)
    items = {}
    for aspect in ac_paths.ASPECTS:
        r11_text = rt_crl.text_array(aspect, ac_paths.R11_OUT)
        ref = rs_vh.array_sha(rt_data.embed('crl', 'val', aspect))
        got = rs_vh.array_sha(vectors(rt_data.embed('clip', 'val', aspect), r11_text))
        items[f'a_vectors_sha_{aspect}'] = dict(value=got, reference=ref, ok=got == ref)
    ours = np.load(out / 'crl/clip_painting_emotion.npy', allow_pickle=False)
    r11 = rt_crl.text_array('emotion', ac_paths.R11_OUT)
    items['a_clip_painting_text_sha_emotion'] = dict(
        value=rs_vh.array_sha(ours), reference=rs_vh.array_sha(r11), ok=rs_vh.array_sha(ours) == rs_vh.array_sha(r11))
    fresh = encode_clip(json.loads((out / 'crl/clip_painting_emotion.json').read_text())['texts'], device)
    cos = float((fresh * r11).sum(1).min())
    items['a_clip_painting_reencode_min_cos_emotion'] = dict(value=cos, reference=0.9999, ok=cos >= 0.9999)
    # (b) Qwen text path against the store (texts = the CSV utterance of each text row's csv_row)
    store = ac_paths.STORE
    index = json.loads((store / 'index.json').read_text())
    rows = [t['csv_row'] for t in index['texts'][:n_rows]]
    csv = pd.read_csv(ac_paths.ANNOTATIONS, usecols=['utterance'], dtype=str)
    texts = [csv.utterance[r] for r in rows]
    assert len(texts) == n_rows and all(isinstance(t, str) and t.strip() for t in texts)
    stored = np.asarray(np.load(store / 'txt.npy', mmap_mode='r')[:n_rows]).astype(np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    cos = (encode_qwen(load_qwen(device), texts) * stored).sum(1)
    items['b_qwen_text_min_cos'] = dict(value=float(cos.min()), reference=COS_MIN, ok=bool(cos.min() >= COS_MIN))
    items['b_qwen_text_mean_cos'] = dict(value=float(cos.mean()), reference=COS_MIN, ok=bool(cos.mean() >= COS_MIN))
    doc = dict(items=items, ok=all(v['ok'] for v in items.values()), n_rows=n_rows)
    p = out / 'stage0a/crl.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + '\n')
    for k, v in items.items():
        print(k, v['value'] if isinstance(v['value'], float) else v['value'][:12], v['ok'])
    print('ok:', doc['ok'])
    return doc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('texts', 'check'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.step == 'texts':
        write_texts(a.device, a.overwrite)
    else:
        raise SystemExit(0 if check(a.device)['ok'] else 1)
