"""Local viewer for Qwen3-VL-Embedding attention maps. Usage:
  PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python att_viewer.py --root <folder> --port 8765
Binds 127.0.0.1 only. No GPU. Reads p{k:04d}.npz files written by att_extract.py (format: att_core.py).
"""
import argparse, io, json, sys, threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAV = ROOT / 'src/test/20261017_prompted_aspect_views'
APC = ROOT / 'src/test/20261017_aspect_premise_checks'
for p in (str(ROOT), str(PAV), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import att_core as C

NAMES_JSON = HERE / 'att_label_names.json'
WORD_BAND = 'layers 16-27'


# ---------------------------------------------------------------- shared context (sample, texts, tokenizer, names)
def build_label_names(s, root=None):
    """Emotion, style, genre names from the sample's own encoders; asserts the codes agree with the painting files."""
    from apc_data import load_sample
    from src.data.wikiart_genre import GENRE_NAMES
    s3 = load_sample(min_caps=3)
    names = dict(emo=[str(x) for x in s3['emo_names']], sty=[str(x) for x in s3['sty_names']],
                 gen=list(GENRE_NAMES))
    assert np.array_equal(s3['sample_ids'], s['sample_ids'])
    for key in ('emo', 'sty', 'gen'):
        assert np.array_equal(s3[key], s[key]), key
    json.dump(names, open(NAMES_JSON, 'w'), indent=1)
    return names


class Ctx:
    def __init__(self, root):
        import pav_sample
        pav_sample.SAMPLE = PAV / 'sample.npz'
        self.s = pav_sample.load_frozen()
        from scripts.extract_features import adapter_artelingo_full
        self.spec = adapter_artelingo_full(sample_ids=self.s['sample_ids'], expected=None)
        if NAMES_JSON.exists():
            self.names = json.load(open(NAMES_JSON))
        else:
            self.names = build_label_names(self.s)
        from pav_prompts import PROMPTS
        self.prompts = {v: PROMPTS[v] for v in C.VIEWS}
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-Embedding-2B')
        self.tokcache = {}
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.imgcache = OrderedDict()
        self.root = Path(root)

    def tok_text(self, i):
        i = int(i)
        if i not in self.tokcache:
            self.tokcache[i] = self.tok.decode([i])
        return self.tokcache[i]

    def painting(self, k):
        with self.lock:
            if k in self.cache:
                self.cache.move_to_end(k)
                return self.cache[k]
        P = C.load_painting(self.root, k)
        pm = P['painting']
        rows = pm['cap_rows']
        for key in ('emo', 'sty', 'gen'):
            code = np.asarray(pm['cap_emo'] if key == 'emo' else pm[key])
            assert np.array_equal(code, self.s[key][rows]), (k, key)
        with self.lock:
            self.cache[k] = P
            while len(self.cache) > 12:
                self.cache.popitem(last=False)
        return P

    def image_jpeg(self, k):
        if k in self.imgcache:
            return self.imgcache[k]
        from src.data.feature_extract import _qwen_prepare_image, QWEN_MAX_PIXELS
        pm = self.painting(k)['painting']
        im = self.spec.load_image(int(pm['image_index']))
        pim = _qwen_prepare_image(im, QWEN_MAX_PIXELS)
        r, c = (int(x) for x in pm['grid_hw'])
        assert pim.size == (c * 32, r * 32), (pim.size, r, c)
        b = io.BytesIO()
        pim.save(b, 'JPEG', quality=90)
        self.imgcache[k] = b.getvalue()
        while len(self.imgcache) > 40:
            self.imgcache.popitem(last=False)
        return self.imgcache[k]


def r5(a):
    """Round to 5 significant digits; returns nested lists."""
    a = np.asarray(a, np.float64)
    with np.errstate(divide='ignore'):
        e = np.floor(np.log10(np.abs(a)))
    e = np.where(np.isfinite(e), e, 0)
    sc = 10.0 ** (4 - e)
    return (np.round(a * sc) / sc).tolist()


def layer_range(layer):
    """layer: band name or integer string -> (name, a, b)."""
    for b in C.BANDS:
        if b[0] == layer:
            return b
    l = int(layer)
    assert 0 <= l < C.N_LAYERS
    return (f'layer {l}', l, l + 1)


def maps_for(ctx, k, mod, cap, layer):
    P = ctx.painting(k)
    pm = P['painting']
    gr, gc = (int(x) for x in pm['grid_hw'])
    band = layer_range(layer)
    _, a, b = band
    out = {}
    wband_idx = [x[0] for x in C.BANDS].index(WORD_BAND)
    for v in C.VIEWS:
        if v not in P:
            continue
        p = P[v][mod][0 if mod == 'img' else cap]
        tt = np.asarray(p['ttype'])
        L = np.asarray(p['layers'][a:b], np.float64)          # (n_layers, L)
        s = L.sum(0)
        pos0 = np.zeros(len(tt), bool)
        pos0[0] = True
        sh = {'first token': L[:, pos0].sum() / (b - a),
              'instruction': L[:, tt == 1].sum() / (b - a),
              'image': L[:, tt == 2].sum() / (b - a),
              'caption': L[:, tt == 3].sum() / (b - a),
              'other template': L[:, (tt == 0) & ~pos0].sum() / (b - a)}
        d = {'image_map': None, 'caption_tokens': None, 'word_maps': None, 'word_maps_band': None,
             'shares': {kk: float(x) for kk, x in sh.items()}}
        if mod != 'txt':
            m = s[tt == 2]
            assert m.size == gr * gc
            d['image_map'] = r5((m / m.sum()).reshape(gr, gc))
        if mod != 'img':
            m = s[tt == 3]
            m = m / m.sum()
            ids = np.asarray(p['ids'])[tt == 3]
            d['caption_tokens'] = [{'text': ctx.tok_text(i), 'weight': float(r5(w))} for i, w in zip(ids, m)]
        if mod == 'joint':
            cr = np.asarray(p['cross'][wband_idx], np.float64)      # (n_cap_tok, n_img_tok)
            cr = cr / np.maximum(cr.sum(1, keepdims=True), 1e-300)
            d['word_maps'] = r5(cr.reshape(cr.shape[0], gr, gc))
            d['word_maps_band'] = band[0] if band[0] in [x[0] for x in C.BANDS] else WORD_BAND
        out[v] = d
    return {'band': band[0], 'views': out}


def meta(ctx):
    ks = []
    for f in sorted(ctx.root.glob('p[0-9][0-9][0-9][0-9].npz')):
        if '.tmp' in f.name:
            continue
        k = int(f.stem[1:])
        if C.is_complete(ctx.root, k):
            ks.append(k)
    return {'ks': ks, 'views': ctx.prompts, 'bands': [x[0] for x in C.BANDS], 'layers': list(range(C.N_LAYERS)),
            'word_band': WORD_BAND}


def painting_info(ctx, k):
    P = ctx.painting(k)
    pm = P['painting']
    n = ctx.names
    caps = []
    for i, r in enumerate(pm['cap_rows']):
        caps.append({'text': str(ctx.spec.texts[int(r)]), 'emotion': n['emo'][int(pm['cap_emo'][i])]})
    g = int(pm['gen'][0])
    return {'k': k, 'painting_id': str(pm['painting_id']), 'style': n['sty'][int(pm['sty'][0])],
            'genre': n['gen'][g] if g >= 0 else 'unknown', 'captions': caps,
            'grid_hw': [int(x) for x in pm['grid_hw']]}


class H(BaseHTTPRequestHandler):
    ctx = None

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {a: b[0] for a, b in parse_qs(u.query).items()}
        try:
            if u.path == '/':
                return self.send(200, (HERE / 'att_viewer.html').read_bytes(), 'text/html; charset=utf-8')
            if u.path in ('/att_viewer.js', '/att_viewer.css'):
                f = HERE / u.path[1:]
                if not f.exists():
                    return self.send(404, 'not found', 'text/plain')
                ct = 'application/javascript' if u.path.endswith('js') else 'text/css'
                return self.send(200, f.read_bytes(), ct)
            ctx = self.ctx
            if u.path == '/api/meta':
                return self.send(200, json.dumps(meta(ctx)), 'application/json')
            if u.path == '/api/image':
                return self.send(200, ctx.image_jpeg(int(q['k'])), 'image/jpeg')
            if u.path == '/api/painting':
                return self.send(200, json.dumps(painting_info(ctx, int(q['k']))), 'application/json')
            if u.path == '/api/maps':
                mod = q.get('mod', 'img')
                assert mod in C.MODS
                r = maps_for(ctx, int(q['k']), mod, int(q.get('cap', 0)), q.get('layer', C.BANDS[0][0]))
                return self.send(200, json.dumps(r), 'application/json')
            return self.send(404, 'not found', 'text/plain')
        except Exception as e:
            return self.send(400, json.dumps({'error': repr(e)}), 'application/json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    H.ctx = Ctx(a.root)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    print(f'serving {a.root} on http://127.0.0.1:{a.port}/', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
