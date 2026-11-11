"""Step `viewer`: static review page for the qualitative-figure candidates. Reads candidates.json, makes thumbnails
with r20's `thumb_b64` / `pack` (height 128 px, width capped at 220 px, JPEG quality 72; the panel (a) queries and the
panel (b) pair members at 320 px height) and writes <out>/viewer/index.html + data/{candidates.json, manifest.json,
thumbs_<i>.json, thumbs_big.json}. Relative paths only; serve the folder with `python3 -m http.server`.
"""
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROUND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND))
import qf_cand as Q  # noqa: E402  (also puts r20 on sys.path)
import r20_view as V  # noqa: E402

TEMPLATE = ROUND / 'qf_viewer_template.html'


def run(smoke=False):
    root = Q.OUT / ('smoke' if smoke else '')
    doc = json.loads((root / 'candidates' / 'candidates.json').read_text())
    out = root / 'viewer'
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    P = doc['paintings']
    big_keys = [c['query'] for c in doc['panel_a']] + [k for c in doc['panel_b'] for k in (c['key_i'], c['key_j'])]
    big_keys = list(dict.fromkeys(big_keys))
    first = [c['query'] for c in doc['panel_a']] + [n['key'] for c in doc['panel_a'] for n in c['plain'] + c['ours']] + big_keys
    first = list(dict.fromkeys(first))
    order = first + sorted(set(P) - set(first))
    with ThreadPoolExecutor(8) as ex:
        b64 = list(ex.map(lambda k: V.thumb_b64(P[k]['img']), order))
        big = list(ex.map(lambda k: V.thumb_b64(P[k]['img'], V.BIG_HEIGHT, 2 * V.MAXW, V.BIG_QUALITY), big_keys))
    files = []
    for i, chunk in enumerate(V.pack(list(zip(order, b64)))):
        f = f'thumbs_{i}.json'
        (data / f).write_text(json.dumps(chunk, separators=(',', ':')))
        files.append(f)
    (data / 'thumbs_big.json').write_text(json.dumps(dict(zip(big_keys, big)), separators=(',', ':')))
    (data / 'manifest.json').write_text(json.dumps(dict(thumbs=files, big='thumbs_big.json', count=len(order),
                                                        big_count=len(big_keys)), indent=1))
    shutil.copyfile(root / 'candidates' / 'candidates.json', data / 'candidates.json')
    shutil.copyfile(TEMPLATE, out / 'index.html')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    return dict(files=len(files) + 1, thumbnails=len(order), big=len(big_keys), bytes=size, mb=round(size / 1e6, 2))
