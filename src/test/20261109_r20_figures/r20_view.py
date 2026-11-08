"""Step `viewer`: static review page. Reads candidates.json, makes thumbnails (height 128 px, width kept and capped at
220 px, JPEG quality 72) packed as base64 in JSON files of at most 12 MB, and writes
<out>/viewer/index.html + data/{candidates.json, manifest.json, thumbs_<set>_<i>.json, thumbs_<set>_q.json}.
`thumbs_<set>_q.json` holds the teaser queries at 320 px height (they are shown larger). Relative paths only.
"""
import base64
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import r20_data as D

HEIGHT, MAXW, QUALITY = 128, 220, 72
BIG_HEIGHT, BIG_QUALITY = 320, 80
MAX_FILE = 12_000_000
TEMPLATE = D.ROUND / 'r20_viewer_template.html'


def thumb_b64(path, height=HEIGHT, maxw=MAXW, quality=QUALITY):
    """Base64 JPEG of the image: height `height`, width kept, capped at `maxw` (then height shrinks, ratio kept)."""
    with Image.open(path) as im:
        im.draft('RGB', (maxw * 2, height * 2))            # JPEG: decode at reduced size
        im = im.convert('RGB')
        w, h = im.size
        nw = max(1, round(w * height / h))
        nh = height
        if nw > maxw:
            nw, nh = maxw, max(1, round(h * maxw / w))
        im = im.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def pack(items, limit=MAX_FILE):
    """Split an ordered list of (key, b64) into chunks whose JSON stays under `limit` bytes."""
    chunks, cur, size = [], {}, 2
    for k, v in items:
        add = len(json.dumps(k)) + len(v) + 4
        if cur and size + add > limit:
            chunks.append(cur)
            cur, size = {}, 2
        cur[k] = v
        size += add
    if cur:
        chunks.append(cur)
    return chunks


def run(smoke=False):
    root = D.OUT / ('smoke' if smoke else '')
    doc = json.loads((root / 'candidates' / 'candidates.json').read_text())
    out = root / 'viewer'
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    manifest = dict(thumbs={}, big={}, counts={})
    for name, S in doc['sets'].items():
        tz = []
        for t in S['teaser']:
            tz += [t['query']] + sum((h['neighbours'] for h in t['heads'].values()), []) + t['plain']['neighbours']
        first = list(dict.fromkeys(tz))                              # teaser paintings first, so chunk 0 serves that tab
        rest = sorted(set(S['paintings']) - set(first))
        order = first + rest
        paths = [S['paintings'][k]['img'] for k in order]
        with ThreadPoolExecutor(8) as ex:
            b64 = list(ex.map(thumb_b64, paths))
        files = []
        for i, chunk in enumerate(pack(list(zip(order, b64)))):
            f = f'thumbs_{name}_{i}.json'
            (data / f).write_text(json.dumps(chunk, separators=(',', ':')))
            files.append(f)
        manifest['thumbs'][name] = files
        manifest['counts'][name] = len(order)
        queries = [t['query'] for t in S['teaser']]
        big = {k: thumb_b64(S['paintings'][k]['img'], BIG_HEIGHT, 2 * MAXW, BIG_QUALITY) for k in queries}
        f = f'thumbs_{name}_q.json'
        (data / f).write_text(json.dumps(big, separators=(',', ':')))
        manifest['big'][name] = f
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    shutil.copyfile(root / 'candidates' / 'candidates.json', data / 'candidates.json')
    shutil.copyfile(TEMPLATE, out / 'index.html')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    return dict(files=sum(len(v) for v in manifest['thumbs'].values()), thumbnails=manifest['counts'], bytes=size)
