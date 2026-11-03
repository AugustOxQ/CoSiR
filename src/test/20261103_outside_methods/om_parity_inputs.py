"""r18 stage 0 parity inputs (spec 2026-11-03-outside-methods-design.md, gates (i) and (ii)).

Writes /data/SSD2/cosir_r18/ext/parity/inputs.npz: the first 2,000 sorted ArtELingo validation paintings,
their unit CLIP L/14 rows (CLAY, gate (i)) and unit plain Qwen rows (TURTLE, gate (ii)), and r11's unit emotion
text vectors of text set A (CRL's 50 values). Our code only; the official code reads this file elsewhere.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND.parent / '20261029_retrieval'))
import rt_crl  # noqa: E402
import rt_data  # noqa: E402

OUT = Path('/data/SSD2/cosir_r18/ext/parity')
N = 2000


def main():
    names = rt_data.names('val')[:N]
    clip = rt_data.embed('clip', 'val', 'emotion')[:N]
    qwen = rt_data.embed('plain', 'val', 'emotion')[:N]
    text = rt_crl.text_array('emotion')
    for x in (clip, qwen, text):
        assert x.dtype == np.float32 and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-3), x.dtype
    assert clip.shape == (N, 768) and qwen.shape == (N, 2048) and text.shape == (50, 768)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'inputs.npz'
    np.savez(path, clip=clip, qwen=qwen, text=text, names=np.array(names))
    meta = {'n': N, 'split': 'val', 'order': 'sorted painting names, first 2,000', 'aspect_text': 'emotion, text set A',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / 'inputs.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
