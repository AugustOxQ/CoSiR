"""Shared frames, explicit key/name joins, and the r23 forward recipe on cut rows."""
from functools import lru_cache

import numpy as np

from mc_common import OUT, P
import tcl_data as D
import tcl_forward as F
import mt_data
import mt_paths
import tx_data
import tx_model
from pav_prompts import PROMPTS


def store(instruction):
    return OUT / 'stores/mmart/qwen3vl_emb_2b_text/narrative' / instruction / 'shared_held'


def mapped_labels(keys):
    out = {}
    for a in P.ASPECTS:
        y = mt_data.labels(keys, a)
        names = D.r22_names(a)
        mapping = D.class_map(a)
        codes = np.asarray([names.index(mapping[c]) for c in mt_paths.CLASSES[a]], dtype=np.int64)
        out[a] = np.where(y < 0, -1, codes[np.maximum(y, 0)])
    return out


@lru_cache(maxsize=2)
def shared(smoke=0):
    keys = mt_data.keys('shared_held')
    assert len(keys) == len(set(keys)) == 11528
    art = D.frame('art_held', 'caption')
    assert art['n_t'] == 61744
    pos = {n: i for i, n in enumerate(art['names'])}
    assert len(pos) == len(art['names'])
    names = [mt_data.artelingo_name(k) for k in keys]
    assert len(set(names)) == len(names) and all(n in pos for n in names)
    if smoke:
        keys, names = keys[:smoke], names[:smoke]
    image_rows = np.asarray([pos[n] for n in names], dtype=np.int64)
    to_shared = {p: j for j, p in enumerate(image_rows)}
    text_rows = np.asarray([j for j, p in enumerate(art['tp']) if p in to_shared], dtype=np.int64)
    tp = np.asarray([to_shared[p] for p in art['tp'][text_rows]], dtype=np.int64)
    ml = mapped_labels(keys)
    al = {a: dict(t=v['t'][text_rows], i=v['i'][image_rows]) for a, v in art['labels'].items()}
    n = len(keys)
    common = dict(smoke=smoke, n_i=n, names=names, keys=keys, ip=np.arange(n, dtype=np.int64), image_rows=image_rows)
    af = dict(common, n_t=len(tp), tp=tp, text_rows=text_rows,
              text=[art['text'][i] for i in text_rows], sample_id=art['sample_id'][text_rows],
              labels=dict(A=al, B={a: dict(t=y[tp], i=y) for a, y in ml.items()}))
    # The raw text join happens only for extraction; scoring needs no parquet read.
    mf = dict(common, n_t=n, tp=np.arange(n, dtype=np.int64),
              labels=dict(C={a: dict(t=v['i'].copy(), i=v['i']) for a, v in al.items()},
                          D={a: dict(t=y.copy(), i=y) for a, y in ml.items()}))
    return dict(A=af, M=mf)


def narrative(keys):
    import pandas as pd
    f = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet', columns=['image_id', 'e_narrative'],
                        filters=[('image_id', 'in', list(keys))])
    assert f.image_id.is_unique and not f.e_narrative.isna().any()
    rows = tx_data.gather(f.image_id.tolist(), keys)
    assert f.iloc[rows].image_id.tolist() == keys
    return f.iloc[rows].e_narrative.tolist()


class Rows:
    """Cut the base rows before forwarding, using exactly tcl_forward.rows' recipe."""
    def __init__(self, text_kind, fr):
        self.kind, self.fr, self.cache = text_kind, fr, {}

    def base(self, aspect=None):
        key = aspect or 'plain'
        if key not in self.cache:
            if self.kind == 'A':
                full = D.plain_text('art_held', 'caption') if aspect is None else D.teacher_text('art_held', 'caption', aspect)
                text = full[self.fr['text_rows']]
            else:
                instruction = 'default' if aspect is None else P.INSTRUCTIONS[aspect]
                old_smoke = P.SMOKE
                try:
                    # Only the dedicated MC_OUTPUTS store reader accepts smoke metadata.
                    P.SMOKE = self.fr['smoke']
                    text = D.read_store(store(instruction), self.fr['keys'], 'keys', PROMPTS[instruction], 2048, np.float16)
                finally:
                    P.SMOKE = old_smoke
            image = D.image('art_held', 'plain' if aspect is None else 'teacher', aspect or 'emotion')[self.fr['image_rows']]
            self.cache[key] = text, image
        return self.cache[key]

    def rows(self, row, head, seed, device):
        if row == 'teacher':
            return self.base(head)
        text, image = self.base()
        if row == 'plain':
            return text, image
        image_side, caption_side = tx_model.load_sides(head, seed, device)
        ct, ci = caption_side(text), image_side(image)
        if row == 'cstar':
            return ct, ci
        assert row == 'late'
        return F.fuse(ct, text), F.fuse(ci, image)
