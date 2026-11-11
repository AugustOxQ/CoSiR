"""Table 3 rows, input readers: the VLM2Vec-V2 stores and the PercepT probability files, aligned to the project's keys.

VLM2Vec store (written by r25): `<root>/<aspect>/{emb.npy, keys.json, meta.json}`, float32 [n, 1536] unit rows, keys =
painting names. Rows are reordered by name to `keys(split)` (sorted painting names, the order of every label), after
asserting the same key set. `meta.json` is written last, so its `n` and `emb_sha256` double as a completeness check.

PercepT file (r26): `<dir>/<file-pattern with {seed}>`, an npz with `prob` [n, 40] and `names` [n]. Rows = `ns_ret._unit(prob)`
(r17's matched form: probabilities over their float32 row norm). Names must equal the project's keys; rows are reordered by
name if the file lists the same names in another order.
"""
import json
from pathlib import Path

import numpy as np

import t3_core as C

N_TOPICS = 40
VLM2VEC_DIM = 1536


def align(names, x, want, what):
    """Rows of `x` (named `names`) in the order of `want`; asserts the same key set and no duplicates."""
    names = [str(n) for n in names]
    want = list(want)
    assert len(names) == len(x) and len(set(names)) == len(names), f'{what}: duplicate names or wrong count'
    assert len(want) == len(set(want)), f'{what}: duplicate project keys'
    assert set(names) == set(want), (f'{what}: key sets differ ({len(set(names) - set(want))} extra, '
                                     f'{len(set(want) - set(names))} missing)')
    if names == want:
        return x
    pos = {n: i for i, n in enumerate(names)}
    return np.ascontiguousarray(x[[pos[n] for n in want]])


class Vlm2VecStore:
    """One split's VLM2Vec-V2 stores (three aspects): `rows(aspect)` aligned to `keys(split)`, `provenance()` with hashes."""

    def __init__(self, root, split, n_expected=None):
        self.root, self.split = Path(root), split
        self.want = C.keys(split)
        assert n_expected is None or len(self.want) == n_expected, (split, len(self.want))
        self._rows, self._prov = {}, {}

    def rows(self, aspect):
        if aspect not in self._rows:
            d = self.root / aspect
            for f in ('emb.npy', 'keys.json', 'meta.json'):
                assert (d / f).is_file(), f'missing {d / f} (store not written yet?)'
            meta = json.loads((d / 'meta.json').read_text())
            assert meta.get('aspect') == aspect, (d, meta.get('aspect'))
            assert meta.get('n') == len(self.want) and meta.get('limit') in (None, 0), (d, meta.get('n'), meta.get('limit'))
            assert meta.get('split', self.split) == self.split, (d, meta.get('split'), self.split)
            sha = C.file_sha(d / 'emb.npy')
            assert meta.get('emb_sha256') == sha, f'{d}/emb.npy hash differs from meta.json emb_sha256'
            emb = np.load(d / 'emb.npy')
            names = json.loads((d / 'keys.json').read_text())
            assert emb.dtype == np.float32 and emb.shape == (len(names), VLM2VEC_DIM), (d, emb.dtype, emb.shape)
            x = align(names, emb, self.want, str(d))
            C.check_rows(x, len(self.want), str(d), VLM2VEC_DIM)
            self._rows[aspect] = x
            self._prov[aspect] = dict(
                dir=str(d), emb_sha256=sha, keys_sha256=C.file_sha(d / 'keys.json'), meta_sha256=C.file_sha(d / 'meta.json'),
                reordered=bool(names != self.want), instruction=meta.get('instruction'), model=meta.get('model'),
                revision=meta.get('revision'))
        return self._rows[aspect]

    def provenance(self):
        return dict(self._prov)


class PerceptFiles:
    """One split's PercepT probability files of the kept seeds: `rows(seed)` unit rows [n, 40] in `keys(split)` order."""

    def __init__(self, directory, file_pattern, split, seeds, n_expected=None):
        assert '{seed}' in file_pattern, file_pattern
        self.dir, self.pattern, self.split, self.seeds = Path(directory), file_pattern, split, tuple(seeds)
        self.want = C.keys(split)
        assert n_expected is None or len(self.want) == n_expected, (split, len(self.want))
        self._rows, self._prov = {}, {}

    def path(self, seed):
        return self.dir / self.pattern.format(seed=seed)

    def rows(self, seed):
        if seed not in self._rows:
            p = self.path(seed)
            assert p.is_file(), f'missing {p}'
            with np.load(p, allow_pickle=False) as z:
                assert 'prob' in z.files and 'names' in z.files, (p, z.files)
                prob, names = z['prob'], z['names']
            prob = np.asarray(prob)
            assert prob.ndim == 2 and prob.shape == (len(names), N_TOPICS) and np.isfinite(prob).all(), (p, prob.shape)
            x = C.ns_ret._unit(align(names, prob, self.want, str(p)))
            C.check_rows(x, len(self.want), str(p), N_TOPICS)
            self._rows[seed] = x
            self._prov[seed] = dict(path=str(p), sha256=C.file_sha(p), reordered=bool([str(n) for n in names] != self.want),
                                    dtype=str(prob.dtype))
        return self._rows[seed]

    def provenance(self):
        return {str(s): v for s, v in self._prov.items()}
