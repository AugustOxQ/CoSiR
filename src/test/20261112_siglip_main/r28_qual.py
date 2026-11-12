"""r28 job G: Figures 3 and 4 candidates on the SigLIP 1 student (c*, arm k11=40, seed 0), by the unchanged rules of
`20261111_qual_figure/qf_cand.py`. Thin wrapper: patches r20's data door (`r20_data.load`) so c* rows are the SigLIP 1
held exports and the plain image is the plain SigLIP 1 (store `siglip_so400m`), then runs r20 partitions + candidates
(k-means on the SigLIP 1 rows, into our own folder), qf_cand.run and qf_view.run. Nothing of the Qwen3 round is edited.

    r28_qual.py partitions | r20cand | candidates | viewer | all     (env R20_OUTPUTS / QF_OUTPUTS are set here)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUTROOT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/qual')
os.environ['R20_OUTPUTS'] = str(OUTROOT / 'r20x')
os.environ['QF_OUTPUTS'] = str(OUTROOT)
ROUND = Path(__file__).resolve().parent
QF = Path('/project/CoSiR/src/test/20261111_qual_figure')
R20 = Path('/project/CoSiR/src/test/20261109_r20_figures')
for p in (QF, R20, ROUND):
    sys.path.insert(0, str(p))

import r20_data as D  # noqa: E402

HELD = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/held/k11=40')
PLAIN_STORE = Path('/data/SSD2/pre_extract/artelingo_full/siglip_so400m')
_orig_load = D.load


def load(name):
    assert name == 'artelingo', name
    q = _orig_load(name)  # labels, names, artists, paths: the same held list as the Qwen3 version
    files = {a: HELD / f'{a}__s{D.SEED}.npz' for a in D.ASPECTS}
    for a, f in files.items():
        assert f.exists(), f
    sys.path.insert(0, '/project/CoSiR/src/test/20261025_recipe_backbones')
    D.setup_paths()
    import rb_encoders as E

    def c_loader(aspect):
        with np.load(files[aspect]) as z:
            x = z['img']
        assert x.shape[0] == len(q.keys), (x.shape, len(q.keys))
        return x

    def plain_loader():
        return E.read_store(PLAIN_STORE, 1152, q.keys, E.FULL_COUNT)

    src = dict(c_star={a: str(files[a]) for a in D.ASPECTS}, plain=str(PLAIN_STORE / 'img.npy'),
               held_list_sha256=q.sources['held_list_sha256'])
    return D.Set('artelingo', q.keys, q.labels, q.class_names, q.artists, q.image_paths, c_loader, plain_loader, src)


D.load = load


def main(step):
    if step in ('partitions', 'all'):
        import r20_part
        print(r20_part.run(sets=['artelingo']))
    if step in ('r20cand', 'all'):
        import r20_cand
        r20_cand.run(sets=['artelingo'])
    if step in ('candidates', 'all'):
        import qf_cand
        qf_cand.R20_CANDIDATES = D.OUT / 'candidates' / 'candidates.json'
        doc = qf_cand.run()
        doc['header']['model'] = 'SigLIP 1 student c* (arm k11=40, seed 0); plain = SigLIP 1 image (siglip_so400m)'
        (Path(qf_cand.OUT) / 'candidates' / 'candidates.json').write_text(json.dumps(doc, separators=(',', ':')))
        print(json.dumps(doc['counts'], default=str))
    if step in ('viewer', 'all'):
        import qf_view
        print(qf_view.run())


if __name__ == '__main__':
    main(sys.argv[1])
