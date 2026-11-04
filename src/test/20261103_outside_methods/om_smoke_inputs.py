"""r18 ticket 11: the SYNTHETIC inputs of the end-to-end smoke (OM_SMOKE=1, OM_OUTPUTS=<scratch>; refuses to run otherwise).

  python om_smoke_inputs.py link                    OUT/textset_b -> the real set B folder (read only; smoke finds lists.json there)
  python om_smoke_inputs.py tpips                   TPIPS stores OUT/tpips_store_smoke/<set>/<factor>/{emb.npy, keys.json, meta.json}:
                                                    random unit rows [n, 4096], meta fake true (om_tpips.load accepts that only in smoke);
                                                    ArtELingo held + val keys, MMArt test keys; checked through om_data.tpips
  python om_smoke_inputs.py parts SETS ENCS         TURTLE partition files (random labels) through om_data.save_parts, every file the
                                                    cells of those sets and encoders read; an existing file is kept (a real short fit stays)
  python om_smoke_inputs.py gates                   stub gate files (iii a, b ok; iii c with a stale code hash; iv ok; ii CUDA part ok)
  python om_smoke_inputs.py ranges                  om_stage1.build_ranges with the slow Qwen k-means range left out (r null for 3 keys)
Real parts of the smoke: gate (i), gate (ii) CPU, TURTLE fit (om_turtle_run fit --outer-steps 30 --only-k 20 --only-start 0 on CPU),
stage 1 picks, stage 2 cells, om_diag.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import om_paths as op  # noqa: E402

REAL_OUT = '/data/SSD2/cosir_r18/20261103_outside_methods/outputs'


def need_smoke():
    assert op.SMOKE and Path(op.OUT).resolve() != Path(REAL_OUT).resolve(), 'smoke inputs: set OM_SMOKE=1 and OM_OUTPUTS=<scratch>'


def link():
    l = op.OUT / 'textset_b'
    if not l.exists():
        l.symlink_to(REAL_OUT + '/textset_b')
    print('link', l, '->', os.readlink(l))


def tpips():
    import om_data as od
    import om_tpips
    rng = np.random.default_rng(0)
    keys = {'artelingo': od.keys('artelingo', 'held') + od.keys('artelingo', 'val'), 'mmart': od.keys('mmart', 'test')}
    for s, ks in keys.items():
        assert len(set(ks)) == len(ks)
        for f in om_tpips.FACTORS:
            d = om_tpips.store_root(s) / f
            d.mkdir(parents=True, exist_ok=True)
            x = rng.standard_normal((len(ks), op.TPIPS_WIDTH))
            x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
            np.save(d / 'emb.npy', x)
            (d / 'keys.json').write_text(json.dumps(ks))
            (d / 'meta.json').write_text(json.dumps({'set': s, 'factor': f, 'snapshot': 'fake', 'n': len(ks), 'width': op.TPIPS_WIDTH,
                                                      'fake': True, 'sha256': op.sha256_file(d / 'emb.npy')}))
            print('store', s, f, x.shape)
    print('readers ok', od.tpips('mmart', 'val', 'emotion').shape, od.tpips('artelingo', 'held', 'genre').shape)


def parts(sets, encs):
    import om_data as od
    rng = np.random.default_rng(1)
    for s in sets:
        n = len(od.keys(s, op.HELD_SPLIT[s]))
        for e in encs:
            todo = [('plain', k, None, st, None) for k in op.plain_ks(s) for st in op.TURTLE_PLAIN_STARTS]
            todo += [('cstar', k, a, 0, sd) for a in op.ASPECTS for k in op.label_ks(s, a) for sd in op.SEEDS]
            made = 0
            for arm, k, a, st, sd in todo:
                if od._path(None, 'turtle', s, e, arm, od._name(arm, k, a, st, sd)).exists():
                    continue
                y = rng.integers(0, k, n).astype(np.int32)
                od.save_parts(y, method='turtle', set_=s, enc=e, arm=arm, k=k, aspect=a, start=st, seed=sd, input_sha256='synthetic',
                              loss=0.0, entropy=0.0, n_clusters=int(len(np.unique(y))), seconds=0.0, device='synthetic')
                made += 1
            print(f'parts {s}/{e}: {made} written, {len(todo) - made} kept; read back',
                  od.turtle_parts(s, e, 'plain', 20).shape, od.turtle_parts(s, e, 'cstar', op.label_ks(s, 'emotion')[0], 'emotion').shape)


def gates():
    import om_gate_iii
    import om_run
    st = op.OUT / 'stage0'
    h = om_gate_iii.code_sha256()
    op.write_json(st / 'gate_iii.json', {'a': {'pass': True, 'code_sha256': h, 'stub': True}, 'b': {'pass': True, 'code_sha256': h, 'stub': True},
                                         'c': {'pass': True, 'code_sha256': '0' * 64, 'stub': True}, 'pass': False})
    op.write_json(st / 'gate_iv.json', {'pass': True, 'stub': True})
    g2 = json.loads((st / 'gate_ii.json').read_text())
    g2['cuda'] = {'pass': True, 'code_sha256': om_run.turtle_code_sha256(), 'stub': True}
    g2['pass'] = True
    op.write_json(st / 'gate_ii.json', g2)
    print('stub gate files written: iii c has a stale code hash on purpose')


def ranges():
    import ns_paths
    import om_run
    import om_stage1 as s1
    table = s1.build_ranges({})
    ns_paths.write_json(op.OUT / 'stage1' / 'om_ranges.json', table)
    print(len(table), 'keys,', sum(v['r'] is None for v in table.values()), 'with r null')
    om_run.write_step('stage1_ranges', 'ok', None, 0.0, note='smoke stand-in: Qwen k-means range not computed')


if __name__ == '__main__':
    need_smoke()
    cmd = sys.argv[1]
    if cmd == 'parts':
        parts(sys.argv[2].split(','), sys.argv[3].split(','))
    else:
        {'link': link, 'tpips': tpips, 'gates': gates, 'ranges': ranges}[cmd]()
