"""r28 job B part 1: Table 4 (MMArt) cells on SigLIP 1, read from r17's records (no rescoring).

r17 (`src/test/20261103_new_scores/records/verdicts.json`) scored MMArt for the four encoders with one code path, so
SigLIP 1 and Qwen3 cells come from the same cell families. Checked at source (see the r28 job B hand-back):
  AMI at 9 / 20 / 10  SC1 (c* vs plain: `cand` = c*, `comp` = plain) and SC2 (c* vs matched PercepT: `comp` = PercepT)
  image mAP           SC8 (`cand` = c*, `comp` = matched PercepT) and SC7 (`comp` = plain), each at the read-out picked on the 5,769
                      ArtELingo validation paintings that MMArt labels (c*, plain: r15 `records/stage1_<enc>.json` picks.arms;
                      matched PercepT: r17 `records/stage1_picks.json` key `matched/mmart/<enc>`), the same picks route for Qwen.
The reader asserts that the same ids for Qwen reproduce the paper's Qwen3 MMArt rows (`Tables/mmart.tex`) at three decimals,
so the cell mapping is the one the paper uses. TURTLE: r18 `records/labels.json` cells `SC5/mmart/<enc>/.../plain_turtle` (`comp`).

CLI: python r28_mmart.py   -> records/mmart.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
R17 = TEST / '20261103_new_scores/records/verdicts.json'
R17_PICKS = TEST / '20261103_new_scores/records/stage1_picks.json'
R15 = TEST / '20261102_mmart_test/records'
R18 = TEST / '20261103_outside_methods/records/labels.json'
R18_TURTLE_DIR = Path('/data/SSD2/cosir_r18/20261103_outside_methods/outputs/parts/turtle/mmart')
RECORD = HERE / 'records/mmart.json'

ASPECTS = (('emotion', 9), ('style', 20), ('genre', 10))
# (arm, metric) -> (cell id template, side of the cell that holds the arm)
CELL = {
    ('cstar', 'ami'): ('SC1/mmart/{e}/{a}/ami/{k}/plain', 'cand'),
    ('plain', 'ami'): ('SC1/mmart/{e}/{a}/ami/{k}/plain', 'comp'),
    ('percept', 'ami'): ('SC2/mmart/{e}/{a}/ami/{k}/matched', 'comp'),
    ('cstar', 'map'): ('SC8/mmart/{e}/{a}/map/-/matched', 'cand'),
    ('plain', 'map'): ('SC7/mmart/{e}/{a}/map/-/plain', 'comp'),
    ('percept', 'map'): ('SC8/mmart/{e}/{a}/map/-/matched', 'comp'),
    ('turtle', 'ami'): ('SC5/mmart/{e}/{a}/ami/{k}/plain_turtle', 'comp'),
}
# Qwen3 rows now in the paper's Tables/mmart.tex (Emo., Style, Genre; AMI then mAP)
PAPER_QWEN = {
    ('plain', 'ami'): (0.038, 0.231, 0.458), ('plain', 'map'): (0.194, 0.205, 0.515),
    ('percept', 'ami'): (0.039, 0.192, 0.439), ('percept', 'map'): (0.193, 0.167, 0.504),
    ('cstar', 'ami'): (0.053, 0.423, 0.531), ('cstar', 'map'): (0.201, 0.347, 0.620),
    ('turtle', 'ami'): (0.038, 0.223, 0.422),
}


def _index(path):
    """{cell id: (cell, line number of its "id" line)}; the file is the committed json with one key per line (indent 1)."""
    d = json.loads(path.read_text())
    lines = path.read_text().splitlines()
    where = {}
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith('"id": "'):
            where[s[len('"id": "'):].rstrip(',').rstrip('"')] = n
    out = {}
    for c in d['cells']:
        assert c['id'] in where and c['id'] not in out, c['id']
        out[c['id']] = (c, where[c['id']])
    return out


def read_cells(enc, arms=('cstar', 'plain', 'percept'), metrics=('ami', 'map')):
    r17, r18 = _index(R17), _index(R18)
    out = {}
    for arm in arms:
        for metric in metrics:
            if (arm, metric) not in CELL:
                continue
            tpl, side = CELL[(arm, metric)]
            vals, src = [], []
            for a, k in ASPECTS:
                cid = tpl.format(e=enc, a=a, k=k)
                idx = r18 if arm == 'turtle' else r17
                cell, line = idx[cid]
                assert cell['set'] == 'mmart' and cell['encoder'] == enc and cell['aspect'] == a, cid
                if arm == 'turtle':
                    assert not cell['void'] and cell['comparator'] == 'plain_turtle', cid
                else:
                    assert cell['status'] == 'decided' and cell['present'] and not cell['void'], (cid, cell['status'])
                assert cell['n_rows'] == 16384, (cid, cell['n_rows'])
                if metric == 'ami':
                    assert cell['k'] == k, cid
                vals.append(float(cell[side]))
                src.append(f"{R18 if arm == 'turtle' else R17}:{line} {cid} .{side}")
            out[f'{arm}_{metric}'] = dict(values=vals, sources=src)
    return out


def picks(enc):
    """The validation picks behind the mAP cells: c* and plain from r15, matched PercepT from r17 (all n_val 5769)."""
    r15 = json.loads((R15 / f'stage1_{enc}.json').read_text())['picks']
    assert r15['n_val'] == 5769 and r15['kind'] == 'mmart_retrieval_picks'
    p17 = json.loads(R17_PICKS.read_text())[f'matched/mmart/{enc}']
    assert p17['n_val'] == 5769 and p17['scores']['kept']
    return dict(cstar=r15['arms']['cstar']['pick'], plain=r15['arms']['frozen']['pick'], percept=p17['pick'],
                n_val=5769, sources=[f'{R15}/stage1_{enc}.json picks.arms.cstar|frozen', f'{R17_PICKS} matched/mmart/{enc}'])


def check_against_paper(qwen):
    for key, want in PAPER_QWEN.items():
        arm, metric = key
        name = f'{arm}_{metric}'
        if name not in qwen:
            continue
        got = tuple(round(v, 3) for v in qwen[name]['values'])
        assert got == want, (name, got, want)


def main():
    qwen = read_cells('qwen', arms=('cstar', 'plain', 'percept', 'turtle'))
    check_against_paper(qwen)                               # same ids, Qwen3: equals the paper's rows at 3 decimals
    sig = read_cells('siglip1', arms=('cstar', 'plain', 'percept', 'turtle'))
    turtle_parts = sorted(R18_TURTLE_DIR.joinpath('siglip1').rglob('*.npz'))
    rec = dict(
        note='r17 / r18 records; Qwen3 values re-read through the same ids and equal the paper rows at 3 decimals',
        siglip1=sig, qwen3=qwen, picks=dict(siglip1=picks('siglip1'), qwen=picks('qwen')),
        turtle_parts_dir=str(R18_TURTLE_DIR / 'siglip1'), turtle_parts_files=len(turtle_parts),
        columns=['emotion (k=9)', 'style (k=20)', 'genre (k=10)'])
    RECORD.parent.mkdir(exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    for name in sorted(sig):
        print(f'{name:14s} siglip1 ' + ' '.join(f'{v:.4f}' for v in sig[name]['values']) +
              '   qwen3 ' + ' '.join(f'{v:.4f}' for v in qwen[name]['values']))
    print('picks siglip1', {k: v for k, v in rec['picks']['siglip1'].items() if k in ('cstar', 'plain', 'percept')},
          'qwen', {k: v for k, v in rec['picks']['qwen'].items() if k in ('cstar', 'plain', 'percept')})
    print('turtle siglip1 parts files', len(turtle_parts))


if __name__ == '__main__':
    main()
