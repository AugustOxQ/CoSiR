"""Top-six neighbours for each existing panel (a) candidate, on CPU."""
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import r28_qual as wrapper  # patches the r20 data door and sets import paths
import qf_cand as QF
import r20_cand as RC
import r20_data as D
import r20_pd as PD

CANDIDATES = wrapper.OUTROOT / 'candidates' / 'candidates.json'
OUTPUT = wrapper.OUTROOT / 'teaser'


def main():
    started = time.monotonic()
    candidates = json.loads(CANDIDATES.read_text())['panel_a']
    candidates = sorted(candidates, key=lambda c: int(c['id'][1:]))
    s = D.load('artelingo')
    death, pd = PD.load_pd(s.name, s.artists)
    use = QF.usable_mask(s.keys, pd)
    positions = {key: i for i, key in enumerate(s.keys)}
    q_rows = np.array([positions[c['query']] for c in candidates])
    assert use[q_rows].all()
    gallery = np.flatnonzero(use)
    plain = s.plain()
    spaces = {'plain': plain, **{a: s.cstar(a) for a in D.ASPECTS}}
    idx, sim = RC.topk_neighbours(q_rows, gallery, spaces, plain, n=6)

    def painting(row):
        return {'key': s.keys[row], **RC.painting_record(s, row, death)}

    queries = []
    for i, candidate in enumerate(candidates):
        for space, saved in (('plain', 'plain'), ('emotion', 'ours')):
            actual = [s.keys[r] for r in idx[space][i, :5]]
            expected = [n['key'] for n in candidate[saved]]
            assert actual == expected, (candidate['id'], space, actual, expected)
        lists = {
            space: [{'cosine': round(float(cos), 4), **painting(int(row))}
                    for row, cos in zip(idx[space][i], sim[space][i])]
            for space in spaces
        }
        counts = {
            space: {a: {f'top{n}': int(RC.agreement(rows[i:i + 1, :n], q_rows[i:i + 1], s.labels[a]).sum())
                        for n in (3, 4)} for a in D.ASPECTS}
            for space, rows in idx.items()
        }
        queries.append({'id': candidate['id'], 'query': painting(int(q_rows[i])),
                        'lists': lists, 'counts': counts})

    pd_path = D.OUT / 'pd' / f'artists_{s.name}.csv'
    if not pd_path.exists():
        pd_path = D.OUT / 'smoke' / 'pd' / f'artists_{s.name}.csv'
    doc = {'header': {
        'rules': ['Existing panel (a) queries in ID order; ArtELingo test gallery, public-domain artists who died by 1955.',
                  'Exclude childe-hassam_old-lyme-bridge, the query and its near copies (plain cosine >= 0.95).',
                  'Cosine on unit-normalised rows; top six for frozen SigLIP 1 and each seed 0 c* aspect head.',
                  'Counts compare labels in the first three or four neighbours; missing labels never match.'],
        'data_paths': {'candidates': str(CANDIDATES), 'embeddings': s.sources,
                       'public_domain_table': str(pd_path), 'annotations': str(D.ANNOTATIONS),
                       'image_root': str(D.ART_IMAGES)},
        'made': datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M'), 'seed': D.SEED,
        'consistency': 'first five plain and emotion keys match panel (a) plain and ours'}, 'queries': queries}
    lines = ['| ID | Query key (shortened) | Emotion / style / genre | Plain E/S/G | Emotion E | Style S | Genre G |',
             '|---|---|---|---|---|---|---|']
    for q in queries:
        key = q['query']['key']
        short = key if len(key) <= 38 else key[:35] + '...'
        labels = ' / '.join(q['query']['labels'][a] or '—' for a in D.ASPECTS)
        counts = q['counts']
        p = '/'.join(str(counts['plain'][a]['top4']) for a in D.ASPECTS)
        own = [str(counts[a][a]['top4']) for a in D.ASPECTS]
        lines.append(f"| {q['id']} | {short} | {labels} | {p} | {' | '.join(own)} |")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'teaser_t1.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    (OUTPUT / 'teaser_t1.md').write_text('\n'.join(lines) + '\n')
    print(f'Consistency assertions passed; {len(queries)} queries; {time.monotonic() - started:.1f} seconds')


if __name__ == '__main__':
    main()
