"""Round 3 teaser search: emotion row must contain a portrait; K = 20 in all rows."""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

import r28_teaser_picks as P
from r28_teaser_picks import D, QF, RC, PD, wrapper, nice, CONTRAST, CATCH_ALL, sheet

OUT = wrapper.OUTROOT / 'teaser'
K = 20


def main():
    t0 = time.monotonic()
    s = D.load('artelingo')
    death, pd = PD.load_pd(s.name, s.artists)
    use = QF.usable_mask(s.keys, pd)
    gallery = np.flatnonzero(use)
    full = np.all([s.labels[a] >= 0 for a in D.ASPECTS], axis=0)
    catch = s.class_names['emotion'].index(CATCH_ALL)
    port = s.class_names['genre'].index('portrait')
    q_rows = np.flatnonzero(use & full & (s.labels['emotion'] != catch) & (s.labels['genre'] != port))
    plain = s.plain()
    spaces = {'plain': plain, **{a: s.cstar(a) for a in D.ASPECTS}}
    idx, sim = RC.topk_neighbours(q_rows, gallery, spaces, plain, n=K)

    def painting(row):
        return {'key': s.keys[row], **RC.painting_record(s, row, death)}

    def pick_query(j):
        qr = int(q_rows[j])
        used, rows_all, meas, share = set(), {}, {}, {}
        for a in D.ASPECTS:
            c = CONTRAST[a]
            ql, qc = s.labels[a][qr], s.labels[c][qr]
            share[a] = int((s.labels[a][idx[a][j]] == ql).sum())
            cands = [(rank, int(r)) for rank, r in enumerate(idx[a][j], 1) if s.labels[a][r] == ql and int(r) not in used]
            chosen = []
            if a == 'emotion':
                ports = [x for x in cands if s.labels['genre'][x[1]] == port]
                if not ports:
                    return None
                chosen.append(ports[0])
            if len(cands) < 4:
                return None
            while len(chosen) < 4:
                rows = [r for _, r in chosen]
                clabs = {int(s.labels[c][r]) for r in rows}
                arts = {s.artists[r] for r in rows}
                ref = plain[rows + [qr]] if rows else plain[[qr]]
                def key(x):
                    rank, r = x
                    cl = int(s.labels[c][r])
                    return (int(cl >= 0 and cl != qc and cl not in clabs), int(s.artists[r] not in arts),
                            -float(np.max(plain[r] @ ref.T)), -rank)
                chosen.append(max([x for x in cands if x[1] not in rows], key=key))
            rows = [r for _, r in chosen]
            used.update(rows)
            rows_all[a] = chosen
            m = plain[rows] @ plain[rows].T
            meas[a] = {'contrast': len({int(s.labels[c][r]) for r in rows if s.labels[c][r] >= 0 and s.labels[c][r] != qc}),
                       'artists': len({s.artists[r] for r in rows}), 'content': float(m[np.triu_indices(4, 1)].mean()),
                       'maxrank': max(rk for rk, _ in chosen)}
        meas['emotion']['genres'] = len({int(s.labels['genre'][r]) for _, r in rows_all['emotion'] if s.labels['genre'][r] >= 0})
        return rows_all, meas, share

    res = {j: pick_query(j) for j in range(len(q_rows))}
    ok = [j for j in res if res[j] is not None]
    def rk(j):
        m = res[j][1]
        return (-m['emotion']['genres'], -min(m['style']['contrast'], m['genre']['contrast']),
                max(m[a]['maxrank'] for a in D.ASPECTS), max(m[a]['content'] for a in D.ASPECTS), s.keys[q_rows[j]])
    ok.sort(key=rk)

    def entry(j, name):
        rows_all, meas, share = res[j]
        qr = int(q_rows[j])
        lists = {sp: [{'cosine': round(float(cs), 4), **painting(int(r))} for r, cs in zip(idx[sp][j], sim[sp][j])] for sp in spaces}
        shown = {a: [{**painting(r), 'cosine': round(float(sim[a][j][list(idx[a][j]).index(r)]), 4), 'rank': rkk}
                     for rkk, r in rows_all[a]] for a in D.ASPECTS}
        with Image.open(s.image_paths[qr]) as im:
            size = list(im.size)
        seen = set()
        for a in D.ASPECTS:
            for e, (rkk, r) in zip(shown[a], rows_all[a]):
                assert e['death_year'] is not None and e['death_year'] <= 1955
                assert s.labels[a][r] == s.labels[a][qr] and rkk <= K and r not in seen
                seen.add(r)
        assert any(e['labels']['genre'] == 'portrait' for e in shown['emotion'])
        assert shown['emotion'][0]['labels']['genre'] == 'portrait'
        return {'id': name, 'K': K, 'query': painting(qr), 'lists': lists, 'shown': shown, 'measures': meas,
                'class_share': share, 'image_size': size}

    queries = [entry(j, f't5_{i}') for i, j in enumerate(ok[:30], 1)]
    pd_path = D.OUT / 'pd' / f'artists_{s.name}.csv'
    if not pd_path.exists():
        pd_path = D.OUT / 'smoke' / 'pd' / f'artists_{s.name}.csv'
    doc = {'header': {'rules': [
        'Round 3. Gallery and queries as plan C (usable rows; three labels; emotion not "something else"), plus query genre not portrait. K = 20 in all rows.',
        'Emotion row: candidates are the 20 nearest in the emotion head with the query emotion; needs a portrait candidate (lowest rank portrait picked first), then 3 more by plan C key (new genre, new artist, low max plain cosine, low rank).',
        'Style and genre rows as plan C with K = 20 (contrast genre / style), no painting twice.',
        'Ranking: distinct genres in emotion row desc, min(style, genre contrast) desc, max rank over rows asc, max content asc, key.',
        'class_share = how many of the 20 in that head carry the query class.'],
        'K': [K], 'made': datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M'), 'seed': D.SEED,
        'counts': {'tried': len(q_rows), 'passing': len(ok)},
        'data_paths': {'embeddings': s.sources, 'public_domain_table': str(pd_path), 'annotations': str(D.ANNOTATIONS), 'image_root': str(D.ART_IMAGES)}},
        'queries': queries}
    (OUT / 'teaser_t5.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    lines = [f'K = 20: queries tried {len(q_rows)}; passing {len(ok)}.', '',
             '| ID | Query | Emotion / style / genre | Emotion-row genres | Contrast S/G | Maxrank E/S/G | Content E/S/G | Artists E/S/G | Share E/S/G | Image px |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for q in queries:
        key = q['query']['key']
        m = q['measures']
        g = ', '.join(nice(e['labels']['genre']) for e in q['shown']['emotion'])
        f3 = lambda k, fm='{}': '/'.join(fm.format(m[a][k]) for a in D.ASPECTS)
        lines.append(f"| {q['id']} | {key if len(key) <= 40 else key[:37] + '...'} | {' / '.join(nice(q['query']['labels'][a]) for a in D.ASPECTS)} | {g} | "
                     f"{m['style']['contrast']}/{m['genre']['contrast']} | {f3('maxrank')} | {f3('content', '{:.3f}')} | {f3('artists')} | "
                     f"{'/'.join(str(q['class_share'][a]) for a in D.ASPECTS)} | {q['image_size'][0]}x{q['image_size'][1]} |")
    (OUT / 'teaser_t5.md').write_text('\n'.join(lines) + '\n')
    for i in range(3):
        sheet(queries[i * 4:i * 4 + 4], OUT / f'teaser_t5_sheet_{i + 1}.png')
    print(len(q_rows), len(ok), f'{time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
