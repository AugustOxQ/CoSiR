"""Round 2 teaser search (plan C): close in one aspect, far in another. Picks 4 per aspect from the top K."""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw

import r28_teaser_wide as W
from r28_teaser_wide import D, QF, RC, PD, wrapper, font, nice, wrap, whole, crop

OUT = wrapper.OUTROOT / 'teaser'
KEY = W.KEY
CONTRAST = {'emotion': 'genre', 'style': 'genre', 'genre': 'style'}
CATCH_ALL = 'something else'


def main():
    t0 = time.monotonic()
    s = D.load('artelingo')
    death, pd = PD.load_pd(s.name, s.artists)
    use = QF.usable_mask(s.keys, pd)
    gallery = np.flatnonzero(use)
    full = np.all([s.labels[a] >= 0 for a in D.ASPECTS], axis=0)
    assert CATCH_ALL in s.class_names['emotion']
    catch = s.class_names['emotion'].index(CATCH_ALL)
    q_rows = np.flatnonzero(use & full & (s.labels['emotion'] != catch))
    n_catch = int((use & full & (s.labels['emotion'] == catch)).sum())
    arow = s.keys.index(KEY)
    assert arow in q_rows
    plain = s.plain()
    spaces = {'plain': plain, **{a: s.cstar(a) for a in D.ASPECTS}}
    idx, sim = RC.topk_neighbours(q_rows, gallery, spaces, plain, n=20)

    def painting(row):
        return {'key': s.keys[row], **RC.painting_record(s, row, death)}

    def pick_query(j, K):
        qr = int(q_rows[j])
        used, shown, meas, rows_all = set(), {}, {}, {}
        for a in D.ASPECTS:
            c = CONTRAST[a]
            ql = s.labels[a][qr]
            qc = s.labels[c][qr]
            cands = [(rank, int(r)) for rank, r in enumerate(idx[a][j, :K], 1)
                     if s.labels[a][r] == ql and int(r) not in used]
            if len(cands) < 4:
                return None
            chosen = []
            while len(chosen) < 4:
                rows = [r for _, r in chosen]
                clabs = {int(s.labels[c][r]) for r in rows}
                arts = {s.artists[r] for r in rows}
                ref = plain[rows + [qr]]
                def key(x):
                    rank, r = x
                    cl = int(s.labels[c][r])
                    nc = int(cl >= 0 and cl != qc and cl not in clabs)
                    na = int(s.artists[r] not in arts)
                    return (nc, na, -float(np.max(plain[r] @ ref.T)), -rank)
                avail = [x for x in cands if x[1] not in rows]
                chosen.append(max(avail, key=key))
            rows = [r for _, r in chosen]
            used.update(rows)
            rows_all[a] = chosen
            m = plain[rows] @ plain[rows].T
            meas[a] = {'contrast': len({int(s.labels[c][r]) for r in rows if s.labels[c][r] >= 0 and s.labels[c][r] != qc}),
                       'artists': len({s.artists[r] for r in rows}),
                       'content': float(m[np.triu_indices(4, 1)].mean()),
                       'maxrank': max(rk for rk, _ in chosen)}
        return rows_all, meas

    def entry(j, name, K, res):
        rows_all, meas = res
        qr = int(q_rows[j])
        lists = {sp: [{'cosine': round(float(cs), 4), **painting(int(r))} for r, cs in zip(idx[sp][j, :K], sim[sp][j, :K])]
                 for sp in spaces}
        shown = {a: [{**painting(r), 'cosine': round(float(sim[a][j][list(idx[a][j]).index(r)]), 4), 'rank': rk}
                     for rk, r in rows_all[a]] for a in D.ASPECTS}
        with Image.open(s.image_paths[qr]) as im:
            size = list(im.size)
        # asserts
        seen = set()
        for a in D.ASPECTS:
            for e, (rk, r) in zip(shown[a], rows_all[a]):
                assert e['death_year'] is not None and e['death_year'] <= 1955
                assert s.labels[a][r] == s.labels[a][qr] and rk <= K
                assert r not in seen
                seen.add(r)
        assert painting(qr)['death_year'] <= 1955
        return {'id': name, 'K': K, 'query': painting(qr), 'lists': lists, 'shown': shown, 'measures': meas, 'image_size': size}

    queries, counts, tables = [], {}, {}
    for K in (10, 20):
        res = {j: pick_query(j, K) for j in range(len(q_rows))}
        ok = [j for j in res if res[j] is not None]
        def rk(j):
            m = res[j][1]
            v = [m[a]['contrast'] for a in D.ASPECTS]
            return (-min(v), -sum(v), max(m[a]['content'] for a in D.ASPECTS), max(m[a]['maxrank'] for a in D.ASPECTS), s.keys[q_rows[j]])
        ok.sort(key=rk)
        mins = [min(res[j][1][a]['contrast'] for a in D.ASPECTS) for j in ok]
        counts[K] = (len(q_rows), len(ok), sum(m == 4 for m in mins), sum(m >= 3 for m in mins))
        for i, j in enumerate(ok[:30], 1):
            queries.append(entry(j, f'k{K}_{i}', K, res[j]))
        ja = int(np.flatnonzero(q_rows == arow)[0])
        assert res[ja] is not None, K
        queries.append(entry(ja, f'a10_k{K}', K, res[ja]))
    pd_path = D.OUT / 'pd' / f'artists_{s.name}.csv'
    if not pd_path.exists():
        pd_path = D.OUT / 'smoke' / 'pd' / f'artists_{s.name}.csv'
    doc = {'header': {'rules': [
        'Plan C. Gallery: all usable ArtELingo test rows (public domain, died by 1955; query and near copies at plain cosine >= 0.95 excluded). Queries: usable rows with all three labels, emotion class "something else" excluded.',
        'Row candidates: top-K neighbours in that head with the query class in that aspect, minus paintings picked in earlier rows (emotion, style, genre). Contrast aspect: genre for emotion and style rows, style for the genre row.',
        'Four greedy picks by key: new contrast label (known, differs from query and from earlier picks in the row), new artist, lowest max plain cosine to picks and query, lowest rank.',
        'Query fails for a K if any row has fewer than 4 candidates. Ranking: min contrast desc, sum contrast desc, max content asc, max maxrank asc, key. K = 10 and 20.'],
        'K': [10, 20], 'made': datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M'), 'seed': D.SEED,
        'excluded_catch_all': {'class': CATCH_ALL, 'queries_excluded': n_catch},
        'data_paths': {'embeddings': s.sources, 'public_domain_table': str(pd_path), 'annotations': str(D.ANNOTATIONS), 'image_root': str(D.ART_IMAGES)}},
        'queries': queries}
    (OUT / 'teaser_t4.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')

    lines = [f'Excluded: {n_catch} usable three-label queries with emotion class "{CATCH_ALL}".', '']
    def fmt(m, k): return '/'.join(str(m[a][k]) if k != 'content' else f'{m[a][k]:.3f}' for a in D.ASPECTS)
    def row(q):
        key = q['query']['key']
        short = key if len(key) <= 40 else key[:37] + '...'
        lab = ' / '.join(nice(q['query']['labels'][a]) for a in D.ASPECTS)
        m = q['measures']
        return f"| {q['id']} | {short} | {lab} | {fmt(m,'contrast')} | {fmt(m,'content')} | {fmt(m,'maxrank')} | {fmt(m,'artists')} | {q['image_size'][0]}x{q['image_size'][1]} |"
    head = ['| ID | Query | Emotion / style / genre | Contrast E/S/G | Content E/S/G | Maxrank E/S/G | Artists E/S/G | Image px |', '|---|---|---|---|---|---|---|---|']
    for K in (10, 20):
        c = counts[K]
        lines += [f'K = {K}: queries tried {c[0]}; passing {c[1]}; passing with min contrast 4: {c[2]}; with min contrast 3: {c[3]}.', ''] + head
        lines += [row(q) for q in queries if q['K'] == K and not q['id'].startswith('a10')]
        lines += [row(q) for q in queries if q['id'] == f'a10_k{K}'] + ['']
    (OUT / 'teaser_t4.md').write_text('\n'.join(lines) + '\n')

    for K in (10, 20):
        qs = [q for q in queries if q['K'] == K and not q['id'].startswith('a10')][:12]
        for i in range(3):
            sheet(qs[i * 4:i * 4 + 4], OUT / f'teaser_t4_k{K}_sheet_{i + 1}.png')
    print(counts, 'excluded', n_catch, f'{time.monotonic() - t0:.0f}s')


def sheet(queries, path):
    w, panel_h, size = 1000, 690, 160
    canvas = Image.new('RGB', (w, panel_h * len(queries)), 'white')
    draw = ImageDraw.Draw(canvas)
    f, small, tiny, title = font(18), font(15), font(13), font(24, True)
    for i, q in enumerate(queries):
        y = i * panel_h
        draw.rectangle((0, y, w - 1, y + panel_h - 1), outline='#b7b7b7', width=2)
        draw.text((16, y + 10), q['id'] + '  Query', fill='black', font=title)
        whole(canvas, q['query'], (16, y + 46, 250, y + 400))
        yy = y + 410
        for a in D.ASPECTS:
            for line in wrap(draw, a.capitalize() + ': ' + nice(q['query']['labels'][a]), small, 228):
                draw.text((16, yy), line, fill='black', font=small)
                yy += 20
        for j, a in enumerate(D.ASPECTS):
            ry = y + 12 + j * 224
            draw.text((272, ry), f"{a.capitalize()}: {nice(q['query']['labels'][a])}", fill='black', font=f)
            c = CONTRAST[a]
            for k, e in enumerate(q['shown'][a]):
                x = 272 + k * 180
                cap = f"#{e['rank']} {nice(e['labels'][c])}"
                lines = wrap(draw, cap, tiny, 172)[:2]
                for n, line in enumerate(lines):
                    draw.text((x, ry + 26 + n * 15), line, fill='black', font=tiny)
                crop(canvas, e, (x, ry + 26 + 32), size)
    canvas.save(path)


if __name__ == '__main__':
    main()
