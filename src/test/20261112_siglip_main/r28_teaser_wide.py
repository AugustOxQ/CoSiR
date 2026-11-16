"""Search the full usable ArtELingo set and draw teaser contact sheets."""
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import r28_qual as wrapper
import qf_cand as QF
import r20_cand as RC
import r20_data as D
import r20_pd as PD

OUT = wrapper.OUTROOT / 'teaser'
OLD = OUT / 'teaser_t1.json'
KEY = 'niko-pirosmani_girl-with-a-baloon'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def nice(s):
    return (s or '—').replace('_', ' ')


def wrap(draw, text, f, width):
    lines = []
    for word in text.split():
        if lines and draw.textlength(lines[-1] + ' ' + word, font=f) <= width:
            lines[-1] += ' ' + word
        else:
            lines.append(word)
    return lines


def whole(canvas, record, box):
    assert record['death_year'] is not None and record['death_year'] <= 1955, record['key']
    with Image.open(record['img']) as im:
        im = ImageOps.contain(im.convert('RGB'), (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
        canvas.paste(im, (box[0] + (box[2] - box[0] - im.width) // 2, box[1] + (box[3] - box[1] - im.height) // 2))


def crop(canvas, record, xy, size):
    assert record['death_year'] is not None and record['death_year'] <= 1955, record['key']
    with Image.open(record['img']) as im:
        im = ImageOps.fit(im.convert('RGB'), (size, size), Image.Resampling.LANCZOS, centering=(.5, .5))
        canvas.paste(im, xy)


def sheet(queries, path):
    w, panel_h, size = 1220, 564, 160
    canvas = Image.new('RGB', (w, panel_h * len(queries)), 'white')
    draw = ImageDraw.Draw(canvas)
    f, small, title = font(18), font(16), font(24, True)
    for i, q in enumerate(queries):
        y = i * panel_h
        draw.rectangle((0, y, w - 1, y + panel_h - 1), outline='#b7b7b7', width=2)
        draw.text((16, y + 10), q['id'] + '  Query', fill='black', font=title)
        whole(canvas, q['query'], (16, y + 46, 250, y + 390))
        yy = y + 400
        for a in D.ASPECTS:
            for line in wrap(draw, a.capitalize() + ': ' + nice(q['query']['labels'][a]), small, 228):
                draw.text((16, yy), line, fill='black', font=small)
                yy += 21
        for j, a in enumerate(D.ASPECTS):
            ry = y + 12 + j * 183
            label = f"{a.capitalize()}: {nice(q['query']['labels'][a])}  |  content {q['variety'][a]['content']:.4f}"
            draw.text((272, ry), label, fill='black', font=f)
            for k, e in enumerate(q['lists'][a][:4]):
                crop(canvas, e, (272 + k * 172, ry + 25), size)
    canvas.save(path)


def sheet_a10(q, path):
    w, h, size = 1980, 630, 160
    canvas = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(canvas)
    f, small, title = font(18), font(15), font(23, True)
    draw.text((15, 12), 'a10  Query', fill='black', font=title)
    whole(canvas, q['query'], (15, 45, 220, 410))
    yy = 420
    for a in D.ASPECTS:
        for line in wrap(draw, a.capitalize() + ': ' + nice(q['query']['labels'][a]), small, 200):
            draw.text((15, yy), line, fill='black', font=small)
            yy += 20
    draw.text((15, 555), 'Green frame: plan B', fill='#146b30', font=small)
    draw.text((15, 576), 'yes/no: class match', fill='black', font=small)
    for j, a in enumerate(D.ASPECTS):
        y = 12 + j * 205
        draw.text((230, y), a.capitalize() + ': ' + nice(q['query']['labels'][a]), fill='black', font=f)
        picked = {p['rank'] for p in q['pick_b'][a]}
        for k, e in enumerate(q['lists'][a]):
            x = 230 + k * 174
            same = e['labels'][a] is not None and e['labels'][a] == q['query']['labels'][a]
            draw.text((x, y + 25), f"#{k + 1}  {'yes' if same else 'no'}", fill='black', font=small)
            crop(canvas, e, (x, y + 45), size)
            if k + 1 in picked:
                draw.rectangle((x, y + 45, x + size - 1, y + 45 + size - 1), outline='#00c43b', width=5)
    canvas.save(path)


def main():
    started = time.monotonic()
    s = D.load('artelingo')
    death, pd = PD.load_pd(s.name, s.artists)
    use = QF.usable_mask(s.keys, pd)
    gallery = np.flatnonzero(use)
    full = np.all([s.labels[a] >= 0 for a in D.ASPECTS], axis=0)
    q_rows = np.flatnonzero(use & full)
    plain = s.plain()
    spaces = {'plain': plain, **{a: s.cstar(a) for a in D.ASPECTS}}
    idx, sim = RC.topk_neighbours(q_rows, gallery, spaces, plain, n=10)
    ok = np.all([RC.agreement(idx[a][:, :4], q_rows, s.labels[a]).all(axis=1) for a in D.ASPECTS], axis=0)
    passing = [j for j in np.flatnonzero(ok) if len(set(np.concatenate([idx[a][j, :4] for a in D.ASPECTS]))) == 12]

    def measures(rows, a):
        rows = list(rows)
        if len(rows) >= 2:
            matrix = plain[rows] @ plain[rows].T
            content = float(matrix[np.triu_indices(len(rows), 1)].mean())
        else:
            content = None
        labels = sum(len(set(int(x) for x in s.labels[b][rows] if x >= 0)) for b in D.ASPECTS if b != a)
        return {'content': content, 'labels': labels, 'artists': len({s.artists[r] for r in rows})}

    def variety(rows_by_a):
        v = {a: measures(rows_by_a[a], a) for a in D.ASPECTS}
        vals = [v[a]['content'] for a in D.ASPECTS]
        v['worst'] = max(vals) if all(x is not None for x in vals) else None
        v['mean'] = float(np.mean(vals)) if all(x is not None for x in vals) else None
        return v

    ranked = [(variety({a: idx[a][j, :4] for a in D.ASPECTS}), j) for j in passing]
    ranked.sort(key=lambda item: (item[0]['worst'], item[0]['mean'], s.keys[q_rows[item[1]]]))
    def painting(row):
        return {'key': s.keys[row], **RC.painting_record(s, row, death)}

    def entry(j, name, v):
        lists = {space: [{'cosine': round(float(cos), 4), **painting(int(row))}
                         for row, cos in zip(idx[space][j], sim[space][j])] for space in spaces}
        counts = {space: {a: {f'top{n}': int(RC.agreement(rows[j:j + 1, :n], q_rows[j:j + 1], s.labels[a]).sum())
                              for n in (3, 4)} for a in D.ASPECTS} for space, rows in idx.items()}
        return {'id': name, 'query': painting(int(q_rows[j])), 'lists': lists, 'counts': counts, 'variety': v}

    queries = []
    for rank, (v, j) in enumerate(ranked[:30], 1):
        q = entry(j, f'w{rank}', v)
        for a in D.ASPECTS:
            count = sum(e['labels'][a] is not None and e['labels'][a] == q['query']['labels'][a] for e in q['lists'][a][:4])
            assert count == q['counts'][a][a]['top4'] == 4
        queries.append(q)
    arow = s.keys.index(KEY)
    j = int(np.flatnonzero(q_rows == arow)[0])
    a10 = entry(j, 'a10', variety({a: idx[a][j, :4] for a in D.ASPECTS}))
    old = json.loads(OLD.read_text())
    old_a10 = next(q for q in old['queries'] if q['id'] == 'a10')
    assert [e['key'] for e in a10['lists']['emotion'][:5]] == [e['key'] for e in old_a10['lists']['emotion'][:5]]
    picked_all, picks, rows_b = set(), {}, {}
    for a in D.ASPECTS:
        candidates = [(rank, int(row)) for rank, row in enumerate(idx[a][j], 1)
                      if s.labels[a][row] >= 0 and s.labels[a][row] == s.labels[a][arow]]
        chosen, artists = [], set()
        while len(chosen) < 4:
            available = [(rank, row) for rank, row in candidates if row not in picked_all and row not in [r for _, r in chosen] and s.artists[row] not in artists]
            if not available:
                break
            if not chosen:
                pick = available[0]
            else:
                pick = min(available, key=lambda x: (float(np.max(plain[x[1]] @ plain[[r for _, r in chosen]].T)), x[0]))
            chosen.append(pick)
            artists.add(s.artists[pick[1]])
        picked_all.update(row for _, row in chosen)
        picks[a] = [{'rank': rank, 'key': s.keys[row]} for rank, row in chosen]
        rows_b[a] = [row for _, row in chosen]
    a10['pick_b'] = picks
    a10['variety_b'] = variety(rows_b)
    queries.append(a10)
    pd_path = D.OUT / 'pd' / f'artists_{s.name}.csv'
    if not pd_path.exists():
        pd_path = D.OUT / 'smoke' / 'pd' / f'artists_{s.name}.csv'
    doc = {'header': {'rules': [
        'ArtELingo test gallery: public-domain artists who died by 1955; exclude childe-hassam_old-lyme-bridge, the query and near copies (plain cosine >= 0.95).',
        'Plan A: all usable three-label queries with 4/4 own-label matches in every head and 12 distinct top-four paintings, ranked by maximum then mean within-row plain cosine (key breaks ties).',
        'Plan B (a10): from matching top-ten neighbours choose the nearest, then greedily minimise maximum plain cosine to prior row picks, excluding repeated artists and paintings picked in earlier rows, in emotion/style/genre order.',
        'Unit-normalised cosine; lists are top ten; counts compare the first three and four labels, with missing labels never matching; variety labels count distinct other-aspect classes, ignoring missing labels.'],
        'data_paths': {'previous_teaser': str(OLD), 'embeddings': s.sources, 'public_domain_table': str(pd_path), 'annotations': str(D.ANNOTATIONS), 'image_root': str(D.ART_IMAGES)},
        'made': datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M'), 'seed': D.SEED,
        'counts': {'usable_queries': len(gallery), 'three_labels': len(q_rows), 'passing': len(passing)},
        'consistency': 'a10 first five emotion keys match teaser_t1.json'}, 'queries': queries}
    (OUT / 'teaser_t3.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    lines = [f'Usable queries: {len(gallery)}; queries with three labels: {len(q_rows)}; passing queries: {len(passing)}.', '',
             '| ID | Query key (shortened) | Emotion / style / genre | Worst | Mean | Content E/S/G | Labels E/S/G | Artists E/S/G |',
             '|---|---|---|---:|---:|---|---|---|']
    def row(q, v, name):
        key = q['query']['key']
        short = key if len(key) <= 45 else key[:42] + '...'
        labels = ' / '.join(nice(q['query']['labels'][a]) for a in D.ASPECTS)
        def fmt(x): return '—' if x is None else f'{x:.4f}'
        cols = [' / '.join(fmt(v[a]['content']) for a in D.ASPECTS),
                ' / '.join(str(v[a]['labels']) for a in D.ASPECTS), ' / '.join(str(v[a]['artists']) for a in D.ASPECTS)]
        return f"| {name} | {short} | {labels} | {fmt(v['worst'])} | {fmt(v['mean'])} | {' | '.join(cols)} |"
    lines += [row(q, q['variety'], q['id']) for q in queries[:-1]]
    lines += [row(a10, a10['variety'], 'a10 true top 4'), row(a10, a10['variety_b'], 'a10 plan B')]
    lines += ['', 'Plan B picks: ' + '; '.join(f"{a}: " + ', '.join(f"#{p['rank']} {p['key']}" for p in picks[a]) + (f' (only {len(picks[a])} picked)' if len(picks[a]) < 4 else '') for a in D.ASPECTS) + '.']
    (OUT / 'teaser_t3.md').write_text('\n'.join(lines) + '\n')
    for i in range(min(4, (len(queries) - 1 + 3) // 4)):
        sheet(queries[i * 4:i * 4 + 4], OUT / f'teaser_t3_sheet_{i + 1}.png')
    sheet_a10(a10, OUT / 'teaser_t3_a10_top10.png')
    print(lines[0], f'Elapsed: {time.monotonic() - started:.1f}s; plan B counts:', {a: len(picks[a]) for a in D.ASPECTS})


if __name__ == '__main__':
    main()
