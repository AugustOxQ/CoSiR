"""Round 4: zero-shot person / setting tags, then the emotion row must contrast in content."""
import json
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw

import r28_teaser_picks as P
from r28_teaser_picks import D, QF, RC, PD, wrapper, nice, CONTRAST, CATCH_ALL
from r28_teaser_wide import font, wrap, whole, crop

OUT = wrapper.OUTROOT / 'teaser'
K = 20
PROMPTS = {
    'person': ['a painting of a person', 'a painting of people', 'a portrait of a person'],
    'noperson': ['a painting of a landscape', 'a painting of a still life', 'a painting of buildings',
                 'a painting of animals', 'a painting of the sea'],
    'indoor': ['a painting of an indoor scene', 'a painting of a room interior'],
    'outdoor': ['a painting of an outdoor scene', 'a painting of a landscape', 'a painting of a street'],
}
DEAD = (40, 60)  # percentiles of the margin over usable rows; inside = unknown
TAGS = OUT / 'teaser_t6_tags.npz'


def text_embeddings():
    import torch
    from transformers import AutoModel, AutoProcessor
    repo = 'google/siglip-so400m-patch14-384'
    p = AutoProcessor.from_pretrained(repo)
    m = AutoModel.from_pretrained(repo, torch_dtype=torch.float32).eval()
    out = {}
    for name, ps in PROMPTS.items():
        t = p.tokenizer([x.lower() for x in ps], return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        with torch.no_grad():
            f = m.get_text_features(**t)
            f = getattr(f, 'pooler_output', f)
        out[name] = torch.nn.functional.normalize(f, dim=-1).numpy()
    return out


def make_tags(s, plain, use):
    te = text_embeddings()
    sc = {n: plain @ e.T for n, e in te.items()}   # [N, n_prompts]
    pm = sc['person'].max(1) - sc['noperson'].max(1)
    sm = sc['indoor'].max(1) - sc['outdoor'].max(1)
    cut = {}
    tags = {}
    for name, m in (('person', pm), ('setting', sm)):
        lo, hi = np.percentile(m[use], DEAD)
        cut[name] = (float(lo), float(hi))
        tags[name] = np.where(m > hi, 1, np.where(m < lo, -1, 0)).astype(np.int8)  # person/indoor = 1; none/outdoor = -1; 0 unknown
    np.savez(TAGS, keys=np.array(s.keys), person_margin=pm, setting_margin=sm, person=tags['person'], setting=tags['setting'],
             dead_zone_percentiles=np.array(DEAD), cut_person=np.array(cut['person']), cut_setting=np.array(cut['setting']))
    return pm, sm, tags['person'], tags['setting'], cut


def tagstr(p, st):
    return ('P' if p == 1 else 'N' if p == -1 else '?') + ' ' + ('I' if st == 1 else 'O' if st == -1 else '?')


def load_ok(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def tagcheck(s, use, tp, ts, path):
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero(use), 24, replace=False)
    canvas = Image.new('RGB', (6 * 200, 4 * 225), 'white')
    d = ImageDraw.Draw(canvas)
    f = font(15, True)
    for n, r in enumerate(rows):
        x, y = (n % 6) * 200, (n // 6) * 225
        try:
            crop(canvas, {'img': s.image_paths[r], 'death_year': 0}, (x + 10, y + 5), 180)
        except Exception:
            pass
        d.text((x + 10, y + 190), tagstr(tp[r], ts[r]) + f'  #{n + 1}', fill='black', font=f)
    canvas.save(path)


def sheet6(queries, path):
    w, panel_h, size = 1000, 690, 160
    canvas = Image.new('RGB', (w, panel_h * len(queries)), 'white')
    draw = ImageDraw.Draw(canvas)
    f, small, tiny, title = font(18), font(15), font(13), font(24, True)
    for i, q in enumerate(queries):
        y = i * panel_h
        draw.rectangle((0, y, w - 1, y + panel_h - 1), outline='#b7b7b7', width=2)
        draw.text((16, y + 10), q['id'] + '  Query  ' + tagstr(*q['tags']), fill='black', font=title)
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
                cap = f"#{e['rank']} {nice(e['labels'][c])} {tagstr(*e['tags'])}"
                for n, line in enumerate(wrap(draw, cap, tiny, 172)[:2]):
                    draw.text((x, ry + 26 + n * 15), line, fill='black', font=tiny)
                crop(canvas, e, (x, ry + 26 + 32), size)
    canvas.save(path)


def main():
    t0 = time.monotonic()
    s = D.load('artelingo')
    death, pd = PD.load_pd(s.name, s.artists)
    use = QF.usable_mask(s.keys, pd)
    gallery = np.flatnonzero(use)
    plain = s.plain()
    if not TAGS.exists():
        make_tags(s, plain, use)
    tg = np.load(TAGS)
    assert list(tg['keys']) == list(s.keys)
    tp, ts = tg['person'], tg['setting']
    full = np.all([s.labels[a] >= 0 for a in D.ASPECTS], axis=0)
    catch = s.class_names['emotion'].index(CATCH_ALL)
    q_rows = np.flatnonzero(use & full & (s.labels['emotion'] != catch) & (tp != 0) & (ts != 0))
    spaces = {'plain': plain, **{a: s.cstar(a) for a in D.ASPECTS}}
    idx, sim = RC.topk_neighbours(q_rows, gallery, spaces, plain, n=K)
    bad = set()

    def painting(row):
        return {'key': s.keys[row], **RC.painting_record(s, row, death), 'tags': [int(tp[row]), int(ts[row])]}

    def pick_query(j):
        qr = int(q_rows[j])
        if qr in bad:
            return None
        used, rows_all, meas, share = set(), {}, {}, {}
        for a in D.ASPECTS:
            c = CONTRAST[a]
            ql, qc = s.labels[a][qr], s.labels[c][qr]
            share[a] = int((s.labels[a][idx[a][j]] == ql).sum())
            cands = [(rank, int(r)) for rank, r in enumerate(idx[a][j], 1)
                     if s.labels[a][r] == ql and int(r) not in used and int(r) not in bad]
            chosen = []
            opp_p = lambda r: tp[r] != 0 and tp[r] != tp[qr]
            opp_s = lambda r: ts[r] != 0 and ts[r] != ts[qr]
            if a == 'emotion':
                if not any(opp_p(r) for _, r in cands) or not any(opp_s(r) for _, r in cands):
                    return None
                both = [x for x in cands if opp_p(x[1]) and opp_s(x[1])]
                if both:
                    chosen.append(both[0])
                else:
                    chosen.append(next(x for x in cands if opp_p(x[1])))
                    chosen.append(next(x for x in cands if opp_s(x[1])))
                if sum(opp_p(r) for _, r in chosen) < 2:
                    rest = [x for x in cands if opp_p(x[1]) and x not in chosen]
                    if rest:
                        chosen.append(rest[0])
            if len(cands) < 4:
                return None
            while len(chosen) < 4:
                rows = [r for _, r in chosen]
                clabs = {int(s.labels[c][r]) for r in rows}
                arts = {s.artists[r] for r in rows}
                ref = plain[rows + [qr]]
                def key(x):
                    rank, r = x
                    cl = int(s.labels[c][r])
                    return (int(cl >= 0 and cl != qc and cl not in clabs), int(s.artists[r] not in arts),
                            -float(np.max(plain[r] @ ref.T)), -rank)
                chosen.append(max([x for x in cands if x not in chosen], key=key))
            rows = [r for _, r in chosen]
            used.update(rows)
            rows_all[a] = chosen
            m = plain[rows] @ plain[rows].T
            meas[a] = {'contrast': len({int(s.labels[c][r]) for r in rows if s.labels[c][r] >= 0 and s.labels[c][r] != qc}),
                       'artists': len({s.artists[r] for r in rows}), 'content': float(m[np.triu_indices(4, 1)].mean()),
                       'maxrank': max(rk for rk, _ in chosen)}
        er = [r for _, r in rows_all['emotion']]
        meas['emotion']['opposite'] = int(sum(bool(opp_p(r) or opp_s(r)) for r in er))
        meas['emotion']['opposite_person'] = int(sum(bool(opp_p(r)) for r in er))
        meas['emotion']['opposite_setting'] = int(sum(bool(opp_s(r)) for r in er))
        return rows_all, meas, share

    def rk(j):
        m = res[j][1]
        return (-m['emotion']['opposite'], -min(m['style']['contrast'], m['genre']['contrast']), -res[j][2]['emotion'],
                max(m[a]['maxrank'] for a in D.ASPECTS), s.keys[q_rows[j]])

    for it in range(15):
        res = {j: pick_query(j) for j in range(len(q_rows))}
        ok = sorted([j for j in res if res[j] is not None], key=rk)
        new = set()
        for j in ok[:30]:
            for r in [int(q_rows[j])] + [r for a in D.ASPECTS for _, r in res[j][0][a]]:
                if r not in bad and not load_ok(s.image_paths[r]):
                    new.add(r)
        if not new:
            break
        bad |= new
    print('image-load iterations', it + 1, 'bad rows', len(bad), [s.keys[r] for r in bad])

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
        return {'id': name, 'K': K, 'query': painting(qr), 'tags': [int(tp[qr]), int(ts[qr])], 'lists': lists, 'shown': shown,
                'measures': meas, 'class_share': share, 'image_size': size}

    queries = [entry(j, f't6_{i}') for i, j in enumerate(ok[:30], 1)]
    pd_path = D.OUT / 'pd' / f'artists_{s.name}.csv'
    if not pd_path.exists():
        pd_path = D.OUT / 'smoke' / 'pd' / f'artists_{s.name}.csv'
    doc = {'header': {'rules': [
        'Round 4. Zero-shot person (person=1, none=-1) and setting (indoor=1, outdoor=-1) tags from SigLIP 1 text prompts against the plain image store; 0 = unknown, margin inside the 40th to 60th percentile over usable rows.',
        'Queries: usable, three labels, emotion not "something else", both tags known. K = 20 in all rows.',
        'Emotion row: from the 20 nearest with the query emotion; needs an opposite-person and an opposite-setting candidate. Pick lowest-rank opposite in both, else lowest-rank opposite-person then lowest-rank opposite-setting; then top up to 2 opposite-person; fill by plan C key.',
        'Style and genre rows as plan C (K = 20, contrast genre / style), no painting twice. Paintings that PIL cannot load fully are excluded.',
        'Ranking: emotion picks opposite in person or setting desc, min(style, genre contrast) desc, emotion class share desc, max rank asc, key.'],
        'prompts': PROMPTS, 'dead_zone_percentiles': list(DEAD), 'dead_zone_cuts': {'person': tg['cut_person'].tolist(), 'setting': tg['cut_setting'].tolist()},
        'tags_file': str(TAGS), 'excluded_unloadable': sorted(s.keys[r] for r in bad),
        'K': [K], 'made': datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M'), 'seed': D.SEED,
        'counts': {'tried': len(q_rows), 'passing': len(ok)},
        'data_paths': {'embeddings': s.sources, 'public_domain_table': str(pd_path), 'annotations': str(D.ANNOTATIONS), 'image_root': str(D.ART_IMAGES)}},
        'queries': queries}
    (OUT / 'teaser_t6.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    lines = [f'K = 20: queries tried {len(q_rows)}; passing {len(ok)}.', '',
             '| ID | Query | Emotion / style / genre | Query tags | Emotion-row tags | Opp | Contrast S/G | Maxrank E/S/G | Share E/S/G | Image px |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for q in queries:
        key = q['query']['key']
        m = q['measures']
        lines.append(f"| {q['id']} | {key if len(key) <= 40 else key[:37] + '...'} | {' / '.join(nice(q['query']['labels'][a]) for a in D.ASPECTS)} | "
                     f"{tagstr(*q['tags'])} | {', '.join(tagstr(*e['tags']) for e in q['shown']['emotion'])} | {m['emotion']['opposite']} | "
                     f"{m['style']['contrast']}/{m['genre']['contrast']} | {'/'.join(str(m[a]['maxrank']) for a in D.ASPECTS)} | "
                     f"{'/'.join(str(q['class_share'][a]) for a in D.ASPECTS)} | {q['image_size'][0]}x{q['image_size'][1]} |")
    (OUT / 'teaser_t6.md').write_text('\n'.join(lines) + '\n')
    for i in range(3):
        sheet6(queries[i * 4:i * 4 + 4], OUT / f'teaser_t6_sheet_{i + 1}.png')
    print(len(q_rows), len(ok), f'{time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
