"""SemArt feasibility check for a c* test (r16 decide, 2026-11-03). CPU only; reads existing stores, writes no store.

    python sc_check.py        # writes sc_check.json and sc_nearcopy.parquet beside this file

1. Near copies: the default Qwen image of each SemArt painting against the 49,121 ArtELingo train and validation
   paintings (ArtELingo minus its 12,281 held paintings), with the same matmul as the MMArt cut
   (`mp_extract.nearest_rows`). Counts at 0.85, 0.90, 0.95, 0.99, and near-copy pairs inside SemArt.
2. Artists: each SemArt author (Web Gallery of Art form, `SURNAME, Given`) against the artists of ArtELingo's own
   42,969 train paintings (`mt_data.artelingo_train_names`, the r15 unseen-artist reference; WikiArt slug before the
   first underscore). Three match tiers, loosest last; a painting is "new" only when no tier matches.
3. Type (SemArt's genre-like label) counts of the kept set and of its new-artist part.
Sample IDs come from the store's own index.json (C6); labels join by image file name.
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'src/test/20261102_mmart_prep'))
sys.path.insert(0, str(REPO / 'src/test/20261102_mmart_test'))
import mp_extract as E  # noqa: E402
import mp_stores as S  # noqa: E402
import mt_data  # noqa: E402

SEMART = Path('/data/SSD/semart/SemArt')
STORE = Path('/data/SSD2/pre_extract/semart/qwen3vl_emb_2b')
HELD_LIST = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/held/held_paintings.json')
CUTS = (.85, .90, .95, .99)
TEST_CUT = .95
SUFFIXES = ('the younger', 'the elder', 'il giovane', 'il vecchio', 'jr', 'sr', 'ii', 'iii')
# WGA name forms that differ from WikiArt's slug (found by eye in the first pass's largest new-artist authors)
ALIASES = {'RAFFAELLO Sanzio': 'raphael', 'TIZIANO Vecellio': 'titian'}


def slug(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9?]+', '-', text).strip('-')


def variants(author):
    """Candidate WikiArt slugs for a WGA author, plus (surname tokens, first given token)."""
    a = str(author).strip()
    if ',' not in a:
        s = slug(a)
        toks = s.split('-')
        return {s, toks[0]}, toks[:1], None
    last, first = (p.strip() for p in a.split(',', 1))
    given, suffix = first, ''
    low = first.lower()
    for suf in SUFFIXES:
        if low.endswith(' ' + suf) or low == suf:
            given, suffix = first[:len(first) - len(suf)].strip(), first[len(first) - len(suf):]
            break
    out = {slug(f'{first} {last}'), slug(f'{given} {last} {suffix}'), slug(last)}
    gtok = slug(given).split('-')[0] if slug(given) else None
    return {v for v in out if v}, slug(last).split('-'), gtok


def wild(token, tokens):
    """WGA names lost some letters to '?' in the CSV (K?BKE = Kobke): '?' matches up to two letters."""
    if '?' not in token:
        return token in tokens
    rx = re.compile('^' + '.{0,2}'.join(re.escape(p) for p in token.split('?')) + '$')
    return any(rx.match(t) for t in tokens)


def artist_tiers(author, train_artists, train_tokens):
    if ALIASES.get(author) in train_artists:
        return 'exact'
    cands, surname, gtok = variants(author)
    if cands & train_artists:
        return 'exact'
    sur = [t for t in surname if t][:1]                     # first surname token (GOYA Y LUCIENTES -> goya)
    hits = [a for a in train_tokens if all(wild(t, train_tokens[a]) for t in sur)] if sur else []
    if hits and (gtok is None or any(wild(gtok, train_tokens[a]) for a in hits)):
        return 'given_surname'
    if hits:
        return 'surname_only'
    return 'none'


def main():
    meta = json.loads((STORE / 'index.json').read_text())
    files = [r['image_file'] for r in meta['images']]
    assert len(files) == len(set(files)) == meta['n_images'] == 21382
    x = np.load(STORE / 'img.npy', mmap_mode='r')
    assert x.shape == (21382, 2048)

    csv = pd.concat([pd.read_csv(SEMART / f'semart_{s}.csv', sep='\t', encoding='latin-1').assign(split=s)
                     for s in ('train', 'val', 'test')])
    csv = csv.set_index('IMAGE_FILE').loc[files]           # store order (C6)
    assert (csv['TYPE'].values == np.array([r['type'] for r in meta['images']])).all()

    names, ref = S.read_store(S.reference_folder('qwen_default'), reference=True)
    held = set(json.loads(HELD_LIST.read_text())['names'])
    assert len(names) == 61402 and len(held) == 12281 and held <= set(names)
    rows = [i for i, n in enumerate(names) if n not in held]
    trainval = [names[i] for i in rows]
    assert len(rows) == 49121
    maxima, ids = E.nearest_rows(x, ref[rows])
    nearest = [trainval[i] for i in ids]

    # near copies inside SemArt (upper triangle, self excluded)
    q = np.asarray(x, np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    inner_pairs, inner_members = 0, set()
    for start in range(0, len(q), 2048):
        s = q[start:start + 2048] @ q.T
        for r in range(s.shape[0]):
            s[r, :start + r + 1] = -1
        ii, jj = np.nonzero(s >= TEST_CUT)
        inner_pairs += len(ii)
        inner_members.update((start + ii).tolist())
        inner_members.update(jj.tolist())

    train_names = mt_data.artelingo_train_names()
    train_artists = {mt_data.artist_of(n) for n in train_names}
    train_tokens = {a: set(a.split('-')) for a in train_artists}
    authors = csv['AUTHOR'].astype(str).values
    tier_of = {a: artist_tiers(a, train_artists, train_tokens) for a in sorted(set(authors))}
    tier = np.array([tier_of[a] for a in authors])
    unknown = np.array([a.upper().startswith('UNKNOWN') for a in authors])

    kept = maxima < TEST_CUT
    new_artist = kept & (tier == 'none') & ~unknown
    near_same_artist = {}
    for lo, hi in ((.85, .90), (.90, .95), (.95, .99), (.99, 1.01)):
        band = np.nonzero((maxima >= lo) & (maxima < hi))[0]
        same = sum(variants(authors[i])[0] & {mt_data.artist_of(nearest[i])} != set() or
                   all(t in nearest[i].split('_')[0].split('-') for t in variants(authors[i])[1] if t)
                   for i in band)
        near_same_artist[f'{lo:.2f}-{min(hi, 1):.2f}'] = dict(n=int(len(band)), same_artist=int(same))

    def types(mask):
        return {k: int(v) for k, v in Counter(csv['TYPE'].values[mask]).most_common()}

    def samples(lo, hi, n=6):
        band = np.nonzero((maxima >= lo) & (maxima < hi))[0]
        pick = band[np.linspace(0, len(band) - 1, min(n, len(band))).astype(int)] if len(band) else []
        return [dict(semart=files[i], author=authors[i], title=str(csv['TITLE'].values[i]),
                     cos=round(float(maxima[i]), 4), nearest=nearest[i]) for i in pick]

    authors_by_tier = {t: sorted(a for a, v in tier_of.items() if v == t) for t in ('exact', 'given_surname',
                                                                                    'surname_only', 'none')}
    out = dict(
        n=len(files), reference_n=len(rows), train_names=len(train_names), train_artists=len(train_artists),
        nearcopy_counts={f'{c:.2f}': int((maxima >= c).sum()) for c in CUTS},
        nearest_cosine_quantiles={f'{p}': round(float(np.quantile(maxima, p)), 4) for p in (.05, .25, .5, .75, .95)},
        inner_pairs_at_cut=int(inner_pairs), inner_paintings_at_cut=len(inner_members),
        kept_at_cut=int(kept.sum()), kept_types=types(kept),
        authors=len(tier_of), author_tiers={t: len(v) for t, v in authors_by_tier.items()},
        paintings_by_tier_kept={t: int((kept & (tier == t)).sum()) for t in ('exact', 'given_surname',
                                                                             'surname_only', 'none')},
        unknown_author_kept=int((kept & unknown).sum()),
        new_artist_kept=int(new_artist.sum()), new_artist_upper_kept=int((kept & (tier != 'exact') & ~unknown).sum()),
        new_artist_artists=len(set(authors[new_artist])), new_artist_types=types(new_artist),
        nearest_same_artist_by_band=near_same_artist,
        samples={'0.85-0.90': samples(.85, .90), '0.90-0.95': samples(.90, .95), '0.95-0.99': samples(.95, .99),
                 '>=0.99': samples(.99, 1.01)},
        author_examples={t: v[:: max(1, len(v) // 15)][:15] for t, v in authors_by_tier.items()},
    )
    (HERE / 'sc_check.json').write_text(json.dumps(out, indent=2) + '\n')
    pd.DataFrame(dict(image_file=files, author=authors, type=csv['TYPE'].values, max_cos=maxima,
                      nearest_painting=nearest, artist_tier=tier)).to_parquet(HERE / 'sc_nearcopy.parquet', index=False)
    print(json.dumps({k: out[k] for k in ('nearcopy_counts', 'kept_at_cut', 'author_tiers', 'new_artist_kept',
                                          'new_artist_upper_kept', 'inner_pairs_at_cut')}))


if __name__ == '__main__':
    main()
