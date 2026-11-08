"""Step `pd`: public-domain table from Wikidata. A painting counts as public domain when its artist's death year is
1955 or earlier. Humans (Q5) with a WikiArt ID (P6002) and a date of death (P570, latest year when several values).
The artist slug of a painting is matched to the WikiArt ID after the same normalisation on both sides
(`r20_data.norm_slug`). Unmatched, or no death date -> not public domain.
"""
import csv
import json
import time
import urllib.parse
import urllib.request

import numpy as np

import r20_data as D

ENDPOINT = 'https://query.wikidata.org/sparql'
USER_AGENT = 'cosir-figure-check/0.1'
PD_LAST_DEATH_YEAR = 1955
PAGE = 20000
QUERY = ('SELECT ?p ?id ?d WHERE { ?p wdt:P31 wd:Q5 ; wdt:P6002 ?id . OPTIONAL { ?p wdt:P570 ?d } } '
         'ORDER BY ?id ?p ?d LIMIT %d OFFSET %d')


def fetch_page(limit, offset, retries=4):
    url = ENDPOINT + '?' + urllib.parse.urlencode({'query': QUERY % (limit, offset), 'format': 'json'})
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/sparql-results+json'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode('utf-8'))['results']['bindings']
        except Exception as e:  # noqa: BLE001 (network errors vary; retry then raise)
            if attempt == retries - 1:
                raise
            # the endpoint can answer 429 (1 request per minute): wait a full minute then
            time.sleep(65 if getattr(e, 'code', None) == 429 else 5 * (attempt + 1))


def fetch_all(page=PAGE):
    rows, offset = [], 0
    while True:
        b = fetch_page(page, offset)
        rows += b
        if len(b) < page:
            return rows
        offset += page


def year_of(value):
    """Year of an xsd:dateTime string like '1983-01-01T00:00:00Z' or '-0500-01-01T00:00:00Z'; None if unparsable."""
    v = str(value).strip()
    neg = v.startswith('-')
    head = v[1:] if neg else v
    try:
        y = int(head.split('-', 1)[0])
    except ValueError:
        return None
    return -y if neg else y


def death_years(bindings):
    """{normalised WikiArt id: latest death year or None}, plus {normalised id: [raw ids]}."""
    years, raw = {}, {}
    for b in bindings:
        rid = b['id']['value']
        nid = D.norm_slug(rid)
        raw.setdefault(nid, set()).add(rid)
        y = None
        d = b.get('d')
        if d is not None and d.get('type') == 'literal':
            y = year_of(d['value'])
        cur = years.get(nid)
        years[nid] = y if cur is None else (cur if y is None else max(cur, y))
    return years, {k: sorted(v) for k, v in raw.items()}


def dotdash(s):
    """Fallback key for slugs whose dots stand for dashes on the other side ('a.y.-jackson' vs 'a-y-jackson')."""
    return D.norm_slug(str(s).replace('.', '-'))


def artist_table(slugs, years, raw_ids, last=PD_LAST_DEATH_YEAR):
    """Per-artist rows for the given slugs: {slug, norm, matched_id, death_year, public_domain, match}. The brief's rule
    first (`norm`: dots dropped on both sides); an artist still unmatched gets a second try with dots read as dashes on
    both sides (`dotdash`), marked in `match`."""
    alt = {}
    for k, ids in raw_ids.items():
        for rid in ids:
            alt.setdefault(dotdash(rid), k)
    rows = []
    for s in sorted(set(slugs)):
        n, how = D.norm_slug(s), 'norm'
        if n not in years and dotdash(s) in alt:
            n, how = alt[dotdash(s)], 'dotdash'
        matched = n in years
        y = years.get(n) if matched else None
        rows.append(dict(slug=s, norm=D.norm_slug(s), matched_id=';'.join(raw_ids[n]) if matched else None,
                         death_year=y, public_domain=bool(y is not None and y <= last),
                         match=how if matched else None))
    return rows


def run(smoke=False):
    out = D.OUT / ('smoke' if smoke else '') / 'pd'
    out.mkdir(parents=True, exist_ok=True)
    raw_file = out / 'wikidata_raw.json'
    full_raw = D.OUT / 'pd' / 'wikidata_raw.json'
    if smoke and full_raw.exists():             # no new query (the endpoint allows 1 per minute): the first 2000 saved rows
        bindings = json.loads(full_raw.read_text())['bindings'][:2000]
    elif smoke:                                 # one small page, enough to test the code path
        bindings = fetch_page(2000, 0)
    elif raw_file.exists():                     # the saved query result is reused (delete it to query again)
        bindings = json.loads(raw_file.read_text())['bindings']
    else:
        bindings = fetch_all()
    if smoke or not raw_file.exists():
        raw_file.write_text(json.dumps(dict(endpoint=ENDPOINT, query=QUERY, page=PAGE, fetched=time.strftime('%F %H:%M'),
                                            n=len(bindings), bindings=bindings)))
    years, raw_ids = death_years(bindings)
    report = dict(wikidata_rows=len(bindings), wikidata_ids=len(years),
                  ids_with_death_year=sum(v is not None for v in years.values()))
    for name in D.SETS:
        s = D.load(name)
        rows = artist_table(s.artists, years, raw_ids)
        by = {r['slug']: r for r in rows}
        pd_mask = np.array([by[a]['public_domain'] for a in s.artists])
        fields = ['slug', 'norm', 'matched_id', 'death_year', 'public_domain', 'match']
        with open(out / f'artists_{name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(fields)
            for r in rows:
                w.writerow(['' if r[k] is None else r[k] for k in fields])
        report[name] = dict(artists=len(rows), matched=sum(r['matched_id'] is not None for r in rows),
                            matched_by_dotdash_fallback=sum(r['match'] == 'dotdash' for r in rows),
                            with_death_year=sum(r['death_year'] is not None for r in rows),
                            public_domain_artists=sum(r['public_domain'] for r in rows),
                            paintings=len(s), public_domain_paintings=int(pd_mask.sum()))
    (out / 'pd_summary.json').write_text(json.dumps(report, indent=1))
    return report


def load_pd(name, artists):
    """(death year as float with nan, public-domain bool) per entry of `artists`, from the saved per-artist table (the
    full table when it exists, else the smoke one)."""
    path = D.OUT / 'pd' / f'artists_{name}.csv'
    if not path.exists():
        path = D.OUT / 'smoke' / 'pd' / f'artists_{name}.csv'
    by = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            by[r['slug']] = (float(r['death_year']) if r['death_year'] else float('nan'), r['public_domain'] == 'True')
    yrs = np.array([by[a][0] for a in artists])
    return yrs, np.array([by[a][1] for a in artists])
