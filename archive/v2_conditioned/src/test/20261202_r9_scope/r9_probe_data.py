"""Development-only CLIP probes. Unit features and caption rows have separate axes.

CUB units are images; species are fit/score grouping IDs, not scoring units.
SemArt units are paintings. Labels are unit-aligned; row_to_unit expands them
(and image features) to caption rows. No test feature/annotation is materialised.
Store headers/index metadata and SHA-256 bytes necessarily cover the full stores.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path(subprocess.check_output(
    ['git', '-C', str(HERE), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
    text=True).strip()).parent
sys.path.insert(0, str(MAIN))
from src.data.cub import MIN_CERTAINTY, dev_species

CUB_ROOT = Path('/data/SSD/cub')
SEMART_ROOT = Path('/data/SSD/semart/SemArt')
STORES = {'cub': Path('/data/SSD2/pre_extract/backbone_check/cub/clip'),
          'semart': Path('/data/SSD2/pre_extract/semart/clip_b32')}
CUB_ASPECTS = {'primary_colour': 'has_primary_color', 'bill_shape': 'has_bill_shape',
               'wing_colour': 'has_wing_color'}
CUB_RULE = ('Per image and aspect, each annotation with presence=1 and certainty>=3 '
            'casts one vote for its attribute value. Accept the unique highest vote count; '
            'no qualifying votes or a tied maximum means no value (-1). '
            'In these data each image/attribute occurs once, so multiple present values tie. '
            'No species aggregation or certainty weighting.')


@dataclass
class ProbeCorpus:
    name: str
    unit_ids: np.ndarray
    groups: np.ndarray
    img_features: np.ndarray  # (units, 512), float32
    txt_features: np.ndarray  # (caption rows, 512), float32
    captions: list[str]
    row_to_unit: np.ndarray
    fit_units: np.ndarray
    scoring_units: np.ndarray
    fit_rows: np.ndarray
    scoring_rows: np.ndarray
    labels: dict[str, np.ndarray]  # unit codes; -1 means no value
    attribute_values: dict[str, list[str]]
    plan: dict


def _pairs(path):
    return [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _features(name, image_rows, text_rows, n_images, text_shape):
    store = STORES[name]
    images = np.load(store / 'img.npy', mmap_mode='r', allow_pickle=False)
    texts = np.load(store / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert images.shape == (n_images, 512) and images.dtype == np.float16
    assert texts.shape == text_shape and texts.dtype == np.float16
    # Select allowed rows BEFORE any cast, finite check or normalisation.
    img = images[image_rows].astype(np.float32)
    txt = texts[text_rows].reshape(-1, 512).astype(np.float32)
    for a in (img, txt):
        assert np.isfinite(a).all()
        assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=0.002)
    return img, txt


def _finish(name, ids, groups, img, txt, caps, row_to_unit, fit, score, labels, values, rules, details):
    assert len(set(ids)) == len(ids) == len(img)
    assert len(caps) == len(txt) == len(row_to_unit)
    assert np.array_equal(np.unique(row_to_unit), np.arange(len(ids)))
    assert not np.intersect1d(fit, score).size
    assert np.array_equal(np.sort(np.r_[fit, score]), np.arange(len(ids)))
    fit_rows = np.flatnonzero(np.isin(row_to_unit, fit))
    score_rows = np.flatnonzero(np.isin(row_to_unit, score))
    aspects = {}
    for aspect, lab in labels.items():
        assert lab.shape == (len(ids),) and np.all((lab >= -1) & (lab < len(values[aspect])))
        counts = Counter(lab[score].tolist())
        aspects[aspect] = {'rule': rules[aspect], 'values': [
            {'code': k, 'value': v, 'count': counts[k]} for k, v in enumerate(values[aspect]) if counts[k] >= 30],
            'scoring_missing_units': counts[-1]}
    splits = {key: {'units': len(u), 'rows': len(r), 'groups': len(np.unique(groups[u]))}
              for key, u, r in [('fit', fit, fit_rows), ('scoring', score, score_rows)]}
    plan = dict(unit='image' if name == 'cub' else 'painting', splits=splits,
                aspects=aspects, pairs=list(combinations(labels, 2)), **details)
    plan['feature_files'] = {f: {'path': str(STORES[name] / f), 'sha256': _sha(STORES[name] / f)}
                             for f in ('img.npy', 'txt.npy', 'index.json')}
    return ProbeCorpus(name, ids, groups, img, txt, caps, row_to_unit, fit, score,
                       fit_rows, score_rows, labels, values, plan)


def load_cub_probe():
    base = CUB_ROOT / 'CUB_200_2011'
    classes = {n: int(i) for i, n in _pairs(base / 'classes.txt')}
    train = {classes[n] for n in (CUB_ROOT / 'xlsa17/trainvalclasses.txt').read_text().split()}
    test = {classes[n] for n in (CUB_ROOT / 'xlsa17/testclasses.txt').read_text().split()}
    assert len(classes) == 200 and len(train) == 150 and len(test) == 50
    assert not train & test and train | test == set(classes.values())
    dev = set(dev_species(sorted(train)).tolist())  # defaults: 30, seed 42
    fit_species = train - dev
    assert len(fit_species) == 120 and len(dev) == 30
    # Split-identification metadata only; never call load_cub(), which loads test captions/labels.
    images = _pairs(base / 'images.txt')
    species_pairs = _pairs(base / 'image_class_labels.txt')
    assert len(images) == len(species_pairs) == 11788
    assert [i for i, _ in images] == [i for i, _ in species_pairs]
    selected = [(k, int(i), p, int(species_pairs[k][1])) for k, (i, p) in enumerate(images)
                if int(species_pairs[k][1]) in train]
    source = np.array([k for k, _, _, _ in selected])
    ids = np.array([i for _, i, _, _ in selected])
    species = np.array([s for _, _, _, s in selected])
    assert not set(species) & test and set(species) == train  # guard:cub_no_test
    fit, score = np.flatnonzero(np.isin(species, list(fit_species))), np.flatnonzero(np.isin(species, list(dev)))
    assert (len(fit), len(score), len(ids)) == (7071, 1750, 8821)
    index = json.loads((STORES['cub'] / 'index.json').read_text())
    paths = index['image_paths']
    assert len(paths) == len(set(paths)) == 11788
    caps = []
    for k, i, p, s in selected:
        assert paths[k] == str(base / 'images' / p), (i, paths[k], p)
        assert int(Path(p).parent.name.split('.')[0]) == s
        lines = (CUB_ROOT / 'captions/extracted/text_c10' / Path(p).with_suffix('.txt')).read_text().splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        assert len(lines) >= 10
        caps.extend(lines[:10])  # exact round-8 extraction order
    attr_pairs = _pairs(CUB_ROOT / 'attributes.txt')
    assert len(attr_pairs) == 312
    values, aid_to_value = {}, {}
    for aspect, group in CUB_ASPECTS.items():
        values[aspect] = []
        for aid, name in attr_pairs:
            g, v = name.split('::')
            if g == group:
                aid_to_value[int(aid)] = (aspect, len(values[aspect]))
                values[aspect].append(v)
        assert values[aspect]
    pos = {int(i): k for k, i in enumerate(ids)}
    votes = {a: np.zeros((len(ids), len(v)), dtype=np.int16) for a, v in values.items()}
    seen = set()
    annotation_widths = Counter()
    with (base / 'attributes/image_attribute_labels.txt').open() as f:
        for line in f:
            parts = line.split()
            image_id = int(parts[0])
            if image_id not in pos:
                continue  # discard test lines before parsing/retaining their attribute labels
            # CUB has both five- and six-field records; the label contract
            # is the first four fields, as in src.data.cub and round 8.
            assert len(parts) in (5, 6), parts
            annotation_widths[len(parts)] += 1
            aid, present, certainty = map(int, parts[1:4])
            assert 1 <= aid <= 312 and present in (0, 1) and 1 <= certainty <= 4
            key = (image_id, aid)
            assert key not in seen, key
            seen.add(key)
            if aid in aid_to_value and present == 1 and certainty >= MIN_CERTAINTY:
                a, v = aid_to_value[aid]
                votes[a][pos[image_id], v] += 1
    assert len(seen) == len(ids) * 312
    labels = {}
    for a, v in votes.items():
        highest = v.max(axis=1)
        labels[a] = np.where((highest > 0) & ((v == highest[:, None]).sum(axis=1) == 1), v.argmax(axis=1), -1)
    img, txt = _features('cub', source, source, 11788, (11788, 10, 512))
    assert len(txt) == 88210
    return _finish('cub', ids, species, img, txt, caps, np.repeat(np.arange(len(ids)), 10),
                   fit, score, labels, values, {a: CUB_RULE for a in labels},
                   dict(split_rule='xlsa17 trainval; dev_species(n_dev=30, seed=42) defaults',
                        fit_species=sorted(fit_species), scoring_species=sorted(dev), excluded_test_species=50,
                        join_check='All 8821 selected index paths equal images.txt paths; class IDs match path prefixes; '
                                   'ten caption rows per image in first-ten nonempty caption-file order.',
                        asserted_attribute_rows=len(seen), annotation_field_counts=dict(annotation_widths)))


def load_semart_probe():
    index = json.loads((STORES['semart'] / 'index.json').read_text())
    assert (index['dataset'], index['backbone'], index['dim'], index['dtype'], index['smoke']) == (
        'semart', 'clip_b32', 512, 'float16', False)
    assert index['n_images'] == index['n_texts'] == len(index['images']) == len(index['texts']) == 21382
    source = np.array([k for k, r in enumerate(index['images']) if r['split'] in ('train', 'val')])
    selected = [index['images'][k] for k in source]
    assert len(selected) == 20313 and all(r['split'] != 'test' for r in selected)  # guard:semart_no_test
    frames = []
    for split, expected in [('train', 19244), ('val', 1069)]:
        df = pd.read_csv(SEMART_ROOT / f'semart_{split}.csv', sep='\t', encoding='latin-1',
                         dtype=str, keep_default_na=False)
        assert list(df.columns) == ['IMAGE_FILE', 'DESCRIPTION', 'AUTHOR', 'TITLE', 'TECHNIQUE', 'DATE',
                                    'TYPE', 'SCHOOL', 'TIMEFRAME']
        assert len(df) == expected
        df['split'] = split
        frames.append(df)
    metadata = pd.concat(frames, ignore_index=True)
    assert metadata.IMAGE_FILE.is_unique
    metadata = metadata.set_index('IMAGE_FILE').loc[[r['image_file'] for r in selected]]
    spec = importlib.util.spec_from_file_location('r9_probe_extract', MAIN / 'scripts/extract_features.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    caps = []
    for k, r, (_, m) in zip(source, selected, metadata.iterrows()):
        assert index['texts'][k] == {'img': int(k)}, (k, index['texts'][k])
        assert r['split'] == m['split']
        for a in ('type', 'school', 'timeframe'):
            assert r[a] == m[a.upper()], (k, a)
        caps.append(module.scrub_semart(m.DESCRIPTION, m.AUTHOR, m.TITLE, m.DATE))
    labels, values, rules = {}, {}, {}
    for a in ('type', 'school', 'timeframe'):
        raw = metadata[a.upper()].to_numpy()
        values[a] = sorted(set(raw) - {''})
        mapping = {v: k for k, v in enumerate(values[a])}
        labels[a] = np.array([mapping.get(v, -1) for v in raw])
        rules[a] = f'Metadata {a.upper()} verbatim, no parsing or binning; empty string means no value (-1).'
    fit = np.flatnonzero(metadata['split'].to_numpy() == 'train')
    score = np.flatnonzero(metadata['split'].to_numpy() == 'val')
    assert (len(fit), len(score)) == (19244, 1069)
    img, txt = _features('semart', source, source, 21382, (21382, 512))
    ids = metadata.index.to_numpy()
    return _finish('semart', ids, ids.copy(), img, txt, caps, np.arange(len(ids)), fit, score,
                   labels, values, rules, dict(split_rule='official train fit; official val scoring',
                   join_check='All 20313 selected IMAGE_FILE IDs unique; each caption index img equals its source '
                              'image row; split, TYPE, SCHOOL and TIMEFRAME agree with metadata by IMAGE_FILE.'))


def load_probe_data(dataset):
    assert dataset in ('cub', 'semart'), dataset
    return load_cub_probe() if dataset == 'cub' else load_semart_probe()
