"""r25 inputs (CoSiR env): the held paintings' image files and caption documents, in r17's key order.

Writes, once, for the side envs (which never import r17's code):
    OUT/inputs/held_images.json  {names, images, names_sha256}          image = path relative to pb_paths.IMAGE_ROOT
    OUT/inputs/held_docs.json    {names, docs, n_captions, names_sha256}  doc = the painting's captions joined by ' '
                                                                           in annotation-file order
`names` is `ns_data.keys('artelingo', 'held')` (sorted held painting names, the order of every r17 label and c* row).
Joins are by painting name (C6: the annotation rows' own `painting` field), never by position.

    python pb_inputs.py [--force] [--split val]   (val: OUT/inputs/val_images.json only, rows in `keys('artelingo','val')`)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_paths as P  # noqa: E402


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def build(names, rows):
    """(images, docs, n_captions) per name; every name must have one image file and at least one caption."""
    want = set(names)
    image, caps = {}, {n: [] for n in names}
    for r in rows:
        n = r['painting']
        if n not in want:
            continue
        if n in image:
            assert image[n] == r['image'], f'{n}: two image files, {image[n]} and {r["image"]}'
        image[n] = r['image']
        caps[n].append(r['caption'].strip())
    missing = [n for n in names if n not in image]
    assert not missing, f'{len(missing)} held paintings without annotation rows, first {missing[:3]}'
    assert all(caps[n] for n in names)
    return [image[n] for n in names], [' '.join(caps[n]) for n in names], [len(caps[n]) for n in names]


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def write_val(force):
    """Validation image list for the VLM2Vec validation store (Table 3's read-out picks); no caption documents."""
    path = P.INPUTS / 'val_images.json'
    if path.exists() and not force:
        print(f'exists: {path}')
        return
    P.add_scorer_paths()
    import ns_data
    names = ns_data.keys(P.SET, 'val')
    assert len(names) == P.N_VAL and names == sorted(names)
    images, _, n_caps = build(names, json.loads(P.ANNOTATIONS.read_text()))
    missing_files = [i for i in images if not (P.IMAGE_ROOT / i).is_file()]
    assert not missing_files, f'{len(missing_files)} image files missing, first {missing_files[:3]}'
    sha = names_sha(names)
    _write(path, dict(names=names, images=images, names_sha256=sha))
    print(f'val paintings {len(names)}; captions {sum(n_caps)}; sha {sha[:12]}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--split', choices=('held', 'val'), default='held')
    a = ap.parse_args(argv)
    if a.split == 'val':
        return write_val(a.force)
    img_path, doc_path = P.INPUTS / 'held_images.json', P.INPUTS / 'held_docs.json'
    if img_path.exists() and doc_path.exists() and not a.force:
        print(f'exists: {img_path}, {doc_path}')
        return
    P.add_scorer_paths()
    import ns_data
    names = ns_data.keys(P.SET, P.SPLIT)
    assert len(names) == P.N_HELD and names == sorted(names)
    rows = json.loads(P.ANNOTATIONS.read_text())
    images, docs, n_caps = build(names, rows)
    missing_files = [i for i in images if not (P.IMAGE_ROOT / i).is_file()]
    assert not missing_files, f'{len(missing_files)} image files missing, first {missing_files[:3]}'
    sha = names_sha(names)
    _write(img_path, dict(names=names, images=images, names_sha256=sha))
    _write(doc_path, dict(names=names, docs=docs, n_captions=n_caps, names_sha256=sha))
    print(f'held paintings {len(names)}; captions {sum(n_caps)} (min {min(n_caps)}, max {max(n_caps)}); sha {sha[:12]}')


if __name__ == '__main__':
    main()
