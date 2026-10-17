"""Freeze the earlier check-1 sample and an independently seeded image subset."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EARLIER = HERE.parent / '20261017_aspect_premise_checks'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EARLIER))
SAMPLE = HERE / 'sample.npz'


def load_frozen():
    with np.load(SAMPLE, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    assert (len(s['sample_ids']), len(s['first']), len(s['subset_pid'])) == (99999, 19889, 5000)
    assert len(np.unique(s['sample_ids'])) == 99999
    assert len(np.unique(s['subset_pid'])) == 5000
    assert np.array_equal(s['all_painting_ids'][s['pid']], s['painting_ids'])
    assert np.array_equal(s['all_painting_ids'][s['subset_pid']], s['subset_painting_ids'])
    return s


def main():
    if SAMPLE.exists():
        s = load_frozen()
    else:
        from apc_data import load_sample
        from scripts.extract_features import adapter_artelingo_full
        s0 = load_sample(min_caps=3)
        spec = adapter_artelingo_full(sample_ids=s0['sample_ids'], expected=None)
        pid = s0['pid']
        first = np.full(s0['n_paint'], -1)
        first[pid[::-1]] = np.arange(len(pid))[::-1]
        paintings = np.asarray(s0['d'].paintings[s0['rows']], dtype=str)
        image_index = np.asarray([m['img'] for m in spec.text_meta])
        assert np.array_equal(image_index, image_index[first][pid])
        assert len(np.unique(image_index[first])) == len(first)
        assert np.array_equal(paintings, np.asarray([m['painting'] for m in spec.text_meta]))
        subset = np.random.default_rng(0).choice(len(first), 5000, replace=False)
        s = dict(sample_ids=s0['sample_ids'], rows=s0['rows'], pid=pid,
                 painting_ids=paintings, all_painting_ids=paintings[first], first=first,
                 subset_pid=subset, subset_painting_ids=paintings[first[subset]],
                 subset_image_indices=image_index[first[subset]],
                 emo=s0['emo'], gen=s0['gen'], sty=s0['sty'])
        np.savez(SAMPLE, **s)
    assert (len(s['sample_ids']), len(s['first']), len(s['subset_pid'])) == (99999, 19889, 5000)
    print(f"{len(s['sample_ids']):,} captions, {len(s['first']):,} paintings, {len(s['subset_pid']):,} subset", flush=True)


if __name__ == '__main__':
    main()
