"""Shared k20 image/caption scorer and image-side comparator back-fill (CPU)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (adjusted_mutual_info_score, v_measure_score,
                             adjusted_rand_score, fowlkes_mallows_score)
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot',
               '20261018_aspect_graph_sheet', '20261018_percept_core'):
    sys.path.insert(0, str(HERE.parent / folder))
from itp_data import atomic_json, fingerprint, load_frozen, prompted
from pbp_graphs import label_vectors, sha_file
from pav_score import partitions, summary
from apc_data import l2
from pc_data import image_features
import pc_score

# Earlier-round code comes from this worktree; ignored artifacts only from main.
MAIN_TEST = Path('/project/CoSiR/src/test')
PC = MAIN_TEST / '20261018_percept_core/outputs'
GENRE = MAIN_TEST / '20261018_aspect_graph_sheet/outputs_fu'
CEILING = MAIN_TEST / '20261019_emotion_ceiling/outputs/ec_results.json'
REPORT = Path('/project/CoSiR/docs/user_read/2026-10-19_percept_core_k40.md')
OUTPUTS = HERE / 'outputs'
ASPECTS = ('emotion', 'style', 'genre')
PROMPTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
BEST = dict(emotion='P3', style='P2', genre='P3')
STAGE1 = dict(emotion='Fres1', style='Fres05', genre='R0o')
METRICS = dict(ami=adjusted_mutual_info_score, v=v_measure_score,
               ari=adjusted_rand_score, fmi=fowlkes_mallows_score)
SCORING = 'pav k20 seeds012 n_init3 batch4096; float32; y>=0; population sd; v1'


def array_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def check_view(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and x.shape[0] == 5000 and x.shape[1] > 0
    assert np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < .01
    return x


def measures(x, labels, cache_dir=None):
    """One set of three partitions, shared by all metrics and labels.

    Optional disk cache binds array bytes, labels, protocol and sklearn version.
    Without cache_dir this is the two-argument scorer imported by the runner.
    """
    x = check_view(x)
    assert labels
    for y in labels.values():
        assert y.shape == (5000,) and np.issubdtype(y.dtype, np.integer)
        assert (y >= 0).any()
    identity = dict(source_sha256=array_sha(x), scoring=SCORING, sklearn=sklearn.__version__,
                    labels={k: array_sha(v) for k, v in sorted(labels.items())})
    path = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(cache_dir) / f'{key}.json'
        if path.exists():
            r = json.loads(path.read_text())
            assert r['identity'] == identity, 'score cache identity mismatch'
            assert set(r['scores']) == set(labels)
            for scores in r['scores'].values():
                assert set(scores) == set(METRICS)
                assert all(set(v) == {'mean', 'sd'} and np.isfinite(list(v.values())).all()
                           and v['sd'] >= 0 for v in scores.values())
            return r['scores']
    with threadpool_limits(8):
        ls = partitions(x, 20)
    assert len(ls) == 3 and all(lab.shape == (5000,) for lab in ls)
    scores = {name: {metric: summary([fn(y[y >= 0], lab[y >= 0]) for lab in ls])
                     for metric, fn in METRICS.items()} for name, y in labels.items()}
    if path is not None:
        atomic_json(path, dict(identity=identity, scores=scores))
        print(f'scored {identity["source_sha256"][:12]}', flush=True)
    return scores


def percept_vectors(s, arm):
    recorded = json.loads((PC / 'pc_results.json').read_text())
    assert recorded['sample_sha256'] == fingerprint(s)
    arrays = []
    for seed in range(3):
        path = PC / 'stage2' / f'{arm}__s{seed}.npz'
        entry = recorded['arms'][arm]['seeds'][str(seed)]
        assert sha_file(path) == entry['source_sha256']['stage2'], path
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['config']['arm'] == arm and meta['config']['seed'] == seed
        assert meta['config']['eval_order'] == 'subset_pid' and meta['K'] == 40
        stage1 = PC / 'stage1' / f'{pc_score.ARMS[arm]}__s{seed}.npz'
        assert sha_file(stage1) == meta['stage1_sha256'] == entry['source_sha256']['stage1']
        assert json.loads(stage1.with_suffix('.json').read_text())['sample_sha256'] == fingerprint(s)
        with np.load(path, allow_pickle=False) as z:
            prob = z['eval_prob']
        assert prob.shape == (5000, 40) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        arrays.append(check_view(l2(prob)))
    return arrays


def network_vectors(s, aspect, stage0_dir):
    base = GENRE if aspect == 'genre' else stage0_dir
    if base is None:
        return None
    paths = [Path(base) / f'{aspect}__{STAGE1[aspect]}__s{seed}.npz' for seed in range(3)]
    if not any(p.exists() or p.with_suffix('.json').exists() for p in paths):
        return None
    arrays = []
    for seed, path in enumerate(paths):
        meta = json.loads(path.with_suffix('.json').read_text())
        config = meta['config']
        assert config['sample_sha256'] == fingerprint(s)
        assert config['aspect'] == aspect and config['seed'] == seed
        assert config.get('inputs', 'default') == 'default'
        assert config.get('input_img', 'default') == config.get('input_cap', 'default') == 'default'
        assert config.get('graph_row', config.get('row')) == STAGE1[aspect]
        assert sha_file(path) == meta['output_sha256'], path
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'img', 'cap'}
            for side in ('img', 'cap'):
                assert z[side].shape == (5000, 2048) and z[side].dtype == np.float16
                check_view(z[side])
            arrays.append(check_view(z['img']))
    return arrays


def encoder_vectors(s, kind):
    x = image_features(s, kind)
    # CLIP follows pc_score.row1; default_images already normalizes Qwen.
    if kind == 'clip_image':
        x = l2(np.asarray(x, dtype=np.float32))
    return check_view(x[s['subset_pid']])


def bar_vectors(s, stage0_dir=None):
    """bar -> aspect -> array, three seed arrays, or None (pending/AMI-only).

    Probe has no stored vectors; read probe_ami() for its recorded AMI.
    Stage 0 folder contains the flat filenames specified by the shared index.
    """
    clip = encoder_vectors(s, 'clip_image')
    qwen = encoder_vectors(s, 'qwen_default')
    percept = {arm: percept_vectors(s, arm) for arm in set(BEST.values())}
    return dict(percept={a: percept[BEST[a]] for a in ASPECTS},
                image_encoder=dict(emotion=qwen, style=clip, genre=qwen),
                target={a: check_view(prompted(s, PROMPTS[a])[0]) for a in ASPECTS},
                linear_probe={a: None for a in ASPECTS},
                stage1={a: network_vectors(s, a, stage0_dir) for a in ASPECTS})


def probe_ami():
    """Default Qwen supervised probe; AMI only, vectors were not saved."""
    r = json.loads(CEILING.read_text())
    result = {a: r[a]['qwen_default']['ami_k20_prob'] for a in ASPECTS}
    assert all(np.isfinite(list(v.values())).all() for v in result.values())
    return result


def check_percept(s, labels, outdir=OUTPUTS):
    outdir = Path(outdir)
    recorded = json.loads((PC / 'pc_results.json').read_text())
    checks = {}
    for arm in pc_score.ARMS:
        for seed, x in enumerate(percept_vectors(s, arm)):
            scores = measures(x, labels, outdir / 'cache')
            for a in labels:
                got = scores[a]['ami']['mean']
                expected = recorded['arms'][arm]['seeds'][str(seed)]['img'][a]['mean']
                checks[f'{arm}/s{seed}/{a}'] = dict(actual=got, expected=expected,
                    status='PASS' if round(got, 6) == round(expected, 6) else 'FAIL')
    # Direct API comparison on one real input; persisted so a rerun needs no k-means.
    x = percept_vectors(s, 'P1')[0]
    direct_path = outdir / 'pc_ami_reference.json'
    identity = dict(source_sha256=array_sha(x), sample_sha256=fingerprint(s),
                    labels={k: array_sha(v) for k, v in labels.items()},
                    sklearn=sklearn.__version__, scoring=SCORING)
    if direct_path.exists():
        direct = json.loads(direct_path.read_text())
        assert direct['identity'] == identity
    else:
        direct = dict(identity=identity, scores=pc_score.ami_scores(x, labels))
        atomic_json(direct_path, direct)
    ours = measures(x, labels, outdir / 'cache')
    equal = all(ours[a]['ami'] == direct['scores'][a] for a in labels)
    result = dict(status='PASS' if equal and all(v['status'] == 'PASS' for v in checks.values()) else 'FAIL',
                  sample_sha256=fingerprint(s), checks=checks, pc_ami_scores_equal=equal)
    atomic_json(outdir / 'stage0_scorer.json', result)
    assert result['status'] == 'PASS', 'PercepT scorer regression; see stage0_scorer.json'
    return result


def aggregate(arrays, labels, cache_dir):
    arrays = arrays if isinstance(arrays, list) else [arrays]
    assert len(arrays) in (1, 3)
    scores = [measures(x, labels, cache_dir) for x in arrays]
    if len(scores) == 1:
        return scores[0]
    return {a: {m: summary([r[a][m]['mean'] for r in scores]) for m in METRICS} for a in labels}


def backfill(s, labels, stage0_dir=None, outdir=OUTPUTS):
    outdir = Path(outdir)
    vectors = {name: {a: x for a in ASPECTS} for name, x in
               ((kind, encoder_vectors(s, kind))
                for kind in ('clip_image', 'qwen_default'))}
    vectors.update({arm: {a: x for a in ASPECTS} for arm in pc_score.ARMS
                    for x in [percept_vectors(s, arm)]})
    vectors['our_network'] = {a: network_vectors(s, a, stage0_dir) for a in ASPECTS}
    # Bind assertions to the actual report cells; margin rows are not methods.
    names = ('CLIP image', 'Qwen default image', 'P1:', 'P2:', 'P3:', 'P4:', 'P5:',
             'Our network, no instruction at test time')
    table = REPORT.read_text().split('| Group | Method | Emotion ↑ | Style ↑ | Genre ↑ |')
    assert len(table) == 2, 'two-bar table header changed'
    lines = table[1].split('| Margin |', 1)[0].splitlines()
    expected = {}
    for key, name in zip(vectors, names):
        matched = [line for line in lines if line.startswith('|') and line.split('|')[2].strip().startswith(name)]
        assert len(matched) == 1, (key, matched)
        expected[key] = dict(zip(ASPECTS, [float(v.strip().replace('*', ''))
                                          for v in matched[0].split('|')[3:6]]))
    rows = {}
    for key, aspects in vectors.items():
        rows[key] = {}
        for a, x in aspects.items():
            if x is None:
                rows[key][a] = dict(status='pending', reason='stage 0 arrays not yet available',
                                    expected_ami=expected[key][a])
                continue
            values = aggregate(x, labels, outdir / 'cache')[a]
            got = values['ami']['mean']
            assert round(got, 3) == expected[key][a], (key, a, got, expected[key][a])
            rows[key][a] = dict(status='complete', measures=values, expected_ami=expected[key][a])
    result = dict(sample_sha256=fingerprint(s), scoring=SCORING, rows=rows,
                  status='pending' if any(v['status'] == 'pending' for r in rows.values() for v in r.values()) else 'complete',
                  conventions='Single arrays: sd over k-means seeds; learned rows: mean and population sd '
                              'over three model-seed means. Network uses its own aspect head; image side only.')
    atomic_json(outdir / 'backfill.json', result)
    md = ['# K = 40 table back-fill', '', result['conventions'], '',
          '| Row | Aspect | Status | AMI | V | ARI | FMI |', '|---|---|---|---:|---:|---:|---:|']
    for key, row in rows.items():
        for a, v in row.items():
            cells = [f"{v['measures'][m]['mean']:.6f} ± {v['measures'][m]['sd']:.6f}"
                     for m in METRICS] if v['status'] == 'complete' else ['pending'] * 4
            md.append(f"| {key} | {a} | {v['status']} | " + ' | '.join(cells) + ' |')
    pending = outdir / 'backfill.pending.md'
    pending.write_text('\n'.join(md) + '\n')
    pending.replace(outdir / 'backfill.md')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-percept', action='store_true')
    p.add_argument('--backfill', action='store_true')
    p.add_argument('--stage0-dir', type=Path)
    p.add_argument('--outdir', type=Path, default=OUTPUTS)
    args = p.parse_args()
    if not (args.check_percept or args.backfill):
        p.error('choose --check-percept and/or --backfill')
    s = load_frozen()
    labels = label_vectors(s, s['subset_pid'])
    if args.check_percept:
        r = check_percept(s, labels, args.outdir)
        print(f"PercepT: {r['status']}; {len(r['checks'])} checks; direct API equal={r['pc_ami_scores_equal']}")
    if args.backfill:
        r = backfill(s, labels, args.stage0_dir, args.outdir)
        complete = sum(v['status'] == 'complete' for row in r['rows'].values() for v in row.values())
        print(f"Back-fill: {r['status']}; {complete}/24 aspect cells complete")


if __name__ == '__main__':
    with threadpool_limits(8):
        main()
