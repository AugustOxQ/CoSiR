"""Compare the Codex re-derivation (run 3) with pc_results.json; prints max abs differences."""
import json
import numpy as np
R = '/tmp/claude-0/-project-CoSiR/724fc430-8911-4bb5-ba46-fe27b9e6c38a/scratchpad/rederive3/rederive_results.json'
cx = json.load(open(R))
ours = json.load(open('/project/CoSiR/src/test/20261018_percept_core/outputs/pc_results.json'))['arms']
A = ('emotion', 'style', 'genre')
d = {'file_img': [], 'file_argmax': [], 'arm_img_mean': [], 'arm_img_sd': [], 'arm_argmax_mean': [], 'arm_argmax_sd': [],
     'stage1_share': [], 'stage1_argmax': []}
n = {k: 0 for k in d}
for f in cx['stage2']['files']:
    o = ours[f['arm']]['seeds'][str(f['training_seed'])]
    assert f['K'] == o['K'] == 40
    for a in A:
        d['file_img'].append(abs(np.mean([k['ami'][a] for k in f['kmeans']]) - o['img'][a]['mean']))
        d['file_argmax'].append(abs(f['argmax_ami'][a] - o['native_img'][a]))
for arm, v in cx['stage2']['arms'].items():
    for a in A:
        d['arm_img_mean'].append(abs(v['labels'][a]['image_side']['mean'] - ours[arm]['img'][a]['mean']))
        d['arm_img_sd'].append(abs(v['labels'][a]['image_side']['sd'] - ours[arm]['img'][a]['sd']))
        d['arm_argmax_mean'].append(abs(v['labels'][a]['argmax']['mean'] - ours[arm]['native_img'][a]['mean']))
        d['arm_argmax_sd'].append(abs(v['labels'][a]['argmax']['sd'] - ours[arm]['native_img'][a]['sd']))
version_arm = {'clip_roberta': 'P1', 'qwen_qwen': 'P3', 'qwen_roberta': 'P5'}
cover = []
for f in cx['stage1']['files']:
    o = ours[version_arm[f['version']]]['seeds'][str(f['training_seed'])]
    kt = o['kept_topics']
    cover.append(f['with_fit_pair'] == kt['with_fit_pair'] and f['with_1pct'] == kt['with_1pct'] and f['top40_matches'])
    d['stage1_share'].append(abs(f['share_before_cut'] - kt['share_before_cut']))
    am = f.get('stage1_argmax_ami') or f.get('argmax_ami')
    for a in A:
        d['stage1_argmax'].append(abs(am[a] - o['native_topic'][a]))
    print(f"{f['version']} s{f['training_seed']}: 40th {f['count_40th']} 41st {f['count_41st']} gap {f['cut_gap']}")
for k, v in d.items():
    print(f'{k}: {len(v)} values, max abs diff {max(v):.3g}')
print('stage1 coverage counts and top-40 equal:', sum(cover), 'of', len(cover))
print('codex comparators:', json.dumps(cx['stage2']['comparators'])[:900])
print('problems:', cx['problems'], '| checks ok:', sum(c['ok'] for c in cx['checks']), 'of', len(cx['checks']))
