"""fx-c13: B2's phase-1 probe (final_review/fr_b2_phase1_probe.py) rerun on the fixed comparer with the real pins files.

Per backbone: rd9_compare.py --phase1 on rederive/rd9_out/rd9_<B>_{phase1.json,seed42_arrays.npz} (made by
rd9_phase1.py in this worktree) against the real F/pins_<B>.json. Writes rd9_<B>_phase1_agreement.json under rederive/rd9_out
(ignored) and fx_c13_phase1_probe.json here. Usage: python fx_c13_phase1_probe.py clip_b32 qwen3vl_emb_2b
"""
import json, runpy, sys, traceback
from pathlib import Path
F = Path(__file__).resolve().parent
RD = F / 'rederive'
OUT = RD / 'rd9_out'
sys.path.insert(0, str(RD))
res = {}
for backbone in sys.argv[1:]:
    pins = F / ('pins_clip.json' if backbone == 'clip_b32' else f'pins_{backbone}.json')
    out = OUT / f'rd9_{backbone}_phase1_agreement.json'
    sys.argv = ['rd9_compare.py', '--phase1', '--backbone', backbone, '--pins', str(pins),
                '--derived', str(OUT / f'rd9_{backbone}_phase1.json'),
                '--derived-arrays', str(OUT / f'rd9_{backbone}_seed42_arrays.npz'), '--out', str(out)]
    try:
        try:
            runpy.run_path(str(RD / 'rd9_compare.py'), run_name='__main__')
        except SystemExit as e:
            res.setdefault(backbone, {})['exit'] = e.code
        rec = json.loads(out.read_text())
        res[backbone] = {**res.get(backbone, {}), 'ran': True, 'agreement': rec['agreement'], 'compared': rec['compared'],
                         'failed': rec['failed'], 'failed_keys': [x['key'] for x in rec['items'] if not x['equal']][:60],
                         'item_groups': sorted({'.'.join(x['key'].split('.')[:2]) for x in rec['items']})}
    except BaseException as e:
        res[backbone] = {'ran': False, 'error': f'{type(e).__name__}: {str(e)[:400]}',
                         'where': traceback.format_exc().strip().splitlines()[-3][:200]}
print(json.dumps(res, indent=1))
Path(__file__).with_suffix('.json').write_text(json.dumps(res, indent=1) + '\n')
