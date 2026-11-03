import sys
T='/project/CoSiR/src/test'
for d in ['20261102_mmart_test','20261102_backbones_sweep','20261030_affection_confirmation','20261029_retrieval','20261027_recipe_sweep']:
    sys.path.insert(0, f'{T}/{d}')
import ac_paths
sys.path.append('/project/CoSiR')
import numpy as np
import rt_data
lv, lh = rt_data.labels('val'), rt_data.labels('held')
for a in ('emotion','style','genre'):
    yv, yh = np.asarray(lv[a]), np.asarray(lh[a])
    cv, ch = sorted(set(yv[yv>=0].tolist())), sorted(set(yh[yh>=0].tolist()))
    print('ArtELingo', a, 'val', len(cv), 'held', len(ch), 'same', cv==ch, 'val min', np.bincount(yv[yv>=0]).min(), 'held min', np.bincount(yh[yh>=0]).min())
try:
    import ac_data
    for sp in ('val','test'):
        y=np.asarray(ac_data.labels(sp)); print('Affection', sp, len(y), sorted(set(y.tolist())))
except Exception as e: print('aff err', e)
try:
    import bs_emoset
    for sp in ('val','test'):
        k=bs_emoset.keys(sp, cut=True); y=bs_emoset.labels(k); print('EmoSet', sp, len(k), sorted(set(y)))
except Exception as e: print('emoset err', repr(e)[:300])
