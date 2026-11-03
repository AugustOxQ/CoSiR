import sys, os
T='/project/CoSiR/src/test'
for d in ['20261102_mmart_test','20261102_backbones_sweep','20261030_affection_confirmation','20261029_retrieval','20261027_recipe_sweep']:
    sys.path.insert(0, f'{T}/{d}')
sys.path.append('/project/CoSiR')
import numpy as np
import mt_data, mt_paths as P
idx = mt_data._index()
val = mt_data.keys('val')
print('val keys', len(val), 'unique', len(set(val)))
rows = mt_data._rows(val, idx)
names = rows.painting.astype(str).tolist()
print('artelingo col values', rows.artelingo.value_counts().to_dict())
print('unique painting names', len(set(names)))
import rt_data
an = rt_data.names('val')
print('artelingo val names', len(an), 'unique', len(set(an)))
s = set(an)
missing = [n for n in names if n not in s]
print('MMArt val names not in ArtELingo val:', len(missing), missing[:5])
# class sets per aspect: MMArt val vs test
for a in ('emotion','style','genre'):
    yv = mt_data.labels(val, a); yt = mt_data.labels(mt_data.keys('test'), a)
    cv = sorted(set(yv[yv>=0].tolist())); ct = sorted(set(yt[yt>=0].tolist()))
    print('MMArt', a, 'val classes', len(cv), 'test classes', len(ct), 'same set', cv==ct, 'val labelled', int((yv>=0).sum()))
