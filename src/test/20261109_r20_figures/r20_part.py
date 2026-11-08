"""Step `partitions`: k-means partitions of c* (per aspect) and of the plain Qwen image, for k = 20 and the aspect's
class count. r17's `ns_metrics.partitions(x, k, set_name)` (MiniBatchKMeans, n_init 3, batch 4,096, states 0 to 2);
state 0 (row 0) is kept. All rows are clustered; labels are masked later.

The three states are independent fits (`random_state=state`), so fitting state 0 alone gives the same labels as row 0
of the three-state call; `fit_state0` does that (a third of the time) and the smoke run checks the two agree. Use
`exact=True` for the literal three-state call.

Cache: <out>/partitions/<set>/<model>__<aspect or plain>__k<k>.npy (int32 [n]) with a .json sidecar (rows, keys sha256,
input file sha256s). Count 20 and the class count are the same partition where they coincide (MMArt style: 20).
"""
import json
import sys
import time

import numpy as np

import r20_data as D

NS_ROUND = D.MAIN_TEST / '20261103_new_scores'
COUNTS_NOTE = 'class count equals 20 where they coincide (MMArt style): one partition serves both'


def _ns():
    if str(NS_ROUND) not in sys.path:
        sys.path.insert(0, str(NS_ROUND))
    D.setup_paths()
    import ns_metrics
    import ns_paths
    return ns_metrics, ns_paths


def counts_for(s, aspect):
    """Sorted distinct k values: 20 and the class count of the aspect (classes present among labelled rows)."""
    lab = s.labels[aspect]
    return sorted({20, int(len(np.unique(lab[lab >= 0])))})


def jobs(s):
    """[(model, aspect or 'plain', k)]: c* per aspect at its counts, plain at the union of the counts."""
    out = [('cstar', a, k) for a in D.ASPECTS for k in counts_for(s, a)]
    ks = sorted({k for a in D.ASPECTS for k in counts_for(s, a)})
    out += [('plain', 'plain', k) for k in ks]
    return out


def path_for(root, set_name, model, aspect, k):
    return root / 'partitions' / set_name / f'{model}__{aspect}__k{k}.npy'


def fit_state0(x, k, set_name, exact=False):
    """int32 [n] labels of state 0. exact=True calls r17's three-state `partitions` and takes row 0."""
    ns_metrics, ns_paths = _ns()
    if exact:
        return ns_metrics.partitions(x, k, set_name)[0]
    old = ns_paths.KMEANS_STATES
    ns_paths.KMEANS_STATES = (0,)
    try:
        return ns_metrics.partitions(x, k, set_name)[0]
    finally:
        ns_paths.KMEANS_STATES = old


def load_partition(root, set_name, model, aspect, k):
    p = path_for(root, set_name, model, aspect, k)
    return np.load(p)


def run(smoke=False, sets=None, all_jobs=False):
    root = D.OUT / ('smoke' if smoke else '')
    report = []
    for name in (sets or D.SETS):
        s = D.load(name)
        if smoke:
            s = s.smoke()
        shas = D.input_shas(s)
        ksha = s.keys_sha256()
        x_cache = {}

        def rows_of(model, aspect):
            key = (model, aspect)
            if key not in x_cache:
                x_cache.clear()
                x_cache[key] = s.plain() if model == 'plain' else s.cstar(aspect)
            return x_cache[key]

        todo = jobs(s)
        if smoke and not all_jobs:                  # one aspect: two counts of c* and the plain image
            todo = [('cstar', 'emotion', 20), ('cstar', 'emotion', counts_for(s, 'emotion')[0]), ('plain', 'plain', 20)]
            todo = list(dict.fromkeys(todo))
        for model, aspect, k in todo:
            p = path_for(root, name, model, aspect, k)
            side = p.with_suffix('.json')
            in_files = ([s.sources['plain']] if model == 'plain' else [s.sources['c_star'][aspect]])
            want = dict(rows=len(s), keys_sha256=ksha, input_sha256={f: shas[f] for f in in_files})
            if p.exists() and side.exists():
                have = json.loads(side.read_text())
                if all(have.get(key) == v for key, v in want.items()):
                    report.append(dict(set=name, model=model, aspect=aspect, k=k, status='cached'))
                    continue
            t = time.time()
            x = rows_of(model, aspect)
            lab = fit_state0(x, k, name)
            if smoke:
                agree = bool(np.array_equal(lab, fit_state0(x, k, name, exact=True)))
                assert agree, 'state-0 shortcut differs from row 0 of the three-state call'
            assert lab.shape == (len(s),) and lab.dtype == np.int32 and lab.min() >= 0
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, lab)
            import sklearn
            side.write_text(json.dumps(dict(
                **want, set=name, model=model, aspect=aspect, k=k, state=0, n_clusters_found=int(len(np.unique(lab))),
                seconds=round(time.time() - t, 1), sklearn=sklearn.__version__, note=COUNTS_NOTE,
                kmeans=dict(cls='MiniBatchKMeans', n_init=3, batch_size=4096, random_state=0,
                            via='ns_metrics.partitions (r17)')), indent=1))
            report.append(dict(set=name, model=model, aspect=aspect, k=k, status='done', seconds=round(time.time() - t, 1)))
    return report
