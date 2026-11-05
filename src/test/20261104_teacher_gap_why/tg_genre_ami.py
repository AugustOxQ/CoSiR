"""Quick add-on to tg_genre_link: AMI and NMI of genre labels used directly as an emotion partition, beside Qwen c* and
the teacher on the same rows (held rows with both labels). A diagnostic; no claim."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R17 = HERE.parent / '20261103_new_scores'
sys.path.insert(0, str(R17))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu_cells  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402

out = {}
for set_ in P.PAINTING_SETS:
    split, k = P.HELD_SPLIT[set_], P.CLASS_COUNT[(set_, 'emotion')]
    e = np.asarray(ns_data.labels(set_, split, 'emotion'))
    g = np.asarray(ns_data.labels(set_, split, 'genre'))
    both = (e >= 0) & (g >= 0)
    t = ns_metrics.contingency(e[both], g[both], shape=(9, 10))
    doc = dict(genre_labels={m: ns_metrics.from_contingency(t, m) for m in ('ami', 'nmi', 'ari', 'acc')})
    for name, parts in (('teacher', ns_clu_cells._comp_parts(set_, split, 'qwen', 'emotion', 'teacher', k)),
                        ('cstar_qwen', ns_clu_cells._cand_parts(set_, split, 'qwen', 'emotion', k))):
        flat = parts.reshape(-1, parts.shape[-1])
        tabs = [ns_metrics.contingency(e[both], p[both], shape=(9, k)) for p in flat]
        doc[name] = {m: float(np.mean([ns_metrics.from_contingency(x, m) for x in tabs])) for m in ('ami', 'nmi', 'ari', 'acc')}
    out[set_] = doc
    print(set_, json.dumps(doc), flush=True)
(HERE / 'outputs/genre_ami.json').write_text(json.dumps(out, indent=1))
