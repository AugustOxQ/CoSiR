"""Thirteen D2 ablation recipes; seed placeholders resolve only at run time."""
import copy

from ab_paths import ms_rows

ASPECTS = ms_rows.ASPECTS
DEFAULT_SWITCHES = dict(teacher='instructed', buddy=1.0, own=1.0, endpoints='both')


def _row(*, base=None, switches=None, store='aspect', **changes):
    base = copy.deepcopy(ms_rows.ROWS['D2'] if base is None else base)
    base.update(changes)
    return dict(base=base, switches=dict(DEFAULT_SWITCHES, **(switches or {})), store=store)


ROWS = {
    'R0': _row(),
    'R1': _row(share='separate'),
    'R2': _row(base=ms_rows.ROWS['T1']),
    'R3': _row(switches=dict(teacher='default')),
    'R4': _row(graphs=None, graph_terms=False),
    'R5': _row(switches=dict(buddy=0.0)),
    'R6': _row(switches=dict(own=0.0)),
    'R7': _row(switches=dict(endpoints='image')),
    'R8': _row(graphs=dict(emotion='R0', style='R0o', genre='R0o')),
    'R9': _row(graphs=dict.fromkeys(ASPECTS, 'Dview'), store='ab'),
    'R10': _row(graphs=dict.fromkeys(ASPECTS, 'rand_s{seed}'), store='ab'),
    'R11': _row(graphs=dict.fromkeys(ASPECTS, 'Dview'), store='ab',
                distill='none', distill_weight=0.0),
    'R12': _row(inputs='instructed'),
}


def validate(row):
    assert set(row) == {'base', 'switches', 'store'}
    ms_rows.validate(row['base'])
    switches = row['switches']
    assert set(switches) == set(DEFAULT_SWITCHES)
    assert switches['teacher'] in ('instructed', 'default')
    assert switches['endpoints'] in ('both', 'image')
    assert all(switches[k] in (0.0, 1.0) for k in ('buddy', 'own'))
    assert row['store'] in ('aspect', 'ab')
    if switches != DEFAULT_SWITCHES:
        assert row['base']['form'] == 'a' and row['base']['graph_terms']
        assert row['base']['distill'] == 'direct'
    if switches['teacher'] == 'default':
        assert row['base']['inputs'] == 'default'
    return row


def resolve(name, seed):
    assert seed in (0, 1, 2)
    row = copy.deepcopy(ROWS[name])
    if row['base']['graphs'] is not None:
        row['base']['graphs'] = {a: graph.format(seed=seed)
                                 for a, graph in row['base']['graphs'].items()}
    return validate(row)


for _recipe in ROWS.values():
    validate(_recipe)
