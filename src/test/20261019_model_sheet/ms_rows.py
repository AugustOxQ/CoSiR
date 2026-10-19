"""Model-sheet row definitions; CLI-derived rows keep the same schema."""
import copy

ASPECTS = ('emotion', 'style', 'genre')
IMAGE = dict(emotion='Gres', style='Fres05', genre='R0o')
STAGE1 = dict(emotion='Fres1', style='Fres05', genre='R0o')
SECOND = dict(emotion='R1', style='Gres', genre='Fside025')


def row(share, graphs):
    return dict(share=share, form='a', graphs=copy.deepcopy(graphs), graph_terms=True,
                distill='none', distill_weight=0.0, inputs='default')


ROWS = dict(A1=row('separate', IMAGE), T1=row('trunk', IMAGE),
            T2=row('trunk', STAGE1), T3=row('trunk', SECOND))


def validate(row):
    assert set(row) == {'share', 'form', 'graphs', 'graph_terms', 'distill', 'distill_weight', 'inputs'}
    assert row['share'] in ('separate', 'trunk'), 'ticket 02 implements separate/trunk only'
    assert row['form'] == 'a' and row['graph_terms'] is True, 'ticket 02 implements form a graph loss only'
    assert row['distill'] == 'none' and row['distill_weight'] == 0, 'distillation belongs to ticket 03'
    assert row['inputs'] == 'default', 'instructed inputs belong to ticket 03'
    assert isinstance(row['graphs'], dict) and set(row['graphs']) == set(ASPECTS)
    assert all(isinstance(v, str) and v for v in row['graphs'].values())
    return row
