"""Model-sheet row definitions; CLI-derived rows keep the same schema."""
import copy
import math

ASPECTS = ('emotion', 'style', 'genre')
IMAGE = dict(emotion='Gres', style='Fres05', genre='R0o')
STAGE1 = dict(emotion='Fres1', style='Fres05', genre='R0o')
SECOND = dict(emotion='R1', style='Gres', genre='Fside025')


def row(share, graphs, **changes):
    result = dict(share=share, form='a', graphs=copy.deepcopy(graphs), graph_terms=True,
                  distill='none', distill_weight=0.0, inputs='default')
    result.update(changes)
    return result


ROWS = dict(A1=row('separate', IMAGE), T1=row('trunk', IMAGE),
            T2=row('trunk', STAGE1), T3=row('trunk', SECOND),
            T4=row('trunk', IMAGE, form='b'), T5=row('mask', IMAGE, form='c'),
            D1=row('trunk', None, graph_terms=False, distill='direct', distill_weight=1.0),
            D2=row('trunk', IMAGE, distill='direct', distill_weight=0.5),
            D3=row('trunk', IMAGE, distill='direct', distill_weight=1.0),
            D4=row('trunk', IMAGE, distill='direct', distill_weight=2.0),
            D5=row('trunk', IMAGE, distill='relational', distill_weight=1.0))


def validate(row):
    assert set(row) == {'share', 'form', 'graphs', 'graph_terms', 'distill', 'distill_weight', 'inputs'}
    assert row['share'] in ('separate', 'trunk', 'mask')
    assert row['form'] in ('a', 'b', 'c')
    assert (row['form'] == 'c') == (row['share'] == 'mask')
    assert isinstance(row['graph_terms'], bool)
    assert row['distill'] in ('none', 'direct', 'relational')
    weight = row['distill_weight']
    assert isinstance(weight, (float, int)) and not isinstance(weight, bool) and math.isfinite(weight)
    assert weight >= 0 and (weight > 0) == (row['distill'] != 'none')
    assert row['inputs'] in ('default', 'instructed')
    graphs = row['graphs']
    if graphs is None:
        assert not row['graph_terms']
    else:
        assert isinstance(graphs, dict) and set(graphs) == set(ASPECTS)
        assert all(isinstance(v, str) and v for v in graphs.values())
    assert row['form'] != 'b' or graphs is not None
    assert row['graph_terms'] or row['distill'] != 'none'
    return row
