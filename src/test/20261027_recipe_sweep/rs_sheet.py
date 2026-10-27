"""The number sheet (spec §3.2 at the rule's spec commit), model G's settings, run configurations and their names, and
the ablation's removals (spec §3.2, rule §2.5).

A configuration is a dict with every knob k1 to k13 and `remove`, a tuple of removal names in REMOVALS order. Its
name is the run folder's name: 'G' for G itself; otherwise the changed knobs in sheet order, 'k2=8+k6=0.0005'; a
removal adds '-no_<removal>' ('G-no_buddy'). `parse(name(c)) == c` for every configuration.
"""
KNOBS = tuple(f'k{i}' for i in range(1, 14))

# G's value of each knob (spec §3.2 column "G").
G = dict(k1=0, k2=0, k3=8, k4=0.02, k5=0.25, k6=1e-3, k7=0.05, k8=512, k9=1024, k10=0.1, k11=20, k12=0.05, k13=15)

# The three stage 1 values of each knob, in the sheet's order (rule §4.2's tie-breaks use these positions).
VALUES = dict(
    k1=(2, 8, 16),                 # relational term on the emotion head: weight (off in G)
    k2=(2, 8, 16),                 # relational term on the style head: weight (off in G)
    k3=(4, 12, 16),                # the genre relational weight
    k4=(0.015, 0.025, 0.035),      # the relational teacher row's temperature (every relational term in the run)
    k5=(0.0625, 0.125, 0.5),       # the direct teacher weight, all three heads
    k6=(3e-4, 5e-4, 2e-3),         # learning rate
    k7=(0.0, 0.01, 0.2),           # weight decay (AdamW)
    k8=(256, 1024, 2048),          # batch, in graph pairs (rows are twice the pairs); training and stopping loss
    k9=(512, 1536, 2048),          # shared trunk width
    k10=(0.0, 0.2, 0.3),           # dropout
    k11=(10, 30, 40),              # graph neighbours: mutual top-k in every aspect graph and lambda graph
    k12=(0.03, 0.07, 0.1),         # the four buddy terms' temperature (direct and relational temperatures stay)
    k13=(5, 10, 30),               # early stopping patience, epochs
)
INT_KNOBS = frozenset({'k1', 'k2', 'k3', 'k8', 'k9', 'k11', 'k13'})

# The ablation's removals (spec §3.2 table, in its order). Each removes one part of the final model F.
REMOVALS = (
    'relational',          # weight 0 on every head that has a relational term
    'sharp_row',           # every relational teacher row at 0.05, the student's temperature
    'direct',              # direct teacher weight 0 on all three heads
    'buddy',               # the four buddy terms off
    'caption_endpoints',   # only the image-image buddy term (D2's image-only endpoints)
    'content_correction',  # emotion and style graphs built without the content correction (genre's has none)
    'instructed_graphs',   # one default-view graph for all three aspects
    'shared_trunk',        # separate heads: one trunk per aspect
)

ASPECTS = ('emotion', 'style', 'genre')
RANGES = dict(emotion=0.0046, style=0.0082, genre=0.0128)   # the seed ranges r_a (rule §3)
TOL = 1e-6
SWEEP_SEED = 0
DEV_SEEDS = (1, 2, 3)
FULL_SEEDS = (0, 1, 2, 3, 4)

assert set(G) == set(KNOBS) == set(VALUES)
assert all(len(v) == 3 and G[k] not in v for k, v in VALUES.items())


def _value(k, v):
    assert k in KNOBS, k
    if k in INT_KNOBS:
        assert float(v) == int(v), (k, v)
        return int(v)
    return float(v)


def config(changes=None, remove=()):
    """G with the knobs in `changes` set and the removals in `remove`; values typed as the sheet's."""
    changes = dict(changes or {})
    assert set(changes) <= set(KNOBS), changes
    assert set(remove) <= set(REMOVALS) and len(set(remove)) == len(tuple(remove)), remove
    cfg = {k: _value(k, changes.get(k, G[k])) for k in KNOBS}
    cfg['remove'] = tuple(r for r in REMOVALS if r in set(remove))
    return cfg


def changed(cfg):
    """The knobs set away from G's value, in sheet order."""
    return tuple(k for k in KNOBS if cfg[k] != G[k])


def _fmt(k, v):
    return str(int(v)) if k in INT_KNOBS else repr(float(v))


def name(cfg):
    assert set(cfg) == set(KNOBS) | {'remove'}, sorted(cfg)
    knobs = changed(cfg)
    base = '+'.join(f'{k}={_fmt(k, cfg[k])}' for k in knobs) if knobs else 'G'
    return base + ''.join(f'-no_{r}' for r in cfg['remove'])


def parse(text):
    base, *removed = text.split('-no_')
    changes = {}
    if base != 'G':
        for part in base.split('+'):
            k, v = part.split('=')
            changes[k] = float(v)
    cfg = config(changes, removed)
    assert name(cfg) == text, (text, name(cfg))
    return cfg


def stage1():
    """The 39 stage 1 configurations c(k, v), in sheet order (k1 first, values in listed order)."""
    return [config({k: v}) for k in KNOBS for v in VALUES[k]]


def positions(cfg):
    """Rule §4.2's last tie-break: the sheet positions (0, 1, 2) of the changed knobs' values, in knob order."""
    return tuple(VALUES[k].index(cfg[k]) for k in changed(cfg))


def knob_numbers(cfg):
    """Rule §4.2's second tie-break: the sorted tuple of the changed knobs' numbers."""
    return tuple(sorted(int(k[1:]) for k in changed(cfg)))
