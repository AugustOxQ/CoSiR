"""Stage iii's variants, ladders and orders (rule §2.3, §4.3)."""
BASE = 'R6-5'
TEACHER_WEIGHT = 0.25                       # R6-5's genre teacher weight; GT-m uses 0.25 * m
FIRST = ('GT-2', 'GT-4', 'GR-0.5', 'GR-2', 'GA-1')   # the five first variants, in the tie order
LADDERS = {'GT': (1.5, 2, 3, 4, 6, 8), 'GR': (0.25, 0.5, 1, 2, 4, 8), 'GA': (0.25, 0.5, 1, 2, 4)}
FRESH_SEEDS = (1, 2, 3)
T_G = 0.4989161752444698                    # reported.json references_p.genre_word.genre (frozen, rule §3.3)
RANGES = {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}


def name(method, rung):
    """'GT', 2 -> 'GT-2'; 'GR', 0.5 -> 'GR-0.5' (rungs as the ladder writes them)."""
    assert method in LADDERS and rung in LADDERS[method], (method, rung)
    return f'{method}-{rung:g}'


def parse(variant):
    """'GR-0.5' -> ('GR', 0.5); the base 'R6-5' -> (None, None)."""
    if variant == BASE:
        return None, None
    method, rung = variant.split('-', 1)
    rung = float(rung)
    rung = int(rung) if rung == int(rung) else rung
    assert method in LADDERS and rung in LADDERS[method], variant
    return method, rung


def switches(variant):
    """The genre-loss switches of a variant: teacher multiplier m, relational weight lam, adaptor weight w."""
    method, rung = parse(variant)
    return {'m': rung if method == 'GT' else 1, 'lam': rung if method == 'GR' else 0,
            'w': rung if method == 'GA' else 0}


for _v in FIRST:
    assert name(*parse(_v)) == _v, _v
