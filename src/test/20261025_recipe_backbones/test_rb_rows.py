from fractions import Fraction

import rb_paths
import ab_rows
import rb_rows


def test_regression_recipes_are_1222_rows():
    for name, old in rb_rows.REGRESSION.items():
        assert rb_rows.recipe(name) == ab_rows.resolve(old, 0)


def test_teacher_shares_and_weights():
    shares = {n: Fraction(rb_rows.teacher_share(n)).limit_denominator(10) for n in rb_rows.ORDER}
    assert shares == {'R0-5': Fraction(1, 5), 'R0-3': Fraction(1, 3), 'R6-5': Fraction(1, 5),
                      'R6-3': Fraction(1, 3), 'R7-5': Fraction(1, 5), 'R7-3': Fraction(1, 3)}
    weights = {n: rb_rows.RECIPES[n]['base']['distill_weight'] for n in rb_rows.ORDER}
    assert weights == {'R0-5': 0.5, 'R0-3': 1.0, 'R6-5': 0.25, 'R6-3': 0.5, 'R7-5': 0.25, 'R7-3': 0.5}


def test_only_weight_or_switch_differs_from_d2():
    d2 = rb_rows.recipe('R0-5')
    for name in rb_rows.ORDER:
        row = rb_rows.recipe(name)
        base = dict(row['base'], distill_weight=d2['base']['distill_weight'])
        assert base == d2['base'] and row['store'] == d2['store']
    assert rb_rows.TIE_ORDER == ('R0-3', 'R6-5', 'R6-3', 'R7-5', 'R7-3')


def test_imports_resolve_in_this_checkout():
    rb_paths.check_imports()
