"""The six part 1 recipes (rule §2): 12-22's R0, R6 and R7 at teacher shares 1/5 and 1/3."""
import copy

import rb_paths  # noqa: F401  (sys.path)
import ab_rows

RECIPES = {
    'R0-5': ab_rows.ROWS['R0'],
    'R0-3': ab_rows._row(distill_weight=1.0),
    'R6-5': ab_rows._row(switches=dict(own=0.0), distill_weight=0.25),
    'R6-3': ab_rows.ROWS['R6'],
    'R7-5': ab_rows._row(switches=dict(endpoints='image'), distill_weight=0.25),
    'R7-3': ab_rows.ROWS['R7'],
}
ORDER = tuple(RECIPES)
REFERENCE = 'R0-5'
TIE_ORDER = ORDER[1:]                                   # rule §4.1: order on equal round(g, 2)
REGRESSION = {'R0-5': 'R0', 'R6-3': 'R6', 'R7-3': 'R7'}  # stage 0: 12-22's rows, seed 0
IMAGE_ONLY = ('R7-5', 'R7-3')                            # no caption network; K5 exists only under these
SEED = 0


def teacher_share(name):
    """w over the summed weights of the loss groups the recipe keeps (buddy, own-caption, teacher)."""
    row = RECIPES[name]
    groups = 1.0 + row['switches']['own'] if row['switches']['endpoints'] == 'both' else 1.0
    w = row['base']['distill_weight']
    return w / (groups + w)


def recipe(name):
    return ab_rows.validate(copy.deepcopy(RECIPES[name]))


for _name in ORDER:
    recipe(_name)
