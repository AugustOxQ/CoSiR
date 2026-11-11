"""r21 arm names and settings (index, "Shared interfaces"): `parse_arm(name) -> settings`, `arm_name(settings) -> name`.

Settings: form in off | plain | start | teacher; lam (float, DEC weight); warmup (int W); stop in S1 | S2; balance
(bool). `off` (the c* rerun) has the other four as None. Name: `dec-off`, else `dec-{form}_l{lam:g}_w{W}_{stop}` plus
`_bal` when balance is on. The two functions are inverse; `parse_arm` refuses any name that `arm_name` would not write.
Pure Python, no file reads: ticket 03's runner imports it (as `r21_run.parse_arm`, `r21_run.arm_name`).
"""
import re

FORMS = ('off', 'plain', 'start', 'teacher')
STOPS = ('S1', 'S2')
_NAME = re.compile(r'dec-(?P<form>plain|start|teacher)_l(?P<lam>[0-9.]+(?:e[+-]?[0-9]+)?)_w(?P<w>[0-9]+)'
                   r'_(?P<stop>S1|S2)(?P<bal>_bal)?')


def arm_name(settings):
    assert set(settings) == {'form', 'lam', 'warmup', 'stop', 'balance'}, sorted(settings)
    assert settings['form'] in FORMS, settings['form']
    if settings['form'] == 'off':
        assert all(settings[k] is None for k in ('lam', 'warmup', 'stop', 'balance')), settings
        return 'dec-off'
    assert isinstance(settings['lam'], (int, float)) and not isinstance(settings['lam'], bool) and settings['lam'] > 0
    assert isinstance(settings['warmup'], int) and not isinstance(settings['warmup'], bool) and settings['warmup'] >= 0
    assert settings['stop'] in STOPS and isinstance(settings['balance'], bool), settings
    return (f"dec-{settings['form']}_l{settings['lam']:g}_w{settings['warmup']}_{settings['stop']}"
            + ('_bal' if settings['balance'] else ''))


def parse_arm(name):
    if name == 'dec-off':
        settings = dict(form='off', lam=None, warmup=None, stop=None, balance=None)
    else:
        m = _NAME.fullmatch(name)
        assert m, f'not an r21 arm name: {name!r}'
        settings = dict(form=m['form'], lam=float(m['lam']), warmup=int(m['w']), stop=m['stop'], balance=bool(m['bal']))
    assert arm_name(settings) == name, f'{name!r} is not in canonical form ({arm_name(settings)!r})'
    return settings
