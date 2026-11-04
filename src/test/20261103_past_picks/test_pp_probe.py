"""Tests of pp_probe (r19 ticket 10): the timing arithmetic of SC15 on fake timers (CPU only).

Fake clocks advance by amounts the tests choose, so every sum, maximum, minimum and rate is known in advance. The GPU
parts (encoders, nvidia-smi, the networks) are plumbing and are covered by the smoke run, not here."""
import hashlib
import json
import sys
import threading
from types import SimpleNamespace

import pytest

import pp_paths as PP
import pp_probe as P
import pp_outcome as O


class FakeClock:
    """A clock that moves only when the test says so (`cost` seconds per `run` call)."""

    def __init__(self):
        self.t = 0.0
        self.syncs = 0

    def __call__(self):
        return self.t

    def sync(self):
        self.syncs += 1


# ------------------------------------------------------------------------------------------------ batches
def batch_sizes(n, batch):
    return [min(batch, n - i) for i in range(0, n, batch)]


def test_batches_split_2000_into_63_with_a_short_last_batch():
    sizes = batch_sizes(2000, 32)
    assert len(sizes) == 63 and sizes[:62] == [32] * 62 and sizes[-1] == 16 and sum(sizes) == 2000
    assert batch_sizes(64, 32) == [32, 32]


def test_warmup_batches_are_run_but_out_of_the_timings_and_sums():
    clock = FakeClock()
    calls = []

    def run(batch):
        calls.append(batch)
        clock.t += 1000.0 if len(calls) <= 3 else 0.5          # the first three calls (warm-up) are slow
    batches = list(range(10))
    secs = P.time_batches(run, batches, warmup=3, sync=clock.sync, clock=clock)
    assert len(calls) == 13 and calls[:3] == [0, 1, 2]           # three warm-up runs, then all ten timed
    assert secs == [0.5] * 10 and sum(secs) == 5.0               # nothing of the 3000 s warm-up is in the sums
    assert clock.syncs >= 2 * 10 + 1                             # synchronised before and after every timed batch


def test_each_timed_batch_is_bracketed_by_a_synchronisation():
    events = []
    clock = FakeClock()
    clock.sync = lambda: events.append('sync')

    def run(batch):
        events.append('run')
        clock.t += 1.0
    P.time_batches(run, [0, 1], warmup=0, sync=clock.sync, clock=clock)
    for i, e in enumerate(events):
        if e == 'run':
            assert i > 0 and events[i - 1] == 'sync' and i + 1 < len(events) and events[i + 1] == 'sync'


def test_warmup_cycles_when_there_are_fewer_batches_than_warmup_runs():
    seen = []
    clock = FakeClock()
    P.time_batches(lambda b: seen.append(b), ['a', 'b'], warmup=3, sync=clock.sync, clock=clock)
    assert seen == ['a', 'b', 'a', 'a', 'b']


# ------------------------------------------------------------------------------------------------ records
def rec(name, secs, n=None, kind='model', repeat=0, **kw):
    n = n or [32] * len(secs)
    return P.batch_record(name, kind, repeat, n, secs, batch_size=32, warmup_batches=3, **kw)


def test_ms_per_image_is_total_over_images_not_the_mean_of_batch_rates():
    r = rec('x', [0.032, 0.016], n=[32, 16])                     # 32 ms/img, 32 ms/img
    assert r['total_seconds'] == pytest.approx(0.048) and r['n_images'] == 48
    assert r['ms_per_image'] == pytest.approx(1000 * 0.048 / 48)
    assert r['images_per_second'] == pytest.approx(48 / 0.048)
    r2 = rec('x', [0.064, 0.001], n=[32, 1])                     # mean of rates would be 1500; the sum rule gives 1000*0.065/33
    assert r2['ms_per_image'] == pytest.approx(65.0 / 33)


def test_record_refuses_wrong_lengths_and_bad_numbers():
    with pytest.raises(AssertionError):
        P.batch_record('x', 'model', 0, [32, 32], [0.1], batch_size=32, warmup_batches=3)
    with pytest.raises(AssertionError):
        rec('x', [0.1, float('nan')])
    with pytest.raises(AssertionError):
        rec('x', [0.1, -0.1])


# ------------------------------------------------------------------------------------------------ polling
def test_foreign_processes_exclude_our_own_pid():
    text = '123, 300 MiB\n456, 2048 MiB\n'
    assert P.parse_compute_apps(text) == [(123, 300), (456, 2048)]
    assert P.foreign(P.parse_compute_apps(text), own_pid=123) == [(456, 2048)]
    assert P.foreign(P.parse_compute_apps(''), own_pid=1) == []
    assert P.foreign(P.parse_compute_apps('\n'), own_pid=1) == []
    with pytest.raises(AssertionError):
        P.parse_compute_apps('garbage line')


def make_measure(values, log=None):
    """measure(): returns the next record; values are total seconds per attempt."""
    it = iter(values)

    def measure():
        s = next(it)
        if log is not None:
            log.append(s)
        return rec('p', [s / 2, s / 2])
    return measure


def test_clean_repeat_is_kept_with_no_discard():
    out, discarded = P.run_repeat('p', 0, make_measure([1.0]), poll=lambda: [], own_pid=1)
    assert out['total_seconds'] == 1.0 and out['attempt'] == 1 and discarded == []


@pytest.mark.parametrize('when', ['before', 'after'])
def test_repeat_with_another_process_is_discarded_logged_and_rerun(when):
    polls = []
    state = {'n': 0}

    def poll():
        state['n'] += 1
        polls.append(state['n'])
        # attempt 1 uses polls 1 (before) and 2 (after); attempt 2 uses polls 3 and 4
        bad = (state['n'] == 1) if when == 'before' else (state['n'] == 2)
        return [(999, 5000)] if bad else []
    log = []
    out, discarded = P.run_repeat('p', 1, make_measure([7.0, 1.0], log), poll=poll, own_pid=1)
    assert log == [7.0, 1.0]                                      # measured twice
    assert out['total_seconds'] == 1.0 and out['attempt'] == 2    # the clean rerun is the one kept
    assert len(discarded) == 1
    d = discarded[0]
    assert d['pass'] == 'p' and d['repeat'] == 1 and d['attempt'] == 1
    assert d['processes_' + when] == [[999, 5000]] and d['record']['total_seconds'] == 7.0


def test_own_pid_alone_does_not_discard_and_endless_interference_stops():
    out, discarded = P.run_repeat('p', 0, make_measure([1.0]), poll=lambda: [(5, 100)], own_pid=5)
    assert discarded == []
    with pytest.raises(RuntimeError, match='another process'):
        P.run_repeat('p', 0, make_measure([1.0] * 9), poll=lambda: [(6, 1)], own_pid=5, max_attempts=3)


# ------------------------------------------------------------------------------------------------ end to end loop
def test_end_to_end_batch_seconds_are_the_gaps_between_completions_and_include_loading():
    clock = FakeClock()
    loaded = []

    def load(batch):
        loaded.append((batch, threading.current_thread().name))
        return [f'im{batch}']

    def run(ims):
        clock.t += 2.0
    secs = P.end_to_end(run, [0, 1, 2], load, sync=clock.sync, clock=clock)
    assert secs == [2.0, 2.0, 2.0] and sum(secs) == 6.0
    assert [b for b, _ in loaded] == [0, 1, 2]
    assert {name for _, name in loaded} != {threading.current_thread().name}   # loaded on the prefetch thread


def test_end_to_end_closes_every_batch_of_images():
    closed = []

    class Im:
        def __init__(self, i):
            self.i = i

        def close(self):
            closed.append(self.i)
    clock = FakeClock()
    P.end_to_end(lambda ims: None, [0, 1], lambda b: [Im(b), Im(b + 10)], sync=clock.sync, clock=clock)
    assert sorted(closed) == [0, 1, 10, 11]


def test_end_to_end_prefetches_the_next_batch_while_the_current_batch_runs():
    next_loaded = threading.Event()
    clock = FakeClock()

    def load(batch):
        if batch == 1:
            next_loaded.set()
        return [batch]

    def run(ims):
        if ims == [0]:
            assert next_loaded.wait(timeout=0.2), 'next decode must overlap run(0)'
        clock.t += 1.0

    assert P.end_to_end(run, [0, 1], load, sync=clock.sync, clock=clock) == [1.0, 1.0]


# ------------------------------------------------------------------------------------------------ the model pass
def test_model_repeat_uses_held_images_and_a_fake_encoder_with_a_cold_start():
    clock = FakeClock()
    calls = {'n': 0}
    images = list(range(70))                                      # batches of 32: 32, 32, 6

    def encode(ims):
        calls['n'] += 1
        clock.t += (50.0 if calls['n'] <= 3 else 0.001 * len(ims))
    r = P.model_repeat('enc', encode, images, repeat=0, batch_size=32, warmup=3, sync=clock.sync, clock=clock)
    assert r['batch_n'] == [32, 32, 6] and r['n_images'] == 70
    assert r['total_seconds'] == pytest.approx(0.070) and r['ms_per_image'] == pytest.approx(1.0)
    assert calls['n'] == 3 + 3


# ------------------------------------------------------------------------------------------------ cells from raw files
def write_raw(root, c_secs, t_secs, heads_secs, e2e=None, n=(32, 32), discarded=(), enc='clip_b32', stray=None,
              passes_over=None):
    """Raw files of one encoder X and the teacher: seconds per repeat for X's default pass (c_secs), each of the three
    teacher passes (t_secs: dict view -> seconds per repeat) and X's three networks (heads_secs). Two batches of n."""
    e2e = e2e or {'x': 3.0, 'teacher': 9.0}
    batches = root / 'probe' / 'batches'
    batches.mkdir(parents=True)
    passes = {}

    def put(name, kind, secs, repeats=True):
        entry = {'encoder': enc if name != 'qwen' and not name.startswith('qwen_') and name != 'teacher' else 'qwen',
                 'kind': kind, 'repeats': {}, 'e2e': None}
        for k, s in enumerate(secs if repeats else []):
            r = rec(name, [s / len(n)] * len(n), n=list(n), kind=kind, repeat=k)
            (batches / f'{name}__r{k}.json').write_text(json.dumps(r))
            entry['repeats'][str(k)] = {'file': f'{name}__r{k}.json', 'ms_per_image': r['ms_per_image'],
                                        'total_seconds': r['total_seconds'], 'attempt': 1}
        passes[name] = entry
        return entry

    put(enc, 'model', c_secs)
    put(f'heads_{enc}', 'heads', heads_secs)
    for view, s in t_secs.items():
        put(f'qwen_{view}', 'model', s)
    for name, total in (('x', e2e['x']), ('teacher', e2e['teacher'])):
        pname = enc if name == 'x' else 'teacher'
        r = rec(pname, [total / len(n)] * len(n), n=list(n), kind='e2e', repeat=None)
        (batches / f'{pname}__e2e.json').write_text(json.dumps(r))
        passes.setdefault(pname, {'encoder': 'qwen', 'kind': 'model', 'repeats': {}, 'e2e': None})
        passes[pname]['e2e'] = {'file': f'{pname}__e2e.json', 'ms_per_image': r['ms_per_image'],
                                'total_seconds': r['total_seconds'], 'attempt': 1}
    doc = {'meta': {'n_images': sum(n), 'batch_size': 32, 'repeats': 3, 'warmup_batches': 3, 'smoke': False},
           'passes': passes, 'repeats_discarded': list(discarded)}
    if passes_over:
        doc.update(passes_over)
    (root / 'probe' / 'passes.json').write_text(json.dumps(doc))
    if stray:
        (batches / stray[0]).write_text(json.dumps(stray[1]))
    return doc


Z = 1e-9                                                       # a network time that is not exactly zero
T3 = {'emotion_focus': [1.0, 1.2, 1.1], 'style_word': [1.0, 1.0, 1.0], 'genre_word': [0.9, 1.0, 1.0]}


def cells(root, **kw):
    kw.setdefault('smoke', True)                                  # the fake sets are small
    return P.build_cells(root, **kw)


def test_cell_is_rebuilt_from_the_raw_files_alone(tmp_path):
    # 64 images; X pass seconds per repeat 0.064, 0.128, 0.096 -> 1, 2, 1.5 ms/img; networks 0.0064 s -> 0.1 ms/img
    write_raw(tmp_path, c_secs=[0.064, 0.128, 0.096], t_secs=T3, heads_secs=[0.0064, 0.0064, 0.0128],
              e2e={'x': 0.32, 'teacher': 0.96})
    cell = {c['encoder']: c for c in cells(tmp_path, encoders=['clip_b32'])}['clip_b32']
    assert cell['sc'] == 'SC15'
    assert cell['cstar_ms'] == pytest.approx([1.0 + 0.1, 2.0 + 0.1, 1.5 + 0.2])        # X pass (r) + networks (r)
    assert cell['teacher_ms'] == pytest.approx([2.9 / 0.064, 3.2 / 0.064, 3.1 / 0.064])  # sum of three passes (r)
    assert cell['cstar_ms_max'] == pytest.approx(2.1) and cell['teacher_ms_min'] == pytest.approx(2.9 / 0.064)
    assert cell['cstar_ms_mean'] == pytest.approx((1.1 + 2.1 + 1.7) / 3)
    assert cell['teacher_ms_mean'] == pytest.approx((2.9 + 3.2 + 3.1) / 3 / 0.064)
    assert cell['cstar_ms_e2e'] == pytest.approx(1000 * 0.32 / 64 + (0.1 + 0.1 + 0.2) / 3)
    assert cell['teacher_ms_e2e'] == pytest.approx(1000 * 0.96 / 64)
    assert cell['repeats_discarded'] == 0
    assert O.check_cost_cell(cell) == ('SC15', 'clip_b32')       # the outcome code accepts it, maxima follow the timings


def test_sc15_reads_the_slowest_cstar_repeat_against_the_fastest_teacher_repeat(tmp_path):
    # c*: 10, 10, 100 ms/img (mean 40); teacher: 50, 60, 70 (mean 60, min 50). The mean favours c*, the rule says fails.
    write_raw(tmp_path, c_secs=[0.64, 0.64, 6.4], t_secs={'emotion_focus': [1.6, 1.6, 1.6], 'style_word': [0.8, 1.2, 1.6],
                                                         'genre_word': [0.8, 1.0, 1.28]},
              heads_secs=[Z, Z, Z], e2e={'x': 1.0, 'teacher': 3.0}, n=(32, 32))
    cell = cells(tmp_path, encoders=['clip_b32'])[0]
    assert cell['cstar_ms_max'] == pytest.approx(100.0, abs=1e-3)
    assert cell['teacher_ms_min'] == pytest.approx(1000 * (1.6 + 0.8 + 0.8) / 64, abs=1e-3)     # 50
    assert cell['cstar_ms_mean'] < cell['teacher_ms_mean']
    assert O.cost_verdict(cell) == O.FAILS
    # swap in a faster c*: slowest repeat 40 ms < fastest teacher 50 ms -> holds
    import shutil
    shutil.rmtree(tmp_path / 'probe')
    write_raw(tmp_path, c_secs=[0.64, 0.64, 2.56], t_secs={'emotion_focus': [1.6, 1.6, 1.6], 'style_word': [0.8, 1.2, 1.6],
                                                          'genre_word': [0.8, 1.0, 1.28]},
              heads_secs=[Z, Z, Z], e2e={'x': 1.0, 'teacher': 3.0})
    assert O.cost_verdict(cells(tmp_path, encoders=['clip_b32'])[0]) == O.HOLDS


def test_repeats_pair_by_index_not_by_sorted_order(tmp_path):
    # Networks repeat 0 is slowest: c*(r) = X(r) + networks(r) must preserve the repeat index.
    write_raw(tmp_path, c_secs=[0.064, 0.064, 0.064], t_secs=T3, heads_secs=[0.64, Z, Z])
    cell = cells(tmp_path, encoders=['clip_b32'])[0]
    assert cell['cstar_ms'] == pytest.approx([11.0, 1.0, 1.0], abs=1e-3)


def test_rebuild_ignores_files_that_are_not_accepted_repeats(tmp_path):
    write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3,
              stray=('clip_b32__r0__a1.json', rec('clip_b32', [9.0, 9.0])))
    cell = cells(tmp_path, encoders=['clip_b32'])[0]
    assert cell['cstar_ms'] == pytest.approx([1.0] * 3, abs=1e-3)


def test_rebuild_refuses_a_passes_json_that_disagrees_with_the_batch_files(tmp_path):
    doc = write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3)
    doc['passes']['clip_b32']['repeats']['1']['ms_per_image'] = 0.5
    (tmp_path / 'probe' / 'passes.json').write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='disagrees'):
        cells(tmp_path, encoders=['clip_b32'])


def test_rebuild_refuses_missing_repeat_wrong_image_count_and_missing_pass(tmp_path):
    write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3)
    (tmp_path / 'probe' / 'batches' / 'clip_b32__r2.json').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        cells(tmp_path, encoders=['clip_b32'])
    other = tmp_path / 'o2'
    other.mkdir()
    write_raw(other, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3, n=(32, 16))     # 48 images, not 2,000
    with pytest.raises(AssertionError, match='2000'):
        cells(other, encoders=['clip_b32'], smoke=False)
    assert cells(other, encoders=['clip_b32'], smoke=True)                                # smoke allows it
    third = tmp_path / 'o3'
    third.mkdir()
    write_raw(third, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3)
    with pytest.raises(AssertionError):
        cells(third, encoders=['siglip2'], smoke=True)                                    # that encoder was not timed


def test_discards_are_counted_per_cell_for_the_passes_it_uses(tmp_path):
    log = [dict(**{'pass': 'clip_b32', 'repeat': 1, 'attempt': 1}), dict(**{'pass': 'qwen_style_word', 'repeat': 0,
                                                                            'attempt': 1}),
           dict(**{'pass': 'clip_l14', 'repeat': 0, 'attempt': 1}), dict(**{'pass': 'heads_clip_b32', 'repeat': 2,
                                                                          'attempt': 1})]
    write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3, discarded=log)
    cell = cells(tmp_path, encoders=['clip_b32'])[0]
    assert cell['repeats_discarded'] == 3                         # its pass, its networks, one teacher pass; not clip_l14
    assert sorted(d['pass'] for d in cell['repeats_discarded_log']) == ['clip_b32', 'heads_clip_b32', 'qwen_style_word']


def test_qwen_cell_uses_its_own_default_pass_and_the_same_teacher(tmp_path):
    write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3, enc='qwen')
    cell = cells(tmp_path, encoders=['qwen'])[0]
    assert cell['encoder'] == 'qwen' and cell['cstar_ms'] == pytest.approx([1.0] * 3, abs=1e-3)


# ------------------------------------------------------------------------------------------------ gate (viii)
@pytest.mark.parametrize('ms,ok', [(86.3, True), (103.5, True), (103.57, False), (69.05, True), (69.0, False),
                                   (200.0, False)])
def test_gate_viii_is_within_20_percent_of_86_3(ms, ok):
    g = P.gate_viii(ms)
    assert g['pass'] is ok and g['reference_ms'] == 86.3 and g['tolerance'] == 0.20
    assert g['qwen_default_e2e_ms'] == ms


def test_gate_viii_refuses_nonsense_timings():
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(AssertionError):
            P.gate_viii(bad)


# ------------------------------------------------------------------------------------------------ names
def test_pass_names_and_the_teacher_passes():
    assert P.TEACHER_PASSES == ('qwen_emotion_focus', 'qwen_style_word', 'qwen_genre_word')
    assert P.default_pass('qwen') == 'qwen' and P.default_pass('clip_b32') == 'clip_b32'
    assert P.passes_of('qwen')[0] == 'qwen' and set(P.TEACHER_PASSES) <= set(P.passes_of('qwen'))
    assert P.passes_of('clip_b16') == ['clip_b16']
    assert set(P.ENCODERS) == set(PP.ENCODERS)


def test_a_full_2000_image_set_is_accepted_outside_smoke_and_standins_are_not(tmp_path):
    n = tuple(batch_sizes(2000, 32))
    doc = write_raw(tmp_path, c_secs=[2.0, 2.0, 2.0], t_secs=T3, heads_secs=[Z] * 3, n=n)
    cell = P.build_cells(tmp_path, encoders=['clip_b32'], smoke=False)[0]
    assert cell['n_images'] == 2000 and cell['cstar_ms'] == pytest.approx([1.0] * 3, abs=1e-3)
    doc['passes']['heads_clip_b32']['standin'] = True
    (tmp_path / 'probe' / 'passes.json').write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='stand-in'):
        P.build_cells(tmp_path, encoders=['clip_b32'], smoke=False)
    assert P.build_cells(tmp_path, encoders=['clip_b32'], smoke=True)


def test_cost_table_whole_set_seconds_rates_and_passes_per_aspect(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, 'MMART_PRE', tmp_path / 'prep')
    write_raw(tmp_path, c_secs=[0.064, 0.128, 0.192], t_secs=T3, heads_secs=[0.0064, 0.0128, 0.0192],
              e2e={'x': 0.32, 'teacher': 0.96})
    table = P.cost_table(tmp_path, cells(tmp_path, encoders=['clip_b32']))
    whole = table['whole_set_16384']['clip_b32']
    assert whole == pytest.approx({'cstar_s': 2.2 * 16384 / 1000,
                                  # Three instructed passes: means 1.1 + 1 + 2.9/3 seconds over 64 images.
                                  'teacher_s': (1.1 + 1 + 2.9 / 3) / 64 * 16384,
                                  'cstar_e2e_s': 5.2 * 16384 / 1000,
                                  'teacher_e2e_s': 15 * 16384 / 1000, 'paintings': 16384})
    expected_ms = {'clip_b32': 2, 'heads_clip_b32': 0.2, 'qwen_emotion_focus': 17.1875,
                   'qwen_style_word': 15.625, 'qwen_genre_word': 1000 * 2.9 / 3 / 64}
    for name, ms in expected_ms.items():
        assert table['passes'][name]['ms_per_image_mean'] == pytest.approx(ms)
        assert table['passes'][name]['images_per_second'] == pytest.approx(1000 / ms)
    for name, ms in [('clip_b32', 5), ('teacher', 15)]:
        assert table['passes'][name]['ms_per_image_e2e'] == pytest.approx(ms)
        assert table['passes'][name]['images_per_second_e2e'] == pytest.approx(1000 / ms)
    assert table['passes_per_aspect'] == {'cstar': 1 / 3, 'teacher': 1}


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
                           get_device_name=lambda _: 'fake GPU')
    torch = SimpleNamespace(cuda=cuda, get_num_threads=lambda: 8, __version__='fake')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='fake'))
    return torch


@pytest.mark.parametrize('command', ['passes', 'heads'])
@pytest.mark.parametrize('variable,value', [('OMP_NUM_THREADS', None), ('OMP_NUM_THREADS', '4'),
                                            ('MKL_NUM_THREADS', None), ('MKL_NUM_THREADS', '4')])
def test_commands_require_store_thread_settings_before_loading_models(monkeypatch, command, variable, value):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '8')
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    with pytest.raises(AssertionError, match=variable + ' must be set to 8'):
        if command == 'passes':
            P.cmd_passes('clip_b32', 64, False)
        else:
            P.cmd_heads(['clip_b32'], 64, False)


def test_cpu_load_is_logged_and_saved_for_model_and_e2e_without_discarding(tmp_path, monkeypatch, capsys, fake_torch):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    # Pin recording the actual torch value, independently of the required environment setting.
    fake_torch.get_num_threads = lambda: 7
    clock = FakeClock()
    monkeypatch.setattr(P.time, 'perf_counter', clock)
    loads = iter([(100, 90, 80), (99, 89, 79), (98, 88, 78), (97, 87, 77)])
    monkeypatch.setattr(P.os, 'getloadavg', lambda: next(loads))

    def encode(batch):
        clock.t += 0.1

    entry, discards = P.run_one_pass('clip_b32', 'default', encode, list(range(64)), 1,
                                     e2e=lambda: rec('clip_b32', [0.1, 0.2], kind='e2e', repeat=None),
                                     poll=lambda: [], encoder='clip_b32')
    assert discards == []
    P.update_passes(['a', 'b'], 64, lambda doc: doc['passes'].update(clip_b32=entry))
    doc = P._read(tmp_path / 'probe' / 'passes.json')
    assert doc['meta']['torch_num_threads'] == 7
    for summary, tag, before, after in [(entry['repeats']['0'], 'r0', [100, 90, 80], [99, 89, 79]),
                                        (entry['e2e'], 'e2e', [98, 88, 78], [97, 87, 77])]:
        raw = P._read(tmp_path / 'probe' / 'batches' / summary['file'])
        saved = doc['passes']['clip_b32']['repeats']['0'] if tag == 'r0' else doc['passes']['clip_b32']['e2e']
        for record in (raw, summary, saved):
            assert record['cpu_load_before'] == before and record['cpu_load_after'] == after
        assert P.read_timing(tmp_path, 'clip_b32', 0 if tag == 'r0' else 'e2e', saved, 64, True) > 0
    output = capsys.readouterr().out
    for load in ([100, 90, 80], [99, 89, 79], [98, 88, 78], [97, 87, 77]):
        assert str(load) in output


@pytest.mark.parametrize('failure', ['model', 'e2e'])
def test_exhausted_attempts_persist_current_and_earlier_repeat_discards(tmp_path, monkeypatch, fake_torch, failure):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    clock = FakeClock()
    monkeypatch.setattr(P.time, 'perf_counter', clock)
    monkeypatch.setattr(P.os, 'getloadavg', lambda: (1, 2, 3))
    original = P.run_repeat

    def without_wait(*args, **kw):
        kw['sleep'] = lambda _: None
        return original(*args, **kw)

    monkeypatch.setattr(P, 'run_repeat', without_wait)
    polls = iter([[(999, 1)], [], [], []] + [[(999, 1)]] * 10)

    def encode(batch):
        clock.t += 0.1

    with pytest.raises(RuntimeError, match='all 5 attempts'):
        P.run_one_pass('clip_b32', 'default', encode, list(range(64)), 2 if failure == 'model' else 1,
                       e2e=(lambda: rec('clip_b32', [0.1], kind='e2e', repeat=None)) if failure == 'e2e' else None,
                       poll=lambda: next(polls), encoder='clip_b32', on_discard=P.discard_logger(['a', 'b'], 64))
    doc = P._read(tmp_path / 'probe' / 'passes.json')
    discarded = doc['repeats_discarded']
    assert len(discarded) == 6
    assert [(d['repeat'], d['attempt']) for d in discarded] == [(0, 1)] + [
        (1 if failure == 'model' else 'e2e', i) for i in range(1, 6)]
    for d in discarded:
        raw = P._read(tmp_path / 'probe' / 'discarded' / d['file'])
        assert raw['total_seconds'] == d['total_seconds'] > 0
        assert raw['cpu_load_before'] == d['cpu_load_before'] == [1, 2, 3]
    assert len(list((tmp_path / 'probe' / 'discarded').glob('*.json'))) == 6


def test_real_checkpoint_requires_a_matching_sha256_sidecar(tmp_path, monkeypatch, fake_torch):
    monkeypatch.setattr(P, 'weights_dir', lambda _: tmp_path)
    path = tmp_path / f'{PP.ASPECTS[0]}__s0.pt'
    path.write_bytes(b'checkpoint')
    monkeypatch.setitem(sys.modules, 'rb_rows', SimpleNamespace(REFERENCE='ref', recipe=lambda _: {'base': 'base'}))
    monkeypatch.setitem(sys.modules, 'rb_student', SimpleNamespace(
        build_model=lambda *a, **kw: (None, {aspect: object() for aspect in PP.ASPECTS})))
    # Even --standin must not bypass verification of a real checkpoint.
    with pytest.raises(AssertionError, match='missing sha256 sidecar'):
        P.load_networks('clip_b32', 'cpu', standin=True)
    P.write_json(path.with_suffix('.json'), {'weights_sha256': 'wrong'})
    with pytest.raises(AssertionError, match='sidecar sha256 differs'):
        P.load_networks('clip_b32', 'cpu', standin=False)
    loaded = []
    fake_torch.load = lambda *a, **kw: loaded.append(a[0]) or {}
    for aspect in PP.ASPECTS:
        checkpoint = tmp_path / f'{aspect}__s0.pt'
        checkpoint.write_bytes(b'checkpoint')
        P.write_json(checkpoint.with_suffix('.json'), {'weights_sha256': hashlib.sha256(b'checkpoint').hexdigest()})
    # A dict models the real view's 'img' lookup as well as state loading and eval.
    class View(dict):
        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(sys.modules['rb_student'], 'build_model',
                        lambda *a, **kw: (None, {aspect: View(img='net') for aspect in PP.ASPECTS}))
    nets, provenance = P.load_networks('clip_b32', 'cpu', standin=False)
    assert nets == ['net'] * 3 and len(loaded) == len(provenance) == 3


def test_teacher_checks_instructions_once_before_the_timed_loop(tmp_path, monkeypatch, fake_torch):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    clock = FakeClock()
    monkeypatch.setattr(P.time, 'perf_counter', clock)
    monkeypatch.setattr(P.os, 'getloadavg', lambda: (1, 2, 3))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '8')
    fake_torch.cuda.is_available = lambda: True
    instructions = {v: f'exact instruction for {v}' for v in P.TEACHER_VIEWS}
    checks, calls = [], []

    def check(enc, text):
        assert enc.instruction == text
        checks.append(text)
        clock.t += 0.0012

    def encode(ims, batch_size):
        assert checks == list(instructions.values()), 'all views must be checked before timing'
        calls.append(enc.instruction)
        clock.t += 0.1

    enc = SimpleNamespace(instruction='default instruction', encode_images=encode)
    monkeypatch.setitem(sys.modules, 'mp_extract', SimpleNamespace(
        held_identity=lambda v: {'instruction': instructions[v]}, assert_image_instruction=check))
    monkeypatch.setitem(sys.modules, 'pp_extract', SimpleNamespace(load_encoder=lambda _: (enc, {}, None)))
    monkeypatch.setattr(P, 'smoke_guard', lambda _: None)
    monkeypatch.setattr(P, 'file_done', lambda name: name != P.TEACHER_E2E)
    monkeypatch.setattr(P, 'any_file', lambda _: False)
    monkeypatch.setattr(P, 'probe_keys', lambda n: [f'key{i}' for i in range(n)])
    monkeypatch.setattr(P, 'image_paths', lambda keys: keys)
    monkeypatch.setattr(P, 'image_path_params', lambda _: (1, 2))
    monkeypatch.setattr(P, 'decode_all', lambda paths: [SimpleNamespace(size=(1, 1)) for _ in paths])
    monkeypatch.setattr(P, 'load_images_from_files', lambda batch: batch)
    monkeypatch.setattr(P, 'poll_gpu', lambda: [])
    assert P.cmd_passes('qwen', 64, resume=True) == 0
    assert calls == list(instructions.values()) * 2
    raw = P._read(tmp_path / 'probe' / 'batches' / 'teacher__e2e.json')
    assert raw['total_seconds'] == pytest.approx(0.6)
    for v in P.TEACHER_VIEWS:
        assert raw['pass_seconds'][v] == pytest.approx([0.1, 0.1])
    entry = P._read(tmp_path / 'probe' / 'passes.json')['passes']['teacher']['e2e']
    assert entry['cpu_load_before'] == entry['cpu_load_after'] == [1, 2, 3]


# ------------------------------------------------------------------------------------------------ cmd_cells (fix wave B)
def clone_pass(root, src, dst):
    """Give pass `dst` the raw files of pass `src` (same timings), in passes.json and on disk."""
    doc = json.loads((root / 'probe' / 'passes.json').read_text())
    entry = json.loads(json.dumps(doc['passes'][src]))
    entry['encoder'] = dst.replace('heads_', '')
    batches = root / 'probe' / 'batches'
    for k, r in entry['repeats'].items():
        raw = json.loads((batches / r['file']).read_text())
        raw['pass'] = dst
        r['file'] = f'{dst}__r{k}.json'
        (batches / r['file']).write_text(json.dumps(raw))
    if entry.get('e2e'):
        raw = json.loads((batches / entry['e2e']['file']).read_text())
        raw['pass'] = dst
        entry['e2e']['file'] = f'{dst}__e2e.json'
        (batches / entry['e2e']['file']).write_text(json.dumps(raw))
    doc['passes'][dst] = entry
    (root / 'probe' / 'passes.json').write_text(json.dumps(doc))


def qwen_root(tmp_path, monkeypatch, others=('clip_b32',), heads_for=None):
    """Raw files for Qwen (its default pass end to end at the prep log's 86.3 ms) and `others` (default and heads passes)."""
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'SMOKE', 1)
    write_raw(tmp_path, c_secs=[0.064] * 3, t_secs=T3, heads_secs=[Z] * 3, e2e={'x': 0.0863 * 64, 'teacher': 9.0}, enc='qwen')
    for e in others:
        clone_pass(tmp_path, 'qwen', e)
        if heads_for is None or e in heads_for:
            clone_pass(tmp_path, 'heads_qwen', f'heads_{e}')
    return tmp_path


def test_cells_writes_gate_viii_first_and_leaves_out_encoders_without_passes(tmp_path, monkeypatch, capsys):
    root = qwen_root(tmp_path, monkeypatch, others=('clip_b32', 'dinov2_large'), heads_for=('clip_b32',))
    assert P.cmd_cells(None) == 0
    out = capsys.readouterr().out
    assert json.loads((root / 'stage0' / 'gate_viii.json').read_text())['pass'] is True
    got = json.loads((root / 'stage3' / 'cells_cost.json').read_text())
    assert [c['encoder'] for c in got] == ['qwen', 'clip_b32']                     # in ENCODERS order
    for e in ('siglip2', 'siglip1', 'clip_l14', 'dinov2_large', 'clip_b16'):
        assert f'SC15 {e}: LEFT OUT' in out
    assert 'SC15 qwen: LEFT OUT' not in out and 'SC15 clip_b32: LEFT OUT' not in out
    assert (root / 'stage3' / 'cost_table.json').exists()


def test_cells_gate_viii_is_written_even_when_the_cells_cannot_be_built(tmp_path, monkeypatch):
    root = qwen_root(tmp_path, monkeypatch, others=())
    doc = json.loads((root / 'probe' / 'passes.json').read_text())
    del doc['passes']['qwen_style_word']                                  # a teacher pass is required
    (root / 'probe' / 'passes.json').write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='qwen_style_word'):
        P.cmd_cells(None)
    assert json.loads((root / 'stage0' / 'gate_viii.json').read_text())['pass'] is True
    assert not (root / 'stage3' / 'cells_cost.json').exists()


def test_cells_with_all_seven_present_equal_build_cells_byte_for_byte(tmp_path, monkeypatch):
    others = tuple(e for e in PP.ENCODERS if e != 'qwen')
    root = qwen_root(tmp_path, monkeypatch, others=others)
    assert P.cmd_cells(None) == 0
    expected = P.build_cells(root, encoders=None, smoke=True)
    assert len(expected) == 7
    P.write_json(root / 'expected.json', expected)
    assert (root / 'stage3' / 'cells_cost.json').read_bytes() == (root / 'expected.json').read_bytes()


def test_a_missing_cost_cell_is_excused_only_by_a_failed_cstar_run():
    """The encoder left out of the cells stays loud in `decide` unless runs[X].cstar (or gate ii, iii, viii) failed."""
    import test_pp_outcome as TO
    with pytest.raises(AssertionError, match='cost cells missing that no failed gate excuses'):
        TO.run(drop=[('SC15', 'clip_l14')])                              # an old encoder has no c* run to excuse it
    with pytest.raises(AssertionError, match='cost cells missing that no failed gate excuses'):
        TO.run(drop=[('SC15', 'dinov2_large')])
    status = TO.failed('runs', 'dinov2_large', 'cstar')
    status['runs']['dinov2_large']['photo'] = 'failed'
    res = TO.run(drop=[('SC15', 'dinov2_large')], status=status, over=None)
    assert res['verdicts']['cost']['dinov2_large']['verdict'] == 'inconclusive'
