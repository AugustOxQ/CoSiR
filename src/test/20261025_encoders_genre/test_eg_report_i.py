"""Tests of the reporter's new parts (ticket 06, fix round): the V5 and X0 `stores` block, the check of a saved
reported block against the verdict block, and the smoke V2 file. Small synthetic files."""
import json
import shutil
from pathlib import Path

import pytest

import eg_paths
import eg_report_i as rep
import eg_v5
import rb_part2_report as rpr

REAL_X0 = Path('/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/stores/x0.json')
STRUCT = {k: True for k in rep.STRUCTURE}


def x0_doc(**change):
    doc = {'pass': True, 'code_commit': 'e4e84a5' + '0' * 33, 'old_commit': '4572189' + '0' * 33, 'n_images': 512,
           'working_tree_dirty': False, 'stores': {
               'clip_b32': {'equal': True, 'new_vs_old': {'max_abs': 0.0, 'rows_affected': 0},
                            'vs_stored': {'new': {'max_abs': 0.0, 'rows_affected': 0},
                                          'old': {'max_abs': 0.0, 'rows_affected': 0}}},
               'siglip2_so400m': {'equal': True, 'new_vs_old': {'max_abs': 0.0, 'rows_affected': 0},
                                  'vs_stored': {'new': {'max_abs': 0.01123, 'rows_affected': 512},
                                                'old': {'max_abs': 0.01123, 'rows_affected': 512}}}}}
    doc.update(change)
    return doc


def entry(top1=None, n=1000, bar_void=False, ok=True, dino=None, failing=()):
    checks = dict(STRUCT)
    out = {'checks': checks, 'pass': ok}
    if top1 is not None:
        out['retrieval'] = {'top1': top1, 'n_paintings': n, 'chance': 1 / n}
        if bar_void is not None:
            out['bar_void'] = bar_void
            checks['retrieval_ge_1pct'] = bar_void or top1 >= .01
    if dino:
        out['dino_vs_archive'] = dino
        checks['dino_cosine_ge_0.99'] = dino['min_cosine'] >= .99 and dino['n'] == dino['archive_n']
    for k in failing:
        checks[k] = False
    out['pass'] = all(checks.values())
    return out


def v5_doc():
    full, a28 = eg_v5.FULL, eg_v5.A28
    cal = {'clip_b32': .159, 'siglip2_so400m': .296}
    stores = {f'{full}/clip_b32': entry(.159), f'{full}/siglip2_so400m': entry(.296),
              f'{a28}/clip_b32': entry(.55, 746), f'{a28}/siglip2_so400m': entry(.85, 746),
              f'{full}/clip_b16': entry(.2), f'{full}/clip_l14': entry(.004),
              f'{full}/dinov2_large': entry(dino={'min_cosine': .995, 'mean_cosine': .999, 'n': 61402,
                                                  'archive_n': 61402}),
              f'{a28}/dinov2_large': entry(failing=('finite',))}
    for s in ('clip_b32', 'siglip2_so400m'):                  # a calibration store has no bar
        for ds in (full, a28):
            stores[f'{ds}/{s}']['checks'].pop('retrieval_ge_1pct', None)
            stores[f'{ds}/{s}'].pop('bar_void', None)
    return {'bar_void': False, 'calibration': cal, 'calibration_a28': {'clip_b32': .55, 'siglip2_so400m': .85},
            'code_commit': 'abc', 'rule_commit': 'def', 'stores': stores}


def write_stores(out, v5=None, x0=None):
    (out / 'stores').mkdir(parents=True, exist_ok=True)
    (out / 'stores/v5.json').write_text(json.dumps(v5 or v5_doc()))
    (out / 'stores/x0.json').write_text(json.dumps(x0 or x0_doc()))


def test_the_stores_block_holds_v5_per_dataset_and_x0_with_the_file_hashes(tmp_path):
    write_stores(tmp_path)
    b = rep.stores_block(tmp_path)
    assert b['v5_sha256'] == rep.sha256_of(tmp_path / 'stores/v5.json') and b['x0_sha256']
    full = b['datasets'][eg_v5.FULL]
    assert full['clip_b32']['retrieval']['top1'] == .159 and 'bar' not in full['clip_b32']['retrieval']
    assert full['clip_b16']['retrieval']['bar_ok'] is True and full['clip_b16']['pass'] is True
    assert full['clip_l14']['retrieval']['bar_ok'] is False and full['clip_l14']['failing_checks'] == ['retrieval_ge_1pct']
    assert full['dinov2_large']['dino_vs_archive']['ok'] is True
    assert full['siglip_so400m'] == dict(role='K8', checked=False)           # no entry: not checked
    assert b['datasets'][eg_v5.A28]['dinov2_large']['failing_checks'] == ['finite']
    assert b['x0']['pass'] is True and b['x0']['stores']['siglip2_so400m']['vs_stored']['new']['rows_affected'] == 512
    md = rep.stores_markdown(b)
    assert 'not checked' in md and 'Decides nothing' in md and '512 of 512' in md
    assert '| clip_l14 | K7 | 0.004 | 1000 | 0.0010 | missed |' in md


def test_a_missing_file_or_a_failed_x0_refuses(tmp_path):
    with pytest.raises(AssertionError, match='v5.json'):
        rep.stores_block(tmp_path)
    write_stores(tmp_path, x0=x0_doc(**{'pass': False}))
    with pytest.raises(AssertionError, match='X0 does not pass'):
        rep.stores_block(tmp_path)
    (tmp_path / 'stores/x0.json').unlink()
    with pytest.raises(AssertionError, match='x0.json'):
        rep.stores_block(tmp_path)


def test_the_real_x0_file_reads_as_the_brief_says():
    if not REAL_X0.exists():
        pytest.skip('no real x0.json')
    x = json.loads(REAL_X0.read_text())
    assert x['pass'] and x['n_images'] == 512 and x['old_commit'].startswith('4572189') and x['code_commit'].startswith('e4e84a5')
    assert x['stores']['siglip2_so400m']['vs_stored']['new'] == {'max_abs': 0.01123046875, 'rows_affected': 512}


# --- a saved reported block against the verdict block now in the file (finding 4) ---------------------------------------

def verdict_blocks(d=.02):
    return {'K6': dict(outcome_id='S1-GO', d=d, interval=[0., .04]), 'K7': dict(outcome_id='S1-NO', d=-.01, interval=[-.03, .01])}


def saved_for(vb, stage=3):
    out = {}
    for c, b in vb.items():
        out[c] = dict(verdict_block_sha256=rpr.block_sha(b))
        if stage == 5:
            out[c].update(verdict_d=b['d'], verdict_outcome=b['outcome_id'], verdict_interval=b['interval'])
    out[rep.STORES] = {}                                              # the stores block is not a candidate
    return out


def test_a_saved_block_made_from_another_verdict_block_is_refused_naming_the_candidate():
    vb = verdict_blocks()
    rep.check_saved_against_verdict(3, saved_for(vb), vb)
    changed = verdict_blocks()
    changed['K7']['d'] = -.02
    with pytest.raises(AssertionError, match='K7'):
        rep.check_saved_against_verdict(3, saved_for(vb), changed)
    rep.check_saved_against_verdict(3, saved_for(vb), changed, skip=['K7'])      # about to be recomputed
    with pytest.raises(AssertionError, match='K6'):
        rep.check_saved_against_verdict(3, saved_for(vb), {'K7': vb['K7']})


def test_at_stage_5_the_saved_d_and_outcome_are_compared_too():
    vb = verdict_blocks()
    saved = saved_for(vb, 5)
    rep.check_saved_against_verdict(5, saved, vb)
    saved['K6']['verdict_d'] = .03                                    # same hash, another recorded d
    with pytest.raises(AssertionError, match='K6'):
        rep.check_saved_against_verdict(5, saved, vb)
    saved = saved_for(vb, 5)
    saved['K7']['verdict_outcome'] = 'S1-GO'
    with pytest.raises(AssertionError, match='K7'):
        rep.check_saved_against_verdict(5, saved, vb)


# --- a smoke V2 file (finding 5) ----------------------------------------------------------------------------------------

def v2_file(out, smoke):
    f = out / 'part2/rederive_v2_stage3.json'
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(stage=3, smoke=smoke, candidates={})))
    return f


def test_a_smoke_v2_file_passes_only_a_smoke_reporter_under_the_scratch_root(tmp_path, monkeypatch):
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path)
    v2_file(tmp_path, True)
    with pytest.raises(AssertionError, match='smoke V2 file'):
        rep.require_v2(3, tmp_path, [], tmp_path / 'v.json')                       # the reporter is not a smoke
    rep.require_v2(3, tmp_path, [], tmp_path / 'v.json', smoke=True)
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path / 'elsewhere')
    with pytest.raises(AssertionError, match='smoke V2 file'):
        rep.require_v2(3, tmp_path, [], tmp_path / 'v.json', smoke=True)           # not under the scratch root
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path)
    v2_file(tmp_path, False)
    rep.require_v2(3, tmp_path, [], tmp_path / 'v.json')                           # a real V2 file passes a real run
