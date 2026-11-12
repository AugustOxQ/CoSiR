"""Tests of r29_model: r14's factory with an independent caption input width, and the sidecar-driven view loader."""
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_model as M  # noqa: E402
import rb_student  # noqa: E402
import rb_rows  # noqa: E402

ASPECTS = ['emotion', 'style', 'genre']
ROW = rb_rows.recipe(rb_rows.REFERENCE)['base']
CPU = torch.device('cpu')


def r14(width, seed=0):
    torch.manual_seed(seed)
    return rb_student.build_model(ROW, ASPECTS, CPU, width=width)


def r29(width, cap_width, seed=0, **kw):
    torch.manual_seed(seed)
    return M.build_model(ASPECTS, CPU, width=width, cap_width=cap_width, **kw)


def named(models, views):
    return [(k, v) for k, v in models.state_dict().items()], [
        (a, [(k, v) for k, v in views[a].state_dict().items()]) for a in ASPECTS]


@pytest.mark.parametrize('width', [1152, 2048])
def test_cap_width_2048_is_r14s_model_same_seed(width):
    a, b = named(*r14(width)), named(*r29(width, 2048))
    assert [k for k, _ in a[0]] == [k for k, _ in b[0]]
    for (_, x), (_, y) in zip(a[0], b[0]):
        assert torch.equal(x, y)
    for (_, xs), (_, ys) in zip(a[1], b[1]):
        assert [k for k, _ in xs] == [k for k, _ in ys]
        assert all(torch.equal(x, y) for (_, x), (_, y) in zip(xs, ys))
    pa, pb = list(r14(width)[0].parameters()), list(r29(width, 2048)[0].parameters())
    assert [tuple(p.shape) for p in pa] == [tuple(p.shape) for p in pb]   # optimizer parameter order


def test_cap_width_1152_changes_only_the_caption_trunks_first_layer():
    base, new = r14(1152)[0].state_dict(), r29(1152, 1152)[0].state_dict()
    assert list(base) == list(new)
    differ = [k for k in base if base[k].shape != new[k].shape]
    assert differ == ['cap.trunk.0.weight']
    assert tuple(new['cap.trunk.0.weight'].shape) == (1024, 1152) and tuple(base['cap.trunk.0.weight'].shape) == (1024, 2048)
    # the image side is created first, so under one seed it is the same tensors; outputs stay 2048
    assert all(torch.equal(base[k], new[k]) for k in base if k.startswith('img.'))
    assert all(tuple(v.shape)[0] == 2048 for k, v in new.items() if k.startswith('cap.heads') and k.endswith('weight'))


def test_forward_shapes_at_1152_and_the_view_keeps_shared_parameters():
    models, views = r29(1152, 1152)
    out = views['style']['cap'](torch.randn(4, 1152))
    assert out.shape == (4, 2048)
    assert views['style']['cap'].trunk is models['cap'].trunk
    with pytest.raises(RuntimeError):
        views['style']['cap'](torch.randn(4, 2048))


def test_image_width_2048_with_other_caption_width_builds():
    models, _ = r29(2048, 768)
    assert tuple(models['img'].trunk[0].weight.shape) == (1024, 2048)
    assert tuple(models['cap'].trunk[0].weight.shape) == (1024, 768)


# --- load_view --------------------------------------------------------------------------------------------------

def save(tmp_path, width, cap_width, claimed_cap=None, source='siglip1', name='emotion__s0'):
    torch.manual_seed(0)
    _, views = M.build_model(ASPECTS, CPU, width=width, cap_width=cap_width)
    w = tmp_path / f'{name}.pt'
    state = views['emotion'].state_dict()
    torch.save(state, w)
    rb = dict(width=width, cap_width=cap_width if claimed_cap is None else claimed_cap, cap_source=source,
              cap_store_sha256='ab' * 32)
    w.with_suffix('.json').write_text(json.dumps(dict(config=dict(aspect='emotion', seed=0, rb=rb))))
    return w, state


def test_load_view_roundtrip(tmp_path):
    w, state = save(tmp_path, 1152, 1152)
    view = M.load_view(w)
    assert not view.training
    for k, v in view.state_dict().items():
        assert torch.equal(v, state[k])
    assert view['cap'](torch.randn(2, 1152)).shape == (2, 2048)


def test_load_view_refuses_sidecar_cap_width_that_disagrees_with_weights(tmp_path):
    w, _ = save(tmp_path, 1152, 1152, claimed_cap=2048, source='qwen')   # consistent with its source, not with the weights
    with pytest.raises(AssertionError, match='weights take'):
        M.load_view(w)
    w, _ = save(tmp_path, 1152, 2048, claimed_cap=1152, name='emotion__s1')
    with pytest.raises(AssertionError, match='weights take'):
        M.load_view(w)


def test_load_view_refuses_image_width_that_disagrees(tmp_path):
    w, _ = save(tmp_path, 1152, 1152)
    rb = json.loads(w.with_suffix('.json').read_text())
    rb['config']['rb']['width'] = 2048
    w.with_suffix('.json').write_text(json.dumps(rb))
    with pytest.raises(AssertionError, match='width'):
        M.load_view(w)


def test_load_view_refuses_sidecar_without_cap_fields(tmp_path):
    w, _ = save(tmp_path, 1152, 1152)
    d = json.loads(w.with_suffix('.json').read_text())
    del d['config']['rb']['cap_width']
    w.with_suffix('.json').write_text(json.dumps(d))
    with pytest.raises(AssertionError):
        M.load_view(w)


def test_load_view_refuses_cap_width_that_is_not_its_sources(tmp_path):
    w, _ = save(tmp_path, 1152, 1152, claimed_cap=2048)      # siglip1 takes 1152
    with pytest.raises(AssertionError, match="input width"):
        M.load_view(w)
