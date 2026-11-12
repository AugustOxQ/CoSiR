"""r29 caption-width model (ticket 01; spec §3.2: only the caption input and its width change).

`build_model` is r14's student factory (`rb_student.build_model`) with the caption trunk's first layer taking its own
width `cap_width`. At `cap_width=2048` it returns r14's own objects. `load_view` rebuilds one saved best view from its
weights file and the sidecar JSON beside it, and refuses a sidecar that disagrees with the weights' layer shapes.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402
sys.path.insert(0, str(P.R28_ROUND))
import r28_trainer  # noqa: E402  (puts r14's and the older rounds' folders on sys.path, as r28 does)
import rb_rows  # noqa: E402
import rb_student  # noqa: E402
import ms_train  # noqa: E402

REFERENCE_ROW = rb_rows.recipe(rb_rows.REFERENCE)['base']
SIDECAR_KEYS = ('width', 'cap_width', 'cap_source', 'cap_store_sha256')   # in the sidecar's config['rb'] block
SHEET_WIDTH = ms_train.CONFIG['architecture'][0]
assert tuple(r28_trainer.B.ASPECTS) == P.ASPECTS
assert SHEET_WIDTH == 2048 and (rb_student.HIDDEN, rb_student.OUT) == (1024, P.OUTPUT_WIDTH)


def build_model(*args, width, cap_width):
    """`build_model(aspects, device, *, width, cap_width)`; also `build_model(row, aspects, device, ...)` so that it can
    stand in for `ms_train.build_model` (patched with `partial(build_model, width=.., cap_width=..)`, as r14 does)."""
    row, aspects, device = args if len(args) == 3 else (REFERENCE_ROW, *args)
    if cap_width == SHEET_WIDTH:
        return rb_student.build_model(row, aspects, device, width=width)       # r14's objects and creation order
    assert row['share'] == 'trunk' and row['form'] == 'a', 'only the trunk-shared form-a cohort is a student'
    img = ms_train.TrunkSide(aspects) if width == SHEET_WIDTH else rb_student.WideTrunkSide(aspects, width)
    models = nn.ModuleDict({'img': img.to(device), 'cap': rb_student.WideTrunkSide(aspects, cap_width).to(device)})
    views = {a: nn.ModuleDict({side: ms_train.TrunkHead(models[side], a) for side in ('img', 'cap')}) for a in aspects}
    return models, views


def read_sidecar(weights):
    """(config, rb block) of the sidecar JSON beside a weights file; the width fields must all be there."""
    info = json.loads(Path(weights).with_suffix('.json').read_text())
    config = info['config']
    rb = config['rb']
    missing = [k for k in SIDECAR_KEYS if k not in rb]
    assert not missing, f'{weights}: sidecar rb block lacks {missing}'
    assert rb['cap_source'] in P.CAPTION_SOURCES, f'unknown caption source {rb["cap_source"]!r}'
    assert rb['cap_width'] == P.CAPTION_SOURCES[rb['cap_source']][1], \
        f'{weights}: sidecar cap_width {rb["cap_width"]} is not {rb["cap_source"]}\'s input width'
    return config, rb


def load_view(weights, device='cpu'):
    """One saved best view (`nn.ModuleDict` of `img` and `cap`, eval mode) of `<aspect>__s<seed>.pt`, built at the widths
    its sidecar records. The sidecar's `cap_width` and `width` must equal the weights' first-layer input sizes."""
    config, rb = read_sidecar(weights)
    device = torch.device(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    got_cap, got_img = state['cap.trunk.0.weight'].shape[1], state['img.trunk.0.weight'].shape[1]
    assert got_cap == rb['cap_width'], f'{weights}: sidecar cap_width {rb["cap_width"]}, weights take {got_cap}'
    assert got_img == rb['width'], f'{weights}: sidecar width {rb["width"]}, weights take {got_img}'
    _, views = build_model(list(P.ASPECTS), device, width=rb['width'], cap_width=rb['cap_width'])
    view = views[config['aspect']]       # KeyError for an aspect outside the cohort
    view.load_state_dict(state)             # strict: every key and shape
    return view.eval()

