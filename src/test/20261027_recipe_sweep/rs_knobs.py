"""The knob layer (rule §2.2 "Knob notes"): the patches that turn G's sample-scale training (12-27's GS-0.02 through
`rg_train`) into the training of another `rs_sheet` configuration.

`patches(cfg, stack, scale=...)` enters, into an `ExitStack`, one patch per changed part of the frozen modules and
nothing else. For `cfg == config()` it enters none, so G runs the frozen code path untouched (V1a, V4a). Where each
knob lives (facts brief §A):
  k1, k2   the relational term on the emotion / style head: a wrapper of `ab_train.switched_head_loss`, which
           `rg_loss.switched_head_loss` calls for every head but genre (genre keeps its own body)
  k3, k4   `rg_variants.ARMS['GS-0.02']` (lam, tau_t): read by `GenreLoss`; k4 is also the teacher row of k1, k2
  k5       the recipe row's `distill_weight` (all heads) and `rg_variants.TEACHER_WEIGHT`, which only
           `rg_loss`'s assert reads
  k6, k7, k8, k12, k13   `ms_train.CONFIG` (= `itp_train.CONFIG`): lr, weight_decay, batch_size, temperature,
           patience. Readers of 'temperature': `ms_train.symmetric` and `extra_symmetric`, `ags_head.symmetric`
           (all buddy terms) and `itp_train.loss_fn` (not on this path); the direct and relational terms hard-code
           0.05. The export's inference batch reads 'batch_size' too, so k8 pins it to G's value there.
  k9, k10  `ms_train.TrunkSide` (hard-coded 1024 and 0.1); `CONFIG['architecture'/'dropout']` follow for the records
  k11      `rg_train.load_graphs` reads the graph store of `rs_graphs.sample_store(k)` (ticket 03)
The stopping loss of a head with a relational term includes that term (as genre's does at G): the loss that is
trained is the loss that stops.
"""
import sys
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch import nn

import rs_paths  # noqa: F401  (sys.path)
import rs_sheet as S
import ab_train
import eg_train3 as eg
import rg_loss
import rg_train
import rg_variants as V
from ab_paths import ms_train

CONFIG_KEYS = dict(k6='lr', k7='weight_decay', k8='batch_size', k12='temperature', k13='patience')
G_BATCH = S.G['k8']
assert {key: ms_train.CONFIG[key] for key in CONFIG_KEYS.values()} == {
    key: S.G[k] for k, key in CONFIG_KEYS.items()}, 'the sheet\'s G differs from the frozen CONFIG'
assert ms_train.CONFIG['dropout'] == S.G['k10'] and ms_train.CONFIG['architecture'] == [2048, S.G['k9'], 2048]
assert V.ARMS[V.BASE]['lam'] == 0 and V.ARMS['GS-0.02'] == dict(m=1, lam=S.G['k3'], tau_t=S.G['k4'], anchors=0)
assert V.TEACHER_WEIGHT == S.G['k5']


def sample_store(k):
    """The `base` folder of the sample graph store at mutual top-k `k` (ticket 03's `rs_graphs.sample_store`), in the
    form `ab_train.require_graph_store` returns: `<base>/graphs/{manifest.json, <aspect>.npz}`. The one seam to ticket
    03; k = 20 never comes here (the frozen store is read by the frozen loader)."""
    assert k != S.G['k11'], 'k = 20 is the frozen store'
    try:
        import rs_graphs
    except ImportError:
        raise AssertionError(f'k11 = {k}: rs_graphs (ticket 03) is not merged; only k = 20 can run') from None
    return rs_graphs.sample_store(k=k, variant='aspect')


def load_graphs_k(s, recipe, k):
    """`rg_train.load_graphs` with the store of `k` (copied from it: same loader and checks, another base)."""
    if k == S.G['k11']:
        return rg_train.load_graphs(s, recipe)
    base = sample_store(k)
    graphs, manifests = {}, {}
    for aspect in rg_train.ASPECTS:
        graphs[aspect], manifests[aspect] = ms_train.head.load_graph(s, base, aspect, [recipe['base']['graphs'][aspect]])
    return graphs, manifests


def with_relational(original, lam, tau_t):
    """Wrap `ab_train.switched_head_loss`: after the frozen terms, add lam[aspect] * the in-batch relational term
    (rows of the same painting excluded, teacher row at tau_t, student row at 0.05: `rg_loss.relational_loss`) on the
    head's own image output z. z is taken from the frozen call's one forward pass by a hook, as `rg_loss` observes
    it: no second forward, so the dropout draws and the random streams are the frozen call's. Heads without a
    weight (genre never comes here) are the frozen call itself."""
    def switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None,
                           include_mask_penalty=True, *, switches=None):
        weight = lam.get(aspect, 0)
        if not weight:
            return original(models, img, cap, pairs, weights, row, aspect, teacher, extras, include_mask_penalty,
                            switches=switches)
        outputs = []
        handle = models['img'].register_forward_hook(lambda _m, _a, z: outputs.append(z))
        try:
            loss = original(models, img, cap, pairs, weights, row, aspect, teacher, extras, include_mask_penalty,
                            switches=switches)
        finally:
            handle.remove()
        assert len(outputs) == 1 and teacher is not None
        ids = torch.cat(tuple(torch.as_tensor(pairs, device=img.device).unbind(1)))
        return loss + weight * rg_loss.relational_loss(outputs[0], teacher[ids], ids, tau_t)
    return switched_head_loss


def trunk_side(width, dropout):
    """`ms_train.TrunkSide` with this width and dropout: the same layers in the same creation order."""
    class TrunkSide(nn.Module):
        def __init__(self, aspects):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(2048, width), nn.GELU(), nn.Dropout(dropout))
            self.heads = nn.ModuleDict({a: nn.Linear(width, 2048) for a in aspects})
    return TrunkSide


def pinned_batch(infer_img, batch):
    """`eg.infer_img` reads CONFIG['batch_size'] for the export's inference batches; k8 must not change them."""
    def pinned(*args, **kwargs):
        with patch.dict(ms_train.CONFIG, batch_size=batch):
            return infer_img(*args, **kwargs)
    return pinned


def with_direct_weight(resolved, weight):
    """`eg.resolved` with the direct teacher weight of every head set (k5)."""
    def weighted(seed):
        recipe = resolved(seed)
        assert recipe['base']['distill'] == 'direct' and recipe['base']['distill_weight'] == S.G['k5']
        recipe['base']['distill_weight'] = weight
        return recipe
    return weighted


def _plan(cfg):
    """[(description, enter)] for the changed knobs: description {knob: {target: value}} (JSON), enter(stack)
    enters the patch. Empty for G."""
    changed = S.changed(cfg)
    plan = []
    relational = [k for k in ('k1', 'k2') if k in changed]
    if relational:
        lam = dict(emotion=cfg['k1'], style=cfg['k2'])
        target = 'ab_train.switched_head_loss'
        plan.append(({k: {target: f'+ {cfg[k]} * relational term, teacher row at {cfg["k4"]}'} for k in relational},
                     lambda stack: stack.enter_context(patch.object(
                         ab_train, 'switched_head_loss',
                         with_relational(ab_train.switched_head_loss, lam, cfg['k4'])))))
    arm = {key: cfg[k] for k, key in (('k3', 'lam'), ('k4', 'tau_t')) if k in changed}
    if arm:
        target = f'rg_variants.ARMS[{rs_paths.G_ARM}]'
        plan.append(({k: {target: {key: arm[key]}} for k, key in (('k3', 'lam'), ('k4', 'tau_t')) if key in arm},
                     lambda stack: stack.enter_context(patch.dict(V.ARMS[rs_paths.G_ARM], arm))))
    if 'k5' in changed:
        plan.append(({'k5': {'rg_train.resolved': f'direct weight {cfg["k5"]}',
                             'rg_variants.TEACHER_WEIGHT': cfg['k5']}},
                     lambda stack: (stack.enter_context(patch.object(
                         rg_train, 'resolved', with_direct_weight(rg_train.resolved, cfg['k5']))),
                         stack.enter_context(patch.object(V, 'TEACHER_WEIGHT', cfg['k5'])))))
    values = {key: cfg[k] for k, key in CONFIG_KEYS.items() if k in changed}
    description = {k: {'ms_train.CONFIG': {key: cfg[k]}} for k, key in CONFIG_KEYS.items() if k in changed}
    if 'k8' in changed:
        description['k8']['eg_train3.infer_img'] = f'export batches pinned at {G_BATCH}'
    if 'k9' in changed:
        values['architecture'] = [2048, cfg['k9'], 2048]
        description['k9'] = {'ms_train.CONFIG': {'architecture': values['architecture']},
                             'ms_train.TrunkSide': f'width {cfg["k9"]}'}
    if 'k10' in changed:
        values['dropout'] = cfg['k10']
        description['k10'] = {'ms_train.CONFIG': {'dropout': cfg['k10']}, 'ms_train.TrunkSide': f'dropout {cfg["k10"]}'}
    if values:
        # The three parts of this group (CONFIG, the export's batch, the trunk) are entered together.
        def enter_config(stack):
            stack.enter_context(patch.dict(ms_train.CONFIG, values))
            if 'k8' in changed:
                stack.enter_context(patch.object(eg, 'infer_img', pinned_batch(eg.infer_img, G_BATCH)))
            if 'k9' in changed or 'k10' in changed:
                stack.enter_context(patch.object(ms_train, 'TrunkSide', trunk_side(cfg['k9'], cfg['k10'])))
        plan.append((description, enter_config))
    if 'k11' in changed:
        k = cfg['k11']
        plan.append(({'k11': {'rg_train.load_graphs': f'store of rs_graphs.sample_store(k={k})'}},
                     lambda stack: stack.enter_context(patch.object(
                         rg_train, 'load_graphs', lambda s, recipe: load_graphs_k(s, recipe, k)))))
    return plan


def describe(cfg):
    """What the configuration patches, {knob: {target: value}} (empty for G); recorded in every run.json."""
    description = {}
    for part, _ in _plan(cfg):
        description.update(part)
    assert set(description) == set(S.changed(cfg)), (sorted(description), S.changed(cfg))
    return description


def patches(cfg, stack, *, scale):
    """Enter every patch `cfg` needs into `stack`; none for G. Returns `describe(cfg)`. Sample scale only: the
    full-scale seams (ticket 08) extend this."""
    assert scale in ('sample', 'full'), scale
    assert not cfg['remove'], 'removal switches: ticket 07'
    description = describe(cfg)
    if description:
        assert scale == 'sample', 'knobs at full scale: ticket 08'
    for _, enter in _plan(cfg):
        enter(stack)
    return description
