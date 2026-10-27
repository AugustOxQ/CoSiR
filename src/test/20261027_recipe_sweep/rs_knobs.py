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
  k11      `rg_train.load_graphs` is replaced by `rs_graphs.sample_store(k, 'aspect', s=, outputs=)` (ticket 03)
Full scale (ticket 08; `rs_full` runs `rg_full.train_cohort` and `rg_full.infer_held` under these patches): the same
patches except where the full path reads another object. The loss patches (k1 to k4) and CONFIG / TrunkSide (k6 to k10,
k12, k13) are the objects both paths share. The differences:
  k5       `rb_rows.recipe(V.BASE)` (the row `rg_full.train_cohort` hands to `ct2_d2.run`) instead of `rg_train.resolved`
  k8       the export batches are two functions: `rb_student.CT2_SAVE_OUTPUT` (training's exports) and `rb_student.infer`
           (held inference), instead of `eg.infer_img`
  k11      `rb_student.load_scale` (the CT's full loader, which `rs_full` calls for the data and the graphs) is replaced
           by `rs_graphs.full_training_data(k, 'aspect', outputs=)`. Ticket 07's graph removals enter at the same seam.
The stopping loss of a head with a relational term includes that term (as genre's does at G): the loss that is
trained is the loss that stops.

Removals (ticket 07; `cfg['remove']`, at most one: the ablation takes one part out of F and retunes nothing). The same
patches at both scales; the section "removals" below. Each is F's configuration with one part out:
  relational          the lambdas k1, k2, k3 enter the knob patches as 0 (no relational term on any head)
  sharp_row           k4 enters the knob patches as 0.05 (every relational teacher row at the student's temperature)
  direct              the genre arm's m = 0 (the frozen switch of 12-27's GN-8) and, on emotion and style, the recipe row's
                      distill off inside `rg_loss.switched_head_loss` (the row itself keeps its distill, the frozen genre
                      body asserts on it)
  buddy, caption_endpoints   `ab_train`'s switches buddy = 0 and endpoints = 'image' (D2's R5 and R7) on emotion and
                      style; `genre_loss` (a copy of the frozen genre body with those two switches) on genre
  content_correction, instructed_graphs   the k11 seam with ticket 03's variants 'uncorrected' and 'default_view' (at the
                      run's k11) and the recipe row's graph names; they refuse until gates/ablation_graphs.json passes
  shared_trunk        `ms_train.TrunkSide` / `TrunkHead` with one trunk per aspect and side in the one joint cohort (the
                      sheet's 'separate' mode trains three cohorts, which neither trainer's loop does): every aspect's
                      parameters are its own, so the joint loop trains each as it would alone
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
import rb_rows
import rb_student
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


def store_training_data(scale, k, outputs, variant='aspect'):
    """The full-scale (s, inputs, graphs, manifests) of mutual top-k `k` as `rb_student.load_scale('full')` returns them,
    from ticket 03's `rs_graphs.full_training_data` (which enforces gates/k11_full.json, V4c, and for a graph variant
    gates/ablation_graphs.json). k = 20 of the aspect graphs never comes here: the frozen loader reads the CT's graphs."""
    assert scale == 'full' and (k, variant) != (S.G['k11'], 'aspect'), 'k = 20 is the frozen store'
    try:
        import rs_graphs
    except ImportError:
        raise AssertionError(f'k11 = {k}: rs_graphs (ticket 03) is not merged; only k = 20 can run') from None
    return rs_graphs.full_training_data(k, variant, outputs=outputs)


def store_graphs(s, k, outputs, variant='aspect'):
    """The sample graphs of mutual top-k `k` as `rg_train.load_graphs(s, recipe)` returns them, (graphs, manifests),
    from ticket 03's `rs_graphs.sample_store` (imported here: the one seam; it enforces its own gates, check K
    included, and for a graph variant the ablation gate). k = 20 of the aspect graphs never comes here: the frozen
    loader reads the frozen store."""
    assert (k, variant) != (S.G['k11'], 'aspect'), 'k = 20 is the frozen store'
    try:
        import rs_graphs
    except ImportError:
        raise AssertionError(f'k11 = {k}: rs_graphs (ticket 03) is not merged; only k = 20 can run') from None
    return rs_graphs.sample_store(k, variant, s=s, outputs=outputs)


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


def with_recipe_weight(recipe, weight):
    """`rb_rows.recipe` with the direct teacher weight of the training recipe's row set (k5), as `with_direct_weight` does
    for the sample scale's `resolved`. Other recipes (held inference builds the model from R0-5's row) are untouched."""
    def weighted(name):
        row = recipe(name)
        if name == V.BASE:
            assert row['base']['distill'] == 'direct' and row['base']['distill_weight'] == S.G['k5']
            row['base']['distill_weight'] = weight
        return row
    return weighted


# ------------------------------------------------------------------------------------------------- removals
GRAPH_REMOVALS = dict(content_correction='uncorrected', instructed_graphs='default_view')    # -> ticket 03's variant
LOSS_SWITCHES = dict(buddy=dict(buddy=0.0), caption_endpoints=dict(endpoints='image'))       # ab_train's switches (D2 R5, R7)
KNOBS_OF = dict(relational=('k1', 'k2', 'k3'), sharp_row=('k4',))     # the knobs a removal sets (as the knob patches see them)


def check_removal(cfg):
    """Refuse a removal with no part to remove on F (cfg without its removal), and a combination of removals."""
    remove = cfg['remove']
    assert len(remove) <= 1, f'{S.name(cfg)}: one removal at a time (the ablation takes one part out of F)'
    if remove in (('relational',), ('sharp_row',)):
        assert any(cfg[k] for k in KNOBS_OF['relational']), \
            f'{S.name(cfg)}: F has no relational term to remove (k1 = k2 = k3 = 0)'
    if remove == ('sharp_row',):
        assert cfg['k4'] != V.TAU, f'{S.name(cfg)}: every relational teacher row is already at 0.05'
    if remove == ('direct',):
        assert cfg['k5'] > 0, f'{S.name(cfg)}: F has no direct term to remove (k5 = 0)'


def effective(cfg):
    """The knobs the knob patches see: `cfg` with the relational removals applied to k1 to k4 (lambda 0 on every head;
    every teacher row at the student's 0.05). The other removals do not touch a knob."""
    eff = dict(cfg, remove=())
    if 'relational' in cfg['remove']:
        eff.update(k1=0, k2=0, k3=0)
    if 'sharp_row' in cfg['remove']:
        eff.update(k4=V.TAU)
    return eff


def genre_loss(models, img, cap, pairs, weights, row, teacher, switches, genre):
    """The genre head's loss with `ab_train`'s buddy switches (copied from `rg_loss.switched_head_loss`'s genre body, which
    asserts R6-5's switches): the operations and their order are the frozen body's, so at R6-5's switches the loss is
    the frozen loss bit for bit (a test checks it, in training mode). `buddy` 0 drops the four buddy terms (the caption
    side is still run, as in `ab_train`); `endpoints` 'image' keeps the image-image term alone, not averaged (D2's R7).
    The direct term is `genre.m`'s, the relational term `genre.lam`'s at `genre.tau_t`."""
    log = genre.term_log
    if log is not None:
        log.clear()
    assert genre.n_anchors == 0 and not switches['own'] and switches['teacher'] == 'instructed'
    assert row['form'] == 'a' and row['graph_terms'] and row['distill'] == 'direct'
    assert row['distill_weight'] == V.TEACHER_WEIGHT
    assert teacher is not None and teacher.shape == img.shape and not teacher.requires_grad
    a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
    ids = torch.cat((a, b))
    z = models['img'](img[ids])
    ia, ib = z.chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    sym = ms_train.symmetric
    terms = []
    if switches['endpoints'] == 'image':
        if switches['buddy']:
            terms.append(sym(ia, ib, w))
    else:
        ca, cb = models['cap'](cap[ids]).chunk(2)
        if switches['buddy']:
            terms.append((sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4)
    if genre.m:
        direct = ms_train.distill_loss(z, teacher[ids], ids, 'direct')
        terms.append(row['distill_weight'] * direct)
        if log is not None:
            log['direct'] = direct.detach()
    if genre.lam:
        relational = rg_loss.relational_loss(z, teacher[ids], ids, genre.tau_t)
        terms.append(genre.lam * relational)
        if log is not None:
            log['relational'] = relational.detach()
    assert terms, 'the removal left no loss term'
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def with_removed_terms(original, removal):
    """Wrap `rg_loss.switched_head_loss` (both trainers call it by that name) for the loss removals `direct`, `buddy` and
    `caption_endpoints`. Emotion and style: the same call with the row's distill off (direct) or `ab_train`'s switch
    (the other two) set; it reaches `ab_train.switched_head_loss` as before, so k1 and k2's relational term stays.
    Genre: direct is the arm's m = 0 (not here); the buddy switches go to `genre_loss`."""
    def switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None,
                           include_mask_penalty=True, *, switches=None, genre=None):
        if removal == 'direct':
            if aspect != 'genre':
                row = dict(row, distill='none', distill_weight=0.0)
        else:
            switches = dict(switches, **LOSS_SWITCHES[removal])
            if aspect == 'genre':
                return genre_loss(models, img, cap, pairs, weights, row, teacher, switches, genre)
        return original(models, img, cap, pairs, weights, row, aspect, teacher, extras, include_mask_penalty,
                        switches=switches, genre=genre)
    return switched_head_loss


def separate_trunks(width, dropout):
    """(TrunkSide, TrunkHead) with one trunk per aspect on each side (the trunk of `ms_train.TrunkSide`, width and dropout
    as the run's), heads as before. A view keeps `ms_train.TrunkHead`'s attribute names, so its state keys, the saved best
    states and the held inference's loading are unchanged. The layers are created trunks first, then heads."""
    class TrunkSide(nn.Module):
        def __init__(self, aspects):
            super().__init__()
            self.trunks = nn.ModuleDict({a: nn.Sequential(nn.Linear(2048, width), nn.GELU(), nn.Dropout(dropout))
                                         for a in aspects})
            self.heads = nn.ModuleDict({a: nn.Linear(width, 2048) for a in aspects})

    class TrunkHead(nn.Module):
        def __init__(self, side, aspect):
            super().__init__()
            self.trunk, self.output = side.trunks[aspect], side.heads[aspect]

        def forward(self, x):
            return nn.functional.normalize(self.output(self.trunk(x)), dim=1)
    return TrunkSide, TrunkHead


def with_graph_rows(resolved, rows):
    """`rg_train.resolved` with the recipe row's graph names set to a variant's rows (`rs_graphs.row_names`)."""
    def rowed(seed):
        recipe = resolved(seed)
        assert set(recipe['base']['graphs']) == set(rows)
        recipe['base']['graphs'] = dict(rows)
        return recipe
    return rowed


def with_recipe_graph_rows(recipe, rows):
    """`rb_rows.recipe` with the training recipe's graph names set to a variant's rows (other recipes untouched)."""
    def rowed(name):
        row = recipe(name)
        if name == V.BASE:
            assert set(row['base']['graphs']) == set(rows)
            row['base']['graphs'] = dict(rows)
        return row
    return rowed


def _removal_plan(cfg):
    """[(description, enter)] of the loss and trunk removals. The relational removals enter through the knob patches
    (`effective`) and the graph removals through the k11 seam (both in `_plan`); their descriptions are here."""
    plan = []
    arm = f'rg_variants.ARMS[{rs_paths.G_ARM}]'
    for removal in cfg['remove']:
        key = f'no_{removal}'
        if removal == 'relational':
            plan.append(({key: {arm: {'lam': 0}, 'ab_train.switched_head_loss': 'no relational term on emotion and style'}},
                         lambda stack: None))
        elif removal == 'sharp_row':
            plan.append(({key: {arm: {'tau_t': V.TAU}, 'ab_train.switched_head_loss': f'teacher row at {V.TAU}'}},
                         lambda stack: None))
        elif removal == 'direct':
            def enter(stack):
                stack.enter_context(patch.dict(V.ARMS[rs_paths.G_ARM], m=0))
                stack.enter_context(patch.object(rg_loss, 'switched_head_loss',
                                                 with_removed_terms(rg_loss.switched_head_loss, 'direct')))
            plan.append(({key: {arm: {'m': 0}, 'rg_loss.switched_head_loss': 'direct term off on emotion and style'}},
                         enter))
        elif removal in LOSS_SWITCHES:
            plan.append(({key: {'rg_loss.switched_head_loss': f'ab_train switches {LOSS_SWITCHES[removal]}; genre: genre_loss'}},
                         lambda stack, removal=removal: stack.enter_context(patch.object(
                             rg_loss, 'switched_head_loss', with_removed_terms(rg_loss.switched_head_loss, removal)))))
        elif removal == 'shared_trunk':
            def enter(stack):
                side, head = separate_trunks(cfg['k9'], cfg['k10'])
                stack.enter_context(patch.object(ms_train, 'TrunkSide', side))
                stack.enter_context(patch.object(ms_train, 'TrunkHead', head))
            plan.append(({key: {'ms_train.TrunkSide': f'one trunk per aspect, width {cfg["k9"]}, dropout {cfg["k10"]}',
                                'ms_train.TrunkHead': 'the view of its own trunk'}}, enter))
        else:
            assert removal in GRAPH_REMOVALS, removal      # in _plan's k11 seam
    return plan


def _plan(cfg, outputs=None, scale='sample'):
    """[(description, enter)] for the changed knobs: description {knob: {target: value}} (JSON), enter(stack)
    enters the patch. Empty for G. A removal's parts follow the knobs' (the relational removals act through the knobs,
    `effective`; the k1 to k4 they set are described under the removal, not as a knob of the configuration)."""
    check_removal(cfg)
    removal, cfg = cfg['remove'], effective(cfg)
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
        recipe_target, recipe_module, recipe_name, wrap = (
            ('rg_train.resolved', rg_train, 'resolved', with_direct_weight) if scale == 'sample' else
            ('rb_rows.recipe', rb_rows, 'recipe', with_recipe_weight))
        plan.append(({'k5': {recipe_target: f'direct weight {cfg["k5"]}', 'rg_variants.TEACHER_WEIGHT': cfg['k5']}},
                     lambda stack: (stack.enter_context(patch.object(
                         recipe_module, recipe_name, wrap(getattr(recipe_module, recipe_name), cfg['k5']))),
                         stack.enter_context(patch.object(V, 'TEACHER_WEIGHT', cfg['k5'])))))
    values = {key: cfg[k] for k, key in CONFIG_KEYS.items() if k in changed}
    description = {k: {'ms_train.CONFIG': {key: cfg[k]}} for k, key in CONFIG_KEYS.items() if k in changed}
    if 'k8' in changed:
        for target in (('eg_train3.infer_img',) if scale == 'sample' else
                       ('rb_student.CT2_SAVE_OUTPUT', 'rb_student.infer')):
            description['k8'][target] = f'export batches pinned at {G_BATCH}'
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
                for module, name in (((eg, 'infer_img'),) if scale == 'sample' else
                                     ((rb_student, 'CT2_SAVE_OUTPUT'), (rb_student, 'infer'))):
                    stack.enter_context(patch.object(module, name, pinned_batch(getattr(module, name), G_BATCH)))
            if 'k9' in changed or 'k10' in changed:
                stack.enter_context(patch.object(ms_train, 'TrunkSide', trunk_side(cfg['k9'], cfg['k10'])))
        plan.append((description, enter_config))
    variant = GRAPH_REMOVALS.get(removal[0], 'aspect') if removal else 'aspect'
    if 'k11' in changed or variant != 'aspect':
        k = cfg['k11']
        seam, recipe_target = (('rg_train.load_graphs', 'rg_train.resolved') if scale == 'sample' else
                               ('rb_student.load_scale', 'rb_rows.recipe'))
        text = (f"rs_graphs.sample_store(k={k}, '{variant}')" if scale == 'sample' else
                f"rs_graphs.full_training_data(k={k}, '{variant}')")
        description = {}
        if 'k11' in changed:
            description['k11'] = {seam: text}
        if variant != 'aspect':
            description[f'no_{removal[0]}'] = {seam: text, recipe_target: f'graph rows of {variant}'}

        def enter_graphs(stack):
            if scale == 'sample':
                stack.enter_context(patch.object(rg_train, 'load_graphs',
                                                 lambda s, recipe: store_graphs(s, k, outputs, variant)))
            else:
                stack.enter_context(patch.object(rb_student, 'load_scale',
                                                 lambda scale: store_training_data(scale, k, outputs, variant)))
            if variant != 'aspect':
                import rs_graphs
                rows = rs_graphs.row_names(variant)
                if scale == 'sample':
                    stack.enter_context(patch.object(rg_train, 'resolved', with_graph_rows(rg_train.resolved, rows)))
                else:
                    stack.enter_context(patch.object(rb_rows, 'recipe', with_recipe_graph_rows(rb_rows.recipe, rows)))
        plan.append((description, enter_graphs))
    induced = {k for r in removal for k in KNOBS_OF.get(r, ())}
    plan = [({k: v for k, v in d.items() if k not in induced}, enter) for d, enter in plan]
    return plan + _removal_plan(dict(cfg, remove=removal))


def describe(cfg, scale='sample'):
    """What the configuration patches, {knob or no_<removal>: {target: value}} (empty for G); recorded in every run.json."""
    description = {}
    for part, _ in _plan(cfg, scale=scale):
        description.update(part)
    induced = {k for r in cfg['remove'] for k in KNOBS_OF.get(r, ())}
    assert set(description) == (set(S.changed(cfg)) - induced) | {f'no_{r}' for r in cfg['remove']}, \
        (sorted(description), S.changed(cfg), cfg['remove'])
    return description


def patches(cfg, stack, *, scale, outputs=None):
    """Enter every patch `cfg` needs into `stack`; none for G. Returns `describe(cfg, scale)`. `outputs` (the run's output
    folder, where the graph stores and gates live) is needed for k11 and the graph removals. A graph removal refuses
    here, before any data is read, unless `gates/ablation_graphs.json` passes for its variant (`rs_graphs.GraphRefusal`)."""
    assert scale in ('sample', 'full'), scale
    description = describe(cfg, scale)
    graph = [GRAPH_REMOVALS[r] for r in cfg['remove'] if r in GRAPH_REMOVALS]
    assert outputs is not None or not ('k11' in description or graph), 'k11 and the graph removals: needs the outputs folder'
    for variant in graph:
        import rs_graphs
        rs_graphs.require_gate(outputs, 'ablation_graphs', variant)
    for _, enter in _plan(cfg, outputs, scale):
        enter(stack)
    return description
