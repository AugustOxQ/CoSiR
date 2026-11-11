"""r21 trainer (ticket 02): c*'s training loop with a DEC loss per head.

`run` is a copy of `ct2_d2.run` (`20261020_claim_test/ct2_d2.py`, frozen, imported, never edited) with hooks for DEC.
With `dec=None` (or `dec['form'] == 'off'`) every line the frozen loop executes is executed unchanged: the same RNG
draws, the same tensor ops, the same history and outputs (the hooks are list appends and `if` tests only). The
patchable helpers are called through their module attributes (`ct2_d2.make_config`, `ct2_d2.save_output`,
`ct2_d2.verify_output`, `ms_train.head_loss`, `ms_train.build_model`, ...), so the patches of c*'s chain
(`rg_full.train_cohort`) apply when `ct2_d2.run` is replaced by this `run`.

Arguments added to `ct2_d2.run`'s signature:
  dec       None, or the arm's settings dict: `form` in off | plain | start | teacher, `lam` (DEC weight, also the
            pull and balance weight), `warmup` W (epochs of c*'s losses only), `stop` in S1 | S2, `balance` (bool).
  dec_data  per aspect (all three heads, one joint run), required when DEC is on:
              dec_data[aspect]['k']        int, the class count (number of centres), all forms
              dec_data[aspect]['feats']    teacher form only: float32 [n_fit, d] unit rows, the teacher's image features
                                           of the head's fit paintings, in the head's fit-node order
                                           (`graphs[aspect]['fit_node_ids']`, which equals `s['fit']`)
              dec_data[aspect]['centres']  teacher form only: float32 [k, d], the cached teacher k-means centres
            The teacher's cluster of a painting is its nearest centre (the argmax of t), as k-means labels it.

Per head, on its own epoch clock (the heads' fit_pairs counts differ):
  epochs 1..W        c*'s losses only (the frozen loop). No best-state tracking: neither stop rule counts them.
  end of epoch W     after the dev pass: eval-mode full pass of the head's image view over the fit paintings, the
                     centres (k-means on the unit outputs for plain and start, teacher-cluster means of them for
                     teacher; one nn.Parameter per head in its own AdamW group, c*'s lr, weight decay 0), q, P,
                     and q0 (start; this q, fixed) or t (teacher; the teacher's soft assignments, fixed).
  DEC epochs W+1..   per batch: loss = ms_train.head_loss (unchanged call) + r21_dec.dec_block on the batch's unique
                     fit image paintings, from a separate train-mode forward pass of the head's image view on those
                     rows. P, q0 and t rows are indexed by fit-node position.
  end of a DEC epoch the dev loss as c*'s (no DEC term); an eval-mode full pass gives the new P, the argmax, the
                     changed fraction (against the previous epoch's argmax; the first DEC epoch against the pass at
                     DEC start), the cluster shares and the fraction away from the DEC-start clusters.
  stop S1            changed fraction < S1_CHANGED or S1_MAX_DEC_EPOCHS DEC epochs: best state = last state.
  stop S2            c*'s dev loss, patience from DEC start, best state among DEC epochs only.
                     Both: the cap `max_epochs` holds (reason 'max_epochs').

Recorded: every history entry gets a `dec` dict when DEC is on. Warm-up entries: `phase='warmup'` (epoch W also
`start_epoch` and the shares at DEC start). DEC entries: `phase='dec'`, `start_epoch`, `batches`, `kl`, `pull`,
`balance` (means over the epoch's batches of the unweighted terms; None when the form has no such term), `changed`,
`away`, `share_max`, `share_min`; the last entry also `stop_reason` in changed | dec_epochs | patience | max_epochs.
`train_loss` and `dev_loss` stay c*'s loss alone. The state carries `dec_start_epoch` and `stop_reason`.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20261020_claim_test'))
sys.path.insert(0, str(HERE.parent / '20261019_model_sheet'))
import ct2_d2  # noqa: E402
import ct2_data  # noqa: E402
import r21_dec  # noqa: E402

ms_train = ct2_d2.ms_train
S1_CHANGED = 0.001  # S1: stop when fewer than this fraction of the fit paintings change cluster
S1_MAX_DEC_EPOCHS = 50
FULL_PASS_CHUNK = 8192
STOP_RULES = ('S1', 'S2')


def validate_dec(dec, aspects, dec_data, max_epochs, graphs):
    """Assert the arm settings and `dec_data`; return True when DEC is on."""
    if dec is None or dec['form'] == 'off':
        return False
    assert set(dec) == {'form', 'lam', 'warmup', 'stop', 'balance'}, f'dec keys {sorted(dec)}'
    assert dec['form'] in r21_dec.FORMS and dec['stop'] in STOP_RULES
    assert isinstance(dec['warmup'], int) and not isinstance(dec['warmup'], bool)
    assert 1 <= dec['warmup'] < max_epochs, 'warm-up W must leave at least one DEC epoch'
    assert dec['warmup'] <= ms_train.CONFIG['patience'], 'a warm-up longer than the patience could stop a head before DEC'
    assert isinstance(dec['balance'], bool) and float(dec['lam']) >= 0.0
    assert dec_data is not None and set(dec_data) == set(aspects), 'dec_data needs every aspect'
    for aspect in aspects:
        d = dec_data[aspect]
        assert isinstance(d['k'], int) and d['k'] >= 2
        if dec['form'] == 'teacher':
            assert set(d) == {'k', 'feats', 'centres'}
            n = len(graphs[aspect]['fit_node_ids'])
            assert d['feats'].shape[0] == n and d['feats'].ndim == 2 and d['feats'].dtype == np.float32
            assert d['centres'].shape == (d['k'], d['feats'].shape[1]) and d['centres'].dtype == np.float32
            assert np.isfinite(d['feats']).all() and np.isfinite(d['centres']).all()
            assert np.max(np.abs(np.linalg.norm(d['feats'], axis=1) - 1)) < 1e-3, 'teacher features are not unit rows'
        else:
            assert set(d) == {'k'}
    return True


def full_pass(view_img, x):
    """Eval-mode pass of the image view over all rows of `x`; unit outputs on the CPU (float32)."""
    # `models.eval()` sets the shared modules; a view's own flag is not reliable, so test the layers that differ.
    assert not any(m.training for m in view_img.modules() if isinstance(m, nn.Dropout)), 'eval-mode pass needed'
    with torch.no_grad():
        return torch.cat([view_img(x[a:a + FULL_PASS_CHUNK]) for a in range(0, len(x), FULL_PASS_CHUNK)]).cpu()


class HeadDec:
    """One head's DEC state: centres, the target P, the fixed pull target, the cluster bookkeeping.

    The full-pass math (k-means, q, P) runs on CPU tensors; P, q0 and t are stored on `device` for batch rows.
    """
    def __init__(self, settings, k, seed, device, teacher=None):
        self.settings, self.k, self.seed, self.device, self.teacher = settings, k, seed, device, teacher
        self.centres = None
        self.n_batches, self.sums = 0, dict(kl_p=0.0, pull=0.0, balance=0.0)

    @property
    def started(self):
        return self.centres is not None

    @torch.no_grad()  # a full pass of soft_assign needs grad off (it chunks rows only then); P, q0, t come out detached
    def start(self, z):
        """DEC start from the eval-mode full pass `z` (CPU, unit rows [n_fit, d]); returns the start shares."""
        assert not self.started and z.shape[0] >= self.k
        form = self.settings['form']
        t = None
        if form == 'teacher':
            t = r21_dec.teacher_soft(self.teacher['feats'], self.teacher['centres'])
            centres = r21_dec.cluster_means(z, t.argmax(1).numpy(), self.k)
        else:
            centres = r21_dec.kmeans_centres(z, self.k, self.seed)
        self.centres = nn.Parameter(centres.to(self.device))
        q = r21_dec.soft_assign(z, centres)
        self.p = r21_dec.target(q).to(self.device)
        self.r = {'plain': None, 'start': q, 'teacher': t}[form]
        if self.r is not None:
            self.r = self.r.to(self.device)
        self.start_argmax = self.prev_argmax = q.argmax(1).numpy()
        share_max, share_min = r21_dec.cluster_shares(self.start_argmax, self.k)
        return dict(share_max=share_max, share_min=share_min)

    def block(self, view_img, x_fit, ids):
        """The DEC term on the fit rows `ids` (unique, a torch index on the device): a separate train-mode pass."""
        z = view_img(x_fit[ids])
        q = r21_dec.soft_assign(z, self.centres)
        total, parts = r21_dec.dec_block(q, self.p[ids], None if self.r is None else self.r[ids], self.settings)
        assert torch.isfinite(total), 'nonfinite DEC term'
        self.n_batches += 1
        for key in self.sums:
            self.sums[key] += parts.get(key, 0.0)
        return total

    @torch.no_grad()
    def end_epoch(self, z):
        """From the eval-mode full pass `z`: new P, the argmax and the epoch's statistics (batch means reset)."""
        q = r21_dec.soft_assign(z, self.centres.detach().cpu())
        self.p = r21_dec.target(q).to(self.device)
        argmax = q.argmax(1).numpy()
        share_max, share_min = r21_dec.cluster_shares(argmax, self.k)
        n = self.n_batches
        assert n > 0, 'a DEC epoch without a DEC batch'
        stats = dict(batches=n, kl=self.sums['kl_p'] / n,
                     pull=self.sums['pull'] / n if self.settings['form'] != 'plain' else None,
                     balance=self.sums['balance'] / n if self.settings['balance'] else None,
                     changed=r21_dec.changed_fraction(self.prev_argmax, argmax),
                     away=r21_dec.changed_fraction(self.start_argmax, argmax),
                     share_max=share_max, share_min=share_min)
        self.prev_argmax = argmax
        self.n_batches, self.sums = 0, dict(kl_p=0.0, pull=0.0, balance=0.0)
        return stats


def run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir,
        smoke=False, max_epochs=200, dec=None, dec_data=None):
    """`ct2_d2.run` with DEC hooks per head: one independent separate head, or one joint cohort with per-head clocks."""
    ms_train.validate(row)
    assert aspects and aspects == [a for a in ms_train.ASPECTS if a in aspects]
    assert row['share'] != 'separate' or len(aspects) == 1
    assert 1 <= max_epochs <= ms_train.CONFIG['max_epochs']
    dec_on = validate_dec(dec, aspects, dec_data, max_epochs, graphs)
    assert not dec_on or row['graph_terms'], 'DEC rows need pair batches'
    configs = {a: ct2_d2.make_config(s, row, name, a, aspects, manifests, seed, device, smoke, max_epochs)
               for a in aspects}
    paths = {a: ms_train.output_path(outdir, name, a, configs[a]['graph_row'], seed) for a in aspects}
    # Verify existing files against their recorded config before replacing stale cohorts.
    old = {a: ct2_d2.verify_output(paths[a], json.loads(paths[a].with_suffix('.json').read_text())['config'])
           if paths[a].exists() else None for a in aspects}
    if all(old[a] is not None and old[a]['config'] == configs[a] for a in aspects):
        print(f'{name} s{seed} {aspects}: verified identical outputs; skip', flush=True)
        return old
    # A missing or stale head replays and replaces the whole cohort.
    np.random.seed(seed)
    torch.manual_seed(seed)
    rngs = {a: np.random.default_rng(seed if len(aspects) == 1 else [seed, ms_train.ASPECTS.index(a)])
            for a in aspects}
    models, views = ms_train.build_model(row, aspects, device)
    if row['share'] == 'mask':
        masks = list(models['masks'].parameters())
        mask_ids = {id(p) for p in masks}
        params = [p for p in models.parameters() if id(p) not in mask_ids]
        opt = torch.optim.AdamW([dict(params=params), dict(params=masks, weight_decay=0.0)],
                                lr=ms_train.CONFIG['lr'], weight_decay=ms_train.CONFIG['weight_decay'])
    else:
        opt = torch.optim.AdamW(models.parameters(), lr=ms_train.CONFIG['lr'], weight_decay=ms_train.CONFIG['weight_decay'])
    tensors, states = {}, {}
    batch_size = ms_train.CONFIG['batch_size'] if row['graph_terms'] else 1024
    neg_rngs = {a: np.random.default_rng([seed, ms_train.ASPECTS.index(a), 1]) for a in aspects}
    if row['distill'] != 'none':
        fit, dev = s['fit'], s['dev']  # Before reading any teacher row.
    for aspect in aspects:
        graph = graphs[aspect]
        tensors[aspect] = {
            part: {side: torch.from_numpy(np.ascontiguousarray(x[graph[f'{part}_node_ids']])).to(device)
                   for side, x in inputs[aspect].items()} for part in ('fit', 'dev')}
        if row['distill'] != 'none':
            for part, ids in (('fit', fit), ('dev', dev)):
                assert set(graph[f'{part}_node_ids']) == set(ids)
            teacher = (inputs[aspect]['img'] if row['inputs'] == 'instructed' else
                       ct2_data.instructed_images(s, ms_train.head.ASPECTS[aspect]))
            assert teacher.shape == (len(s['first']), 2048) and teacher.dtype == np.float32
            assert np.isfinite(teacher).all()
            assert ct2_d2._sha_array(teacher) == s['data_sha256'][f'{aspect}.img'], f'{aspect}: teacher differs from the recorded hash'
            for part in ('fit', 'dev'):
                ids = graph[f'{part}_node_ids']
                tensors[aspect][part]['teacher'] = torch.from_numpy(
                    np.ascontiguousarray(teacher[ids])).to(device)
            del teacher
        if row['graph_terms']:
            fit_pairs, fit_w = graph['rows'][row['graphs'][aspect]]['fit']
            dev_pairs, dev_w = graph['rows'][row['graphs'][aspect]]['dev']
            if smoke:
                fit_pairs, fit_w = fit_pairs[:5000], fit_w[:5000]
        else:
            fit_pairs, dev_pairs = np.arange(len(fit)), np.arange(len(dev))
            fit_w, dev_w = np.ones(len(fit), dtype=np.float32), np.ones(len(dev), dtype=np.float32)
        states[aspect] = dict(fit_pairs=fit_pairs, fit_w=fit_w, dev_pairs=dev_pairs, dev_w=dev_w,
                              history=[], best=float('inf'), best_epoch=0, stale=0, best_state=None,
                              epoch=1, pos=0, total=0., order=None, tick=None, active=True)
        if row['form'] == 'b' and row['graph_terms']:
            states[aspect]['candidates'] = {part: ms_train.negative_candidates(graphs, row, aspect, part)
                                           for part in ('fit', 'dev')}
        if dec_on:
            d = dec_data[aspect]
            teacher_part = (dict(feats=torch.from_numpy(d['feats']), centres=torch.from_numpy(d['centres']))
                            if dec['form'] == 'teacher' else None)
            states[aspect]['hd'] = HeadDec(dec, d['k'], seed, device, teacher_part)
    start, step = time.monotonic(), 0
    while any(st['active'] for st in states.values()):
        models.train()
        active = [a for a in aspects if states[a]['active']]
        for aspect in active:
            st = states[aspect]
            if st['order'] is None:
                st['tick'] = time.monotonic()
                st['order'] = rngs[aspect].permutation(len(st['fit_pairs']))
                if row['form'] == 'b' and row['graph_terms']:
                    st['extras'] = ms_train.draw_extras(st['candidates']['fit'], st['fit_pairs'],
                                              neg_rngs[aspect], device)
        opt.zero_grad(set_to_none=True)
        losses, values, sizes = [], [], {}
        for aspect in active:
            st = states[aspect]
            ix = st['order'][st['pos']:st['pos'] + batch_size]
            extra_args = {}
            if row['form'] == 'b' and row['graph_terms']:
                extra_args['extras'] = st['extras'][torch.as_tensor(ix, device=device)]
            loss = ms_train.head_loss(views[aspect], **tensors[aspect]['fit'],
                             pairs=st['fit_pairs'][ix], weights=st['fit_w'][ix], row=row, aspect=aspect,
                             **extra_args)
            assert torch.isfinite(loss), 'nonfinite train loss'
            values.append(loss)  # c*'s loss alone: the train loss in the history
            if dec_on and st['hd'].started:
                # The batch's unique fit image paintings (the pair endpoints), in fit-node positions.
                ids = torch.from_numpy(np.unique(st['fit_pairs'][ix])).to(device)
                loss = loss + st['hd'].block(views[aspect]['img'], tensors[aspect]['fit']['img'], ids)
            losses.append(loss)
            sizes[aspect] = len(ix)
        # Do not introduce even an add-to-zero operation on the single-head parity path.
        loss = losses[0]
        for other in losses[1:]:
            loss = loss + other
        loss.backward()
        opt.step()
        step += 1
        for aspect, value in zip(active, values):
            st = states[aspect]
            st['total'] += float(value.detach()) * sizes[aspect]
            st['pos'] += sizes[aspect]
        for aspect in active:
            st = states[aspect]
            if st['pos'] < len(st['order']):
                continue
            models.eval()
            total = 0.
            if row['form'] == 'b' and row['graph_terms']:
                dev_extras = ms_train.draw_extras(st['candidates']['dev'], st['dev_pairs'],
                                        np.random.default_rng([10000, ms_train.ASPECTS.index(aspect)]), device)
            with torch.no_grad():
                for a in range(0, len(st['dev_pairs']), batch_size):
                    ps = st['dev_pairs'][a:a + batch_size]
                    ws = st['dev_w'][a:a + batch_size]
                    extra_args = {}
                    if row['form'] == 'b' and row['graph_terms']:
                        extra_args['extras'] = dev_extras[a:a + batch_size]
                    total += float(ms_train.head_loss(views[aspect], **tensors[aspect]['dev'],
                                             pairs=ps, weights=ws, row=row, aspect=aspect,
                                             include_mask_penalty=False,
                                             **extra_args)) * len(ps)
            dev_loss = total / len(st['dev_pairs'])
            assert np.isfinite(dev_loss)
            st['history'].append(dict(epoch=st['epoch'], step=step,
                                      train_loss=st['total'] / len(st['fit_pairs']), dev_loss=dev_loss,
                                      seconds=time.monotonic() - st['tick']))
            if row['form'] == 'c':
                st['history'][-1]['mask_penalty'] = float((5e-4 * F.relu(views[aspect]['img'].mask).abs().sum()).detach())
            if dec_on:
                st['active'] = _dec_epoch_end(st, dec, views[aspect], tensors[aspect]['fit']['img'], opt,
                                              dev_loss, max_epochs)
            elif dev_loss < st['best']:
                st['best'], st['best_epoch'], st['stale'] = dev_loss, st['epoch'], 0
                st['best_state'] = copy.deepcopy(views[aspect].state_dict())
            else:
                st['stale'] += 1
            h = st['history'][-1]
            print(f'{name}/{aspect} s{seed}: epoch={st["epoch"]} step={step} '
                  f'train={h["train_loss"]:.6f} dev={dev_loss:.6f} seconds={h["seconds"]:.2f}'
                  + (f' dec={json.dumps(h["dec"])}' if dec_on else ''), flush=True)
            if not dec_on:
                st['active'] = st['stale'] < ms_train.CONFIG['patience'] and st['epoch'] < max_epochs
            st.update(epoch=st['epoch'] + 1, pos=0, total=0., order=None)
    result = {}
    for aspect in aspects:
        result[aspect] = ct2_d2.save_output(s, views[aspect], inputs[aspect], states[aspect],
                                            paths[aspect], configs[aspect], start)
    return result


def _dec_epoch_end(st, dec, view, x_fit, opt, dev_loss, max_epochs):
    """The DEC hooks at the end of one head's epoch (eval mode is on, the dev pass and history entry are done).

    Returns whether the head stays active. Warm-up: no best state. End of epoch W: DEC start. DEC epoch: the
    statistics and the stop rule.
    """
    hd, epoch, entry = st['hd'], st['epoch'], st['history'][-1]
    if not hd.started:
        entry['dec'] = dict(phase='warmup')
        if epoch < dec['warmup']:
            return True
        shares = hd.start(full_pass(view['img'], x_fit))
        # Centres: their own AdamW group at c*'s lr, no weight decay.
        opt.add_param_group(dict(params=[hd.centres], lr=ms_train.CONFIG['lr'], weight_decay=0.0))
        st['dec_start_epoch'] = epoch + 1
        entry['dec'].update(start_epoch=epoch + 1, **shares)
        return True
    stats = hd.end_epoch(full_pass(view['img'], x_fit))
    entry['dec'] = dict(phase='dec', start_epoch=st['dec_start_epoch'], **stats)
    n_dec = epoch - st['dec_start_epoch'] + 1
    if dec['stop'] == 'S1':
        reason = ('changed' if stats['changed'] < S1_CHANGED else 'dec_epochs' if n_dec >= S1_MAX_DEC_EPOCHS
                  else 'max_epochs' if epoch >= max_epochs else None)
        if reason is not None:  # best state = last state
            st['best'], st['best_epoch'] = dev_loss, epoch
            st['best_state'] = copy.deepcopy(view.state_dict())
    else:
        # c*'s rule on c*'s dev loss, counted from DEC start (best and stale were left at their start values).
        if dev_loss < st['best']:
            st['best'], st['best_epoch'], st['stale'] = dev_loss, epoch, 0
            st['best_state'] = copy.deepcopy(view.state_dict())
        else:
            st['stale'] += 1
        reason = ('patience' if st['stale'] >= ms_train.CONFIG['patience']
                  else 'max_epochs' if epoch >= max_epochs else None)
    if reason is not None:
        st['stop_reason'] = entry['dec']['stop_reason'] = reason
    return reason is None
