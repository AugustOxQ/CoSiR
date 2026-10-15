"""Minimal k-shot adapters around round 6; original modules remain unchanged."""
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/project/CoSiR')
R6 = ROOT / 'src/test/20261125_artelingo_held_test'
sys.path.insert(0, str(R6))
# r6_common's import resolves a worktree through Git. This workspace is given by
# the task, so return that path without executing Git; reject other Git calls.
_run = subprocess.run
def _workspace_path(args, **kwargs):
    if args[0] != 'git':
        return _run(args, **kwargs)
    assert args == ['git', '-C', str(R6), 'rev-parse', '--path-format=absolute', '--git-common-dir']
    return subprocess.CompletedProcess(args, 0, str(ROOT / '.git') + '\n', '')

with patch('subprocess.run', _workspace_path):
    import r6_common as R
import r6_bundle as B
import r6_context as X
import r6_heads as H
import r6_picks as P
import r6_score as S
import r6_stats as ST
import numpy as np
import torch
from src.eval.aspect_episodes import concat_episodes
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_quick_checks import centered_term, uniform_probe_scores, _code_sides
from src.eval.aspect_scorers import EvalInputs, cosine_scores
from src.eval.pair_metric_baselines import rca_term, fit_pca_basis, _proj_pairs
from src.train.train_factors import encode_rows, load_factor_checkpoint
import run_r6_dts as D

NAMES = ('cosine', 'RCA', 'B', 'B0', 'AFF', 'CF')
PAIR_FIELDS = ('pairs_a_img', 'pairs_a_txt', 'pairs_b_img', 'pairs_b_txt')
SPREAD_COLUMNS = np.array([6*j+i for j in range(len(B.A0)) for i in (3, 4)])


def prefix(ep, k):
    """Replace NUM_PAIRS=4 validation with exact-prefix and dimension checks.

    The full arrays have r6's validated episode SHA; prefixing inherits their
    painting/value constraints, including exclusion of the anchor's value.
    """
    assert k in (1, 2, 3, 4)
    out = replace(ep, **{f: np.ascontiguousarray(getattr(ep, f)[:, :k]) for f in PAIR_FIELDS})
    assert np.array_equal(out.anchor, ep.anchor)
    assert np.array_equal(out.candidates, ep.candidates) and out.candidates.shape == (len(ep.anchor), 13)
    for cond in CONDITIONS:
        si, st, ci, ct, target = out.condition(cond)
        assert target == (0 if cond == 'a' else 1)
        assert all(a.shape == (len(ep.anchor), k) for a in (si, st, ci, ct))
    for f in PAIR_FIELDS:
        assert np.array_equal(getattr(out, f), getattr(ep, f)[:, :k])
    return out


def features(post, ep, k):
    """rb_features.episode_features, with k replacing its four-pair guard.

    Preserve float32 agreement means/deltas, float64 ddof=1 spreads and
    the support image/caption argmax coincidence share.
    """
    out = {}
    for cond in CONDITIONS:
        si, st, ci, ct, _ = ep.condition(cond)
        assert all(a.shape == (len(ep.anchor), k) for a in (si, st, ci, ct))
        cols = []
        for h in B.A0:
            pi, pt = post[h]['img'], post[h]['txt']
            ai, at = pi[si], pt[st]
            sup = np.einsum('nsc,nsc->ns', ai, at)
            con = np.einsum('nsc,nsc->ns', pi[ci], pt[ct])
            assert sup.shape == con.shape == (len(ep.anchor), k)
            s, c = sup.mean(axis=1), con.mean(axis=1)
            # At k=1 these are placeholders only: each half-reader replaces
            # them with its own scaler means inside the reader call below.
            sd_sup = np.zeros(s.shape) if k == 1 else sup.astype(np.float64).std(axis=1, ddof=1)
            sd_con = np.zeros(c.shape) if k == 1 else con.astype(np.float64).std(axis=1, ddof=1)
            cols += [s, c, s-c, sd_sup, sd_con,
                     (ai.argmax(axis=-1) == at.argmax(axis=-1)).mean(axis=1)]
        out[cond] = np.stack([np.asarray(x, dtype=np.float64) for x in cols], axis=1)
        assert out[cond].shape == (len(ep.anchor), 18) and np.isfinite(out[cond]).all()
    for j in range(len(B.A0)):
        assert np.array_equal(out['b'][:, 6*j+2], -out['a'][:, 6*j+2])
    return out


def reader(b, readers):
    """Original reader, with half-specific spread neutralisation only at k=1."""
    if b.pooled.pairs_a_img.shape[1] != 1:
        return R.RF.reader(b, readers)

    def half_probs(pk, features, n_classes):
        out = []
        for half in pk['halves']:
            assert np.array_equal(half['model'].classes_, np.arange(n_classes))
            raw = features.copy()
            raw[:, SPREAD_COLUMNS] = half['scaler'].mean_[SPREAD_COLUMNS]
            scaled = half['scaler'].transform(raw)
            assert np.all(scaled[:, SPREAD_COLUMNS] == 0), 'C4 k=1 spread columns must scale to exactly zero'
            p = half['model'].predict_proba(scaled)
            assert np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)
            out.append(p)
        return out

    with patch.object(R.rbe, 'half_reader_probs', half_probs):
        return R.RF.reader(b, readers)


def load():
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    H.require_threads()
    torch.set_num_threads(8)
    R.assert_inputs()
    eps = D.load_episodes_checked(42, R6 / 'results/dts_inputs/episodes_seed42.npz')
    assert eps.n == 12288 and all(len(ep.anchor) == 4096 for ep in eps.per_pair)
    with np.load(R.INPUT_PATHS[R.PREPARE_REL]) as z:
        groups, selection, train = (z[f] for f in ('groups', 'selection', 'scorer_train'))
    assert np.isin(eps.pooled.rows(), selection).all()
    assert not np.intersect1d(selection, train).size
    # Read only scorer-train and selection feature rows from read-only shards;
    # no annotations or held-row feature values are opened through the loader.
    rows = np.union1d(selection, train)
    data = {}
    from src.data.artelingo import FEATURE_DIR
    for m in ('img', 'txt'):
        full = np.full((R.N_ROWS, 512), np.nan, np.float32)
        offset = 0
        for shard in sorted((Path(FEATURE_DIR) / 'shards').glob('shard_*')):
            mm = np.load(shard / f'{m}_features.npy', mmap_mode='r')
            take = rows[(rows >= offset) & (rows < offset + len(mm))]
            full[take] = mm[take-offset]
            offset += len(mm)
        assert offset == R.N_ROWS and np.isfinite(full[rows]).all()
        data[f'{m}_features'] = full
    data = SimpleNamespace(**data)
    ctx = SimpleNamespace(groups=groups, selection=selection)
    post_e2 = R.run_n6.load_posteriors(R.INPUT_PATHS[H.N6_REL], ctx)
    # Saved later, but these exact coefficient bytes are round 6's frozen head.
    head_path = ROOT / 'src/test/20261126_design_l_style_source/results/heads/head_affect.pkl'
    with head_path.open('rb') as f:
        head = pickle.load(f)
    import json
    recorded = json.loads((R6 / 'results/refit_check.json').read_text())
    assert H.coef_sha256({'affect': head})['affect'] == recorded['coef_sha256']['affect']
    post_aff = H.predict({'affect': head}, data, selection)['affect']
    post = {'affect': post_aff, 'image': post_e2['image'], 'caption': post_e2['caption']}
    masked = []
    for m in ('img', 'txt'):
        a = np.full((R.N_ROWS, 512), np.nan, np.float32)
        a[selection] = getattr(data, f'{m}_features')[selection]
        masked.append(a)
    base = EvalInputs(*masked)
    # Restore the original training-only PCA calculation once: exact r6 draw,
    # seed, normalisation, algorithm. No episode-dependent fit or selection.
    draw = B.fit_rows(train)
    baseline = json.loads(R.INPUT_PATHS[B.BASELINES42_REL].read_text())
    assert R.sha256_bytes(np.ascontiguousarray(draw, dtype=np.int64).tobytes()) == baseline['fit_rows_sha256']
    fi, ft = (getattr(data, f'{m}_features')[draw] for m in ('img', 'txt'))
    fi, ft = (a / np.linalg.norm(a, axis=1, keepdims=True) for a in (fi, ft))
    basis = fit_pca_basis(fi, ft)
    model, _ = load_factor_checkpoint(R.INPUT_PATHS[B.A3_REL], device='cpu')
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=selection, device='cpu')
    codes = []
    for a in (ic, tc):
        full = np.full((R.N_ROWS, a.shape[1]), np.nan, np.float32)
        full[selection] = a
        codes.append(full)
    inputs = SimpleNamespace(img_codes=codes[0], txt_codes=codes[1])
    return SimpleNamespace(eps=eps, base=base, inputs=inputs, basis=basis, post=post,
                           post_e2=post_e2, readers=B.load_readers(), picks=P.load_picks(),
                           lambdas=P.frozen_lambdas(), cl=groups[eps.pooled.anchor])


def build(env, k):
    """The relevant build_bundle_r6 sequence, without unused scorer work."""
    ep = concat_episodes([prefix(p, k) for p in env.eps.per_pair])
    for c in CONDITIONS:
        assert all(a.shape == (12288, k) for a in ep.condition(c)[:4])
    # C3: inspect the real arrays passed into factor centring and RCA as well.
    for d in DIRECTIONS:
        _, _, ex = _code_sides(env.inputs, ep, d)
        assert ex.shape[:2] == (12288, 2*k)
    for c in CONDITIONS:
        assert all(a.shape[:2] == (12288, k) for a in _proj_pairs(env.base, ep, c, env.basis))
    F = features(env.post, ep, k)
    if k == 4:
        original, _ = R.rbe.seed42_features(SimpleNamespace(ctx=SimpleNamespace(pooled=ep), post=env.post), B.A0)
        assert all(np.array_equal(F[c], original[c]) for c in CONDITIONS)
    b = SimpleNamespace(n=12288, mode='selection', seed=42, smoke=False, pooled=ep,
                        parity=env.eps.parity, pair_index=env.eps.pair_index, cl=env.cl,
                        cos=cosine_scores(env.base, ep), t_n1u=centered_term(env.inputs, ep, uniform=True),
                        t6u_B=uniform_probe_scores(env.post_e2, ep, B.E2_PARTS),
                        t6u_B0=uniform_probe_scores(env.post, ep, B.A0),
                        stack=R.C.grouping_stack(env.post, ep, B.A0), F=F,
                        pm_terms={'rca': rca_term(env.base, ep, env.basis)})
    for name in ('cos', 't_n1u', 't6u_B', 't6u_B0'):
        B.check_scores(getattr(b, name), b.n, name, np.float32)
    B.check_scores(b.pm_terms['rca'], b.n, 'rca')
    return b


def score(env, b):
    """Six-scorer subset of score_seed, using the original frozen functions."""
    frozen = {n: S.frozen_nested(b.cos, b.t_n1u, getattr(b, P.TERM_OF[n]), env.picks[n], b.parity)
              for n in ('B', 'B0')}
    rd = reader(b, env.readers)
    gates = R.RF.gates_aff(rd['m'], rd['pick'], R.R3.assert_taus())
    aff, cf = S.frozen_cells(frozen['B'], b.parity, rd['T'], gates,
                            S.CELLS['aff']['fused'], S.CELLS['aff']['cf'])
    raw = {'cosine': b.cos, 'RCA': S.frozen_fused(b.cos, b.pm_terms['rca'], env.lambdas['rca'], b.parity),
           **frozen, 'AFF': aff, 'CF': cf}
    pa = {n: per_anchor(raw[n]) for n in NAMES}
    assert all(np.all(pa[n]['gain'] == 0) for n in ('cosine', 'B', 'B0', 'CF'))
    diag = {f'tau_{t}': {c: np.count_nonzero(gates[t][c]) / b.n for c in CONDITIONS} for t in (0, 2)}
    return pa, diag, frozen
