"""R3 D3–D11 scoring, independent of the implementation chain.

Reader features and metric formulas reuse R6's independent rd code.
"""
import rd9_inputs
import numpy as np
import torch
from src.model.aspect_rule import zscore_rows

CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")
NESTED_U = (0., .5, 1., 2., 4., 8., 16.)
NESTED_A = (0., .25, .5, 1., 2., 4., 8., 16.)
A0 = ("affect", "image", "caption")
def unit32(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def cosine(img_u, txt_u, ep):
    out = {}
    for d in DIRS:
        if d == "i2t":
            q, c = img_u[ep.anchor], txt_u[ep.candidates]
        else:
            q, c = txt_u[ep.anchor], img_u[ep.candidates]
        out[d] = np.einsum("nd,nkd->nk", q, c)
    return {c: {d: out[d] for d in DIRS} for c in CONDS}

def _first(s, col):
    t = s[:, col:col + 1]
    rest = np.concatenate([s[:, :col], s[:, col + 1:]], axis=1)
    return ((rest < t).all(axis=1) & np.isfinite(s).all(axis=1)).astype(np.float64)

def metrics(scores):
    """Per-anchor r1, gain, other, swap, strict: per direction, then the mean of the two directions."""
    per = {m: [] for m in METRICS}
    for d in DIRS:
        sa = np.asarray(scores["a"][d], dtype=np.float64)
        sb = np.asarray(scores["b"][d], dtype=np.float64)
        aa, bb, ab, ba = _first(sa, 0), _first(sb, 1), _first(sb, 0), _first(sa, 1)
        r1 = (aa + bb) / 2
        other = (ba + ab) / 2
        fin = np.isfinite(sa).all(axis=1) & np.isfinite(sb).all(axis=1)
        swap = ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0]) & fin).astype(np.float64)
        for m, v in (("r1", r1), ("gain", r1 - other), ("other", other), ("swap", swap), ("strict", aa * bb)):
            per[m].append(v)
    return {m: (v[0] + v[1]) / 2 for m, v in per.items()}

def by_parity(full_by_half, parity):
    """full_by_half[h]: scores of the pick made on tune half h, which scores the episodes of parity 1 - h."""
    out = {c: {d: np.empty(np.asarray(full_by_half[0][c][d]).shape, np.float32) for d in DIRS} for c in CONDS}
    for h in (0, 1):
        rows = parity == 1 - h
        for c in CONDS:
            for d in DIRS:
                out[c][d][rows] = np.asarray(full_by_half[h][c][d], dtype=np.float32)[rows]
    return out

def cell_params(cell):
    t, rem = divmod(int(cell), 56)
    return t, NESTED_U[rem // 8], NESTED_A[rem % 8]

def features(post, parts, si, st, ci, ct):
    cols = []
    for h in parts:
        pi, pt = post[h]["img"], post[h]["txt"]
        ai, at = pi[si], pt[st]
        sup = np.einsum("nsc,nsc->ns", ai, at)
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        s, c = sup.mean(axis=1), con.mean(axis=1)
        cols += [s, c, s - c, sup.astype(np.float64).std(axis=1, ddof=1), con.astype(np.float64).std(axis=1, ddof=1),
                 (ai.argmax(axis=-1) == at.argmax(axis=-1)).mean(axis=1)]
    return np.stack([np.asarray(x, dtype=np.float64) for x in cols], axis=1)

def stack_dots(post, ep, parts):
    out = {}
    for d in DIRS:
        q, c = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][q][ep.anchor], post[h][c][ep.candidates]) for h in parts],
                          axis=1)
    return out

def condition_free(s):
    return all(np.array_equal(s["a"][d], s["b"][d], equal_nan=True) for d in DIRS)


def select_cells(rho, gamma, cf_rho, control_rho):
    assert rho.dtype.kind in 'iu' and gamma.dtype.kind in 'iu' and cf_rho.dtype.kind in 'iu'
    assert rho.shape == gamma.shape == cf_rho.shape and rho.shape[1] == 2
    margin = np.minimum(rho - np.asarray(control_rho)[None,:], gamma)
    return np.argmax(margin, axis=0).tolist(), np.argmax(cf_rho, axis=0).tolist()


def zdict(scores):
    return {c:{d:zscore_rows(torch.as_tensor(np.asarray(scores[c][d]),dtype=torch.float32)) for d in DIRS} for c in CONDS}


def combine(zbase,term,u,a):
    out={c:{} for c in CONDS}
    for c in CONDS:
        for d in DIRS:
            score=zbase[c][d]
            if u>0: score=score+u*zbase[c][d]
            if a>0: score=score+a*term[c][d]
            out[c][d]=score.numpy().astype(np.float32)
    return out


def nested(zcos,zu,za,u,a):
    out={c:{} for c in CONDS}
    for c in CONDS:
        for d in DIRS:
            score=zcos[c][d]
            if u>0: score=score+u*zu[c][d]
            if a>0: score=score+a*za[c][d]
            out[c][d]=score.numpy().astype(np.float32)
    assert condition_free(out)
    return out


def integer_statistics(score,parity):
    per=metrics(score)
    counts=[]
    for field in ('r1','gain'):
        q=np.rint(4*per[field]).astype(np.int64)
        assert np.array_equal(q/4,per[field])
        counts.append(np.array([q[parity==h].sum() for h in (0,1)],np.int64))
    return counts


def baseline(cos,tu,tc,parity,picks=None):
    zc,zu,za=zdict(cos),zdict(tu),zdict(tc)
    if picks is None:
        best=np.full(2,-1,np.int64)
        picks=[None,None]
        for u in NESTED_U:
            for a in NESTED_A:
                score=nested(zc,zu,za,u,a)
                rho,_=integer_statistics(score,parity)
                for h in (0,1):
                    if rho[h]>best[h]:
                        best[h],picks[h]=rho[h],[u,a]
    return by_parity({h:nested(zc,zu,za,*picks[h]) for h in (0,1)},parity),picks


def reader_state(post,ep,readers,taus=None):
    probabilities, margins, picks = {},{},{}
    for c in CONDS:
        si,st,ci,ct,_=ep.condition(c)
        X=features(post,A0,si,st,ci,ct)
        probabilities[c]=np.stack([np.asarray(h['model'].predict_proba(h['scaler'].transform(X)),np.float64)
                                  for h in readers['halves']]).mean(0)
        p=probabilities[c]
        assert p.shape==(len(ep.anchor),3) and np.isfinite(p).all()
        ordered=np.sort(p,axis=1)
        margins[c]=ordered[:,-1]-ordered[:,-2]
        picks[c]=p.argmax(1)
    if taus is None:
        taus=np.percentile(np.concatenate([margins[c] for c in CONDS]),[0,25,50,75])
    assert np.asarray(taus).shape==(4,) and np.isfinite(taus).all()
    stack=stack_dots(post,ep,A0)
    term={c:{d:np.einsum('nh,nhk->nk',probabilities[c],stack[d].astype(np.float64)).astype(np.float32)
             for d in DIRS} for c in CONDS}
    zterm=zdict(term)
    gated,cf,gates=[],[],[]
    for tau in taus:
        gate={c:((margins[c]>=tau)&(picks[c]==0)).astype(np.float32) for c in CONDS}
        g={c:{d:torch.as_tensor(gate[c])[:,None]*zterm[c][d] for d in DIRS} for c in CONDS}
        mean={d:((g['a'][d].numpy().astype(np.float64)+g['b'][d].numpy().astype(np.float64))*0.5).astype(np.float32) for d in DIRS}
        matched={c:{d:torch.as_tensor(mean[d].copy()) for d in DIRS} for c in CONDS}
        gated.append(g);cf.append(matched);gates.append(gate)
    extra={f'P__{c}':probabilities[c] for c in CONDS}
    extra.update({f'margin__{c}':margins[c] for c in CONDS})
    extra.update({f'pick__{c}':picks[c] for c in CONDS})
    extra.update({f'aff_gate__{c}':np.stack([g[c] for g in gates]) for c in CONDS})
    return gated,cf,np.asarray(taus),extra


def reader_cells(B,gated,cf,parity,frozen=None):
    zB=zdict(B)
    if frozen is None:
        sums=sorted({u+a for u in NESTED_U for a in NESTED_A})
        control=np.stack([integer_statistics(combine(zB,zB,s,0),parity)[0] for s in sums])
        sigma_idx=control.argmax(0)
        sigma=[sums[int(j)] for j in sigma_idx]
        rho_ctrl=control[sigma_idx,np.arange(2)]
        rho,gamma,cf_rho=[],[],[]
        for cell in range(224):
            t,u,a=cell_params(cell)
            r,g=integer_statistics(combine(zB,gated[t],u,a),parity)
            control_score=combine(zB,cf[t],u,a)
            assert condition_free(control_score)
            c,_=integer_statistics(control_score,parity)
            rho.append(r);gamma.append(g);cf_rho.append(c)
        aff_cells,cf_cells=select_cells(np.stack(rho),np.stack(gamma),np.stack(cf_rho),rho_ctrl)
        frozen=dict(aff=aff_cells,cf=cf_cells,sigma_star=sigma,
                    switched_off=all(cell_params(c)[2]==0 for c in aff_cells))
    scores={}
    for name,terms in (('aff_fused',gated),('aff_cf',cf)):
        cells=frozen['aff' if name=='aff_fused' else 'cf']
        by={}
        for h,cell in enumerate(cells):
            t,u,a=cell_params(cell)
            by[h]=combine(zB,terms[t],u,a)
            if name=='aff_cf': assert condition_free(by[h])
        scores[name]=by_parity(by,parity)
    return scores,frozen


def rca_fusion(cos,term,parity,picks=None):
    from src.eval.aspect_scorers import fused_scores
    if picks is None:
        candidates=[0.,.25,.5,1.,2.,4.,8.,16.,float('inf')]
        cache={}
        def criterion(lam,h):
            if lam not in cache: cache[lam]=metrics(fused_scores(cos,term,lam))
            m=cache[lam]
            return .5*(m['r1'][parity==h].mean()+m['gain'][parity==h].mean())
        picks=[]
        for h in (0,1):
            best=max(candidates,key=lambda lam:criterion(lam,h))
            if best==16.: best=max(candidates+[32.,64.],key=lambda lam:criterion(lam,h))
            picks.append('inf' if np.isinf(best) else best)
    decoded=[float(v) for v in picks]
    return by_parity({h:fused_scores(cos,term,decoded[h]) for h in (0,1)},parity),picks
