"""Independent integer arithmetic of R9 §4 and §5.1, R6 §3."""
import numpy as np


def shares(sizes, total=4096):
    assert isinstance(total, int) and total > 0
    assert all(isinstance(n, (int, np.integer)) and n >= 0 for n in sizes)
    denominator = sum(int(n) for n in sizes)
    assert denominator > 0
    numerators = [total * int(n) for n in sizes]
    counts = [n // denominator for n in numerators]
    order = sorted(range(len(sizes)), key=lambda j: (-(numerators[j] % denominator), j))
    for j in order[:total - sum(counts)]:
        counts[j] += 1
    assert sum(counts) == total
    return counts


def holm(counts, ids):
    assert len(ids) == len(set(ids)) and ids
    assert all(isinstance(counts[j], (int, np.integer)) and 0 <= counts[j] <= 5000 for j in ids)
    order = sorted(ids, key=lambda j: (counts[j], ids.index(j)))
    ok, checks = True, {}
    for k, j in enumerate(order, 1):
        factor = len(ids) + 1 - k
        own = 40 * (int(counts[j]) + 1) * factor <= 5001
        ok = ok and own
        boundary = 5001 // (40 * factor) - 1
        checks[j] = dict(holm_k=k, pass_=ok, own_count_passes=own,
                         n_star_k=boundary, within_one_of_boundary=abs(counts[j]-boundary)<=1,
                         not_reached=own and not ok, alpha=0.025/factor)
        checks[j]['pass'] = checks[j].pop('pass_')
    return dict(order=order, checks=checks, passed=ok)


CHECK_IDS = [f'P{i}' for i in range(1, 8)]


def decide(counts, a28=False):
    assert set(counts) == set(CHECK_IDS)
    if not a28:
        family = holm(counts, CHECK_IDS)
        return dict(outcome='GO' if family['passed'] else 'NO-GO', families={'seven': family}, tested=CHECK_IDS)
    one, two = ['P4','P6'], ['P1','P2','P3','P5','P7']
    level1 = holm(counts, one)
    level2 = holm(counts, two) if level1['passed'] else None
    claim = 'NO-GO' if level2 is None else ('full claim' if level2['passed'] else 'small claim')
    return dict(outcome=claim, families={'level1':level1, 'level2':level2}, tested=one + (two if level2 else []))


N_BOOT, BOOT_SEED, CHUNK = 5000, 42, 250

class Draws:
    """The cluster bootstrap's resample indices (rule §3 'Draws'): clusters by numpy.unique(..., return_inverse),
    rng = default_rng(42), chunks of 250 rows of rng.integers(0, k, size=(rows, k)). Generated once and shared by
    every check, as the rule requires."""

    def __init__(self, clusters):
        _, self.idx = np.unique(np.asarray(clusters), return_inverse=True)
        self.k = int(self.idx.max()) + 1
        rng = np.random.default_rng(BOOT_SEED)
        self.chunks = []
        for start in range(0, N_BOOT, CHUNK):
            self.chunks.append(rng.integers(0, self.k, size=(min(CHUNK, N_BOOT - start), self.k)))
        self.counts = np.bincount(self.idx, minlength=self.k).astype(np.float64)

    def boots(self, values):
        values = np.asarray(values, dtype=np.float64)
        sums = np.bincount(self.idx, weights=values, minlength=self.k)
        return np.concatenate([sums[dr].sum(axis=1) / self.counts[dr].sum(axis=1) for dr in self.chunks])

    def int_sums(self, values):
        """Per resample, the integer 4·Σ (cluster sums of the drawn clusters)."""
        values = np.asarray(values, dtype=np.float64)
        v4 = np.rint(4 * values).astype(np.int64)
        if not np.array_equal(v4.astype(np.float64) / 4, values):
            raise AssertionError("a per-episode difference is not a multiple of 0.25")
        s4 = np.zeros(self.k, dtype=np.int64)
        np.add.at(s4, self.idx, v4)
        return np.concatenate([s4[dr].sum(axis=1) for dr in self.chunks])

def variance_split(y, clusters):
    """R3 rule §6.1 one-way decomposition by anchor painting: σ_ε² = within mean square; σ_a² = max(0, (between mean
    square − σ_ε²) / n₀), n₀ = (n − Σ m_p² / n) / (P − 1)."""
    y = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    P = int(inv.max()) + 1
    n = len(y)
    m = np.bincount(inv, minlength=P).astype(np.float64)
    means = np.bincount(inv, weights=y, minlength=P) / m
    grand = y.mean()
    msb = float(np.sum(m * (means - grand) ** 2) / (P - 1))
    msw = float(np.sum((y - means[inv]) ** 2) / (n - P))
    n0 = float((n - np.sum(m ** 2) / n) / (P - 1))
    return {"sigma_a2": max(0.0, (msb - msw) / n0), "sigma_e2": msw, "n0": n0, "n": n, "P": P,
            "msb": msb, "msw": msw, "sum_m2": float(np.sum(m ** 2))}


CHECKS = {'P1':('r1','cosine'), 'P2':('r1','rca'), 'P3':('r1','B'), 'P4':('r1','B0'),
          'P5':('r1','aff_cf'), 'P6':('gain','aff_cf'), 'P7':('gain','rca')}


def pass_record(arrays,a28=False,sensitivity=False):
    from src.eval.aspect_metrics import cluster_bootstrap
    clusters=np.asarray(arrays['cl'])
    assert len(clusters)>0
    for name in ('cosine','B','B0','aff_cf'):
        assert np.array_equal(arrays[name+'__gain'],np.zeros(len(clusters)))
    draws=Draws(clusters)
    checks,boots_of,counts,variance={},{},{},{}
    for j,(metric,comparator) in CHECKS.items():
        difference=arrays['aff_fused__'+metric]-arrays[comparator+'__'+metric]
        assert difference.shape==clusters.shape and np.isfinite(difference).all()
        boots=draws.boots(difference)
        ints=draws.int_sums(difference)
        counts[j]=int(np.count_nonzero(ints<=0))
        assert counts[j]==int(np.count_nonzero(boots<=0))
        ci=np.percentile(boots,[2.5,97.5])
        ref=cluster_bootstrap(difference,clusters)
        assert list(ci)==ref['ci95']
        checks[j]=dict(n=counts[j],point=100*float(difference.mean()),ci95=(100*ci).tolist(),tested=False)
        boots_of[j]=boots
        if sensitivity:
            assert draws.k>1 and len(clusters)>draws.k
            variance[j]=variance_split(100*difference,clusters)
    decision=decide(counts,a28)
    families=dict(decision['families'])
    if a28 and families['level2'] is None:
        # The runner computes level 2's Holm (m = 5) even when level 1 fails; the verdict step decides what is tested.
        # decision keeps level2 None (nothing tested); the quantities below exist for all seven checks.
        families['level2']=holm(counts,['P1','P2','P3','P5','P7'])
    for family in families.values():
        for j,details in family['checks'].items():
            row=checks[j]
            alpha=details['alpha']
            row.update(details)
            row['tested']=True    # 'computed': all seven in every record; the runner's schema carries no such flag
            row['passes']=row.pop('pass')
            row['near_boundary']=row.pop('within_one_of_boundary')
            row['ci_holm']=(100*np.percentile(boots_of[j],[100*alpha,100*(1-alpha)])).tolist()
            row['level_two_sided']=1-2*alpha
    out=dict(n_episodes=len(clusters),n_clusters=draws.k,checks=checks,decision=decision)
    if not a28: out['holm_order']=decision['families']['seven']['order']
    else:
        out['holm_order']={'level1':families['level1']['order'],'level2':families['level2']['order']}
    if sensitivity: out['sensitivity']=variance
    return out
