import numpy as np
import pytest
import rd9_stats as S


def test_largest_remainder_shares_use_language_order_for_ties():
    assert S.shares([1, 1, 1], 5) == [2, 2, 1]
    assert S.shares([2, 3, 5], 7) == [1, 2, 4]
    assert S.shares([0, 1, 1], 3) == [0, 2, 1]
    with pytest.raises(AssertionError):
        S.shares([0, 0], 4096)


@pytest.mark.parametrize('m, boundaries', [(7, [16,19,24,30,40,61,124]), (5, [24,30,40,61,124]), (2, [61,124])])
def test_holm_rule_integer_boundaries(m, boundaries):
    ids = [f'P{j}' for j in range(m)]
    result = S.holm(dict(zip(ids, boundaries)), ids)
    assert result['order'] == ids
    assert all(row['pass'] for row in result['checks'].values())
    for k in range(m):
        counts = [0] * k + [boundaries[k] + 1] * (m-k)
        result = S.holm(dict(zip(ids, counts)), ids)
        assert [result['checks'][j]['pass'] for j in ids] == [True]*k + [False]*(m-k)


def test_holm_tie_order_and_stop_even_when_later_count_passes():
    result = S.holm({'P4':62, 'P6':62}, ['P4','P6'])
    assert result['order'] == ['P4','P6']
    assert not result['checks']['P6']['pass']
    assert result['checks']['P6']['own_count_passes']
    assert result['checks']['P6']['not_reached']


@pytest.mark.parametrize('counts, claim', [([0]*7, 'full claim'), ([25,25,25,0,25,0,25], 'small claim'), ([0,0,0,125,0,125,0], 'NO-GO')])
def test_two_levels_gate_claims_and_do_not_test_level_two_after_failure(counts, claim):
    result = S.decide(dict(zip(S.CHECK_IDS, counts)), a28=True)
    assert result['outcome'] == claim
    if claim == 'NO-GO':
        assert result['families']['level2'] is None
        assert result['tested'] == ['P4','P6']
    else:
        assert result['families']['level1']['order'] == ['P4','P6']
        assert result['families']['level2']['order'] == ['P1','P2','P3','P5','P7']


def test_cluster_counts_include_zero_and_pool_repeated_paintings():
    draws = S.Draws(np.array(['same','same','other']))
    # Every mean is positive, and the observation-weighted point is 1/6.
    values = np.array([0.25, 0.25, 0.0])
    assert np.count_nonzero(draws.int_sums(np.zeros(3)) <= 0) == 5000
    assert np.count_nonzero(draws.int_sums(np.full(3, 0.25)) <= 0) == 0
    assert set(np.unique(draws.boots(values))) == {0.0, 1/6, 0.25}
    with pytest.raises(AssertionError):
        draws.int_sums(np.array([0.1, 0.25, 0.0]))


def test_strict_ties_miss_and_gain_is_quarter_integer():
    from rd9_scores import metrics
    scores = {c: {d: np.array([[3,3,0],[3,2,0]], np.float32) for d in ('i2t','t2i')} for c in ('a','b')}
    got = metrics(scores)
    np.testing.assert_array_equal(got['r1'], [0,0.5])
    np.testing.assert_array_equal(got['gain'], [0,0])


def test_cell_selection_uses_integer_minimum_and_first_tie():
    from rd9_scores import select_cells
    # cell 0 wins R@1 but loses gain; cell 1 wins min-margin; cell 2 ties it.
    rho = np.array([[20,20],[18,19],[18,19]])
    gamma = np.array([[0,0],[6,7],[6,7]])
    cf = np.array([[10,10],[11,11],[11,11]])
    assert select_cells(rho, gamma, cf, [12,12]) == ([1,1], [1,1])


def test_agreement_is_exact_for_discrete_values_and_arrays():
    from rd9_compare import compare_value,compare_array
    assert not compare_value(1,2,'checks.P4.n')
    assert not compare_value(True,False,'checks.P4.passes')
    assert compare_value(1.0,1.0+5e-10,'checks.P4.point')
    assert not compare_value(1.0,1.0+2e-9,'checks.P4.point')
    assert compare_value(0.5,0.5+1e-10,'taus.1')
    assert not compare_array(np.array([1.0]),np.array([1.0],np.float32))
    assert not compare_array(np.array([1.0]),np.array([1.0+1e-12]))


def test_unexpected_runtime_error_does_not_set_language_aside():
    from rd9_episodes import failed_draw_count
    assert failed_draw_count('emotion x style: could not fill 4 episodes (41 failed draws); the pools are too small for these constraints', 'emotion','style',4) == 41
    assert failed_draw_count('unexpected failed IO', 'emotion','style',4) is None
    assert failed_draw_count('emotion x style: could not fill 4 episodes (40 failed draws); the pools are too small for these constraints', 'emotion','style',4) is None


def test_agreement_requires_exact_frozen_weights_and_sigma():
    from rd9_compare import compare_value
    assert not compare_value(0.5,0.5+1e-12,'picks.rca_lambda.0')
    assert not compare_value(0.0,1e-12,'picks.sigma_star.0')


def test_pin_paths_keep_round9_outputs_in_the_running_worktree(monkeypatch,tmp_path):
    import rd9_inputs as I
    here=tmp_path/'checkout/src/test/20261202_r9_scope'
    main=tmp_path/'main'
    monkeypatch.setattr(I,'HERE',here);monkeypatch.setattr(I,'MAIN',main)
    assert I.resolve_pin('src/test/20261202_r9_scope/results/new.npz') == tmp_path/'checkout/src/test/20261202_r9_scope/results/new.npz'
    assert I.resolve_pin('src/test/20261031_pseudo_partitions/results/partitions.npz') == tmp_path/'main/src/test/20261031_pseudo_partitions/results/partitions.npz'


def test_h5_archive_uses_concatenated_arrays_and_checks_stored_hashes():
    from rd9_phase2 import h5_parts
    from rd9_episodes import FIELDS
    # Minimal schema fixture; rows retain their original global positions.
    archive={'pair_index':np.repeat(np.arange(3),2), 'pair_names':np.array(['emotion__style','emotion__genre','style__genre'])}
    for f in FIELDS:
        archive[f]=np.arange(6) if f=='anchor' else np.zeros((6,13 if f=='candidates' else 4),np.int64)
    parts=h5_parts(archive,per_pair=2,check_hashes=False)
    np.testing.assert_array_equal(parts['style__genre'].anchor,[4,5])
    assert parts['emotion__genre'].candidates.shape==(2,13)
    with pytest.raises(AssertionError):
        h5_parts(archive,per_pair=3,check_hashes=False)


def test_arm_l_draw_identity_hashes_sorted_row_ids_without_changing_fit_order():
    from rd9_bundle import head_draw_hash
    rows=np.array([2,0,1],np.int64)
    assert head_draw_hash(rows)=='ab25350e3e65efebe24584461683ecda68725576e825e550038b90e7b1479946'
    np.testing.assert_array_equal(rows,[2,0,1])


@pytest.mark.parametrize('backbone', ['clip_b32','qwen3vl_emb_2b','siglip2_so400m'])
def test_phase1_agreement_checks_every_pinned_quantity(backbone):
    from rd9_compare import compare_phase1, Comparison
    from rd9_inputs import sha_bytes
    arrays={f'{name}__{metric}':np.array([0.25,0.5]) for name in ('B','B0','B1','aff_fused','aff_cf','cosine','rca') for metric in ('r1','gain','other','swap','strict')}
    for name,values in {'B':[0.,.5],'B0':[.25,.5],'B1':[.25,.75],'aff_fused':[.5,.75],'aff_cf':[.5,1.]}.items():
        arrays[name+'__r1']=np.array(values)
    own={'backbone':backbone,'picks':{'aff':[1,2],'cf':[3,4],'sigma_star':[0.,0.], 'switched_off':False,
        'gate_share':0.25,'B':[[1.,2.],[3.,4.]],'B0':[[2.,4.],[4.,8.]],'rca_lambda':[0.5,1.], 'taus':[0.,0.2,0.4,0.6]},
        'stats':{'checks':{j:{'point':float(k)+.125,'n':k} for k,j in enumerate(S.CHECK_IDS)},'n_episodes':2,'n_clusters':2,
                 'sensitivity':{j:{'sigma_a2':k+.5,'sigma_e2':k+.25} for k,j in enumerate(S.CHECK_IDS)}}}
    sigma={j:{'sigma_a2':k+.5,'sigma_eps2':k+.25} for k,j in enumerate(S.CHECK_IDS)}|{'N':2,'n_paintings':2}
    cells={k:own['picks'][k] for k in ('aff','cf','sigma_star','switched_off','gate_share')}
    picks={name:{str(h):own['picks'][name][h] for h in (0,1)} for name in ('B','B0')}
    base={'lambda_picks':{'rca':{'0':0.5,'1':1.}}}
    record={'mean_r1':{'B':25.,'B0':37.5,'B1':50.,'AFF':62.5,'CF':75.},
            'checks':{j:{'point':float(k)+.125,'n_j':k} for k,j in enumerate(S.CHECK_IDS)},
            'array_hashes':{key:{'sha256':sha_bytes(v),'dtype':str(v.dtype),'shape':list(v.shape)} for key,v in arrays.items()}}
    c=Comparison(); compare_phase1(c,own,arrays,cells,picks,base,{'taus':own['picks']['taus']},record,sigma)
    assert c.record()['agreement']
    # Every required field can independently turn the result into a disagreement.
    cells['switched_off']=True
    record['checks']['P7']['n_j']=999
    record['array_hashes']['B__r1']['sha256']='0'*64
    record['mean_r1']['B1']=38.
    sigma['P2']['sigma_eps2']=9.
    c=Comparison(); compare_phase1(c,own,arrays,cells,picks,base,{'taus':own['picks']['taus']},record,sigma)
    failed={i['key'] for i in c.record()['items'] if not i['equal']}
    assert {'picks.switched_off','seed42.checks.P7.n','seed42.arrays.B__r1.sha256','seed42.mean_r1.B1','seed42.sigma.P2.sigma_e2'} <= failed


def test_pin_roles_reject_ambiguous_names_and_choose_backbone_recipe():
    from pathlib import Path
    from rd9_inputs import role_paths
    originals={'n6_posteriors':Path('/main/n6_posteriors.npz')}
    with pytest.raises(AssertionError,match='ambiguous'):
        role_paths(originals,[Path('/x/n6_posteriors.npz'),Path('/y/n6_posteriors.npz')],'clip_b32',Path('/round'))
    selected=role_paths(originals,[Path('/main/n6_posteriors.npz'),Path('/round/results/recipe/qwen3vl_emb_2b/n6_posteriors.npz')],'qwen3vl_emb_2b',Path('/round'))
    assert str(selected['n6_posteriors'])=='/round/results/recipe/qwen3vl_emb_2b/n6_posteriors.npz'


def test_library_provenance_is_sha256_of_loaded_src_files():
    import hashlib,src.eval.aspect_episodes as ae
    from pathlib import Path
    from rd9_inputs import library_sha256
    provenance=library_sha256()
    path=Path(ae.__file__)
    assert provenance['src/eval/aspect_episodes.py']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(not key.startswith('src/test/') for key in provenance)


def test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level():
    arrays={'cl':np.arange(20),'aff_fused__r1':np.full(20,.5),'aff_fused__gain':np.full(20,.25)}
    for name in ('cosine','B','B0','aff_cf','rca'):
        arrays[name+'__r1']=np.zeros(20);arrays[name+'__gain']=np.zeros(20)
    arrays['aff_cf__r1']=.5-np.tile([-.25,0,.25,.5],5)
    arrays['rca__r1']=np.full(20,.5)
    arrays['rca__gain']=.25-np.array([-.25]*10+[.25]*8+[0]*2)
    got=S.pass_record(arrays)
    assert got['checks']['P5']['n']==133  # includes 66 zero-sum draws
    assert got['checks']['P7']['n']==3575
    assert got['checks']['P5']['point']==12.5
    assert got['checks']['P7']['point']==-2.5
    assert got['holm_order']==['P1','P3','P4','P6','P5','P7','P2']
    assert got['checks']['P5']['holm_k']==5
    assert got['checks']['P5']['level_two_sided']==1-.05/3
    np.testing.assert_allclose(got['checks']['P5']['ci_holm'],[-1.677083333333336,27.500000000000004],rtol=0,atol=1e-14)
    assert got['checks']['P5']['ci95']==[0.,25.]


def score_fixture(rows=4):
    return {c:{d:np.zeros((rows,13),np.float32) for d in ('i2t','t2i')} for c in ('a','b')}


def test_baseline_ties_choose_first_cell():
    import rd9_scores as F
    zero=score_fixture()
    _,picks=F.baseline(zero,zero,zero,np.arange(4)%2)
    assert picks==[[0.,0.],[0.,0.]]


def test_sigma_ties_choose_smallest_and_switched_off_requires_both_halves():
    import torch,rd9_scores as F
    B=score_fixture()
    for c in B:
        for d in B[c]: B[c][d][:,2]=1
    zero={c:{d:torch.zeros((4,13)) for d in B[c]} for c in B}
    _,picks=F.reader_cells(B,[zero]*4,[zero]*4,np.arange(4)%2)
    assert picks['sigma_star']==[0.,0.]
    assert picks['switched_off'] is True
    term={c:{d:torch.zeros((4,13)) for d in B[c]} for c in B}
    for c in term:
        for d in term[c]: term[c][d][[1,3],0 if c=='a' else 1]=1
    cf={c:{d:(term['a'][d]+term['b'][d])/2 for d in B[c]} for c in B}
    _,picks=F.reader_cells(B,[term]*4,[cf]*4,np.arange(4)%2)
    assert F.cell_params(picks['aff'][0])[2]==0
    assert F.cell_params(picks['aff'][1])[2]>0
    assert picks['switched_off'] is False


def test_assembly_uses_opposite_parity():
    from rd9_scores import by_parity
    one=score_fixture();two=score_fixture()
    for c in one:
        for d in one[c]: one[c][d][:]=1;two[c][d][:]=2
    got=by_parity({0:one,1:two},np.arange(4)%2)
    for c in got:
        for d in got[c]: np.testing.assert_array_equal(got[c][d][:,0],[2,1,2,1])


def test_reader_tau_percentiles_inclusive_gate_and_affect_pick():
    from types import SimpleNamespace
    import rd9_scores as F
    from rd9_episodes import concat,FIELDS
    from src.eval.aspect_episodes import AspectEpisodes
    probs=np.array([[.5,.25,.25],[.25,.5,.25],[.625,.25,.125],[.75,.125,.125]])
    class Predictor:
        def predict_proba(self,X): return probs
    reader={'halves':[{'scaler':SimpleNamespace(transform=lambda X:X),'model':Predictor()}]*2}
    ep=AspectEpisodes('emotion','style',np.arange(4),np.tile(np.arange(13),(4,1)),*[np.zeros((4,4),np.int64) for _ in range(4)])
    p=np.tile([.25,.75],(13,1)).astype(np.float32)
    post={g:{m:p for m in ('img','txt')} for g in F.A0}
    _,_,taus,extra=F.reader_state(post,ep,reader)
    np.testing.assert_array_equal(taus,[.25,.25,.3125,.4375])
    for c in ('a','b'):
        np.testing.assert_array_equal(extra['aff_gate__'+c][0],[1,0,1,1])
        np.testing.assert_array_equal(extra['aff_gate__'+c][3],[0,0,0,1])
        np.testing.assert_array_equal(extra['pick__'+c],[0,1,0,0])


@pytest.mark.parametrize('gap,expected',[(.125,32.),(.08,64.)])
def test_rca_extends_grid_after_sixteen(gap,expected):
    from rd9_scores import rca_fusion
    cos=score_fixture(6);term=score_fixture(6)
    for c in cos:
        target=0 if c=='a' else 1
        for d in cos[c]:
            for row in range(6):
                case=row//2
                cos[c][d][row,target if case==0 else 2]=1
                term[c][d][row,:]=-1
                term[c][d][row,target]=1+(gap if case==1 else .25 if case==2 else 0)
                term[c][d][row,2]=1.02 if case==0 else 1
    _,picks=rca_fusion(cos,term,np.arange(6)%2)
    assert picks==[expected,expected]
