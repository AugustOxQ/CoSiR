"""Rule §5.1 on small, actual language-local episode builds (no feature scores)."""
import numpy as np
import pandas as pd
import pytest
import rd9_episodes as E
from src.eval.aspect_episodes import PaintingValueIndex


@pytest.fixture
def table():
    rows=[]
    for language in ('Burmese','Darija','Hindi'):
        for emotion in range(5):
            for style in range(5):
                for genre in range(5):
                    for caption in range(2):
                        rows.append(dict(row_id=len(rows),painting=f'{language}_{emotion}_{style}_{genre}',
                                         language=language,emotion=emotion,style=style,genre=genre,
                                         held=False,dropped=language=='Hindi' and caption==1))
    return pd.DataFrame(rows)


VALUES={a:list(range(5)) for a in ('emotion','style','genre')}


def test_a28_seeds_post_drop_shares_and_language_order(table):
    episodes,records=E.build_a28(table,VALUES,9001,total=12,real=False)
    labels,groups,_=E.table_arrays(table,real=False)
    for a,b,third in E.PAIRS:
        key=a+'__'+b
        assert records[key]['passes'][0]['shares']=={'Burmese':5,'Darija':5,'Hindi':2}
        expected=[]
        for lang,n,seed in [('Burmese',5,9001001),('Darija',5,9001002),('Hindi',2,9001005)]:
            R=np.flatnonzero(table.language.to_numpy()==lang)
            local={x:labels[x][R] for x in labels}
            idx=PaintingValueIndex(local,groups[R])
            kept=np.flatnonzero(~table.dropped.to_numpy()[R])
            part=E.restricted(local,groups[R],kept,a,b,n,seed,third,idx,(VALUES[a],VALUES[b]),min_paintings=1)
            expected.append({f:R[getattr(part,f)] for f in E.FIELDS})
        for f in E.FIELDS:
            np.testing.assert_array_equal(getattr(episodes[key],f),np.concatenate([part[f] for part in expected]))
        assert not table.dropped.iloc[episodes[key].rows()].any()


def test_a28_dropped_labels_still_count_for_lacks_and_index_precedes_drops(table,monkeypatch):
    # Hindi's first painting has emotion 0 on the kept row and emotion 1 on a dropped row.
    first=int(np.flatnonzero(table.language.to_numpy()=='Hindi')[0])
    table.loc[first+1,'emotion']=1
    original=E.restricted
    seen=[]
    def check(labels,groups,rows,a,b,n,seed,third,index,values,min_paintings=30):
        assert min_paintings==1
        assert len(index.labels['emotion'])==250  # full language, including its 125 drops
        if seed==9001005:
            assert len(rows)==125
            assert groups[0] not in index.lacks('emotion',1)
            assert index.labels['emotion'][1]==1
            seen.append(a+'__'+b)
        return original(labels,groups,rows,a,b,n,seed,third,index,values,min_paintings)
    monkeypatch.setattr(E,'restricted',check)
    E.build_a28(table,VALUES,9001,total=12,real=False)
    assert seen==['emotion__style','emotion__genre','style__genre']


def test_a28_both_set_aside_causes_rebuild_with_new_shares(table):
    table.loc[table.language=='Hindi','emotion']=0
    episodes,records=E.build_a28(table,VALUES,9001,total=12,real=False)
    for key,record in records.items():
        assert len(record['passes'])==2
        assert record['passes'][0]['shares']=={'Burmese':5,'Darija':5,'Hindi':2}
        assert record['passes'][1]['shares']=={'Burmese':6,'Darija':6}
        assert record['set_aside']==[{'language':'Hindi','reason':'failed-draw' if key=='style__genre' else 'values-subset','seed':9001005}]
        langs=table.language.iloc[episodes[key].anchor].tolist()
        assert langs==['Burmese']*6+['Darija']*6


def test_a28_third_set_aside_stops(table):
    fourth=table[table.language=='Burmese'].copy()
    fourth['language']='Igbo';fourth['painting']=fourth.painting.str.replace('Burmese','Igbo')
    table=pd.concat([table,fourth],ignore_index=True);table['row_id']=np.arange(len(table))
    table.loc[table.language!='Igbo','emotion']=0
    with pytest.raises(AssertionError,match='more than two set aside'):
        E.build_a28(table,VALUES,9001,total=12,real=False)


def test_restricted_subset_guard_and_minimum_paintings(table):
    local=table[table.language=='Burmese'].reset_index(drop=True)
    labels,groups,_=E.table_arrays(local,real=False)
    rows=np.arange(len(local));index=PaintingValueIndex(labels,groups)
    with pytest.raises(E.ValuesSubsetError):
        E.restricted(labels,groups,rows,'emotion','style',4,9001001,'genre',index,([0,1,2,3,4,5],VALUES['style']),min_paintings=1)
    with pytest.raises(E.ValuesSubsetError):
        E.restricted(labels,groups,rows,'emotion','style',4,9001001,'genre',index,(VALUES['emotion'],VALUES['style']),min_paintings=30)
    ep=E.restricted(labels,groups,rows,'emotion','style',4,9001001,'genre',index,(VALUES['emotion'],VALUES['style']),min_paintings=1)
    assert len(ep.anchor)==4


@pytest.mark.parametrize('hashes,prior', [({'55':{'a':'x','b':'x'}},set()),({'55':{'a':'x'}},{'x'})])
def test_episode_hash_guard_rejects_duplicates_and_prior_collisions(hashes,prior):
    from rd9_phase2 import hash_guard
    with pytest.raises(AssertionError): hash_guard(hashes,prior)
    hash_guard({'55':{'a':'x','b':'y'}},{'older'})


# --- value_codes reads ticket 14's record list: each aspect is [{"code": int, "name": str}, ...] ---
def _records(codes): return [dict(code=c,name=f'n{c}') for c in codes]


def test_value_codes_reads_list_of_code_name_records():
    rec={'emotion':_records([3,0,1]),'style':_records([9,12]),'genre':_records([7,2])}
    assert E.value_codes(rec)=={'emotion':[0,1,3],'style':[9,12],'genre':[2,7]}


def test_value_codes_still_reads_plain_code_lists():
    assert E.value_codes(VALUES)=={a:list(range(5)) for a in VALUES}


@pytest.mark.parametrize('bad',[
    _records([1,1]),                          # duplicate code
    [dict(code=-1,name='x')],                 # negative
    [dict(code=1.0,name='x')],                # not an int
    [dict(code=1)],                           # name missing
    [dict(code=1,name='x',extra=2)],          # extra key
    [dict(code=1,name='x'),2],                # mixed forms
    {'codes':[1],'names':['x']},              # the old dict form is gone
])
def test_value_codes_rejects_malformed_aspects(bad):
    rec={'emotion':[0],'style':bad,'genre':[0]}  # only `style` is bad; plain lists elsewhere
    with pytest.raises((AssertionError,TypeError)): E.value_codes(rec)


def test_value_codes_on_real_ticket14_file():
    import json
    path=E.MAIN/'src/test/20261202_r9_scope/results/value_sets_a28.json'
    if not path.exists(): pytest.skip('ticket 14 file absent')
    got=E.value_codes(json.loads(path.read_text()))
    assert got['emotion']==list(range(8))
    assert got['style']==[9,12,17,20,21,23,24]
    assert got['genre']==[0,1,2,4,6,7,8,9]


# --- phase 2 loads the read set and the value sets through the pinned reader (ticket 25d review, LOW A) ---
def _a28_input_files(folder):
    import hashlib,json
    folder.mkdir(exist_ok=True)
    pd.DataFrame(dict(row_id=[1,2],painting=['w0','w1'])).to_parquet(folder/'read_set_a28.parquet')
    (folder/'value_sets_a28.json').write_text(json.dumps({'emotion':[{'code':0,'name':'n0'}]}))
    return {f'results/{p.name}':hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_phase2_a28_inputs_are_loaded_from_the_pinned_bytes(tmp_path):
    from rd9_phase2 import load_a28_inputs
    entries=_a28_input_files(tmp_path)
    table,values=load_a28_inputs(entries,tmp_path)
    assert list(table.row_id)==[1,2] and values['emotion'][0]['code']==0


@pytest.mark.parametrize('name',['read_set_a28.parquet','value_sets_a28.json'])
def test_phase2_a28_inputs_refuse_a_file_that_differs_from_its_pins_entry(tmp_path,name):
    from rd9_phase2 import load_a28_inputs
    entries=_a28_input_files(tmp_path)
    entries[f'results/{name}']='0'*64
    with pytest.raises(AssertionError,match=name):
        load_a28_inputs(entries,tmp_path)
