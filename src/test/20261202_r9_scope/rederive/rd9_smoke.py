"""Small labels-only A28 smoke on the exact shared-interface columns."""
import numpy as np
import pandas as pd
from rd9_episodes import build_a28,FIELDS


def main():
    records=[]
    # Disjoint paintings spanning all 5³ label combinations; language-local lacks is unambiguous.
    for lang in ('Burmese','Darija','Hindi'):
        for e in range(5):
            for s in range(5):
                for g in range(5):
                    for painting in range(8):
                        for caption in range(2):
                            records.append(dict(row_id=len(records),painting=f'{lang}_{e}_{s}_{g}_{painting}',
                                                language=lang,emotion=e,style=s,genre=g,held=False,dropped=caption==1))
    table=pd.DataFrame(records)
    values={a:list(range(5)) for a in ('emotion','style','genre')}
    for seed in (9001,9002,9003):
        eps,rec=build_a28(table,values,seed,total=12,real=False)
        again,repeat=build_a28(table,values,seed,total=12,real=False)
        assert rec==repeat
        for key,ep in eps.items():
            assert len(ep.anchor)==12
            for f in FIELDS: np.testing.assert_array_equal(getattr(ep,f),getattr(again[key],f))
            assert not table.dropped.iloc[ep.rows()].any()
            assert rec[key]['passes'][0]['shares']=={'Burmese':4,'Darija':4,'Hindi':4}
            for j in range(12):
                member=np.concatenate([getattr(ep,f)[j].reshape(-1) for f in FIELDS])
                assert len(table.language.iloc[member].unique())==1
    print('A28 synthetic smoke: 3 seeds × 3 pairs; deterministic hashes, language membership and drop masks pass')


if __name__=='__main__': main()
