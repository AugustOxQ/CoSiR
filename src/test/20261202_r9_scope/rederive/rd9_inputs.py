"""Direct data boundaries; never imports the round implementation modules."""
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RD = HERE/'rederive'
OUT = RD/'rd9_out'
MAIN = Path('/project/CoSiR')
T = MAIN/'src/test'
sys.path.insert(0,str(MAIN))
BACKBONES = ('clip_b32','qwen3vl_emb_2b','siglip2_so400m')
ROLES = {
    'cells_seed42': '20261202_r9_scope/results/cells_seed42.json',
    'picks_seed42': '20261125_artelingo_held_test/results/picks_seed42.json',
    'step1_heads_style': '20261116_grouping_step1_style/results/step1_heads_style.npz',
    'prepare': '20261013_stage_d_selection/cache/prepare.npz',
    'held_codes': '20261014_stage_d_final/cache/held_codes.npz',
    'partition_L': '20261111_community_told_oracle/results/per_anchor_told_oracle.npz',
    'told_oracle': '20261111_community_told_oracle/results/told_oracle.json',
    'partitions': '20261031_pseudo_partitions/results/partitions.npz',
    'n6_posteriors': '20261108_new_method_quick_checks/results/n6_posteriors.npz',
    'A3_seed42': '20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt',
    'picked': '20261101_aspect_factor_gonogo/results/picked.json',
    'rb_reader_A0': '20261117_reader_fix_csd/results/rb_reader_A0.pkl',
    'rb_reader_A0_record': '20261117_reader_fix_csd/results/rb_reader_A0.json',
    'rc_tau': '20261117_reader_fix_csd/results/rc_tau.json',
    'episodes_seed42': '20261030_aspect_baselines/results/episodes_seed42.npz',
    'baselines_seed42': '20261030_aspect_baselines/results/baselines_seed42.json',
}
FROZEN = {'prepare','held_codes','partition_L','told_oracle','episodes_seed42'}
RULE_HASHES = {
    '20261121_round3_affect_gate':'2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925',
    '20261125_artelingo_held_test':'7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724',
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_reads(record):
    for name, metadata in record.items():
        path=Path(name)
        assert path.suffix!='.py'
        assert sha_file(path)==metadata['sha256'], f'cached input changed: {path}'


def jsonable(value):
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return jsonable(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    return value


def check_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(RD), f'output outside rederive: {path}'
    assert not path.exists(), f'refusing overwrite: {path}'
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def save_json(path, value):
    with check_output(path).open('x') as f:
        json.dump(jsonable(value),f,indent=2,allow_nan=False)
        f.write('\n')


def save_npz(path, value):
    with check_output(path).open('xb') as f:
        np.savez(f,**value)


def resolve_pin(relative):
    path=Path(relative)
    if path.is_absolute(): return path
    if str(path).startswith(f'src/test/{HERE.name}/'):
        return HERE.parents[2]/path
    return (MAIN/path) if str(path).startswith('src/') else HERE/path


def read_pinned(path,entries):
    """The bytes of `path`, after checking their SHA-256 against its one entry in `entries` (path -> SHA-256).

    The entry is found by file name, so a copy kept in another folder is accepted only if it is byte-equal.
    Callers parse these bytes, not the file again, so the checked bytes are the parsed bytes.
    """
    path=Path(path)
    matches=[digest for key,digest in entries.items() if Path(key).name==path.name]
    assert len(matches)==1, f'{path.name}: expected exactly one pins entry, found {len(matches)}'
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==matches[0], f'{path.name}: SHA-256 differs from its pins entry ({path})'
    return raw


def pin_entries(record):
    """Accept explicit {path,sha256} entries or path→SHA maps; no implicit recipe lookup."""
    entries = {}
    def visit(obj):
        if isinstance(obj,list):
            for item in obj: visit(item)
        if not isinstance(obj,dict): return
        if 'path' in obj and 'sha256' in obj:
            entries[str(obj['path'])] = obj['sha256']
        for key,value in obj.items():
            if isinstance(value,str) and len(value)==64 and ('/' in key or '.' in Path(key).name):
                entries[key] = value
            elif isinstance(value,dict) and 'sha256' in value and ('/' in key or '.' in Path(key).name):
                entries[value.get('path',key)] = value['sha256']
            visit(value)
    visit(record)
    return entries


def role_paths(originals,entries,backbone,here):
    result=dict(originals)
    for role,old in originals.items():
        if role in FROZEN: continue
        candidates=[p for p in entries if p.name==old.name]
        if len(candidates)>1:
            recipe=Path(here)/'results/recipe'/backbone
            preferred=[p for p in candidates if p.is_relative_to(recipe)]
            if preferred: candidates=preferred
        assert len(candidates)<=1, f'ambiguous pin role {role}: {candidates}'
        if candidates: result[role]=candidates[0]
    return result


def library_sha256():
    result={}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__name__','')
        filename=getattr(module,'__file__',None)
        if not name.startswith('src.') or name.startswith('src.test.') or not filename: continue
        path=Path(filename).resolve()
        if path.suffix!='.py': continue
        roots=(MAIN/'src',HERE.parents[1])
        root=next((r for r in roots if path.is_relative_to(r)),None)
        if root is not None:
            relative='src/'+str(path.relative_to(root))
            assert not relative.startswith('src/test/')
            result[relative]=sha_file(path)
    assert result, 'no loaded src libraries'
    return dict(sorted(result.items()))


class Inputs:
    def __init__(self,backbone='clip_b32',pins=None):
        assert backbone in BACKBONES
        self.backbone, self.reads = backbone, {}
        self.paths = {role:T/relative for role,relative in ROLES.items()}
        self.hashes = {}
        self.pin_record = None
        if pins is None:
            name = 'clip' if backbone=='clip_b32' else backbone
            pins = HERE/f'pins_{name}.json'
        pins = Path(pins)
        if pins.exists():
            self.pin_record=json.loads(pins.read_text())
            entries = pin_entries(self.pin_record)
            assert entries, 'pins schema must contain path and sha256 entries'
            self.reads[str(pins)] = dict(sha256=sha_file(pins), keys=['file pins'])
            for relative,digest in entries.items():
                path = resolve_pin(relative)
                if path.suffix == '.py':
                    continue  # source digests are metadata; never opened
                assert sha_file(path)==digest, f'pin mismatch: {path}'
                self.hashes[path.resolve()] = digest
            self.paths=role_paths(self.paths,list(self.hashes),backbone,HERE)
            assert all(self.paths[r].resolve() in self.hashes for r in ROLES), 'missing phase-1 pin'
        else:
            assert backbone=='clip_b32', 'new backbones require pins_<backbone>.json'
            # The independent R6 code's constants are data; it is parsed, never imported.
            import ast
            source=(T/'20261125_artelingo_held_test/rederive/rd_common.py').read_text()
            tree=ast.parse(source)
            env={'T':T,'R1DIR':T/'20261117_reader_fix_csd','R3DIR':T/'20261121_round3_affect_gate',
                 'E1':T/'20261030_aspect_baselines/results'}
            node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SHA' for t in n.targets))
            # Literal path expressions and literal digests only; never executes that module.
            for k,v in zip(node.keys,node.values):
                assert isinstance(k,ast.BinOp) and isinstance(k.op,ast.Div)
                self.hashes[(env[k.left.id]/ast.literal_eval(k.right)).resolve()]=ast.literal_eval(v)
            # picked.json and reader record have their checkpoint/pickle identity checked below.
        for folder,digest in RULE_HASHES.items():
            assert sha_file(T/folder/'DECISION_RULE.md') == digest
        assert sha_file(HERE/'DECISION_RULE.md') == R9_RULE_SHA

    def path(self,role,keys=()):
        p=self.paths[role]
        got=sha_file(p)
        if p.resolve() in self.hashes:
            assert got==self.hashes[p.resolve()], f'input hash differs: {p}'
        previous=self.reads.get(str(p),{}).get('keys',[])
        self.reads[str(p)] = dict(sha256=got,keys=sorted(set(previous)|set(keys)))
        return p

    def npz(self,role,keys):
        with np.load(self.path(role,keys),allow_pickle=False) as z:
            assert set(keys)<=set(z.files), (role,keys,z.files)
            return {k:z[k] for k in keys}

    def json(self,role,keys=()):
        return json.loads(self.path(role,keys).read_text())

    def readers(self):
        import sklearn
        record=self.json('rb_reader_A0_record',['config','groupings','feature_names','sklearn_version'])
        reader=pickle.loads(self.path('rb_reader_A0',['config','groupings','feature_names','halves.*.scaler','halves.*.model','sklearn_version']).read_bytes())
        assert reader['config']=='A0' and tuple(reader['groupings'])==('affect','image','caption')
        assert record['config']==reader['config'] and record['groupings']==reader['groupings']
        assert record['feature_names']==reader['feature_names'] and len(reader['feature_names'])==18
        assert record['sklearn_version']==reader['sklearn_version']==sklearn.__version__
        assert len(reader['halves'])==2
        for h in reader['halves']:
            assert np.array_equal(h['model'].classes_,np.arange(3))
        return reader


def feature_data(backbone):
    from src.data.artelingo import load_artelingo, ArtelingoData
    clip=load_artelingo()
    if backbone=='clip_b32': return clip
    folder=Path('/data/SSD2/pre_extract/artelingo_full')/backbone
    return store_data(folder,clip)


def store_data(folder,clip):
    folder=Path(folder)
    index=json.loads((folder/'index.json').read_text())['texts']
    ids=np.array([r['sample_id'] for r in index],np.int64)
    assert len(ids)==len(clip.sample_ids)==len(np.unique(ids))==308723
    loc={int(i):j for j,i in enumerate(ids)}
    assert set(loc)==set(clip.sample_ids.tolist())
    order=np.array([loc[int(i)] for i in clip.sample_ids])
    assert np.array_equal(ids[order],clip.sample_ids)
    assert all(index[j]['painting']==str(clip.paintings[k]) for k,j in enumerate(order))
    image=np.load(folder/'img.npy',mmap_mode='r')
    text=np.load(folder/'txt.npy',mmap_mode='r')
    assert len(text)==len(ids) and image.ndim==text.ndim==2 and image.shape[1]==text.shape[1]
    joined=np.array([index[j]['img'] for j in order])
    assert joined.min()>=0 and joined.max()<len(image)
    img,txt=np.asarray(image[joined],np.float32),np.asarray(text[order],np.float32)
    assert np.isfinite(img).all() and np.isfinite(txt).all()
    from src.data.artelingo import ArtelingoData
    return ArtelingoData(img,txt,clip.sample_ids,clip.emotions,clip.paintings,clip.art_styles)

R9_RULE_SHA = 'f9b07b34e645bc8e881fb13c7e7a085eabea6a52f4d93445305944cf1aea030c'
