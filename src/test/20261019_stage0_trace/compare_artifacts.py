from pathlib import Path
import subprocess, json, re, hashlib
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
R=Path('/project/CoSiR')
B=Path('/project/CoSiR-buddy_prototype_conditioning')
D=Path('/project/CoSiR-wt/r10-pc-08-queue/src/test/20261018_percept_core/sept_stage0/stage0_outputs/1792363233321524805')
P='src/test/20260922_percept_topic_pipeline/'
U='src/test/20260923_artelingo_buddy_analysis/'
paths=[P+x for x in ['run_percept_stage1_embedding_snapshot_pilot.py','run_percept_stage2_extended_seed_pilot.py','run_percept_stage1_pilot.py','run_percept_stage1_cluster_count_sweep_pilot.py','run_percept_stage2_sweep_pilot.py']]+[U+x for x in ['run_pipeline.py','run_affect_pilot.py','run_cca_audit_pilot.py']]
def git(*args): return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()
for p in paths:
    last=git('log','-1','--format=%h','1b70d7f','--',p)
    refs=['3b3ce61','4e4480b','1b70d7f'] if 'embedding_snapshot' not in p else ['4e4480b','1b70d7f']
    blobs=[git('rev-parse',f'{rev}:{p}') for rev in refs]
    actual=hashlib.sha1(b'blob '+str(len((B/p).read_bytes())).encode()+b'\0'+(B/p).read_bytes()).hexdigest()
    print('CODE',p,'last='+last,'same_across_'+','.join(refs)+'='+str(len(set(blobs))==1),'worktree_same='+str(actual==blobs[-1]),'blob='+blobs[-1][:12])
print('TRANSITIVE_DIFF',repr(git('diff','--name-only','3b3ce61','1b70d7f','--','src/utils','src/conditional_buddy')))
for stage in ['stage1_embedding_snapshot','stage2_extended_seed']:
    name='percept_'+stage+'_pilot_report.md'
    def rows(file):
        return [[x.strip() for x in line.split('|')[1:-1]] for line in file.read_text().splitlines() if line.startswith('|') and ('held-out emotion AMI' in line or 'held-out genre AMI' in line)]
    old,new=rows(B/'res/percept'/name),rows(D/name)
    for a,b in zip(old,new):
        print('REPORT_ROW',stage,'September='+json.dumps(a),'today='+json.dumps(b))
with np.load(B/'res/percept/percept_stage1_embedding_snapshot.npz',allow_pickle=True) as z:
    print('SNAPSHOT', 'K=',z['n_initial_clusters'].item(),z['n_surviving_clusters'].item(),'seed=',z['seed'].item())
    for split in ['train','heldout']:
        print('NPZ',split,'N=',len(z[split+'_paintings']),'emotion_AMI=',adjusted_mutual_info_score(z[split+'_emotion'],z[split+'_topic']))
        mask=z[split+'_genre']!=''
        print('NPZ',split,'genre_N=',int(mask.sum()),'genre_AMI=',adjusted_mutual_info_score(z[split+'_genre'][mask],z[split+'_topic'][mask]))
    print('SEPT_SURVIVING',z['surviving_indices'].tolist())
old=(B/(P+'percept_stage1_cluster_count_sweep_v2_pilot_report.md')).read_text()
oldpre=re.findall(r'epoch (\d+): ([0-9.]+)',old.split('## Phase 1:')[1].split('\n\n|')[0])
newlog=(D/'run_percept_stage1_embedding_snapshot_pilot.log').read_text()
newpre=re.findall(r'Pretrain epoch (\d+)/100: mean reconstruction=([0-9.]+)',newlog)
print('ORIGINAL_LOCAL_CITATION_PRETRAIN',[(int(e),v.rstrip(".")) for e,v in oldpre])
print('TODAY_PRETRAIN',[(int(e),v.rstrip(".")) for e,v in newpre])
print('LOCAL_CITATION_PRETRAIN_MATCH',[(int(e),v.rstrip(".")) for e,v in oldpre]==[(int(e),v.rstrip(".")) for e,v in newpre])
