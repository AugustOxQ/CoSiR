import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
with np.load('/project/CoSiR/res/percept/percept_stage1_embedding_snapshot.npz',allow_pickle=True) as z:
    x=z['train_latent_pre']; old=z['train_initial_label']
km=KMeans(n_clusters=60,n_init=10,random_state=42).fit(x)
print('same historical pretrain latents; today KMeans threads=8; ARI=',adjusted_rand_score(old,km.labels_),'exact_label_fraction=',float(np.mean(old==km.labels_)))
