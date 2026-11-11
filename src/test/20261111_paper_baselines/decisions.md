# r25 decisions

One line per result-shaping choice: when · the choice · the main alternative · where · who · could it change a verdict
or claim. User decisions (2026-11-11, paper chat) are in the handoff's section 2 and are not repeated here.

| When | Choice | Main alternative | Where | Who | Changes a claim? |
|---|---|---|---|---|---|
| 2026-11-11 05:10 | BERTopic per painting: one document per held painting, its captions joined with a space in annotation-file order; one topic per painting | per-caption topics, painting topic by majority vote | `pb_inputs.py`, `pb_bertopic.py` | run chat (handoff default) | maybe |
| 2026-11-11 05:10 | BERTopic topic count: default pipeline (MiniLM, UMAP, HDBSCAN, c-TF-IDF), reduced to 20 topics, outliers reassigned with `reduce_outliers` so every painting has a topic; three UMAP seeds (0, 1, 2) averaged like the three k-means states | k-means with 20 clusters as BERTopic's cluster model | `pb_bertopic.py` | run chat (handoff default) | maybe |
| 2026-11-11 05:10 | BERTopic is scored by its own topics (no k-means on top), like PercepT's own-topic read-out | also k-means at 20 on its MiniLM document embeddings | `pb_score.py` | run chat | maybe |
| 2026-11-11 05:10 | VLM2Vec-V2 prompt: Table 1's aspect instruction verbatim, placed in VLM2Vec's image-query format (its image token, then the instruction); exact strings in each store's `meta.json` | rewording into VLM2Vec's "Represent the given image ..." style | `pb_paths.INSTRUCTIONS`, `pb_vlm2vec.py` | run chat | maybe |
| 2026-11-11 05:10 | VLM2Vec-V2 read-out: k-means K = 20 as c* (MiniBatchKMeans, n_init 3, batch 4,096, states 0 to 2), on its unit image embeddings | none | `pb_score.py` | run chat | no |
| 2026-11-11 05:10 | Interval and label: r17's `ns_clu.paired_cell` (2,000 joint row and c*-seed draws, seed 0); r = Qwen c*'s own validation AMI-at-20 seed range from r18's stage 1 record; WIN / LEVEL / BELOW by r10's rule | r17's recorded CLIP L/14 AMI-at-20 range | `pb_score.py`, `pb_paths.RANGES` | run chat | maybe |
| 2026-11-11 05:10 | Side envs: venvs over the CoSiR env (`--system-site-packages`, r18's TPIPS precedent), one for VLM2Vec, one for BERTopic; the CoSiR env is not changed | a separate conda env | `pb_paths.VLM2VEC_ENV`, `pb_paths.BERTOPIC_ENV` | run chat (handoff minor point) | no |
