# Paper related-work sources: aspect and instruction-conditioned embeddings, distillation, neighbour graphs, affective and art analysis

Literature check for the paper's first stage (related work and method), 2026-11-02, outside the research loop's
steps. Three deep-research agents in parallel, exhaustive depth by the user's request: (a) Opus high, aspect-specific
and conditional representations, instruction-conditioned multimodal embedders, and the closest-prior check; (b)
Sonnet high, distillation of embeddings and neighbour-graph contrastive learning, plus the method's citations; (c)
Sonnet high, affective and artistic image analysis. Each work carries its status: in the earlier verified
bibliography, from an earlier literature note, verified now (page opened), or excluded. The three reports follow
verbatim (the paper chat saved them; agent c's addendum replaces two of its entries).

Closest-prior answer (agent a, part C): no work found that does any of our three points in our form, each with close
prior work: (i) distilling one instruction embedder's per-aspect views into a light multi-head student (closest
FocalLens, DIOR, UNIC and AM-RADIO, Eddy-VL); (ii) per-aspect neighbour graphs from instructed embeddings as
contrastive positives (closest language-guided sampling, X-Sample, TAC, guideline-driven clustering); (iii) a student
that exceeds its teacher, known in general (Born-Again, RKD, AM-RADIO, weak-to-strong), so only our setting can be
claimed. An absence in about 15 targeted searches is not proof of novelty; a final arXiv sweep is due the week
before submission.


---

## Sources, H1 and H2 (aspect-specific and conditional representations; instruction-conditioned embedders) and the closest-prior check

Deep-research gap-fill (Opus high), 2026-11-02, for the first draft. Saved verbatim from the agent's report by the paper chat. Key aliases: abbasikoohpayegani2020compress = koohpayegani2020compress, tian2019crd = tian2020crd, furlanello2018bornagain = furlanello2018born (use the second key of each pair; refs.bib holds only those).

## Related Work sources for H1 and H2, with the closest-prior check (CoSiR, CVPR draft)

I checked 74 works for H1, H2 and the closest-prior question. The searches found no paper that does any of our three points in our form, but each point has close prior work.

- **Point (i), distilling one embedder's instructed views into a multi-head student:** closest are FocalLens, DIOR, UNIC/AM-RADIO and Eddy-VL.
- **Point (ii), one neighbour graph per aspect used as contrastive positives:** closest are language-guided sampling, X-Sample, TAC and guideline-driven clustering.
- **Point (iii), a student that beats its teacher:** this is known in general (Born-Again, RKD, AM-RADIO, weak-to-strong), so only our specific setting can be claimed.

Budget: 24 of 40 searches and all 30 page reads. Depth: exhaustive (capped by the brief). Model: Opus.

Status codes used below:
- `bib`: in the archived refs.bib with a `% verified:` line.
- `note: <file>`: a literature note states it and does not flag it.
- `verified now: <URL>`: page opened in this session.
- "API": the arXiv export API record (title, authors, comment, journal_ref), fetched at `https://export.arxiv.org/api/query?id_list=...`.

## A. Sources per heading

### H1. Aspect-specific, conditional and perceptual image representations

**Conditional similarity**
- `veit2017csn` | bib | Conditional Similarity Networks: a condition id gates embedding dimensions; image only (CVPR 2017). | Closest to our "one head per aspect". Their condition is a supervised id and is an input at test time; our heads are trained without labels from instructed teacher views.
- `tan2019scenet` | bib | SCE-Net mixes learned masks, with weights inferred from the compared items and no condition labels (ICCV 2019). | Label-free conditions, but learned from triplets in image space. No teacher, no captions.
- `ye2022discovernet` | bib | DiscoverNet learns condition embeddings from triplets without condition labels (CVPR 2022). | Same difference as SCE-Net.
- `shi2025generalizedcsl` | bib | Generalized Conditional Similarity Learning via Semantic Matching (TPAMI 2025); conditions run from supervised to semi-supervised. | Image only; triplet-based.
- `ma2020asen` | bib | ASEN (AAAI 2020): an attribute id drives attention to give attribute-specific fashion embeddings. | Supervised, given attributes; ours learns emotion, style and genre without attribute labels.
- `dong2021asenpp` | bib | ASEN++ (TIP 2021), the extended version. | Same difference.
- `plummer2018conditional` | bib | Conditional Image-Text Embedding Networks (ECCV 2018): a phrase softly selects one of K learned subspaces. | Cross-modal and multi-subspace like ours, but the subspaces are driven by the phrase itself and are not named aspects.
- `vaze2023genecis` | bib | GeneCIS (CVPR 2023): a benchmark for focusing on or changing an attribute or object, given by text. | The standard conditional-similarity benchmark. Our condition is a head choice, not text.
- `liu2025crl` | bib | CRL (NeurIPS 2025): an LLM writes descriptive texts for a criterion; frozen CLIP projects onto them. Training-free. | A named-criterion comparator on frozen CLIP. Its "head" is a text basis; ours is distilled from an MLLM embedder.
- `wang2026tpips` | bib | TPIPS (preprint, 2026): a free-text aspect, fine-tuned from Qwen3-VL-8B-Embedding on human odd-one-out votes. Qwen3-VL-8B-Embedding zero-shot reaches GeneCIS focus attribute 30.85 (note: 2026-12-04_conditional_similarity_update.md). | It fine-tunes the instruction embedder itself, with human labels. We distil it into a small label-free student. Cite as concurrent.
- `lim2026clay` | bib | CLAY (CVPR 2026): training-free text-conditional modulation of a frozen VL embedding space; includes a mood task. | Text condition at test; no training signal from an MLLM embedder.
- `ruthardt2026steervit` | bib | SteerViT (ECCV 2026): text steers a frozen ViT through cross-attention. | Text input at test; ours needs no condition input.
- `fioresi2026controlling` | bib | Fioresi et al. (ECCV 2026): a hypernetwork maps frozen embeddings by attribute text, trained with attribute labels. | Per-attribute transforms of a frozen embedding, close in shape to our heads. Theirs need attribute labels and text; ours need neither.
- `hsieh2025focallens` | bib | FocalLens (ICLR 2025 workshop): instruction tuning gives zero-shot conditional image representations (CLIP-L with condition tokens; an MLLM variant). | Closest on "instruction-conditioned image embedding on a CLIP-class encoder". It takes the instruction as input and trains on instruction data; ours distils fixed instruction views into heads.
- `li2025cocofacet` | bib | COCO-Facet (NeurIPS 2025): attribute-focused retrieval. A gallery-side prompt raises VLM2Vec R@1 from 44.5 to 53.4; MM-Embed does not respond (note: 2026-12-07_prompted_aspect_views.md). | Evidence that image-side instructions move MLLM embeddings. Our teacher uses exactly such image-side instructions.
- `kawarada2025training` (DIOR) | bib, plus verified now: https://wacv.thecvf.com/virtual/2026/poster/654 (WACV 2026 poster). | Training-free: prompts an LVLM "describe in one word regarding {condition}" and uses the last-token state as the conditional embedding. On WikiArt style it loses to CLIP-L (MAP@1 45.6 vs 59.3; note 2026-12-07). | Our style instruction has the same "in one word" form. DIOR stays a per-query LVLM pass; we distil into a cheap student. **The bib entry is @misc; update it to WACV 2026 (pages not seen).**

**Multiple and alternative clustering**
- `kwon2024ictc` | bib | IC\|TC (ICLR 2024): clusters images by a text criterion through a VLM and an LLM. | The criterion must be named and the clusters are per-run; we learn reusable aspect embeddings.
- `yao2024multimap` | bib | Multi-MaP (CVPR 2024): frozen CLIP plus GPT-4 proxies select the clustering matching a user keyword. | Keyword-driven; no MLLM-embedder teacher; no retrieval.
- `yao2024multisub` | verified now: https://proceedings.neurips.cc/paper_files/paper/2024/hash/96b8167534ef3cc30c230bbeb55a524d-Abstract-Conference.html | Multi-Sub (NeurIPS 2024): CLIP plus GPT-4 proxy words as subspace bases for customized multiple clustering. | One subspace per user interest, close to our per-aspect heads; theirs are text-proxy driven.
- `liu2026xcluster` | bib | X-Cluster (CVPR 2026 Findings per arXiv comment): an MLLM and an LLM discover criteria and cluster per criterion; mood is the hardest criterion on COCO-4c. | Discovers criteria but stops at clusters.
- `wang2026esmc` | note: 2026-12-07 and 2026-12-04 (AAAI 2026, OJS 40(31):26588); authors from API 2512.00725 | ESMC: hidden states of an MLLM, prompted with a criterion, plus a small clustering head, for multiple clustering. | Prompted MLLM view per criterion plus a light head, structurally close. It clusters one dataset per criterion with no frozen-encoder student and no retrieval.
- `zhong2026guideline` | note: 2026-12-07 (CVPR 2026, controller-checked); authors from API 2606.24094 | Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent: guideline-aware embeddings (GME under an instruction, per the note) plus a minimum-spanning-tree step, training-free. | Instructed embeddings feed a graph per guideline, close to point (ii). No student, no retrieval, no art.
- `xu2026cimembed` | verified now: https://arxiv.org/abs/2610.21805 (preprint, 15 Oct 2026) | CIM-Embed: an instruction-conditioned image+text embedding, clustered with HDBSCAN. Trained with supervised contrastive learning on product-metadata groupings; 449 grouping tasks. | Concurrent. The instruction is an input and supervision comes from labels; no teacher distillation, no neighbour graph.
- `cui2007orthogonalization` | verified now: Crossref 10.1109/icdm.2007.94 | Non-redundant multi-view clustering via orthogonalization (ICDM 2007). | Classic precedent for removing a known (content) structure to reveal another. Our content-similarity subtraction for emotion and style is a soft analogue.

**PercepT and what it builds on**
- `mohamed2026percept` | bib | PercepT (preprint): perceptual topics on ArtELingo from a fused CLIP-L/14 plus GoEmotions-ModernBERT embedding, IDEC-style clustering, AMI against emotion and genre; baselines include VLM2Vec v1/v2 (note: 2026-12-05_percept_neighbourhood.md). | Direct prior. One fused partition with a supervised affect model. **Caveat from the bib: do not call its CLIP encoder "frozen".**
- `xie2016dec` | verified now: https://proceedings.mlr.press/v48/xieb16.html | DEC (ICML 2016): jointly learns features and cluster assignments. | The clustering family PercepT uses.
- `guo2017idec` | verified now: Crossref 10.24963/ijcai.2017/243 | IDEC (IJCAI 2017): DEC plus local structure preservation by reconstruction. | PercepT's Stage 1 is IDEC, which PercepT does not cite (note 2026-12-05).

**Style beyond semantics**
- `somepalli2024csd` | bib | CSD (ECCV 2024): a style descriptor for style similarity and retrieval. | Frames "aspect beyond semantics"; a style-only, supervised descriptor.

### H2. Instruction-conditioned and universal multimodal embedders

**Our teacher and backbones**
- `li2026qwen3vlembedding` | bib | Qwen3-VL-Embedding 2B/8B. A custom instruction replaces the default "Represent the user's input."; MMEB-V2 73.2 (2B) per the model card (note: 2026-10-21_cvpr_literature_review.md). | Our teacher.
- `bai2025qwen3vl` | bib | The Qwen3-VL technical report. | Base of the teacher.
- `radford2021clip`, `zhai2023siglip`, `tschannen2025siglip2` | bib | CLIP, SigLIP, SigLIP 2. | Our frozen input encoders.

**MLLM-based universal embedders**
- `jiang2025vlm2vec` | note: 2026-10-21 (ICLR 2025); API 2410.05160 | VLM2Vec: task-instruction-aware embeddings ("Instruct: {task} Query: {input}") and the MMEB benchmark. | Instruction-conditioned embedder; a PercepT baseline.
- `meng2025vlm2vecv2` | API 2507.04590 (technical report, preprint) | VLM2Vec-V2 and MMEB-V2 (video, visual documents). | Benchmark family on which our teacher reports.
- `jiang2024e5v` | API 2407.12580 (preprint) | E5-V: prompt-based universal embeddings from an MLLM, trained on text pairs only. | One-word prompt, as in our style instruction.
- `zhang2025gme` | note: 2026-10-21 (CVPR 2025); API 2412.16855 (comment: CVPR 2025) | GME: Qwen2-VL universal retriever with an instruction argument. | Sibling of our teacher from the same group.
- `lin2025mmembed` | note (ICLR 2025); API 2411.02571 (comment: ICLR 2025) | MM-Embed: an MLLM universal retriever with modality-aware hard negatives. | Did not respond to image-side prompts on COCO-Facet.
- `wei2024uniir` | note: 2026-10-21 (ECCV 2024); API 2311.17136 | UniIR and the M-BEIR benchmark: instruction-guided universal retrieval. | Early instruction-guided retriever.
- `liu2025lamra` | note (CVPR 2025); API 2412.01720 | LamRA: an LMM as retriever and reranker. | Instruction through the query.
- `gunther2025jinav4` | API 2506.18902 (preprint) | jina-embeddings-v4: 3.8B, single- and multi-vector, LoRA task adapters. | Task adapters, but no free instruction.
- `zhang2024magiclens` | note (ICML 2024); API 2403.19651 (comment: ICML 2024 Oral) | MagicLens: retrieval with open-ended instructions, trained on mined image pairs. | Text-instructed relations.
- `gu2025unime` | note (ACM MM 2025); API 2504.17432 (comment: ACM MM 2025) | UniME: stage 1 distils from an LLM text embedder, stage 2 uses hard negatives. | Distils an embedder into an MLLM: the reverse direction and scale from ours.
- `lan2025llave` | note (EMNLP 2025 Findings); API 2503.04812 | LLaVE: hardness-weighted contrastive training. | MMEB state of the art for its time.
- `chen2025mme5` | API 2502.08468 (preprint per the note) | mmE5: synthetic multilingual multimodal data. | MMEB row.
- `thirukovalluru2025b3` | API 2505.11293 (preprint) | B3: batches built by clustering a teacher-derived similarity graph. | Uses a teacher similarity graph for contrastive batches, a weak relative of point (ii).
- `schneider2025abc` | API 2503.00329 (comment "TMLR 2025"; note: aggregator only) → preprint | ABC and CtrlBench: instruction control of embeddings. VLM2Vec scores 2.95 and GME-2B 20.92 on CtrlBench (note 2026-12-07). | Evidence that some embedders ignore instructions.
- `cui2025thinkthenembed` | API 2510.05014 (preprint; ICLR 2026 seen only on mlanthology) | Think Then Embed: generated reasoning conditions the embedding; claims MMEB-V2 state of the art. | Recent row.
- `zhou2026wemm` | API 2608.24053 (preprint) | WeMM-Embedding, 2B to 9B. A search snippet claims MMEB-V2 80.6 (9B) and 2B +4.7 over Qwen3-VL-Embedding-2B; the numbers are not checked. | Newest state of the art; cite without numbers.

**Text instruction embedders**
- `su2023instructor` | verified now: https://aclanthology.org/2023.findings-acl.71/ | INSTRUCTOR: one embedder conditioned on task instructions; 70 tasks. | Origin of instruction-conditioned embeddings.
- `wang2024e5mistral` | verified now: https://aclanthology.org/2024.acl-long.642/ | Fine-tunes decoder LLMs on synthetic data into embedders. | LLM-as-embedder recipe.
- `peng2024inbedder` | note: 2026-12-07 (ACL 2024); API 2402.09642 | InBedder: instruction-following text embeddings learned by answering a question about the input. Clustering one corpus under different instructions gives different clusters. | Text analogue of our per-aspect instructed views.

**Already in the bib, optional:** `zhu2026freeret`, `liu2026lens` (training-free MLLM embedders, the second steered by a task phrase).

### Works for the closest-prior question (part C)

- `elbanani2023lgssl` | verified now: https://cvpr.thecvf.com/virtual/2023/poster/23019 | Language-guided sampling: contrastive positives are image pairs whose captions are close under a pretrained language model. | **Closest to point (ii).** It uses one graph from a plain text model; ours uses one graph per aspect from instructed MLLM embeddings, cross-modal.
- `dwibedi2021nnclr` | verified now: Crossref 10.1109/iccv48922.2021.00945 | NNCLR: nearest neighbours in the learned space serve as positives. | Neighbour positives from the model's own space.
- `sobal2025xsample` | verified now: https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe37ac3ce109cd33a23a6b3ed0cfc21-Abstract-Conference.html | X-Sample contrastive loss: encodes a sample-similarity graph (from class or caption similarity) instead of one positive. | Graph-defined targets. One graph, not per aspect, not from an instructed teacher.
- `li2024tac` | note: 2026-12-05 (ICML 2024); API 2310.11989 (journal_ref ICML 2024 Oral) | TAC: WordNet-noun text counterparts plus cross-modal neighbour distillation for clustering. | Cross-modal neighbour distillation for clustering; one semantic view.
- `vangansbeke2020scan` | verified now: Crossref 10.1007/978-3-030-58607-2_16 | SCAN: neighbours mined from pretext features, then clustering with self-labelling. | Neighbour-consistency precedent.
- `adaloglou2023temi` | note: 2026-12-05 (BMVC 2023); API 2303.17896 | TEMI: self-distilled clustering heads over neighbour structure on pretrained features; improves on k-means. | Clustering heads on frozen features that beat clustering of the raw features.
- `park2019rkd` | verified now: Crossref 10.1109/cvpr.2019.00409 | Relational KD: distance and angle relations. The abstract summary says students match or beat their teachers in metric learning. | Our genre relational term and point (iii).
- `abbasikoohpayegani2020compress` | verified now: https://proceedings.neurips.cc/paper_files/paper/2020/hash/975a1c8b9aee1c48d32e13ec30be7905-Abstract.html | CompRess: the student matches the teacher's similarity distribution over anchors. Its AlexNet student beats supervised AlexNet (59.0 vs 56.5 linear). | Direct precedent for similarity-distribution distillation, our genre term.
- `tian2019crd` | API 1910.10699 (comment "ICLR 2020"; OpenReview blocked) → preprint | Contrastive Representation Distillation. | Precedent for our contrastive distillation term.
- `furlanello2018bornagain` | verified now: https://proceedings.mlr.press/v80/furlanello18a.html | Born-Again Networks: students of the same architecture outperform their teachers. | Point (iii) precedent.
- `burns2024weaktostrong` | verified now: https://proceedings.mlr.press/v235/burns24b.html | Strong models fine-tuned on a weak model's labels outperform the weak supervisor. | Point (iii). Their student is the larger model; ours is smaller but sees a different input and neighbour structure.
- `ranzinger2024amradio` | verified now: https://cvpr.thecvf.com/virtual/2024/poster/30113 | AM-RADIO: multi-teacher distillation into one student that can outperform individual teachers. | Point (i) shape (heads per teacher) and point (iii). Teachers are different models, not instruction views of one model; the backbone is trained.
- `sariyildiz2024unic` | verified now: Crossref 10.1007/978-3-031-73235-5_20 | UNIC: multi-teacher distillation with extra projectors and teacher dropping. | Per-teacher projector heads; same difference as AM-RADIO.
- `cho2026eddyvl` | verified now: https://arxiv.org/abs/2607.16316 (preprint) | Eddy-VL: prunes and distils Qwen3-VL-Embedding-2B into a 1.9B model. | **Same teacher as ours.** Their student is a general compressed copy; ours is per-aspect heads on a small frozen encoder.
- `kim2026mllmclip` | verified now: https://arxiv.org/abs/2608.25575 (preprint; comment says EMNLP 2026) | Feature-level distillation of a generative MLLM into a CLIP student (CKA loss). | MLLM to CLIP distillation, with no aspects per the abstract.
- `he2025embeddingteacher` | verified now: https://arxiv.org/abs/2510.12014 (preprint) | Distils a VLM's preference rankings into an embedding-based retrieval student. | VLM teacher to embedding student; no aspects; no teacher comparison in the abstract.
- `huang2024llm2clip` | API 2411.04997 (journal_ref "AAAI 2026", not checked on an official page) → preprint | An LLM is coupled to the CLIP vision encoder through an adaptor. | Distant relative of point (i).

**Note-only work without BibTeX:** Multiview Triplet Embedding (Amid, Ukkonen, ICML 2015, PMLR v37; the 2026-11-25 note says opened) learns one map per hidden attribute. The authors' first names were not seen, so its BibTeX needs a read of the PMLR v37 page.

## B. BibTeX (only works not already in the archived refs.bib)

```bibtex
% from note: 2026-10-21_cvpr_literature_review.md (ICLR 2025); page: https://export.arxiv.org/api/query?id_list=2410.05160
@inproceedings{jiang2025vlm2vec,
  author    = {Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  title     = {{VLM2Vec}: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

% verified: https://export.arxiv.org/api/query?id_list=2507.04590 (comment: Technical Report)
@misc{meng2025vlm2vecv2,
  author        = {Meng, Rui and Jiang, Ziyan and Liu, Ye and Su, Mingyi and Yang, Xinyi and Fu, Yuepeng and Qin, Can and Chen, Zeyuan and Xu, Ran and Xiong, Caiming and Zhou, Yingbo and Chen, Wenhu and Yavuz, Semih},
  title         = {{VLM2Vec-V2}: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
  year          = {2025},
  eprint        = {2507.04590},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2407.12580 (no venue)
@misc{jiang2024e5v,
  author        = {Jiang, Ting and Song, Minghui and Zhang, Zihan and Huang, Haizhen and Deng, Weiwei and Sun, Feng and Zhang, Qi and Wang, Deqing and Zhuang, Fuzhen},
  title         = {{E5-V}: Universal Embeddings with Multimodal Large Language Models},
  year          = {2024},
  eprint        = {2407.12580},
  archivePrefix = {arXiv}
}

% from note: 2026-10-21_cvpr_literature_review.md (CVPR 2025); page: https://export.arxiv.org/api/query?id_list=2412.16855 (comment: Accepted to CVPR 2025)
@inproceedings{zhang2025gme,
  author    = {Zhang, Xin and Zhang, Yanzhao and Xie, Wen and Li, Mingxin and Dai, Ziqi and Long, Dingkun and Xie, Pengjun and Zhang, Meishan and Li, Wenjie and Zhang, Min},
  title     = {{GME}: Improving Universal Multimodal Retrieval by Multimodal {LLMs}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}

% from note: 2026-10-21_cvpr_literature_review.md (ICLR 2025); page: https://export.arxiv.org/api/query?id_list=2411.02571 (comment: Accepted at ICLR 2025)
@inproceedings{lin2025mmembed,
  author    = {Lin, Sheng-Chieh and Lee, Chankyu and Shoeybi, Mohammad and Lin, Jimmy and Catanzaro, Bryan and Ping, Wei},
  title     = {{MM-Embed}: Universal Multimodal Retrieval with Multimodal {LLMs}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

% from note: 2026-10-21_cvpr_literature_review.md (ECCV 2024); page: https://export.arxiv.org/api/query?id_list=2311.17136
@inproceedings{wei2024uniir,
  author    = {Wei, Cong and Chen, Yang and Chen, Haonan and Hu, Hexiang and Zhang, Ge and Fu, Jie and Ritter, Alan and Chen, Wenhu},
  title     = {{UniIR}: Training and Benchmarking Universal Multimodal Information Retrievers},
  booktitle = {Computer Vision -- ECCV 2024},
  year      = {2024}
}

% from note: 2026-10-21_cvpr_literature_review.md (CVPR 2025); page: https://export.arxiv.org/api/query?id_list=2412.01720
@inproceedings{liu2025lamra,
  author    = {Liu, Yikun and Chen, Pingan and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Yao, Jiangchao and Wang, Yanfeng and Xie, Weidi},
  title     = {{LamRA}: Large Multimodal Model as Your Advanced Retrieval Assistant},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}

% verified: https://export.arxiv.org/api/query?id_list=2506.18902 (no venue)
@misc{gunther2025jinav4,
  author        = {G{\"u}nther, Michael and Sturua, Saba and Akram, Mohammad Kalim and Mohr, Isabelle and Ungureanu, Andrei and Wang, Bo and Eslami, Sedigheh and Martens, Scott and Werk, Maximilian and Wang, Nan and Xiao, Han},
  title         = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
  year          = {2025},
  eprint        = {2506.18902},
  archivePrefix = {arXiv}
}

% from note: 2026-10-21_cvpr_literature_review.md (ICML 2024); page: https://export.arxiv.org/api/query?id_list=2403.19651 (comment: ICML 2024 Oral)
@inproceedings{zhang2024magiclens,
  author    = {Zhang, Kai and Luan, Yi and Hu, Hexiang and Lee, Kenton and Qiao, Siyuan and Chen, Wenhu and Su, Yu and Chang, Ming-Wei},
  title     = {{MagicLens}: Self-Supervised Image Retrieval with Open-Ended Instructions},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}

% from note: 2026-10-21_cvpr_literature_review.md (ACM MM 2025); page: https://export.arxiv.org/api/query?id_list=2504.17432 (comment: Accepted by ACM MM2025)
@inproceedings{gu2025unime,
  author    = {Gu, Tiancheng and Yang, Kaicheng and Feng, Ziyong and Wang, Xingjun and Zhang, Yanzhao and Long, Dingkun and Chen, Yingda and Cai, Weidong and Deng, Jiankang},
  title     = {Breaking the Modality Barrier: Universal Embedding Learning with Multimodal {LLMs}},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year      = {2025}
}

% from note: 2026-10-21_cvpr_literature_review.md (EMNLP 2025 Findings); page: https://export.arxiv.org/api/query?id_list=2503.04812 (comment: Accepted by Findings of EMNLP 2025)
@inproceedings{lan2025llave,
  author    = {Lan, Zhibin and Niu, Liqiang and Meng, Fandong and Zhou, Jie and Su, Jinsong},
  title     = {{LLaVE}: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025}
}

% verified: https://export.arxiv.org/api/query?id_list=2502.08468 (no venue)
@misc{chen2025mme5,
  author        = {Chen, Haonan and Wang, Liang and Yang, Nan and Zhu, Yutao and Zhao, Ziliang and Wei, Furu and Dou, Zhicheng},
  title         = {{mmE5}: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data},
  year          = {2025},
  eprint        = {2502.08468},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2505.11293 (no venue)
@misc{thirukovalluru2025b3,
  author        = {Thirukovalluru, Raghuveer and Meng, Rui and Liu, Ye and K, Karthikeyan and Su, Mingyi and Nie, Ping and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu and Dhingra, Bhuwan},
  title         = {Breaking the Batch Barrier ({B3}) of Contrastive Learning via Smart Batch Mining},
  year          = {2025},
  eprint        = {2505.11293},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2503.00329 (arXiv comment "TMLR 2025"; official TMLR page not opened)
@misc{schneider2025abc,
  author        = {Schneider, Benjamin and Kerschbaum, Florian and Chen, Wenhu},
  title         = {{ABC}: Achieving Better Control of Multimodal Embeddings using {VLMs}},
  year          = {2025},
  eprint        = {2503.00329},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2510.05014 (no venue in feed; ICLR 2026 seen only on mlanthology)
@misc{cui2025thinkthenembed,
  author        = {Cui, Xuanming and Cheng, Jianpeng and Chen, Hong-you and Shukla, Satya Narayan and Awasthi, Abhijeet and Pan, Xichen and Ahuja, Chaitanya and Mishra, Shlok Kumar and Yang, Yonghuan and Xiao, Jun and Guo, Qi and Lim, Ser-Nam and Singh, Aashu and Fan, Xiangjun},
  title         = {Think Then Embed: Generative Context Improves Multimodal Embedding},
  year          = {2025},
  eprint        = {2510.05014},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2608.24053 (no venue)
@misc{zhou2026wemm,
  author        = {Zhou, Junjie and Mei, Ke and Li, Lei and Wang, Tianyi and Rao, Fengyun and Lyu, Jing},
  title         = {{WeMM-Embedding}: {WeChat} Multi-Modal Embedding Technical Report},
  year          = {2026},
  eprint        = {2608.24053},
  archivePrefix = {arXiv}
}

% verified: https://aclanthology.org/2023.findings-acl.71/
@inproceedings{su2023instructor,
  author    = {Su, Hongjin and Shi, Weijia and Kasai, Jungo and Wang, Yizhong and Hu, Yushi and Ostendorf, Mari and Yih, Wen-tau and Smith, Noah A. and Zettlemoyer, Luke and Yu, Tao},
  title     = {One Embedder, Any Task: Instruction-Finetuned Text Embeddings},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  pages     = {1102--1121},
  year      = {2023},
  doi       = {10.18653/v1/2023.findings-acl.71}
}

% verified: https://aclanthology.org/2024.acl-long.642/
@inproceedings{wang2024e5mistral,
  author    = {Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  title     = {Improving Text Embeddings with Large Language Models},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {11897--11916},
  year      = {2024},
  doi       = {10.18653/v1/2024.acl-long.642}
}

% from note: 2026-12-07_prompted_aspect_views.md (ACL 2024); page: https://export.arxiv.org/api/query?id_list=2402.09642
@inproceedings{peng2024inbedder,
  author    = {Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
  title     = {Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

% from note: 2026-12-04_conditional_similarity_update.md (AAAI 2026, https://ojs.aaai.org/index.php/AAAI/article/view/39867, vol 40 no 31); page: https://export.arxiv.org/api/query?id_list=2512.00725
@inproceedings{wang2026esmc,
  author    = {Wang, Xinyue and Jia, Yuheng and Liu, Hui and Hou, Junhui},
  title     = {{ESMC}: {MLLM}-Based Embedding Selection for Explainable Multiple Clustering},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {31},
  year      = {2026}
}

% from note: 2026-12-07_prompted_aspect_views.md (CVPR 2026; https://cvpr.thecvf.com/virtual/2026/poster/37785); page: https://export.arxiv.org/api/query?id_list=2606.24094 (comment: CVPR 2026)
@inproceedings{zhong2026guideline,
  author    = {Zhong, Wenliang and Barton, Rob and Goncalves, Lucas and Kumar, Kushal and Jiang, Feng and Ma, Hehuan and Guo, Yuzhi and Bansal, Vidit and Bouyarmane, Karim and Huang, Junzhou},
  title     = {Universal Guideline-Driven Image Clustering via a Hybrid {LLM} Agent},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}

% verified: https://arxiv.org/abs/2610.21805 (no venue; v1 15 Oct 2026)
@misc{xu2026cimembed,
  author        = {Xu, Yinghua and Xie, Pengyang and Zhou, Yanxiang and Tong, Chuanzhi and Xiao, Xiao and Feng, Chaoqun and Wang, Xiaoqing},
  title         = {{CIM-Embed}: Learning Instruction-Conditioned Multimodal Embeddings for Clustering},
  year          = {2026},
  eprint        = {2610.21805},
  archivePrefix = {arXiv}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2024/hash/96b8167534ef3cc30c230bbeb55a524d-Abstract-Conference.html
@inproceedings{yao2024multisub,
  author    = {Yao, Jiawei and Qian, Qi and Hu, Juhua},
  title     = {Customized Multiple Clustering via Multi-Modal Subspace Proxy Learning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024}
}

% verified: https://api.crossref.org/works?query.bibliographic=Non-redundant+Multi-view+Clustering+via+Orthogonalization (Crossref record 10.1109/icdm.2007.94)
@inproceedings{cui2007orthogonalization,
  author    = {Cui, Ying and Fern, Xiaoli Z. and Dy, Jennifer G.},
  title     = {Non-redundant Multi-view Clustering via Orthogonalization},
  booktitle = {Seventh IEEE International Conference on Data Mining (ICDM 2007)},
  pages     = {133--142},
  year      = {2007},
  doi       = {10.1109/ICDM.2007.94}
}

% verified: https://proceedings.mlr.press/v48/xieb16.html
@inproceedings{xie2016dec,
  author    = {Xie, Junyuan and Girshick, Ross and Farhadi, Ali},
  title     = {Unsupervised Deep Embedding for Clustering Analysis},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {48},
  pages     = {478--487},
  year      = {2016}
}

% verified: https://api.crossref.org/works/10.24963/ijcai.2017/243 (Crossref record)
@inproceedings{guo2017idec,
  author    = {Guo, Xifeng and Gao, Long and Liu, Xinwang and Yin, Jianping},
  title     = {Improved Deep Embedded Clustering with Local Structure Preservation},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {1753--1759},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/243}
}

% from note: 2026-12-05_percept_neighbourhood.md (ICML 2024); page: https://export.arxiv.org/api/query?id_list=2310.11989 (journal_ref: ICML 2024 (Oral))
@inproceedings{li2024tac,
  author    = {Li, Yunfan and Hu, Peng and Peng, Dezhong and Lv, Jiancheng and Fan, Jianping and Peng, Xi},
  title     = {Image Clustering with External Guidance},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}

% from note: 2026-12-05_percept_neighbourhood.md (BMVC 2023); page: https://export.arxiv.org/api/query?id_list=2303.17896 (comment: Accepted at BMVC2023)
@inproceedings{adaloglou2023temi,
  author    = {Adaloglou, Nikolas and Michels, Felix and Kalisch, Hamza and Kollmann, Markus},
  title     = {Exploring the Limits of Deep Image Clustering using Pretrained Models},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2023}
}

% verified: https://api.crossref.org/works?query.bibliographic=SCAN+Learning+to+Classify+Images+without+Labels+Van+Gansbeke (Crossref record 10.1007/978-3-030-58607-2_16)
@inproceedings{vangansbeke2020scan,
  author    = {Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  title     = {{SCAN}: Learning to Classify Images Without Labels},
  booktitle = {Computer Vision -- ECCV 2020},
  series    = {Lecture Notes in Computer Science},
  pages     = {268--285},
  publisher = {Springer},
  year      = {2020},
  doi       = {10.1007/978-3-030-58607-2_16}
}

% verified: https://cvpr.thecvf.com/virtual/2023/poster/23019 ; https://export.arxiv.org/api/query?id_list=2302.12248
@inproceedings{elbanani2023lgssl,
  author    = {El Banani, Mohamed and Desai, Karan and Johnson, Justin},
  title     = {Learning Visual Representations via Language-Guided Sampling},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

% verified: https://api.crossref.org/works?query.bibliographic=With+a+Little+Help+from+My+Friends+Nearest-Neighbor+Contrastive+Learning+Dwibedi (Crossref record 10.1109/iccv48922.2021.00945)
@inproceedings{dwibedi2021nnclr,
  author    = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
  title     = {With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {9568--9577},
  year      = {2021},
  doi       = {10.1109/ICCV48922.2021.00945}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe37ac3ce109cd33a23a6b3ed0cfc21-Abstract-Conference.html
@inproceedings{sobal2025xsample,
  author    = {Sobal, Vlad and Ibrahim, Mark and Balestriero, Randall and Cabannes, Vivien and Bouchacourt, Diane and Astolfi, Pietro and Cho, Kyunghyun and LeCun, Yann},
  title     = {{$\mathbb{X}$}-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

% verified: https://api.crossref.org/works?query.bibliographic=Relational+Knowledge+Distillation+Park+Kim+Lu+Cho (Crossref record 10.1109/cvpr.2019.00409)
@inproceedings{park2019rkd,
  author    = {Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  title     = {Relational Knowledge Distillation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3962--3971},
  year      = {2019},
  doi       = {10.1109/CVPR.2019.00409}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2020/hash/975a1c8b9aee1c48d32e13ec30be7905-Abstract.html
@inproceedings{abbasikoohpayegani2020compress,
  author    = {Abbasi Koohpayegani, Soroush and Tejankar, Ajinkya and Pirsiavash, Hamed},
  title     = {{CompRess}: Self-Supervised Learning by Compressing Representations},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  year      = {2020}
}

% verified: https://export.arxiv.org/api/query?id_list=1910.10699 (comment "ICLR 2020"; the OpenReview page was blocked, so the venue is not checked)
@misc{tian2019crd,
  author        = {Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  title         = {Contrastive Representation Distillation},
  year          = {2019},
  eprint        = {1910.10699},
  archivePrefix = {arXiv}
}

% verified: https://proceedings.mlr.press/v80/furlanello18a.html
@inproceedings{furlanello2018bornagain,
  author    = {Furlanello, Tommaso and Lipton, Zachary and Tschannen, Michael and Itti, Laurent and Anandkumar, Anima},
  title     = {Born Again Neural Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {80},
  pages     = {1607--1616},
  year      = {2018}
}

% verified: https://proceedings.mlr.press/v235/burns24b.html
@inproceedings{burns2024weaktostrong,
  author    = {Burns, Collin and Izmailov, Pavel and Kirchner, Jan Hendrik and Baker, Bowen and Gao, Leo and Aschenbrenner, Leopold and Chen, Yining and Ecoffet, Adrien and Joglekar, Manas and Leike, Jan and Sutskever, Ilya and Wu, Jeffrey},
  title     = {Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {4971--5012},
  year      = {2024}
}

% verified: https://cvpr.thecvf.com/virtual/2024/poster/30113 (CVPR 2024) ; pages from the arXiv journal_ref of 2312.06709
@inproceedings{ranzinger2024amradio,
  author    = {Ranzinger, Mike and Heinrich, Greg and Kautz, Jan and Molchanov, Pavlo},
  title     = {{AM-RADIO}: Agglomerative Vision Foundation Model -- Reduce All Domains Into One},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {12490--12500},
  year      = {2024}
}

% verified: https://api.crossref.org/works?query.bibliographic=UNIC+Universal+Classification+Models+via+Multi-teacher+Distillation+Sariyildiz (Crossref record 10.1007/978-3-031-73235-5_20)
@inproceedings{sariyildiz2024unic,
  author    = {Sar{\i}y{\i}ld{\i}z, Mert B{\"u}lent and Weinzaepfel, Philippe and Lucas, Thomas and Larlus, Diane and Kalantidis, Yannis},
  title     = {{UNIC}: Universal Classification Models via Multi-teacher Distillation},
  booktitle = {Computer Vision -- ECCV 2024},
  series    = {Lecture Notes in Computer Science},
  pages     = {353--371},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-73235-5_20}
}

% verified: https://arxiv.org/abs/2607.16316 (no venue)
@misc{cho2026eddyvl,
  author        = {Cho, HanYeong and Kim, Changwoo and Chu, Taeuk and Park, Jimin},
  title         = {{Eddy-VL} 1.9B: Structural Pruning and Layered Distillation for Edge-Deployable Multimodal Embedding},
  year          = {2026},
  eprint        = {2607.16316},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2608.25575 (arXiv comment: EMNLP 2026 Main Conference; official page not opened)
@misc{kim2026mllmclip,
  author        = {Kim, Jongsuk and Wu, Qiyu and Mao, Zhuoyuan and Wakaki, Hiromi and Kim, Junmo and Mitsufuji, Yuki},
  title         = {{MLLMCLIP}: Feature-Level Distillation of {MLLM} for Robust Vision-Language Representations},
  year          = {2026},
  eprint        = {2608.25575},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2510.12014 (no venue)
@misc{he2025embeddingteacher,
  author        = {He, Eric and Gupta, Akash and Liusie, Adian and Raina, Vatsal and Molenda, Piotr and Chabra, Shirom and Raina, Vyas},
  title         = {Embedding the Teacher: Distilling {vLLM} Preferences for Scalable Image Retrieval},
  year          = {2025},
  eprint        = {2510.12014},
  archivePrefix = {arXiv}
}

% verified: https://export.arxiv.org/api/query?id_list=2411.04997 (v6 title; journal_ref "AAAI 2026" not checked on an official page). The arXiv feed lists "Chunyu Wang" twice; kept once here. Check the author list before use.
@misc{huang2024llm2clip,
  author        = {Huang, Weiquan and Wu, Aoqi and Yang, Yifan and Luo, Xufang and Yang, Yuqing and Naseem, Usman and Wang, Chunyu and Dai, Qi and Dai, Xiyang and Chen, Dongdong and Luo, Chong and Qiu, Lili and Hu, Liang},
  title         = {{LLM2CLIP}: Powerful Language Model Unlocks Richer Cross-Modality Representation},
  year          = {2024},
  eprint        = {2411.04997},
  archivePrefix = {arXiv}
}
```

**Change to an existing bib entry:** `kawarada2025training` (DIOR) is @misc in refs.bib. https://wacv.thecvf.com/virtual/2026/poster/654 shows WACV 2026, so make it `@inproceedings`, booktitle "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)", year 2026, no pages.

## C. Closest prior work and novelty wording

**What I searched (24 web searches).** The queries, by theme:
- distil an instruction-following multimodal embedder into a light multi-head student on a frozen encoder;
- aspect-specific embeddings distilled from MLLM prompts (emotion, style), and students that surpass their teacher;
- multi-head students with one head per instruction or prompt;
- Qwen3-VL-Embedding as a teacher;
- condition-specific LVLM embeddings distilled into a light encoder;
- frozen-CLIP attribute heads trained from VLM pseudo-similarity;
- kNN-graph or teacher-neighbour positives in contrastive learning, and positives from prompted embeddings;
- MLLM pseudo-positives for emotion retrieval (EmoSet);
- art emotion, style and genre embeddings under MLLM supervision;
- multi-teacher distillation with per-teacher heads;
- student-beats-teacher in embedding distillation.

I also re-read the notes of 2026-12-07, 2026-12-04, 2026-12-05 and 2026-11-28. **None of these found a work matching any of points (i) to (iii) in full. Absence in about 15 targeted searches is not proof of novelty.**

**(i) Distilling one instruction embedder's per-aspect views into a light multi-head student on a frozen encoder.**
- FocalLens: a conditional CLIP-class embedding learned by instruction tuning. The instruction is an input at test time, and training uses instruction data.
- DIOR: per-condition LVLM embeddings, training-free, with an LVLM pass per query. It loses to CLIP on WikiArt style.
- UNIC and AM-RADIO: a multi-head student with one projector per teacher. Their teachers are different models, the student backbone is trained, and the targets are general features.
- Eddy-VL: distils our exact teacher, Qwen3-VL-Embedding-2B, into one smaller general embedder.
- MLLMCLIP: MLLM-to-CLIP feature distillation with no aspects (per its abstract).
- TPIPS: fine-tunes the instruction embedder itself on human votes.
- Fioresi et al.: per-attribute transforms of frozen embeddings, trained with attribute labels.
- **Our difference:** one teacher queried under several aspect instructions; each view becomes a linear head on a shared trunk over a frozen image encoder; no instruction at test; no aspect labels.
- **Wording:** "To our knowledge, prior work either queries an instruction-following embedder at test time for each condition [DIOR, COCO-Facet], fine-tunes it [TPIPS], compresses it into one general model [Eddy-VL], or distils several different teachers into per-teacher heads [AM-RADIO, UNIC]. We instead distil the aspect-instructed views of a single multimodal embedder into a lightweight multi-head student on a frozen image encoder, which needs no instruction at inference."

**(ii) Per-aspect neighbour graphs from instructed embeddings, used as contrastive positives.**
- Language-guided sampling: positives are caption neighbours under a pretrained language model. This is the closest; it uses one graph from an uninstructed text model.
- X-Sample: one similarity graph from class or caption similarity, used as soft contrastive targets.
- NNCLR, SCAN, TEMI: neighbours from the model's own or pretrained features.
- TAC: cross-modal neighbour distillation with WordNet text counterparts, for clustering.
- Guideline-driven clustering (CVPR 2026): instructed embeddings and a spanning tree per guideline, training-free, clustering only.
- B3: a teacher similarity graph used for batch mining, not positives.
- PercepT: one fused partition.
- The content subtraction has a classic analogue in non-redundant clustering (Cui et al. 2007).
- **Our difference:** one reciprocal-kNN graph per aspect, built from instruction-conditioned MLLM embeddings, with content similarity subtracted for emotion and style, used as the positive set of a cross-modal (image and caption) contrastive loss.
- **Wording:** "Neighbour graphs as contrastive positives are well established [NNCLR, language-guided sampling, X-Sample, TAC]. To our knowledge, we are the first to build one such graph per perceptual aspect from instruction-conditioned embeddings, with the content structure removed, and to use each as the positive set of a separate image-caption contrastive objective."
- Do not claim the graph construction or neighbour positives themselves.

**(iii) A student that surpasses its embedding teacher on the teacher's own aspect.**
- Born-Again Networks: same-architecture students beat their teachers.
- RKD: its abstract reports students outperforming teachers in metric learning.
- CompRess: its student beats a supervised model.
- AM-RADIO: the merged student can outperform individual teachers.
- Weak-to-strong: strong students beat weak supervisors.
- TEMI: clustering heads improve on k-means over the same features.
- No case found where a small student on a different frozen encoder's features beats an MLLM instruction embedder on the aspect named in the teacher's instruction (clustering AMI against human labels).
- **Wording:** "Students that exceed their teachers are known [Born-Again, RKD, AM-RADIO, weak-to-strong]. In our setting, to our knowledge, this is the first report of a lightweight student on frozen CLIP-class features that exceeds the instruction-following embedder it is distilled from, on clustering by the very aspect named in the teacher's instruction, while matching it in retrieval."
- Present this as a finding, not a mechanism. Say that it holds for clustering (AMI), with retrieval level, so the claim is not overstated.

**Concurrent works to name:**
- CIM-Embed (15 Oct 2026, preprint): instruction-conditioned embeddings for clustering, supervised by labels.
- Eddy-VL (Jul 2026, preprint).
- TPIPS (preprint).

## D. Open

- **Excluded because not opened** (seen only in search results or snippets; do not cite):
  - distillation and multi-teacher: SEED (ICLR 2021), Theia, "Learning Task-Agnostic Representations through Multi-Teacher Distillation" (2510.18680), CLIP-RD, AMMKD, VL2Lite;
  - emotion: EmoVLM-KD, Niu and Provost (2505.18040);
  - embedders: MetaEmbed (the note has its ICLR 2026 row but no authors), UniME-V2 (AAAI 2026), mEOL;
  - other: Gondek and Hofmann (ICDM 2004), US patent 12670695, MMEB-V3, ArtECulture.
- **Venues not checked on an official page, so cited as preprints:** CRD (OpenReview blocked), ABC (TMLR), LLM2CLIP (AAAI 2026), Think Then Embed (ICLR 2026), MLLMCLIP (EMNLP 2026), X-Cluster (CVPR 2026 Findings, already @misc in the bib). One fetch each of the OpenReview or TMLR page would settle them once OpenReview stops blocking.
- **Venues that come from notes, not from an official page opened now:** VLM2Vec, MM-Embed, UniIR, LamRA, GME, MagicLens, UniME, LLaVE, InBedder, ESMC, guideline clustering, TAC, TEMI. Pages are omitted for all of them; the proceedings pages would add pages.
- **Multiview Triplet Embedding** (Amid, Ukkonen, ICML 2015) needs its PMLR v37 page for full author names before BibTeX.
- **Still unknown:** whether Qwen3-VL-Embedding was trained with image-side instructions, which matters for how we describe the teacher. Read the appendix of arXiv 2601.04720.
- **WeMM-Embedding's numbers** (snippet only): read arXiv 2608.24053 before citing any figure.
- **A final arXiv sweep the week before submission** for "instruction embedding distillation multi-head" and "per-aspect neighbour graph" (CIM-Embed appeared only on 15 Oct 2026).

---

## Sources, H3 and H4 (distillation; neighbour graphs) and the method citations

Deep-research gap-fill (Sonnet high), 2026-11-02, for the first draft. Saved verbatim from the agent's report by the paper chat.

## Checked source list for Related Work (H3, H4) and Method citations

Depth: exhaustive, capped at 50 searches and 40 pages.

**Budget:** I used 23 searches. I made about 48 fetch calls, of which about 38 returned a page. CVF (the open-access repository) blocked three fetches with a 403, OpenReview blocked one, and Crossref returned 429 or 404 on four. If failed fetches count against the cap, I went over it, so a few entries below are "listing" status (defined next).

**Status words:**
- `verified now: <URL>` means I opened that page.
- `note + listing` means a literature note states the work and does not flag it. The author list, pages and venue come from an official-site listing that appeared in a search result, and I did not open it. Open the page before final submission if you want it fully checked.
- `excluded` means it will not be cited as it stands.

Venue rule: a venue counts only when an opened page or an unflagged note shows it. Otherwise the BibTeX entry is an arXiv `@misc`.

## A. Sources per heading

### H3. Knowledge distillation of embeddings and representations

`hinton2015distill` | verified now: https://arxiv.org/abs/1503.02531 | Transfers an ensemble's predictive power into one smaller model. Also proposes specialist models. The arXiv comment says "NIPS 2014 Deep Learning Workshop". | Origin of distillation. Ours distils embeddings and similarity structure, not class posteriors.

`romero2015fitnets` | verified now: https://arxiv.org/abs/1412.6550 | A thinner, deeper student learns from the teacher's intermediate representations (hints) and outputs, with extra mapping parameters. The arXiv page has no venue; ICLR 2015 reached me only through a summary of DBLP, so I use `@misc`. | Feature-level distillation. Our direct term is a contrastive distillation of the output embedding, with no hint layers.

`tian2020crd` | verified now: https://ar5iv.labs.arxiv.org/html/1910.10699 (the page shows no venue; use `@misc`) | Contrastive Representation Distillation (CRD). Positives are the teacher and student representations of the same input. Negatives pair a teacher representation with the student representation of a different input. The critic is h(T,S) = exp(g^T(T)'g^S(S)/τ) / [exp(g^T(T)'g^S(S)/τ) + N/M]. It is trained by maximising E_{q(T,S|C=1)}[log h] + N·E_{q(T,S|C=0)}[log(1−h)], with a memory buffer and N from 16 to 16,384. | Closest to our contrastive distillation term (teacher image embedding, student image embedding). CRD is NCE-style over a memory buffer. Ours uses in-batch InfoNCE. The CRD name clashes with CLIP-KD's "CRD".

`passalis2018pkt` | verified now (text): https://ar5iv.labs.arxiv.org/html/1803.10837 ; venue from the Springer listing https://link.springer.com/chapter/10.1007/978-3-030-01252-6_17 (seen in search, not opened) | Probabilistic Knowledge Transfer (PKT). Kernel K(a,b) = ½(aᵀb/(‖a‖‖b‖)+1). p_{i|j} = K(x_i,x_j)/Σ_{k≠j}K(x_k,x_j), with x the teacher features, and q_{i|j} the same with y, the student features. The loss is the KL divergence, L = Σ_i Σ_{j≠i} p_{j|i} log(p_{j|i}/q_{j|i}). It uses mini-batches of 64 to 128. It also reports cross-modal transfer from text attributes into a visual student on SUN Attribute. | Closest in-batch KL form to our relational term, and cross-modal. Differences: PKT normalises kernel values by their sum, with no softmax and no temperature. Ours uses a softmax-style distribution. This resolves the "from memory only" flag in the 12-26 note.

`chen2018darkrank` | verified now (abstract): https://arxiv.org/abs/1707.01220 ; venue from https://ojs.aaai.org/index.php/AAAI/article/view/11783 (seen in search, not opened; DOI 10.1609/aaai.v32i1.11783) | Cross-sample similarities are knowledge to transfer to a smaller student. It applies learning-to-rank in deep metric learning and is tested on re-identification, retrieval and clustering. The abstract names no specific ranking loss. | A ranking-based relational precedent. I will not state its loss form because I did not see it.

`park2019rkd` | verified now (text): https://ar5iv.labs.arxiv.org/html/1904.05068 ; note: 2026-12-26_relational_genre_term.md (CVF page) | Distance-wise loss: ψ_D = ‖t_i−t_j‖₂/μ, with μ the mean pairwise distance in the mini-batch. Angle-wise loss: ψ_A = cos∠(t_i t_j t_k) = ⟨e^ij, e^kj⟩. Both go through a Huber loss l_δ (½(x−y)² if |x−y| ≤ 1, else |x−y|−½). Total: L_task + λ·L_KD. Metric learning uses λ_D = 1 and λ_A = 2 with no task loss. | Relational family umbrella. RKD compares distances and angles with Huber loss. Ours compares similarity distributions with KL.

`tung2019spkd` | note + listing: 2026-12-26_relational_genre_term.md (read through ar5iv); CVF listing https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html (ICCV 2019, pp. 1365–1374) | A b×b in-batch similarity matrix of layer activations, rows L2-normalised. Loss: squared Frobenius distance divided by b². | In-batch similarity, as ours. SP-KD matches normalised similarity values with a squared error. Ours matches softmax rows with KL.

`koohpayegani2020compress` | note + listing: 2026-12-26_relational_genre_term.md ; https://arxiv.org/abs/2010.14713 | Per query, KL(p_teacher ‖ p_student). Each p is a softmax over cosine similarities to anchors, divided by a temperature. The anchors are a FIFO queue of 128,000 teacher features. The temperature is 0.04, or 0.007 for SwAV. Venue: NeurIPS 2020, vol. 33, pp. 12980–12992. | Closest softmax-KL form to ours. Difference: CompRess uses a queue of anchors and one temperature on both sides. Ours is in-batch with no anchors.

`fang2021seed` | note + listing: 2026-12-26_relational_genre_term.md ; https://arxiv.org/abs/2101.04731v1 (ICLR 2021 per the arXiv listing) | Cross-entropy between teacher and student similarity distributions over a queue of 65,536 teacher features, with the teacher feature appended. Teacher temperature 0.01, student 0.2. Equals KL up to a constant. | Same family as CompRess. Differences: queue anchors and unequal temperatures. Ours is in-batch.

`yang2024clipkd` | note + listing: 2026-12-26_relational_genre_term.md ; https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html (pp. 15952–15962) | Their "contrastive relational distillation" aligns teacher and student in-batch image-to-text and text-to-image softmax distributions with KL, teacher first. The note reports feature regression beating the relational term alone (34.2 against 31.9 ImageNet zero-shot). | Closest in-batch softmax-KL precedent, cross-modal. Ours acts within the image tower, per aspect, with a teacher that sees an instruction.

`wu2023tinyclip` | verified now: https://arxiv.org/abs/2309.12314 (comment "Accepted by ICCV 2023") | TinyCLIP: affinity mimicking plus weight inheritance. Halves CLIP ViT-B/32 while keeping comparable zero-shot performance. | Cross-modal affinity distillation. I saw only the abstract, so I do not state the loss form.

`vasu2024mobileclip` | verified now: https://arxiv.org/abs/2311.17049 ; https://api.crossref.org/works/10.1109/cvpr52733.2024.01511 (CVPR 2024, pp. 15963–15974) | Multi-modal reinforced training. It transfers knowledge from a captioner and an ensemble of CLIP encoders, stored in a reinforced dataset. | A caption-aware teacher signal. In ours, captions enter only as a training tower.

`ranzinger2024amradio` | note + listing: 2026-12-20_ways_to_improve.md ; https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html (CVPR 2024, pp. 12490–12500) | Multi-teacher distillation of CLIP, DINOv2 and SAM into one student. The note says it beats its teachers on 6 of 9 tasks. | A student can pass a teacher when the teachers are complementary. In ours there is one teacher viewed under three instructions.

`sariyildiz2024unic` | note + listing: 2026-12-20_ways_to_improve.md ; https://link.springer.com/chapter/10.1007/978-3-031-73235-5_20 (ECCV 2024, pp. 353–371) ; arXiv 2408.05088 | Multi-teacher distillation. It adds a ladder of expendable projectors and teacher dropping, and the student keeps or improves on the best teacher per task. The note records that standardising the teacher targets helps (+2.7 top-1). | Same remark as AM-RADIO. UNIC's standardised targets could help our direct term.

`ranzinger2024phis` | note: 2026-12-20_ways_to_improve.md (preprint) ; https://arxiv.org/abs/2410.01680 | Studies how teacher activation statistics interact with the loss in label-free multi-teacher distillation. Proposes a PCA-Hadamard normalisation. | Preprint. Target-normalisation precedent only.

`shang2025theia` | verified now: https://proceedings.mlr.press/v270/shang25a.html | Theia distils several vision foundation models for robot learning (CoRL 2024, PMLR 270, pp. 724–748). It reports outperforming its teachers. | Multi-teacher student passing its teachers. This resolves the "secondary only" flag.

`zhang2024jasper` | note: 2026-12-30_signals_beyond_teacher.md (preprint, read in the note) ; https://arxiv.org/abs/2412.19048 (authors seen in search) | Distils Stella and NV-Embed-v2 into the 2B Jasper. Per the note, the student passes the weaker teacher on the MTEB average by 0.35. | Preprint. A text-embedding student that slightly passes one of its teachers.

`akram2026jina` | note: 2026-12-20_ways_to_improve.md (preprint) ; https://arxiv.org/abs/2602.15547 | jina-embeddings-v5-text, task-targeted embedding distillation. | Preprint. Embedding distillation toward task-specific targets.

`furlanello2018born` | note + listing: 2026-12-30_signals_beyond_teacher.md (ar5iv) ; https://proceedings.mlr.press/v80/furlanello18a.html (ICML 2018, PMLR 80:1607–1616) | The student has the same architecture as the teacher and uses true labels. The note records CIFAR-100 gains of 0.80 to 1.30 error points. | A student can pass its teacher, but labels are in the loss there. Ours has none.

`xie2020noisystudent` | verified now: https://api.crossref.org/works/10.1109/cvpr42600.2020.01070 (CVPR 2020, pp. 10684–10695) | Self-training with a noised student of equal or larger size. This resolves the note's "snippet" flag. I did not open the abstract, so I give no result numbers. | Self-training can pass the teacher, but it needs strong noise and pseudo-labels. Ours uses neither.

`vapnik2009lupi` | verified now: https://api.crossref.org/works/10.1016/j.neunet.2009.06.042 (Neural Networks, 2009, pp. 544–557) | Learning using privileged information (LUPI). The title is "A new learning paradigm: Learning using privileged information". | Our captions are privileged information, available in training only.

`lopezpaz2016unifying` | verified now: https://arxiv.org/abs/1511.03643 | Generalized distillation. Unifies distillation and privileged information (ICLR 2016, per the arXiv page). | The closest frame for "teacher sees more than the student". Our teacher sees the instruction, and our caption tower sees captions.

### H4. Neighbourhood graphs and graph-supervised contrastive learning

`oord2018cpc` | verified now: https://arxiv.org/abs/1807.03748 (no venue on the page: preprint) | Contrastive Predictive Coding. A contrastive objective with negative sampling, used for speech, images, text and reinforcement learning. | Origin of InfoNCE. I did not confirm the "InfoNCE" name on the page.

`chen2020simclr` | verified now: https://proceedings.mlr.press/v119/chen20j.html | SimCLR, ICML 2020 (PMLR 119, pp. 1597–1607). In-batch contrastive learning with a nonlinear projection head, no memory bank. | Our in-batch symmetric loss is its form, with graph neighbours as positives instead of two augmented views.

`khosla2020supcon` | verified now: https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html | SupCon. Class labels define the positives, pulling same-class embeddings together. | Multi-positive contrastive loss. Our positives come from a graph, not labels.

`dwibedi2021nnclr` | verified now: https://arxiv.org/abs/2104.14548 (comment "Accepted at ICCV 2021") | NNCLR. Nearest neighbours of an embedding in latent space serve as positives. | Neighbour positives from the model's own space, image-only. Ours uses a teacher's graph, mutual, cross-modal.

`koohpayegani2021msf` | verified now: https://arxiv.org/abs/2105.07269 (no venue on the page: preprint) | Mean Shift for Self-Supervised Learning. Moves each embedding toward the mean of its nearest neighbours. | Neighbours as a target, not as contrastive positives.

`li2021comatch` | verified now: https://arxiv.org/abs/2011.11183 (no venue on the page: preprint) | CoMatch. Semi-supervised learning with contrastive graph regularisation. Class probabilities and embeddings shape each other. | Graph regularisation of embeddings. Needs labelled data.

`vangansbeke2020scan` | verified now: https://arxiv.org/abs/2005.12320 (comment "Accepted at ECCV 2020") | SCAN. A two-step method that separates feature learning from clustering, using nearest neighbours. | Neighbour-based clustering on features. Ours has no clustering loss.

`xie2016dec` | verified now: https://proceedings.mlr.press/v48/xieb16.html | DEC, ICML 2016 (PMLR 48, pp. 478–487). Jointly learns representations and cluster assignments. | PercepT's clustering base. We replace it with graph supervision.

`guo2017idec` | verified now: https://www.ijcai.org/proceedings/2017/243 | IDEC, IJCAI 2017, pp. 1753–1759. Adds a reconstruction loss to DEC to preserve local structure. | Same remark as DEC. The 12-05 note says PercepT's method is IDEC.

`zhong2017kreciprocal` | verified now: https://arxiv.org/abs/1701.08398 (comment "To appear in CVPR 2017") | k-reciprocal encoding of mutual nearest neighbours, blended into a Jaccard distance for re-ranking. | The mutual-kNN idea, used for re-ranking. Ours uses reciprocal kNN as a training graph.

`huh2024platonic` | verified now: https://proceedings.mlr.press/v235/huh24a.html (ICML 2024, PMLR 235, pp. 20617–20642) ; https://ar5iv.labs.arxiv.org/html/2405.07987 | The mutual-kNN metric is "the mean intersection of the k-nearest neighbor sets induced by two kernels", normalised by k. The exact formula is in Appendix A, which I did not see. It is applied to paired image and caption data. | A mutual-kNN metric between two representations, to measure alignment. Ours builds a reciprocal graph inside one embedding to supervise training.

`mcinnes2018umap` | verified now: https://arxiv.org/abs/1802.03426 (no journal reference: preprint) | UMAP builds a neighbour graph, then embeds it. | The fuzzy neighbour graph precedent. I did not verify the model-intersection feature (only the 12-05 note mentions it).

`hao2021wnn` | verified now: https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/ (Cell 184(13):3573–3587.e29, 2021) | Weighted nearest neighbour analysis. It learns the relative weight of each data type per cell and builds a graph from the weighted combination. | A multi-modality neighbour graph. Ours builds one graph per aspect from one instructed model.

`li2024tac` | note + listing: 2026-12-05_percept_neighbourhood.md ; https://proceedings.mlr.press/v235/li24aa.html (ICML 2024 oral, PMLR 235, pp. 27890–27902) | Text-Aided Clustering. WordNet nouns are retrieved, then text and image distil cross-modal neighbourhood information from each other. The note reports ImageNet-1K NMI 79.9 against 72.3 for k-means on CLIP. | The nearest published cross-modal neighbour distillation. TAC needs a WordNet text side and a cluster head. Ours has captions only in training and graph positives.

`su2019djsrh` | note + listing: 2026-12-05_percept_neighbourhood.md ; CVF listing https://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Deep_Joint-Semantics_Reconstructing_Hashing_for_Large-Scale_Unsupervised_Cross-Modal_Retrieval_ICCV_2019_paper.pdf (my fetch got a 403; DOI 10.1109/ICCV.2019.00312) | Deep Joint-Semantics Reconstructing Hashing. A joint image-text affinity matrix supervises binary codes for unsupervised cross-modal retrieval. | A neighbourhood affinity across modalities as supervision, for hashing. Ours uses contrastive learning on a teacher graph.

`li2022declip` | verified now (text): https://ar5iv.labs.arxiv.org/html/2110.05208 ; venue ICLR 2022 from https://iclr.cc/virtual/2022/poster/6406 (seen in search, not opened) | Nearest-neighbour supervision (NNS). For a text feature, the nearest text feature in a 64K FIFO queue (cosine) serves as the positive. The pairs (z^I, z^{T'}) and (z̃^I, z^{T'}) use the InfoNCE form. L = (1−α−β−γ)L_CLIP + α(L_ISS+L_TSS) + βL_MVS + γL_NNS, with α = β = γ = 0.2. NNS adds about 1.8% ImageNet zero-shot (25.4 to 27.2). | The clearest image-text precedent for neighbour positives. Neighbours come from the model's own text space, one direction, found in a queue. Ours uses a fixed teacher graph, with four directions including caption-caption.

`yang2022unicl` | verified now: https://arxiv.org/abs/2204.03610 (comment "CVPR 2022") | UniCL. Image-label and image-text data in one shared space with a single contrastive objective. | Multi-positive cross-modal contrastive loss with label-defined positives. Ours has graph-defined positives, no labels.

### Method citations

`loshchilov2019adamw` | verified now: https://iclr.cc/virtual/2019/poster/935 (ICLR 2019 poster) ; arXiv 1711.05101 | Decoupled Weight Decay Regularization. Decoupling weight decay from the loss-gradient update restores weight decay for Adam. | Our optimiser.

`hendrycks2016gelu` | verified now: https://arxiv.org/abs/1606.08415 (no venue: preprint; v5 revised June 2023) | GELU. Scales each input by the probability that a standard Gaussian is below it. | Our trunk activation.

`srivastava2014dropout` | verified now: https://jmlr.org/papers/v15/srivastava14a.html (JMLR 15(56):1929–1958, 2014) | Dropout. Randomly deactivating units during training curbs overfitting. | Our regulariser.

`vinh2010ami` | verified now: https://jmlr.org/papers/v11/vinh10a.html (JMLR 11:2837–2854, 2010) | Compares clusterings with information-theoretic measures and argues for correcting them for chance. | Our clustering metric (AMI). The page text I saw does not use the words "adjusted mutual information".

`chum2007aqe` | verified now: https://api.crossref.org/works/10.1109/ICCV.2007.4408891 (ICCV 2007, pp. 1–8) | "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval". The Crossref record has no abstract. | Source of average query expansion. I did not see the text, so the average-over-top-results description comes from Radenović's page (below). The "average query expansion" name is from Radenović's paper.

`radenovic2018finetune` | verified now: https://ar5iv.labs.arxiv.org/html/1711.02512 ; https://arxiv.org/abs/1711.02512 (comment "TPAMI 2018") | Introduces a "novel α-weighted query expansion": the i-th ranked image gets weight (f̄(q)ᵀf̄(i))^α, and α = 0 reduces to average query expansion (AQE). | Source of αQE, and it describes AQE as average pooling of the top-ranked descriptors.

`jegou2012whitening` | verified now: https://api.crossref.org/works/10.1007/978-3-642-33709-3_55 (ECCV 2012, LNCS, pp. 774–787) | "Negative Evidences and Co-occurences in Image Retrieval: The Benefit of PCA and Whitening". The Crossref record has no abstract; the abstract summary in my search ties mean subtraction to negative evidence and whitening to co-occurrences, which I did not open. | Standard source for centering plus whitening in retrieval. Cite for the idea only.

### Excluded (not cited as it stands)

- k-means++ (Arthur and Vassilvitskii, SODA 2007). Not opened. Crossref 404 on the guessed DOI. A bibliography listing gave pp. 1027–1035 and DOI 10.1145/1283383.1283494 (not confirmed).
- ALBEF (Li et al., NeurIPS 2021, pp. 9694–9705). The page https://proceedings.neurips.cc/paper_files/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html appeared in search only. It uses momentum-model pseudo-targets, so it is an image-text soft-target precedent. Open it and add it.
- Provodin et al. (TMLR 2025). Authors and title not seen in a primary page.
- All-but-the-Top (Mu and Viswanath, ICLR 2018, arXiv 1702.01417). Seen only in search. It removes the common mean vector and top directions from word vectors.
- Cho and Hariharan (ICCV 2019). Not rechecked.

## B. BibTeX (works not in the archived refs.bib)

```bibtex
% verified: https://arxiv.org/abs/1503.02531 (comment: NIPS 2014 Deep Learning Workshop)
@misc{hinton2015distill, title={Distilling the Knowledge in a Neural Network}, author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff}, year={2015}, eprint={1503.02531}, archivePrefix={arXiv}}
% verified: https://arxiv.org/abs/1412.6550 (no venue on the page)
@misc{romero2015fitnets, title={FitNets: Hints for Thin Deep Nets}, author={Romero, Adriana and Ballas, Nicolas and Ebrahimi Kahou, Samira and Chassang, Antoine and Gatta, Carlo and Bengio, Yoshua}, year={2014}, eprint={1412.6550}, archivePrefix={arXiv}}
% verified: https://ar5iv.labs.arxiv.org/html/1910.10699 (no venue or year on the page; arXiv id)
@misc{tian2020crd, title={Contrastive Representation Distillation}, author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip}, eprint={1910.10699}, archivePrefix={arXiv}}
% verified: https://ar5iv.labs.arxiv.org/html/1803.10837 ; venue from https://link.springer.com/chapter/10.1007/978-3-030-01252-6_17 (seen in search, not opened)
@inproceedings{passalis2018pkt, title={Learning Deep Representations with Probabilistic Knowledge Transfer}, author={Passalis, Nikolaos and Tefas, Anastasios}, booktitle={Computer Vision -- ECCV 2018}, year={2018}}
% verified: https://arxiv.org/abs/1707.01220 ; venue from https://ojs.aaai.org/index.php/AAAI/article/view/11783 (seen in search, not opened)
@inproceedings{chen2018darkrank, title={DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer}, author={Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang}, booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, volume={32}, number={1}, year={2018}, doi={10.1609/aaai.v32i1.11783}}
% verified: https://ar5iv.labs.arxiv.org/html/1904.05068 ; from note: 2026-12-26_relational_genre_term.md; page: https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html (pages 3967-3976 from the CVF listing in search)
@inproceedings{park2019rkd, title={Relational Knowledge Distillation}, author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu}, booktitle={CVPR}, year={2019}, pages={3967--3976}}
% from note: 2026-12-26_relational_genre_term.md; page: https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html (listing in search, not opened)
@inproceedings{tung2019spkd, title={Similarity-Preserving Knowledge Distillation}, author={Tung, Frederick and Mori, Greg}, booktitle={ICCV}, year={2019}, pages={1365--1374}}
% from note: 2026-12-26_relational_genre_term.md; page: https://arxiv.org/abs/2010.14713 ; NeurIPS pages from a reference entry seen in search
@inproceedings{koohpayegani2020compress, title={CompRess: Self-Supervised Learning by Compressing Representations}, author={Abbasi Koohpayegani, Soroush and Tejankar, Ajinkya and Pirsiavash, Hamed}, booktitle={Advances in Neural Information Processing Systems}, volume={33}, pages={12980--12992}, year={2020}}
% from note: 2026-12-26_relational_genre_term.md; page: https://arxiv.org/abs/2101.04731v1 (listing says accepted at ICLR 2021)
@inproceedings{fang2021seed, title={SEED: Self-supervised Distillation For Visual Representation}, author={Fang, Zhiyuan and Wang, Jianfeng and Wang, Lijuan and Zhang, Lei and Yang, Yezhou and Liu, Zicheng}, booktitle={ICLR}, year={2021}}
% from note: 2026-12-26_relational_genre_term.md; page: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html (listing in search; only the first author and "seven co-authors" seen)
@inproceedings{yang2024clipkd, title={{CLIP-KD}: An Empirical Study of {CLIP} Model Distillation}, author={Yang, Chuanguang and others}, booktitle={CVPR}, year={2024}, pages={15952--15962}}
% verified: https://arxiv.org/abs/2309.12314 (comment: Accepted by ICCV 2023)
@inproceedings{wu2023tinyclip, title={{TinyCLIP}: {CLIP} Distillation via Affinity Mimicking and Weight Inheritance}, author={Wu, Kan and Peng, Houwen and Zhou, Zhenghong and Xiao, Bin and Liu, Mengchen and Yuan, Lu and Xuan, Hong and Valenzuela, Michael and Chen, Xi (Stephen) and Wang, Xinggang and Chao, Hongyang and Hu, Han}, booktitle={ICCV}, year={2023}}
% verified: https://api.crossref.org/works/10.1109/cvpr52733.2024.01511 ; https://arxiv.org/abs/2311.17049 (comment: CVPR 2024)
@inproceedings{vasu2024mobileclip, title={{MobileCLIP}: Fast Image-Text Models through Multi-Modal Reinforced Training}, author={Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, Fartash and Vemulapalli, Raviteja and Tuzel, Oncel}, booktitle={CVPR}, year={2024}, pages={15963--15974}, doi={10.1109/cvpr52733.2024.01511}}
% from note: 2026-12-20_ways_to_improve.md; page: https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html (listing in search)
@inproceedings{ranzinger2024amradio, title={{AM-RADIO}: Agglomerative Vision Foundation Model -- Reduce All Domains Into One}, author={Ranzinger, Mike and Heinrich, Greg and Kautz, Jan and Molchanov, Pavlo}, booktitle={CVPR}, year={2024}, pages={12490--12500}}
% from note: 2026-12-20_ways_to_improve.md; page: https://link.springer.com/chapter/10.1007/978-3-031-73235-5_20 (listing in search)
@inproceedings{sariyildiz2024unic, title={{UNIC}: Universal Classification Models via Multi-teacher Distillation}, author={Sar{\i}y{\i}ld{\i}z, Mert Bulent and Weinzaepfel, Philippe and Lucas, Thomas and Larlus, Diane and Kalantidis, Yannis}, booktitle={ECCV}, year={2024}, pages={353--371}, doi={10.1007/978-3-031-73235-5_20}}
% from note: 2026-12-20_ways_to_improve.md (preprint); page: https://arxiv.org/abs/2410.01680
@misc{ranzinger2024phis, title={{PHI-S}: Distribution Balancing for Label-Free Multi-Teacher Distillation}, author={Ranzinger, Mike and Barker, Jon and Heinrich, Greg and Molchanov, Pavlo and Catanzaro, Bryan and Tao, Andrew}, year={2024}, eprint={2410.01680}, archivePrefix={arXiv}}
% verified: https://proceedings.mlr.press/v270/shang25a.html
@inproceedings{shang2025theia, title={Theia: Distilling Diverse Vision Foundation Models for Robot Learning}, author={Shang, Jinghuan and Schmeckpeper, Karl and May, Brandon B. and Minniti, Maria Vittoria and Kelestemur, Tarik and Watkins, David and Herlant, Laura}, booktitle={Proceedings of The 8th Conference on Robot Learning}, series={PMLR}, volume={270}, pages={724--748}, year={2025}}
% from note: 2026-12-30_signals_beyond_teacher.md (preprint); page: https://arxiv.org/abs/2412.19048 (authors from citing papers seen in search)
@misc{zhang2024jasper, title={Jasper and Stella: distillation of {SOTA} embedding models}, author={Zhang, Dun and Li, Jiacheng and Zeng, Ziyang and Wang, Fulong}, year={2024}, eprint={2412.19048}, archivePrefix={arXiv}}
% from note: 2026-12-20_ways_to_improve.md (preprint); page: https://arxiv.org/abs/2602.15547 (author list from the arXiv listing seen in search)
@misc{akram2026jina, title={jina-embeddings-v5-text: Task-Targeted Embedding Distillation}, author={Akram, Mohammad Kalim and Sturua, Saba and Havriushenko, Nastia and Herreros, Quentin and G{\"u}nther, Michael and Werk, Maximilian and Xiao, Han}, year={2026}, eprint={2602.15547}, archivePrefix={arXiv}}
% from note: 2026-12-30_signals_beyond_teacher.md; page: https://proceedings.mlr.press/v80/furlanello18a.html (listing in search, not opened)
@inproceedings{furlanello2018born, title={Born Again Neural Networks}, author={Furlanello, Tommaso and Lipton, Zachary and Tschannen, Michael and Itti, Laurent and Anandkumar, Anima}, booktitle={Proceedings of the 35th International Conference on Machine Learning}, series={PMLR}, volume={80}, pages={1607--1616}, year={2018}}
% verified: https://api.crossref.org/works/10.1109/cvpr42600.2020.01070
@inproceedings{xie2020noisystudent, title={Self-Training With Noisy Student Improves {ImageNet} Classification}, author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V.}, booktitle={CVPR}, year={2020}, pages={10684--10695}, doi={10.1109/cvpr42600.2020.01070}}
% verified: https://api.crossref.org/works/10.1016/j.neunet.2009.06.042
@article{vapnik2009lupi, title={A new learning paradigm: Learning using privileged information}, author={Vapnik, Vladimir and Vashist, Akshay}, journal={Neural Networks}, year={2009}, pages={544--557}, doi={10.1016/j.neunet.2009.06.042}}
% verified: https://arxiv.org/abs/1511.03643 (page: published in ICLR 2016 proceedings)
@inproceedings{lopezpaz2016unifying, title={Unifying distillation and privileged information}, author={Lopez-Paz, David and Bottou, L{\'e}on and Sch{\"o}lkopf, Bernhard and Vapnik, Vladimir}, booktitle={ICLR}, year={2016}}
% verified: https://arxiv.org/abs/1807.03748 (no venue on the page)
@misc{oord2018cpc, title={Representation Learning with Contrastive Predictive Coding}, author={van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol}, year={2018}, eprint={1807.03748}, archivePrefix={arXiv}}
% verified: https://proceedings.mlr.press/v119/chen20j.html
@inproceedings{chen2020simclr, title={A Simple Framework for Contrastive Learning of Visual Representations}, author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey}, booktitle={Proceedings of the 37th International Conference on Machine Learning}, series={PMLR}, volume={119}, pages={1597--1607}, year={2020}}
% verified: https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
@inproceedings{khosla2020supcon, title={Supervised Contrastive Learning}, author={Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip}, booktitle={Advances in Neural Information Processing Systems}, volume={33}, year={2020}}
% verified: https://arxiv.org/abs/2104.14548 (comment: Accepted at ICCV 2021)
@inproceedings{dwibedi2021nnclr, title={With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations}, author={Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew}, booktitle={ICCV}, year={2021}}
% verified: https://arxiv.org/abs/2105.07269 (no venue on the page)
@misc{koohpayegani2021msf, title={Mean Shift for Self-Supervised Learning}, author={Abbasi Koohpayegani, Soroush and Tejankar, Ajinkya and Pirsiavash, Hamed}, year={2021}, eprint={2105.07269}, archivePrefix={arXiv}}
% verified: https://arxiv.org/abs/2011.11183 (no venue on the page)
@misc{li2021comatch, title={{CoMatch}: Semi-supervised Learning with Contrastive Graph Regularization}, author={Li, Junnan and Xiong, Caiming and Hoi, Steven}, year={2020}, eprint={2011.11183}, archivePrefix={arXiv}}
% verified: https://arxiv.org/abs/2005.12320 (comment: Accepted at ECCV 2020)
@inproceedings{vangansbeke2020scan, title={{SCAN}: Learning to Classify Images without Labels}, author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc}, booktitle={ECCV}, year={2020}}
% verified: https://proceedings.mlr.press/v48/xieb16.html
@inproceedings{xie2016dec, title={Unsupervised Deep Embedding for Clustering Analysis}, author={Xie, Junyuan and Girshick, Ross and Farhadi, Ali}, booktitle={Proceedings of The 33rd International Conference on Machine Learning}, series={PMLR}, volume={48}, pages={478--487}, year={2016}}
% verified: https://www.ijcai.org/proceedings/2017/243
@inproceedings{guo2017idec, title={Improved Deep Embedded Clustering with Local Structure Preservation}, author={Guo, Xifeng and Gao, Long and Liu, Xinwang and Yin, Jianping}, booktitle={Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17)}, year={2017}, pages={1753--1759}}
% verified: https://arxiv.org/abs/1701.08398 (comment: To appear in CVPR 2017)
@inproceedings{zhong2017kreciprocal, title={Re-ranking Person Re-identification with k-reciprocal Encoding}, author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi}, booktitle={CVPR}, year={2017}}
% verified: https://proceedings.mlr.press/v235/huh24a.html ; https://ar5iv.labs.arxiv.org/html/2405.07987
@inproceedings{huh2024platonic, title={Position: The Platonic Representation Hypothesis}, author={Huh, Minyoung and Cheung, Brian and Wang, Tongzhou and Isola, Phillip}, booktitle={Proceedings of the 41st International Conference on Machine Learning}, series={PMLR}, volume={235}, pages={20617--20642}, year={2024}}
% verified: https://arxiv.org/abs/1802.03426 (no journal reference on the page)
@misc{mcinnes2018umap, title={{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction}, author={McInnes, Leland and Healy, John and Melville, James}, year={2018}, eprint={1802.03426}, archivePrefix={arXiv}}
% verified: https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/ (author list on the page was cut with an ellipsis: complete it before submission)
@article{hao2021wnn, title={Integrated analysis of multimodal single-cell data}, author={Hao, Yuhan and Hao, Stephanie and Andersen-Nissen, Erica and others}, journal={Cell}, volume={184}, number={13}, pages={3573--3587.e29}, year={2021}}
% from note: 2026-12-05_percept_neighbourhood.md; page: https://proceedings.mlr.press/v235/li24aa.html (listing in search)
@inproceedings{li2024tac, title={Image Clustering with External Guidance}, author={Li, Yunfan and Hu, Peng and Peng, Dezhong and Lv, Jiancheng and Fan, Jianping and Peng, Xi}, booktitle={Proceedings of the 41st International Conference on Machine Learning}, series={PMLR}, volume={235}, pages={27890--27902}, year={2024}}
% from note: 2026-12-05_percept_neighbourhood.md; page: CVF PDF (my fetch got a 403; authors and DOI 10.1109/ICCV.2019.00312 from a search summary)
@inproceedings{su2019djsrh, title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval}, author={Su, Shupeng and Zhong, Zhisheng and Zhang, Chao}, booktitle={ICCV}, year={2019}}
% verified: https://ar5iv.labs.arxiv.org/html/2110.05208 ; venue from https://iclr.cc/virtual/2022/poster/6406 (seen in search, not opened)
@inproceedings{li2022declip, title={Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm}, author={Li, Yangguang and Liang, Feng and Zhao, Lichen and Cui, Yufeng and Ouyang, Wanli and Shao, Jing and Yu, Fengwei and Yan, Junjie}, booktitle={ICLR}, year={2022}}
% verified: https://arxiv.org/abs/2204.03610 (comment: CVPR 2022)
@inproceedings{yang2022unicl, title={Unified Contrastive Learning in Image-Text-Label Space}, author={Yang, Jianwei and Li, Chunyuan and Zhang, Pengchuan and Xiao, Bin and Liu, Ce and Yuan, Lu and Gao, Jianfeng}, booktitle={CVPR}, year={2022}}
% verified: https://iclr.cc/virtual/2019/poster/935 ; https://arxiv.org/abs/1711.05101
@inproceedings{loshchilov2019adamw, title={Decoupled Weight Decay Regularization}, author={Loshchilov, Ilya and Hutter, Frank}, booktitle={ICLR}, year={2019}}
% verified: https://arxiv.org/abs/1606.08415 (no venue on the page)
@misc{hendrycks2016gelu, title={Gaussian Error Linear Units ({GELU}s)}, author={Hendrycks, Dan and Gimpel, Kevin}, year={2016}, eprint={1606.08415}, archivePrefix={arXiv}}
% verified: https://jmlr.org/papers/v15/srivastava14a.html
@article{srivastava2014dropout, title={Dropout: A Simple Way to Prevent Neural Networks from Overfitting}, author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan}, journal={Journal of Machine Learning Research}, volume={15}, number={56}, pages={1929--1958}, year={2014}}
% verified: https://jmlr.org/papers/v11/vinh10a.html
@article{vinh2010ami, title={Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance}, author={Vinh, Nguyen Xuan and Epps, Julien and Bailey, James}, journal={Journal of Machine Learning Research}, volume={11}, pages={2837--2854}, year={2010}}
% verified: https://api.crossref.org/works/10.1109/ICCV.2007.4408891
@inproceedings{chum2007aqe, title={Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval}, author={Chum, Ondrej and Philbin, James and Sivic, Josef and Isard, Michael and Zisserman, Andrew}, booktitle={ICCV}, year={2007}, pages={1--8}}
% verified: https://ar5iv.labs.arxiv.org/html/1711.02512 ; https://arxiv.org/abs/1711.02512 (comment "TPAMI 2018"; volume and pages not seen)
@article{radenovic2018finetune, title={Fine-tuning {CNN} Image Retrieval with No Human Annotation}, author={Radenovi{\'c}, Filip and Tolias, Giorgos and Chum, Ond{\v r}ej}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, year={2018}}
% verified: https://api.crossref.org/works/10.1007/978-3-642-33709-3_55
@inproceedings{jegou2012whitening, title={Negative Evidences and Co-occurences in Image Retrieval: The Benefit of {PCA} and Whitening}, author={J{\'e}gou, Herv{\'e} and Chum, Ond{\v r}ej}, booktitle={Computer Vision -- ECCV 2012}, series={Lecture Notes in Computer Science}, year={2012}, pages={774--787}}
```

## C. Closest precedents

**Graph-positive four-way cross-modal contrastive loss.** Four sources are closest. None uses a teacher-built graph in both modalities.
- **DeCLIP, nearest-neighbour supervision (`li2022declip`).** The positive for an image is the nearest text feature (cosine) in a 64K FIFO queue of the model's own text embeddings. It is InfoNCE with that neighbour text as positive. It works in the image-to-text direction only and takes neighbours from the model's own, changing embedding space. Ours uses a fixed graph from a separate instructed teacher, per aspect, reciprocal, with a content similarity subtracted. We use all four directions, including caption-caption.
- **NNCLR (`dwibedi2021nnclr`).** The nearest neighbour of an embedding is the positive. It is image-only, with no teacher and no captions.
- **SupCon and UniCL (`khosla2020supcon`, `yang2022unicl`).** Multi-positive contrastive loss, with positives set by labels. UniCL does this across image and text. Ours has no labels. The graph replaces them.
- **TAC and DJSRH (`li2024tac`, `su2019djsrh`).** Cross-modal neighbourhood information supervises a cluster head or hash codes. Both are described from notes and listings only, so I would not describe their losses.

**Relational term.** The closest forms, as the sources give them:
- **PKT.** KL divergence between mini-batch conditional probabilities built from a cosine kernel (cos+1)/2 divided by the sum over the other samples, p_{j|i} log(p_{j|i}/q_{j|i}). It includes a cross-modal case. Closest in-batch KL.
- **CompRess and SEED.** KL(p_teacher ‖ p_student) over softmaxed cosine similarities to a queue of teacher anchors, with a temperature. CompRess uses 128,000 anchors and SEED 65,536 (the SEED form is cross-entropy, equal to KL up to a constant, with unequal temperatures).
- **CLIP-KD.** In-batch image-text softmax rows aligned with KL, teacher first (from the note).
- **SP-KD.** The same in-batch similarity matrix, compared by squared Frobenius distance. **RKD** uses Huber loss on normalised distances and angles. **DarkRank** is ranking-based, form unseen.

Ours: KL between in-batch teacher and student similarity distributions for genre, with no anchors. It sits between PKT (in-batch, KL, but sum-normalised, not softmax) and CompRess (softmax and KL, but a queue). The teacher distribution is built from the instructed view and the student sees only the plain image. The 12-26 note says the temperature of our relational term was not given. Confirm it before the method section compares it with CompRess and SEED.

## D. Open

- Four entries are `note + listing` and one is excluded without being opened: DJSRH (CVF 403), CLIP-KD (author list incomplete), PKT, DarkRank, RKD, SP-KD, CompRess, SEED, AM-RADIO, UNIC, Born-Again, TAC (listing only). Open each official page to move it to fully verified.
- Excluded without opening: k-means++, ALBEF, All-but-the-Top, Provodin et al., and Cho and Hariharan.
- Not opened: the exact temperatures and weights in TinyCLIP and CLIP-KD, and whether TinyCLIP's affinity loss is KL.
- Not found: a clear precedent for a four-direction (image-image, image-caption, caption-image, caption-caption) graph-positive loss. DeCLIP is the closest. One more search for "multi-positive cross-modal contrastive neighbour" would settle it.
- The WNN author list, FitNets and CRD venues, and Radenović's TPAMI volume and pages are incomplete on the pages I saw.
- Working note for later: `hendrycks2016gelu`, `oord2018cpc`, `koohpayegani2021msf`, `li2021comatch`, `mcinnes2018umap` and `ranzinger2024phis` are arXiv `@misc` because no opened page shows a venue.

---

## Sources, H5a and H5b (affective and artistic image analysis)

Deep-research gap-fill (Sonnet high), 2026-11-02, for the related work of the first draft. Saved verbatim from the agent's report by the paper chat.

Related Work source check, H5a and H5b. Budget used: 13 of 35 web searches, 25 of 25 page reads (several reads were blocked, see C).

Summary: 34 works are checked (12 already in the archived refs.bib, 22 new with BibTeX in B). Four seed items are excluded or dropped (iMemNet as a memorability work, StyleDecoupler/MegaStyle as unopened, Huckle, PolyArt as a name). MMArt is on arXiv (2608.10706) and its author list includes Wangyuan Ding, so cite it as your own preprint and mind the double-blind rule.

Conventions: "bib" means the work is in the archived refs.bib. "note: X" means a literature note states it and does not flag it. "verified now" means I opened the page; the Crossref API counts as a page, following the precedent in refs.bib. Where I only saw a venue in an arXiv comments field, the entry says so, as in refs.bib.

## A. Sources per heading

### H5a. Affective image analysis and affective captions

- `zhao2022affective` | verified now: https://arxiv.org/abs/2106.16125 ; https://api.crossref.org/works?filter=doi:10.1109/tpami.2021.3094362 | Survey of affective image content analysis. It covers emotion models, features, learning methods and applications, and names the affective gap, perception subjectivity and label noise. IEEE TPAMI 44(10), pp. 6729-6751, 2022. | Cite as the field survey; it anchors the claim that labels are subjective and noisy.
- `machajdik2010affective` | verified now: https://api.crossref.org/works?filter=doi:10.1145/1873951.1873965 | "Affective image classification using features inspired by psychology and art theory" (Machajdik, Hanbury), ACM Multimedia 2010, pp. 83-92. I saw only the Crossref record, not the abstract. The ArtPhoto name and size (806 images) come only from the 2027-01-02 note, which does not mark them as snippets, but I did not see them on a page. | Early hand-crafted features for affect; the origin of the ArtPhoto set.
- `you2016building` | verified now: https://arxiv.org/abs/1605.02677 | "Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark" (You, Luo, Jin, Yang), AAAI 2016 per the arXiv comments field. It builds a dataset from over 3 million weakly labeled images and benchmarks CNNs. The abstract does not name the "FI" label set or give 23,308 images (that is from the note, from a snippet). | Photo emotion classification baseline. Sizes conflict: the note says FI has 23,308 images and 23,164 in another table, and the controller's data log says 21,808 FI images inside Affection. Do not state an FI size in the paper. FI is an Affection source, so it cannot be a held-out test set.
- `peng2015emotion6` | verified now: https://api.crossref.org/works?filter=doi:10.1109/cvpr.2015.7298687 (the CVF page returned 403) | "A mixed bag of emotions: Model, predict, and transfer emotion distributions" (Peng, Chen, Sadovnik, Gallagher), CVPR 2015, pp. 860-868. Per the note it introduces Emotion6 (1,980 Flickr images, Ekman 6 plus neutral, distributions); Crossref does not show this. | Emotion distributions per image, not one label.
- `panda2018webemo` | verified now: https://arxiv.org/abs/1808.02212 (venue ECCV 2018 from the arXiv comments field) | "Contemplating Visual Emotions: Understanding and Overcoming Dataset Bias" (Panda et al.). It shows that models trained on small emotion datasets generalize badly, and proposes webly supervised training on stock images with a curriculum. The abstract does not use the name WEBEmo or give a size. The note says about 268k images and 25 classes, labels from search keywords. | Weakly labeled, web-scale emotion data. Use the abstract's wording and avoid the size.
- `yang2023emoset` | verified now: https://arxiv.org/abs/2307.07961 (accepted to ICCV 2023 per the comments field; the CVF page returned 403) ; note: 2027-01-02_photo_aspect_datasets.md | EmoSet: 3.3 million images in total, 118,102 labeled by human annotators. Each labeled image has brightness, colorfulness, scene type, object class, facial expression and human action. The abstract does not list the emotion names; the note gives Mikels' 8 classes. | Our second photo emotion test set. No captions, and attributes are machine-predicted per the note.
- `achlioptas2023affection` | note: 2026-12-01_affect_caption_datasets.md (the controller checked the arXiv abstract on 2026-10-13); https://arxiv.org/abs/2210.01946 ; CVF page in the note | Affection (CVPR 2023): 85,007 images, 526,749 explanations, 6,283 annotators, 8 emotions plus "something-else". Explanations are written in the ArtEmis protocol. | The photo counterpart of ArtEmis; our photo training source. Note: PercepT says it lacks genre labels.
- `achlioptas2021artemis` | bib | ArtEmis. The counts differ by version: 454,684 utterances on 80,031 artworks in the CVPR PDF, 439,121 on 81,446 in the latest arXiv PDF. | Our painting source. Quote the CVPR numbers or avoid exact counts.
- `mohamed2022okay` | verified now: https://arxiv.org/abs/2204.07660 | "It is Okay to Not Be Okay: Overcoming Emotional Bias in Affective Image Captioning by Contrastive Data Collection" (ArtEmis v2.0). CVPR 2022 per the comments field. It adds 260,533 instances on similar images with opposing emotions, and reports CIDEr +20% and METEOR +7%. | Emotion bias in affective captions. Correction: the affect-caption note gives arXiv 2101.07396 for ArtEmis v2.0, which is wrong. That id is ArtEmis. The correct id is 2204.07660, as in the literature review.
- `mohamed2022artelingo` | bib | ArtELingo (EMNLP 2022). Per the digest: Arabic, Chinese and Spanish annotations on ArtEmis. | Our data.
- `mohamed2024artelingo28` | bib | ArtELingo-28 (EMNLP 2024), 28 languages. The paper states 2,000 images (body, Table 1, Appendix C.1), but about 200K annotations at 140 per image gives about 1,430. | Our data. Write "2,000 images" only with the caveat.
- `mohamed2026percept` | bib | PercepT, arXiv only. Do not call its CLIP encoder "frozen". | The direct prior work.
- `mohammad2018wikiart` | verified now: https://aclanthology.org/L18-1197/ (title, authors, venue; the abstract is not on the page) ; note: 2026-12-01_affect_caption_datasets.md (4,105 art pieces, mostly paintings, emotions evoked by the image and title, liking score; checked on saifmohammad.com) | WikiArt Emotions, LREC 2018, Mohammad and Kiritchenko. The only text is the art title. | Emotion labels for art without captions.
- `mertens2024findingemo` | note: 2027-01-02_photo_aspect_datasets.md; the NeurIPS abstract URL in the note returned 404 for me. A search result lists the proceedings page and gives the authors. | FindingEmo, NeurIPS 2024 Datasets and Benchmarks track. 25,869 images with Plutchik's 24 emotions, valence and arousal; only a URL list is released. The note says its licence was not found. | A recent in-the-wild emotion set. The 8-class map is coarse.
- `zhang2023emotionclip` | bib | EmotionCLIP (CVPR 2023). Per the digest, a frozen text sentiment model gives distant supervision. | Precedent for using a text emotion model as a teacher. Our GoEmotions teacher is a similar case.
- `demszky2020goemotions` | bib | GoEmotions (ACL 2020). Cite only as the text classifier PercepT uses. | Label space of the caption side.
- `xie2024emovit` | verified now: https://arxiv.org/abs/2404.16670 (Accepted by CVPR 2024 in the comments; the CVF page returned 403) | EmoVIT: visual instruction tuning for emotion. A GPT-assisted pipeline makes emotion instruction data, and the model builds on InstructBLIP. It covers emotion classification, affective reasoning and humor. | Multimodal-LLM emotion work (2024).
- `lee2025emovlmkd` | verified now: https://arxiv.org/abs/2505.07164 (ABAW workshop, CVPR 2025, from the comments field) | EmoVLM-KD (Lee, Lee, Park). It fine-tunes a VLM on emotion instruction data and distills a vision model into a lightweight module attached to the frozen VLM, with a gate between the two. | Multimodal-LLM emotion work (2025). It predicts the dominant emotion; it does not discover aspects.
- `zhao2020imemnet` | bib | The bib verified line says it is IMEMNet, emotion-based image-music matching in valence-arousal space (ACM MM 2020). | Not an image memorability work, as the seed list assumed. Drop from H5a, or cite only as emotion-based cross-modal matching.

### H5b. Artistic style and genre recognition

- `karayev2014style` | verified now: https://arxiv.org/abs/1311.3715 | "Recognizing Image Style" (7 authors), BMVC 2014 per the page. It introduces 80K Flickr photographs with 20 curated style labels and 85K paintings with 25 style/genre labels. It finds that features learned with object labels work best. | The origin of Flickr Style. The 2027-01-02 note could not open the author's page (404), so the data may not be obtainable.
- `saleh2015largescale` | verified now: https://arxiv.org/abs/1505.00855 | "Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature" (Saleh, Elgammal), arXiv, May 2015 (the seed list says 2016; the page shows 2015, no journal reference). It predicts style, genre and artist with metric learning. | Early WikiArt style/genre/artist classification. Cite as arXiv only.
- `tan2016ceci` | verified now: https://api.crossref.org/works?filter=doi:10.1109/icip.2016.7533051 | "Ceci n'est pas une pipe: A deep convolutional network for fine-art paintings classification" (Tan, Chan, Aguirre, Tanaka), IEEE ICIP 2016, pp. 3703-3707. Crossref only; I did not see the abstract. | First end-to-end CNN on WikiArt.
- `tan2019artgan` | bib | ArtGAN (IEEE TIP). Its `genre_class` file is the genre source for our ArtELingo genre ids. | Where our genre labels come from.
- `mao2017deepart` | verified now: https://api.crossref.org/works?filter=doi:10.1145/3123266.3123405 | "DeepArt" (Mao, Cheung, She), ACM Multimedia 2017, pp. 1183-1191. The Crossref title is the single word "DeepArt"; I did not see the abstract. | Joint visual-art representation (title only).
- `elgammal2018shape` | verified now: https://arxiv.org/abs/1801.07729 ; https://api.crossref.org/works?filter=doi:10.1609/aaai.v32i1.11894 | "The Shape of Art History in the Eyes of the Machine", AAAI 2018 (vol. 32, issue 1). CNNs are trained on about 77,000 paintings, and their representations are compared with art-historical concepts. The author order differs: Crossref has Elgammal, Liu, Kim, Elhoseiny, Mazzone; arXiv has Elgammal, Mazzone, Liu, Kim, Elhoseiny. I used the Crossref order. | Style representations follow art-historical order.
- `cetinic2018finetuning` | verified now: https://api.crossref.org/works?filter=doi:10.1016/j.eswa.2018.07.026 | "Fine-tuning Convolutional Neural Networks for fine art classification" (Cetinic, Lipic, Grgic), Expert Systems with Applications 114, pp. 107-118, 2018. Crossref only; the abstract text was not opened. | Fine-art classification baseline.
- `garcia2018semart` | bib | SemArt (ECCVW 2018): 21,384 paintings with comments; text to image R@1 0.144 per the digest. | Art retrieval with text, no affect.
- `strezoski2017omniart` | verified now: https://arxiv.org/abs/1708.00684 | "OmniArt: Multi-task Deep Learning for Artistic Data Analysis" (Strezoski, Worring), arXiv 2017. A multi-task shared representation, plus an aggregated dataset of "almost half a million samples". No journal reference on the page, so cite it as arXiv. | Multi-task art analysis; the dataset aggregates sources.
- `wu2023goya` | verified now: https://arxiv.org/abs/2304.10278 ; https://api.crossref.org/works?filter=doi:10.1145/3591106.3592262 | GOYA, "Not Only Generative Art: Stable Diffusion for Content-Style Disentanglement in Art Analysis", ICMR 2023, pp. 199-208 (Wu, Nakashima, Garcia). It distills a generative model to separate content and style. Per note 2026-11-27: frozen CLIP B/32, style 50.90 against 51.23 for CLIP, and WikiArt is evaluation only. | The closest published content/style split on WikiArt.
- `ruta2021aladin` | verified now: https://arxiv.org/abs/2103.09776 (8 authors, arXiv only on this page). A search result listed the CVF page as ICCV 2021, but I could not open it. | ALADIN: weakly supervised style similarity search. It introduces BAM-FG, "a new 2.62 million image dataset of 310,000 fine-grained style groupings". | Style retrieval with content/style separation (trained on Behance, not WikiArt, per note 2026-11-27).
- `somepalli2024csd` | bib | CSD, ECCV 2024 (Crossref record), arXiv v1 title "Measuring Style Similarity in Diffusion Models". Numbers differ between sources (note 2026-11-27 §6). | Strongest style-embedding baseline.
- `kawarada2025training` | bib | DIOR, arXiv only. Per the digest it loses to plain CLIP on WikiArt style (mAP@1 45.6 against 59.3). | A conditional embedding on art style.
- `wang2026mmart` | verified now: https://arxiv.org/abs/2608.10706 | "MMArt: A Multi-Perspective Multimodal Dataset for Visual Art Understanding" (Shuai Wang, Wangyuan Ding, Yixian Shen, Jia-Hong Huang, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring). arXiv only, no venue; v1 11 Aug 2026, v3 21 Sep 2026. 74,234 WikiArt paintings, four independently annotated perspectives (narrative, formal, emotional, historical) plus a unified caption, from specialized VLMs or human annotators. Retrieval R@1 is 44.0% for narrative and 7.8% for formal descriptions. | Our next evaluation set. The name PolyArt appears on neither this page nor in the searches (two searches). | Check whether PolyArt is an earlier internal name for MMArt.

Excluded or dropped:
- `huckle2020style` (Huckle, Garcia, Nakashima, ECCVW 2020; AMI against style 0.191): excluded, a search snippet only (note 2026-11-27) and not opened here.
- `StyleDecoupler` (2601.17697) and `MegaStyle` (2604.08364): not opened; the notes mark them as snippets and summaries. They are excluded from the checked list.
- `AICA-Bench` (2604.05900), `EmoCaliber`, `ArtECulture` (2608.03358): no page opened; I did not get a title and author list. Excluded.
- `FlickrLDL/TwitterLDL`, `EMOTIC`, `MELD`, `EmoVerse` (TAffC 2026): the notes did not re-check them, and I did not open them. Excluded.

## B. BibTeX (new entries only)

```bibtex
% verified: https://arxiv.org/abs/2106.16125 ; https://api.crossref.org/works?filter=doi:10.1109/tpami.2021.3094362 (IEEE TPAMI 44(10), pp 6729-6751, 2022)
@article{zhao2022affective,
  author  = {Zhao, Sicheng and Yao, Xingxu and Yang, Jufeng and Jia, Guoli and Ding, Guiguang and Chua, Tat-Seng and Schuller, Bj{\"o}rn W. and Keutzer, Kurt},
  title   = {Affective Image Content Analysis: Two Decades Review and New Perspectives},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {44},
  number  = {10},
  pages   = {6729--6751},
  year    = {2022}
}

% verified: https://api.crossref.org/works?filter=doi:10.1145/1873951.1873965 (Crossref record, ACM)
@inproceedings{machajdik2010affective,
  author    = {Machajdik, Jana and Hanbury, Allan},
  title     = {Affective image classification using features inspired by psychology and art theory},
  booktitle = {Proceedings of the 18th ACM International Conference on Multimedia},
  pages     = {83--92},
  year      = {2010}
}

% verified: https://arxiv.org/abs/1605.02677 (venue AAAI 2016 from the arXiv comments field; proceedings page not opened)
@inproceedings{you2016building,
  author    = {You, Quanzeng and Luo, Jiebo and Jin, Hailin and Yang, Jianchao},
  title     = {Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2016},
  eprint    = {1605.02677},
  archivePrefix = {arXiv}
}

% verified: https://api.crossref.org/works?filter=doi:10.1109/cvpr.2015.7298687 (Crossref record, CVPR 2015, pp 860-868); CVF page returned 403
@inproceedings{peng2015emotion6,
  author    = {Peng, Kuan-Chuan and Chen, Tsuhan and Sadovnik, Amir and Gallagher, Andrew},
  title     = {A Mixed Bag of Emotions: Model, Predict, and Transfer Emotion Distributions},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {860--868},
  year      = {2015}
}

% verified: https://arxiv.org/abs/1808.02212 (arXiv comments field: ECCV 2018; proceedings page not opened)
@inproceedings{panda2018webemo,
  author    = {Panda, Rameswar and Zhang, Jianming and Li, Haoxiang and Lee, Joon-Young and Lu, Xin and Roy-Chowdhury, Amit K.},
  title     = {Contemplating Visual Emotions: Understanding and Overcoming Dataset Bias},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018},
  eprint    = {1808.02212},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2307.07961 (comments: Accepted to ICCV 2023; CVF page returned 403)
@inproceedings{yang2023emoset,
  author    = {Yang, Jingyuan and Huang, Qirui and Ding, Tingting and Lischinski, Dani and Cohen-Or, Daniel and Huang, Hui},
  title     = {{EmoSet}: A Large-scale Visual Emotion Dataset with Rich Attributes},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  eprint    = {2307.07961},
  archivePrefix = {arXiv}
}

% from note: 2026-12-01_affect_caption_datasets.md; page: https://arxiv.org/abs/2210.01946 (the note says the controller checked the abstract). Surnames only: given names were not seen on any page; copy them from the CVF page before use.
@inproceedings{achlioptas2023affection,
  author    = {Achlioptas and Ovsjanikov and Guibas and Tulyakov},
  title     = {Affection: Learning Affective Explanations for Real-World Visual Data},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

% verified: https://arxiv.org/abs/2204.07660 (CVPR 2022 per the listing, 8 pages)
@inproceedings{mohamed2022okay,
  author    = {Mohamed, Youssef and Khan, Faizan Farooq and Haydarov, Kilichbek and Elhoseiny, Mohamed},
  title     = {It is Okay to Not Be Okay: Overcoming Emotional Bias in Affective Image Captioning by Contrastive Data Collection},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  eprint    = {2204.07660},
  archivePrefix = {arXiv}
}

% verified: https://aclanthology.org/L18-1197/ (title, authors, LREC 2018, ELRA; no pages on the page)
@inproceedings{mohammad2018wikiart,
  author    = {Mohammad, Saif M. and Kiritchenko, Svetlana},
  title     = {{WikiArt} Emotions: An Annotated Dataset of Emotions Evoked by Art},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  publisher = {European Language Resources Association (ELRA)},
  address   = {Miyazaki, Japan},
  year      = {2018}
}

% from note: 2027-01-02_photo_aspect_datasets.md; page: https://proceedings.neurips.cc/paper_files/paper/2024/hash/08a7229eaba3b35cd8f933ac678f2096-Abstract.html (the note's URL; I could not open it). Authors from a search-result listing of that page; given names not checked.
@inproceedings{mertens2024findingemo,
  author    = {Mertens, Laurent and Yargholi, Elahe' and Op de Beeck, Hans and Van den Stock, Jan and Vennekens, Joost},
  title     = {{FindingEmo}: An Image Dataset for Emotion Recognition in the Wild},
  booktitle = {Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year      = {2024}
}

% verified: https://arxiv.org/abs/2404.16670 (comments: Accepted by CVPR 2024; CVF page returned 403)
@inproceedings{xie2024emovit,
  author    = {Xie, Hongxia and Peng, Chu-Jun and Tseng, Yu-Wen and Chen, Hung-Jen and Hsu, Chan-Feng and Shuai, Hong-Han and Cheng, Wen-Huang},
  title     = {{EmoVIT}: Revolutionizing Emotion Insights with Visual Instruction Tuning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  eprint    = {2404.16670},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2505.07164 (comments: accepted at the ABAW workshop, CVPR 2025; workshop proceedings page not opened)
@inproceedings{lee2025emovlmkd,
  author    = {Lee, SangEun and Lee, Yubeen and Park, Eunil},
  title     = {{EmoVLM-KD}: Fusing Distilled Expertise with Vision-Language Models for Visual Emotion Analysis},
  booktitle = {Workshop and Competition on Affective \& Behavior Analysis in-the-wild (ABAW), CVPR Workshops},
  year      = {2025},
  eprint    = {2505.07164},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/1311.3715 (page: "Proc. British Machine Vision Conference (BMVC) 2014")
@inproceedings{karayev2014style,
  author    = {Karayev, Sergey and Trentacoste, Matthew and Han, Helen and Agarwala, Aseem and Darrell, Trevor and Hertzmann, Aaron and Winnemoeller, Holger},
  title     = {Recognizing Image Style},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2014},
  eprint    = {1311.3715},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/1505.00855 (arXiv only; no journal reference on the page)
@misc{saleh2015largescale,
  author       = {Saleh, Babak and Elgammal, Ahmed},
  title        = {Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature},
  year         = {2015},
  eprint       = {1505.00855},
  archivePrefix = {arXiv}
}

% verified: https://api.crossref.org/works?filter=doi:10.1109/icip.2016.7533051 (Crossref record, IEEE ICIP 2016, pp 3703-3707)
@inproceedings{tan2016ceci,
  author    = {Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan E. and Tanaka, Kiyoshi},
  title     = {Ceci n'est pas une pipe: A deep convolutional network for fine-art paintings classification},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  pages     = {3703--3707},
  year      = {2016}
}

% verified: https://api.crossref.org/works?filter=doi:10.1145/3123266.3123405 (Crossref record, ACM MM 2017, pp 1183-1191; Crossref title is the single word "DeepArt")
@inproceedings{mao2017deepart,
  author    = {Mao, Hui and Cheung, Ming and She, James},
  title     = {{DeepArt}},
  booktitle = {Proceedings of the 25th ACM International Conference on Multimedia},
  pages     = {1183--1191},
  year      = {2017}
}

% verified: https://api.crossref.org/works?filter=doi:10.1609/aaai.v32i1.11894 (Crossref record, AAAI vol 32 issue 1; author order as in Crossref, arXiv 1801.07729 orders Elgammal, Mazzone, Liu, Kim, Elhoseiny) ; https://arxiv.org/abs/1801.07729
@inproceedings{elgammal2018shape,
  author    = {Elgammal, Ahmed and Liu, Bingchen and Kim, Diana and Elhoseiny, Mohamed and Mazzone, Marian},
  title     = {The Shape of Art History in the Eyes of the Machine},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  year      = {2018}
}

% verified: https://api.crossref.org/works?filter=doi:10.1016/j.eswa.2018.07.026 (Crossref record, Expert Systems with Applications vol 114, pp 107-118)
@article{cetinic2018finetuning,
  author  = {Cetinic, Eva and Lipic, Tomislav and Grgic, Sonja},
  title   = {Fine-tuning Convolutional Neural Networks for fine art classification},
  journal = {Expert Systems with Applications},
  volume  = {114},
  pages   = {107--118},
  year    = {2018}
}

% verified: https://arxiv.org/abs/1708.00684 (arXiv only; 9 pages; no journal reference on the page)
@misc{strezoski2017omniart,
  author       = {Strezoski, Gjorgji and Worring, Marcel},
  title        = {{OmniArt}: Multi-task Deep Learning for Artistic Data Analysis},
  year         = {2017},
  eprint       = {1708.00684},
  archivePrefix = {arXiv}
}

% verified: https://api.crossref.org/works?filter=doi:10.1145/3591106.3592262 (Crossref record, ICMR 2023, pp 199-208) ; https://arxiv.org/abs/2304.10278
@inproceedings{wu2023goya,
  author    = {Wu, Yankun and Nakashima, Yuta and Garcia, Noa},
  title     = {Not Only Generative Art: Stable Diffusion for Content-Style Disentanglement in Art Analysis},
  booktitle = {Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
  pages     = {199--208},
  year      = {2023}
}

% verified: https://arxiv.org/abs/2103.09776 (arXiv only on the page; ICCV 2021 was seen only in a search-result listing of the CVF page, which returned 403, so not stated here)
@misc{ruta2021aladin,
  author       = {Ruta, Dan and Motiian, Saeid and Faieta, Baldo and Lin, Zhe and Jin, Hailin and Filipkowski, Alex and Gilbert, Andrew and Collomosse, John},
  title        = {{ALADIN}: All Layer Adaptive Instance Normalization for Fine-grained Style Similarity},
  year         = {2021},
  eprint       = {2103.09776},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2608.10706 (arXiv only; no venue on the page; v1 2026-08-11, v3 2026-09-21)
@misc{wang2026mmart,
  author       = {Wang, Shuai and Ding, Wangyuan and Shen, Yixian and Huang, Jia-Hong and Rudinac, Stevan and Kackovic, Monika and Wijnberg, Nachoem and Worring, Marcel},
  title        = {{MMArt}: A Multi-Perspective Multimodal Dataset for Visual Art Understanding},
  year         = {2026},
  eprint       = {2608.10706},
  archivePrefix = {arXiv}
}
```

## C. Open

1. Blocked pages. CVF returned 403 (EmoVIT, EmoSet, Emotion6, ALADIN). IEEE Xplore returned 418 (Tan 2016). The AAAI OJS page was blocked by Anubis. The NeurIPS FindingEmo URL in the note returned 404. For these I relied on arXiv or Crossref. A fresh session could try `papers.nips.cc/paper_files/paper/2024/hash/08a7229eaba3b35cd8f933ac678f2096-Abstract-Datasets_and_Benchmarks_Track.html` for FindingEmo, and the CVF ALADIN and EmoSet pages. That would confirm ALADIN as ICCV 2021 and the author given names for FindingEmo.
2. Affection given names (Achlioptas, Ovsjanikov, Guibas, Tulyakov): read the CVF or arXiv abs page, which I did not open (page budget).
3. Not opened for dataset facts: the abstracts of Machajdik 2010, Tan 2016, Mao 2017 and Cetinic 2018 (Crossref only). Emotion6's size (1,980) and WEBEmo's size and classes rest on the 2027-01-02 note and do not appear on the pages I opened. Use the notes' figures only after checking the PDFs.
4. PolyArt: no hit in two searches. Ask whether it is the former name of MMArt. If MMArt gets a venue, update the bib.
5. Unchecked seed items: published versions of Saleh and Elgammal 2016 (the Int. J. Digital Art History version) and of OmniArt (possibly ACM TOMM); Huckle et al. 2020; StyleDecoupler and MegaStyle (preprints); recent emotion-MLLM work such as EmoCaliber and AICA-Bench, which the notes mention but I did not open; ArtECulture (2608.03358). A search for "multimodal LLM visual emotion benchmark 2025 2026" plus one page read each would settle them.
6. Contradictions to keep visible in the paper: the ArtEmis counts differ by version (CVPR 454,684 on 80,031 against arXiv 439,121 on 81,446). ArtELingo-28 has 2,000 images in the body but about 1,430 by arithmetic. The FI size is 23,308, 23,164 or 21,808 depending on the source. The Elgammal 2018 author order differs between arXiv and AAAI. The affect-caption note gives the wrong arXiv id for ArtEmis v2.0 (it should be 2204.07660).

## Addendum: corrected entries (these replace the two entries above)

Both entries are done. The CVF page for Affection returned 403, so the Affection venue still rests on the note and not on a page I opened.

**1. Affection**
Opened: arXiv abs 2210.01946. It gave the title and four authors with given names. Its comments field shows only the project link and no venue. The CVF CVPR 2023 page returned 403. The CVPR 2023 venue comes from the note 2026-12-01_affect_caption_datasets.md, which cites that CVF URL, and not from a page I opened now.

```bibtex
% verified: https://arxiv.org/abs/2210.01946 (title, authors, 2022-10-04; the page lists no venue); venue CVPR 2023 from note: 2026-12-01_affect_caption_datasets.md (CVF page https://openaccess.thecvf.com/content/CVPR2023/html/Achlioptas_Affection_Learning_Affective_Explanations_for_Real-World_Visual_Data_CVPR_2023_paper.html returned 403 on 2026-11-02)
@inproceedings{achlioptas2023affection,
  author    = {Achlioptas, Panos and Ovsjanikov, Maks and Guibas, Leonidas and Tulyakov, Sergey},
  title     = {Affection: Learning Affective Explanations for Real-World Visual Data},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  eprint    = {2210.01946},
  archivePrefix = {arXiv}
}
```

**2. FindingEmo**
Opened: https://papers.nips.cc/paper_files/paper/2024/hash/08a7229eaba3b35cd8f933ac678f2096-Abstract-Datasets_and_Benchmarks_Track.html. It gave the title, five authors in order, and "NeurIPS 2024, Datasets and Benchmarks Track". Also opened arXiv abs 2402.01355, which has the same five authors and no venue. This page lists "Advances in Neural Information Processing Systems 37" as the volume.

```bibtex
% verified: https://papers.nips.cc/paper_files/paper/2024/hash/08a7229eaba3b35cd8f933ac678f2096-Abstract-Datasets_and_Benchmarks_Track.html (NeurIPS 2024, Datasets and Benchmarks Track, vol. 37) ; https://arxiv.org/abs/2402.01355
@inproceedings{mertens2024findingemo,
  author    = {Mertens, Laurent and Yargholi, Elahe' and Op de Beeck, Hans and Van den Stock, Jan and Vennekens, Joost},
  title     = {{FindingEmo}: An Image Dataset for Emotion Recognition in the Wild},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume    = {37},
  year      = {2024},
  eprint    = {2402.01355},
  archivePrefix = {arXiv}
}
```

The arXiv abstract says 25,000 images, with valence, arousal and an emotion category per image, collected through Prolific. The note's 25,869 (public set) comes from a search listing, not from a page I opened, so quote "about 25,000" or check the PDF before giving the exact number.
