# Brainstorm brief: can buddy-graph init replace a PercepT-style latent Z?

You are doing an **open-ended research brainstorm**, not implementation. Do not
write or modify any training/model code. Do not schedule or launch anything.
Output is a single markdown file: `src/test/20260922_percept_brainstorm/codex_findings.md`.
Write directly to that file — do not pause to ask for confirmation first.

## Context: our system (CoSiR, Experiment 18, branch `experiment/buddy_prototype_conditioning`)

CoSiR conditions a frozen CLIP embedding for contrastive image-text retrieval.
Read these files yourself for exact mechanics before writing anything:

- `src/model/prototype_bank.py` — the prototype bank module (16 learned key/value
  vectors, softmax attention over them keyed by a query projection).
- `src/hook/train_cosir.py` lines ~325-360 — how the bank is *seeded*: build a
  mutual-KNN "buddy graph" over frozen CLIP image+text features, run community
  detection on it, average each community's CLIP features, coarsen to exactly
  16 communities, project through `query_proj`, and overwrite the bank's 16
  key/value rows with those 16 vectors. This happens ONCE at training start.
- `src/hook/train_cosir.py` lines ~1651-1666 — how the bank is *used* at every
  training step after that: `query_features = 0.5*(img_features+txt_features)`,
  `label_embeddings = model.prototype_bank(query_features)` (softmax attention,
  differentiable, gradients flow into the bank's keys/values every step via the
  normal retrieval loss — there is NO separate clustering loss, NO reconstruction
  loss, nothing pulls the bank's structure toward being a good clustering other
  than whatever the retrieval loss happens to reward).
- `docs/reports/2026-09-16_stage_report_prototype_conditioning.md` — full writeup
  of what we found: with a bug (prototype bank's LR was 1000x too small), the
  space collapsed to ~1 effective dimension, useless for interpretability. After
  fixing the LR and lowering softmax temperature, silhouette improved from
  ~0.1-0.3 (near-random) to ~0.55-0.70 across 3 seeds — real progress — but
  oracle image-to-text Recall@1 dropped from 16.8 to 10.4 (raw unconditioned
  CLIP baseline is 17.8) — a serious, seed-replicated retrieval regression.
  We have NOT resolved this trade-off.

## Context: the paper we're comparing against

"Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from
Vision-Language Data" (arXiv 2606.03345), method = **PercepT**. Two-stage:

**Stage 1 — P-Topic Formation (unsupervised, decoupled from any downstream task):**
- Fused embedding h = weighted combination of CLIP image (768d), CLIP text (768d),
  AND a GoEmotions-finetuned ModernBERT text encoder (768d, explicitly affect-aware) —
  h = (2*h_C' + h_E)/3 where h_C' is a norm-rescaled CLIP factual component and h_E
  is the norm-rescaled emotion component (Eq. 2). This is deliberately built to carry
  affect, not just factual/semantic content.
- An autoencoder (encoder G_E: MLP [500,500,500,2000] -> latent Z, dim=128; symmetric
  decoder G_D) is pretrained 100 epochs on pure reconstruction loss L_R = ||h - h_hat||_2.
- Then K-means finds k0=100 initial cluster centers mu_j in the pretrained Z.
- Then 200 epochs of **Deep Embedded Clustering (DEC)**: iterative self-training —
  E-step computes soft cluster assignment q_ij via a Student's-t kernel on distance
  to mu_j; a sharpened target distribution P is derived from Q; the loss
  L_total = KL(P||Q) + lambda_R * L_R is minimized, jointly pulling Z points toward
  their nearest centroid AND reshaping the encoder so Z stays a valid reconstruction
  of h. This is a **dedicated, ongoing clustering-shaping loss** — not just an init —
  that runs for 200 epochs and is what actually produces tight, separated clusters
  (paper reports going from silhouette ~unclustered to 0.97 final; Figure 2 is a
  before/after tSNE of exactly this transformation).
- Finally, a norm-thresholding heuristic prunes noise centers, leaving ~67 real
  P-Topics out of the original 100.

**Stage 2 — P-Topic Mapping (supervised, on FROZEN Stage-1 pseudo-labels):**
- Cluster assignments from Stage 1 become fixed multi-label targets O ∈ {0,1}^k
  per training image (an image can belong to zero, one, or several topics).
- A separate small network learns to predict topic membership from a NEW image's
  patch embeddings alone (no access to text at inference): attention-pooling over
  patches -> a single pooled vector -> linear projection -> per-topic sigmoid score
  s_i, trained with binary cross-entropy against the frozen Stage-1 labels.
- They ALSO tried a closer analogue to our prototype bank: K learnable "topic vector"
  queries doing multi-head cross-attention against the image's patch
  embeddings-as-keys/values (Appendix A.1) — and this scored WORSE (AUC 0.91) than
  the simpler single-pooling-head + linear multi-label classifier (AUC 0.94).

Results: silhouette 0.97 vs. best baseline (BERTopic) 0.37; mapping AUC 0.94 vs 0.77;
human eval preferred PercepT 58.4% vs BERTopic 19.5%. Dataset: ArtELingo (~1.2M art
images, 28-language captions, emotion labels) and Affection (~500K realistic images,
English emotional captions).

## The question to brainstorm, honestly and skeptically

1. Can buddy-graph initialization functionally substitute for what Z/DEC does in
   PercepT? Where exactly does the analogy hold and where does it break down?
   (Hint at one obvious structural gap: buddy-init is a ONE-SHOT seed of a fixed
   set of centers in frozen raw CLIP space; DEC is a 200-epoch ONGOING loss that
   actively reshapes a *learned, reconstruction-grounded* latent space toward
   clean clusters, decoupled entirely from any downstream task loss. Our
   prototype bank has no clustering-shaping loss at all post-init — its only
   training signal is the retrieval loss. Do you agree this is the likely root
   cause of Experiment 18's silhouette-vs-retrieval trade-off? Push back if you
   see it differently.)
2. Is there a reason our `warmth`/`register` proxy probes came back mostly as
   content detectors rather than real affect signals, in light of PercepT
   deliberately injecting an affect-aware encoder (GoEmotions-tuned text model)
   into its fused embedding h — something CoSiR has never done? Could this
   explain more of our null/weak affect results than any clustering mechanism
   issue?
3. Propose at least 3 concrete, genuinely new directions for CoSiR inspired by
   this paper. You are explicitly told to ignore the currently planned
   experiment list and publication plan — think about what CoSiR *could become*,
   not just how to patch Experiment 18. Be specific: what would change
   architecturally, what new loss terms, what new data/encoders would be needed,
   what's the smallest test that would tell us fast whether a direction is
   promising. Include at least one idea that is a real pivot, not an incremental
   fix.
4. Be honest about downsides/risks of each direction (compute cost, need for new
   encoders/datasets, RedCaps not having emotion labels like ArtELingo, etc.)

Write your full reasoning and conclusions to
`src/test/20260922_percept_brainstorm/codex_findings.md`. Use clear headers.
This is meant to be read and critiqued by a human, so be concrete and avoid
hedging filler — take real positions.
