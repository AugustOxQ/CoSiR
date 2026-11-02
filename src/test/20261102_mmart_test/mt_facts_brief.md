# MMArt facts brief

Sources below are repo-relative `path:line` unless absolute. Abbreviations: **S** = `docs/superpowers/specs/2026-11-02-backbones-sweep-design.md`; **L** = `src/test/20261102_backbones_sweep/bs_run_log.md`; **P** = `src/test/20261102_mmart_prep/20261102_mmart_prep_log.md`; **B** = `src/test/20261102_backbones_sweep/`; **M** = `src/test/20261102_mmart_prep/`. No extraction outputs were inspected. Nothing is proposed for removal.

## 1. Labels and coverage

- **Count source C (executed, CPU, pandas):** environment `OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, interpreter `/root/miniconda3/envs/CoSiR/bin/python`; the following reproduces the counts below (before any copy removal):
```python
import pandas as pd
D = pd.read_parquet('/data/SSD/mmart/mmart_index.parquet')
R = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet')
print(D.columns.tolist(), R.columns.tolist())
for split in ['new', 'val']:
    t = D[D.split == split]; print(split, len(t))
    for c in ['emotion', 'style', 'genre_name']:
        print(c, t[c].fillna('').replace('', '(missing)').value_counts().sort_index().to_dict())
    print('missing genre IDs', (t.genre < 0).sum(), 'exact copies', t.exact_dup_trainval.sum())
```
- Raw label fields: `dominant_emotion`, `style`; no raw genre. Index fields: `emotion` (raw dominant emotion, empty string means missing), `style`, `genre` (integer; negative means unknown), `genre_name`; also `artelingo_emotion`/`artelingo_style` for shared paintings, not new-test ground truth. Source: C; M`mp_index.py:124`.
- **New test: 16,782 paintings. Emotion counts:** amusement 875; anger 422; awe 1,293; contentment 3,475; disgust 1,420; excitement 650; fear 2,907; sadness 4,576; something else 974; missing 190. Thus 16,592 labelled, 9 observed classes. Source: C (labelled total = total minus missing).
- **New-test style counts:** abstract_expressionism 625; art_nouveau_modern 888; baroque 989; color_field_painting 372; cubism 557; early_renaissance 300; expressionism 1,461; fauvism 172; high_renaissance 292; impressionism 2,801; mannerism_late_renaissance 267; minimalism 310; naive_art_primitivism 481; northern_renaissance 614; pop_art 339; post_impressionism 1,431; realism 2,578; rococo 438; romanticism 1,591; ukiyo_e 276. All 16,782 labelled, 20 classes, missing 0. Source: C.
- **New-test genre counts:** abstract_painting 1,158; cityscape 967; genre_painting 2,253; illustration 359; landscape 2,546; nude_painting 420; portrait 3,151; religious_painting 1,405; sketch_and_study 884; still_life 596; missing 3,043. Thus 13,739 labelled, 10 classes. Source: C.
- **Style sets differ:** MMArt has 20 classes versus ArtELingo's 27; shared-painting agreement 55,681/57,452 (96.9%). Examples of merges: Pointillism → post_impressionism; Contemporary/New Realism → realism; Synthetic/Analytical Cubism → cubism; Action painting → abstract_expressionism. Source: P:35–37.
- **Genre set is the same project WikiArt set**, added to MMArt via `src.data.wikiart_genre.load_wikiart_genre`, not supplied by MMArt. Its ArtGAN genre CSVs and 10 class names are the source for both datasets; unknown includes conflicting CSV assignments. Sources: M`mp_index.py:39`, :112, :125; `src/data/wikiart_genre.py:1`, :15–32.
- **Validation: 5,769 shared ArtELingo val paintings**, with the same MMArt emotion/style and added WikiArt genre columns. Emotion: 5,720 labelled, 49 missing, all 9 classes; style: 5,769 labelled, 0 missing, all 20 classes; genre: 4,679 labelled, 1,090 missing, all 10 classes. Source: C; split construction M`mp_index.py:117–120`.
- **Val emotion counts:** amusement 273; anger 168; awe 519; contentment 1,310; disgust 472; excitement 272; fear 995; sadness 1,454; something else 257. Source: C.
- **Val style counts:** abstract_expressionism 219; art_nouveau_modern 333; baroque 339; color_field_painting 112; cubism 214; early_renaissance 118; expressionism 486; fauvism 72; high_renaissance 103; impressionism 915; mannerism_late_renaissance 111; minimalism 113; naive_art_primitivism 177; northern_renaissance 208; pop_art 123; post_impressionism 496; realism 857; rococo 161; romanticism 519; ukiyo_e 93. Source: C.
- **Val genre counts:** abstract_painting 390; cityscape 331; genre_painting 756; illustration 148; landscape 885; nude_painting 143; portrait 1,018; religious_painting 489; sketch_and_study 279; still_life 240. Source: C.

## 2. Extracted views and copy checks

- Planned seven views: `qwen_default`, `emotion_focus`, `style_word`, `genre_word` use `Qwen/Qwen3-VL-Embedding-2B`; only the last three are teacher views. `siglip2` uses `google/siglip2-so400m-patch14-384`; `siglip1` uses `google/siglip-so400m-patch14-384`; `clip_l14` uses `openai/clip-vit-large-patch14`. Sources: M`mp_stores.py:16–23`; M`mp_extract.py:114–126`; B`bs_extract.py:171–187`; `src/data/feature_extract.py:20–25`.
- **Coverage by design: all four encoders have default-image views**, with Qwen instructed targets shared across encoders; there are no instructed SigLIP/CLIP teacher views. Actual completion/coverage of the running stores: **not checked**. Val default views reuse ArtELingo stores; val teacher views are unavailable through the current loader and were deferred, as were caption embeddings. Sources: M`mp_stores.py:75–93`; P:66–68.
- Teacher instructions come from ArtELingo held-store metadata, checked against the prompt definitions; Qwen revision is pinned. Source: P:75–77; M`mp_extract.py:119–125`.
- Near-copy comparison: each new painting's default Qwen image against all ArtELingo paintings outside held (train plus val); nearest cosine after temporary float32 L2 normalization. Reference counts asserted: 61,402 total minus 12,281 held = 49,121 train/val. Sources: M`mp_extract.py:236–270`.
- Inclusive exclusion cut is cosine ≥0.85; diagnostic counts at 0.85, 0.90, 0.95, 0.99. Expected output `/data/SSD/mmart/mmart_nearcopy.parquet`; actual results **not checked**. Sources: M`mp_extract.py:23`, :27, :272–280.
- Independently, 34 new paintings have identical image-byte MD5 to ArtELingo train/val, despite different painting names; these are designated for exclusion alongside near-copies. Sources: P:30, :57–60; M`mp_index.py:111`, :127; C.

## 3. Scoring protocol and decision bars

- r14 inherits r10 clustering, r11 retrieval and r12 photo evaluation. Each painting aspect has AMI and mAP cells. Plain image = same encoder's default image; teacher = aspect-instructed Qwen image; matched PercepT = same input encoder with the existing Qwen-topic PercepT mapper. Sources: S:121–128, :178–198.
- Retrieval is **image-to-image, label based**: every labelled painting queries the other labelled paintings in the same scored split; relevance means equal aspect label. Self excluded; stable ties preserve gallery row order; full-ranking AP averaged over queries, then seeds. Missing labels and queries with no relevant gallery item are dropped. It is not text/caption retrieval. Sources: S:187–192; `src/test/20261029_retrieval/rt_metrics.py:21–42`, :75–96; `src/test/20261030_affection_confirmation/ac_stats.py:209–234`.
- K is **fixed at 20**, not selected on validation and not equal to each aspect's class count; MiniBatchKMeans states 0,1,2, mean AMI, model seeds 0–4. EmoSet's K=8 class-count row is descriptive. Sources: S:178–180, :198–201; `src/test/20261030_affection_confirmation/ac_cluster.py:39–54`; B`bs_emoset_score.py:8–10`.
- Validation selects retrieval read-outs and measures candidate seed ranges, not K. Read-outs use r11/r12 cross-fit validation selection; the chosen map is fit on all validation features then applied to test, with query expansion on the test gallery. Teacher wording selection on EmoSet is task-specific. Sources: S:187–214; B`bs_emoset_score.py:11–20`; `src/test/20261030_affection_confirmation/ac_stats.py:209–224`.
- AMI intervals: paired bootstrap with fixed clustering partitions, 2,000 draws, shared sampled paintings, then candidate seed resampling; frozen teacher/plain arms have no seed draw. Matched PercepT seeds are drawn independently from student seeds, with the stronger of its own-topic argmax and 20-cluster readings recomputed in every draw. Sources: S:178–186; B`bs_artclu.py:8–17`.
- mAP intervals: 10,000 paired draws of per-query AP, shared query draws and (when both seeded) shared seed draws; percentile 95% interval. This resamples stored AP, without reranking a bootstrap gallery. Sources: S:190; `src/test/20261029_retrieval/rt_stats.py:45–85`.
- Brief §3.4 quote: “WIN when the lower 95% bound > 0”; LEVEL if not WIN and lower bound > −r; BELOW otherwise. r is candidate max–min across validation seeds, per aspect/task (mAP at R0); no multiplicity correction. Earlier r10 genre LEVEL used 0.0128; r14 uses each encoder's own range. Source: S:211–216.
- **Painting bars:** all three AMI and three mAP cells must WIN over plain image; all three AMI cells must WIN over matched PercepT for “holds”. If plain-image cells all WIN but matched cells miss, “qualified”; otherwise “fails”. Teacher AMI/mAP comparisons are separate per-cell LEVEL-or-WIN claims. Sources: S:340–354.
- **Part c photo bars:** WIN over plain Qwen and teacher in both AMI and mAP = holds; both plain wins plus one teacher WIN and one LEVEL = qualified; otherwise fails. “The verdict stands” after a scientific failure, with next steps discussed, rather than changing the bar. Sources: S:44–47, :296, :359–361.
- Brief §3.5 quote: “seed ranges and read-out picks written and committed before any held number”; development gates precede claim tests. Brief §6 quote: “Binding rule: this section with §3.4 and §3.5”. Sources: S:229–231, :336.
- Execution failure differs from a failed claim: retry a crashed/nonfinite run once with same seed; repeated run failure or failed gate makes the affected verdict inconclusive; other encoders proceed. Sources: S:175–177, :355–361.
- EmoSet near-copies: remove val/test photos whose default-Qwen cosine to any of all 72,255 Affection training photos is ≥0.85, including the 2,328 previously removed training photos; use same cut sets for every arm, without labels. Removed 135 val and 368 test; kept 5,770 val and 17,348 test. Sources: S:167–171; `docs/reports/auto/percept/2027-01-02_backbones_sweep.md:61–62`.

## 4. Existing c* weights and reusable scoring code

- **Checkpoint existence source E (executed):** Python `Path(root/'weights/k11=40').glob('*.pt')` plus `is_file()` for every `{emotion,style,genre}__s{0..4}.pt`. Each root below has all 15 expected files; seed coverage 0,1,2,3,4 for every aspect. This checked existence, not checkpoint contents or hashes.
- Qwen root: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full`; reused c*, not a newly trained r14 Qwen arm. Sources: E; B`bs_paths.py:21`; B`bs_photo.py:343–348`.
- SigLIP 2 root: `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip2`. Source: E.
- SigLIP 1 root: `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1`. Source: E.
- CLIP L/14 root: `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/clip_l14`. Source: E.
- Complete path template: `<root>/weights/k11=40/<aspect>__s<seed>.pt`. Run-log corroboration: each new encoder's 10 training units (c* plus photo) succeeded; 30/30 total. Sources: E; L:37–48; B`bs_cstar.py:216–221`.
- Frozen all-aspect forward inference: `bs_cstar._infer_sides(weights, aspect, w, inputs, device)`; accepts checkpoint, aspect, encoder width, dict such as `{'img': float32[n,w]}`, and CPU device. Builds the trained view, eval/no-grad, returns float16 exports. No captions/teacher inputs needed for image inference. Source: B`bs_cstar.py:321–339`.
- Emotion-specific CPU inference also exists: `bs_photo.infer(state, x, width)`; `export_checkpoint` wraps loading/export. Source: B`bs_photo.py:395–414`, :464–476.
- Dataset-independent scoring pieces: `ac_cluster.partitions`, `ami`, `paired` accept feature rows/partitions and integer labels; `rt_metrics.retrieval` accepts unit float32 rows, labels, optional query expansion; `ac_stats.arm_retrieval` accepts validation/evaluation feature callbacks, read-out, labels and seeds; `rt_stats.paired` accepts aligned per-query AP arrays. Sources: `src/test/20261030_affection_confirmation/ac_cluster.py:44–108`, `ac_stats.py:209–234`; `src/test/20261029_retrieval/rt_metrics.py:75–96`, `rt_stats.py:45–85`.
- Existing r14 orchestration `bs_artclu.score_encoder` / `bs_artret.score_encoder` is ArtELingo-specific; `bs_emoset_score.RealSource` is EmoSet-specific. MMArt needs keyed feature/label adaptation and val exports/read-out records, not simply changing a dataset CLI flag. Sources: B`bs_artclu.py:320–344`; B`bs_artret.py:339–373`; B`bs_emoset_score.py:145–175`. A ready-made MMArt scorer: **not checked**.

## 5. Matched PercepT requirements

- Training recipe: Stage 1 topic model on caption embeddings of ArtELingo **fit paintings**, with caption topics converted into each fit/dev painting's multi-hot targets; Stage 2 image mapper learns those targets. Matched arms reuse the saved Qwen Stage 1 per seed and change Stage 2 image encoder. Sources: `src/test/20261020_claim_test/ct2_percept.py:93–100`, :171–199, :305–314; `src/test/20261025_recipe_backbones/rb_matched.py:1–6`, :57–75, :95–108.
- Existing matched arms in r14: SigLIP 2 M2, SigLIP 1 M8, CLIP L/14 M7; Qwen matched P3/B1 is the prior-round reference. Sources: S:124; B`bs_artclu.py:59–69`.
- **Frozen transfer to MMArt can run without new caption embeddings:** saved Stage 2 mapper + matching normalized image features suffice; topic fit stays ArtELingo's, not MMArt's. Inference returns topic probabilities for own-topic or fixed-cluster scoring. Sources: `src/test/20261025_recipe_backbones/rb_matched.py:144–159`; `src/test/20261030_affection_confirmation/ac_cluster.py:51–54`.
- A *new MMArt topic fit* would require caption embeddings and a specified fit partition; the prep queue deferred caption embeddings. Source: same training code above; P:66–68. Actual mapper-file existence for every seed: **not checked**. Existing `rb_matched.infer` insists on its recorded training device, so CPU portability is not supplied by that wrapper as-is; no mapper inference was run. Source: `rb_matched.py:148–151`.

## 6. CPU scoring cost

- Run log: stage 3 took 3,060 s elapsed; each ArtELingo encoder clustering 390–520 s, retrieval about 120 s, Affection about 90 s; EmoSet part c 1,098 s. CPU scoring used 8 threads. These are logged elapsed durations, not measured aggregate CPU-seconds. Sources: L:60–61, :74–79.
- ArtELingo painting scoring alone: about 510–640 s (8.5–10.7 min) per encoder, excluding its Affection photo scoring, validation picks and extraction; sum of logged clustering and retrieval. Source: L:78–79; held size 12,281, S:155.
- **Rough extrapolation**, same arms/seeds/bootstrap settings, before MMArt exclusions: scale clustering linearly by 16,782/12,281 and retrieval quadratically by that ratio. About 757–935 s (13–16 min) per encoder; four encoders sequentially about 3,027–3,738 s (50–62 min). Arithmetic: `(390..520)*16782/12281 + 120*(16782/12281)**2`; sizes from C and S:155, times L:78–79. Not a benchmark; different label masks/arm count/cache reuse can change it.
- EmoSet 1,098 s (~18 min) covers its emotion-only multi-arm part c on 17,348 test photos; it is not a direct three-aspect MMArt estimate. Sources: L:78–79; report:62. The above estimate excludes validation selection, export inference, independent re-derivation and diagnostics; their MMArt cost is **not checked**.

## 7. Overlap and caveats recorded in preparation

- Same-name split overlap: 40,155 train, 5,769 val, 11,528 held, 16,782 new; train is not a test because c* trained there; val was used for selection. Sources: P:28–30, :57–60; M`mp_index_summary.json:10–15`.
- New does not guarantee image novelty: 34 exact byte copies of train/val; cosine near-copy check targets train **and val**. Actual cross-split near-copy counts remain **not checked** while extraction runs. Sources: P:30, :49–52, :60; M`mp_extract.py:258–274`.
- Metadata/path caveats: 547 shared paintings have another style-folder path; 97 MMArt rows participate in within-MMArt MD5 duplication (rows, not 97 duplicate pairs). Sources: P:32–33; M`mp_index.py:138–145`.
- Emotion caveat: MMArt labels agree with ArtELingo majority emotion on only 43.4% of 11,398 labelled held paintings; prep log says the paper does not further define `dominant_emotion`. User fixed its treatment as an independent label set. Source: P:40–44.
- **Same-artist overlap:** no artist-overlap analysis or artist-disjoint claim is recorded in this prep log; actual overlap **not checked**. The log's cross-split novelty evidence is painting-name, byte-copy and planned feature-neighbor checks, not artist separation. Source: P:24–44, :49–60.

## Open design choices

- Which historical comparison bars and verdict structure bind this MMArt transfer test, and which cells are descriptive.
- Whether to retain fixed-cluster scoring alone or also report label-count cluster diagnostics.
- Whether read-outs and LEVEL ranges transfer from ArtELingo or are picked/measured on MMArt-labelled validation.
- Whether validation teacher extraction is included, given its deferred status.
- Which matched PercepT encoders/readings enter the comparison; frozen ArtELingo-topic transfer versus a separately defined topic refit.
- How missing-label masks, exact/near-copy exclusions, internal duplicates and optional shared-held results are represented in reported populations.
