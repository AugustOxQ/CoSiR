# r14 facts brief
Read 2026-11-02. Repository paths below are relative to /project/CoSiR; /data and /tmp paths are absolute. “Inference” marks a derived conclusion. No experiment was run and no deletion was performed. Source: commands used for this brief (file reads, metadata/header reads, directory listings, git log and df only).

Terms: c* = G with graph neighbours changed from 20 to 40, selected in r10; teacher = instructed Qwen3-VL-Embedding-2B image; photo model = c*'s emotion pathway fine-tuned on Affection train photos. Sources: docs/reports/auto/percept/2026-12-28_recipe_sweep.md:7; docs/reports/auto/percept/2027-01-01_photo_finetune.md:7.

## Q1. Backbones

### a. Feature stores and missing coverage
Inventory command: Python pathlib iterated the immediate directories of /data/SSD2/pre_extract/{artelingo_full,affection_local}, printed small progress.json/meta.json/merge.json files, and read only .npy magic/version/header bytes using open('rb'), read(8), header-length read and header read. Large index.json files and feature array bodies were not read. This command is the source for the inventory and header shapes below.

ArtELingo default stores are all in /data/SSD2/pre_extract/artelingo_full/<backbone>/. All seven have 61,402 image rows; train/validation/held roles are 42,969/6,152/12,281 paintings. Coverage of those roles is also enforced by the named-image loader. Sources: inventory command above; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:63; src/test/20261025_recipe_backbones/rb_encoders.py:29.

| Backbone folder | Image dimension | Text rows | Splits covered | Source |
|---|---:|---:|---|---|
| qwen3vl_emb_2b | 2048 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/progress.json:1 |
| siglip2_so400m | 1152 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/siglip2_so400m/progress.json:1 |
| siglip_so400m | 1152 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/siglip_so400m/progress.json:1 |
| clip_l14 | 768 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/clip_l14/progress.json:1 |
| clip_b16 | 512 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/clip_b16/progress.json:1 |
| clip_b32 | 512 | 308,723 | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/clip_b32/progress.json:1 |
| dinov2_large | 1024 | none | train, validation, held | inventory/header command; /data/SSD2/pre_extract/artelingo_full/dinov2_large/progress.json:1 |

Other ArtELingo folders: qwen3vl_emb_2b_aspect_prompts (instructed views), qwen3vl_emb_2b_attention (attention artifacts), roberta_go_emotions and roberta_go_emotions_s2 (text only: 99,999/146,980 rows, dimension 768). Source: inventory/header command above.

| Affection folder under /data/SSD2/pre_extract/affection_local | Coverage/count | Dimension | Source |
|---|---|---:|---|
| qwen3vl_emb_2b | all 85,007 images; train 72,255, validation 4,251, test 8,501; 526,749 text rows | 2048 | header command; src/test/20261031_photo_finetune/pf_paths.py:36; src/test/20261031_photo_finetune/pf_paths.py:49 |
| clip_l14_valtest | validation + test, 12,752 images; train absent | 768 | /data/SSD2/pre_extract/affection_local/clip_l14_valtest/meta.json:1; src/test/20261030_affection_confirmation/ac_extract.py:37 |
| qwen3vl_emb_2b_aspect_prompts/emotion_focus | validation + test, 12,752 images; artwork wording | 2048 | /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus/meta.json:1 |
| qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo | validation + test, 12,752 images; photo wording | 2048 | /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo/meta.json:1 |
| qwen3vl_emb_2b_aspect_prompts/emotion_focus_train | train, 72,255 images before near-copy cut | 2048 | /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_train/meta.json:1 |
| qwen3vl_emb_2b_aspect_prompts/emotion_focus_txt | train + validation explanations, 474,051 rows | 2048 | /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_txt/meta.json:1; src/test/20261031_photo_finetune/pf_extract.py:5 |

Inference from this inventory: no default ArtELingo image extraction is missing for SigLIP 2, SigLIP 1 or CLIP L/14 c* runs. For photo fine-tuning, both SigLIP image stores are missing for all Affection splits; CLIP L/14 train images are missing. The Qwen teacher and explanation views already exist. Changing code and obtaining matching c* checkpoints are separate from having feature stores. Sources: inventory command; src/test/20261031_photo_finetune/pf_paths.py:23.

### b. Earlier held encoder results and r11
The 10-25 tests used R6-5: remove own-caption loss terms, direct teacher weight 0.25 (share 1/5). The 10-26 encoder extension used the same recipe, not G or c*. Sources: docs/reports/auto/percept/2026-12-24_recipe_backbones.md:64; docs/reports/auto/percept/2026-12-24_recipe_backbones.md:83; docs/reports/auto/percept/2026-12-25_encoders_genre.md:119.

All following rows are emotion AMI on 12,281 held paintings, five seeds. “Sample/full” refers to training scale, not the held set. The strongest comparator was Qwen PercepT core B1's own reading: 0.0855 sample, 0.0874 full; matched own-encoder arms were included in the maximum. Sources: docs/reports/auto/percept/2026-12-24_recipe_backbones.md:92; docs/reports/auto/percept/2026-12-24_recipe_backbones.md:104; docs/reports/auto/percept/2026-12-25_encoders_genre.md:58.

| Encoder; recipe R6-5 throughout | Sample AMI; difference [95%]; verdict | Full AMI; difference [95%]; verdict | Source |
|---|---|---|---|
| SigLIP 2 | .1020; +.0165 [.0110,.0177]; GO | .1024; +.0150 [.0096,.0163]; GO | docs/reports/auto/percept/2026-12-24_recipe_backbones.md:97; docs/reports/auto/percept/2026-12-24_recipe_backbones.md:100 |
| CLIP B/32 | .0901; +.0046 [-.0006,.0057]; inconclusive | .0885; +.0011 [-.0043,.0025]; inconclusive | docs/reports/auto/percept/2026-12-24_recipe_backbones.md:98; docs/reports/auto/percept/2026-12-24_recipe_backbones.md:101 |
| Qwen | .1045; +.0190 [.0137,.0201]; GO | .1029; +.0154 [.0100,.0169]; GO | docs/reports/auto/percept/2026-12-24_recipe_backbones.md:99; docs/reports/auto/percept/2026-12-24_recipe_backbones.md:102 |
| CLIP B/16 | .0944; +.0089 [.0036,.0101]; inconclusive | .0945; +.0070 [.0016,.0085]; inconclusive | docs/reports/auto/percept/2026-12-25_encoders_genre.md:64 |
| CLIP L/14 | .0992; +.0138 [.0083,.0150]; GO | .0996; +.0122 [.0068,.0136]; GO | docs/reports/auto/percept/2026-12-25_encoders_genre.md:67 |
| SigLIP 1 | .1046; +.0191 [.0137,.0202]; GO | .1036; +.0162 [.0106,.0179]; GO | docs/reports/auto/percept/2026-12-25_encoders_genre.md:70 |
| DINOv2-large | .0814; -.0041 [-.0094,-.0030]; NO-GO | not run after sample NO-GO | docs/reports/auto/percept/2026-12-25_encoders_genre.md:56; docs/reports/auto/percept/2026-12-25_encoders_genre.md:73 |

GO required difference ≥ .0100 and positive interval lower bound. These are k-means claims dependent on Qwen training views; they do not establish encoder-independent training. Sources: docs/reports/auto/percept/2026-12-24_recipe_backbones.md:92; docs/reports/auto/percept/2026-12-25_encoders_genre.md:22.

A non-Qwen c* (40-neighbour) run was not found in the specified rounds/reports. The c* report restricts its claim to Qwen and lists SigLIP 2/SigLIP 1/CLIP L/14 tests as future work. Sources: docs/reports/auto/percept/2026-12-28_recipe_sweep.md:33; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:38.

r11 did report CLIP L/14 and SigLIP 2 rows. They were frozen plain image embeddings, not c* students; CRL was a separate CLIP L/14 image–value-text cosine representation. Stage-5 emotion/style/genre mAP: CLIP .2479/.2732/.4981; SigLIP 2 .2443/.2576/.4968; CRL .2554/.2816/.5033. Sources: src/test/20261029_retrieval/rt_data.py:227; docs/reports/auto/percept/2026-12-29_retrieval.md:153.

### c. Inputs and comparators for a student on X
Encoder X supplies its default image vector to the student's image network. Qwen supplies the default caption mean to the caption network, instructed aspect graphs and instructed image teacher targets; output vectors stay 2048-dimensional. Only the image network's first layer changes width in the earlier cross-encoder tests. Source: docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md:108.

The graph code uses instructed emotion caption means E, instructed style/genre images, and default Qwen caption/image content views D/I. Emotion and style subtract a content-similarity correction; genre uses mutual neighbours of the instructed genre image. Source: src/test/20261027_recipe_sweep/rs_graphs.py:93.

Comparators: X's frozen image clustered directly; X's matched PercepT core P3 (Qwen Stage-1 topics retained, Stage 2 retrained on X images); instructed Qwen teacher; and the other eligible PercepT entries. The held claim's deciding maximum was over PercepT entries including the matched arm; frozen X and teacher were reported separately. Source: docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md:117.

### d. Encoder configuration and hard-coded paths
| Code | Does it accept another encoder as used? | Relevant keys, paths and hard-coding | Source |
|---|---|---|---|
| r10 recipe sweep | No encoder knob/CLI in its 13-setting grid; full trainer resolves fixed Qwen | rs_full.load_data() passes rg_full.ENCODER; that constant is qwen3vl_emb_2b. The inherited rb block records encoder/width/recipe; image store is PRE/artelingo_full/<encoder>. | src/test/20261027_recipe_sweep/rs_full.py:233; src/test/20261027_relational_genre/rg_full.py:40; src/test/20261025_recipe_backbones/rb_student.py:112 |
| Earlier generic student underneath r10 | Yes, --encoder; width from rb_encoders.WIDTH | full_images reads img.npy/index.json by painting name; train_cohort uses width. Base widths: Qwen 2048, SigLIP2 1152, CLIP B/32 512, DINO 1024. | src/test/20261025_recipe_backbones/rb_student.py:498; src/test/20261025_recipe_backbones/rb_student.py:154; src/test/20261025_recipe_backbones/rb_encoders.py:18; src/test/20261025_recipe_backbones/rb_encoders.py:52 |
| r13 photo training | No encoder setting in GRID/BASE | I/D/T tensors explicitly require width 2048; model uses c* factory; checkpoint root is Qwen c* full/weights/k11=40. STORE and instructed views resolve to affection_local Qwen. | src/test/20261031_photo_finetune/pf_train.py:54; src/test/20261031_photo_finetune/pf_train.py:80; src/test/20261031_photo_finetune/pf_paths.py:23; src/test/20261031_photo_finetune/pf_paths.py:36; src/test/20261031_photo_finetune/pf_paths.py:84 |
| r11 retrieval | Fixed named arms; array metric/read-out functions accept matrices | raw maps clip/siglip2/plain to fixed stores, dimensions 768/1152/2048. cstar reads fixed r10 k11=40 outputs of width 2048; instructed loader checks Qwen model/revision. No SigLIP1 arm in ARMS. | src/test/20261029_retrieval/rt_data.py:19; src/test/20261029_retrieval/rt_data.py:149; src/test/20261029_retrieval/rt_data.py:176; src/test/20261029_retrieval/rt_data.py:245 |
| r12 Affection test | Fixed c*, plain Qwen, two teacher views, CLIP and CRL/PercepT paths | c* export calls rb_student.infer(...,2048,...); data/views are bound to affection_local, fixed keys/sizes and Qwen identity. CLIP view width 768 is separately supported. | src/test/20261030_affection_confirmation/ac_export.py:108; src/test/20261030_affection_confirmation/ac_paths.py:29; src/test/20261030_affection_confirmation/ac_extract.py:34 |

Hard-coded Qwen elements also include caption inputs and teacher width 2048 in rg_full, and an nn.Linear(2048,width) construction in the sweep's alternate trunk implementation. Thus a generic loader beneath r10 does not make the current sweep runner encoder-configurable. Sources: src/test/20261027_relational_genre/rg_full.py:229; src/test/20261027_recipe_sweep/rs_knobs.py:114. Last sentence: inference.

### e. Cost per training run
c*, one full-scale seed-0 three-head cohort: 316.97 s = 5.28 min; seeds 0–4: 316.97/243.47/212.37/296.09/241.31 s = 3.54–5.28 min. Device cuda; this stage ran locally on RTX 3090. These are cohort wall times, not sums of per-head timers. Sources: /data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/train/k11=40/run__s0.json:104; command: Python json.load of run__s*.json in that directory, printing seconds and host; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:78; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:81. Minute conversions: inference (seconds/60).

Photo model, final seed 0: 138.36 s = 2.31 min; five final runs: 138.36/141.83/142.57/142.18/142.09 s = 2.31–2.38 min each, local RTX 3090. Stage 2 as a whole took 910 s including other work. Sources: /data/SSD2/cosir_r13/20261031_photo_finetune/outputs/runs/final/s0/run.json:121; command: Python json.load of outputs/runs/final/s*/run.json printing seconds/history length; src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:85; src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:96. Minute conversions: inference.

Peak GPU memory per full c*/photo run: not found in the inspected logs/records. A6000 capacity was 48 GB, not measured per-run consumption; photo smoke explicitly says peak memory was not measured. Sources: command rg -n 'peak|memory|MiB|GiB|VRAM|GB' on the two round logs and rs_node_log.md; src/test/20261027_recipe_sweep/rs_node_log.md:32; src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:39.

## Q2. The large sweep

### a. r10 grid, pick and fixed settings
Stage 1 changed each of 13 knobs separately at three non-reference values (39 configurations plus G); G-repeat was an additional repeat. Stage 2 added 18 joint points; node verification covered 59 runs including the repeat. Source: docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:104; src/test/20261027_recipe_sweep/rs_node_log.md:35; src/test/20261027_recipe_sweep/rs_node_log.md:46; src/test/20261027_recipe_sweep/rs_node_log.md:54.

| Axis | G reference | Three stage-1 values | Source |
|---|---|---|---|
| Emotion relational weight | 0 | 2,8,16 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:108 |
| Style relational weight | 0 | 2,8,16 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:109 |
| Genre relational weight | 8 | 4,12,16 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:110 |
| Relational teacher temperature | .02 | .015,.025,.035 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:111 |
| Direct teacher weight | .25 | .0625,.125,.5 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:112 |
| Learning rate | .001 | .0003,.0005,.002 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:113 |
| AdamW weight decay | .05 | 0,.01,.2 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:114 |
| Batch graph pairs | 512 | 256,1024,2048 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:115 |
| Shared trunk width | 1024 | 512,1536,2048 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:116 |
| Dropout | .1 | 0,.2,.3 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:117 |
| Mutual graph neighbours | 20 | 10,30,40 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:118 |
| Buddy temperature | .05 | .03,.07,.1 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:119 |
| Stopping patience | 15 | 5,10,30 | docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:120 |

The actual joint grid used neighbours 20/40 × teacher temperature .02/.035 × style relational weight 0/2/8 × trunk width 1024/512, reusing earlier points. Source: src/test/20261027_recipe_sweep/rs_node_log.md:43.

Score M = mean across emotion/style/genre of AMI gain over same-machine G divided by r=(.0046,.0082,.0128). Stage-1/2 guard: no aspect loses more than half its r. Positive counting runs supply the top four knobs and up to two positive levels plus G; top three configurations are confirmed locally on seeds 1–3. Sources: src/test/20261027_recipe_sweep/DECISION_RULE.md:107; src/test/20261027_recipe_sweep/DECISION_RULE.md:121; src/test/20261027_recipe_sweep/DECISION_RULE.md:223.

Final qualification: each aspect's gain lower bound > −r and mean-gain lower bound > 0; largest unrounded three-seed mean gain wins (ties follow TOP order). Pick: k11=40 only, M=.683; all other knobs remain G's settings. Sources: src/test/20261027_recipe_sweep/DECISION_RULE.md:253; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:99.

Held fixed in this sweep: encoder/default inputs, teacher model/instruction texts, AdamW optimizer family, 200-epoch maximum, output width 2048 and clustering K=20. Structural components such as content correction, instructed graphs, caption endpoints and trunk sharing were removed separately in ablations, not searched jointly in the knob grid. Sources: docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:101; docs/superpowers/specs/2026-10-27-recipe-sweep-design.md:130. “Never swept” across the entire project's history: not established by these round records.

### b. r13 nine photo rows, pick and stopping
All rows below use validation seed 0; row 0 is a frozen reference outside the nine training rows. Score = min((AMI−teacher AMI)/.00343079, (mAP−teacher mAP)/.00101032); teacher validation AMI/mAP .1719/.3044; sweep read-out center+k10. Source: docs/reports/auto/percept/2027-01-01_photo_finetune.md:76.

| Row | Change from c* fine-tune baseline | AMI | mAP | Score | Source |
|---|---|---:|---:|---:|---|
| 0 reference | frozen c* | .1427 | .2856 | -18.65 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:80 |
| 1 | none: c* start, lr .001, k40, correction on, teacher .25 | .1769 | .3227 | 1.45 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:81 |
| 2 | lr .0003 | .1873 | .3225 | 4.49 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:82 |
| 3 | lr .0001 | .1841 | .3240 | 3.55 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:83 |
| 4 | k20 | .1857 | .3171 | 4.04 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:84 |
| 5 PICK | k80 | .1905 | .3280 | 5.42 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:85 |
| 6 | correction off | .1570 | .2932 | -11.06 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:86 |
| 7 | teacher weight 1 | .1806 | .3147 | 2.52 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:87 |
| 8 | random start | .1801 | .3227 | 2.40 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:88 |
| 9 | lr .0003 + k80 + teacher 1 | .1837 | .3167 | 3.43 | docs/reports/auto/percept/2027-01-01_photo_finetune.md:89 |

Pick lead over row 2: .93 score units, .0032 AMI, less than half the photo model's validation seed range .00707. Device variation in one GPU/CPU comparison was about .6 score units; the tie margin was .5. Sources: docs/reports/auto/percept/2027-01-01_photo_finetune.md:91; src/test/20261031_photo_finetune/pf_paths.py:71.

All five final runs selected best epoch 1; validation loss rose from epoch 2. Budget 200 epochs, patience 15, strict loss improvement, batch-weighted validation loss on fixed graph-pair order; epoch 0 excluded. The saved histories have 16 epochs each, so “best epoch 1” does not mean training executed one epoch. Sources: docs/reports/auto/percept/2027-01-01_photo_finetune.md:93; src/test/20261031_photo_finetune/pf_paths.py:81; src/test/20261031_photo_finetune/pf_train.py:96; src/test/20261031_photo_finetune/pf_train.py:140; command: Python json.load of five final run.json files printing len(history).

### c. Previously named late analysis
On 10-27 the user named two analysis lines: ablation of useful components, and distributions/statistics of what the model learned; other passing encoders were to follow the final recipe. Source: docs/superpowers/direction_log.md:227.

On 10-29 the recorded late plan was a large-scale sweep, deep analysis and qualitative results between 11-10 and the paper deadline, with the main recipe frozen 11-10 and the late sweep serving analysis/robustness only. “c* without the caption endpoints” was explicitly assigned to r13 analysis. Source: docs/superpowers/direction_log.md:174.

Already recorded held ablations: removing direct teacher raised emotion by .0035; removing caption endpoints raised style by .0072. Removing relational loss lowered genre by .0440; instructed graphs, buddy terms and content correction carried emotion/style. These ablations decided nothing. Sources: docs/superpowers/direction_log.md:162; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:26; docs/reports/auto/percept/2026-12-28_recipe_sweep.md:50.

On 10-31 photo training took r13, moving the large sweep to r14 after the freeze. The later r14 entry moved it before the freeze, alongside frozen-recipe backbone tests; whether it could replace c* or serve analysis only remained open pending rerun costs. These are recorded directions, not choices made in this brief. Sources: docs/superpowers/direction_log.md:93; docs/superpowers/direction_log.md:68.

### d. Throughput
Observed node402 stage-1 training plus scoring: about 3 min/run per RTX A6000; three GPUs, each 48 GB. Inference: approximately 20 runs/hour/GPU, or 60/hour across three GPUs, excluding synchronization and separate verification. Stage 1 launched 17:08–17:10 and shards ended about 17:55; the plan finished 18:29 after verification. Per-run memory usage: not found. Source: src/test/20261027_recipe_sweep/rs_node_log.md:32; src/test/20261027_recipe_sweep/rs_node_log.md:35; src/test/20261027_recipe_sweep/rs_node_log.md:37; src/test/20261027_recipe_sweep/rs_node_log.md:45.

## Q3. Frozen photo model on EmoSet-118K

README facts: 118,102 labelled images; eight classes amusement, anger, awe, contentment, disgust, excitement, fear, sadness; info.json and train/val/test.json; image and annotation subfolders per emotion. Split sizes and 11.3 GB zip size are not stated in this saved README. The Dropbox link is present. Source: /tmp/claude-0/-project-CoSiR/231885ae-9168-4719-bb17-fc882c65ec41/scratchpad/emoset_check/readme.md:5; /tmp/claude-0/-project-CoSiR/231885ae-9168-4719-bb17-fc882c65ec41/scratchpad/emoset_check/readme.md:7; /tmp/claude-0/-project-CoSiR/231885ae-9168-4719-bb17-fc882c65ec41/scratchpad/emoset_check/readme.md:17; /tmp/claude-0/-project-CoSiR/231885ae-9168-4719-bb17-fc882c65ec41/scratchpad/emoset_check/readme.md:44; command: full numbered README read (167 lines).

### a. Affection r12/r13 protocol
AMI: cluster each evaluated split's image representations with MiniBatchKMeans, K=20, n_init=3, batch 4096, one thread, random states 0/1/2; average AMI across states, then model seeds. Five c*/photo seeds; frozen plain/teacher views have no training-seed ensemble. Sources: docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:135; src/test/20261030_affection_confirmation/ac_cluster.py:39; docs/reports/auto/percept/2027-01-01_photo_finetune.md:55.

mAP: each split's images serve as queries and gallery; remove query itself; relevant = same plurality emotion label. AP sums precision at relevant ranks divided by all relevant gallery items; mAP averages query AP (not class-balanced). Cosine ranking is stable in key order; expansion adds the top k neighbours to the query and normalizes before reranking. Sources: docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:138; src/test/20261029_retrieval/rt_metrics.py:21; src/test/20261029_retrieval/rt_metrics.py:31.

Read-out menu: none/centering/PCA whitening (64,256,1024 dimensions, clipped to input width and deduplicated), each crossed with query expansion k=0/1/3/5/10: up to 25. Whitening epsilon = 1e-6 × largest covariance eigenvalue. Source: src/test/20261029_retrieval/rt_readout.py:15; src/test/20261029_retrieval/rt_readout.py:50; src/test/20261029_retrieval/rt_readout.py:74.

Read-out selection uses validation cross-fit: sorted-key even/odd halves; fit map on one half, score retrieval within the other, swap and average; seeded arms averaged across seeds. Score is emotion mAP gain over R0 divided by validation seed range; choose maximum with .001 score tie tolerance, preferring R0/no expansion, then simpler transforms/smaller k/larger whitening width. Finally fit chosen map on all validation features and apply to test. Sources: src/test/20261029_retrieval/rt_readout.py:20; src/test/20261029_retrieval/rt_readout.py:44; src/test/20261029_retrieval/rt_readout.py:111; src/test/20261029_retrieval/rt_readout.py:119; src/test/20261029_retrieval/rt_readout.py:169; docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:248.

r12 picked artwork wording in both tasks: validation AMI .1719 vs photo .1629, cross-fit mAP .3044 vs .2995. Wordings picked separately per task; photo only wins selection if better by >.001 r. r13 retained r12's teacher/CRL picks. Sources: src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md:171; docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:253; docs/reports/auto/percept/2027-01-01_photo_finetune.md:116.

Artwork prompt: “Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion.” Photo variant substitutes “photo” for both occurrences of “artwork”. Sources: /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus/meta.json:3; /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo/meta.json:3.

r12 read validation/test and did not train on Affection. r13 removed a train photo and its explanations when maximum cosine to any validation or test photo was ≥.85. Cosine used float32 normalized default Qwen image features joined by key, blockwise CPU matrix products; labels were not read. Removal: 2,328 of 72,255, leaving 69,927 fit photos. Sources: docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:117; src/test/20261031_photo_finetune/pf_nearcopy.py:1; src/test/20261031_photo_finetune/pf_nearcopy.py:19; src/test/20261031_photo_finetune/pf_nearcopy.py:55; docs/reports/auto/percept/2027-01-01_photo_finetune.md:72.

Affection's ninth class “something else” was retained, not mapped away: alphabetical plurality tie-break; test had 198 such labels. Ties-dropped diagnostics remove tied photos, not this class as a rule. Sources: docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:130; docs/reports/auto/percept/2027-01-01_photo_finetune.md:158; docs/reports/auto/percept/2027-01-01_photo_finetune.md:189.

Uncertainty: 2,000 paired image/seed bootstrap draws for AMI with partitions fitted once; 10,000 query/seed draws for retrieval, 95% intervals. r13 starts each final run from matching c* seed 0–4 with train seed 100+s; no sweep run enters held test. Sources: docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:135; docs/superpowers/specs/2026-10-30-affection-confirmation-design.md:138; docs/reports/auto/percept/2027-01-01_photo_finetune.md:55; docs/reports/auto/percept/2027-01-01_photo_finetune.md:57.

### b. Extraction and frozen inference inputs
extract_features.py has an ADAPTERS registry and --dataset choices=sorted(ADAPTERS), not an arbitrary dataset-path interface. Registry: artelingo28, artelingo_full, semart, impressions, affection_local, affection_fi, coco_train2014, genecis_coco, genecis_vg_crops; EmoSet is not present. ALLOWED further restricts backbone per dataset. Sources: scripts/extract_features.py:453; scripts/extract_features.py:622; scripts/extract_features.py:653.

Inference: adding EmoSet to this extractor entails supplying a Spec loader/metadata adapter, registering its dataset name and allowed backbone(s). Current affection_local ALLOWED lists CLIP B/32, Qwen and SigLIP2; CLIP L/14 and SigLIP1 are listed for ArtELingo but not Affection. Sources: scripts/extract_features.py:97; scripts/extract_features.py:76; scripts/extract_features.py:453.

Instructed image extraction used ac_extract.py extract --view emotion_focus --keys valtest --out <folder>, its photo variant, and pf_extract.py images for train. pf_extract.py texts extracted instructed explanation rows for train+validation; these reuse pinned Qwen revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda. Batches: images 32, texts 128. Sources: src/test/20261030_affection_confirmation/ac_extract.py:15; src/test/20261031_photo_finetune/pf_extract.py:3; src/test/20261031_photo_finetune/pf_extract.py:16; src/test/20261031_photo_finetune/pf_extract.py:20; /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_train/meta.json:5.

Photo-model inference feeds only default Qwen image features, normalized float32 [N,2048], into the checkpoint's image network; outputs float16 [N,2048]. It does not read test captions, graphs or instructed test images in that path. Teacher comparison separately requires an instructed image view. Picked retrieval additionally uses the validation-fitted centering and k10 expansion. Sources: src/test/20261031_photo_finetune/pf_export.py:33; src/test/20261031_photo_finetune/pf_export.py:75; docs/reports/auto/percept/2027-01-01_photo_finetune.md:93.

Training, in contrast, uses I=default images, D=default explanation mean, E=instructed explanation mean and T=instructed emotion images; E/D/I construct graphs and I/D/T enter the training loss. Sources: src/test/20261031_photo_finetune/pf_data.py:229; src/test/20261031_photo_finetune/pf_graphs.py:18; src/test/20261031_photo_finetune/pf_train.py:80.

### c. Disk
| Mount | Filesystem | Size | Used | Available | Use | Source |
|---|---|---:|---:|---:|---:|---|
| /data/SSD | /dev/sdc | 3.6T | 1.1T | 2.4T | 31% | command: df -h /data/SSD /data/SSD2 |
| /data/SSD2 | /dev/sda | 3.6T | 1.7T | 1.8T | 48% | same command |

## Q4. Timing of recent loops
Chat timestamps are registry launch times, not measured active work. Auto-report endpoints below use the first-add git commit, not the sequence-number date in filenames. Sources: docs/superpowers/handoffs/chats.tsv:193; docs/superpowers/handoffs/chats.tsv:202; command: git log --diff-filter=A --format='%h %aI %cI' -- <the r12 and r13 auto-report paths>.

| Loop | Decide start → build start | Build start → run-chat start | Run queue wall time | Auto report committed | Decide start → auto report | Source |
|---|---|---|---|---|---|---|
| r12 | Oct 29 23:15 → Oct 30 21:19 (22h04m) | Oct 30 21:19 → Oct 31 00:56 (3h37m) | Oct 31 00:57–03:33, 2h36m | Oct 31 03:41:35 +01:00, 354257c | 28h26m35s | docs/superpowers/handoffs/chats.tsv:193; docs/superpowers/handoffs/chats.tsv:194; docs/superpowers/handoffs/chats.tsv:199; src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md:161; src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md:183; git log command above |
| r13 | Oct 31 03:49 → Nov 1 15:34 (35h45m) | Nov 1 15:34 → 18:42 (3h08m) | Nov 1 18:43–21:24, 2h41m | Nov 1 21:36:35 +01:00, 79f0c49 | 41h47m35s | docs/superpowers/handoffs/chats.tsv:202; docs/superpowers/handoffs/chats.tsv:204; docs/superpowers/handoffs/chats.tsv:209; src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:83; src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:102; git log command above |

Durations calculated by timestamp subtraction: inference. r12 auto report was already done by its 03:45 handoff; r13 both reports were committed by its 21:42 handoff. Sources: docs/superpowers/handoffs/2026-10-31-r12-run-2-handoff.md:9; docs/superpowers/handoffs/2026-11-01-r13-run-2-handoff.md:4.

r12 step wall times: FI extraction 1,553 s; artwork/photo teacher extraction 656/652 s; CLIP 92 s; total GPU steps about 51 min. CPU: c* export 35 s; stage 0a 1,215 s; stage 0b 28 s; stage 1 306 s; stage 2 184 s; stage 3 168 s; corresponding independent checks 28/1,110/1,430/1,585 s; diagnostics 168 s. Source: src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md:162.

r13 step wall times: image extraction 3,630 s; text extraction 2,142 s; stage 0a 646 s; nine-row stage 1 1,840 s; five-run stage 2 910 s; verdict 95 s; independent derivation/compare 266/4 s; diagnostics 115 s. Individual final training times are listed in Q1e. Source: src/test/20261031_photo_finetune/20261031_photo_finetune_log.md:90.

Removal list for this task: none; no files need removal to produce this brief. Historical cleanup candidates, not a new recommendation: r12 pre-FI store 1.8 GB and FI-only store 626 MB; r13 build scratch/worktrees about 12.6 GB. Their current presence was not checked outside the requested feature-store listing. Sources: src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md:185; docs/superpowers/handoffs/2026-11-01-r13-run-2-handoff.md:69.

## Gaps
- EmoSet train/validation/test sizes and zip byte size: not found in the supplied README (full numbered read, lines 1–167).
- Measured peak GPU memory for full c* or photo training, and full-scale A6000 c*/photo timing: not found in inspected round logs/run records; node timing cited above is sample-scale training plus scoring (src/test/20261027_recipe_sweep/rs_node_log.md:37).
- Non-Qwen c* checkpoints/results and photo models on SigLIP2/SigLIP1/CLIP L/14: not found in the specified reports/rounds; these tests are named as future work (docs/reports/auto/percept/2026-12-28_recipe_sweep.md:38; docs/superpowers/direction_log.md:68).
- A defined r14 large-sweep grid or a settled authority to replace c*: not found in the inspected specifications/handoffs; the current direction entry explicitly leaves its role open (docs/superpowers/direction_log.md:70).
