# CoSiR new-line experiment facts inventory

Repository evidence through r15. Sequence dates are report identifiers. Paths relative to `/project/CoSiR`; “not found” means not located in inspected evidence.

Sources (brief=Results brief): **Rdd** = `docs/reports/auto/percept/2026-12-dd_<suffix>.md`, with suffixes: 06 new_line_candidates; 08 prompted_aspect_views_check; 09 image_transfer_and_word_masking; 10 purified_buddy_pilot; 11 aspect_graph_sheet; 12 aspect_graph_followup; 13 percept_core; 14 percept_core_arms; 15 percept_core_k40; 16 model_sheet; 18 claim_test_stage1; 19 claim_test_stage2; 21 advantages_framing; 22 d2_ablation; 23 attention_maps; 24 recipe_backbones; 25 encoders_genre; 27 relational_genre; 28 recipe_sweep; 29 retrieval; 31 affection_confirmation. **P13** = `docs/reports/auto/percept/2027-01-01_photo_finetune.md`; **B14** = same directory, `2027-01-02_backbones_sweep.md`; **M15** = same directory, `2027-01-03_mmart_test.md`. **L(date, suffix)** = `docs/reports/literature/<date>_<suffix>.md`; **S(date, suffix)** = `docs/superpowers/specs/<date>-<suffix>-design.md`; **DL** = `docs/superpowers/direction_log.md`. **T** = `src/test/`. Section references apply to the named file. All numbers are copied, not recomputed.

AE=ArtELingo; AF=Affection; EM=EmoSet; MM=MMArt; dev=development; val=validation; comparator*=claim comparator. E/S/G = emotion/style/genre; Q = Qwen3-VL-Embedding-2B; S2/S1 = SigLIP 2/1 so400m; L14/B32/B16 = CLIP L/14, B/32, B/16; DN = DINOv2-large. A triple is E/S/G. Model seeds are distinguished from k-means states. Unless stated otherwise clustering is AMI, k=20, three k-means states; learned final rows use seeds 0–4. [R28 What ran; B14 What ran; M15 What ran]

Table abbreviations: AE28=ArtELingo-28; descr.=descriptive; diag.=diagnostic; clust.=clustering. Additional source aliases: **RSlog**=`T/20261027_recipe_sweep/20261027_recipe_sweep_log.md`; **BSlog**=`T/20261102_backbones_sweep/bs_run_log.md`; **PFlog**=`T/20261031_photo_finetune/20261031_photo_finetune_log.md`; **AClog**=`T/20261030_affection_confirmation/20261030_affection_confirmation_log.md`; **RTlog**=`T/20261029_retrieval/20261029_retrieval_log.md`; **MPlog**=`T/20261102_mmart_prep/20261102_mmart_prep_log.md`; **BFacts**=`T/20261102_backbones_sweep/bs_facts_brief.md`; **EGlog**=`T/20261025_encoders_genre/20261025_encoders_genre_run_log.md`; **EmoRun**=`T/20261102_backbones_sweep/bs_emoset_score.py`; **CeilingLog**=`T/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md`.

Spec aliases: r11 spec=S(2026-10-29,retrieval); r14 spec=S(2026-11-02,backbones-sweep); model-sheet spec=S(2026-10-19,stage2-model-sheet).

## A. Results inventory

### A1. Clustering

|Round|Dataset and split|Aspects|Encoder|Method row|Metric|Seeds|Role|Headline number or outcome|Source, section|
|---|---|---|---|---|---|---|---|---|---|
|r10 premise|AE dev, ~19.9k paintings|ESG|B32; CSD; VGG Gram|Plain; mean caption; style descriptors|AMI|3 k-means|diag.|Image 0.067/0.302/0.421; mean text 0.074/0.115/0.330; CSD 0.047/0.315/0.249; VGG 0.034/0.123/0.149 (reordered as ESG)|R06 §2|
|r10 prompt check|AE dev, 5,000 images; 99,999 captions|ESG|Q; B32; CSD|Default/instructed image and caption, ten instructions|AMI k=9/20/40|3 k-means; no model seeds|diag.|Caption E 0.278 instructed vs 0.122 default; image E 0.093 vs 0.069; image S 0.395 vs 0.222; CSD S 0.317|R08 §brief; Results|
|r10 pilots|AE dev, 5,000 images|E|Q|Own-caption student; default-caption control; neighbour transfer|AMI|3 model seeds|diag.|Student 0.074 ± 0.000; control 0.074 ± 0.002; instructed-image input 0.074/control 0.075; nearest-20 teacher average 0.102; below 0.12 bar|R09 §brief|
|r10 pilot|AE dev, 5,000|E|Q|Purified buddy; control; unpurified; hard negatives|AMI|3|ablation|Purified 0.094 ± 0.002 vs control 0.070 ± 0.002; purification +0.004; hard negatives +0.001; fails 0.12|R10 §brief|
|r10 graph sheet/follow-up|AE dev, 5,000|ESG|Q|21 graph rows/aspect; separate heads, default or instructed inputs|AMI, mean image/caption sides|1 initially; top/reference rows 0–2|ablation|Default residual E 0.137 ± 0.002 vs aspect graph 0.118 ± 0.002; S 0.260 ± 0.003 vs 0.213 ± 0.005; image-only G graph 0.428 ± 0.003; prompted residual E/S 0.151/0.284|R11 §brief; R12 §brief (latest reference repeats)|
|r10 first core attempt|AE dev|ESG|Q; B32+RoBERTa|Default graph; other-side graph; prompted inputs; fused baseline|AMI, mean sides|3|ablation|Default graph 0.082/0.163/0.388 vs default embedding 0.089/0.171/0.393; B32+RoBERTa 0.127/0.174/0.240. PercepT blocked by failed regression at this attempt|R13 §brief|
|r10 core arms|AE dev, 5,000|ESG|Q; B32|PercepT P1–P5, automatic count|AMI|0–2|descr.|P2 0.036/0.162/0.217; counts 1,2,3,99; P5 never trains (one topic)|R14 §brief|
|r10 core K40|AE dev, 5,000|ESG|Q; B32|PercepT core P1–P5, fixed K=40|AMI|0–2|descr.|P1 0.053/0.167/0.356; P2 0.066/0.229/0.400; P3 0.077/0.189/0.436; P4 0.054/0.162/0.361; P5 0.074/0.181/0.435|R15 Results §image-side table|
|r10 model sheet|AE dev, 5,000 (G 4,053)|ESG|Q|13 settings (sharing,masks,negatives,distillation,inputs)|AMI|A1,T1,T2,T5,D2,D3,D5: 3; T3,T4,D1,D4,D17,I1: 1|ablation|D2 0.0962/0.3697/0.4540; D5 best G 0.4891; direct-only D1 0.0825/0.2751/0.4366; T1 0.0930/0.3591/0.4603; A1 0.0894/0.3539/0.4574|R16 Results, aspect tables|
|r10 claim stage 1→2|AE held 12,281; G 9,949; sample→full training|ESG|Q|D2|AMI|0–4|claim E; descr. SG|Full 0.1008/0.3685/0.4438; E GO +0.0133 [+0.0081,+0.0146]; earlier sample E0.1001,GO|R19 Results/Analysis; R18 Results|
|r10 claim stage 1→2|AE held; sample→full training|ESG core; E paper|Q; B32 patches/global|PercepT core; strict paper; paper K40 P1/P4|AMI own assignment and k20|0–4|comparator* E; descr. SG|Full E core0.0799/0.0874(k20/own);paper P1 0.0699/0.0711,P4 0.0644/0.0707;strict10/10collapsed. Sample core0.0779/0.0855;paper0.0693/0.0707,0.0652/0.0687|R19 Results; R18 Results|
|r10→r14|AE held|ESG|Q|Instructed Q image (teacher); plain|AMI|3 k-means|comparator*|Teacher 0.0921/0.4026/0.5062; plain 0.0708/0.2233/0.4521. Reused teacher in later reports|R19 Analysis target table; R28 stage 4; B14 Results|
|r10 D2 ablation|AE dev, 5,000|ESG|Q|12 changes incl. no instruction, no buddy, no teacher, default teacher, image-only endpoints, sharing|AMI|0–2|ablation|No instruction0.0733/0.1848/0.4241;image-only S0.3945 vs teacher0.3952|R22 §brief/Results|
|r10 recipe pick|AE dev P=29,232|ESG|Q|Five recipe candidates vs D2, teacher share fixed/changed|AMI|0|ablation|No own-caption at share 1/5 chosen; gains +0.0061/+0.0240/+0.0070; mean gain 1.60 ranges, image-only endpoints 1.55|R24 §brief; Part 1|
|r10 backbone tests|AE held; sample/full training|ESG (claims E)|Q,S2,B32|R6-5 student (pre-G, pre-c*)|AMI|0–4|claim E; descr. SG|Full E Q0.1029/S2 0.1024 GO,B32 0.0885 inconclusive;sample0.1045/0.1020/0.0901;SG both scales|R24 stages 3–4; Analysis|
|r10 encoder extension|AE held; sample/full (DN sample only)|ESG (claims E)|B16,L14,S1,DN|R6-5 student|AMI|0–4|claim E; descr. SG|Full E B16 0.0945 inconclusive, L14 0.0996 GO, S1 0.1036 GO; sample 0.0944/0.0992/0.1046; DN 0.0814 NO-GO, full not run|R25 stage i; Analysis stage i|
|r10 matched arms/plains|AE held; sample/full (DN sample)|ESG|S2,B32,B16,L14,S1,DN|Matched PercepT; frozen plain encoder|AMI, own/k20 PercepT|0–4 model; 3 k-means plain|descr./comparator E|R24 full own E S2 0.0793, B32 0.0716; r14 strongest held S L14 matched 0.2306, G S2 matched 0.4558; other cells exist in reported comparisons|R24 matched table/Analysis; R25 Analysis; B14 Analysis strongest PercepT|
|r10 dev lift|AE dev P=29,232|E|Q,S2,B32,B16,L14,S1,DN|R6-5 / matched PercepT / plain|AMI|0–4|descr.|Student 0.1050/0.1042/0.0905/0.0952/0.1031/0.1073/0.0839; every own-arm lift interval positive|R25 stage ii|
|r10 transfer|AE28, 746|E|Q,S2,B32,B16,L14,S1,DN|D2/R6-5 student; PercepT core/matched; frozen images|AMI|0–4; DN sample weights|descr. confirmation|Q D2 0.1344, Q R6-5 0.1443, S2 0.1536, B32 0.1198; extension B16 0.1276, L14 0.1402, S1 0.1477, DN 0.1253. No holds: qualified except B32/B16 fail|R24 stage 5; R25 stage i|
|r10 genre loss|AE dev P|ESG|Q|Seven genre variants: direct teacher x2/x4, relational 0.5/2/4/8, regression 1|AMI|0; GR8 fresh 1–3|ablation|GR8 seed0 G 0.5018 vs teacher 0.4989; fresh G 0.5022 vs R6-5 0.4694, +0.0328 [+0.0235,+0.0442]|R25 stage iii|
|r10 relational G|AE dev P; held|ESG|Q|Six variants;G teacher0.02/weight8|AMI|sweep0; confirm1–3; held0–4|ablation; claim G|G held 0.1039/0.4057/0.5001; G genre vs teacher −0.0060 [−0.0105,−0.0014], G-GO within 0.0128 range|R27 §brief; R28 stage 4 G comparison|
|r10 recipe sweep|AE dev P; sample/full held|ESG|Q|39 knob settings, 18 new joint points; three finalists; G emotion retest|AMI|sweep0; finalists1–3; held0–4|ablation; claim E|G E vs core sample +0.0188/full +0.0164; c* (k40) confirmed 0.1089/0.4072/0.5019 on P; replaces G|R28 What ran; stages 0–3|
|r10 final c*|AE held|ESG|Q|c*=G k40|AMI|0–4|claim|0.1066/0.4066/0.5001; vs teacher E/S WIN, G LEVEL (whole interval below 0); vs core E +0.0192 [+0.0137,+0.0207]|R28 §brief; stage 4|
|r10 removals→r14 rescore|AE dev P; val; held|ESG|Q|Eight removals: relational, sharper teacher, direct teacher, buddy, captions, correction, instructed graphs, trunk|AMI|P1–3; val/held0–4|ablation|Held no graphs S −0.1903; no buddy S −0.1512; no correction S −0.0752; no relational G −0.0440. R28 table;r14 reuses runs|R28 Ablation; B14 Analysis|
|r12 transfer|AF val 4,251; test 8,501|E|Q|Frozen c*|AMI|0–4|diag. confirmation|Test 0.1384 vs plain 0.1041 WIN; teacher 0.1666 BELOW; confirmation fails|R31 Results; validation picks|
|r12 comparators|AF val/test|E|Q,L14|Plain Q/L14; teacher artwork/photo; CRL Q/L14; frozen PercepT core|AMI|c*/PercepT0–4; frozen3 k-means|descr. / comparator|Test L14 0.1127; PercepT 0.0784 k20/0.0870 own; teacher photo 0.1581; CRL clip_photo 0.1278, qwen_painting 0.1073, qwen_photo 0.1137; picked CRL clip_painting AMI not found|R31 Analysis, descriptive and unpicked tables|
|r13 fine-tune sweep|AF val|E|Q|Nine photo rows + frozen reference; k, lr, teacher, correction, random start|AMI|0|ablation|k80 pick 0.1905; correction off 0.1570; random start 0.1801 vs c* start 0.1769|P13 sweep table|
|r13 photo claim|AF test|E|Q|Photo model|AMI|0–4, training100+s|claim|0.1838 vs plain 0.1041 (+0.0797) and teacher 0.1666 (+0.0172), both WIN; frozen c* 0.1384 descr.|P13 Results|
|r14 encoder claim|AE held|ESG|S2,S1,L14|c*; plain encoder; matched PercepT|AMI|0–4|claim|All nine aspect/encoder comparisons WIN over plain and matched. Minimum per-encoder plain gains E +0.0477/+0.0399/+0.0377; matched +0.0253/+0.0250/+0.0243|B14 SC1/SC3|
|r14 teacher comparison|AE held|ESG|S2,S1,L14 (teacher Q)|c* vs instructed Q|AMI|0–4|claim|G c* 0.5084/0.5065/0.5010 vs teacher 0.5062: LEVEL for each; E/S WIN|B14 SC6; Licensed sentences|
|r14 photo claim|AF test|E|S2,S1,L14|Photo model; own plain; teacher Q|AMI|0–4|claim|Photo 0.1924/0.1928/0.1941 vs plain 0.1142/0.1200/0.1127; all WIN; teacher comparisons level or better|B14 SC4; Licensed sentences|
|r14 fresh transfer|EmoSet val 5,770; test17,348 after cut|E|Q|Frozen photo model; plain Q; instructed teacher|AMI|0–4|claim|Test 0.4571/0.3918/0.4390 (photo/plain/teacher); photo WIN vs both|B14 SC7/SC9|
|r14 descr. transfer|EM val/test|E|Q,S2,S1,L14|c*Q;photo/plain S2/S1/L14;CRL L14;photo teacher|AMI|seeded0–4; fixeds3 states|descr.|Q photo−c* +0.0098;other points not found in report|B14 What ran/Analysis; EmoRun:24–38,80–86; bs_run_log.md:66,77–79|
|r14 full-scale analysis|AE val; AF val|ESG / E|Q|39 knob rows + reference (120 units); 14 photo rows + reference (45 units)|AMI|knob1–3; photo0–2|ablation|Complete 39/39 and 14/14, no selection; buddy temperature 0.1: S −0.0163; photo correction off E −0.0320|B14 Analysis|
|r15 new art|MM val5,769; test16,384|ESG|Q,S2,S1,L14|Frozen c*|AMI|0–4|claim|Q 0.0609/0.4227/0.5131; S2 0.0608/0.4504/0.5181; S1 0.0620/0.4546/0.5213; L14 0.0590/0.4345/0.5115; all WIN over plain/matched|M15 SC1/SC3|
|r15 comparators|MM test|ESG|Q,S2,S1,L14|Plain|AMI|3 k-means states|comparator*|Q 0.0407/0.2314/0.4498; S2 0.0332/0.3243/0.3993; S1 0.0370/0.3915/0.4325; L14 0.0386/0.3558/0.4322|M15 SC1|
|r15 comparators|MM test|ESG|Q,S2,S1,L14|Frozen PercepT core/matched, max own/20 reading|AMI|0–4|comparator*|Q 0.0514/0.1959/0.4257; S2 0.0451/0.2298/0.4611; S1 0.0468/0.2297/0.4594; L14 0.0454/0.2329/0.4506|M15 SC3; Analysis strongest PercepT|
|r15 teacher|MM test|ESG|Q;4students|Instructed Q image|AMI|3 k-means states|comparator*|All E/S teacher comparisons WIN; G Q/S2/S1 WIN under LEVEL-or-WIN bar, L14 LEVEL; teacher absolute AMI not tabulated|M15 SC4|
|r15 diagnostics|MMArt far10,403; shared-held11,528|ESG|Q,S2,S1,L14|c*/plain/teacher; shared images under MMArt vs ArtELingo labels|AMI|0–4|diag.|Far all plain cells and pre-set better-than teacher cells retain positive lower bounds; shared Q E gain 0.0192 MMArt labels vs 0.0364 ArtELingo|M15 Analysis|

### A2. Label retrieval (query by image)

R0=plain cosine; picked=validation read-out, usually center+k10; frozen arms have no training seeds. [R29/R31 What ran]

|Round|Dataset and split|Aspects|Encoder|Method row|Metric|Seeds|Role|Headline number or outcome|Source, section|
|---|---|---|---|---|---|---|---|---|---|
|r11→r12|AE held12,281; G9,949|ESG|Q|c*, R0 / picked|mAP|0–4|claim|R0 0.2642/0.3110/0.5896; picked 0.2691/0.3235/0.6151. Vs teacher R0 BELOW/WIN/WIN; picked LEVEL/LEVEL/WIN; all plain-image WIN. r12 repeats picked points|R29 stages1/5; R31 SC5–7|
|r11|AE val6,152 / held|ESG|Q|Plain, R0 / picked|mAP|fixed|comparator*|Held R0 0.2424/0.1801/0.4760; picked 0.2483/0.1924/0.5361|R29 Results|
|r11|AE val/held|ESG|Q|Instructed teacher, R0 / picked|mAP|fixed|comparator*|Held R0 0.2665/0.3091/0.5860; picked 0.2684/0.3217/0.6124|R29 Results|
|r11|AE val/held|ESG|L14,S2|Plain encoder, R0 / picked|mAP|fixed|descr.|L14 0.2424/0.2421/0.4272 → 0.2479/0.2732/0.4981; S2 0.2364/0.2298/0.4327 → 0.2443/0.2576/0.4968|R29 Results descriptive arms|
|r11|AE val/held|ESG|Q|PercepT core, R0 / picked|mAP|0–4|descr.|0.2415/0.1486/0.4982 → 0.2483/0.1523/0.5049|R29 Results descriptive arms|
|r11→r12|AE val/held|ESG|L14|CRL reimplementation, painting values, R0 / picked|mAP|fixed|descr.→comparator*|0.2538/0.2722/0.4744 → 0.2554/0.2816/0.5033; r12 c* WIN all three; bars set after r11 values seen|R29 Results; R31 SC5–7|
|r12|AE val/held|ESG|Q|CRL painting version|mAP|fixed|descr.|Picked held 0.2659/0.2284/0.4840; unselected against CLIP CRL|R31 Analysis unpicked table|
|r11 read-outs|AE val|ESG|Q,L14,S2|Up to25 maps/expansions per arm (7 arms)|mAP|c*/PercepT0–4; others fixed|ablation|All picks center+k10 except CRL none+k10; whitening fails every guard|R29 stage4; Analysis|
|r12→r13|AF val4,251/test8,501|E|Q|Frozen c*, R0 / picked|mAP|0–4|diag. confirmation|Test 0.2670/0.2755; picked plain 0.2452 WIN; teacher 0.2945 BELOW; photo confirmation fails; r13 repeats frozen row|R31 Results/Analysis; P13 Results|
|r12→r13|AF val/test|E|Q|Plain; instructed teacher artwork/photo|mAP|fixed|comparator / descr.|Plain R0 0.2383, picked 0.2452; artwork teacher 0.2884/0.2945; photo wording picked 0.2897, not selected|R31 Results/Analysis; P13 Analysis|
|r12|AF val/test|E|L14|Plain L14; CRL clip_painting / clip_photo|mAP|fixed|descr.; CRL comparator|Plain 0.2410 R0/0.2581 picked; picked CRL painting 0.2739, photo 0.2744; frozen c* vs selected CRL LEVEL (WIN bar fails)|R31 SC8; Analysis|
|r12|AF val/test|E|Q|CRL qwen_painting / qwen_photo|mAP|fixed|descr.|Picked 0.2655/0.2678 (both unselected)|R31 Analysis unpicked table|
|r12|AF val/test|E|Q|Frozen PercepT core|mAP|0–4|descr.|0.2345 R0/0.2386 picked|R31 Analysis descriptive arms|
|r12 read-outs|AF val|E|Q,L14|Two teacher wordings; four CRL versions; up to25 read-outs/arm|mAP|c*/PercepT0–4; others fixed|ablation|Artwork wins both tasks; CLIP painting CRL selected; PercepT center+k0, others center+k10|R31 What ran picks|
|r12 figure|Impressions822 IDs; six queries/816gallery|E|Q|c* emotion seed0; plain Q|Top5 ranking/cosine, no label mAP|0|descr.|Figure only; many shared top5 images; no claim|R31 Analysis figure; Deviations|
|r13 sweep|AF val|E|Q|Nine photo fine-tunes + frozen reference|mAP|0|ablation|k80 pick 0.3280; random start 0.3227; c* start 0.3227; correction off 0.2932|P13 What ran sweep|
|r13 photo claim|AF test|E|Q|Photo model R0 / picked; CRL L14 comparator|mAP|0–4|claim; R0 diag.|0.3068/0.3172; picked vs plain +0.0720 WIN, teacher +0.0227 WIN, CRL 0.2739 +0.0433 WIN|P13 Results; Analysis|
|r14 encoder claim|AE val/held|ESG|S2,S1,L14|c*; own plain; Q teacher|mAP|0–4|claim|All nine plain-image cells WIN; minimum gains E +0.0249/+0.0236/+0.0192. E c*/teacher 0.2692/0.2684 LEVEL, 0.2688/0.2684 LEVEL, 0.2671/0.2684 BELOW; S/G teacher WIN|B14 SC2/SC6|
|r14 photo claim|AF val/test|E|S2,S1,L14|Photo model; own plain; Q teacher|mAP|0–4|claim|Photo 0.3227/0.3234/0.3221; plain 0.2515/0.2545/0.2581; all WIN vs plain; all level or better than teacher|B14 SC5; Licensed sentences|
|r14 fresh transfer|EM val/test17,348|E|Q|Frozen photo model; plain Q; artwork teacher|mAP|0–4 photo; fixed comparators|claim|0.4838/0.3864/0.4519; photo WIN over both. R0 0.4583/0.3519/0.4313 diag.|B14 SC8/SC10; Analysis plain cosine|
|r14 descr. transfer|EM val/test|E|Q,S2,S1,L14|c*Q;photo/plain S2/S1/L14;CRL L14;photo teacher|mAP|seeded0–4; frozen fixed|descr.|Q photo−c* +0.0425;other report points not found|B14 Analysis; EmoRun:24–38,80–86; bs_run_log.md:77–79|
|r14 analysis|AE val/held; AF val|ESG / E|Q|39 knob, eight stored removals, 14 photo rows|mAP R0 and picked|knob1–3; removals0–4; photo0–2|ablation|No relational held G picked mAP −0.1175 [−0.1214,−0.1134]; photo 20 epochs +0.0056 vs reference 0.3281|B14 Analysis|
|r15 new art|MM val/test|ESG|Q,S2,S1,L14|Frozen c*; plain|mAP|0–4 student; frozen fixed|claim|Q0.2011/0.3472/0.6199;S2 0.2016/0.3744/0.6291;S1 0.2021/0.3831/0.6282;L14 0.2010/0.3637/0.6245;all WIN vs plain|M15 SC2|
|r15 comparator|MM test|ESG|Q,S2,S1,L14|Plain|mAP|fixed|comparator*|Q 0.1935/0.2046/0.5151; S2 0.1897/0.2779/0.5097; S1 0.1913/0.3031/0.4932; L14 0.1912/0.2967/0.5110|M15 SC2|
|r15 teacher|MM test|ESG|Q;4students|Instructed image|mAP|fixed|comparator*|E Q −0.0010 BELOW, S2 −0.0005 LEVEL, S1 −0.0000 LEVEL, L14 −0.0011 BELOW; S Q LEVEL, others WIN; G all WIN|M15 SC4|
|r15 diag.|MMArt far10,403; shared-held11,528; test|ESG|Q,S2,S1,L14|c*/plain/teacher, R0, far subsets|mAP|0–4; comparators fixed|diag.|Far Q teacher gap E −0.0007, S −0.0039; test Q E R0 c*/plain/teacher 0.1998/0.1935/0.2026|M15 Analysis|

### A3. Other measurements (brief)

|Round|Dataset/split|Aspects|Encoder|Method row|Metric|Seeds|Role|Headline|Source/section|
|---|---|---|---|---|---|---|---|---|---|
|r10|AE train captions;AE28 746|E|B32|Raw/residual captions;language groups|Probe accuracy;JS|3kmeans;probe not found|diag.|57.3%→46.6%;JS excess+0.052,se0.001|R06§2|
|r10|AE99,999captions|E|Q|Word masking|Caption AMI|3states|diag.|Broad mask0.187 vs default0.069;label mask0.231/0.095|R09 brief|
|r10|AE dev5,000|ESG|Q/B32|Supervised linear ceilings|AMI;accuracy|3states;C picked on1,489|diag.|Q default0.121/0.527/0.657,instructed0.130/0.569/0.658;default accuracy0.480/0.640/0.849|CeilingLog:22–32;R16 tables|
|r10|AE dev5,000|ESG|Q/B32|PercepT/plain/instructed/model-sheet rows|V/ARI/FMI;caption AMI|0–2;specified rows1|diag.|D2 E0.102/0.038/0.146;S0.381/0.173/0.231;caption E0.1828;full ESG sheets|R15 extra measures;R16 Results|
|r10|AE train/dev|ESG|Q/B32|Graph variants|Lift/overlap/density|as tables|diag.|E2.42,S7.07,Gimage5.35;shared-content E2.55(CLIP2.61)|R10–R12 Results|
|r10|AE held|E|Q|D2/core Leiden;matched-count;train-vote|AMI|0–4;Leiden3|diag.|Full0.0757/0.0762;earlier sample0.0766/0.0748;matched-count0.0980/0.0789;vote0.0678/0.0748|R19 Analysis;R18|
|r10|AE dev/held;AE28|ESG|7encoders|Regression,Leiden,epoch4,views,classF1,head agreement|AMI/F1/cosine/neighbours|1–5|diag.|Ridge G0.5040 vs teacher0.4989(dev);c*GLeiden0.5082 vs PercepT0.4454(held)|R24/R25 Analysis;R27;R28stage4|
|r10|AE dev5,000;304controls|ESG|Q|Four views;image/text/joint;value/border controls|Attention/spatial/word/class statistics|one bf16 pass|diag.|Image L1shift0.68/0.79/0.81;border maxima34–63% vs29%;not embedding importance|R23 brief/checks|
|r11|AE held|ESG|Q/CRL L14|Correct/wrong c*heads;teacher/CRL diagonals|mAP|c*0–4|claim c*;descr.others|Six c*comparisons WIN,0.0103–0.2304;switch GO|R29stage2/Analysis|
|r11|AE held|E|Q|c*/plain/teacher|nDCG;P@10|c*0–4|confirmation/diag.|nDCG0.9343/0.9292/0.9341;both WIN,confirmation qualified;P@10 E/G above teacher,Sbelow|R29stage3/Analysis|
|r12–13|AF test/val|E|Q/CRL L14|Photo/c*/plain/teacher/CRL;untied/source/class/graph rows|nDCG/AP/AMI/mAP/edge share|learned0–4;sweep0|diag.|R0nDCG0.9396/0.9315/0.9250/0.9372/0.9314;contentment photoAP0.3718 vs teacher0.3784;untied6,584allphotoWIN;corrected k40 edges54.2% vs random19.3%|P13 Analysis;R31 Analysis|
|r14|EM test;AE held;AF val|E/ESG|4encoders|k8,R0,AP,cut/uncut,seeds,gain share,photo length,node parity|AMI/mAP/AP/ranges|0–4;photo0–2|diag.|k8photo/teacher/plain0.4403/0.3942/0.3492;amusementAP0.3203/0.3471;uncut plainAMI gain+0.0390(cut+0.0653)|B14 Analysis12diagnostics|
|r15|MM test/far/shared-held|ESG|4encoders|k9,AP,R0,gain share,seeds,artists/copies|AMI/AP/counts|0–4|diag.|k9Qc*/plain/teacher0.0532/0.0381/0.0492;contentmentAP0.2413/0.2547;unseen15paintings/12artists,no cells|M15 Analysis|

## B. Coverage grid (derived from A)

*=claim/comparator cell, including failed tests; not every listed aspect has a licensed claim. Includes descriptive rows with points not found. Dataset shorthand: AH AE held; AT AF test; ET EM test; MT MM test. [A1–A3]

### B1. Clustering

|Method|AH Q|AH S2|AH S1|AH L14|AH B32|AH B16|AH DN|AT Q|AT S2|AT S1|AT L14|ET Q|ET S2|ET S1|ET L14|MT Q|MT S2|MT S1|MT L14|AH CLIP patches|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|c*|ESG*|ESG*|ESG*|ESG*|-|-|-|E*|-|-|-|E|-|-|-|ESG*|ESG*|ESG*|ESG*|-|
|Photo model|-|-|-|-|-|-|-|E*|E*|E*|E*|E*|E|E|E|-|-|-|-|-|
|G|ESG*|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|D2|ESG*|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|R6-5 student|ESG*|ESG*|ESG*|ESG*|ESG*|ESG*|ESG*|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Plain|ESG*|ESG*|ESG*|ESG*|ESG|ESG|ESG|E*|E*|E*|E*|E*|E|E|E|ESG*|ESG*|ESG*|ESG*|-|
|Instructed Q teacher|ESG*|-|-|-|-|-|-|E*|-|-|-|E*|-|-|-|ESG*|-|-|-|-|
|PercepT core / matched|ESG*|ESG*|ESG*|ESG*|ESG*|ESG*|ESG*|E|-|-|-|-|-|-|-|ESG*|ESG*|ESG*|ESG*|-|
|PercepT paper strict/K40|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|E*|
|CRL Q|-|-|-|-|-|-|-|E|-|-|-|-|-|-|-|-|-|-|-|-|
|CRL L14|-|-|-|-|-|-|-|-|-|-|E|-|-|-|E|-|-|-|-|-|
|Eight c* removals|ESG|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|CLAY / TPIPS|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|

P1/P4: CLIP patches, version not found; topics B32+RoBERTa/Q [R19 glossary]. Core global CLIP B32 [T/20261018_percept_core/pc_data.py:29–30].

### B2. Retrieval

|Method|AH Q|AH S2|AH S1|AH L14|AH B32|AH B16|AH DN|AT Q|AT S2|AT S1|AT L14|ET Q|ET S2|ET S1|ET L14|MT Q|MT S2|MT S1|MT L14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|c*|ESG*|ESG*|ESG*|ESG*|-|-|-|E*|-|-|-|E|-|-|-|ESG*|ESG*|ESG*|ESG*|
|Photo model|-|-|-|-|-|-|-|E*|E*|E*|E*|E*|E|E|E|-|-|-|-|
|G / D2 / R6-5|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Plain|ESG*|ESG*|ESG*|ESG*|-|-|-|E*|E*|E*|E*|E*|E|E|E|ESG*|ESG*|ESG*|ESG*|
|Instructed Q teacher|ESG*|-|-|-|-|-|-|E*|-|-|-|E*|-|-|-|ESG*|-|-|-|
|PercepT core|ESG|-|-|-|-|-|-|E|-|-|-|-|-|-|-|-|-|-|-|
|Matched/paper PercepT|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|CRL Q|ESG|-|-|-|-|-|-|E|-|-|-|-|-|-|-|-|-|-|-|
|CRL L14|-|-|-|ESG*|-|-|-|-|-|-|E*|-|-|-|E|-|-|-|-|
|Eight c* removals|ESG|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|CLAY / TPIPS|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|

## C. Ablations and analysis already done

|Set varied/removed|Split, encoder, tasks, seeds|Source|
|---|---|---|
|Caption residual/projection; language grouping|AE train/AE28,B32;probes/JS/clustering,3kmeans|R06§2|
|Ten prompts; word masking; own-caption/control/prompted-input students; neighbour averages|AE dev,Q;clustering/probes/lift,3states;student3seeds|R08/R09|
|Purification/content negatives/CLIP graph|AE dev,Q+B32;clustering/lift,3seeds;CLIP overlap no interval|R10|
|21graphs/aspect:pruning,residual,fusion,other-side,prompted inputs|AE dev,Q;both-side clustering,1seed then top/reference0–2;28+31runs|R11/R12|
|PercepT P1–P5,automatic/K40;feature/affect inputs|AE dev,Q/B32/RoBERTa;clustering/AUC/V/ARI/FMI,0–2|R13–R15|
|13model rows:sharing,mask,negatives,direct/relational,inputs;12D2 changes:teacher/graph/buddy/caption/trunk/instructions|AE dev,Q;clustering both sides;model sheet7rows3seeds+6rows1seed;D2 changes0–2|R16;R22|
|Five recipes:caption/endpoints/teacher share;ridge/4-layerMLP/linear ceiling|AE P+old5k,Q;clustering,captions,Leiden,epoch4,regression;recipe0,scoring3states|R24 Part1/Analysis;CeilingLog|
|Seven encoders/matched arms/sample/full transfer|AE dev/held+AE28;clustering/Leiden,0–4|R24/R25|
|Genre direct/regression/relational weights;sharpness/centering/caption changes|AE P/held,Q;clustering;0sweep,1–3confirm,0–4held|R25stageiii;R27|
|39knobs+18joint points;three finalists,machine parity|AE P,Q;clustering;0sweep,1–3confirm|R28stages1–3|
|Eight c* removals listed A1|AE P/val/held,Q;clustering+retrieval(r14);P1–3,val/held0–4|R28 Ablation;B14 Analysis|
|Read-out grid,head switch,graded targets,classAP;photo wordings/fourCRLs/untied/source checks|AE/AF val/test,Q/L14/S2;retrieval,both tasks for AF;learned0–4|R29/R31|
|Nine photo rows:lr,k,correction,teacher,start|AF val,Q;both tasks;0sweep,0–4final|P13|
|39 full-scale knobs(13axes);14photo rows incl.fixed epochs|AE val/AF val,Q,node402;both tasks;knob1–3,photo0–2;no selection|B14 Analysis;r14 spec§3.2|
|Attention four views/image-text-joint;spatial/value/border|AE dev,Q;attention only;one pass,304controls|R23|
|k8/R0/AP/cut/uncut/gain share/seeds/other encoders;MMArt shared labels/far/k9/artists/copies|EmoSet/MMArt val/test,4encoders;both tasks,PercepT clustering only;learned0–4|B14/M15 Analysis|

## D. Not run: comparators, datasets, metrics, ablations

No new-line run found; reason not found unless given. Completed experiments: A. [A; DL]

D source aliases: L05=L(2026-12-05,percept_neighbourhood); L07=L(2026-12-07,prompted_aspect_views); L17=L(2026-12-17,cluster_level_losses); L20=L(2026-12-20,ways_to_improve); L30=L(2026-12-30,signals_beyond_teacher); LP=L(2027-01-02,photo_aspect_datasets); LR=L(2027-01-03,paper_related_work_sources); LC=L(2026-10-21,cvpr_literature_review). DN03=DL2026-11-03; Park=DL Parked ideas.

|Item|What / unrun scope|Source|Reason / feasibility recorded|
|---|---|---|---|
|CLAY|Keyword-conditioned frozen-VL projection|R29 Scope; r11 spec D5; DN03|User chose CRL only. Later public github.com/kaist-ami/CLAY, training-free CLIP/SigLIP base, Stanford40 release, no license file; earlier LC§2.6 said project page only|
|TPIPS comparator|Human-triplet-tuned Qwen similarity, text aspect|LC§2.6; DN03|Pending timing; only8B released (three sywang models); 2B needs training;7.5–11 GPUh estimate, not measured|
|TPIPS teacher|Replace Qwen teacher and retrain|Park2026-11-03|User deferred: no time before deadline|
|IC\|TC; Text-guided clust.|Text-criteria image clust.|L05 Q3/Q4; LC§2.1|Reason not found|
|Multi-MaP|Multimodal proxies for personalized multiple clust.|L05 Q4; LC§2.1|CVPR2024; reason not found|
|Multi-Sub|Per-interest pseudo-label cluster training|L17 Q4/Novelty|Transductive; reason not found|
|X-Cluster|MLLM/LLM criterion discovery|L07 Q1/Q5; LR H1|Training-free; COCO-4c/Food-4c; reason not found|
|ESMC|Prompted MLLM + pseudo-label head|L17 Q2/Q5|Small head; transductive; reason not found|
|Guideline-driven clust.|Instructed GME + graph/MST agent|L07 Q1/Q5; LR H1|Training-free; reason not found|
|CIM-Embed|Instruction-defined product groups|L17 Q4; LR H1|Supervised Qwen tuning,449 grouping tasks; note excludes label-free comparison|
|TURTLE|Maximum-margin labels on frozen spaces|L17 Q5/idea3; L20 idea10|Precomputed workflow, needs K; own-backbone code change not checked|
|TEMI|Neighbour self-distillation heads|L17 Q5/idea3; L20 idea10|Precomputed features, K required, inductive|
|SCAN|Neighbour consistency/self-labelling|L17 Q1/Q3; L05 Q3|Frozen features omit strong-augmentation safeguard|
|DEC/IDEC standalone|Separate comparator and c* extension|L05 Q4; L17 Q3|Not found as standalone; DEC/IDEC did run inside PercepT|
|CPP; SCP Keep it light|Frozen-feature cluster heads|L17 Q5|CPP estimates K, clusters WikiArt; SCP fixed K; inductive|
|SEIC; TAC|WordNet/text-assisted clust.|L17 Q5; L05 Q3|Text encoder/nouns required; emotion/style suitability untested|
|DELIUS|Art-specific DEC on DenseNet121|L17 Q3/Q5|Frozen CNN features; kmeans comparison not checked in note|
|CAE; BERTopic-MM; CEMTM|clust./topic comparators|L05 Q3/Q4|CAE training-free; other feasibility not found|
|DivClust; MFCVAE; orthogonalized multi-view clust.|Multiple heads/facets/nonredundant partitions, cited precedents|L17 Q4; L20 architecture; LR H1|DivClust shared encoder20 heads; reproduction not found|
|Published PercepT encoders|CLIP L14 + GoEmotions-ModernBERT|R18/R19 limits; L05 Q1|Excluded; reimplementations used own stores/RoBERTa/Qwen|
|Photo-fitted PercepT|Train PercepT on Affection/EmoSet|r14 spec§2|Out of scope; frozen Affection PercepT exists|
|Matched PercepT retrieval|Added encoders, MMArt/EmoSet retrieval|B14 Analysis; M15 Results|B14 strongest-PercepT item explicitly no retrieval; Q core retrieval in A|
|CRL MMArt / same-encoder bootstrap|CRL MMArt; c*L14 vs CRL L14 ArtELingo|DN03|CPU/features available; same-encoder points0.2671/0.2554 exist, bootstrap pending report|
|InDiReCT; DIOR; FocalLens; SteerViT; Fioresi et al.|Text/attribute-conditioned similarity, close priors|LC§2.6; LR H1; drafts/v1/paper_v1.md§2.1|DIOR main Llama11B, training-free; Fioresi attribute labels; FocalLens instruction training|
|GeneCIS Combiner; MagicLens; SEARLE; CIReVL; OSrCIR|Conditional/composed retrieval comparators|LC§2.2/§5|Older episode context; Combiner RN50x4/B16 released; SEARLE B32 weights; CIReVL/OSrCIR use LLMs|
|CSN; SCE-Net; DiscoverNet|Triplet-trained conditional masks/subspaces|LC§2.1; L07 Q4|Model-sheet mask row is not full reproduction|
|Other instructed embedders|VLM2Vec v1/v2,GME,E5-V,MM-Embed,UniIR,LamRA,jina-v4,UniME,LLaVE,mmE5,ABC,Think Then Embed,WeMM,FreeRet/Lens|LR H2; LC§2.8|Candidate teachers/backbones; no new-line runs found|
|Other plain encoders|OpenCLIP L/H,EVA-CLIP,Perception Encoder,jina-clip-v2,BLIP2|LC§2.8|Candidate table; no new-line runs found|
|DEC from good heads: track2|Sharpen learned per-aspect partition|DL2026-10-20; L17 Q3; DN03|Track2 decided; R19 says waiting choice B; latest DL explicitly never run on new-line heads|
|Cluster contrast / prototype anchors: track2|Per-head contrast/KL to teacher centroid assignments|DL2026-10-20; L17 ideas1–2|K20/core0.3/entropy proposed; no execution found; differs from batch relational loss|
|TEMI-style c* head|EMA neighbour teacher, marginal reweighting|L17 idea2; L20 idea12|Proposed EMA0.996,beta0.6; minutes estimated|
|Deeper residual heads|Three-layer residual aspect head/depth changes|DN03|Explicitly never run; width/sharing sweeps exist|
|Joint direct+relational S/G loop|Earlier relational0.5/2 plus direct0.5; style remainder|Park; R16 options|Not chosen2026-10-20; genre relational-alone later ran|
|Align/contrast heads; orthogonality|Common/aspect factor separation|Park; L20 final note|Not chosen: aspects may not be separable; note found no orthogonality support|
|Discovered aspects; PercepT-only paper|LLM aspect list; fair-protocol whole-paper candidate|Park|Deferred2026-10-18; no mapping discovered criteria→named aspects|
|Culture / caption residual training|Language condition, within-painting variation signal|Park; R06§2|Deferred2026-10-18; premise JS/probes exist, training never tried|
|Token importance checks|Ring probe, attended-token masking, class attention/gain correlation|Park; R23 limits|Parked2026-10-25; value weighting/grey border did run|
|Attention-weighted caption loss|Use word attention as training weights|Park; R23|Parked; fit-side extraction needed|
|Caption-side retrieval|Text query through caption trunk/heads; held text-side scoring|DN03 correction r11 D1; r11 spec§2|Excluded r11, pending validation exploration; caption side exists; development caption AMI already measured|
|Untold retrieval|Join three heads, no aspect selector|DN03|CPU saved embeddings; deferred until collaborator/supervisor report|
|Larger read-out/reranking|Neighbours beyond10; standard reranking|R29 options/Analysis; L20 hubness|Picks at largest tested k10; whitening already tested|
|Teacher read-out as target|Distil centered/query-expanded teacher|L30 Lever1/controller note|Minutes estimate; no new information|
|Second / prompt-ensemble teacher|Caption E,CLIP/SigLIP S,teacher dropping;4–8 paraphrases|L20 ideas5/7; L30 Levers2/5|Second-source stores exist; paraphrases require extraction|
|Standardized distillation|PHI-S,standardized dimensions,smooth-L1,discarded adaptor|L20 idea3|Not found; ridge/MLP ceiling and genre regression control did run|
|Centroid/queue ranking targets|K20/~200 anchors or neighbour ranking hinge|L20 idea4; L30 Lever4|Not found; batch relational term already ran|
|Crops; richer default features|K4 consistent crops; intermediate/token states|L20 ideas6/8|Re-extraction needed; six layers~1.4GB estimate|
|Extra art / born-again generation|More unlabeled art; second student generation|L20 idea9; L30 Levers3/6|Not found; photo fine-tuning is completed separate case|
|Graph refinement/random control|Mutual proximity,Jaccard,student refresh,degree-matched random graph|L20 idea11|Not found; residual/pruned/fused graphs already tested|
|Soft-label emotion training|Train on annotator distributions|L30 Lever7|Votes exist; violates label-free constraint per note; nDCG evaluation ran|
|Qwen-free / new unselected encoder|Remove all Qwen supervision; c*B32/B16/DN/fourth encoder|B14 limits; r14 spec§2|Excluded; older R6-5 encoders/no-instruction ablation not equivalent|
|Photo model paintings/from-scratch held|Single fine-tuned model both domains; random-start held test|P13 limits/Analysis|Explicitly not tested; random-start validation exists|
|SemArt unseen-artist genre|11,162 paintings,2,717 artists,type labels|Park; DN03; T/20261103_semart_check/20261103_semart_check_log.md|Deferred until report; stores exist;0.95 cut misses cross-source copies|
|Flickr Style / AVA style|Second photo aspect,20/14 labels|Park; LP B|Deferred; access risk; Flickr80,000 no text; AVA~31k style images, DPC comment join unchecked|
|Emotion6; FindingEmo|Other independent photo-emotion tests|LP A|Not chosen;1,980/request/lossy map;25,869/24classes/URL list respectively|
|FI; EmoVerse; ArtPhoto|Other affective test candidates|LP A/table|FI Affection source, excluded independent test; EmoVerse overlaps EmoSet/Flickr30k+AI; ArtPhoto806|
|WEBEmo; Twitter I/II|Weak25-class / binary emotion images|LP table; L(2026-12-01,affect_caption_datasets)|~268k keyword-labelled; Twitter1,269/603 binary; no evaluation found|
|Photozilla; DataSeeds; Unsplash|Style/mood sources|LP B|>990k classifier styles;10,610 unreleased; local Unsplash no class labels found|
|MMArt perspective queries/aspects|Emotional/formal/narrative/historical text queries or instructions|Park; MPlog|Parked2026-11-02; text features needed; new aspects lack heads|
|MMArt unseen artists scoring|Test artists absent from ArtELingo training|M15 Analysis|15paintings/12artists, fewer than20clusters, no cells|
|Conditional/fine-grained benchmarks|GeneCIS,COCO-Facet,CLAY-EVAL,CORE,ZeroSight,MCMR,LanZ-DML,CUB human captions,Stanford40,CLEVR4,face/fashion sets|LC§2.7/§4; L30 Q4|No new-line scores; older episode context; many face/fashion captions leak labels|
|NMI; Hungarian ACC|Main clust. paper columns|L05 Q4|Results not found; requested alongside AMI/ARI|
|Held ARI paper column|Final dataset/encoder comparison column|L05 Q4; R15/R16|Not found (development ARI: A3)|
|H/C; SI/CHI/DBI|Homogeneity/completeness; internal cluster metrics|model-sheet spec D7; DL2026-10-19 K40|H/C dropped; internal scores set aside: “without DEC I don't think we gain a lot on these”|
|Unified K-pass latency/memory|One student pass vs teacher passes|R21§2; L30 Q4; r14 facts brief Q1e|Partial timings E; complete deployment/peak-memory experiment not found|

## E. Cost facts

Trainer vs queue times are distinct; batch times are not single runs. BFacts quotes external artifacts; these were not opened. “Not recorded” refers to inspected logs. Citations give source lines.

|Work|Recorded time / machine / limit|Source, line|
|---|---|---|
|Full c*, Q|3090 trainer seeds0–4:316.97/243.47/212.37/296.09/241.31 s; train_c queue1,526 s, held export22 s|RSlog:108–109,134–151|
|Full c*, Q|A6000 training+export444 s, seed1 parity run|BSlog:16–17|
|c*, S2/S1/L14|A6000 about516 s/seed, pooled; individual encoder times not recorded|BSlog:30|
|Photo fine-tune Q|3090 seeds138.36/141.83/142.57/142.18/142.09 s, quoted trainer records; five-run stage2 wall910 s|BFacts:78; PFlog:85,96|
|Photo S2/S1/L14|A6000 about215 s/seed; later sweep units~200 s; individual encoder times not recorded|BSlog:30,56|
|ArtELingo+AE28 extraction|Local GPU B16 394+260 s,L14 621+50 s,S1 1,618+228 s,DN543+18 s; Q/S2/B32 individual extraction times not recorded here|EGlog:14|
|Affection val/test extraction|12,752 images:Q artwork656 s/photo652 s (~19.4/s);L14 92 s;FI completion1,553 s;local GPU|AClog:162–164|
|Affection train Q extraction|3090:72,255 instructed images3,630 s (~20/s,~350W);474,051 explanations2,142 s|PFlog:85,92–93|
|MMArt extraction|Local GPU16,782 images/store:Q default1,449 s(86.3ms/image),E1,489 s,S1,472 s,G1,466 s;S2 324 s(19.3ms),S1 323 s(19.2ms),L14 134 s(8.0ms);queue6,816 s|MPlog:84–91|
|r14 extraction|Queue11,601 s; individual Affection/EmoSet store times not recorded in round log|BSlog:28|
|One PercepT run|Single full-scale seed duration not recorded. Aggregate core all arms/seeds3,558 s,paper8,014 s GPU+CPU;D2 all seeds598 s. Not single-run estimates|R19:65–67|
|Earlier relative cost|D2 ~6times less training/seed,2.3times parameters vs core; protocol-specific|R21 Summary5/§2|
|clust.+bootstrap CPU|c* held verdict266 s;report comparisons/Leiden1,019 s;independent241 s;8threads. Earlier D2 stage2 verdict126 s,diagnostics1,870 s|RSlog:110–113;R19:70|
|Retrieval kernel benchmark|CPU8workers,12,281x2,048 random rows:k0 3.1 s,k5 5.7 s,graded4.2 s;437MB max worker;no bootstrap|RTlog:3–4|
|r11 retrieval+bootstrap|CPU8threads,22:10–23:02 incl.checks;stages1–3 and5 each<1min;read-out stage4 11.2min(c*290 s,PercepT139 s,others28–57 s);checks5min;GPU export~1min|RTlog:107–108;R29 What ran|
|r12 CPU stages+bootstrap|Ranges28 s,selection306 s,Affection184 s,CRL168 s;independent1,430/1,585 s,diagnostics168 s;whole2h36min,GPU~51min|AClog:162–184;BFacts:193|
|r13 sweep/scoring|Nine-row sweep1,840 s;final stage2 910 s;verdict95 s,derive266 s/compare4 s,diagnostics115 s;whole2h41min|PFlog:95–102|
|r14 CPU claims+bootstrap|ArtELingo clustering390–520 s/encoder,retrieval~120 s;Affection~90 s;EmoSet1,098 s;stage3 3,060 s;derive5,334 s,8threads|BSlog:77–81|
|r14 knob total|120/120units complete15:26;workers start08:33/08:39/08:52;retry15:09–15:25(282/272/263 s). Summed knob GPU time not recorded. Combined photo+knob19GPUh/~6.5h on3GPUs is estimate. Final CPU knob sheet2,448 s,39/39rows|BSlog:31–32,39,43,49,140–143,156|
|r14 CPU analysis|Three sheets4,534 s:knob2,448 s,photo209 s,ablation1,877 s;spot rederive1,301 s|BSlog:153–159|
|r10 removals|Local development24runs3,763 s(~63GPUmin),score1,528 s/derive647 s;held40runs15,124 s(~252GPUmin),score1,810 s/derive1,048 s|RSlog:393,416,459|
|r10 sweep/confirmation|Three A6000s17:08–18:29/18:30–19:39;local confirmation train932 s/verdict569 s/derive572 s;node exact step durations absent|R28 What ran;RSlog:39,64,84–86|
|r15 CPU+bootstrap|Local8threads,three parallel steps;stage0+1 3,120 s;stage2 2,189 s;derive5,177.9 s overlapping stage2;diagnostics3,053 s;no training|M15:252–258|

Disagreements: r14 handoffs17WIN/6LEVEL/1BELOW vs final18/5/1 [B14 Deviations]. r11 erroneous PercepT replay0.08549055120302276 vs correct full0.08742922158584242; fixed before final scores [R29 Deviations A2]. r13 compare followed verdict (378values,0mismatches) [P13 Deviations].
