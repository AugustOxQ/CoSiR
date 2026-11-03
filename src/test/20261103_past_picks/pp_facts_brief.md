# r19 facts
Sources (repo-relative): S14/S15 = docs/superpowers/specs/2026-11-02-{backbones-sweep,mmart-test}-design.md; R14/R15 = docs/reports/auto/percept/2027-01-{02_backbones_sweep,03_mmart_test}.md; T14/T15 = src/test/20261102_{backbones_sweep,mmart_test}; O14 = /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs; P = /data/SSD2/pre_extract. E/S/G = emotion/style/genre. R24/R25 = docs/reports/auto/percept/2026-12-{24_recipe_backbones,25_encoders_genre}.md. Also: E10=src/test/20261025_encoders_genre; B10=src/test/20261025_recipe_backbones; SC=src/test/20261103_semart_check; V=docs/paper/cvpr2027_aspect_spaces/drafts/v1; INV=src/test/20261103_experiment_overview/facts_inventory.md. Disk sources: ls/du.

## A. r14 bars
Quote: “WIN when the lower 95% bound > 0; LEVEL when not WIN and the lower bound > −r (the candidate's own range); BELOW otherwise.” r is unrounded validation max-minus-min across c* seeds 0 to 4, per aspect/task (AMI; mAP at R0). No multiplicity correction. [S14 §3.4]
Quote: “Paintings: holds when all nine cells of SC1 to SC3 WIN”; “qualified when SC1 and SC2 WIN in all six cells and one or more SC3 cells do not; fails otherwise.” Teacher BELOW gives no claim; failure means NO-GO, PercepT-only misses qualified GO. [S14 §6]
SC1/SC2: AMI/mAP vs plain image; SC3: AMI vs matched PercepT max(own-topic,20-cluster); SC6: Qwen teacher, separate claims. AMI: fixed K20, mean of 3 states; 2,000 painting/seed bootstrap draws. SC3 jointly draws paintings, independently draws model seeds, recomputes max; default_rng(0). Retrieval: 10,000 query/seed draws; r11 validation read-out rule. Validation stops training; held 12,281 (genre 9,949) scores only. “Surprise: a frozen-image miss on any aspect.” [S14 §§3.1,3.3–3.5,6]

## B. Non-Qwen execution
X default image enters trunk; only first-layer input width changes; output 2048. Caption input/graphs/teachers stay Qwen. Buddy graphs: instructed emotion caption mean, style/genre images; emotion/style corrected by default Qwen caption/image similarities; mutual k=40. [T14/bs_cstar.py:1–20; src/test/20261027_recipe_sweep/rs_graphs.py:90–114; src/test/20261020_claim_test/ct2_graphs.py:175–181]
Teacher: Qwen instructed image (emotion_focus/style_word/genre_word). Matched PercepT: P3 Stage2 on X image, saved Qwen Stage1 topics, same scale/seed; r14 reuses full M2/M8/M7 (S2/S1/L14), seeds 0–4. [S14 §3.2; B10/rb_matched.py:1–11]
ArtELingo train/val/held reuse X default images and Qwen supervision. New r14 extractions: Affection S2/S1 train+val+test, L14 train (val/test existed); EmoSet val+test six views: Qwen default, emotion artwork/photo, S2/S1/L14 default. [T14/bs_stores.py §folders/new_folder; bs_extract_queue.sh:43–51]
Commands: `bs_node.py part-a --encoder E --device cuda` trains seeds0–4 via bs_cstar.py; `bash bs_run_cpu.sh stage3` calls bs_artclu.py/bs_artret.py. [T14/bs_node.py:4–14; bs_run_cpu.sh]

## C. r14 costs
Local extraction seconds (pooled splits): Affection S2 1583, S1 1784, L14-train 361; EmoSet default Qwen 2309, emotion artwork 2193, photo 2244, L14 156, S2 431, S1 474; queue 11601s. ArtELingo reused. [O14/stage0/extract_queue_status.txt]
Measured A6000/node402 c* unit seconds, seeds 0–4: S2 466.2/516.0/517.2/518.0/506.5; S1 438.5/503.8/496.1/505.9/499.0; L14 494.5/512.3/505.5/496.8/476.2. Totals include overhead/export. Pure training seed0: S2 369.2657499713823, S1 340.8912113448605, L14 398.13894561491907; other seeds in record. [O14/node/units.jsonl, kind=cstar]
PercepT: reuse 5 runs/encoder. CPU scoring: ArtELingo clustering 390–520s/encoder, retrieval about 120s; stage3 3060s; rederive 5334s, 8 threads. GPU peak absent; capacity 48 GB. Extraction local, training DAS6, scoring local CPU. [S14 §§3.2,3.5; T14/bs_run_log.md:16–30,77–81; bs_facts_brief.md:80]

## D. r15 MMArt
16,782 outside AE -> 16,384: drop 34 byte copies; default-Qwen cosine >=0.95 to all 49,121 AE train+val; remaining md5 duplicates keep first sha256-key (8 dropped). Shared artists; unseen diagnostic 15 paintings/12 artists, no cells. [S15 §§3.3,3.6; R15 Results brief; facts_inventory.md A3]
Frozen c*, seeds0–4; K20, three states; K9/20/10 diagnostic. 5,769 relabelled AE validation paintings pick ranges/retrieval read-out (one per arm across aspects); teacher once with Qwen ranges. Bars own plain + frozen matched PercepT (AE topics), teacher separate. `mt_export.py`, `mt_matched.py`, `mt_clu.py`, `mt_ret.py`; `bash mt_run_cpu.sh first`, then `stage2`. [S15 §§3.2–3.5; T15/mt_run_cpu.sh; round log Run]
Per encoder: Qwen holds (emotion mAP teacher BELOW); S2 holds; S1 holds; L14 holds (emotion mAP teacher BELOW). All nine plain/matched cells WIN per encoder. Teacher 18 WIN/4 LEVEL/2 BELOW; Qwen/L14 emotion retrieval BELOW. [R15 Results tables]

## E. r10 and ArtELingo stores
R6-5 (pre-c*), AE held12,281, five seeds, K20 AMI: B32 full 0.0885 inconclusive; B16 full 0.0945 inconclusive; DINOv2-large sample 0.0814 NO-GO. DINO full skipped because sample failed; sample training 13,400 vs full 42,969 (val 1,489 vs 6,152). [R24 Results/What ran; R25 Results stage i]
P/artelingo_full/: clip_b16 397M, clip_b32 397M (img+txt), dinov2_large 128M (image only); also clip_l14 578M, siglip2_so400m 849M, siglip_so400m 849M, qwen3vl_emb_2b 1.5G, its _aspect_prompts 7.6G, _attention 8.1G, roberta_go_emotions 342M, _s2 501M. Train/val/held coverage inferred from full adapter (61,402 images) and store gates; no split folders. [ls/du P/artelingo_full; scripts/extract_features.py:205–230; R25 Analysis V5]
Other roots: P/artelingo/ cross_encoder_stage1_features 3.1G , features 1.2G (B32), r8_enc_aladin 49M, r8_enc_dinov2l 193M, r8_enc_siglip2 217M, style_csd_vitl 187M, style_vgg19_gram 1007M; P/artelingo_heldout/features 184M (legacy CLIP). [ls/du named roots; features/metadata.json]
Matched outputs exist: B32 B10/outputs/part2/full/held/M3; B16 /data/SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/full/held/M6; DINO same root sample/held/M5. Files P3__s0..4.{npz,json}; sibling matched/M* mappers; metrics part2/reported_stage{3,4}.{json,md} (DINO stage3), part2/partitions. [ls paths; B10/rb_matched.py Files]
Extractor choices: clip_b32,qwen3vl_emb_2b,siglip2_so400m,dinov2_large,clip_b16,clip_l14,siglip_so400m; DINO requires --image-only. [scripts/extract_features.py:44–45,623–624]

## F. SemArt and MMArt stores
/data/SSD/semart/SemArt/semart_{train,val,test}.csv: 19244/1069/1069 samples. Stores pool all three splits: P/semart/clip_b32 45M, qwen3vl_emb_2b 170M, siglip2_so400m 97M; img+txt in each. No DINO/B16/S1/L14/teacher store. Genre teacher needs fresh Qwen genre_word images for evaluation/validation (inferred); log estimate about 35 GPU minutes. [SemArt/README.txt:13–15; ls/du P/semart; extract_features.py:234–249; SC log Open]
CPU check 47s, 21,382 vs AE train+val; >=0.85/0.90/0.95/0.99 counts 12,176/6,292/3,542/1,670; kept 17,840; internal pairs at .95 1,095. New artists after cut, unknowns excluded 11,162 paintings/2,717 artists; 1,249 >=.90, 217 >=.93. Type10, school26, period22; no emotion/style. Copies survive .95 (Last Supper .944); alias/surname/wildcard matching leaves possible name misses. [same check log Results/Findings; sc_check.py artist_tiers/main]
Earlier SemArt use: archived conditioned line had development probes, no claim; not c*. [archive/v2_conditioned/docs/reports/auto/v2/2026-12-02_aff_scope.md:42]
MMArt P/mmart/: Qwen 67M, prompted 200M (emotion_focus/style_word/genre_word), S2 38M, S1 38M, L14 26M, parity 4.5M, logs 12K; main stores test-only 16,782, validation reused AE. No DINO/B32/B16. [ls/du P/mmart; S15 §§3.3,3.5]

## G. Cost-table timing facts
Measured local MMArt 16,782/store: Qwen default 1,449s (86.3ms/image), E 1,489s, S 1,472s, G 1,466s; S2 324s, S1 323s, L14 134s. [src/test/20261102_mmart_prep/20261102_mmart_prep_log.md:84–91; log P/mmart/logs/queue_20261102_180138.log]
Measured Qwen instructed Affection val/test ~19.4 images/s; train ~20/s (3090). [INV E, affection log:162–164; photo log:85,92–93]
One plain encoder pass shared across aspects vs one Qwen instructed pass per aspect; three student trunk/head networks at test, not one shared trunk execution. [V/03_method.md:58,244–248]

## H. Paper
V/03_method.md §3.7: “We changed no setting for any encoder” (274–278; paper_v1.md:418–422), naming S2/S1/L14. DINO/B loss touches broader encoder generalisation (inferred). §3.2: teacher pass claim (:58); §3.6: shared encoder pass (:244–248). MMArt appears as related work, paper_v1.md:120, four annotated perspectives; SemArt :128 also related work. No experiment/limitation sections found (grep/ls). [V listing and keyword grep]

## I. r17
Not written yet; planned src/test/20261103_<name>/; embeddings+labels -> metrics, imported by r18/r19. AMI/NMI/ARI/ACC on claim cells; sweep AMI/ARI, NMI/ACC at class count only (latter recommendation); user emotion K=5,8,16,20, other counts open. [docs/superpowers/handoffs/2026-11-03-r17-decide-handoff.md §§2,4–6; ls specs/src/test]

## Gaps
Missing: split-specific extraction times, DINO/B throughput, isolated c* head cost, GPU peaks. Legacy r8 splits and SemArt residual copies/name misses unresolved. Removal list: none. 

## Added by Claude (r19 decide, 2026-11-03 19:00 to 19:25), checked by hand

- Spot checks of the brief above: r14 bar lines (S14:215, 351 to 352), store shapes, paper §3.6 and §3.7 lines: no
  wrong fact found.
- ArtELingo stores, all 61,402 paintings, float16: `P/artelingo_full/dinov2_large/img.npy` (61402, 1024; image only),
  `clip_b16` and `clip_b32` (61402, 512; img and txt).
- Affection stores (`P/affection_local/`): clip_l14_train, clip_l14_valtest, qwen3vl_emb_2b (+ aspect prompts),
  siglip2_so400m, siglip_so400m. No CLIP B/32, B/16 or DINOv2 store. EmoSet: no store for the three either.
- r10's matched-arm cost: 15 to 16 s per seed at the sample scale, 35 to 46 s at full scale (10-26 run log, lines 20
  and 22). r10 extraction on ArtELingo plus ArtELingo-28: CLIP B/16 394 + 260 s, DINOv2 543 + 18 s (line 14).
- r14 photo fine-tune per encoder and seed on a node402 A6000 (`O14/node/units.jsonl`, kind `xphoto`): 180 to 260 s.
- r10's emotion development lift table (report 12-25, R6-5 students on 13,400 training paintings, 29,232 development
  paintings; student AMI / own PercepT arm's best / frozen image): CLIP B/32 0.0905 / 0.0708 / 0.0624; CLIP B/16
  0.0952 / 0.0728 / 0.0643; DINOv2 0.0839 / 0.0722 / 0.0587 (the other four: SigLIP 2 0.1042 / 0.0756 / 0.0571, CLIP
  L/14 0.1031 / 0.0772 / 0.0648, SigLIP 1 0.1073 / 0.0784 / 0.0631, Qwen 0.1050 / 0.0844 / 0.0690).
- r10's held bar was another bar: emotion AMI at least 0.0100 above PercepT core's strongest entry. B/32 0.0885 vs
  0.0874, +0.0011 [−0.0043, +0.0025]; B/16 0.0945 vs 0.0874, +0.0070 [+0.0016, +0.0085]; DINOv2 (sample) 0.0814 vs
  0.0855, −0.0041 [−0.0094, −0.0030] (report 12-25 lines 64 to 73; report 12-24 line 29).
- SemArt close pairs (`src/test/20261103_semart_check/sc_nearcopy.parquet`, new-artist rows below 0.95, names only,
  images not viewed): my filter gave 11,205 rows and 1,250 at ≥ 0.90 (the log's 11,162 and 1,249 use the log's exact
  filter; the build recomputes from `sc_check.py`). The 12 closest are mostly different paintings of the same subject
  by different artists (Franciabigio's Last Supper vs Andrea del Sarto's, 0.950; Marieschi vs Canaletto, Grand Canal,
  0.948; Grünewald vs Terbrugghen, Crucifixion, 0.949); one family copy (Pieter Brueghel the Younger's Proverbs after
  his father's, 0.947). Type shares, all new-artist rows vs the ≥ 0.90 rows: religious 0.370 vs 0.503, portrait 0.150
  vs 0.210, landscape 0.138 vs 0.101, genre 0.079 vs 0.024. So a 0.90 cut would mostly drop real new-artist paintings
  of common subjects and skew the type mix; the log's "mostly workshop versions" is not borne out at the top.
- MMArt measured extraction on the 3090 (16,782 paintings): Qwen default 1,449 s, emotion 1,489 s, style 1,472 s,
  genre 1,466 s; SigLIP 2 324 s, SigLIP 1 323 s, CLIP L/14 134 s (mmart prep log lines 84 to 91).
- DAS6 node402 at 18:45: all three A6000 GPUs free (0 MiB), 115 h 14 min left (ends about 2026-11-08 14:00), CoSiR
  env ok, no running jobs. The handoff's "blocked" note no longer holds.
- Codex at 18:46: five-hour window 100%, weekly 100% (reset); after this brief 85% and 98%.
