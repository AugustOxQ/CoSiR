# Coverage after r17-r24

Inventory through 2026-11-08. Only this file written; nothing deleted.

Sources: 01-xx = **2027 sequence dates**, not run dates. D11-xx = dated direction entry; DP = "Parked ideas"; DN = "Now".

- 11-04: `docs/reports/stage/2026-11-04_datasets_and_retrieval.md`, coverage tables and earlier results.
- O11-03: `docs/user_read/2026-11-03_experiment_overview.md`, to-do list and decided plan.
Auto-report paths share `docs/reports/auto/percept/`.

- 01-04: `2027-01-04_new_scores.md` (r17).
- 01-05: `2027-01-05_past_picks.md` (r19).
- 01-06: `2027-01-06_outside_methods.md` (r18).
- 01-07: `2027-01-07_teacher_gap.md` (r17 follow-up).
- 01-09: `2027-01-09_text_retrieval.md` (r22).
- 01-10: `2027-01-10_pair_hit_checks.md` (r22 follow-up).
- 01-11: `2027-01-11_text_claim.md` (r23).
- 01-12: `2027-01-12_miss_check.md` (r24).
- D/DP/DN: `docs/superpowers/direction_log.md`.

F = Qwen, SigLIP 2, SigLIP 1, CLIP L/14. N = DINOv2, CLIP B/32, CLIP B/16. Seven = F + N. E/S/G = emotion/style/genre. Paintings: E/S/G; photos: E; SemArt: genre/type. Teacher = instructed Qwen. **Comparator outcomes mean candidate versus comparator.** Claim ref = claim comparator; descriptive gives no claim.

## Clustering coverage

Class-count metrics: AMI/NMI/ARI/ACC. r19 claims use AMI at 20; other metrics descriptive. (01-04, 01-05)

| Method | ArtELingo held | MMArt test | Affection test | EmoSet test | SemArt new artists |
|---|---|---|---|---|---|
| Our painting model | F, claim: plain metrics WIN; teacher 33 WIN/13 LEVEL/2 BELOW. N claim holds vs plain/PercepT. (01-04, 01-05) | F, claim: plain metrics WIN; teacher 35 WIN/10 LEVEL/3 BELOW. N claim holds vs plain/PercepT. (01-04, 01-05) | Qwen failed confirmation, below teacher; others not planned. (11-04) | Qwen descriptive; others not planned. (11-04) | Seven, claim: holds, AMI WIN vs plain/PercepT. F teacher AMI LEVEL; N mostly BELOW. (01-05) |
| Photo model | not planned. (11-04) | not planned. (11-04) | F, claim: plain metrics WIN; teacher 14 WIN/2 LEVEL. N, claim: holds; DINOv2 teacher BELOW. (01-04, 01-05) | Qwen claim: all four metrics WIN vs plain/teacher; other F descriptive. (01-04, 11-04) | not planned. (11-04) |
| Plain image | Seven, claim ref: F all class-count metrics WIN; N AMI WIN. (01-04, 01-05) | Seven, claim ref: F all class-count metrics WIN; N AMI WIN. (01-04, 01-05) | Seven, claim comparator for photo model: F all metrics WIN; N AMI WIN. (01-04, 01-05) | Qwen claim ref: all metrics WIN; other F descriptive. (01-04, 11-04) | Seven, claim ref: AMI WIN. (01-05) |
| Teacher | Qwen claim ref: F 33 WIN/13 LEVEL/2 BELOW; N teacher claims restricted, no "better than". (01-04, 01-05) | Qwen claim ref: F 35 WIN/10 LEVEL/3 BELOW; N mostly BELOW. (01-04, 01-05) | Qwen, photo claim ref: F 14 WIN/2 LEVEL; N mixed; CLIP B/16 AMI WIN, licence "level with or better than". (01-04, 01-05) | Qwen, photo claim ref: all four metrics WIN. (01-04) | Qwen claim ref: F AMI LEVEL; no teacher superiority claim. (01-05) |
| PercepT on the same encoder | Seven, claim ref: F AMI/NMI/ARI WIN; SigLIP 1 E ACC LEVEL. N AMI WIN. (01-04, 01-05) | Seven, claim ref: F AMI/NMI/ARI WIN; SigLIP 2 and CLIP L/14 E ACC LEVEL. N AMI WIN. (01-04, 01-05) | Qwen, descriptive; additions not planned. (11-04) | not planned. (11-04) | Seven, claim ref: AMI WIN. (01-05) |
| CRL | CLIP L/14, claim ref: AMI/NMI WIN; E ARI/ACC LEVEL. (01-04) | CLIP L/14, claim ref: AMI/NMI/ARI WIN; E ACC LEVEL, S ACC BELOW. Qwen c* comparison descriptive. (01-04) | Qwen and CLIP L/14, descriptive. (11-04) | CLIP L/14, descriptive. (11-04) | not planned. (11-04) |
| CLAY | CLIP L/14 ref to Qwen/CLIP L/14 c*: Qwen S AMI LEVEL; other cells WIN. (01-06) | CLIP L/14 features, claim ref: Qwen S AMI/NMI LEVEL; CLIP L/14 S ACC BELOW; other cells WIN. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| TPIPS | Qwen 8B ref to Qwen c*: E ARI LEVEL; other cells WIN. Direction log calls it AMI; see differences. (01-06, D11-05) | Qwen 8B ref to Qwen c*: all cells WIN. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| TURTLE on plain images | F, claim comparator, released defaults: c* k-means WIN in every cell. (01-06) | F, claim comparator, released defaults: c* k-means WIN in every cell. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| TURTLE on c* heads | F claim vs plain TURTLE, defaults: SigLIP 2 E ARI LEVEL; other cells WIN. (01-06) | F claim vs plain TURTLE, defaults: SigLIP 1 E ACC LEVEL; other cells WIN. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| TEMI | not run: deferred. (01-06, D11-05) | not run: deferred. (01-06, D11-05) | not planned. (11-04) | not planned. (11-04) | not planned. (11-04) |

Default TURTLE, no selection: below k-means in 203 of 232 diagnostics; stronger-than-k-means comparator unestablished. (01-06)

## Retrieval coverage

Image-to-image, told unless marked untold. Untold joins heads. No new frozen painting-model photo confirmation. (01-04)

| Method | ArtELingo held | MMArt test | Affection test | EmoSet test | SemArt new artists |
|---|---|---|---|---|---|
| Our painting model | F, claim: plain WIN; teacher E CLIP L/14 BELOW, other E LEVEL. N, claim: holds vs plain. (O11-03, 01-05) | F, claim: plain WIN; teacher E Qwen/CLIP L/14 BELOW, other E LEVEL. N, claim: holds vs plain. (O11-03, 01-05) | Qwen failed confirmation, below teacher; others not planned. (11-04) | Qwen descriptive; others not planned. (11-04) | Seven, claim: plain WIN; F teacher mAP LEVEL except CLIP L/14 BELOW; N mostly BELOW. (01-05) |
| Photo model | not planned. (11-04) | not planned. (11-04) | Seven, claim: holds, plain WIN. F teacher claims include LEVEL; N mixed, CLIP B/32/B/16 mAP WIN with restricted licence. (11-04, 01-05) | Qwen claim: WIN vs plain/teacher; other F descriptive. (11-04) | not planned. (11-04) |
| Plain image | Seven, claim ref: told mAP WIN. (11-04, 01-05) | Seven, claim ref: told mAP WIN. (11-04, 01-05) | Seven, photo claim ref: mAP WIN. (11-04, 01-05) | Qwen, photo claim ref: WIN; other F descriptive. (11-04) | Seven, claim ref: genre mAP WIN. (01-05) |
| Teacher | Qwen claim ref: F E LEVEL except CLIP L/14 BELOW; N restricted licence. (O11-03, 01-05) | Qwen claim ref: F E Qwen/CLIP L/14 BELOW, others LEVEL; N mostly BELOW. (O11-03, 01-05) | Qwen, photo claim ref: F holds; N mixed, DINOv2 BELOW. (11-04, 01-05) | Qwen, photo claim ref: WIN. (11-04) | Qwen claim ref: F LEVEL except CLIP L/14 BELOW; no teacher superiority claim. (01-05) |
| PercepT on the same encoder | F, claim ref: all told cells WIN; N retrieval not planned. (01-04, 11-04) | F, claim ref: all told cells WIN; N retrieval not planned. (01-04, 11-04) | Qwen, descriptive; additions not planned. (11-04) | not planned. (11-04) | not planned. (11-04) |
| CRL | CLIP L/14 same-encoder claim ref: all cells WIN; Qwen CRL descriptive. (01-04, 11-04) | CLIP L/14 same-encoder claim ref: all cells WIN; Qwen c* vs CLIP L/14 CRL descriptive, positive intervals. (01-04) | CLIP L/14, claim ref: photo model WIN; frozen c* LEVEL, failed WIN bar. Qwen CRL descriptive. (11-04) | CLIP L/14, descriptive. (11-04) | not planned. (11-04) |
| CLAY | CLIP L/14 features, claim ref: Qwen/CLIP L/14 c* all WIN. (01-06) | CLIP L/14 features, claim ref: Qwen/CLIP L/14 c* all WIN. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| TPIPS | Qwen 8B, claim ref: Qwen c* all WIN. (01-06) | Qwen 8B, claim ref: Qwen c* all WIN. (01-06) | not planned. (01-06) | not planned. (01-06) | not planned. (01-06) |
| Untold c*, joined heads | F, claim vs plain: E WIN; SigLIP 1 S BELOW, Qwen G LEVEL; remaining S/G WIN. (01-04) | F, claim vs plain: all cells WIN. (01-04) | not run; painting heads only. (01-04) | not run; painting heads only. (01-04) | not run. (01-04, 01-05) |

Untold: 22 WIN/1 LEVEL/1 BELOW. Descriptive untold-minus-told intervals all below zero. No r19 PercepT retrieval on N/SemArt. (01-04, 01-05, 11-04)

## Text retrieval coverage

Slash order: **t2i/i2t/t2t**, text-to-image/image-to-text/text-to-text. Separate number lines: pair R@10; E/S/G identify heads; dash = unprinted. Claims use Qwen. MMArt test uses new paintings/narrative; shared = 11,528 held ArtELingo paintings, descriptive. (01-09 to 01-12)

| Method | ArtELingo validation | ArtELingo held | MMArt test | MMArt shared rows | SemArt validation |
|---|---|---|---|---|---|
| c* caption side | Qwen descriptive: Outcome 1; above plain 9/9, above both 6/9; E below teacher, S/G above. (01-09)<br>c* pair R@10: in report 01-09 (D1: not copied here). | Qwen claim: holds; plain 9 of 9 WIN, teacher 6 of 9 WIN; E BELOW, S/G WIN. (01-11)<br>c* pair R@10: in report 01-11 (D1: not copied here). | Qwen claim: fails; plain 8 of 9 WIN; teacher S WIN, G t2i LEVEL, others BELOW. G t2t vs plain LEVEL. (01-11, 01-12)<br>c* pair R@10: in report 01-11 (D1: not copied here). | Qwen descriptive: captions plain/teacher WIN counts 9/6; narrative 8/3, either label set. G t2t vs plain: captions WIN, narrative BELOW. (01-12) | Qwen, descriptive, descriptions; no label claim. (01-09)<br>c* pair R@10: in report 01-09 (D1: not copied here). |
| Late fusion, a = 0.7 on plain Qwen | Qwen descriptive: pass; 5 of 9 label cells above both. (01-10)<br>0.3215/0.4954/0.1952. (01-10) | Qwen claim: holds; plain 9 of 9 WIN, teacher 5 of 9 WIN; pair bar passes. (01-11)<br>0.2541/0.4106/-. (01-11) | Qwen claim: fails; plain 9 of 9 WIN, teacher 0 of 9 WIN; pair bar passes. (01-11)<br>0.8379/0.8922/-. (01-11) | Qwen descriptive: captions plain/teacher counts 9/4; narrative 9/0, either label set; captions S i2t LEVEL. (01-12) | not run. (01-09, 01-10) |
| Plain Qwen | Qwen descriptive ref: c* above, all labels. (01-09)<br>0.3102/0.5068/0.1823. (01-09) | Qwen claim ref: c*/fusion 9 of 9 WIN each. (01-11)<br>0.2512/0.4136/-. (01-11) | Qwen claim ref: c* 8 of 9 WIN, fusion 9 of 9 WIN. (01-11)<br>0.8493/0.9006/-. (01-11) | Qwen descriptive ref: c* 9 WIN with captions, 8 with narrative; fusion 9 under both. (01-12) | Qwen, descriptive, scrubbed text. (01-09)<br>0.5968/0.5341/-; raw text 0.6268/0.5529/-. (01-09) |
| Teacher's text | Qwen descriptive ref: c* E below, S/G above. (01-09)<br>Best heads G/E/G: 0.1604/0.2466/0.1261. (01-09) | Qwen claim ref: c* 6 WIN, fusion 5 WIN. (01-11) | Qwen claim ref: c* 3 WIN and G t2i LEVEL; fusion all BELOW. (01-11) | Qwen descriptive ref: c* captions/narrative 6/3 WIN; fusion 4/0 WIN. (01-12) | not run. (01-09) |
| Free variant, captions through image side | Qwen descriptive: above both in 3 of 9 label cells. (01-09)<br>Best heads E/E/G: 0.1984/0.3708/0.1394. (01-09) | Qwen, descriptive: c* above, all t2i. (01-11) | not reported. (01-11) | not run. (01-12) | Qwen descriptive G. (01-09)<br>0.4687/0.3994/-. (01-09) |
| CLIP L/14 | Descriptive: c* above, all labels; exact pair row not printed. (01-09) | Descriptive: c* above, all t2i. (01-11) | not run. (01-11) | not run. (01-12) | not checked; text absent. (01-09) |
| SigLIP 2 | Descriptive: c* above, all labels. (01-09)<br>0.3180/0.5382/-. (01-09) | Descriptive: c* above, all t2i. (01-11) | not run. (01-11) | not run. (01-12) | Descriptive. (01-09)<br>0.5154/0.4808/-. (01-09) |
| SigLIP 1 | Descriptive: c* above, all labels. (01-09)<br>-/-/0.1516. (01-09) | Descriptive: c* above, all t2i. (01-11) | not run. (01-11) | not run. (01-12) | not checked; text absent. (01-09) |
| CLIP B/32 | not run. (01-09) | not run. (01-11) | not run. (01-11) | not run. (01-12) | Descriptive. (01-09)<br>0.4331/0.4200/-. (01-09) |
| RoBERTa, 768-d | Descriptive E t2t label mAP 0.3099; no t2i/i2t. (01-09) | Descriptive side row; no claim bar. (01-11) | not run. (01-11) | not run. (01-12) | not run. (01-09) |
| Chance | Descriptive E/S/G label-mAP baseline, all directions. (01-09) | not printed. (01-11) | not printed. (01-11) | not printed. (01-12) | not printed. (01-09) |

Other validation populations and query forms:

| Row | MMArt validation, five perspectives | ArtELingo validation, caption means and words |
|---|---|---|
| c* caption side | Qwen, descriptive t2i, N/F/E/H/U perspectives. Label matches: E head leads emotional E labels; S head leads all perspectives' S labels. (01-09)<br>c* pair R@10: in report 01-09 (D1: not copied here). | Qwen, descriptive caption-mean t2i label mAP E/S/G 0.2905/0.2036/0.5378. Word-query leader: E template t2i, S t2t, G template t2t. (01-09)<br>c* pair R@10: in report 01-09 (D1: not copied here). |
| Free variant | Qwen descriptive: E head leads N/F/U pairs; G leads E/H pairs. (01-09) | Qwen descriptive: leads G template t2i macro mAP 0.7138. (01-09) |
| Plain Qwen | no comparator row printed. (01-09) | Qwen, descriptive caption-mean t2i. (01-09)<br>0.7243. (01-09) |
| Teacher's text | not run by scope. (01-09) | Qwen, descriptive word queries: leads remaining rows named in report. (01-09) |
| Plain CLIP L/14, SigLIP 2, SigLIP 1 | no comparator row printed. (01-09) | Descriptive word queries: SigLIP 2 leads plain encoders in S/G t2i; none leads overall. (01-09) |

N/F/E/H/U = narrative/formal/emotional/historical/unified. Caption-mean pair line: E/S/G t2i. MMArt validation G t2t: c* 0.4628, plain 0.4778; fusion above teacher 0 of 9. (01-09, 01-11)

Older saved recipes, ArtELingo validation only, descriptive; no c* retraining:

| Row | Label outcome and pair R@10, t2i/i2t/t2t |
|---|---|
| R0-5, own-caption on | Label mAP lower in all 9 cells versus off; own-term check pass. (01-10)<br>0.2538/0.4161/0.1748. (01-10) |
| R6-5, own-caption off | Comparator; sample-scale older recipe. (01-10)<br>0.1724/0.2947/0.1450. (01-10) |

## Status of the to-do list

Status records execution, not success.

| Item | Loop | Status | Outcome (at most 15 words) | Source |
|---|---|---|---|---|
| 1. NMI, ARI, ACC | r17 | done | Plain class-count WIN; teacher ARI/ACC mixed; r19 metrics descriptive. | 01-04, 01-05 |
| 2. Cluster-count sensitivity | r17 | done | Class-count claims done; other-count sweeps descriptive. | 01-04 |
| 3. Stronger plain clustering | r18 | done, partly | Default TURTLE done; stronger-than-k-means test open; selected TURTLE/TEMI deferred. | 01-06, D11-05 |
| 4. DEC, with/without initial-cluster pull | r21 | not run: waiting | Head variants wait. | DN |
| 5. Criterion-driven clustering | cite/explain | paper text only | Not run; large build/per-image calls exceed schedule. | 11-04, O11-03 |
| 6. New-artist SemArt genre | r19 | done | Seven hold vs plain/PercepT; no teacher superiority claim. | 01-05 |
| 7. Same-encoder CRL | r17 | done | CLIP L/14: 6 of 6 retrieval cells WIN across both painting sets. | 01-04 |
| 8. CRL on MMArt | r17 | done | Same-encoder retrieval WIN; clustering includes LEVEL and BELOW. | 01-04 |
| 9. Untold retrieval | r17 | done | 22 WIN, 1 LEVEL, 1 BELOW; told comparison descriptive. | 01-04 |
| 10. Caption-side queries | r22, r23, r24 | done | c*/fusion hold ArtELingo, fail MMArt; narrative check done. | 01-09-01-12 |
| 11. Teacher emotion gap | r17 and check | done | Contentment explains ArtELingo clustering gap; MMArt ACC partial. | 01-04, 01-07 |
| 12. MMArt perspective captions | r22, r23, r24 | done, partly | Five perspectives explored; narrative claim tested; shared rows narrative only. | 01-09, 01-11, 01-12 |
| 13. TPIPS timing probe | r18 decide | done | 219 ms per image per factor, batch 1, RTX 3090. | direction log 2026-11-03, r18 grill |
| 14. TPIPS comparator | r18 | done | Qwen c*: clustering 28 WIN/1 LEVEL; retrieval 6 WIN. | 01-06 |
| 15. CLAY comparator | r18 | done | Qwen/CLIP L/14 c*: clustering 54 WIN/3 LEVEL/1 BELOW; retrieval 12 WIN. | 01-06 |
| 16. Previously unsuccessful encoders | r19 | done | All nine new-encoder verdicts hold; teacher mostly BELOW. | 01-05 |
| 17. Second instructed embedder | unscheduled | not run: deferred | Important; priority decision left for later. | O11-03, 11-04 |
| 18. Deeper residual heads | r21 | not run: waiting | Head variants wait. | DN |
| 19. One-pass cost | r19 | done | Seven cost verdicts hold against three teacher passes; shared-image teacher unmeasured. | 01-05 |
| 20. Qualitative cluster figure | r20 | not run: waiting | Figure waits; design with user remains named scope. | DN, O11-03 |
| 21. Second photo aspect | cite/explain | paper text only | Not run; new pathway and access risk; direction log defers it. | 11-04, DP |
| 22. Frozen encoder retrieval note | paper | paper text only | Wording item; no experiment. | 11-04, O11-03 |
| 23. Training-domain/context rows | paper | paper text only | Outside-method rows decided; insertion into paper left to later chat. | 11-04, D11-05 |
| Added: CRL clustering on paintings | r17 | done | CLIP L/14: 26 WIN/3 LEVEL/1 BELOW; AMI/NMI all WIN. | 01-04 |
| Added: PercepT retrieval, F/MMArt | r17 | done | F paintings: 24 of 24 WIN; no photo expansion. | 01-04, 11-04 |

Parked ideas entries dated since 2026-11-02:

| Parked item | Status | Recorded execution/status | Source |
|---|---|---|---|
| Photo style/mood: Flickr Style or AVA | not run: deferred | New pathway/access risk; old stage maps it to cite/explain after paper. | DP 11-02; 11-04 |
| MMArt perspective retrieval | done, partly | DP says "never tried"; perspective validation/narrative claim/shared check ran. | DP 11-02; 01-09, 01-11, 01-12 |
| MMArt perspective names as new teacher instructions | not run: parked | Narrative/formal/historical as new aspects would require new heads. | DP 11-02 |
| SemArt new-artist genre | done | DP says "never tried"; seven-encoder claim ran, 11,159 paintings. | DP 11-03; 01-05 |
| TPIPS as replacement teacher | not run: deferred | Retrain deferred for lack of time before deadline. | DP 11-03 |
| Photo random weights, 80 neighbours, five seeds | not run: waiting | Decided after r19 frees GPU; supplied reports establish no completed run. | DP 11-04 |
| Cost against teacher encoding image once | not run: parked | Not chosen; report has arithmetic, not measured cost. | DP 11-05; 01-05 |
| Emotion/style on new artists | not run: parked | Not chosen; data search needed, SemArt has neither label. | DP 11-05; 01-05 |
| MMArt miss: other texts; author/length/text-count | not run: waiting | Plan update decides; narrative check leaves cause open. | DP 11-08; 01-12 |
| TEMI, r18 stage three | not run: deferred | Worth doing, not top priority; no run. | DP/D11-05; 01-06 |
| Selected TURTLE on validation then plain test | not run: deferred | Not top priority; spec must settle D9 versus D25. | DP/D11-05; 01-06 |

## Open points named by the reports

Duplicate limits grouped.

- Claim cells stand separately; no pooled r17/r18 verdict or multiplicity correction. (01-04, 01-06; correction also 01-05, 01-11, 01-12)
- Candidate test values sometimes preceded bars; ArtELingo CRL retrieval was previously seen. (01-04, 01-06)
- MMArt shares most training artists; new-artist claims cover genre/type only, matched by artist names. (01-04, 01-05)
- SemArt emotion/style/schools/periods untested; close copies and internal near-copy pairs remain. (01-05)
- Teacher superiority is metric-specific; all five r17 teacher BELOW cells are emotion. (01-04)
- LEVEL can have a wholly negative interval; BELOW can reach zero; validation ranges imperfectly reflect test spread. (01-04, 01-05)
- Non-class-count sweeps give point values, no claim labels. (01-04, 01-06)
- Teacher outcome retention for ArtELingo Qwen's 12 cells was not computed; older records were not checked. (01-04)
- CRL clustering had no stored partitions; cross-encoder diagnostics are unlicensed. (01-04)
- PercepT class-count k-means is weaker than its earlier larger-of-two-readings comparison. (01-04)
- AP and class decomposition do not establish training causes of clustering gaps. (01-04, 01-07)
- Post-result masks/weights give no claim; equal-weight/within-genre values lack intervals; control classes are smaller. (01-07)
- Other landscape rows' destinations and emotion splits were not checked; fear's MMArt ACC contribution was not masked. (01-07)
- ArtELingo validation's teacher gap is smaller, mostly absent; genre-as-partition gives no baseline claim. (01-07)
- Whether new heads close the gap, and why teacher gaps vary by encoder, remain untested. (01-07, 01-05)
- r19 teacher WINs outside preset scope license no "better than"; additional metrics are descriptive. (01-05)
- Cost covers RTX 3090, batch 32, model time; no training-cost or shared-image teacher claim. (01-05)
- Shared-image teacher cost is arithmetic; decoding overlaps compute; zero end-to-end warm-up batches recorded instead of three. (01-05)
- Older DINOv2 comparison uses sample-scale student against full-scale c*. (01-05)
- Default TURTLE lacked selection and did not establish stronger-than-k-means clustering. (01-06)
- Re-implemented TURTLE parity covers a short official-code fit and one full cross-machine fit. (01-06)
- TURTLE LEVEL uses k-means ranges; diagnostic c* intervals may be too narrow; unit-row form differs from k-means. (01-06)
- Empty hard clusters remain despite balanced soft assignments; entropy-weight explanation untested. (01-06)
- TPIPS vectors depend on library version; node parity hardware cause and WikiArt training overlap were not checked. (01-06)
- Re-derivation does not repeat TPIPS extraction or TURTLE training; CLAY selected lists contain class names. (01-06)
- Selected TURTLE, TEMI, other sets/encoders and CLAY set B have no claims. (01-06)
- r17 photo replay checks hashes, not row order; PercepT argmax unreplayed; partition diagnostics report nonexistent draws. (01-04)
- r18 part files lack input hashes; CLAY/TPIPS gates lack code hashes; final reviews stopped before full coverage. (01-06)
- r18 input-trust notes remain; unchanged-input checks and re-derivation carry evidence. (01-06)
- r22/follow-ups use one seed and validation only; picks used scored rows; cross-fit followed first reading. (01-09, 01-10)
- Genre i2t/t2t validation advantage depends on read-out; genre t2t needs expansion against teacher. (01-09)
- Caption side trained on means; single captions/words are new; mean comparisons establish no training mismatch. (01-09)
- Painting emotion uses caption plurality; MMArt emotional text leaks labels; generated/conditioned text may favour Qwen encoders. (01-09, 01-11)
- SemArt used validation, not published test; paper/README numbers conflict; metadata rows excluded; Lenses/CANVAS not re-checked. (01-09)
- Own-caption check uses older sample-scale recipes/graphs without genre relational term; effects on c* and clustering remain unknown. (01-10)
- Joined validation labels use means without intervals; only one grid weight passes; follow-up code lacked separate review. (01-10)
- r23 claims: Qwen, two sets, five seeds, validation picks; no other-encoder claim or MMArt human text. (01-11)
- Only plain Qwen was measured as MMArt's plain encoder; its teacher text uses ArtELingo instructions. (01-11)
- Fusion's ArtELingo teacher count depends on read-out; fixed-read-out counts lack intervals; weight 0.7's causal explanation untested. (01-11)
- r23 C13 shares four readers and numerical path; faults there could be shared. (01-11)
- r24 narrative changes counts; author, length, texts per painting and ArtELingo caption-side training effects remain unseparated. (01-12)
- Shared rows tested narrative only; other-text class diagnostics on new paintings lack intervals. (01-12)
- Why portrait/landscape carry the miss is unknown; explicit genre words do not explain it. (01-12)
- Image side alone untested; r23 stands; shared fusion teacher count 4, S i2t LEVEL. (01-12)
- r24 text-specific read-out pairs differ; label-word check followed class-table reading. (01-12)
- Pair-number wording waits for user/coauthors; r24 pair metrics were out of scope. (01-11, 01-12, DN)

## Facts I could not settle

- TPIPS: 01-06 full table says ArtELingo Qwen E **ARI**, LEVEL, 0.0605/0.0550; D11-05 says **AMI**, 0.1024/0.0718. Checked brief, table and log; coverage follows report.
- DP says perspectives/SemArt "never tried"; 01-09/01-05 What ran and DN establish completion. SemArt 11,162 feasibility becomes 11,159 after exact deduplication (01-05).
- DP schedules random-start photo follow-up after r19; supplied reports establish no run. No wider run audit made.
- 01-11 Results/Analysis/What ran leave some pairs/side scores unprinted, referring to its log. Missing numbers not inferred.
- DN says r20/r21 wait, superseding schedules. O11-03/11-04 cite-only photo aspect and DP deferral both retained.
- D11-05/D11-06 postpone paper insertion; completion unestablished. Sequence dates differ from run dates in headers.
- DN excludes r17 from historical 11-04 report. Only this inventory changes.

Removal list: none for this read; nothing deleted. Historical 01-04 candidates: `c13_smoke/`, `c13_round_smoke/` under `/data/SSD2/cosir_r17/20261103_new_scores/`; existence unchecked. 01-06/D11-05 defer `OUT/turtle_pack` and node copy. 01-11 pending r23 storage is superseded by D11-08's recorded cache/scratch/worktree/branch deletions.
