Fidelity check of docs/user_read/2026-11-08_what_is_missing.md against the requested local sources. Findings below identify discrepancies and material omissions; unflagged headline counts match the sources, including four completed original loops, 15/23 done or partly done, teacher 33/13/2 and 35/10/3, default-TURTLE 203/232, text WIN counts, and fusion weight 0.7. The proposed schedule is labelled as a proposal; its new dates are not treated as prior decisions. The deadline dates and Amsterdam conversion are consistent.

Removal list (no deletion performed): the historical r17 candidates are c13_smoke/ and c13_round_smoke/ under /data/SSD2/cosir_r17/20261103_new_scores/ (docs/reports/auto/percept/2027-01-04_new_scores.md:287); existence not checked, and they are outside this task’s working scope. No project file is proposed for removal. Keep/defer r18 OUT/turtle_pack and its node copy (docs/superpowers/direction_log.md:112,215); r23 cleanup is already recorded as completed (same file:112), so do not propose it again.

1. **wrong** — Page: “WIN, LEVEL, BELOW: the interval of the difference is above 0, contains 0, or is below 0.” (page line 142).

   Source: docs/reports/auto/percept/2027-01-04_new_scores.md:90 defines WIN as lower bound > 0, otherwise LEVEL as lower bound > −r, otherwise BELOW; lines 165–172 explicitly show wholly negative LEVEL intervals and BELOW intervals reaching zero. docs/reports/auto/percept/2027-01-06_outside_methods.md:12–13 uses the same rule.

   Suggested fix: Replace the LEVEL/BELOW definitions with the lower-bound/validation-seed-range rule. Retain WIN and specify a 95% interval.

2. **scope** — Page: “Clustering and image retrieval now have every planned comparator … plain image, teacher, PercepT, CRL, CLAY, TPIPS and TURTLE” (lines 10–11; also title line 1 and “grid is full enough”, line 129).

   Source: docs/reports/stage/2026-11-04_datasets_and_retrieval.md:464–466,482–486 scopes CLAY to Qwen/CLIP L/14 candidates, TPIPS to Qwen, TURTLE to clustering, and excludes new-three/SemArt PercepT retrieval. docs/reports/auto/percept/2027-01-06_outside_methods.md:28–34 says stronger-than-k-means evidence is unestablished and TEMI did not run.

   Suggested fix: Say the completed planned rows cover their specified encoders; list TURTLE under clustering only, and qualify that stronger clustering remains partial. Avoid implying a complete cross-product of comparators, tasks and encoders.

3. **scope** — Page: Clustering table: “all four metrics”, “7 encoders: WIN”, “4 encoders LEVEL”, and the CRL/CLAY/TPIPS/TURTLE outcomes (lines 25–38).

   Source: docs/reports/auto/percept/2027-01-04_new_scores.md:75–84,98–120 gives four-metric claims at the class count; lines 188–197 makes other-count sweeps descriptive. docs/reports/auto/percept/2027-01-05_past_picks.md:48,92 says r19 additional metrics are descriptive and AMI claims use 20 clusters. docs/reports/auto/percept/2027-01-06_outside_methods.md:92 includes class-count metrics plus AMI at 20.

   Suggested fix: Add a short metric/count key: r17 four-metric results at class count; r19 new-three, Affection additions and SemArt claims use AMI at 20; r18 and CRL also include AMI at 20. Do not let the seven-encoder WIN or SemArt LEVEL read as four-metric claims.

4. **missing** — Page: Teacher clustering, MMArt: only “4 encoders: 35 WIN, 10 LEVEL, 3 BELOW of 48” (line 32).

   Source: src/test/20261108_what_is_missing/facts_inventory.md:32 records the new three as mostly BELOW on MMArt; docs/reports/auto/percept/2027-01-05_past_picks.md:29–39 reports DINOv2 BELOW in all 16 teacher cells and mostly BELOW for both small CLIPs across sets.

   Suggested fix: Add “3 more: mostly BELOW; no better-than-teacher claim”, with the r19 AMI-at-20 scope.

5. **scope** — Page: “All teacher BELOW cells on paintings are emotion.” (line 40).

   Source: docs/reports/auto/percept/2027-01-04_new_scores.md:29,163 supports this only for r17 class-count clustering on the original four encoders. docs/reports/auto/percept/2027-01-05_past_picks.md:145–146 includes SemArt genre mAP BELOW and MMArt CLIP B/16 genre AMI BELOW.

   Suggested fix: Change to “All five r17 teacher BELOW cells, in class-count clustering on the original four encoders, are emotion.”

6. **scope** — Page: TURTLE row: “4 encoders: WIN” on both painting sets (line 37).

   Source: docs/reports/auto/percept/2027-01-06_outside_methods.md:22–23 distinguishes 116/116 WIN for c* k-means against plain-image TURTLE from 114/116 WIN with TURTLE on both sides; lines 123–124 identifies the two LEVEL cells. The original plan also requested the method on c* heads (docs/user_read/2026-11-03_experiment_overview.md:292–293).

   Suggested fix: Label this row “c* k-means versus plain-image TURTLE”; add the both-sides result, with ArtELingo SigLIP 2 emotion ARI LEVEL and MMArt SigLIP 1 emotion ACC LEVEL.

7. **scope** — Page: CLAY: “Qwen and CLIP L/14” and TPIPS: “Qwen” (clustering lines 35–36 and retrieval lines 52–53).

   Source: docs/reports/stage/2026-11-04_datasets_and_retrieval.md:464–465,482–483 distinguishes CLAY on CLIP L/14 features from the two c* candidates, and TPIPS Qwen 8B from Qwen c*. docs/reports/auto/percept/2027-01-06_outside_methods.md:52,97 confirms TPIPS base size/released model.

   Suggested fix: State that the encoder names identify c* candidates; CLAY uses CLIP L/14 features, and TPIPS is the released Qwen 8B model. This avoids implying Qwen CLAY or a matched-size TPIPS baseline.

8. **missing** — Page: Photo columns say the photo model carries the claim, while “Frozen c* on photos with encoders other than Qwen” is listed as a gap (lines 26–27,101).

   Source: docs/reports/stage/2026-11-04_datasets_and_retrieval.md:458–459,476–477 records frozen painting c* on Affection as a failed Qwen confirmation, and Qwen on EmoSet as descriptive; the photo model is a different, fine-tuned candidate. src/test/20261108_what_is_missing/facts_inventory.md:29–30,49–50 retains this distinction.

   Suggested fix: Add the existing frozen-Qwen Affection failure below the teacher, and identify the replacement photo model as fine-tuned on Affection train photos. Keep the other-encoder gap separate from this completed failed test.

9. **missing** — Page: Image retrieval teacher row, Affection: only “4 encoders: holds”; ArtELingo and MMArt teacher cells also omit the three additions (line 49).

   Source: src/test/20261108_what_is_missing/facts_inventory.md:52 records restricted r19 licences on ArtELingo, mostly BELOW on MMArt, and mixed Affection results including DINOv2 BELOW. docs/reports/auto/percept/2027-01-05_past_picks.md:29–39,147–150,179–181 distinguishes numeric WINs from the permitted “level with or better than” claims.

   Suggested fix: Include the new-three teacher retrieval status: mixed/restricted on ArtELingo and Affection, mostly BELOW on MMArt, and no licensed superiority claim. Explicitly name DINOv2’s failure against the teacher.

10. **scope** — Page: Text row “Other encoders (CLIP L/14, SigLIP 2, SigLIP 1) … descriptive: c* above” and gap “on other encoders as a claim” (lines 67,100).

   Source: docs/reports/auto/percept/2027-01-09_text_retrieval.md:9,78,103–111 compares Qwen c* with three plain encoder rows; docs/reports/auto/percept/2027-01-11_text_claim.md:30,200 limits claims to Qwen and calls these descriptive side rows. These are not results of retraining/running c* on three alternative text encoders.

   Suggested fix: Rename the row “Qwen c* versus other plain text encoders”; say any claim for c* with another encoder remains absent. Preserve validation-all-directions versus held-text-to-image-only scope.

11. **scope** — Page: Shared-painting text columns contain bare counts such as “captions 9; narrative text 8”, alongside claim verdicts (lines 61–66).

   Source: docs/reports/auto/percept/2027-01-12_miss_check.md:7–20,31–32,96 says the 11,528 shared paintings are an exploration with no bar; r23 verdicts stand. The counts hold under either label set, and fusion’s shared-caption fifth cell is LEVEL.

   Suggested fix: Mark the shared column “descriptive; no verdict; 11,528 shared paintings; either label set”. Explain that fusion’s 4 does not overturn its held-set 5/9 claim.

12. **wording** — Page: “The claim counts WIN cells of 9 in label mAP” and “claim holds” for late fusion (lines 59,63–66); pair hits are mentioned only as wording waiting (lines 69–70).

   Source: docs/reports/auto/percept/2027-01-11_text_claim.md:9–16 defines holds as 9/9 over plain and at least 5/9 over teacher; fusion additionally needs a pair-R@10 lower bound above −0.02, which passes on both sets (lines 115–122). Pair-number interpretation waits, not this evaluated clause.

   Suggested fix: State the two label thresholds and fusion’s additional pair bar, passed on both sets. Keep the pending coauthor wording separate from the completed pair-bar test.

13. **wrong** — Page: “12 MMArt texts as queries: narrative text tested; the other four only on validation” (line 77).

   Source: docs/reports/auto/percept/2027-01-11_text_claim.md:165 reports genre text-to-text on formal, historical, unified and emotional text on the new MMArt paintings, descriptively with no bar. docs/reports/auto/percept/2027-01-12_miss_check.md:29 also reports new-painting class diagnostics on those texts; only the shared rows are narrative-only.

   Suggested fix: Replace with “Five perspectives explored on validation; narrative claim and shared-row check done; other texts have limited descriptive new-test diagnostics, but no full shared-row follow-up.”

14. **missing** — Page: “DEC loss on each head, and deeper heads” (lines 78,87,120).

   Source: docs/user_read/2026-11-03_experiment_overview.md:212,226 requires DEC both plain and with a pull toward initial clusters, retrieval reported too, and a 3-layer residual deeper head. docs/reports/stage/2026-11-04_datasets_and_retrieval.md:498,512 retains the two DEC arms and deeper residual heads.

   Suggested fix: Restore the two DEC variants and retrieval evaluation, and specify the 3-layer residual head; all remain waiting under direction_log.md:37.

15. **wording** — Page: “One run … never ran” / photo random weights listed without its picked recipe (lines 14,88,119).

   Source: docs/superpowers/direction_log.md:84–87 decides five seeds at 80 neighbours on Affection test after r19, distinguishing the earlier seed-0, 40-neighbour validation comparison. src/test/20261108_what_is_missing/facts_inventory.md:201 explicitly says supplied reports establish no run and no wider audit was made.

   Suggested fix: Say “No completed run is recorded in these sources”; include five seeds, 80 neighbours and Affection test so the pending decided run cannot be confused with the earlier validation check. Keep 30 minutes labelled as an estimate.

16. **missing** — Page: Parked list has MMArt’s other texts and later items, but no new MMArt instruction/aspect experiment (lines 90–96).

   Source: docs/superpowers/direction_log.md:74–76 explicitly parks narrative/formal/historical names as new Qwen instructions, requiring new aspects and heads. src/test/20261108_what_is_missing/facts_inventory.md:136 distinguishes this from completed perspective retrieval.

   Suggested fix: Add “New MMArt perspective instructions/heads (narrative, formal, historical): parked, not run; separate from querying current c* with MMArt text.”

17. **missing** — Page: “TURTLE with its own setting selection” (line 92) and supplementary checks (line 121).

   Source: docs/superpowers/direction_log.md:96–99 says a small spec must settle D9 versus D25 before the deferred selection; docs/reports/auto/percept/2027-01-06_outside_methods.md:42–44 specifies selection on validation followed by plain-image TURTLE at the chosen setting, with about 4 GPU hours an unmeasured estimate.

   Suggested fix: Retain the deferred status and add the prerequisite spec decision D9 versus D25 and validation-selection/test sequence. Keep the proposed dates explicitly provisional.

18. **missing** — Page: Paper items list a draft, r18/r19 insertion, MMArt wording and items 22/23 (lines 103–108), but no explicit citation/explanation task for criterion-driven clustering or incompatible conditional-retrieval benchmarks.

   Source: docs/reports/stage/2026-11-04_datasets_and_retrieval.md:499,515 marks criterion-driven clustering and the second photo aspect “cite and explain only”. docs/user_read/2026-11-03_experiment_overview.md:165–166 says to explain why GeneCIS and CLAY’s Stanford40 conditions lack c* heads; lines 243,270 repeats cite/explain decisions.

   Suggested fix: Add these paper-only explanations explicitly, naming criterion-driven methods and the out-of-head conditional benchmarks. The status table’s shorthand “paper text only” does not specify the required explanation.

19. **wording** — Page: “only the related work and the method have a first draft (11-02)” and “The draft: abstract, introduction, experiments, limits” (lines 15,105).

   Source: No supplied inventory/report or the requested Now/Parked/2026-11-05–08 direction entries establishes that dated drafting inventory. src/test/20261108_what_is_missing/facts_inventory.md:125–126,204 establishes pending wording/insertion only; docs/superpowers/direction_log.md:29,37 records later insertion and an undecided stage-note form.

   Suggested fix: Remove the unsupported “only”/11-02 first-draft assertion, or supply a dated paper-source citation after a separate authorized check. Present the listed sections as proposed writing work, not verified absent drafts.

20. **missing** — Page: The completed text work is presented only through the main caption-side/fusion grid (lines 56–70); paper work says only “Word MMArt’s text result” (line 107).

   Source: docs/superpowers/direction_log.md:139–141 decides an additional check of why 0.7 helps using query hit splits. docs/reports/auto/percept/2027-01-11_text_claim.md:178–185 reports score-spread and hit-overlap diagnostics, while line 31 says they test no cause. docs/reports/auto/percept/2027-01-10_pair_hit_checks.md:23,43–49 establishes the own-caption trade-off on older recipes only.

   Suggested fix: Briefly account for these decided/completed checks: hit-overlap diagnostics ran but the cause remains open; the own-caption experiment improves pair hits and lowers all nine label cells only on older saved recipes, with no c* retrain. Do not imply the explanation is established or the decided diagnostic never ran.

21. **wording** — Page: “Text retrieval … fails on MMArt, where the counts follow MMArt’s text” and “the main paper can word MMArt as a limit of the text” (lines 12,132).

   Source: docs/reports/auto/percept/2027-01-12_miss_check.md:19–31 establishes the count effect for narrative text and leaves author, length, texts per painting and caption-side training unseparated; other MMArt texts have only limited diagnostics. docs/superpowers/direction_log.md:106–108 explicitly retains the user’s view that the reason is not fully explained.

   Suggested fix: Use “MMArt narrative text” consistently and say the mechanism remains unknown. In paper wording, preserve that limitation rather than treating all MMArt text or an identified text property as the established cause.

21 findings
