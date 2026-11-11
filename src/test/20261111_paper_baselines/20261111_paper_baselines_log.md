# Paper baselines: round r25 run log

> 2026-11-11 · [Design record](../../../docs/superpowers/handoffs/2026-11-11-r25-paper-baselines-handoff.md) · [Auto report](../../../docs/reports/auto/percept/2027-01-14_paper_baselines.md)

## Problem

The paper needed VLM2Vec-V2 and BERTopic clustering rows beside c*, our student with aspect heads.
The user fixed ArtELingo test, aspect instructions for VLM2Vec-V2 and test captions for BERTopic.
The paper cutoff is 2026-11-14. CEMTM was excluded because it needs retraining.

## Investigation timeline

All times below are Amsterdam local time. Git timestamps were read with their explicit local offset.
The timeline distinguishes commit times from recorded execution times.

|Time|Evidence|Step|
|---|---|---|
|2026-11-11 05:02|Design handoff|User design passed to the run chat.|
|2026-11-11 05:07|ff91fd3c|Round folder, paths, decisions and test input writer.|
|2026-11-11 05:10|decisions.md|Defaults set before comparator scores.|
|2026-11-11 05:20|4f6e86e3, 3de4cd52|VLM2Vec extractor, BERTopic partitions and scorer built.|
|2026-11-11 05:21|79f1513c, extract.log|BERTopic branch merged. Local GPU extraction started.|
|2026-11-11 05:22|7a41d3c0|Build choices recorded.|
|2026-11-11 05:26|decisions.md|Missing MiniLM files identified and corrected after review finding B1.|
|2026-11-11 05:32|0943b0f6|Fallback-cache guard and comparator provenance checks committed.|
|2026-11-11 05:33|26783112|B1 decision and independent check script recorded.|
|2026-11-11 05:35|decisions.md|User's class-count extension recorded.|
|2026-11-11 05:41|bertopic_classcount.log|Queued class-count BERTopic fits started.|
|2026-11-11 05:49|5f6e0c16|Class-count cells and partitions committed.|
|2026-11-11 06:16|6b9b909b|Validation extraction option added for the other chat's retrieval work.|
|2026-11-11 07:54|extract.log, score_full.log|Image extraction ended. Full CPU scoring started.|
|2026-11-11 08:05|68cd35f6|Twelve final comparator cells recorded, all WIN, exact candidate gate.|
|2026-11-11 08:06|ac48d7e7|Independent recompute and stale-run size check recorded.|
|2026-11-11 08:06|Report handoff|Main numbers sent to the paper chat.|
|2026-11-11 08:08|Report handoff|Independent check result sent to the paper chat.|

The report handoff is stamped 2026-11-11 08:12.
Its delivery times are handoff statements, rather than git execution timestamps.

## Missing MiniLM files: root cause and fix

Review finding B1 concerned the cached MiniLM snapshot.
It lacked the sentence-transformers files, so the first BERTopic run silently used a fallback model.
That model had mean pooling, no Normalize module and a 512-token limit.
Its row norms ranged from 1.67 to 5.00.

The run chat restored four missing files at the same revision, 1110a243.
The corrected model uses Normalize and a 256-token limit. The longest document has 242 tokens.
A new guard refuses fallback models and stale caches. The run chat reran BERTopic.
The stale outputs remain on disk for the size check.

|Aspect, twenty clusters|Stale AMI|Final AMI|
|---|---:|---:|
|emotion|0.0720|0.0662|
|style|0.0927|0.0936|
|genre|0.3137|0.3092|

These values come from records/check.json. The handoff reports no label changes.
The check file contains stale point values, without stale intervals.

## Class-count extension and final run

The user required class counts for Table 2 through the paper chat. Twenty clusters remain supplementary.
The extension added emotion nine, style twenty-seven and genre ten for both comparators.
Each BERTopic count used fresh fits with the checked MiniLM cache. Existing twenty-topic files stayed unchanged.

The build notes record seventy-five tests in each environment and an exact six-cell candidate gate.
These are prior build checks. This report task did not repeat the test suites.

The local GPU extraction ran for 153.09 queue minutes, at about 250 milliseconds per image.
The handoff records peak memory of twelve GB and power from 264 to 284 watts.
This exceeded the two-hundred-watt light-work bar and ended before the weekday 10:00 limit.
Full scoring took 650.7 seconds, or 10.87 queue minutes, on one CPU slot.

All methods fitted 12,281 test paintings. Emotion and style score all rows, while genre scores 9,949 labelled rows.
Both comparators use three states. c* uses five model seeds.
The scorer uses the same labels, partitions, validation ranges and paired bootstrap as the earlier outside-method comparison.
It uses 2,000 draws and bootstrap seed zero.

The records contain twelve WIN cells. Their candidate gates match the earlier round exactly at all six aspect/count pairs.
The independent check has maximum gap 0.0 over twelve cells.
It independently refits VLM2Vec k-means and reads BERTopic partitions, then calls sklearn AMI.
This checks row order, masks and partition assembly. It does not independently check sklearn or reproduce image extraction.

## Extra MiniLM read-out

The report task adds pb_alt_minilm.py without changing existing round files.
It checks minilm.json's names hash against the scorer's exact row order and checks the embedding-file hash.
It calls the same ns_metrics.partitions function as VLM2Vec and the same pb_score.score_cell function.
Thus k-means settings, labels, c* partitions, validation ranges and bootstrap settings share the existing code path.
The comparator name is minilm_kmeans.

The candidate gate is reused from cells.json to save CPU work. Each resulting record states that choice.
score_cell still checks its candidate mean against the reused gate.
PB_SMOKE uses the scorer's forty draws and writes a separate smoke record. No new tests are required.

The CPU queue rejected four attempts as busy. One retry was too early. The final three attempts used three-minute delays.
The fifth attempt started at 2026-11-11 08:20 and finished before the 2026-11-11 08:28 time check.
It exited zero after 420.4 script seconds, or 7.05 queue minutes.
The full run used 2,000 draws per cell. A separate smoke run was not needed.
The input check in logs/alt_input_check.log confirms matching names and embedding hashes, and shape 12,281 by 384.

records/alt_minilm_kmeans.json contains all six cells:

|Aspect|K|MiniLM k-means AMI|d [95% interval]|Label|
|---|---:|---:|---|---|
|emotion|9|0.0841|+0.0183 [+0.0143, +0.0224]|WIN|
|emotion|20|0.0915|+0.0151 [+0.0111, +0.0191]|WIN|
|style|27|0.1074|+0.3035 [+0.2945, +0.3063]|WIN|
|style|20|0.1010|+0.3056 [+0.2971, +0.3091]|WIN|
|genre|10|0.2884|+0.2216 [+0.2114, +0.2310]|WIN|
|genre|20|0.2949|+0.2052 [+0.1965, +0.2129]|WIN|

Our reading: MiniLM k-means narrows the emotion gap, but all alternative cells remain WIN for c*.
The check changes the read-out only. Per-caption topics, multimodal BERTopic and VLM2Vec extraction alternatives remain unchecked.

## Reports and retained files

Codex drafted the auto report, the user-read report and this log (08:10 to 08:28).
The controller (Claude) did the claim review: five numbers recomputed from the record files, all equal.
The user-read report then got a Haiku cold read and a Sonnet fidelity check, and one fix wave.
No deletion was made. Existing round code and records stayed unchanged.
