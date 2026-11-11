# Ticket B progress

- Read pb_paths.py, decisions.md, pb_inputs.py and r17/r18/r10 scoring references.
- Worktree clean initially; branch r25/bertopic-score. BERTopic venv does not yet exist.
- No full BERTopic or scoring runs authorized; use queued 2,000-doc one-seed smoke and gate only.
- Created venv with the requested CoSiR python and --system-site-packages. Installed current BERTopic 0.17.4 and dependencies with venv pip only; install.log lives in the venv.
- Implemented pb_io.py, pb_bertopic.py, pb_score.py and two stand-in test modules.
- Initial tests: 25 passed in 3.57s (CoSiR); 9 passed in 0.14s (BERTopic pure helpers).
- Queued BERTopic smoke exit 0; seed 0 has 16 real topics and 0 final outliers. No fallback to invent 20 topics.
- Queued candidate gate exit 0 under timeout 235s, 110.211 seconds; all three candidate means equal r18 exactly. No full scorer run.
- Added stand-in tests for default-then-embedding reassignment and stopping when production HDBSCAN finds fewer than 20 real topics.
- Final tests: 27 passed in 3.89s (CoSiR), 11 passed in 0.17s (BERTopic pure helpers).
- Captured metadata in ticket_b_validation.json and defaults, choices and controller instructions in ticket_b_notes.md. No changes to decisions.md or pb_paths.py.
- Reviewed both implementation diffs and all staged paths; staged diff has no whitespace errors. Prepared the task-only commit; no push.

## Fix round
- Working in /project/CoSiR, this round's folder only. Existing user changes include decisions.md and constitution.md; leave these unstaged.
- Read review findings, current loaders/tests, and pb_vlm2vec's metadata writer (read only). No real-data runner or scorer invocation is authorized.
- Regression loop: 30 failed / 28 passed before fixes. Tests use fake SentenceTransformer modules and temporary comparator stores only.
- B1: enforce exact Transformer/Pooling(mean)/Normalize module types, max_seq_length=256 and strict unit-row deviation < 1e-4 on encoded and cached rows; record model specification; omit normalize_embeddings from encode.
- S1: rebuild caches without verified module/length metadata before seed skip checks; mismatching or absent current embedding SHA means a seed is not done.
- S2: require VLM2Vec metadata and validate its writer's aspect/instruction/emb_sha256/n/limit fields; load from the same bytes whose hash was validated.
- M1: require BERTopic sidecars, smoke=false, matching seed and NPZ hash, reading the NPZ from the checked byte snapshot.
- Folder tests after fixes: 58 passed in 4.46s with the requested CoSiR environment and thread settings. No real-data execution, deletions or push.
- Disposable test artifact retained: src/test/20261111_paper_baselines/fix_regression_red.log.
- Corrected historical notes and marked the old fallback smoke invalid in ticket_b_validation.json; retained its metadata for inspection.

## Class-count extension
- Read the current runner, scorer, controller checker and tests in main. User changes outside the task folder are preserved.
- Add new K=9/10/27 artifacts with a fresh fit per target; keep K=20 names and skip validated existing files.
- Full scoring remains controller-owned. Authorized jobs: BERTopic class-count fits first, then all-six candidate gate through the CPU queue.
- Both full folder test runs passed: CoSiR 75 tests in 12.12s, BERTopic venv 75 tests in 12.18s (run concurrently).
- Stored pre-run SHA-256s of minilm.npy/json and the six legacy K=20 files in classcount_validation.json.
- Initial detached shell ended before starting res-queue (empty log, no process); restarted with nohup in a persistent tool session. logs/bertopic_classcount.log and .exit retain output and status.
- BERTopic queue exit 0 in 1.78 minutes: all nine new fits complete, final K=9/10/27. Per-seed HDBSCAN topics 92/87/91; outliers 5384/5102/5134, all reassigned by distributions. No embeddings pass required.
- Verified pre/post SHA-256 equality for minilm.npy/json and every legacy K=20 NPZ/JSON; no changes to these files.
- Started the all-six gate afterwards through the specified no-wait CPU queue command; logs/classcount_gate.log and .exit retain output/status.
- All-six gate exit 0 in 224.237s (3.77 queue minutes); each candidate mean equals r18 exactly, including the preserved K=20 values.
- Recorded all nine run counts, hashes, timing and gate values in classcount_validation.json and classcount_notes.md. Reviewed source changes and scoped staged paths; no full scoring, deletions or push.
