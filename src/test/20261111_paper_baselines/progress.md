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
