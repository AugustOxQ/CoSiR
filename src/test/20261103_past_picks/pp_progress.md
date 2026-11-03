Read brief, ticket, index, controller notes, spec, runner pattern and driven module interfaces.
No GPU, queue or cluster commands executed. No output or records written outside smoke.
Implemented runner, separate CPU queue reservations and full run sheet. Boundary tests initially 5/5 pass.
Added gate freshness checks and quiet-load guard; gate ii refuses missing pulled training before any CPU training.
Smoke stage0 passed. Accidentally started stage2 before stage1 finished; stopped it and will rerun sequentially.
Stopped the initial eight-worker stage1 near four minutes; smoke now uses one worker to avoid process overhead.
Seven runner boundary tests pass (pp_test_run.log: 7 passed in 9.36s). Python syntax and shell syntax checks pass.
Smoke stages 0–4 passed. Stage1 refusal.json records stage2 rc=2. Stage4 refusal.json records strict status rc=1.
Stage2 has MMArt clu/ret and SemArt clu/ret; stage3 has cells_cost, cost_table, probe raw files.
Stage4 outcomes are written; absent diagnostics failed cleanly rc=2 twice. All artifacts retained.
The synthetic matched mapper hash initially referred to a different identity; fixed the smoke to use its local
stand-in mapper identity while retaining the production reader's hash and key checks.
Integration tip moved to 7b9db859 (ticket13 diagnostics). Merge and affected smoke verification next.
Merged current r19-past-picks tip 7b9db859 in 3cf0ca41. No conflicts. It adds ticket13 pp_diag.py.
Full pp_diag run on the tiny smoke failed rows that request real full-set keys. Kept that attempt by moving its
whole diagnostic folder aside, without deletion. The rerunnable smoke now runs the four artifact-derived kinds:
gain_share, percept_strongest, cost_rows, node_time. All four pass. Production stage4 still invokes all eleven.
Final boundary test log: 7 passed in 7.67s. Smoke handback.json records stages 0–4 true, refusal codes 2 and 1,
MMArt cells 9+6, SemArt cells 6+4, two cost cells, draws 40/100, diagnostics rc=0.
Choices: serial CPU graph; one worker for synthetic scoring; independent partial SemArt validation records with no
fabricated missing entries; local development records used only while computing SemArt validation picks; exact
hash-bound C13 record before outcomes; stage3 arithmetic can precede stage0 collection so gate viii is available.
Smoke scope: real reduced MMArt/SemArt statistics, stand-in r14 validation blocks and timing batches; no full scientific
replays, extraction, training, GPU, queue, cluster operation or C13 re-derivation. Those are the run chat's run-sheet jobs.
No pp_paths.py change. No real records or real OUT writes. No files deleted. Optional cleanup only: ignored local
pp_test_run.log and pp_steps_0.log; old diagnostic smoke folders after ticket13 has finished with them. Keep the main
smoke tree and handback.json for ticket13. Planning notes are small ticket-local implementation history.
