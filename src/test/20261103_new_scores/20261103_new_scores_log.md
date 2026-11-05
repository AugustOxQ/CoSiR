# r17 run log: new scores from saved outputs

Spec: `docs/superpowers/specs/2026-11-03-new-scores-design.md`. Handoff: `docs/superpowers/handoffs/2026-11-04-r17-run-handoff.md`.
Code on main at 8cf743b2 (run chat opened at daba7948). Outputs: `/data/SSD2/cosir_r17/20261103_new_scores/outputs`.
Times are Amsterdam time.

## 2026-11-04 00:22: start and end-time estimate

- Machine: run mode night; queue idle, 4 CPU slots free; load 3; 51 GB RAM available; 1.8 TB free on SSD2. r19 build 4
  is open and may take slots.
- Estimate at the start, in three chains (three of the four CPU slots): stages 0 to 2 about 16 to 17 slot-hours, so
  about 06:00; C13 about 10 slot-hours (r15's rate, not measured), so about 09:30; stage 3 about 12 slot-hours, so
  about 13:30.
- Choice (agent, `shared-resources.md`): the run goes on past 11:00 if it must. It is CPU only, so the local GPU stays
  free and the machine is not loud. The report is due 18:00, a split over two nights would miss it, and the spec (§3.6)
  has its own cut: if 12:00 passes with stage 3 unfinished, the photo sets run last and may be cut. The expected end is
  told to the main chat through the handoff brief.

## 2026-11-04 00:56: stage 0 done, all gates pass

16 gates, about 34 min of wall time in three chains (2.1 slot-hours were estimated at load 30 to 60; the load was low).
No failed or missing gate. Status after stage 0: ok 29, failed 0, missing 132 (the stage 1 items).

| Gate | Result |
|---|---|
| (i) a, r14 SigLIP 2 ArtELingo SC1, SC6; r15 CLIP L/14 MMArt SC1, SC4; r10 stage 4 partitions | pass, max abs diff 0.0 |
| (i) b, matched-arm interval | pass, 0.0 |
| (i) c, Affection; EmoSet | pass, 0.0 |
| (i) d, EmoSet K = 8; r15 class count | pass, 0.0 |
| (ii) r15 SC2 and r11 SC10 to SC15; CRL; PercepT | pass, 0.0 |
| (iii) metrics; class counts | ok, 0.0; all eight counts equal §3.3's |
| (iv) 20 mappers on CPU; Qwen validation | ok, max 8.3e-7 (bar 1e-5) |
| (v) untold join | ok on four encoders, max unit deviation 4.5e-5 (bar 1e-3) |
| (vi) read-out rule | pass, 0.0, picks and flags equal |

Record note (build 3's §3): at build time `a_r15` failed first on one cell (SC4 genre teacher, 1.3e-5). r17's replay
normalised r15's rows a second time, which flipped one row in k-means. That was r17's own wiring slip, fixed in 86c3cc0;
the re-run and this real run pass 30 of 30. The real cells were never affected: `ns_clu_cells` feeds rows as `ns_data`
returns them.

## 2026-11-04 02:31: stage 1 done, ranges and picks committed before any new test number

About 95 min of wall time (r18 and a review job took CPU slots from about 01:40). 116 class-count ranges and 16 new picks; status ok 161, failed 0,
missing 0. Picks (all `kept`): CRL on MMArt `none+k10`; matched ArtELingo SigLIP 2 `center+k3`, SigLIP 1 and CLIP L/14
`center+k10`; matched MMArt Qwen and CLIP L/14 `center+k10`, SigLIP 2 and SigLIP 1 `center+k0`; untold ArtELingo
`center+k10` on all four; untold MMArt Qwen and SigLIP 2 `none+k10`, SigLIP 1 `none+k1`, CLIP L/14 `none+k3`.

## 2026-11-04 02:32: stage 2 and the C13 job started

Stage 1 committed at c0ed42d8. Stage 2: 21 queue steps in three chains (`run_logs/stage2_plan.txt`). C13: Codex
`gpt-6.1-sol` high (`codex-usage --room rederivation`: room, 5h 57% left), brief `.scratch/r17-new-scores/c13/task.txt`,
writing `ns_c13.py` and `test_ns_c13.py`. A smoke root at 40 and 100 draws (`c13_round_smoke`, EmoSet Qwen clustering and
ArtELingo Qwen retrieval) gives its subset compare. The full derive runs after its review, section by section in the queue.

## 2026-11-04 03:38: C13 code done, full derive started

Codex hand-back: `ns_c13.py` and `test_ns_c13.py`, 56 tests pass, subset smoke `C13 PASS items=896 failed=0
max_abs=3.6e-15 scope=subset` (counts, Qwen validation probabilities, EmoSet ranges, three picks, the 14 smoke cells). Its
estimate for the full derive: 10.05 slot-hours (52 sections). The code is committed at 33d7155e; the derive runs from that
commit, section by section through the queue, three chains (`run_logs/c13_plan_{a,b,c}.txt`). An Opus `high` worker reviews
the code at the same time. Stage 2 so far: 13 clustering steps done or running, faster than estimated (painting 13 to 24
min against 35 to 50).

## 2026-11-04 03:52: C13 review: pass after fixes

Opus `high` worker: the derive follows the rule and finds no wrong number; the gap is in `compare`. Must-fix 1: the numbers
in `records/outcomes.json` are not compared (only the cell files are), and `verdicts` copies that file. Must-fix 2:
`licensed` and `summary`, `status` and `smoke` of that file are not compared. Notes taken too: 3, `--write-record` passes
with cells absent; 4, an old pass survives a later refusal. Left: 5, a missing range would be judged differently (stage 1
has none, so it does not apply today); 6, minor notes (`rb_matched` and `ct_percept` not in the fingerprint; both frozen).
All fixes stay inside `compare`, `certify` and `Comparison`, so the fingerprint and the running derive stay valid. Codex has
no room (12% left, reset 04:35), so a Sonnet `high` worker makes the fixes (routing: Claude takes it).

## 2026-11-04 04:09: C13 compare fixes merged, re-review passes

bbd4f4f9: `compare` now checks the numbers, `licensed`, `summary`, `status`, `smoke`, the row count and id uniqueness of
`outcomes.json`; `certify` refuses a bare absent cell; a refusal or an unfinished compare replaces an older pass with
`pass: false`. Fingerprint unchanged (c278e4a2...), so the running derive stays valid. 62 tests pass, each fix sabotaged
once and caught. Sonnet `high` re-review: pass. Left as notes: an inherited false FAIL if a stage 1 range were not ok (all
116 are ok, so it does not fire), and hand-edited `sentence` text, `present`, `r_source` or the void reason list would not
be seen (`ns_outcome` writes those).

## 2026-11-04 04:58: stage 2 done, outcomes written (pending C13)

All 21 steps ok, 412 cells, rows decided 412, missing 0 (2.4 h of wall time, others' jobs held a slot part of it).
`records/status.json` committed; `outcomes` wrote `records/outcomes.json`, sha256 f851da2e. No verdict before C13.

## 2026-11-04 06:59: C13 passed, verdicts written

- Derive: 52 of 52 sections ended exit 0 (`run_logs/c13_derive.log`), 02:32 code start, about 05:00 first full plan, 06:59 end.
- Compare (`ns_c13.py compare --write-record`, `run_logs/c13_compare.log`):
  `C13 PASS items=22615 failed=0 missing=0 max_abs=3.52218254562e-14 scope=full`.
- `c13.json` names `outcomes_sha256` f851da2e...5b7f, which equals `sha256sum records/outcomes.json`.
- `c13-check` ok, `verdicts` wrote `records/verdicts.json` (same bytes as `outcomes.json`). `status` rebuilt: ok 161, failed 0,
  missing 0; `status.json` equals HEAD.
- Machine at 06:59: CPU idle, 4 slots free. Stage 3 starts now in three chains.

## 2026-11-04 07:02: stage 3 starts, order and end estimate

- Order (agent): the 39 lines of `ns_run_cpu.sh stage3` split into four parts, each in three chains:
  a. painting sweeps, per-class, told-untold, `qwen_crl`, `r0`; b. painting figures and tables; c. photo sweeps; d. photo
  figures and tables. Why: §3.6 says the photo sets run last and may be cut, and the plan's own order put photo sweeps before
  the painting diagnostics. With this order a cut can lose photo rows only. No step changes.
- End estimate: about 11:30 if the CPU slots stay free (r18 and r19 may take slots). This is past 11:00; the choice made at
  stage 0 holds (CPU only, go on; §3.6's 12:00 cut applies).

## 2026-11-04 08:53: stage 3 done

- 39 of 39 steps exit 0 (`run_logs/stage3.log`), 07:00 to 08:53, about 1 h 53 min in three chains. CPU was free all the way.
- Much faster than the estimate (about 11:30): painting sweeps took 11 to 34 min, not 55 min; photo sweeps about 2 to 3 min.
- Nothing was cut: photo sets all ran. The run's CPU use ended at 08:53, before 11:00.
- Outputs: `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/` (2.0 MB, 39 files).

## 2026-11-04 09:02: fix of the Qwen c* against CRL diagnostic

- Found when I read stage 3's output: `qwen_crl_mmart.json` had clustering rows for emotion and genre only. In `ns_diag.qwen_crl`
  the skip of the 20-cluster row also skipped the class-count row at K = 20, so MMArt style was lost.
- Fix bd8717c2, with a test that failed before the fix. Rerun through the queue: the old rows came back identical, and 4 style
  rows are new. No verdict reads this step. Change log: `.claude/20261104_log.md`.
- `ns_nonwin_cells.md` written from `records/verdicts.json`: 39 of 412 cells, the rest are WIN.

## 2026-11-04 09:31: reports done

- Auto report `docs/reports/auto/percept/2027-01-04_new_scores.md`: Codex draft (`gpt-6.1-sol` medium), the controller rewrote the
  results brief, then a Sonnet `high` claim review (about 330 numbers recomputed; two small errors, four wordings too strong,
  caveats added). User-read report `docs/user_read/2026-11-04_new_scores.md`: Codex draft, a Haiku cold read and a Sonnet `high`
  fidelity check (about 70 numbers, all matched), one fix wave. Time-box 18:00 met.
- The report is about 18 KB against a target of about 15 KB: the many report items that no code writes take the room.
- CPU use ended 09:02. No GPU. Nothing cut. Disk: 183 MB on SSD2.
