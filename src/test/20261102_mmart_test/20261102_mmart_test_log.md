# r15 round log (the MMArt test)

Spec: `docs/superpowers/specs/2026-11-02-mmart-test-design.md`. Build: branch `r15-mmart-test`, merged into main
(bc27203); tickets, final review and smoke in the run handoff
(`docs/superpowers/handoffs/2026-11-03-r15-run-handoff.md`). Run chat "r15 run", started 2026-11-03 01:13 (Amsterdam),
run mode night. Outputs: `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/` (`OUT`). CPU only. Times are Amsterdam.

## Build (summary; details in the handoff)

- 8 tickets (6 of verdict code, each with an Opus review and a fix round); the final whole-branch review (Opus `xhigh`,
  three parts) approved with notes after its fix wave (1ce7712, 72ef2d2, bc27203) and re-review. 274 round tests pass.
- End-to-end smoke (`MT_SMOKE=1`, `smoke_build3/`): `QUEUE smoke OK rc=0 seconds=5671`; stage 1 at full size, stage 2 at
  40 and 100 draws.
- Review notes carried into the run: finding 3 (a failed teacher-aspect gate stops stage 2 with exit 2), finding 8 (the
  frozen and teacher retrieval rows are normalised twice; the C13 brief does the same), finding 11 (the unseen-artist
  subset has 15 paintings, so it gets no cells).

## Run

- 01:13 `uptime` load 0.9 (5 min 5.0), 53 GB free; GPU not used. `bash mt_run_cpu.sh first` started (stage 0, then
  stage 1; `MT_PARALLEL` 3), log `OUT/first_queue.log`.
- 01:18 C13: Codex job (`gpt-6.1-sol`, high; `codex-usage --room rederivation`: room, five-hour window 72% left) writes
  `mt_c13.py` (derive, compare) and `test_mt_c13.py` from the spec's rule, r14's `bs_c13.py` and the frozen r10 to r13
  protocol code, with no `mt_*` import and without the r14 scorer modules the round's code calls. The brief carries
  finding 8. It smokes a subset on `smoke_build3/`; the full derive runs under this chat after stage 2. Brief:
  `.scratch/r15-mmart-test/c13/task.txt`.
- 02:05 `QUEUE first OK rc=0 seconds=3120`. **Stage 0: 22 steps, none failed; every gate ok** (`records/status.json`):
  - (i) stores and near-copy file pass; cut sets 16,384 (test, 0.95) and 10,403 (far, 0.85), as the spec's counts.
  - (ii) c* exports: all four encoders pass, max difference 6.1e-5 (bar 1e-3).
  - (iii) scoring replay at full draws (2,000 clustering, 10,000 retrieval): max difference 0 (bar 1e-9);
    clustering 1,312 s.
  - (iv) mappers on CPU: all 20 pass (a), max 8.3e-7 (Qwen P3; bar 1e-5); (b) passes. No retrain (D11 fallback unused).
  - (v) validation teacher views pass; (vi) read-out replay: max difference 0, 1,484 s.
  - Exports: c* about 72 s per encoder, matched arms about 14 s.
- **Stage 1: 13 steps, none failed** (ranges about 2 min per encoder, picks 540 to 957 s, teacher pick 121 s). Records
  committed and pushed before any test number: 8dd3857 (the runner's printed command has a literal `\n` in its message,
  so the same paths were committed with a proper message).
- 02:07 stage 2 started (`bash mt_run_cpu.sh stage2`, log `OUT/stage2_queue.log`).
- 02:02 Codex C13, first pass: `mt_c13.py`, 43 tests pass; subset smoke on `smoke_build3/` (cut, SigLIP 2 and teacher
  ranges and picks, SigLIP 2's 15 cells at 40 and 100 draws): 2,079 items, 16 failed, max 2.6e-5. Two layout slips fixed
  in compare only. The 16: the three SigLIP 2 emotion clustering cells. Cause: the k-means thread count. r14's C13
  helper fits with one thread; the round fits with 8 (its runner sets `OMP_NUM_THREADS=8`). One thread changes 212 of
  16,384 assignments in seed 2, state 1, so the c* emotion AMI moves 1.3e-5; at 8 threads C13 matches the round
  exactly (max difference 0). Outcomes unchanged. **Decision (agent default):** C13 fits k-means at 8 threads, the
  round's setting; the spec fixes no thread count. The report states the thread sensitivity. 02:12 fix round sent by
  resume (`--kmeans-threads`, default 8, recorded; the record refuses other values).
- 02:21 fix round back: 51 tests pass; smoke 2,079 items, 0 failed, max difference 0 (SigLIP 2 emotion cells included).
- 02:22 full C13 derive started (`mt_c13.py derive --workers 8 --kmeans-threads 8`, log `c13/derive_full.log`); it reads
  no round result and does not need stage 2. Its code is committed as f093fb7 (the exact bytes it loaded).
- 02:28 Claude review of `mt_c13.py` (Sonnet `high` worker): no path lets a wrong verdict pass (independence, full scope,
  compare logic and protocol checked). One must-fix: any BELOW teacher cell makes compare fail (C13 writes `'no claim'`,
  the round `null`), a false FAIL. Should-fix: the matched-arm sidecars are not identity-checked; `derived.json` is not
  tied to its code. Not applicable here: a voided encoder would block the record (every gate passed). 02:31 second fix
  round by resume, compare-side code only (the running derive stays valid): claim vocabulary, 20 matched-sidecar
  items, `--derive-commit f093fb7` source check, unique SC4 identities.
