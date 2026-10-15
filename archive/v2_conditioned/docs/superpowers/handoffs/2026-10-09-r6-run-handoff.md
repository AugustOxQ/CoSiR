# Handoff: run round 6 unattended: rule §6 items 1 to 7 (DTS built by 2026-10-10 12:29), the held read, the phase-2 re-derivation, the verdict, the descriptive pass and the reports; then open `r6 read` with `--notify`

> Written 2026-10-09 23:43 by the `r6 build 6` chat. Loop step reached: step 8 (run), after step 7 (15 tickets,
> the final whole-branch review, one fix wave, the scoped re-review; merged into `main` at df421a7).
> Direction log entry: [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. This handoff's §2, §3 and §5 whole; §4 is your plan.
2. The rule, which governs everything: `src/test/20261125_artelingo_held_test/DECISION_RULE.md` (call it F), §5 to
   §11. Where this handoff and the rule differ, the rule wins.
3. `/project/CoSiR/.scratch/r6-held-test/notes/run_handoff_items.md` (local, outside git): the items the build
   collected for you (H5 cells, crashes, reruns, DTS budget and rerun mechanics, smoke reruns and waivers,
   `external_sources.json`, the descriptive pass's threads, listing caches).
4. `/project/CoSiR/.scratch/r6-held-test/notes/contracts.md` §7 (runner, H5 row, smoke records), §8 (the agreement
   record the re-derivation agent writes), §9 (GPU jobs, job folders, the settings copy), §10 (the fix wave's fields).
5. Each runner's module docstring before you call it: it holds the exact command, the order and every refusal.
   `run_r6_refit.py`, `run_r6_picks.py`, `run_r6_held.py`, `run_r6_sensitivity.py`, `run_r6_dts.py`,
   `run_r6_smoke.py`, `run_r6_apply_rule.py`, `run_r6_descriptive.py`; the GPU wrappers `scripts/run_r6_verbalise.sh`,
   `run_r6_listing.sh`, `run_r6_rerank.sh`, `run_r6_ftfeat.sh` and `scripts/das6_sync_r6.py`.
6. `/project/CoSiR/.scratch/r6-held-test/notes/gpu_path.md` §5 (the cluster CLI), §7 (hazards), §8 (the job
   scripts); the `cluster-run` skill's SKILL.md before the first DAS6 launch.
7. The build's verdicts: `F/final_review/fr_a.md`, `fr_b.md`, `fr_c.md` (final review, three areas) and `rr.md` (the
   scoped re-review), first sections only. Ticket list: `/project/CoSiR/.scratch/r6-held-test/issues/00-index.md`.
8. `~/.claude/references/loop-chats.md` ("Unattended chats"); for step 9, `report-kinds.md`,
   `~/.claude/templates/auto_report_template.md`, `results_brief.md` and `user-read-reports.md`.

## 2. Decided, not to be reopened

- **By the user:** the spec's §4 and the rule (approved 2026-10-09 04:53). Build and run unattended. The run chat
  closes by opening `r6 read` (decision C on round 6's results) with `--notify`. The user starts `r7 decide` (design
  L) by hand; it may run in parallel and stays off DAS6. **Delete nothing**: every item goes on
  `/project/CoSiR/.scratch/pending_deletions.md` (`| added | chat | path or branch | size | what it is | why it can go
  |`) for the next storage summary; the one automatic step is a verified `cluster pull --tag` removing the job's code
  copy on the node.
- **Agent defaults, made in the build (the run report shows each to the user, marked as agent defaults):**
  - "Built" is a one-time event: the DTS budget is judged on `results/dts_first_built.json` while a rerun's chosen
    setting, settings SHA-256 and GPU output fingerprints are unchanged, otherwise on the rerun's own time.
    `DTS_CLOCK_START` is pinned to 2026-10-09 12:29.
  - Smoke: a GPU family whose outputs are not given fails the smoke unless waived (`--waive <family>=<reason>`,
    recorded); a smoke-scale DTS sanity or stop failure retries seeds 9002 and 9003, then marks DTS missing.
  - A crashed `--fix 1` cannot be rerun (stricter than rule §8.1); any changed AB file stops before
    `held_started.json` with exit 1.
  - Time-box (rule §9): a first read refuses after the Amsterdam date 2026-10-15; `--after-crash`, `--fix 1`,
    `--reserve` and the smoke are not boxed.
  - `results/value_sets.json` must also be the file the regression record names (`value_sets_sha256`), stricter
    than rule §5.2; the smoke seeds are exempt from the `dts_settings.json` copy check; each started attempt records
    git HEAD and `git status --porcelain -- src` (provenance only).

## 3. Where things stand

- **Code:** `r6-held-test` merged into `main` at df421a7 and pushed. The run happens in the main checkout
  `/project/CoSiR`, as every runner's docstring command says. `F/results/` on main does not exist yet: every seed-42
  record is made fresh by this chat.
- **Build verdicts:** final review areas A (verdict path), B (scoring path) and C (comparator, GPU jobs, descriptive
  pass, smoke) each confirmed with fixes; every load-bearing number re-derived by the reviewers' own code matched
  exactly (B/B0/B1 18.341064453125 / 18.436686197916664 / 18.804931640625, AFF 19.136555989583336, 9,406
  quarter-hits, the value sets 8 / 23 / 10, the 12 episode hashes). Fix wave merged at ec27ee2 (log row 22:23).
  Scoped re-review (`rr.md`, log row 23:40): **CONFIRMED**, no blocking or should-fix finding; 828 tests passed, 1
  skipped, 4 deselected; five nits left (log row), one of which binds you: step 7's first line.
- **Comparator clock (rule §7.7): started 2026-10-09 12:29 (pinned). DTS must be built (DTS-N sanity passed and the
  chosen setting run on all 12,288 seed-42 episodes) by 2026-10-10 12:29.** At the measured rates the seed-42
  verbaliser work is about 9.5 GPU-hours (about 1.1 h on nine GPUs), plus listing jobs and CPU stages.
- **DAS6** (build 4, log rows 13:38 to 14:10): node401, node402, node408, three free GPUs each (an RTX A6000 on slot
  0), env present, selftest passed. Seed-42 images (5,245, 1.72 GB) staged on all three nodes; the FT checkpoints on
  node401 only (`das6_sync_r6.py --ckpt` copies them). Held images ship after the held episodes exist. Rates:
  verbaliser 0.80 s per call; reranker 4.4 s per episode (held: 36,864 episodes, about 45 GPU-hours, about 5 h on
  nine GPUs); model load 17 s warm, 60 s cold. About 120 hours of reservation were left on the night of 2026-10-09.
- `/project/CoSiR-r6/.../results/prep_gpu/` holds seed-42 episodes and a pre-stage verbaliser job made by older code.
  Reuse a GPU output only if the modules that produced it are unchanged (rule §6.7); otherwise make it again.

## 4. The run, in order

Every step's command is in its runner's docstring. Log one row per step in `F/20261125_artelingo_held_test_log.md`
(Amsterdam time). A stop of rule §9 before the read ends the read part only: write everything up, report, and still
open `r6 read`.

1. **Start.** Log row; `uptime`, `free -g`; `cluster status` on the three nodes.
2. **Rule §6 items 1 and 2:** `run_r6_refit.py` (inputs asserted, heads refit, every posterior bit for bit) →
   `results/refit_check.json`. Any difference: stop before the read; the user decides.
3. **Item 3:** `run_r6_picks.py` → `results/picks_seed42.json` (B, B0, B1 exactly as the rule says).
4. **Item 4 (C12):** `run_r6_held.py --mode regression` → `regression_seed42.json` (131 items), `value_sets.json`,
   the per-episode and count files. A difference is traced; the user decides.
5. **Item 5:** `run_r6_sensitivity.py` → `sensitivity_seed42.json`.
6. **Re-derivation, phase 1 (rule §8.4, C13):** a fresh Opus subagent that has not written or read the
   implementation, with the rule's import list, re-derives §6 items 3 and 4, the seed-42 Holm counts and the
   sensitivity inputs. It may run beside step 7.
7. **Item 6, DTS on seed 42** (`run_r6_dts.py` docstring, stages 1 to 7). First copy F's `dts_settings.json` byte for
   byte to `results/dts_settings.json`, `git add -f` it, commit with a subject containing "cluster run", push, then
   `cluster sync`. **No seed-42 GPU job, the verbaliser included, before that commit** (re-review nit 1: only
   `list-input` checks the copy, not `r6_gpu_inputs.py` or the verbaliser wrapper). Follow the docstring's order
   (the sanity listing job first), not `gpu_path.md`'s J1/J2 numbering; make the job inputs fresh from the merged
   code (build 4's `s42_verbalise_prestage` folders, in `/project/CoSiR-r6` and on the nodes, came from older code). Listing jobs get `--cache` with earlier listing folders. The stop runs with
   `--clock-start "2026-10-09 12:29"`. Stop exit 3, or DTS not built by 2026-10-10 12:29: stop before the read.
8. **Item 7, the smoke, last** (`run_r6_smoke.py`: check, gpu-inputs, the GPU jobs, list-input, the listing job,
   finish). Run every GPU family; waive only what truly cannot run, and say why. Commit the smoke record's copy.
9. **The read (rule §8.1):** commit and push ledger row H5 (contracts §7: runner SHA-256 in the script cell, the latest
   last; GPU-job SHA-256s in the purpose cell; "(pending)" in the episode and report cells). Then
   `run_r6_held.py --mode held` in a background Bash. Fill the episode cell with the nine hashes once they exist.
10. **Held GPU jobs** (rule §8.3), once the held episodes exist: held images with `das6_sync_r6.py`; the verbaliser and
    listing on held episodes (`run_r6_dts.py --stage list-input --for held --seed 52|53|54`, listing with `--cache`
    and the seed-42 listing folders); LB and LoRA held features; the reranker on held seed 52. Seed-42 and held jobs
    of one family on the same node env. Then write `results/external_sources.json` (all four entries).
11. **Re-derivation, phase 2:** the same independent agent (or a new one that has not read the implementation)
    against `held_pass.json`; it writes the agreement record (contracts §8). A disagreement is traced before the
    verdict (rule §9). Then `run_r6_apply_rule.py` → `held_verdict.json`; its SHA-256 into H5's report cell.
12. **Descriptive pass:** `run_r6_descriptive.py` at `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
13. **Step 9, verify and report:** a final review of the run's results (`final-review.md`); the auto report
    `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` opening with the results brief, its index row in
    `docs/reports/reports_sum.md`; the user-read report; the seed ledger note for seeds 52 to 54 (rule §10.3); the
    disclosures of rule §10.4.
14. **Storage:** `cluster du --node <n>` for each node used; every item onto `pending_deletions.md`; nothing deleted.
15. **Close:** commit and push; write the `r6 read` handoff; `loop-next <handoff> --label "r6 read" --notify`; stop.

## 5. Code and pitfalls

- **Unattended:** never stop to wait for the user. A busy GPU or node is waited for (`flock -o -w` locally; launch
  again once jobs end on DAS6). A refused command is done another way or left and noted at the top of the next
  handoff. Every subagent prompt ends with the permission line ("if a command is refused by the permission check, do
  the work another way or report it; do not wait") and the delete-nothing line.
- **DAS6** (build 4): every describe-then-score call goes to DAS6, none to the local GPU (DAS6 runs torch 2.14.0 and
  transformers 5.16.1, local 2.11.0 and 5.6.2; one of four local answers differed). Feed the chosen-setting stage one
  verbaliser job's outputs per key. HEAD's subject must contain "cluster run" before `cluster sync`. `cluster watch`
  and `cluster logs` need `--node` off the default node. `das6_sync_r6.py` runs with `/usr/bin/python3`.
  `/local/wding` is per node. Never hand-rolled ssh or rsync, never `/tmp` on a node; watch the `cluster pull`
  fallback (it once pulled 29.95 GB).
- **Reruns:** a fix to any module the DTS stages ran makes them stale (rule §6.7); move older stage records aside
  (`*.stale<N>`), keep `dts_first_built.json`. Smoke reruns move the record and the smoke folder aside
  (`*.failed<N>`). Results files are never overwritten or deleted.
- **Long jobs** are launched by this chat's main session with `run_in_background: true`; subagents implement and
  monitor only.
- **Interim notifications:** a finished subagent with a leftover background job resends its report; once recorded,
  stop it with `TaskStop`.
- Times from `TZ=Europe/Amsterdam date`; every Python call with `PYTHONDONTWRITEBYTECODE=1`; keep command output
  short.

## 6. State at handoff

- Running: nothing. No subagent, no GPU job, no DAS6 job.
- Uncommitted: nothing; `.scratch/` is local by design.
- Storage (nothing deleted; for the next storage summary): `pending_deletions.md` gained the fix wave's worktree and
  branch (70 MB), this chat's session scratchpad (1.2 GB, almost all the re-reviewer's `rr/`), and `/project/CoSiR-r6`
  with `r6-held-test` (13 GB; keep until the run report is written). 10.2 GB of that worktree is not round 6's:
  `res/coca_pr`, `res/siglip_pr`, `res/siglip2_pr` (dated 2025-12-15, host user, untracked); the storage summary asks
  the user where they belong.
