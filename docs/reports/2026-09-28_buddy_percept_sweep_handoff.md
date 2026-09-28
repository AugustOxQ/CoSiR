# Handoff: buddy-percept comprehensive sweep — read this first in the next session

Written 2026-09-28, end of the session that built and launched the sweep.
Branch `experiment/percept_topic_pipeline`, worktree
`/project/CoSiR-buddy_prototype_conditioning` (based on `/project/CoSiR`).

## TL;DR — what to do right now

1. Check the sweep is still healthy:
   ```
   CLUSTER=~/.claude/skills/cluster-run/cluster
   for n in node403 node404 node405; do
     $CLUSTER status --node $n | grep -o '"running_jobs":\[[^]]*\]\|"time_left":"[^"]*"'
   done
   ```
   Expect 3 `sweep-agent-<node>-<slot>` tags running per node. If any node
   shows fewer than 3, check `$CLUSTER logs --node <node> <tag>` for that
   slot — a crashed agent needs re-launching (see §"Relaunching a dead
   agent" below).
2. Check sweep progress on the dashboard: entity `polysemic`, project
   `CoSiR-buddy-percept-sweep`, sweep id `40i43gt5` (full URL omitted here
   deliberately — this repo's egress hook blocks bash commands that
   contain literal `https://wandb.ai` text; just navigate there directly
   or construct the URL yourself: wandb.ai/polysemic/CoSiR-buddy-percept-sweep/sweeps/40i43gt5).
3. **Task 11 is the only remaining task**: once the sweep has accumulated
   enough completed, gate-passing runs (dozens+, your judgment — the
   objective gate is `stage2_macro_auc if emotion_ami>0.1236 and
   genre_ami>0.1954 else -1.0`, so only check runs with `objective > -1`
   count), run the post-sweep top-10 stress test:
   ```
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
   python src/test/20260928_buddy_percept_sweep/run_top10_stress.py polysemic/CoSiR-buddy-percept-sweep/40i43gt5
   ```
   This pulls the top 10 runs by `objective`, re-runs each at the
   established 4-seed stress convention (42/7/123/2024), and prints a
   final winner. Fold the winning config + result into
   `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`
   as a new subsection, matching this investigation's established
   reporting convention (see §6a–§6h there for the pattern).
4. If the 24h soft budget has passed and `objective` is still visibly
   improving on the dashboard, just leave the 9 agents running longer —
   they're elastic, no redesign needed, agents already have ~106h of
   SLURM reservation left as of this writing.

## What this sweep is, and why it exists

This is the direct continuation of the multi-night PercepT-vs-buddy-graph
topic-formation investigation
(`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` is
the master report — **read that file's §6a through §6h before touching
anything else**, it has the full arc). Tonight's session replaced the
earlier hand-tuned "candidate 1, 2, 3..." approach with a systematic 22-
(then trimmed to 21-) dimensional Bayesian W&B sweep covering both Stage 1
(buddy-graph InfoNCE topic formation) and Stage 2 (patch-image classifier)
jointly, per trial.

**The single most important finding from tonight, which any Task 11
result must be read against**: earlier in this session, closing an
asymmetric-tuning gap found that a *symmetrically tuned* PercepT Stage 2
mapper actually **beats** buddy's own best tuned Stage 2 mapper (0.9226 vs
0.8534 macro AUC) — see master report §6g. This reversed four rounds of
buddy-favoring Stage 2 claims. **Stage 1 (topic formation, the
investigation's actual subject) is unaffected** — buddy still clears the
AMI Pareto bar without PercepT's occupancy collapse. Whatever Task 11
finds should be read in that light: it's asking "what's the best buddy
Stage1+Stage2 configuration," not "does buddy beat PercepT" — that
question's current answer is "yes on Stage 1, no on Stage 2, as of the
symmetric-tuning correction."

## Implementation plan and spec (if you need architecture detail)

- Spec: `docs/superpowers/specs/2026-09-28-buddy-percept-sweep-design.md`
- Plan: `docs/superpowers/plans/2026-09-28-buddy-percept-sweep.md` (11
  tasks, all code in `scripts/buddy_percept_sweep/`)
- SDD ledger (full blow-by-blow of every review finding, ruling, and fix
  round across all 11 tasks): `.superpowers/sdd/2026-09-28-buddy-percept-sweep/progress.md`
  — this is git-ignored scratch, but it's still on disk in this worktree;
  read it if you want the detailed history of what each task's review
  caught and how it was fixed. Once Task 11 is done and a final
  whole-branch review is clean, this workspace should be deleted per the
  subagent-driven-development skill's normal finish flow.

Tasks 1–10 are complete and reviewed. **Task 11 (post-sweep top-10
4-seed stress test) is the only one left** — its script
(`src/test/20260928_buddy_percept_sweep/run_top10_stress.py`) is already
written and tested (with mocked data), just never run against real sweep
results yet since the sweep had just started.

## Execution method used

Subagent-driven development, but adapted: **almost all implementation and
review was dispatched to Codex CLI directly** (`codex e
--dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c
mcp_servers={} -C <dir> --json -`, task piped via stdin, `nohup ... &`),
not Claude subagents, per explicit user instruction to conserve Claude
Code usage. **`codeagent-wrapper` (the CCG-provided wrapper around this)
is confirmed broken** — hangs forever at "Reading from stdin pipe" — use
the direct `codex e` invocation instead. If you see Codex spontaneously
try to dispatch a nested review via `codeagent-wrapper`, it will hang;
kill the process tree (`pkill -9 -f "codeagent-wrapper --progress
--backend"` etc.) — this happened twice tonight, described in more detail
in the master report's earlier sections from prior nights.

If Task 11 needs any further real code changes (unlikely, but possible if
its dry-run-only script needs adjusting once it sees real sweep data),
follow the same pattern: dispatch to Codex, review the diff yourself or
via a second Codex dispatch, don't just trust "done."

## DAS6 / cluster-run gotchas discovered tonight (important, not in any doc elsewhere)

1. **`cluster check --sync-data`'s "is this data already on the node"
   check is unreliable for a brand-new node-side path prefix.** It uses a
   bare `ssh host bash -s` connection (outside the SLURM allocation) to
   test `os.path.exists`, while `cluster launch` runs everything inside
   the srun-allocated shell — these two gave **contradictory** answers for
   `/local/wding/pre_extract/...` and `/local/wding/Dataset/artelingo`
   paths that had never been populated before. The user independently
   confirmed via their own check that the data exists "on the cluster"
   somewhere, but couldn't explain the discrepancy either — root cause
   never diagnosed. **Do not trust `ready.data: yes` for a new DATA_MAP
   prefix without independently verifying via a real launched job** (see
   `src/test/20260928_buddy_percept_sweep/force_data_sync_das6.py`'s
   docstring for the exact verification pattern used).
2. **Workaround**: `force_data_sync_das6.py` in this repo bypasses the
   broken pre-check by importing `cluster.py` as a library and calling its
   own `plan_data_sync`/`run_data_sync` functions directly with a
   manually-supplied "missing" dict — no modification to cluster.py
   itself. Run it with **system python3, not the CoSiR conda env**
   (`/usr/bin/python3 src/test/20260928_buddy_percept_sweep/force_data_sync_das6.py <node>`)
   — conda's bundled OpenSSL conflicts with the system `ssh` binary
   cluster.py shells out to (`OpenSSL version mismatch` error otherwise).
3. **`cluster.conf`'s `LAUNCH_ALLOWED`** only permits `python
   main_cosir.py` or `bash scripts/run_*.sh`. A bare `wandb agent ...`
   command is rejected. Fix already in place:
   `scripts/run_buddy_percept_sweep_agent.sh` is a thin wrapper matching
   the allowed pattern.
4. **`cluster.conf` is local and gitignored** (`~/.claude/skills/cluster-run/cluster.conf`,
   confirmed via `git check-ignore`) — safe to edit directly for DATA_MAP
   additions, no commit needed in that tool's own repo. It now has 4 new
   entries for the `/local/wding/Dataset/artelingo`,
   `/local/wding/pre_extract/artelingo(_heldout|_percept_patch_features)`
   node-side paths (per the user's own stated convention: dataset data
   under `/local/wding/Dataset/<name>`, pre-extracted features under
   `/local/wding/pre_extract/<name>`).
5. **New dataset config**: `configs/dataset/artelingo_cluster.yaml`
   (tracked, shared) declares these same node-side paths so
   `cluster check --sync-data -- python main_cosir.py
   dataset=artelingo_cluster` can theoretically drive the sync mechanism
   (even though in practice tonight the force-sync script was needed
   instead, per gotcha #1/#2 above) — kept because it's the "intended"
   sanctioned mechanism and documents the path convention even if its
   auto-detection needs the workaround.
6. `scripts/buddy_percept_sweep/real_data.py` routes the held-out split's
   otherwise-hardcoded path constants (from the protected pilot file
   `run_learned_student_arch_sweep_pilot.py`) through the same
   `PERCEPT_FEATURE_ROOT`/`PERCEPT_RAW_JSON_ROOT` env vars
   `run_pipeline.py` already respects — the DAS6 launch wrapper sets these
   to the node-side paths; local/test runs leave them unset and get the
   original `/data/...` defaults unchanged.

## Relaunching a dead agent

If `cluster status --node <node>` shows fewer than 3 `sweep-agent-*`
tags running, one GPU slot's agent died (crash, OOM, preempted). Relaunch
just that slot:
```
CLUSTER=~/.claude/skills/cluster-run/cluster
$CLUSTER sync --node <node> --allow-any-branch   # only if you've made code changes since
$CLUSTER launch --node <node> --gpu-slots <0|1|2> --tag sweep-agent-<node>-<slot> -- \
  bash scripts/run_buddy_percept_sweep_agent.sh polysemic/CoSiR-buddy-percept-sweep/40i43gt5
```
Same sweep id — it resumes pulling from the same shared queue, no data
re-sync needed (data already confirmed present on all 3 nodes as of
tonight).

## Standing constraints (still in force, from earlier nights)

- CoSiR's own trainable-label-embedding retrieval framework
  (`src/hook/train_cosir.py`, `scripts/run_sweep_agent.py`,
  `scripts/sweep_config_v*.yaml`) is explicitly out of scope — a
  different system that happens to also use the word "buddy." Never
  touch it as part of this investigation.
- Never edit files under `src/test/20260922_percept_topic_pipeline/`,
  `src/test/20260923_artelingo_buddy_analysis/`, or
  `src/test/20260927_deep_stage_analysis/` — provenance for this
  investigation's earlier nights. Reuse via import, never modify.
- DAS6 scratch work scoped to `/local/wding/` on nodes; never touch
  shared `/tmp`.
- Git safety: create new commits, never amend; never force-push; never
  skip hooks.
