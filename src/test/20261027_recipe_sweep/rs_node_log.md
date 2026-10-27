# Node route log (node402)

One entry per sync and node check of the r10 recipe sweep (rule §4.2 "The node route", spec D19). Times Amsterdam.

## 2026-10-27, build 11: sync before N1 and N2

- Why: the node code was at 3ad016c (build 8). Since then 09, 10, 12 and 13 were merged; the node runs `rs_knobs.py`,
  `rs_train.py`, `rs_sweep.py` (changed) and `rs_vp.py` part (a) (new). `rs_graphs.py` is unchanged, so the k11 gate
  on the node stays valid.
- Reservation: node402, 3 GPUs free, 82 h left (`cluster status`, 14:4x).
- Inputs: `das6_sync_rs.py --node node402 --verify`: all 13 items equal in files and bytes (27.4 GiB).
- Launch fixes (both in `scripts/run_rs_node.sh`, change log `.claude/20261027_log.md`): the echo text `; CUDA_VISIBLE_DEVICES=`
  read as an assignment by the launch's check (now `,`); the CLI dry-runs a runner that mentions CLUSTER_DRY_RUN and
  accepts only `python main_cosir.py` lines (the alias is gone; the plan prints with `--plan`).
- N1 (tag 20261027-134327-ea1bd84, 14:43, 43 s): PASS. Versions equal the local env (numpy 2.2.6, scikit-learn 1.6.1,
  scipy 1.17.1, torch 2.11.0+cu130, Python 3.11); 71 input files equal by SHA-256 (4.3 GB); RTX A6000 on slot 0;
  k11 gate and stores k10, k30, k40 pass; P loads (29,232 rows). `gates/n1.json` fetched home.
- N2 (tag 20261027-134558-ea1bd84, 14:46, 109 s): PASS. G seed 0 smoke trained and scored on the node: A emotion
  0.0949, style 0.3376, genre 0.4988. `gates/n2.json` fetched home.
- Node jobs launch from the worktree `/project/CoSiR-wt-node` (branch `r10-rs-cluster-run`), since `cluster launch`
  needs the node to hold HEAD and HEAD's subject or branch to name "cluster run". The run chat syncs the final
  reviewed code the same way (a branch whose name holds "cluster-run", at the merged tip) before stage 1.

## 2026-10-27, run 3: sync and stage 1

- Code: the node worktree `/project/CoSiR-wt-node` (branch `r10-rs-cluster-run`) fast-forwarded from ea1bd84 to main's
  tip a87cb1d (the build's merged code, eb1d04b, plus handoff commits; no code change after eb1d04b).
- `cluster sync` (17:07): ok, sha a87cb1d; the node's clone holds the commit (`git cat-file -e` passed in the sync).
  `cluster status` still shows `code_head` 1ea7742: that is the clone's checked-out branch, which sync never moves; every
  job runs in its own worktree at the launched commit (each launch's `sha` field says a87cb1d). Not a blocker.
- Inputs: `das6_sync_rs.py --node node402 --verify` (17:07): all 13 items equal in files and bytes (27.4 GiB).
- Reservation: node402, 3 GPUs free (RTX A6000, 48 GB each), 79 h 44 min left at 17:06.
- Deadlines (rule §4.2, exception of 15:31): `RS_START_BY` 2026-10-27 17:08 (the plan's start), `RS_END_BY`
  2026-10-28 01:08 (plus 8 hours).
- Stage 1 launched 17:08 to 17:10, all at sha a87cb1d: `rs1027-stage1-a` (slot 0: G, G-repeat and 12 runs),
  `rs1027-stage1-b` (slot 1: 14 runs), `rs1027-stage1-c` (slot 2: 13 runs).
- Stage 1 shards ended about 17:55 (about 3 min per run on an A6000, scoring included); pulls verified;
  `fetch scores/node402` brought 41 score files home. No run failed (`failed` empty; every export passed V1b).
- N3 (`rs1027-stage1-n3`): PASS. G₀ and G₀′ equal bit for bit (exports and best epoch; difference 0 on each aspect,
  limit r_a / 2). Readings, no gate: trainer parity (G₀ minus 12-27's local seed-0 run) emotion +0.0003, style +0.0020,
  genre −0.0077; scorer parity 0 on each aspect. `gates/n3.json` fetched home.
- V1c part (a) (`rs1027-stage1-vp`, node CPU): pass, 41 runs; `scores/node402/rederive_partitions.json` fetched.
- Local: `rs_sweep stage1 --machine node402`: GRID-4 (grid knobs k11, k4, k2, k9; levels k11 20/40, k4 0.02/0.035,
  k2 0/2/8, k9 1024/512; 18 new points); `rs_vp sweep` (part (b)): pass (`sweep/node402/rederive_stage1.json`).
  The plan ended 18:29.
- Stage 2 started 18:30 (`RS_END_BY` 2026-10-28 01:08): 18 new points in three shards.
- Stage 2 shards (`rs1027-stage2-a`, `-b`, `-c`, slots 0 to 2, at a87cb1d): 18 grid points, all ok (no V1b failure);
  ended about 18:55; pulls verified; scores fetched.
- V1c part (a) (`rs1027-stage2-vp`, node CPU, 59 runs): the plan's `cluster watch` stopped at 19:27 on its default
  30-minute stall limit while the job still ran (V1c part (a) prints only at its end; stage 1's took 31 minutes for 41
  runs). Not a failure: the run chat watched it again with `--stall-minutes 120` and ran the plan's remaining steps by
  hand, in the same order (script `stage2_tail.sh` in the run chat's scratchpad, status lines in
  `queue_node_stage2_status.txt`): pull; the stage 1 home copy moved to `rederive_partitions.stage1.json`; fetch; part
  (a) pass (59 runs); deadline not passed; `rs_sweep stage2`: TOP; `rs_vp sweep` (part (b)): pass. Done 19:39.
- TOP: k11=40 (M 0.92), k2=2+k9=512+k11=40 (0.87), k4=0.035+k11=40 (0.75). Stage 3 runs locally from 19:4x.
- For a later round: a node job that prints nothing for over 30 minutes needs `--stall-minutes` in the plan's watch.
