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
