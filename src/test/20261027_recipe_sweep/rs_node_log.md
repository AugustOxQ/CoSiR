# Node route log (node402)

One entry per sync and node check of the r10 recipe sweep (rule §4.2 "The node route", spec D19). Times Amsterdam.

## 2026-10-27, build 11: sync before N1 and N2

- Why: the node code was at 3ad016c (build 8). Since then 09, 10, 12 and 13 were merged; the node runs `rs_knobs.py`,
  `rs_train.py`, `rs_sweep.py` (changed) and `rs_vp.py` part (a) (new). `rs_graphs.py` is unchanged, so the k11 gate
  on the node stays valid.
- Reservation: node402, 3 GPUs free, 82 h left (`cluster status`, 14:4x).
- Inputs: `das6_sync_rs.py --node node402 --verify`: all 13 items equal in files and bytes (27.4 GiB).
