# Run log: the relational genre round (r10), stages A, B and C

Run chat "r10 run", from the handoff `docs/superpowers/handoffs/2026-10-27-r10-run-handoff.md`. Code: main at 68e8081
(no change under `src/` since c438f29, the code of every gate file). Outputs:
`/data/SSD2/cosir_r10/20261027_relational_genre/outputs` (`outputs/` here links to it).

## Before the start (2026-10-27 00:33)

- GPU check: driver 595.99.02 in `nvidia-smi`, the kernel module and the container library; no compute process, no
  `gpu0.lock` holder. Load 0.18, 54 GB RAM available. System disk 229 GB free, `/data/SSD2` 1.9 TB free.
- Run mode: day. No container restart (handoff §2).
- **End estimate** (`rg_queue.md`, "Expected times"): stage A 82 min (worst 106, a climb to GR-256), stage B 65 to 70
  plus about 10 for `rg_v2 --phase b` at full scale, stage C 52; about 210 min, worst 240. Start 00:33, so the expected
  end is about 04:05, the worst case about 04:35. Without B-PICK the chain ends after stage B, about 03:00. GPU share
  about 1 h, light (spec §9); CPU about 2.5 h. The 11:00 end and the weekday 10:00 to 19:00 rule are not touched.

## The run

- 00:33 `rg_queue_gpu.sh` started from `/project/CoSiR` (background Bash; stdout in the chat's scratchpad `gpu.out`).
- 00:34 VA1: the three runs exist (kept, 37 s); the scorer half is kept with the same code, inputs and values, 16 of
  16 items within 1e-6; the queue notes "gate VA1 holds".
- EQ passes (21 s); GQ-8 can win.
- GR-16 and GR-32 trained (107 s). Climb decision 1 (56 s): genre A on P, seed 0: GR-8 0.5018, GR-16 0.5022,
  GR-32 0.4964; all guarded; newest GR-32 lower, so the climb stops (no GR-64).
- `rg_v2 --phase rung --arm GR-32` (95 s): agree, 0 differences, 2,000 draws.
- GN-8, GS-0.02, GS-0.01, GQ-8 trained in one call (187 s), every `run.json` status ok (no failed run, no VA3 rerun).
- `rg_score a` (321 s): v*_A = GS-0.02, A-NEW, A_genre 0.5074419784910177. GS-0.01 fails the guard (e_style
  −0.0094 < −0.0082). No `VA4 note: interrupted run` line in the `score_a` log. `pending_deletions` empty: every arm
  is at or above T_g − r_genre (0.4861), so every stage A weight is kept.
- `rg_v2 --phase a` (504 s): agree, 0 differences, 2,000 draws. Stage A outcome stated: A-NEW.
- Loss-scale run (GR-1, one epoch, 30 s): status ok; its weights on the pending list. Queue A done 00:56, failed=0
  (23 min against the 82 estimated).
- 00:56 Queue B: VB1 holds (17 s, kept file). GR-2, GR-4 and GS-0.02 on seeds 1 to 3 (387 s): all nine `run.json`
  status ok.
- `rg_score b` (566 s, 01:12): B-PICK, v_held GS-0.02; confirmed GR-8, GR-2, GR-4, GS-0.02 (all C-HOLDS). Ā_genre:
  GS-0.02 0.5050, GR-8 0.5022, GR-4 0.5005, GR-2 0.4965, R6-5 0.4694; GR-8, GR-2 and GR-4 within one seed range of
  GS-0.02 (`within_one_range`).
- `rg_v2 --phase b` (1,366 s, 01:35): agree, 0 differences, 2,000 draws. Stage B outcome stated: B-PICK, G = GS-0.02.
- `report_ab` (504 s). Queue B done 01:43, failed=0.
- 01:43 Queue C, G = GS-0.02. VC1 training half: verified weights, kept, pass (66 s); scorer half: the kept file
  matches this code, nothing recomputed (4 s); "gate VC1 holds". The VC1 run goes on the pending list.
- `train_g` (631 s): five seeds at full scale, no non-finite loss. `held_g` (15 s): 15 output sets, 12,281 rows
  each, `vc3_attempts` 1 and no VC3 problem for each.
- `score_c` (83 s): G-GO. d_T −0.0060 [−0.0105, −0.0014] (A_G 0.5001, A_T 0.5062); d_R +0.0467 [+0.0402, +0.0531]
  (A_R 0.4534); G_half 0.0264 equals the frozen value; PC holds; `matches_or_exceeds` false (lower bound under 0).
  9,949 labelled rows. `stage_c/verdict.json` SHA-256 a6fd488e44620b2c912a014c307ed748c6a5ce26f8a03fabb17a60b9c7455361.
- `vc2` (99 s, 01:59): agree, max abs difference 6.9e-18. Stage C outcome stated: G-GO.
