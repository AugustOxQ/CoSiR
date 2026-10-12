# Review fixes: round 8 decision rule (from `opus_rule_check.md`), applied 2026-10-12 from 20:26

One line per finding. "Applied" = the reviewer's fix as proposed (wording may differ).

1. Applied: SC1a lists R3's ten `aff_*` arrays as R7 `run_r7_stage0.py` l.166 to 173; `aff_bar_v` is not compared. §4 SC1a.
2. Applied: painting index p = position in `rb_halves.npz` `paintings_sorted` (= `g_st`), "M1 order" for all painting-ordered arrays; `meta.json` stores and asserts the painting-id string per row; M13 selection paintings in ascending `artelingo_splits().groups` code. M1, M4, M6, M13; `rb_halves.npz` pinned in §10.
3. Applied: test loader = 36,518 scorer-train paintings in M1 order; φ of any image = unit `encoder_linear[0](encoder[0](x))` in eval / no_grad. M7.
4. Applied, plus `DrawMax = 10000`: the net file imports it from `MainLauncher` too (l.22), so the `r8_` copy has to set it; the value is MainLauncher's. M7.
5. Applied: `CUBLAS_WORKSPACE_CONFIG=:4096:8`; one wiring smoke (2 epochs, 2,048 paintings, `results/smoke/`, decides nothing), allowed alongside stage 0. M7, §4.
6. Applied: SC1d (head path) and SC1e (M3 recipe at r = 1.0 gives R7's R0); `r7_groupings.npz` pinned in §10. §4, §7, §10.
7. Applied: shares by `bincount(..., minlength=K)/36518`, `default_rng(7200+i).choice` / `default_rng(7300+i).choice`; control draws added to the re-derivation (a). M13, §7. (`minlength` added so an empty label count cannot shorten `p`.)
8. Applied: rng call order (permutation, views, own then other negatives per query); pools exclude p and n(p), with n(p) = CLIP neighbour for V and NB, CSD for NBC. M6.
9. Applied: `F.interpolate(..., bicubic, align_corners=False, antialias=True)` then clamp; TF32 off and batch size recorded for every GPU pass. M1 (GPU passes), M4.
10. Applied: TP's img and bank inputs = TP map, same in both halves; `classes_half{0,1}` = arange(K), `filled_by_half` = 1 − half (stated for every X in M12, which matches R7's `r7_heads.py` format); 27 names from `paintings.json` top-level folders. M9, M12.
11. Applied: `transform_subspace` with `V_`, `P_` cited; worker saves `V_`, `P_` and asserts `labels_.shape == (n, 2)`. M8.
12. Applied: strength among evaluated NB scorers; none evaluated → NBC does not run (recorded). §6.3.
13. Applied: `SystemExit` at a search point → r skipped and recorded; final K per grouping recorded; TP ≤ 27 and FC*/NRK below 17 stated plainly and listed for a spec note. M3 (Group counts), §7 report flags.
14. Applied: φ of FCMI = 17-value representation (agent default for "hidden features"); GPU departs from A §9 budget (recorded as a mechanic); checkpoints and `NpPoints` to `/data/SSD2/cosir_r8/fcmi/<run>/`. M7, §7 report flags.
15. Applied: DI and SG load with `PIL.Image.open(p).convert("RGB")`; AL uses `main.load_img`; ESC1(d) hashes contiguous int64. M10.
16. Applied: ESC1 (b) to (d), SC2's label count and finite-loss log added as re-derivation (d); stage 1 now stores the loss log and pre-merge labels. §5, §7.
17. Applied: AL = ALADIN-ViT from StyleBabel (BAM-FG), used as ED5's "official weights from its README" (agent default), named so in the report. M10, §7 report flags.
18. Applied: CLIP B/32 revision 3d74acf (refs/main) with both weight-file SHA-256s (bin; safetensors from refs/pr/66), the loaded one recorded; `rb_halves.npz` pinned; TP text check (256 captions, cosine ≥ 0.9999, else TP dropped). §10, M9, M16. (The drop on failure is added so the check has a consequence; TP is a reference and picks nothing.)
Agent defaults section: applied as the §7 "Report flags" bullet (merge after FCMI/NRK, V100-to-V33 drop, ALADIN model, FCMI φ and GPU).
Controller (a) TP count and FC*/NRK below 17: M3 "Group counts". Controller (b) ALADIN weights: M10 and §7.
- Controller, 2026-10-12 20:34: re-review L1 (SC2 test excludes the wiring smoke), L2 (main.py l.114) and L4 (smokes in §9's order) applied; the M8 line offsets (L2) left as is; L3 (faiss-cpu cost per epoch) goes to ticket 06's smoke.
