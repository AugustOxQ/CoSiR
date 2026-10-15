# Re-review of the fix wave: round 8 gate decision rule (scoped, C7)

> 2026-10-12 20:35 (Amsterdam). Rule under review: `src/test/20261129_r8_grouping/DECISION_RULE.md` (415 lines,
> SHA-256 4a236651870446966670963a025b510c4de2bc938aea862edd5bdb32c9578d2d, uncommitted). Review it answers:
> `opus_rule_check.md`; fixer's record: `fixes.md`. Scope: each finding, plus bars, comparators, candidates, controls
> and data roles against specs A and B. Not a fresh review of the whole rule. Nothing edited; nothing committed.

## Verdict

**commit.** All 18 findings are fixed as proposed or changed with a sound reason. No bar, comparator, candidate,
control or data role of spec A or B moved. No contradiction found. Three low-severity items below are cosmetic and can
be corrected in the same commit or left.

## Findings 1 to 18

1. **Fixed.** SC1a now lists `aff_fused_cells`, `aff_cf_cells`, `aff_sigma`, `aff_gate__a`, `aff_gate__b`,
   `aff_fused__*`, `aff_cf__*` over r1, gain, other, strict, swap, and says `aff_bar_v` is not compared. Checked:
   `numpy.load(seed42_arrays.npz).files` holds exactly these 15 plus `aff_bar_v`; `run_r7_stage0.py` l.166 to 173 and
   `METRICS` (l.41) compare the same set.
2. **Fixed.** M1 defines p as the position in `rb_halves.npz` `paintings_sorted` (= `g_st`) and "M1 order" for every
   painting-ordered array. Checked: `np.searchsorted(paintings_sorted, painting_of_local_row)` equals `g_st`; R7's
   `r7_heads.py` l.63 to 66 builds `g_st` the same way. (`paintings_sorted` holds the raw group values 2 to 61396; `g_st`
   is the 0 to 36517 position. The rule's "position in" wording is the right one.) `meta.json` painting-id check in M4,
   M6's permutation and M13's selection order follow.
3. **Fixed.** M7: test loader = the 36,518 scorer-train paintings in M1 order; φ of any image from the shared encoder in
   eval and no_grad. No selection or held image enters the fit.
4. **Fixed, plus one addition.** `LimitKmeans`, `CodeTest`, `UnDeterministic`, and also `DrawMax = 10000`. Checked:
   `MainLauncher.py` l.13 to 21 holds these values; `networkExplicitDecoupleShuffle.py` l.22 imports all three names,
   so `DrawMax` is needed. Sound.
5. **Fixed.** `CUBLAS_WORKSPACE_CONFIG=:4096:8` (FCMI `main.py` l.155 comment) and a wiring smoke (2 epochs, 2,048
   paintings, `results/smoke/`, decides nothing). Allowed alongside stage 0 in §4. See new item L1.
6. **Fixed.** SC1d (head path) and SC1e (M3 at r = 1.0 gives R7's R0) added, with `r7_groupings.npz` pinned. Checked:
   SHA-256 matches; key `R0` is int64 (183,694,) with 17 groups; R7's own G3 compared the same head outputs at 1e-12
   (`run_r7_stage0.py` l.212 to 224), so the tolerance is one R7 already met.
7. **Fixed.** M13 gives `default_rng(7200 + i).choice(K, size=36518, p=shares)` and `default_rng(7300 + i)` for the
   selection paintings; `minlength=K` added (changed-ok: stops an empty label count shortening `p`). §7(a) re-derives
   the draws from their seeds.
8. **Fixed.** M6 fixes the call order (permutation, views, own then other negatives per query) and the pools without p
   and n(p); n(p) is the CLIP neighbour for V and NB, the CSD neighbour for NBC.
9. **Fixed.** M4 names `F.interpolate(..., "bicubic", align_corners=False, antialias=True)` then clamp. M1 sets TF32 off
   and records the batch size for every GPU pass (M4, M7, M10).
10. **Fixed.** M9 and M12: TP's image side and bank inputs = the TP map, the same in both halves; `classes_half{0,1}` =
    arange(K) and `filled_by_half` = 1 − half. Checked against `r7_heads.py` l.172 to 184: `filled[loc] = k` for the
    rows of half 1 − k, and l.178 asserts `filled == 1 − half_of_local_row`; `classes_half{k}` is the int64 arange(n_cls);
    `r7_reader.py` l.48 to 64 requires the same keys and dtypes; R7's `xheads_R0.npz` holds them (`int64 (17,)`,
    `int8 (183694,)`). The 27 style names: the sorted top-level folders of `paintings.json` give 27 names (I listed them).
11. **Fixed.** M8 cites `transform_subspace`, `V_`, `P_`; worker saves them and asserts `labels_.shape == (n, 2)`.
    Checked: installed ClustPy `nrkmeans.py` has SHA-256 1145bd9f…; `V_` and `P_` are set at l.861 to 863 and
    `transform_subspace` (l.922) returns `X @ V_[:, P_[s]]`. The rule says "l.860 to 862" and "l.944": one to two lines
    off, harmless.
12. **Fixed.** §6.3: "among evaluated NB scorers; if none was evaluated, NBC does not run (recorded)".
13. **Fixed.** M3: `SystemExit` at a search point skips and records that r. "Group counts" paragraph records every
    final K and lists TP ≤ 27 and FC*/NRK below 17 for a spec note. §7 report flags repeat it.
14. **Fixed.** φ of FCMI = the 17-value representation, marked agent default. GPU departure from A §9's CPU budget is
    named (up to 6 GPU hours; a mechanic). Checkpoints and `NpPoints` go to `/data/SSD2/cosir_r8/fcmi/<run>/`.
15. **Fixed.** M10: DI and SG load with `PIL.Image.open(p).convert("RGB")`; AL uses `main.load_img`; ESC1(d) hashes
    contiguous int64.
16. **Fixed.** §7(d) re-derives ESC1 (b) to (d), SC2's label count and the finite-loss log from stored files; §5 has
    stage 1 store the loss log and the pre-merge labels.
17. **Fixed.** M10 names AL as ALADIN-ViT from StyleBabel (BAM-FG), used as ED5's "official weights from its README"
    (agent default); §7 report flag. Checked: ALADIN README l.7 releases only `aladinVIT_bamfg_64.47.pt`. Spec B ED5
    says "ICCV 2021 weights"; this is a wording departure, not a bar or candidate change, and the report must name it
    (it does, in §7).
18. **Fixed.** CLIP B/32 revision 3d74acf9… (refs/main) with both weight SHA-256s, `rb_halves.npz` pinned, TP text check
    (256 captions, cosine ≥ 0.9999, else TP dropped, M16). Checked: `refs/main` = 3d74acf9…, `refs/refs/pr/66` =
    c237dc49…; `sha256sum` of the two blobs equals the rule's a6308213… (bin) and 99d28a65… (safetensors); `rb_halves.npz`
    = ed2d359e…. The TP drop on failure is an addition; it does not move a bar (TP is a reference and picks nothing).
- **Agent defaults section: fixed.** §7 "Report flags" names the merge after FCMI/NRK, the V100-to-V33 drop, ALADIN,
  and FCMI's φ and GPU.

## Fidelity against the specs (re-checked after the wave)

- Bars unchanged: SC3 (lower bound > 0, strict), SC4 (Σ(4d) ≥ −49 on each of two pairs; ⌈−0.3 · 4,096 / 25⌉ = −49),
  ESC2 = same gate, ESC1 items (a) to (e) as B §6, pick order as A §6 and D26, NBC trigger as D22, flag as D27, promising
  as D30. Candidates 12 groupings × 2 (+ NBC) and 3 × 2; references R0own, NRK, TP; one control per candidate grouping;
  comparators AFF, B, B0, B1, BX, CF[s]. FCMI weights 0.05 to 1.60 = 0.25 to 8 × 0.20 (D20). Data roles: groupings and
  centres on scorer-train only; views from the scorer-train images only; held rows and held images untouched (C5);
  TP and M9 read only path folder names. Spec SHA-256s in the rule header match the files (83ff81bb…, e8598116…); R7 rule
  (204a0ac5…), constitution (bdd66f44…) and the ten reused R7 code files match their pins; FCMI (`main.py`,
  `networkExplicitDecoupleShuffle.py`, git head 9750ccfc…), ALADIN (3ddd112, main.py, vit_networks.py, weights),
  ClustPy file, DINOv2 (47b73ee) and SigLIP 2 (e8e4872) revisions match.
- No contradiction found between the new text and §4 to §9. The fix wave moved nothing in §6 (the gate).

## New problems

- **L1 (low, wording).** M7's "SC2 / D21 test (on any FCMI run)" and the wiring smoke's "no SC2 decision" read against
  each other, since the smoke is an FCMI run. Intent is clear (the smoke decides nothing). Optional fix: "on any FCMI run
  except the smoke".
- **L2 (low, citation).** M7 cites `main.py` l.111 for `use_deterministic_algorithms(True)`; it is l.114 (inside the
  `if not UnDeterministic:` branch, l.99). Same slip in `opus_rule_check.md`. Cosmetic. The M8 line numbers (860 to 862,
  944) are also one to two off; cosmetic.
- **L3 (low, not a rule fault).** FCMI's per-epoch k-means now runs on `faiss-cpu` (`IndexFlatL2`) with `niter` 300,
  `nredo` 10, over 300 epochs, against D21's 60-minute test. The rule marks the swap as an agent default, but the wall
  time is unmeasured. The wiring smoke (2 epochs) gives a time per epoch; if it projects over 60 minutes, FC1 to FC6
  drop for a time reason, not a wiring one. Worth reading off the smoke before stage 1. This predates the fix wave
  (the reviewer did not raise it).
- **L4 (nit).** §9 "Order" does not list the smoke; §4 and M7 do. No effect.

## Deletions

None made. One scratch file of mine, `/tmp/claude-0/h.txt` (ten SHA-256 lines), is outside the session temp folder and a
hook refused its removal; it is listed in `/project/CoSiR/.scratch/pending_deletions.md`.
