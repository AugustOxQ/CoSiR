# Handoff: r9 build: write and check the decision rule first, then build the generic AFF chain, the cluster preparation and the probes

## Brief
- Done: grill and spec approved: AFF as a fixed recipe on Qwen and SigLIP 2, the ArtELingo-28 language check, CUB and SemArt probes.
- State: nothing built. node406 and node408 reserved, 3 GPUs each, until 10-18 14:40. Day run: ask the user when unsure.
- Next: the decision rule, checked by Opus and committed; then tickets; cluster wave 1 at once.
- Blocked: Affection waits for the user's access request (dropped if no data by 10-24); not part of this build.

> Written 2026-10-13 15:10 by `r9 decide 2`. Loop step reached: after step 6 (spec approved 15:05), before step 7.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md) (grill and spec bullets).

## 1. Read in this order

1. The spec, from §1: `docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md`. §3.5 gives the stages and gates,
   §4 every decision (not to be reopened), §6 the bars.
2. `docs/superpowers/constitution.md` (C1 to C14; the spec's §7 lists two exceptions, C4 and C9).
3. The round 6 rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md`: the model for the new rule (§3 the bar
   and its integer Holm form, §5.4 the frozen picks, §6 the pre-read checks, §8 the ledger-first runner).
4. The round 3 rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md`: AFF's parts, D1 to D15.
5. The facts brief `.scratch/r9-scope/facts_brief.md` (local only): §1 AFF's parts and what depends on the backbone,
   §2 the hard-wired places (CLIP default paths, SHA-256 pins, the split by CLIP image hash), §3 timings, §4 data.
6. `~/.claude/references/matt-chain.md` (tickets, verdict code, the lean build), `~/.claude/references/codex-jobs.md`,
   the `cluster-run` skill, and `~/.claude/templates/decision_rule_template.md`.

## 2. Decided by the user, not to be reopened

All of spec §4 (D1 to D34), recorded in the spec and the direction log. In short: fixed recipe (D3); stop when the
seed-42 pick sets λ_a = 0 in both halves (D4); seven checks per backbone on ArtELingo (D5); ArtELingo-28 translated to
English with NLLB-200, 746 paintings, two-level test, all three backbones (D6 to D9, D20); held reads local, cluster
for everything before them (D12); no time cap that drops anything (D14); day run (D34).

## 3. Where things stand

- **Features:** Qwen3-VL-Embedding-2B for every ArtELingo row (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b`,
  2048-d, joined by `sample_id`); no loader builds `load_artelingo()`'s object from it yet. SigLIP 2: nothing usable
  (round 8's image-only store is not reused, D19). CUB: CLIP, Qwen, SigLIP 2 in `backbone_check/cub`. SemArt: CLIP and
  Qwen (`/data/SSD2/pre_extract/semart/`), SigLIP 2 to make.
- **ArtELingo-28:** `/data/PDD/artelingo/ArtELingo/semeval-artelingo28/` (CSV with columns `image_id, genre, emotion,
  caption, art_style, painting, language, image_name, split`; `images.zip`, 2,000 images); root-only files on the
  shared disk, read only. No translation exists anywhere (PercepT released none). The 746-painting read set
  (248 held + 498 new) and its value counts: style 7 of 26 values with 30+ paintings, genre 8 of 9, every emotion
  (computed 2026-10-13 by `r9 decide 2`, script `a28_values.py` in that session's scratchpad).
- **Cluster:** node406 and node408 idle, 3 GPUs of 48 GB each, CoSiR env ok, reservations end about 2026-10-18 14:40.
  `DATA_MAP` lacks CUB and SemArt (D24); whether the nodes hold `artelingo_full`, SemArt features and the model
  weights is not checked.
- **Local HF cache:** Qwen3-VL-Embedding-2B, SigLIP 2 so400m/14-384, GoEmotions, CLIP B/32. NLLB-200 3.3B: not
  checked.
- Seeds: 42 development; 52 to 54 for H6 and H7 (H5's episodes, D16); 55 to 57 for H8 to H10 (D29); 9001 to 9003 smoke.

## 4. Open points for the build (no user decision needed; the rule fixes them)

- The A3 tolerance in the recipe replay (stage 0b, D32), traced to GPU non-determinism.
- How an ArtELingo-28 row is formed (a translated caption with its painting's image) and whether the `other` emotion
  value counts (our `artelingo_aspect_labels` excludes only `something else`).
- The per-language coverage rule for the translator (a language NLLB-200 lacks is flagged and reported).
- Which SHA-256 pins are re-made per backbone, and how the switched-off check (D4) is written.

## 5. Code and pitfalls

- New round folder `src/test/20261202_r9_scope/`, module prefix `r9_`; the rule is its `DECISION_RULE.md`.
- The rule comes first: written from the spec, checked by a fresh Opus reviewer, committed before any scoring code
  (C7). Extraction and translation may start before it (D31).
- Stage 0a (round 6's 131-item seed-42 regression, exact) and 0b (recipe replay) gate every use of the generic chain.
- The SHA-256 pins on CLIP-made files fail on other backbones by design. Pass CLIP's leakage groups into the split for
  every backbone and keep the size asserts (D15).
- The `cluster` on PATH is a graph tool; the cluster CLI is `~/.claude/skills/cluster-run/cluster`.
- Held reads run locally from pinned files, ledger row committed and pushed first (D12, R6 rule §8).
- Nothing is deleted: each item goes on `.scratch/pending_deletions.md`. Cleaning the nodes before 10-18 14:40 needs
  the user's approval (day mode: ask).
- The GitHub repo is public: never name it in paper material.

## 6. State at handoff

- Running: nothing.
- Uncommitted: only the untracked round 8 results trees (`src/test/20261129_r8_grouping/results/`,
  `results_m4_09999/`), as before.
- On disk over 1 GB: nothing new.
- The remote branch `origin/r8-grouping` (merged into main) was deleted at the user's approval (D26).
