# Handoff: r17 decide: grill and spec for new scores from saved outputs (NMI, ARI, ACC, cluster counts, CRL, PercepT retrieval, untold retrieval)

## Brief
- Done: decision A (user): six loops r17 to r22 from the experiment overview; r17, r18, r19 decided tonight, in parallel.
- State: r17 rescores saved embeddings only (CPU, no training); its scoring module is used by r18 and r19.
- Next: a short grill, a short spec, the user's approval, then the build tonight (unattended).
- Blocked: nothing.

> Written 2026-11-03 18:35 by tab "r16 report 3". Loop step reached: after decision A, at step 4 (medium loop, step 3
> skipped: the options are clear). Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets
> "The user's reading of the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The overview's section "The plan (decided 2026-11-03)", near the end of
   [`docs/user_read/2026-11-03_experiment_overview.md`](../../user_read/2026-11-03_experiment_overview.md): loop 1 is
   r17. Its to-do table gives each item's cost.
2. The direction log's r16 entry, the two bullets named above.
3. For cell facts: [`src/test/20261103_experiment_overview/facts_inventory.md`](../../../src/test/20261103_experiment_overview/facts_inventory.md),
   sections B (coverage grid) and E (cost).
4. Before the grill: `~/.claude/references/grill.md`. Before the spec: `~/.claude/rules/spec-templates.md`.

Skills: grilling, domain-modeling (as grill-with-docs).

## 2. Decided by the user, not to be reopened

- r17's items (2026-11-03 18:30): 1 (NMI, ARI, ACC beside AMI on every claim cell), 2 (cluster-count sweep), 7 (CRL
  against c* on CLIP L/14, bootstrap with the bar set before looking), 8 (CRL on MMArt), 9 (untold retrieval), 11
  (emotion retrieval below the teacher: a per-class look), plus CRL clustering on paintings and PercepT retrieval on
  SigLIP 2, SigLIP 1, CLIP L/14 and MMArt.
- Cluster counts for emotion: 5, 8, 16, 20 (user's own). Agent recommendation, seen by the user: set per aspect,
  below the class count, at it, and up to 20 (emotion has 9 classes on ArtELingo and MMArt, 8 on EmoSet).
- Metrics across counts (agent recommendation, user's point on chance): AMI and ARI for the sweep; NMI and ACC at the
  class count only.
- Untold retrieval's form (user, earlier 2026-11-03): the image only, c*'s three heads joined into one vector, one
  ranking scored with each aspect's labels, against the plain image (also untold); CPU, saved held embeddings.
- Loops run in parallel; r17's build runs tonight and merges first.
- **Each loop's report states shortly how much it changes the current paper writing: related work, method,
  experiments** (user, 2026-11-03 18:33). Put it in the spec's report section; the draft is
  `docs/paper/cvpr2027_aspect_spaces/drafts/v1/` (`paper_v1.md`, `02_related_work.md`, `03_method.md`).

## 3. Where things stand

- CRL against c* on CLIP L/14, ArtELingo held emotion mAP: c* 0.2671, CRL 0.2554 (point values, stage note).
- Every claim uses AMI at 20 clusters; class-count readings exist only as diagnostics (8 on EmoSet, 9 on MMArt).
- c* held embeddings are saved: r10 `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held`, r15 exports;
  r14 exports not checked. Whether matched PercepT outputs for the three other encoders are saved: not checked.
- Emotion retrieval below the teacher: CLIP L/14 on ArtELingo (0.2671 against 0.2684), Qwen and CLIP L/14 on MMArt
  (the gap mostly in contentment).
- CPU costs: r14 ArtELingo clustering 390 to 520 s per encoder, retrieval about 120 s (8 threads).

## 4. Open points for the next step

- Key: the exact cluster counts per aspect and data set (style and genre class counts: check them first).
- Key: which cells get bootstrap intervals for the new metrics, and which stay point values.
- Key: the CRL bar (WIN, LEVEL) on the same encoder, set before looking.
- Minor: ACC with more clusters than classes (one-to-one matching only at the class count).
- Minor: the per-class emotion look's form (per-class AP, c* minus teacher, per class).

## 5. Code and pitfalls

- A new round folder `src/test/20261103_<name>/` with a new module prefix (r17's own). Earlier round folders stay
  frozen: import their modules (r11 `rt_metrics`, r15 `mt_*`), never edit them; copy and extend in r17's folder.
- The scoring module is shared: r18 and r19 import it read-only after r17 merges into main. Give it a stable
  interface (embeddings and labels in, metric table out) and say so in the spec.
- The build runs unattended tonight (night mode); ask about a container restart before opening it (`loop-chats.md`).
  Codex: weekly limit at 5% until 2026-11-04 08:27, so the build starts on Claude.
- zsh: `git commit -F - <<'EOF'`; `*.json` and `*.log` are gitignored. The paper chat also commits here: commit only
  your own paths.
- Open r17's main chat at brief approval: `loop-next ~/.claude/templates/main_chat_start.md --label "r17 main" --main
  --new-workspace "CoSiR r17" --model sonnet --effort high --notify`.

## 6. State at handoff

- Running: nothing of r17. Beside it: the r18 and r19 decide chats, opened at the same time.
- Uncommitted: nothing.
- On disk over 1 GB: nothing new.
