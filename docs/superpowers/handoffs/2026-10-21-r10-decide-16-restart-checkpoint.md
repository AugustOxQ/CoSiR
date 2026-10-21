# Handoff: start the two fact-finding jobs, do the claim test's wrap-up, then grill and spec the D2 ablation and the backbone build

> Superseded: never opened as a chat. The user resumed "r10 decide 16" after the restart, and that chat used this file
> as its checkpoint. The next phase handoff is [r10 build 14](2026-10-21-r10-build-14-handoff.md).

## Brief
- Done: decision C for the claim test (user, stage 2; agent default: it covers stage 1); decision A, a five-part plan.
- State: the plan is ablation, analysis, improvement list, framing, and a backbone and ArtELingo-28 build. No bar.
- Next: start the two Codex fact jobs (briefs ready), do the wrap-up while they run, then the short grill and spec.
- Blocked: nothing. Cut for a container restart that the user asked for at 02:53.

> Written 2026-10-21 02:55 by "r10 decide 16". Loop step reached: after decision C and decision A, before the facts
> briefs (step 3) and the grill (step 5). Direction log entry: not written yet (§4 b).

## 1. Read in this order

1. This file.
2. [The r10 decide 16 handoff](2026-10-20-r10-decide-16-handoff.md) §3 (the claim test state) and §4 e, f (the
   wrap-up and the routing review, still open).
3. The two Codex briefs, ready to launch: `.scratch/r10-decide-16/f1_task.txt` (D2's parts, the Qwen views, the
   node route; for items 1 and 2) and `.scratch/r10-decide-16/f2_task.txt` (backbones and ArtELingo-28; for item 5).
   Each writes one file beside it (`f1_facts.md`, `f2_facts.md`).
4. `~/.claude/references/grill.md`, `~/.claude/rules/spec-templates.md`, `~/.claude/references/codex-jobs.md`.

Skills: `grilling` and `domain-modeling` for the short grill; `cluster-run` when the spec sets the node run.

## 2. Decided by the user, not to be reopened

- **Decision C, claim test stage 2** (user, 2026-10-21 about 02:30, given before our reading), in the user's words:
  "This seems a good overall result. (1) D2 cleared uncollapsed PercepT, that's a good starting point. But need to
  establish to more backbones and datasets. (2) Leiden community seems not really helping? [...] (3) paper settings
  PercepT seems worse than our PercepT core setting which fixed many things."
- **Agent default (no answer from the user, who went on to the plan):** decide 15 is closed and its points move here;
  the stage 2 reading covers stage 1 too. Decide 15's "where stage 2 runs" point is void.
- **Decision A** (user, 2026-10-21 about 02:50, own pick). "We did not really understand our current design." The
  user calls all of it sweep and analysis work, so it has no bar. In order:
  1. A quick ablation of D2's building parts, on the cluster, 3 seeds: what each part brings.
  2. After it, a deep analysis of the ablation, to find ways to improve (only emotion passes the instructed Qwen
     image now). Also a deep analysis of the four Qwen image embeddings (default and the three instructed ones): how
     they differ and correlate.
  3. A list of ways to improve, with a new deep research if needed. The three cluster-level (contrastive) candidates
     can be in the list, so track 2's decision B waits for this list.
  4. Our advantages and how to show them (storyline, framing, experiment, analysis). PercepT has label-free training
     and good cluster quality. Ours may be: any dataset (Qwen prompting), any backbone (to prove), lightweight (not
     sure).
  5. Code for other backbones and the ArtELingo-28 dataset.
  While item 1 runs on the cluster, items 3, 4 and 5 go on locally.
- The user's question at decision C, answered in chat: does PercepT gain more from more training data? (§3).

## 3. Where things stand

- **Training size, stage 1 (13,400 paintings) against stage 2 (42,969), same 12,281 held paintings** (from
  `verdict_stage1.json`, `verdict_stage2.json`, `diag_stage1.md`, `diag_stage2.md` in `src/test/20261020_claim_test/`):
  emotion D2 0.1001 → 0.1008 (+0.7%), core (own top topic) 0.0855 → 0.0874 (+2.3%), best paper settings K = 40
  0.0707 → 0.0711; style D2 −0.9%, core −0.1%; genre D2 0.449 → 0.444 (−1.2%), core 0.441 → 0.423 (−4.1%). The
  seed spread alone gives about ±0.001 per change, so the two emotion gains do not differ beyond noise (not tested
  formally). Two sizes are not a scaling curve.
- **Our reading, shown after the user's:** passes with a smaller lead than stage 1; no lead under Leiden; D2 below
  the instructed Qwen image on style (0.369 against 0.403) and genre (0.444 against 0.506). Differences told to the
  user: the backbone part is outside the rule's next step; Leiden lowers D2 (0.1008 → 0.0757) far more than core
  (0.0799 → 0.0762), and k-means at Leiden's own cluster count keeps the lead (0.0980 against 0.0789), so the count
  does not explain it; the user's reading leaves out the style and genre gap.
- **Track 2 candidates** (`docs/reports/literature/2026-12-17_cluster_level_losses.md`, "Candidate ideas"): 1 a
  teacher-partition anchor per head; 2 a TEMI-style cluster head per aspect on the buddy pairs; 3 a comparator probe
  (TURTLE, TEMI) on our features. DEC from a good partition has no support. None has run.
- **D2's parts** (model sheet spec `docs/superpowers/specs/2026-10-19-stage2-model-sheet-design.md` §3.2): a shared
  first layer per side (2048 to 1024), one second layer per aspect (1024 to 2048); buddy InfoNCE terms on each
  aspect's graph (emotion Gres, style Fres05, genre R0o) and own-caption terms; direct distillation of the image head
  toward the instructed Qwen image of its aspect, weight 0.5, temperature 0.05. Rows with results on development data:
  A1 (no trunk), T1 (no distillation), D1 (distillation only, one seed), D3 to D5, T2 to T5, I1.
- **Draft ablation (agent, not yet shown to the user):** leave-one-out from D2 on development data (13,400 fit,
  5,000 evaluation): D2; no trunk; no distillation; no buddy-graph terms; no own-caption terms; distillation only;
  graphs built on the default image instead of the instructed one; distillation toward the default image instead
  of the instructed one. About 8 rows × 3 seeds. f1's facts fix the switches and the cost.
- **Facts for item 4, already known:** stage 2 training took D2 598 s and PercepT core 3,558 s (5 seeds, full scale,
  same 3090; stage 2 report "What ran"). The graph picks and the distillation weight were chosen on development
  labels (model sheet D13), which weakens a plain "label-free" claim; f1 question 3 checks the full list.
- **node408:** 3 A6000 GPUs free, reservation about 68 h left at 02:50 (ends about 2026-10-23 23:30). Its A6000 did
  not reproduce stage 1 bit for bit (decide 16 handoff §3), so the ablation reruns D2 itself on the node as its
  reference row.

## 4. Open points for the next step

a. **Start the facts jobs first.** `codex-usage --room other other` (at 02:50: room, 87% left), then launch both
   briefs per `codex-jobs.md` (`-m gpt-6.1-sol -c model_reasoning_effort=medium`, `-C /project/CoSiR`, from a
   background Bash). Add their ledger rows after. While they run, ask the user no grill question (grill.md item 2).
b. **The claim test's wrap-up, while the facts run** (`research-loop.md`): the direction log entry with decision C
   and decision A (who decided: user; the stage 1 coverage: agent default), its "Now" paragraph; the stage 2 paper
   sentence (spec §6, the rule's S2-GO step; agent draft); the storage summary with every row of
   `.scratch/pending_deletions.md` and `cluster du --node node408` (about 33 GB of node inputs: keep them if the
   ablation runs there); a disk ledger row for `src/test/20261020_claim_test/outputs/stage2/` (4.6 GB, keep);
   `wrapup-check`, `disk-check`, `routing-check`; the routing review of decide 16 §4 f; `sync.sh`, commit and push.
c. **The short grill for items 1 and 2 (key):** the row set (show the draft above as a proposed sweep, not a pick
   among options); the analysis list for the ablation and for the four Qwen views. Development data only: the
   ablation will shape the next design, so held data is not read (`held-data.md`). Ask once about a plan page.
d. **The spec for item 5 (key), after f2:** which backbones and how D2 works on one that takes no instruction (keep
   the Qwen-instructed teacher and graphs, or the backbone's own text prompts). The "any backbone" claim depends on
   this. Also what ArtELingo-28 adds over ArtELingo.
e. **Where items 3 and 4 run:** they need no user decision until the list exists. Agent default: the run chat runs
   them while the node trains (item 3's literature part on the `deep-research` agent, Opus at deep depth; item 4 as
   design judgment, Opus `xhigh`); the final improvement list waits for item 2.
f. **Cuts:** one build chat for both specs, then a run chat (node408 for item 1; local CPU for the Qwen view
   analysis). Before each unattended chat, ask about a restart (`loop-chats.md`).

## 5. Code and pitfalls

- **The cluster CLI is `~/.claude/skills/cluster-run/cluster`.** The `cluster` on PATH is
  `/root/miniconda3/bin/cluster`, an unrelated graph tool; `du`, `status` and `logs` need `--node node408`.
- The claim test's round folder is closed (decide 16 handoff §5): do not edit `ct_*.py`, `ct2_*.py`, its queue
  scripts or `DECISION_RULE.md`. The ablation gets a new round folder and module prefix.
- JSON outputs are gitignored: `git add -f` for any new JSON that a report rests on.

## 6. State at handoff

- Running: nothing from this chat (no Codex job, subagent, GPU lock or node job). Other processes that the restart
  ends: the r10 main 3 chat (pid 7429) and the old r10 run 10 and run 11 sessions (pids 540155, 662996), whose work is
  committed. GPU check at 02:54: driver 595.99.02 in all three places, no GPU process.
- Uncommitted: nothing in git. `.scratch/r10-decide-16/` (two briefs, gitignored, on `/project`) survives the
  restart.
- On disk over 1 GB: as in the decide 16 handoff §6 (`outputs/stage2/` 4.6 GB, the pending deletions).
