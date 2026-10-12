# Handoff: r8 decide 2: close step 2 with the user (FCMI's role, the paper plan, placement, loop size), then step 3's literature check on the learned grouping

## Brief
- Done: decision C and the L3 pick (keep the fallback) are recorded; the storage summary is done (11.8 GB deleted); step 2 is half done.
- State: the user picked grouping work. Main line: a learned grouping head on frozen CLIP that cuts across content. In parallel: a screen of fixed encoders on the cluster.
- Next: settle the four open points with the user, then run step 3 (literature check on the learned grouping).
- Blocked: the open points wait for the user.

> Written 2026-10-12 17:41 by the `r8 decide` chat (cut at its context threshold). Loop step reached: step 2, decision
> A in part, before step 3. Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md) ("Decided (A),
> in part").

## 1. Read in this order

1. `docs/superpowers/direction_log.md`: "Now", "Parked ideas" and the 2026-10-12 entry (decision C in the user's
   words, the L3 pick, decision A so far).
2. This handoff's §3 and §4: the step-2 discussion and what is still open.
3. `docs/user_read/2026-10-12_fcmi_intro.html` (sent to the user; our view is folded at its end).
4. For step 3: `docs/reports/literature/2026-11-28_design_l_novelty.md` (§1 item 5 and §2: conditional contrastive
   learning, Tsai et al. 2021 and CCL-K, ICLR 2022; FCMI) and `2026-11-27_style_sources.md` (the ALADIN, DINOv2 and
   SigLIP 2 rows). Round 7's report `docs/reports/auto/v2/2026-11-26_design_l_style_source.md` (Analysis) for the AMIs.
5. Before the grill: `~/.claude/references/grill.md`; before the spec: `~/.claude/rules/spec-templates.md`.

## 2. Decided by the user, not to be reopened

All 2026-10-12, in the `r8 decide` chat; recorded in the direction log entry.

- **Decision C** (user, before seeing ours): the ceiling (+3.92) shows a lot to gain on style × genre; CSD entangles
  style and genre so that it treats them almost as one aspect; go back to grouping work, on which the project rests.
- **L3** (user, own pick, before seeing ours; ours agreed): keep the fallback. Any later reuse of L3 names the rank
  (groups − 1).
- **Storage**: done (ledger rows written; `pending_deletions.md` empty).
- **Decision A, in part:**
  - Main line: idea (ii), a learned grouping head on frozen CLIP whose negatives come from the query's own content
    group, as the reader's fourth grouping in CSD's place. Affect (Leiden on GoEmotions caption probabilities), image
    and caption groupings stay (user, before any recommendation of ours).
  - In parallel: idea (i), a screen of fixed encoders (ALADIN, DINOv2, SigLIP 2; token statistics possible) on the
    round 7 pipeline, run on the cluster after a local extraction. It counts as an engineering finding, not a method
    claim (user: "even it works it does not add novelty").
  - Comparators: X-Cluster criteria (iv) and told prompts (the aspect named to CLIP, as Multi-MaP does) (user).

## 3. Where things stand

- Friday 2026-10-16 is NO-GO under round 7's rule; AFF stays the current best. The user has not yet said what happens
  to the ArtELingo paper.
- Facts shown to the user in the discussion (round 7, descriptive AMIs):
  - The true styles agree with the content grouping at 0.32 and with the true genres at 0.22; raw CSD agrees with
    style 0.34, genre 0.33, content 0.40. So CSD's main gap is too little style. A grouping pushed to zero content
    agreement also moves away from the true styles (V1: style 0.10, content 0.11); the cuts kept style (0.32, 0.33)
    and lowered content a little (0.33, 0.38).
  - Placement: CLIP heads place the true styles at 63.7% (images) and 26.5% (captions); told true styles give +3.92,
    the true labels as the placement (an oracle) +21.74.
  - Told raw CSD (round 1 setup) gave only +0.13: a hint that the grouping limits more than the reader.
- On idea (ii), we told the user: CLIP carries some style (a linear probe with labels reaches 51.2%; CLIP groupings
  agree with style at 0.28 to 0.32), but every clustering mixes it with content; nothing in our method learns new
  directions; the content-group negatives may expose style, or colour, medium or genre. What makes a positive pair is
  not settled. (An earlier, stronger claim that clustering "never reaches" CLIP's style was corrected in chat.)
- On idea (i), we told the user: fix the encoder list before any result; pick by seed-42 R@1; AMI only describes
  (the user's generality principle: encoders fixed in advance, chosen without aspect labels; the grouping step knows
  nothing of aspects). Every combination read on seed 42 is selection to disclose. Extraction is about an hour on
  the local GPU; the rest is mostly CPU, so the cluster helps only for a large screen.
- On (v) prompts: the told form is a privileged comparator; the inferred form is parked idea N5 ("infer the aspect
  name, then embed"; VisDiff is the closest prior). A prompt bank that names aspects conflicts with the principle.

## 4. Open points for the next step

- **Key, FCMI's role** (ask answer-first; the user has the page): a grouping builder beside (ii), a comparator for
  (ii), or drop it. Our view is folded in the page (comparator).
- **Key, the paper plan** (answer-first): grouping work beside the ArtELingo paper, instead of it as the CVPR target,
  or after the abstract (2026-11-10). It sets the budget and the bar of this loop.
- **Key, placement**: in scope for this loop or later (for example, place gallery images by the new grouping's own
  assignment, not by CLIP heads; captions still need CLIP).
- **Type and loop size**: agent default new idea, full loop (step 3 with a literature check; is the learned grouping
  new, and what is its strongest comparator). Confirm with the user.
- **(v) inferred prompts (N5)**: stays parked unless the user pulls it in.
- **Minor, routing review**: `routing-check` printed two items at this wrap-up ("implementer: 4 measured jobs (median
  0% of 5h), no estimate recorded yet"; "2 of the last 10 Codex jobs were taken back"). Propose a review of
  `agent-routing.md` to the user; the user decides.

## 5. Code and pitfalls

- Round 7's code is merged in main: `src/test/20261126_design_l_style_source/` (groupings, heads, readers, scorers,
  gate). A new round needs a new module prefix (`r8_`).
- `results/` of round 7 is write-once and is kept (3.5 GB, ledgered).
- L3's `orth` keeps the literal pinv form where it passes and falls back where it fails (commit e3776ca).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing.
- On disk over 1 GB: round 7's `results/` (3.5 GB, keep), `res/cluster_jobs/` (3.1 GB, keep), the token cache on SSD2
  (1.1 GB, keep); all ledgered.
