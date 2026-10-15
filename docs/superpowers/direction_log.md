# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

The conditioned line's log (2026-09-28 to 2026-10-15, rounds 1 to 9, parked ideas) is archived: [direction log](../../archive/v2_conditioned/docs/superpowers/direction_log.md).

## Now

The conditioned line (CoSiR v2) was stopped on 2026-10-15 by the supervisors' decision and archived (tag
`archive/v2-conditioned`, folder `archive/v2_conditioned/`). The new line builds a condition-free model from the
buddy model, as an improvement over PercepT, on PercepT's task plus retrieval; the goal is a method that discovers
aspects, label-free if possible. CVPR is a soft target: submit if something works by about 2026-10-29, else keep
trying until the abstract deadline (2026-11-10), else no submission. Decision A is a draft under discussion.

## Parked ideas

(none yet for the new line; the v2 parked ideas are in the archived log)

## 2026-10-15 · Round 9 read; the supervisors stop the conditioned line, archive it and turn to a condition-free model

- **Read:** r9 (fixed AFF recipe): GO on Qwen3-VL-Embedding-2B, AFF − B′(A0) +0.63 [+0.49, +0.76] R@1, and SigLIP 2,
  +0.51 [+0.36, +0.66] (CLIP in round 6: +0.60). ArtELingo-28: full claim only on Qwen (+0.37 [+0.23, +0.51]); CLIP and
  SigLIP 2 inconclusive (+0.06, +0.15). AFF − B′(A1), the strongest condition-free scorer: −0.52 to +0.13 over the
  five reads.
- **Meaning (C):** the supervisors' decision at the weekly meeting (2026-10-14), relayed by the user: the line became
  too specific and too complicated to make a good CVPR paper; stop it and archive the CVPR plan (user). Our reading
  recommended stage 3; it agrees that B′(A1), condition-free, is level with AFF or ahead of it.
- **Decided (A):** draft, still to be discussed (user): a condition-free model built from the buddy model (redesigned,
  not a pull of its branch), as an improvement over PercepT; PercepT's task plus retrieval, with aspect-conditioned
  retrieval as a third candidate; a method that discovers aspects, label-free if possible, a looser use of labels if
  that is too hard. CVPR is soft: submit if something works by about 2026-10-29, else try until the abstract deadline
  2026-11-10, else no submission. Type new idea, full loop (agent default).
- **Also:** the r9 choices left open (the SigLIP 2 refit, H9's claim wording) are closed by the stop, not decided.
  Held data may be re-read; only tuning or sweeps on it are flagged (user; global rule `held-data.md`).
- **Why:** a narrower, harder task with a small margin over its own bar, and no margin over a condition-free scorer.
- **Links:** [r9 report](../../archive/v2_conditioned/docs/reports/auto/v2/2026-12-02_aff_scope.md) ·
  [archive index](../../archive/v2_conditioned/README.md) · paper plan: `docs/paper/README.md` (private repo) ·
  [r10 decide handoff](../../archive/v2_conditioned/docs/superpowers/handoffs/2026-10-14-r10-decide-handoff.md)

