# Handoff: r23 build: tickets, build and final review of the text claim loop, from the approved spec

## Brief
- Done: grill, spec and rule review. The user approved the spec at 05:09.
- State: no r23 code and no held score exist. On validation, c* is above both comparators in 6 of 9 label cells.
- Next: tickets from the spec, the build, the final review, then cut to the run chat.
- Blocked: nothing. The pair-hit limit still waits for the user's talk with the other authors.

> Written 2026-11-06 05:12 by tab "r23 decide 2". Loop step reached: after steps 5 and 6, before step 7.
> Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule. Skip the brief.
2. `~/.claude/references/matt-chain.md`, for the chain: tickets, build, ticket reviews, final review.
3. `src/test/20261105_text_retrieval/`: `tx_metrics.py`, `tx_core.py`, `tx_data.py`, `tx_followup.py`, `tx_extract.py`, `tx_model.py`. The code to reuse.
4. `src/test/20261102_mmart_test/`: `mt_data.py`, `mt_paths.py`, `mt_c13.py`. The test keys, their order, and the form of a re-derivation.
5. The [rule review](../../../src/test/20261106_text_claim/tcl_rule_review.md), only to see why a sentence of the spec is exact.

Skills: tdd (verdict code).

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. The rule was committed before any code (C7). A change to it is an amendment (hard rule 5).

## 3. Where things stand

- Validation numbers only, so no claim. c* against plain Qwen and the teacher's text: report 2027-01-09. Late fusion: report 2027-01-10.
- Pair bar estimate, validation, five seeds: image to text on ArtELingo is -0.0086 [-0.0155, -0.0016] against a bar of -0.02. Script: `src/test/20261106_text_claim/tcl_review_pair_estimate.py`. It decides nothing.
- Label-word shares in spec D9: measured inline on 2026-11-06 with r22's patterns, no script kept. Stage 3 reports them again.

## 4. Open points for the next step

None for the user. Agent defaults are in the spec's §4.

## 5. Code and pitfalls

- Round folder `src/test/20261106_text_claim/`, prefix `tcl_`. Outputs `/data/SSD2/cosir_r23/20261106_text_claim/`. New stores go in the shared cache (D17). The folder goes on `sys.path` like the earlier rounds.
- `tx_metrics.evaluate` fits its means on the evaluated rows. Stage 2 needs means from validation, so add optional means and keep the old behaviour, or the replay gate (iii) breaks.
- `tx_metrics.painting_bootstrap` draws with `multinomial`. The spec's interval uses `rt_stats.paired`'s draw order on paintings, with seeds: write it new, and test it on the spec's hand case.
- `tx_data.py` reads validation only, and `mmart_texts` reads validation keys only. Held caption ids come from the store's `index.json` by painting name. `rt_data.names('held')` gives the 12,281 paintings.
- Late fusion's read-outs act on the concatenated vector, as `tx_followup.check_joined` does. Do not center each part alone.
- Unlabelled rows stay in ranking and expansion, and are masked only at AP (r22). This differs from r11 and r15. Pair cells keep every row.
- `encode_store` records no truncation count; the build adds it. Text limit: 512 tokens; the longest MMArt text is 1,310 characters. RoBERTa uses max length 64.
- c*'s saved `config.loss` text says "buddy+own-caption". That label is stale: `switches.own` is 0.0 for all three heads.
- `*.json` is gitignored, so results files need `git add -f`.
- Device (D13): local GPU, jobs under 1 hour each. A heavy job that would fall between 10:00 and 19:00 on a weekday waits for 19:00. Report by 2026-11-07 12:00, at the latest 2026-11-08.
- Valid `codex-usage --room` kinds: figure, rederivation, fidelity, mechanical, tool, review, bug, report, implementer, other.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat. Old untracked files under `archive/` and three `queue_status.txt` files are not from this chat.
- On disk over 1 GB: nothing new. `.scratch/pending_deletions.md` keeps three deferred rows.
- Open from earlier: the `routing-check` items of the last handoff stay with the tooling chat after 2026-11-09.
