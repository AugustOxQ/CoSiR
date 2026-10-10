# Engine switch check: round 6 on Matt's chain cost about twice round 5 and took 2.5 times as long, but the engine was not the cause

> 2026-11-26 · Spec: none (process check, descriptive) · Decision rule: none (descriptive) · Run log: none; numbers are extracted from the chat transcripts, scripts and tables are in the scratchpad (see Files and storage) · Round reports: [round 4a](../v2/2026-11-22_round4_aff_vetoes.md), [round 5](../v2/2026-11-23_idea3_goemotions.md), [round 4b](../v2/2026-11-24_clip_lightweight_ft.md), [round 6](../v2/2026-11-25_artelingo_held_test.md) · User-read report: [engine_switch_check](../../../user_read/2026-10-10_engine_switch_check.md)

## Results brief

**What we tried.** We compared the cost (tokens, dollar weight, GPU) and the time of rounds 4 and 5, built on the Superpowers chain, with round 6, the first loop on Matt Pocock's chain ([round 6 spec](../../../superpowers/specs/2026-10-09-r6-held-test-design.md)). The aim: what to change to cut time and cost while keeping quality.

**Outcome against the bar.** There was no pre-set bar. Round 6 cost more and took longer on every measure, and quality did not drop:

| | Round 4 (2 experiments) | Round 5 | Round 6 |
|---|---|---|---|
| Dollar weight | $295 | $341 | $706 |
| Whole loop, wall clock | 3h22 and ~2h26 | 12h09 (5h57 idle) | 30h31 |
| Final review, blocking findings | 0 and 0 | 0 | 0 |

**What it does not show.** The rounds are not like for like: round 6 was one claim test (a 27.7 KB spec, 15 tickets, 43.6 GPU-hours), against two medium loops in round 4 and one in round 5. Dollar weights are API-price equivalents, not what the Max plan charges.

**Options for next.** Keep Matt's chain and adopt some of the eight changes below (each needs your approval), or switch back to Superpowers.

**Your reading.** (left for the user)

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The engine switch is not what doubled the cost. The drivers were the round's size, Opus on nearly every ticket, five very long implementers, cache rewrites in waiting subagents, and the coordinator's babysitting of GPU jobs. Time was lost mainly to one 6-hour stall and to post-verdict GPU work. Matt's own overhead was small next to the round's $706 ($10 for code mapping, about 12 minutes of ticket writing), even though code mapping cost twice round 5's $5.

**Our recommendation.** Keep Matt's chain and change how we use it. Changes 1 to 3 below have the largest expected effect, with no quality risk (1) or a neutral one (2, 3). This is our view; the decisions are yours.

</details>

## What was compared

- **Round 4 (Superpowers):** one chat (10-07 00:17 to 16:20) running two medium experiments: round 4a, the AFF vetoes (7 tasks, killed at development), and round 4b, CLIP lightweight fine-tuning (6 tasks). Specs 13.1 KB and 7.4 KB.
- **Round 5 (Superpowers):** one medium loop, 10 plan tasks, killed at development. Spec 14.4 KB.
- **Round 6 (Matt's chain):** a full loop, a claim test (the one-time ArtELingo held-split read of AFF), 15 tickets, 14 chats (decide 1, decide 2, build 1 to 6, run 1 to 5), verdict GO.
- **Dollar weight** (new term): input, output and cache tokens priced at API rates, used only to weight the token kinds. Opus 5.5 $4 fresh / $20 output / $0.20 cache read per MTok; Sonnet 5.5 $2 / $10 / $0.20; Haiku 5.5 $0.10 / $0.50 / $0.01. Cache writes: main chats at the 1-hour price (2x input), subagents at the 5-minute price (1.25x), as the usage records show.
- **Source:** usage extracted from the session transcripts; token counts are input tokens (fresh, cache writes, cache reads).

## Results

Round 6 cost twice round 5 and used twice its input tokens; the extra went to the build.

| | Round 4 (2 experiments) | Round 5 | Round 6 |
|---|---|---|---|
| Input tokens | 792M | 874M | 1,731M |
| Output tokens | 1.58M | 1.19M | 3.33M |
| Dollar weight | $295 | $341 | $706 |
| of which cache writes / cache reads / output | $110 / $154 / $31 | $148 / $169 / $24 | $306 / $334 / $66 |
| Main-chat calls, average context | 764, 500k | 592, 453k | 2,422 over 14 chats, 151k |
| Subagents: count, calls | 36, 2,240 | 29, 2,221 | 50, 6,500 |
| Subagents on Opus | 15 of 36 | 12 of 29 | 41 of 50 |
| Tasks or tickets | 13 (7+6) | 10 | 15 |
| Implementer plus ticket review, per task | $4.2 (55/13) | $10 (100/10) | $24.5 (367/15) |
| GPU | 1.3 GPU-h (4b only) | none | 43.6 GPU-h on DAS6 (89 jobs, up to 9 at once) |

Round 6 by phase: decide $64, build $556 (79%), run including the report $85.

![Dollar weight per round by role](../../assets/2026-11-26_engine_switch_check/fig_dollar_by_role.png)

*Figure 1. Dollar weight by role; the roles sum to the totals above.*

| Role (input tokens, dollar weight) | Round 4 | Round 5 | Round 6 |
|---|---|---|---|
| Main chat (coordinator) | 382M, $139 | 268M, $111 | 368M, $167 (14 chats) |
| Decide (facts, literature, rule draft and review) | 23M, $9 | 87M, $27 | 107M, $39 |
| Code mapping before the build | 1M, $1 | 14M, $5 | 33M, $10 |
| Implementers | 87M, $39 (8) | 191M, $80 (6) | 709M, $293 (16) |
| Ticket or task reviews and re-reviews | 31M, $16 (12) | 50M, $20 (14) | 186M, $74 (14) |
| Final whole-branch review, fix wave, re-review | 132M, $39 | 108M, $41 | 285M, $107 |
| Independent re-derivation | 46M, $22 | 47M, $22 | 37M, $13 |
| Reports (writer, user-read, checks) | 89M, $31 | 108M, $35 | 6M, $3 |

Round 6 took 30.5 hours against 12h09 (round 5) and 2h26 to 3h22 (round 4).

| Phase | Round 4a | Round 4b | Round 5 | Round 6 |
|---|---|---|---|---|
| Decide | 1h11 | ~0 | 2h18 | 2h36 |
| Idle, waiting for the user | none | none | 5h57 | none |
| Build | 54 min | 52 min | 2h20 | 18h49 |
| Build per task | ~8 min | ~8 min | ~14 min | ~51 min (excluding the stall) |
| Run start to verdict | 11 min | 32 min | 53 min | 4h13 |
| Verdict to final report commit | 1h03 | 53 min | 1h18 | 4h50 |
| Whole loop | 3h22 (shared chat) | ~2h26 | 12h09 | 30h31 |

![Wall-clock time per round by phase](../../assets/2026-11-26_engine_switch_check/fig_time_by_phase.png)

*Figure 2. Wall-clock hours by phase. The marks show the 6-hour stall inside round 6's build and the post-verdict GPU time inside its last phase. Round 5's run began before its build ended, so its bars overlap by about 40 minutes and add up to more than its 12h09 wall time.*

**Quality held.** Every final review ended "confirmed with fixes", and no review changed a decision number.

| Check | Round 4a | Round 4b | Round 5 | Round 6 |
|---|---|---|---|---|
| Final review: blocking / should-fix / nit | 0/6/13 | 0/4/13 | 0/2/8 | 0/6/~11 (re-review 0/0/5) |
| Pre-build rule review findings | 22 | n/a | 19 | 32 (2 blocking) |
| Independent re-derivation | agreed | agreed | agreed | agreed (389 and 249 quantities) |

## Analysis

Our reading throughout; the numbers are in the tables above.

**Cost.**

1. **Implementers are the bulk.** They took 709M tokens, 41% of round 6. Five long ones (tickets 15, 14, 13, 08, 10: 128M, 97M, 94M, 76M, 74M) took 470M, 27% of the loop. Their contexts grew to 400 to 590k tokens over 250 to 330 calls. Cost grows roughly with the square of context length, so one long subagent costs about twice two half-length ones. Round 5 had the same pattern (tasks 3 and 5: 73M and 98M), so it is not new with Matt's chain.
2. **Opus on nearly every ticket.** 14 of 16 implementers and all 14 ticket reviewers ran on Opus (round 4: implementers 3 of 8, reviews 6 of 12; round 5: 2 of 6 and 4 of 14). Small Opus tickets (02, 03, 04) cost 10 to 13M each; round 4's Sonnet tasks 1 to 5M. Ticket 12 on Sonnet still cost 12M, so ticket size matters as much as the model. A Sonnet cache read costs the same as an Opus one, so Sonnet saves on output and cache writes, not on re-reading context.
3. **Cache rewrites in waiting subagents.** Subagents use the 5-minute cache. 52 times a subagent waited more than 5 minutes (polling loops on long test or mutation runs; 16 resumes of a handed-back subagent for a fix round) and rewrote its whole context: 16.3M tokens, about $80, 11% of round 6. Another 66 single-step writes over 40k tokens (mostly large Bash output, and hand-backs) added 13.8M. Rounds 4 and 5 had 3.5M and 10.9M of such large writes in total.
4. **Coordinator cost did not fall.** Total main-chat input was 368M against 382M and 268M, although the per-call context fell from about 500k to 151k, because calls tripled to 2,422. The controller checked 15 diffs, ran suites, orchestrated 89 GPU jobs and was woken by many notifications (run 3: 36, including nine 30-minute Monitor expiries). Run chats alone took 152M tokens over 1,066 calls. Each of the 14 chat starts re-reads a handoff.
5. **Report drafting on Sonnet worked.** Drafting, user-read report and checks cost $3, against $31 to $35 for the report roles in rounds 4 and 5. The Opus claim review that followed found 6 major and 8 minor findings (all fixed), and a Sonnet number check re-derived about 330 numbers with 1 mismatch.
6. **Matt-specific overhead is small next to $706.** Code mapping $10 (round 5's equivalent $5, so twice as much); the controller wrote tickets and contracts in about 12 minutes.

**Time.**

1. **A 6-hour stall.** The first review of ticket 01 sat from 05:49 to 11:50 on a permission prompt in an unattended chat (its last command was a forced `git worktree remove`). Tickets 02 and 03 depended on 01, so the whole build waited.
2. **Build time per ticket was about 51 minutes** (excluding the stall) against 8 to 14 minutes in rounds 4 and 5: heavier tickets (guards, refusals, provenance for a one-time read), long Opus implementers, reviews on 14 of 15 tickets with 9 "changes needed" and about 12 fix rounds, dependency waves, and 5 context cuts in the build.
3. **Run and report.** The run took 4h13 to the verdict, then 4h50 more to the report: 28.8 of the 43.6 GPU-hours ran after the verdict (held verbaliser, reranker and fine-tuned features, 03:45 to 07:44 on nine GPUs) and served only descriptive rows. A listing-cache bug across held seeds cost 44 minutes.
4. **Codex was never used.** The auto-mode permission check refused the Codex command with the sandbox bypass, so the planned Codex re-derivation went back to Opus. At 03:03 you asked to move subagents to Codex because Claude usage was short.

**Quality.** We see no sign that round 6 lost quality. The ticket-15 review caught a blocking defect: a smoke run with no GPU outputs wrote "passed": true, which would have unlocked the read. So ticket reviews earned their cost in round 6 (one blocking defect caught, on ticket 15).

**Correction to the interim answer** (2026-10-09, mid-build): "per ticket about the same as round 5" held per implementer dispatch at that point. With the full loop, round 6 cost about 2.5 times round 5 per plan task ($24.5 against $10).

## Proposed changes

Ours, ranked by expected effect; estimates are rough, from round 6's numbers. Each needs your approval; rule changes go through `maintaining-rules.md`.

| # | Change | Expected effect | Quality risk |
|---|---|---|---|
| 1 | **Stuck-subagent watchdog, no prompts in unattended chats.** The controller checks each running subagent's transcript age every ~20 min; one silent for over 20 min with no command of its own running is treated as stuck and replaced. Briefs forbid deletions and forced commands (your "delete nothing" rule) so nothing triggers a prompt. | Time: would have saved about 6 of 30.5 hours | none |
| 2 | **Size tickets so an implementer ends under ~200k context;** one passing ~250k hands back a progress note and a fresh one continues. | Cost: 10 to 20% of a loop (the 27% spent in five long implementers) | neutral or better |
| 3 | **No waits over ~4 minutes inside subagents.** Long suites, mutation runs and GPU checks run in the background owned by the controller, or the subagent hands back and is re-dispatched; a fix round after a long pause goes to a fresh fixer with a short brief and the findings, not a resume of a 300k context. | Cost: about $80 (11%) of cache rewrites | neutral |
| 4 | **Model tier per ticket:** Sonnet for plumbing tickets (job wrappers, sync, smoke plumbing, file formats) and re-reviews of small fix rounds; Opus for verdict code, its reviews and the final review. | Cost: modest (output and writes at half price, shorter runs) | small; covered by the controller's diff check and the final review |
| 5 | **Run chats:** one queue script per GPU stage that notifies once at the end; no Monitor re-arming every 30 min. | Cost: maybe half of the 152M-token run chats | none |
| 6 | **Descriptive pass after the verdict** (28.8 GPU-h, ~4 h): your call, since it changes what the descriptive table shows. Options: more GPUs, a held subsample for the descriptive baselines, or running it in the next chat while other work proceeds. | Time: up to ~4 h | depends on option |
| 7 | **Codex:** allow the Codex command for unattended chats (it bypasses Codex's sandbox, which cannot run in this container), or drop Codex jobs from unattended chats. | Unblocks or removes a planned path | none |
| 8 | **Keep:** the rule review before the build, ticket reviews on verdict code, the final whole-branch review with re-derivation, Sonnet report drafting with an Opus claim review. | | |

## Deviations

- **Not like for like:** one claim test against two medium loops (round 4 is two experiments in one chat). Compare per task and per role as well as in total.
- **Dollar weights** are API-price equivalents used to weight token kinds, not what the Max plan charges.
- **Scope:** the round 6 read chat (the next loop's start, about 0.5M tokens) is excluded; round 6's decide 1 shares a chat with round 5's read chat.
- **Time sources:** round 6's log rows for builds 2 to 4 were stamped ahead of the real clock; git times are used where they differ. Round 5's phase bars in Figure 2 overlap by about 40 minutes (its run started before its build ended).
- **Task counts:** round 6's count of 9 "changes needed" reviews comes from log rows; ticket 04's review row says "approve with nits" and still got a fix round.

## Decisions for you

1. **Keep Matt's chain, or switch back to Superpowers?** Our recommendation: keep it (our view; see the folded block above).
2. **Which of changes 1 to 7 to adopt?** Each needs your approval; 6 and 7 are yours by nature (they change what the descriptive table shows, and Codex's sandbox bypass).

## Files and storage

- **Outputs:** this report; the user-read report ([engine_switch_check](../../../user_read/2026-10-10_engine_switch_check.md)); figures and their scripts in `docs/reports/assets/2026-11-26_engine_switch_check/` (`fig_dollar_by_role.py`, `fig_time_by_phase.py`); new colours in `docs/reports/assets/palette.md`.
- **Analysis data and scripts:** in the controller's scratchpad `.../scratchpad/cost/` (`analysis_notes.md`, `rounds_facts.md` with file and line sources, `data.json`, `extract.py`, `roles.py`, `dollars.txt`, `subs_r6.txt`, `gaps_r6.txt`). Scratchpad files are temporary.
- **Storage:** no project data written and nothing left on disk beyond the files above; no disk ledger rows.
