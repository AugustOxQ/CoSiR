# Handoff: r22 decide: decide with the user what to do for text-to-text and cross-modal retrieval, now top priority

## Brief
- Done: r17 is read and closed. The teacher gap check (report 2027-01-07) is committed. r17's storage is decided.
- State: r17, r18 and r19 are closed; r20, r21 and r22 have not started. After the weekly meeting, text-to-text and cross-modal retrieval are top priority.
- Next: step 2 with the user: what r22 covers, its type and size, and what happens to the other open items.
- Blocked: nothing.

> Written 2026-11-05 16:26 by tab "r20 decide". Loop step reached: r17's wrap-up done, step 2 next, decision A open. Direction log
> entry: [r17 read](../direction_log.md) ("New scores read", 2026-11-05).

## 1. Read in this order

1. The direction log's "Now" and the entry "New scores read" (2026-11-05), for the state of all loops.
2. The [experiment overview](../../user_read/2026-11-03_experiment_overview.md): "Retrieval: what is missing", to-do items 10, 12
   and 17, and "The plan".
3. The direction log's entry of 2026-11-03, the "Correction to r11 D1" and "Decision A list" lines (near line 208 to 226): what we
   know about c*'s caption side and the first design of a text-query test.
4. Parked ideas in the direction log: "MMArt perspective retrieval" and "MMArt perspective names as new Qwen instructions".
5. `docs/reports/literature/`: skim first, as step 3 says, if the direction needs a literature check.

Skills: none for step 2; `grilling` and `domain-modeling` at step 5 if the loop has a grill.

## 2. Decided by the user, not to be reopened

- **r22, plus more, is top priority** (user, 2026-11-05, after the weekly meeting). The user's words: "from the weekly meeting, my
  supervisor and collaborators think it might not be enough to do the image-to-image retrieval. It's still important to try
  text-to-text and also cross-modal retrieval. So this turns r22 (Plus more) into top priority."
- **Decision A for this step is made in this chat**, not in the r20 decide chat (user, 2026-11-05).
- r17 is closed (user, 2026-11-05). r18 and r19 are closed (direction log, 2026-11-05).
- The weekly report keeps r17's numbers without the contentment reason (user, against the recommendation, 2026-11-05).
- r18's rows go into the paper in a later, different chat; TEMI and the TURTLE selection stay parked (direction log).

## 3. Where things stand

- **Deadlines:** abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23.
- **Retrieval so far is image to image only:** all claim cells use an image query, told (the aspect head) or untold (the heads
  joined). r17: told c* against PercepT 24 of 24 WIN; untold c* against the plain image 22 of 24 WIN; c* against CRL on CLIP L/14,
  6 of 6 WIN.
- **c*'s caption side:** trunk and three heads on the Qwen caption mean, trained by the buddy terms and never scored (direction log,
  2026-11-03). It exists; no text retrieval result exists.
- **First design on record, not approved as a spec:** text queries by aspect through the caption side, a small exploration first:
  emotion, ArtELingo validation, a hit is an image with the caption's own emotion label, against plain Qwen text and the
  teacher's text. Estimated under 1 h. This is image retrieval from text, so it covers one cross-modal direction only.
- **Teacher gap (r17):** on ArtELingo emotion, contentment carries the ARI and ACC gap; genre labels alone match both models in ACC.
  Report: [2027-01-07_teacher_gap.md](../../reports/auto/percept/2027-01-07_teacher_gap.md). The user has read the numbers and said
  yes to the report. A genre baseline row is proposed, not decided.

The plan's other open items, each still to be placed by decision A:

| Item | Cost (our estimate) | Note |
|---|---|---|
| r20 qualitative figure (item 20) | CPU minutes and design work | planned by 11-12 |
| Paper chat: r17, r18, r19 rows and the teacher gap text | no compute | a `claude-paper` chat |
| Genre baseline row | short spec, CPU under 1 h | from the teacher gap report |
| r21 head variants (items 4, 18) | GPU about 2 h | planned 11-17 to 11-20, supplementary |
| Item 17, a second instructed embedder | GPU 3 to 4 h | the user sets its priority; we advised after 11-16 |
| The note for the collaborator and supervisor (item 22) | none | its form still waits for the user |

## 4. Open points for the next step

Key, answer-first (the user answers before our view):

1. What do "text-to-text" and "cross-modal" mean for the paper? Which query and which target (text to image, image to text, text to
   text), and is a hit decided by an aspect label, or by a known pair such as a painting and its own caption?
2. Which text: ArtELingo captions, MMArt perspective captions, or both? Which comparators (plain Qwen text, the teacher's text,
   CLIP text, CLAY, TPIPS)?
3. Does this become a main claim of the paper before 2026-11-16, or supplementary material? Does the abstract change?
4. How large is "plus more"? Its answer sets the type and the loop size. Is a literature check needed on how the field scores
   text-to-text retrieval by aspect?

Minor: the order of the other open items in the table above. Date note: the plan says r21 and r22 run from 11-17 and the old "Now"
said from 11-06; this decision replaces both for r22.

The r20 decide chat recommended, before the weekly news: the paper chat and the r20 figure now, the genre row before the paper, and
item 17 after 11-16. The news changes the priority, so treat this as old advice.

## 5. Code and pitfalls

- A new round needs a new module prefix (round folders go on `sys.path`). The r17 to r19 prefixes `ns_`, r18's and r19's are taken;
  `tg_` is the teacher gap check.
- `ns_score` is frozen for scoring; reuse it, do not edit it. `ns_c13.py`'s source hash covers all but `compare`, `certify` and `main`.
- The r17 reports' numbers replay exactly: `src/test/20261104_teacher_gap_why/tg_why.py` shows how to import the r17 scorer without
  changing it.
- Whether Qwen text embeddings of the captions exist for each set is not checked; MMArt's perspective captions need them (Parked ideas).
- Tuning and picks read the development split; held rows are for the test (`held-data.md`).

## 6. State at handoff

- Running: nothing of this chat (no subagent, queue job or Codex job).
- Uncommitted: nothing of this chat's. `cluster-run/cluster.conf` in `claude-config` is another chat's change.
- On disk over 1 GB: nothing new. `.scratch/pending_deletions.md` holds no r17 row now.
- `routing-check` printed three items at the wrap-up; they wait for the tooling chat after 2026-11-09.
