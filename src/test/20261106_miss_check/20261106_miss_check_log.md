# r24: short check of the MMArt miss (exploration, small loop)

> Written 2026-11-06 18:40, before any number of this round. Decided by the user in the r24 decide chat (options b and c of
> report 2027-01-11). Type and size: agent default. No claim, no bar, no verdict changes. Label mAP only: no pair numbers
> (spec 2026-11-06-text-claim-design, D1). Nothing here tunes anything: picks and validation means are r23's, committed.

## Questions

- **b, per class.** Does c*'s miss against plain Qwen (MMArt test, narrative text, genre, text to text, -0.0137) sit in a few frequent genres?
- **c, shared rows.** Does r23's MMArt failure come from the new paintings, or from MMArt's text or labels?

## Design

**b.** Saved per-query AP from r23, at r23's picks, c* against plain Qwen. Per genre class: query count and share, class mean AP of both, difference with the
D15 interval restricted to the class, contribution to the gap (share times difference), and the gap left when the class is dropped. Beside it: the same class differences on
MMArt validation, on the other four MMArt text kinds and on held ArtELingo captions (no intervals there).

**c.** The 11,528 paintings in both ArtELingo held and MMArt (r15's `shared_held`), gallery limited to them. Rows c*, late fusion, plain Qwen and the teacher's text, seeds 0 to 4, r23's picks and
validation means of the same text kind. Text and labels are crossed:

| Name | Text | Labels | Paintings |
|---|---|---|---|
| A | ArtELingo captions | ArtELingo | shared |
| B | ArtELingo captions | MMArt | shared |
| C | MMArt narrative | ArtELingo | shared |
| D | MMArt narrative | MMArt | shared |
| E | MMArt narrative | MMArt | new (r23's MMArt test, 16,384) |

Narrative text only (r23's bar text): 4 new stores, about 10 minutes of GPU. All 9 cells, four differences (c* and late fusion, each against plain Qwen and the teacher's text), D15 intervals.

## Answers that change our reading (written before any number)

1. **D against E.** D looks like A (c* 9 of 9 over plain Qwen, 5 or more over the teacher's text): the failure comes with the new paintings. D looks like E: the paintings are not the cause.
2. **If D looks like E:** B against D isolates the text, and C against D isolates the labels.
3. **b.** One or two classes hold 100% or more of the gap: the miss is a class effect. Otherwise it is broad.

None of these changes an r23 verdict. Each decides only how the paper words MMArt.

## Log

### 2026-11-06 19:15, results

Local GPU, 18:45 to 18:58. Commits on the build branch: part b 21abd9e0, part c d9cb9c9e, gates and label-word read d2b9f193. Outputs: `/data/SSD2/cosir_r24/20261106_miss_check/`
(277 MB: stores, scores, gates, logs). Records: `records/b_classes.json`, `records/c_cells.json`, `records/c_text_label_facts.json`.

- **Gates.** Image rows: minimum cosine 0.99999988 on 200 shared paintings. Rows of the new path against r23's forward, cut by the same row lists: largest difference 7.5e-8. Scoring path against r23's saved per-query AP: 12 units
  (c* and plain Qwen, 3 directions, both sets), largest difference 0.
- **Answers to the three questions.** Shared paintings with MMArt text and labels equal r23's new paintings in all four counts (8, 3, 9, 0), so the paintings do not change the counts. Text changes the counts, and labels change none (label swaps move 24 of 36 point estimates and 4 and 1 outcome words).
  Portrait holds 102% of the gap and portrait plus landscape 167%, so the miss is a class effect.
- **Labels agree:** genre 100% of 9,401 paintings, style 96.8% of 11,528, emotion 43.4% of 11,398. Genre cells are the same under both label sets for this reason, which is not a bug.
- **Words:** captions mean 14.5 and median 14; narrative mean 70.6 and median 71.

### Deviations

1. The brief asked for the new text stores in the shared cache's layout; they sit under the round's output folder (`stores/`). Nothing else reads them.
2. The label-word read in part b was added after the class table was read, so it is outside the design above: exploration inside an exploration.
3. The controller widened the gates after the diff read: all three directions, and a row-parity check for ArtELingo text. Codex's sampled CPU gate failed at 1e-5 on ArtELingo (1.0e-4, near ties depend on the matrix shape); the full GPU gates give 0.
