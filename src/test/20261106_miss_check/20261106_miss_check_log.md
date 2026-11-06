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

