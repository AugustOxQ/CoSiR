# Short check of the MMArt miss: the failure comes with MMArt's text, not its paintings or labels

> 2026-11-06 (sequence name 2027-01-12) · Spec: none, small loop; design in the [round log](../../../../src/test/20261106_miss_check/20261106_miss_check_log.md), written before any number · Decision rule: none (exploration) · Checks the failed verdicts of [report 2027-01-11](2027-01-11_text_claim.md)

## Results brief

**What we tried.** r23's verdicts failed on MMArt, and we did not know why. We ran two short checks at r23's picks, in label mAP. First, we split c*'s one miss against plain Qwen (genre, text to text, -0.0137) by genre class. Second, we scored the 11,528 paintings that are in both held ArtELingo and MMArt, with text and labels crossed, beside r23's new MMArt paintings.

**Outcome.** There is no bar, and no verdict changes. Each number below counts WIN cells of 9.

| Paintings | Text | Labels | c* over plain Qwen | c* over the teacher's text | Late fusion over plain Qwen | Late fusion over the teacher's text |
|---|---|---|---|---|---|---|
| Shared | ArtELingo captions | ArtELingo | 9 | 6 | 9 | 4 |
| Shared | ArtELingo captions | MMArt | 9 | 6 | 9 | 4 |
| Shared | MMArt narrative | ArtELingo | 8 | 3 | 9 | 0 |
| Shared | MMArt narrative | MMArt | 8 | 3 | 9 | 0 |
| New (r23) | MMArt narrative | MMArt | 8 | 3 | 9 | 0 |

- **The paintings are not the cause.** On paintings from held ArtELingo, MMArt's text and labels give the same four counts as r23's new paintings.
- **The labels are not the cause.** A label swap changes no count. Genre labels agree on 100% of 9,401 paintings, style on 96.8% of 11,528, emotion on 43.4% of 11,398.
- **The text changes the counts.** With MMArt labels on both sides, c* loses 3 WINs over the teacher's text, and all 3 are genre cells (table in Results). Its 3 style WINs stay.
- **The miss sits in two large classes.** Portrait is -0.0605 [-0.0688, -0.0519] and landscape is -0.0483 [-0.0567, -0.0393]. Together they are 42% of queries and hold 167% of the gap; portrait alone holds 102%. Without portrait the gap is +0.0003.
- **The other classes.** Abstract is also below, -0.0272 [-0.0410, -0.0089]. The other 7 of 10 classes are above plain Qwen.
- **Label words are not the cause.** With texts that name their own genre removed, the gap stays: -0.0143 [-0.0194, -0.0084]. Few narrative texts name it: 4.9% for portrait, 7.5% for landscape.

**What it does not show.**

- **Which property of the text matters.** The two texts differ in several ways at once: author (humans or a Qwen model), length (mean 14.5 words or 70.6) and texts per painting (about 5 or 1). c*'s caption side also trained on ArtELingo captions. This check cannot separate these.
- **Other MMArt texts.** The shared rows ran narrative text only. On the new paintings, portrait and landscape are also below plain Qwen on formal, historical and unified text, and above it on emotional text (means, no intervals).
- **Why portrait and landscape.** We only know where the gap sits. Plain Qwen's AP is high in these classes (0.6220, 0.6503), but abstract and still life are high too, and they go opposite ways.
- **Limits of the type.** An exploration: five seeds, read-outs picked on validation by r23, no multiplicity correction. Pair numbers are out of scope (decision D1).
- **A thin cell moved.** Late fusion's count over the teacher's text is 4 on the shared rows against r23's 5 on all held rows. Its fifth cell, style image to text, is LEVEL here: +0.0003 [-0.0001, +0.0007]. r23's verdict stands, since it is a different row set.

**Options for next.**

a. **The what-is-missing table.** Your stated next step. No GPU; it takes the coverage tables of the 11-04 stage report up to r24.
b. **The shared rows on the other four MMArt texts.** About 50 minutes of GPU with the same code (our estimate). It could show whether all MMArt texts behave as narrative does.
c. **Take the claims to the paper (r23's option a).** One writing chat. The text limit above changes how MMArt is worded. The wording of c*'s pair numbers still waits for your talk with the other authors.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** r23's MMArt failure is a text effect, not a failure on new paintings. We are confident in this for narrative text: the counts match exactly on paintings the claim already covers, and a label swap moves nothing. The firm wording is that c*'s caption side holds on ArtELingo's captions and loses its genre WINs on MMArt's narrative text. Its style WINs and its image side are not touched by this result.

The reason is open. The spec's prior named it before the run: the text is generated, long and new to the caption side. Nothing here tests which of these matters. The class pattern fits a shift in how the caption side treats long text, but we did not test that either.

**Our recommendation.** Option a, as you planned, then option c. Word MMArt as a limit of the text. Option b can wait: it adds detail, and the wording does not depend on it.

</details>

## What ran

- **Stages and types:** one exploration, two parts. No spec; small loop.
- **Data and seeds:** r23's new MMArt test rows (16,384, narrative text); 11,528 shared paintings, 58,002 ArtELingo captions; seeds 0 to 4. No tuning: picks and validation means are r23's committed ones.
- **Code:** `src/test/20261106_miss_check/`, merged into main; local GPU, 13 minutes. Codex wrote it; the controller read the diff and widened the gates.
- **Checks:** image rows, minimum cosine 0.99999988 on 200 paintings; rows against r23's forward, largest difference 7.5e-8; 12 scoring units against r23's saved per-query AP, largest difference 0.

## Results

### Part b: the missed cell per class

MMArt test, narrative text, genre, text to text, c* against plain Qwen, 13,420 queries. Overall 0.4564 against 0.4701, -0.0137; macro mAP 0.3921 against 0.3901. Contribution is share times difference, and contributions sum to the gap. The last two columns are means with no intervals.

| Class | Queries | Share | c* | Plain Qwen | Difference | Contribution | Validation | ArtELingo captions |
|---|---|---|---|---|---|---|---|---|
| portrait | 3,092 | 0.2304 | 0.5615 | 0.6220 | -0.0605 [-0.0688, -0.0519] | -0.0139 | -0.0620 | +0.0479 |
| landscape | 2,491 | 0.1856 | 0.6020 | 0.6503 | -0.0483 [-0.0567, -0.0393] | -0.0090 | -0.0556 | +0.0216 |
| genre_painting | 2,215 | 0.1651 | 0.2656 | 0.2464 | +0.0192 [+0.0156, +0.0228] | +0.0032 | +0.0236 | +0.0107 |
| religious_painting | 1,370 | 0.1021 | 0.5269 | 0.4925 | +0.0344 [+0.0107, +0.0572] | +0.0035 | +0.0110 | +0.0672 |
| abstract_painting | 1,141 | 0.0850 | 0.5829 | 0.6101 | -0.0272 [-0.0410, -0.0089] | -0.0023 | -0.0307 | +0.1164 |
| cityscape | 883 | 0.0658 | 0.4159 | 0.3843 | +0.0316 [+0.0166, +0.0477] | +0.0021 | +0.0422 | +0.0419 |
| sketch_and_study | 866 | 0.0645 | 0.0899 | 0.0822 | +0.0077 [+0.0058, +0.0099] | +0.0005 | +0.0024 | +0.0163 |
| still_life | 592 | 0.0441 | 0.6173 | 0.5892 | +0.0280 [+0.0145, +0.0415] | +0.0012 | +0.0350 | +0.1037 |
| nude_painting | 414 | 0.0308 | 0.1882 | 0.1662 | +0.0220 [+0.0130, +0.0313] | +0.0007 | +0.0223 | +0.0558 |
| illustration | 356 | 0.0265 | 0.0707 | 0.0579 | +0.0128 [+0.0098, +0.0160] | +0.0003 | +0.0320 | +0.0102 |

Every class keeps its sign on validation. Every class is above plain Qwen on ArtELingo captions. Macro mAP is above plain Qwen because each class counts once, and the three classes below are outnumbered.

### Part c: the genre and style cells, c* against the teacher's text

Shared paintings, MMArt labels, so only the text differs. Emotion is not in the table: its cells are not WIN under either text.

| Cell | ArtELingo captions | MMArt narrative |
|---|---|---|
| genre/t2i | +0.0403 [+0.0380, +0.0427] WIN | +0.0001 [-0.0030, +0.0031] LEVEL |
| genre/i2t | +0.0079 [+0.0058, +0.0103] WIN | -0.0346 [-0.0400, -0.0283] BELOW |
| genre/t2t | +0.0181 [+0.0136, +0.0229] WIN | -0.0260 [-0.0318, -0.0196] BELOW |
| style/t2i | +0.0158 [+0.0144, +0.0173] WIN | +0.0175 [+0.0146, +0.0205] WIN |
| style/i2t | +0.0141 [+0.0136, +0.0147] WIN | +0.0186 [+0.0171, +0.0201] WIN |
| style/t2t | +0.0119 [+0.0112, +0.0126] WIN | +0.0179 [+0.0163, +0.0196] WIN |

c* against plain Qwen on genre text to text, the missed cell: -0.0176 [-0.0230, -0.0115] BELOW on the shared rows with MMArt's text, -0.0137 [-0.0190, -0.0078] LEVEL on r23's new rows, and +0.0445 [+0.0399, +0.0493] WIN on the shared rows with ArtELingo's captions. All 180 cells are in `records/c_cells.json`.

**Verdict under the rule:** none; exploration. r23's verdicts stand.

## Analysis

Our reading: the check answers where the failure comes from, and it leaves why open. The three questions written before the run have clear answers: the shared rows equal the new rows, text changes the counts, and labels do not. The genre labels are identical on both sets, so genre cells cannot move under a label swap; this fits r23's cause sitting in the text.

## Deviations

- The label-word read in part b was added after the class table was read, so it is outside the design. It is labelled as exploration.
- New text stores sit in the round's output folder, not in the shared cache. Nothing else reads them.
- The gates were widened after the diff read: all three directions, and a row check. See the round log.

## Files and storage

- **Outputs:** `/data/SSD2/cosir_r24/20261106_miss_check/` (277 MB). Records in `src/test/20261106_miss_check/records/`.
- **Storage:** under 1 GB, so no ledger row; keep, since this report rests on it.
