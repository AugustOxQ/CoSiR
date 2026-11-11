# Handoff: r25 report, write the short auto report, the user-read report, the run log and the wrap-up for the paper-baseline cells

## Brief
- Done: VLM2Vec-V2 and BERTopic scored beside c* on the ArtELingo test split, AMI at 20 and at each class count.
- State: c* is above both in all 12 cells (WIN); d from +0.036 (VLM2Vec, emotion, 20) to +0.313 (BERTopic, style, 20).
- Next: short auto report with results brief, half-page user-read report, run log, direction log entry, wrap-up.
- Blocked: nothing; the paper tables chat already has the numbers.

> Written 2026-11-11 08:12 by the "r25 paper baselines" tab. Loop step reached: r25 is a small loop opened from a paper
> chat; step 8 is done (runs, gate, independent check, records committed); step 9 (reports) and the wrap-up are next.
> Direction log entry: none yet; this chat adds a short one.

## Flags for the top of the report and the next handoff

Agent defaults that shape results (all in `src/test/20261111_paper_baselines/decisions.md`):
1. BERTopic per painting: one document per test painting, its captions joined; one topic per painting (maybe).
2. BERTopic topic count: its default pipeline reduced to K topics, outliers reassigned so every painting has a topic;
   three UMAP seeds (0, 1, 2) in place of the three k-means states (maybe).
3. VLM2Vec-V2 prompt: Table 1's instruction in the repo's image-query form, `<|image_pad|> <instruction>\n`; the
   repo's default image size (3,136 to 1,003,520 pixels) (maybe).
4. VLM2Vec-V2 numerics: bf16, batch 16. Batch make-up moves rows a little (cosine 0.9986 to 0.9997 against batch 1;
   the build's bar was 0.999; fp32 gives 1.0). Accepted by the run chat (no).

Fix after the verdict-code review (B1): the cached MiniLM snapshot lacked its sentence-transformers files, so the first
BERTopic run used a fallback model (no Normalize, 512 tokens). The run chat added the four files at the same revision,
added a guard, and reran. Size check at K = 20 (stale against final): emotion 0.0720 against 0.0662, style 0.0927
against 0.0936, genre 0.3137 against 0.3092. No label changes.

Scope change: both comparators are also scored at each aspect's class count (emotion 9, style 27, genre 10). User
decision of 2026-11-11, passed on by the paper chat (cosir-paper README, commit 71be7b4).

## 1. Read in this order

1. `src/test/20261111_paper_baselines/records/cells.json` (12 cells) and `records/check.json` (the recompute). Every
   number in the reports comes from these two files.
2. `src/test/20261111_paper_baselines/decisions.md`: 17 lines, for the report's "Agent choices checked" section.
3. `docs/superpowers/handoffs/2026-11-11-r25-paper-baselines-handoff.md`, sections 2 and 5: the user's decisions and
   what the reports must hold (the numbers the paper needs go in the results table).
4. `docs/reports/auto/percept/2027-01-06_outside_methods.md` (r18): the model for a comparator report on these cells.
5. `~/.claude/templates/results_brief.md`, `~/.claude/templates/auto_report_template.md`, the rules `report-kinds.md`,
   `report-writing.md`, `user-read-reports.md`, and `~/.claude/references/ste-writing.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- The r25 design of 2026-11-11 (paper-baselines handoff, section 2): BERTopic and VLM2Vec as comparators in the main
  clustering table, CEMTM skipped; VLM2Vec-V2 2B under each aspect instruction; BERTopic on the test captions; the
  ArtELingo test split only; cutoff 2026-11-14.
- Table 2 uses each aspect's class count; K = 20 stays for the supplementary (cosir-paper README row of 2026-11-11,
  commit 71be7b4, "user, after seeing the recommendation").
- VLM2Vec's retrieval row in Table 3 belongs to the "CoSiR paper sec4 tables 2" chat (README commit fa0fac0; scorer
  commit e061d41c). The r25 report names it in one line only.

## 3. Where things stand

Mean AMI on the 12,281 test paintings; d = c* minus the comparator, with r17's paired bootstrap (2,000 draws); r = c*'s
own validation seed range (r18 record). All labels WIN.

| Aspect, K | c* | VLM2Vec-V2 | d [95%] | BERTopic | d [95%] | r |
|---|---:|---:|---|---:|---|---:|
| emotion, 9 | 0.1024 | 0.0638 | +0.0386 [+0.0349, +0.0423] | 0.0375 | +0.0649 [+0.0603, +0.0691] | 0.0036 |
| emotion, 20 | 0.1066 | 0.0710 | +0.0356 [+0.0320, +0.0391] | 0.0662 | +0.0404 [+0.0360, +0.0443] | 0.0059 |
| style, 27 | 0.4109 | 0.3730 | +0.0380 [+0.0335, +0.0421] | 0.1050 | +0.3059 [+0.2978, +0.3096] | 0.0087 |
| style, 20 | 0.4066 | 0.3700 | +0.0366 [+0.0320, +0.0411] | 0.0936 | +0.3131 [+0.3053, +0.3176] | 0.0107 |
| genre, 10 | 0.5101 | 0.4593 | +0.0508 [+0.0424, +0.0590] | 0.2605 | +0.2496 [+0.2387, +0.2604] | 0.0136 |
| genre, 20 | 0.5001 | 0.4595 | +0.0406 [+0.0346, +0.0469] | 0.3092 | +0.1909 [+0.1816, +0.1994] | 0.0139 |

- Gate: c*'s mean AMI equals r18's SC1 cells exactly (difference 0.0) at all six (aspect, K).
- Independent recompute (`pb_check.py`): sklearn AMI on the labelled rows, its own MiniBatchKMeans refit for VLM2Vec,
  BERTopic partitions reread: max gap 0.0 over 12 cells. It uses the same sklearn functions as r17's scorer, so it
  checks the plumbing (names, labels, masks, partition stacking), not the library.
- Run facts: VLM2Vec extraction 05:21 to 07:54 on the local GPU (153 minutes, about 250 ms per image, peak 12 GB,
  264 to 284 W: heavy by the 200 W bar, ended before the weekday 10:00 limit). Scoring 11 minutes on one CPU slot.
  BERTopic at K = 20: HDBSCAN found 92, 87 and 91 topics; outliers 5,384, 5,102 and 5,134, all reassigned.
- The paper tables chat got the numbers at 08:06 and the check result at 08:08.

## 4. Points the report must state (caveats)

- Emotion words in captions: ArtELingo captions often name the emotion, so BERTopic's emotion cells read label words;
  c* and VLM2Vec read only the image. Size not measured.
- BERTopic's topics are very unbalanced. At K = 20 the four largest topics hold 7,955 to 9,024 of 12,281 paintings
  (largest 2,478 to 3,269, smallest 14 to 21); at K = 9 and 10 the largest holds 5,111 to 6,050. Source: the npz
  sidecars `OUT/bertopic/topics_*.json`. This lowers its AMI and is part of its default pipeline.
- BERTopic reads text and c* reads the image: a modality difference set by the user's design. PercepT ran BERTopic
  multimodal (ViT-B/32 and MiniLM, 100 topics); ours is text only.
- VLM2Vec bf16 batch noise: its effect on AMI is not checked; it is far smaller than any d, which is at least 0.035.
- c*'s test values were known from earlier rounds. No r25 choice read a test result: the defaults were set at 05:10,
  before any score, and the class counts came from the user.

## 5. What to write, and pitfalls

1. Short auto report `docs/reports/auto/percept/<date>_paper_baselines.md`. CoSiR's sequence-number date is the next
   after the latest report there (now 2027-01-13, so 2027-01-14 unless another chat took it first). Open it with the
   results brief; add the `reports_sum.md` row; run `python scripts/check_reports_sum.py` and `docs-check`. Claim
   review of the draft against the two record files.
2. Half-page user-read report `docs/user_read/2026-11-11_paper_baselines.md`: a Haiku cold read, then a Sonnet `high`
   fidelity check against the auto report.
3. Run log `src/test/20261111_paper_baselines/20261111_paper_baselines_log.md`: problem, steps, the B1 root cause and
   fix. Timeline: `git log -- src/test/20261111_paper_baselines` (commits ff91fd3c to ac48d7e7) and the times above.
4. Direction log: a short entry (`~/.claude/templates/direction_log_entry.md`); no "Now" rewrite beyond one line, since
   decision C waits for the user.
5. Wrap-up: storage summary (`storage.md`), `wrapup-check`, project memory pointer, `/project/claude-config/sync.sh`
   (the Codex ledger `/project/claude-config/codex/jobs.md` has three new r25 rows to commit).

Pitfalls:
- `records/*.json` are gitignored by the repo's `*.json` rule; older rounds force-add their records (`git add -f`).
- Do not name or link the public GitHub repo in anything the paper may use.
- The round folder also holds Codex's notes (`findings.md`, `progress.md`, `task_plan.md`, `ticket_b_notes.md`,
  `classcount_notes.md`); they are build records, not results.

## 6. State at handoff

- Running: nothing of this chat. The GPU queue holds `r25:extract_val` (the VLM2Vec validation store for Table 3); it
  belongs to the "CoSiR paper sec4 tables 2" chat, which watches it.
- Uncommitted: nothing of this chat (`logs/*.exit` and `*.log` are small, untracked run files;
  `docs/superpowers/constitution.md` was modified before r25 and is not ours).
- On disk over 1 GB: nothing. r25 left 218 MB of VLM2Vec stores (`/data/SSD2/pre_extract/artelingo/vlm2vec_v2/`, kept),
  48 MB of outputs (`/data/SSD2/cosir_r25/20261111_paper_baselines/outputs/`), 26 MB of VLM2Vec code
  (`/data/SSD2/cosir_r25/ext/`), and the venvs `/root/venvs/vlm2vec` (148 MB) and `/root/venvs/bertopic` (177 MB).
- Pending deletions: three r25 rows in `.scratch/pending_deletions.md` (the stale BERTopic run, the two build
  worktrees and branches, the scratch smoke files).
