# Text claim round log

> 2026-11-06 · Spec: [design](../../../docs/superpowers/specs/2026-11-06-text-claim-design.md) · Run outputs: `/data/SSD2/cosir_r23/20261106_text_claim`

## Problem and rule

The frozen caption side and late fusion were tested against plain Qwen and the teacher’s text on held ArtELingo and MMArt narrative text. The rule is spec §3.4, §3.5 and §6. Each set stands on its own.

| ID | Row / set | P / T / Q | Bar | Verdict |
|---|---|---|---|---|
| SC1 | c* / ArtELingo | 9 / 6 / none | P=9; T≥5 holds, T=3 or 4 qualified; fusion also Q | holds |
| SC2 | c* / MMArt | 8 / 3 / none | P=9; T≥5 holds, T=3 or 4 qualified; fusion also Q | fails |
| SC3 | late fusion / ArtELingo | 9 / 5 / true | P=9; T≥5 holds, T=3 or 4 qualified; fusion also Q | holds |
| SC4 | late fusion / MMArt | 9 / 0 / true | P=9; T≥5 holds, T=3 or 4 qualified; fusion also Q | fails |


## Measured run order and commands

Times are Amsterdam local time. The `at` field is the recorded completion time. Cache skips remain in the table.

| Stage | Step | Completed | Seconds | Exit |
|---|---|---|---|---|
| 0 | extract_art_teacher | 2026-11-06 09:15 | 682.6 | 0 |
| 0 | extract_mm_plain | 2026-11-06 09:30 | 941.2 | 0 |
| 0 | extract_mm_teacher_narrative_val | 2026-11-06 09:33 | 173.7 | 0 |
| 0 | extract_mm_teacher_narrative_test | 2026-11-06 09:41 | 476.8 | 0 |
| 0 | extract_mm_teacher_formal_val | 2026-11-06 09:45 | 250.5 | 0 |
| 0 | extract_art_teacher | 2026-11-06 12:50 | 5.1 | 0 |
| 0 | extract_mm_plain | 2026-11-06 12:50 | 3.3 | 0 |
| 0 | extract_mm_teacher_narrative_val | 2026-11-06 12:50 | 3.2 | 0 |
| 0 | extract_mm_teacher_narrative_test | 2026-11-06 12:50 | 3.3 | 0 |
| 0 | extract_mm_teacher_formal_val | 2026-11-06 12:50 | 3.3 | 0 |
| 0 | extract_mm_teacher_formal_test | 2026-11-06 13:01 | 680.3 | 0 |
| 0 | extract_mm_teacher_emotional_val | 2026-11-06 13:04 | 164.8 | 0 |
| 0 | extract_mm_teacher_emotional_test | 2026-11-06 13:12 | 452.7 | 0 |
| 0 | extract_mm_teacher_historical_val | 2026-11-06 13:15 | 188.9 | 0 |
| 0 | extract_mm_teacher_historical_test | 2026-11-06 13:23 | 519.9 | 0 |
| 0 | extract_mm_teacher_unified_val | 2026-11-06 13:29 | 310.1 | 0 |
| 0 | extract_mm_teacher_unified_test | 2026-11-06 13:43 | 860.5 | 0 |
| 0 | extract_roberta | 2026-11-06 13:44 | 38.4 | 0 |
| 0 | gate_i | 2026-11-06 13:45 | 105.5 | 0 |
| 0 | extract_art_teacher | 2026-11-06 13:52 | 5.1 | 0 |
| 0 | extract_mm_plain | 2026-11-06 13:52 | 3.3 | 0 |
| 0 | extract_mm_teacher_narrative_val | 2026-11-06 13:52 | 3.3 | 0 |
| 0 | extract_mm_teacher_narrative_test | 2026-11-06 13:52 | 3.3 | 0 |
| 0 | extract_mm_teacher_formal_val | 2026-11-06 13:52 | 3.3 | 0 |
| 0 | extract_mm_teacher_formal_test | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_mm_teacher_emotional_val | 2026-11-06 13:53 | 3.4 | 0 |
| 0 | extract_mm_teacher_emotional_test | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_mm_teacher_historical_val | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_mm_teacher_historical_test | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_mm_teacher_unified_val | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_mm_teacher_unified_test | 2026-11-06 13:53 | 3.3 | 0 |
| 0 | extract_roberta | 2026-11-06 13:53 | 5.1 | 0 |
| 0 | gate_i | 2026-11-06 13:56 | 172.8 | 0 |
| 0 | gate_ii | 2026-11-06 13:57 | 40.6 | 0 |
| 0 | gate_iii | 2026-11-06 13:59 | 131.8 | 0 |
| 0 | gate_iv | 2026-11-06 13:59 | 7.7 | 0 |
| 1 | score_art_val | 2026-11-06 14:07 | 512.3 | 0 |
| 1 | score_mm_val_narrative | 2026-11-06 14:08 | 57.4 | 0 |
| 1 | score_mm_val_formal | 2026-11-06 14:09 | 58.9 | 0 |
| 1 | score_mm_val_emotional | 2026-11-06 14:10 | 58.3 | 0 |
| 1 | score_mm_val_historical | 2026-11-06 14:11 | 58.1 | 0 |
| 1 | score_mm_val_unified | 2026-11-06 14:12 | 58.9 | 0 |
| 1 | stage1_art | 2026-11-06 14:12 | 1.7 | 0 |
| 1 | stage1_mm | 2026-11-06 14:12 | 2.4 | 0 |
| 2 | score_art_held | 2026-11-06 14:34 | 1079.3 | 0 |
| 2 | score_mm_test_narrative | 2026-11-06 14:37 | 206.1 | 0 |
| 2 | decide | 2026-11-06 14:38 | 55.1 | 0 |
| 3 | score_art_held_side | 2026-11-06 15:09 | 564.7 | 0 |
| 3 | score_mm_test_formal | 2026-11-06 15:12 | 208.2 | 0 |
| 3 | score_mm_test_emotional | 2026-11-06 15:16 | 205.6 | 0 |
| 3 | score_mm_test_historical | 2026-11-06 15:19 | 206.4 | 0 |
| 3 | score_mm_test_unified | 2026-11-06 15:23 | 207.7 | 0 |
| 3 | diag | 2026-11-06 15:37 | 843.1 | 0 |

```text
2026-11-06 10:52 chain val start
2026-11-06 12:49 chain val end rc=143
2026-11-06 12:49 chain val start (D13 lifted)
2026-11-06 12:49 chain val end rc=127
2026-11-06 12:50 chain val start (D13 lifted, absolute path)
2026-11-06 13:45 chain val end rc=1
2026-11-06 13:52 chain val start (after the gate i fix)
2026-11-06 14:12 chain val end rc=0
2026-11-06 14:13 c13 compare val start
2026-11-06 14:15 c13 compare val end rc=0
2026-11-06 14:16 chain held start
2026-11-06 14:38 chain held end rc=0
2026-11-06 14:38 c13 derive held start
2026-11-06 14:49 c13 derive held end rc=0
2026-11-06 14:49 c13 compare all start
2026-11-06 14:59 c13 compare all end rc=0
2026-11-06 14:59 chain post start
2026-11-06 15:37 chain post end rc=0
```

The failed gate check returned zero as a process; the chain stopped on the gate result. This differs from treating the step exit code as a gate pass.

Commands recorded by the runner, in step order:

```sh
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py art_teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_plain
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind formal --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py art_teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_plain
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind formal --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind formal --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind emotional --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind emotional --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind historical --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind historical --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind unified --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind unified --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py roberta
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py gate_i
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py art_teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_plain
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind narrative --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind formal --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind formal --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind emotional --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind emotional --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind historical --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind historical --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind unified --split val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py mm_teacher --kind unified --split test
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py roberta
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_extract.py gate_i
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_forward.py cuda
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_replay.py cuda
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_run.py _gate_iv
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set art_val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_val --kind narrative
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_val --kind formal
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_val --kind emotional
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_val --kind historical
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_val --kind unified
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_run.py _stage1 --set art_val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_run.py _stage1 --set mm_val
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set art_held --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_test --kind narrative --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_outcome.py
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set art_held --rows free clip_l14 siglip2_so400m siglip_so400m roberta
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_test --kind formal --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_test --kind emotional --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_test --kind historical --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_score.py --set mm_test --kind unified --rows cstar late plain teacher
/root/miniconda3/envs/CoSiR/bin/python /project/CoSiR/src/test/20261106_text_claim/tcl_diag.py --device cuda
```

## Gates and investigation

The user lifted the weekday wait in the amendment at 2026-11-06 12:49, commit 60b22ac2. This changed when jobs ran, not any rule, row, pick or bar. Steps ran on Friday between the restricted hours on the user's word; earlier extraction also appears in the measured steps.

Gate (i) stopped the chain at 13:45 on `mm_val/formal/genre_word/teacher`, a diagnostic-only store. Its minimum was 0.99847019 against 0.999, a shortfall of 0.00052981; 1 of 200 rows failed. Length sorting changed padding: row 113 padded to 196 tokens in the gate versus 189 in the store. The trace's repeated gate batches give 0.99999994; the full store-batch rerun gives 0.99999988. Commit cbe5afce makes same-store batches decide this MMArt teacher rerun check; reference checks stay unchanged. The rerun passes 39 of 39. The bar stays 0.999, ticket 03's bar as r15; spec §3.5 names none.

The chain stopped and restarted for the context cut, for the lifted wait, and after a failed start where no job ran. No result depends on these stops. The chain log and all command arguments are in the round log.

C13 passed with 0 failures. Codex wrote it and fixed it in one round. Its numeric path means no numerical diversity: separate rule logic uses the run's numerical path because matrix shapes can change near-tied ranks.

C13's inputs include the frozen readers `rt_data.names`, `rt_data.labels`, `mt_data.keys`, `mt_data.labels`. Keys and painting labels are inputs; joins, caption labels and all decision logic are separate. A fault in these readers would be shared.

C13's `pass` permits a difference only when every SC that reads it is in `c13_void`; it then enters `void_differences`. Here both the void list and void differences are empty.

Measured cost differs by stage from the spec estimates. Extraction totals 95.67 minutes, against about 1.5 h. Scoring totals 58.03 minutes; `decide` takes 55.1 seconds against the CPU estimate of about 1 h. These are step wall times, not device utilisation. The longest step is `score_art_held`, 17.99 minutes; no logged job exceeds an hour. C13 held derive and comparison times are measured by the chain log, with full cost reconciliation in the round log.

The remaining required handoff readings apply as follows. `c13_void` also withholds cell claims. RoBERTa is normalised on read. A bad non-void score file stops `decide`. Plain Qwen LEVEL and BELOW words make no claim. Ticket 01 has no review trace; the independent numbers review stands in. C13's cache exceeded the output estimate. Weak tests remain disclosed. Full held-size C13 checks now completed, as its categories table shows.

Exploration: `label_word_share`, the within-query spread and `tcl_review_pair_estimate.py` were unrequested. The helper is off the run path. No reading above the spec's diagnostics is used to change a verdict.

The user-read report path uses the real date, `docs/user_read/2026-11-06_text_claim.md`, instead of the planned date in the spec. This task writes only the round log and auto report.

### Gate 1: recorded result

```json
{
  "passed": {
    "art_held/caption/emotion_focus/teacher": true,
    "art_held/caption/style_word/teacher": true,
    "art_held/caption/genre_word/teacher": true,
    "art_held/caption/none/roberta": true,
    "mm_test/narrative/default/plain": true,
    "mm_test/formal/default/plain": true,
    "mm_test/emotional/default/plain": true,
    "mm_test/historical/default/plain": true,
    "mm_test/unified/default/plain": true,
    "mm_val/narrative/emotion_focus/teacher": true,
    "mm_val/narrative/style_word/teacher": true,
    "mm_val/narrative/genre_word/teacher": true,
    "mm_val/formal/emotion_focus/teacher": true,
    "mm_val/formal/style_word/teacher": true,
    "mm_val/formal/genre_word/teacher": true,
    "mm_val/emotional/emotion_focus/teacher": true,
    "mm_val/emotional/style_word/teacher": true,
    "mm_val/emotional/genre_word/teacher": true,
    "mm_val/historical/emotion_focus/teacher": true,
    "mm_val/historical/style_word/teacher": true,
    "mm_val/historical/genre_word/teacher": true,
    "mm_val/unified/emotion_focus/teacher": true,
    "mm_val/unified/style_word/teacher": true,
    "mm_val/unified/genre_word/teacher": true,
    "mm_test/narrative/emotion_focus/teacher": true,
    "mm_test/narrative/style_word/teacher": true,
    "mm_test/narrative/genre_word/teacher": true,
    "mm_test/formal/emotion_focus/teacher": true,
    "mm_test/formal/style_word/teacher": true,
    "mm_test/formal/genre_word/teacher": true,
    "mm_test/emotional/emotion_focus/teacher": true,
    "mm_test/emotional/style_word/teacher": true,
    "mm_test/emotional/genre_word/teacher": true,
    "mm_test/historical/emotion_focus/teacher": true,
    "mm_test/historical/style_word/teacher": true,
    "mm_test/historical/genre_word/teacher": true,
    "mm_test/unified/emotion_focus/teacher": true,
    "mm_test/unified/style_word/teacher": true,
    "mm_test/unified/genre_word/teacher": true
  },
  "overall": true,
  "n_stores": 39,
  "smoke": false,
  "parity_bar": 0.999,
  "detail": {
    "art_held/caption/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/emotion_focus_heldtxt",
      "parity_min_cosine": 0.9994956851005554,
      "parity_mean_cosine": 0.9998153448104858,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "art_held/caption/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/style_word_heldtxt",
      "parity_min_cosine": 0.9995563626289368,
      "parity_mean_cosine": 0.9998185038566589,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "art_held/caption/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/genre_word_heldtxt",
      "parity_min_cosine": 0.9991364479064941,
      "parity_mean_cosine": 0.9998477697372437,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "art_held/caption/none/roberta": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions_s2_heldtxt",
      "parity": null,
      "pass": true
    },
    "mm_test/narrative/default/plain": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/default/test",
      "parity_min_cosine": 0.9996649622917175,
      "parity_mean_cosine": 0.9998697638511658,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/formal/default/plain": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/default/test",
      "parity_min_cosine": 0.9996045827865601,
      "parity_mean_cosine": 0.9999377727508545,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/emotional/default/plain": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/default/test",
      "parity_min_cosine": 0.9996684193611145,
      "parity_mean_cosine": 0.9999051690101624,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/historical/default/plain": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/default/test",
      "parity_min_cosine": 0.9997226595878601,
      "parity_mean_cosine": 0.9999120831489563,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/unified/default/plain": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/default/test",
      "parity_min_cosine": 0.9996850490570068,
      "parity_mean_cosine": 0.9999894499778748,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/narrative/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/emotion_focus/val",
      "parity_min_cosine": 0.9997824430465698,
      "parity_mean_cosine": 0.9999107122421265,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/narrative/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/style_word/val",
      "parity_min_cosine": 0.9996577501296997,
      "parity_mean_cosine": 0.9999098181724548,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/narrative/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/genre_word/val",
      "parity_min_cosine": 0.9995320439338684,
      "parity_mean_cosine": 0.9998704791069031,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/formal/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/emotion_focus/val",
      "parity_min_cosine": 0.9997557997703552,
      "parity_mean_cosine": 0.9999856352806091,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/formal/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/style_word/val",
      "parity_min_cosine": 0.9996320605278015,
      "parity_mean_cosine": 0.9999620318412781,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/formal/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/genre_word/val",
      "parity_min_cosine": 0.9984701871871948,
      "parity_mean_cosine": 0.9999514818191528,
      "parity_n": 200,
      "parity": true,
      "parity_same_batches_min_cosine": 0.9999998807907104,
      "pass": true
    },
    "mm_val/emotional/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/emotion_focus/val",
      "parity_min_cosine": 0.9997543096542358,
      "parity_mean_cosine": 0.999927282333374,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/emotional/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/style_word/val",
      "parity_min_cosine": 0.9996941089630127,
      "parity_mean_cosine": 0.9999350905418396,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/emotional/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/genre_word/val",
      "parity_min_cosine": 0.9996041059494019,
      "parity_mean_cosine": 0.9999166131019592,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/historical/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/emotion_focus/val",
      "parity_min_cosine": 0.9996278285980225,
      "parity_mean_cosine": 0.9999383687973022,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/historical/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/style_word/val",
      "parity_min_cosine": 0.9994951486587524,
      "parity_mean_cosine": 0.999894380569458,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/historical/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/genre_word/val",
      "parity_min_cosine": 0.999653697013855,
      "parity_mean_cosine": 0.9999201893806458,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/unified/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/emotion_focus/val",
      "parity_min_cosine": 0.9999998807907104,
      "parity_mean_cosine": 1.0,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/unified/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/style_word/val",
      "parity_min_cosine": 0.9997302293777466,
      "parity_mean_cosine": 0.9999912977218628,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_val/unified/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/genre_word/val",
      "parity_min_cosine": 0.9997524619102478,
      "parity_mean_cosine": 0.9999921321868896,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/narrative/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/emotion_focus/test",
      "parity_min_cosine": 0.9997717142105103,
      "parity_mean_cosine": 0.9999599456787109,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/narrative/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/style_word/test",
      "parity_min_cosine": 0.999646008014679,
      "parity_mean_cosine": 0.9998992681503296,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/narrative/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/narrative/genre_word/test",
      "parity_min_cosine": 0.9995197057723999,
      "parity_mean_cosine": 0.999845027923584,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/formal/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/emotion_focus/test",
      "parity_min_cosine": 0.9997512698173523,
      "parity_mean_cosine": 0.9999542236328125,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/formal/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/style_word/test",
      "parity_min_cosine": 0.9995086193084717,
      "parity_mean_cosine": 0.9999342560768127,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/formal/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/formal/genre_word/test",
      "parity_min_cosine": 0.9995748996734619,
      "parity_mean_cosine": 0.9999435544013977,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/emotional/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/emotion_focus/test",
      "parity_min_cosine": 0.9997690320014954,
      "parity_mean_cosine": 0.9999548196792603,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/emotional/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/style_word/test",
      "parity_min_cosine": 0.9996063709259033,
      "parity_mean_cosine": 0.9999380707740784,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/emotional/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/emotional/genre_word/test",
      "parity_min_cosine": 0.9995846748352051,
      "parity_mean_cosine": 0.9998860955238342,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/historical/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/emotion_focus/test",
      "parity_min_cosine": 0.999610424041748,
      "parity_mean_cosine": 0.9999223351478577,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/historical/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/style_word/test",
      "parity_min_cosine": 0.9995704889297485,
      "parity_mean_cosine": 0.9998763799667358,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/historical/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/historical/genre_word/test",
      "parity_min_cosine": 0.999698281288147,
      "parity_mean_cosine": 0.9999168515205383,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/unified/emotion_focus/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/emotion_focus/test",
      "parity_min_cosine": 0.9999999403953552,
      "parity_mean_cosine": 1.0,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/unified/style_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/style_word/test",
      "parity_min_cosine": 0.9997270107269287,
      "parity_mean_cosine": 0.9999920129776001,
      "parity_n": 200,
      "parity": true,
      "pass": true
    },
    "mm_test/unified/genre_word/teacher": {
      "static": true,
      "why": "",
      "path": "/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text/unified/genre_word/test",
      "parity_min_cosine": 0.9997467994689941,
      "parity_mean_cosine": 0.9999920725822449,
      "parity_n": 200,
      "parity": true,
      "pass": true
    }
  }
}
```

### Gate 2: recorded result

```json
{
  "tolerance": 0.001,
  "files": {
    "emotion__s0": {
      "art_val_image": 5.936622619628906e-05,
      "art_held_image": 5.893409252166748e-05,
      "mm_test_image": 6.015598773956299e-05,
      "art_val_caption": 6.0945749282836914e-05
    },
    "emotion__s1": {
      "art_val_image": 5.970895290374756e-05,
      "art_held_image": 5.9857964515686035e-05,
      "mm_test_image": 6.0930848121643066e-05,
      "art_val_caption": 6.0886144638061523e-05
    },
    "emotion__s2": {
      "art_val_image": 5.9410929679870605e-05,
      "art_held_image": 5.792081356048584e-05,
      "mm_test_image": 6.099045276641846e-05,
      "art_val_caption": 5.8770179748535156e-05
    },
    "emotion__s3": {
      "art_val_image": 6.057322025299072e-05,
      "art_held_image": 6.073713302612305e-05,
      "mm_test_image": 6.076693534851074e-05,
      "art_val_caption": 6.0439109802246094e-05
    },
    "emotion__s4": {
      "art_val_image": 5.9992074966430664e-05,
      "art_held_image": 6.085634231567383e-05,
      "mm_test_image": 6.0871243476867676e-05,
      "art_val_caption": 6.099045276641846e-05
    },
    "style__s0": {
      "art_val_image": 5.9977173805236816e-05,
      "art_held_image": 6.115436553955078e-05,
      "mm_test_image": 6.060302257537842e-05,
      "art_val_caption": 3.053247928619385e-05
    },
    "style__s1": {
      "art_val_image": 6.002187728881836e-05,
      "art_held_image": 6.122887134552002e-05,
      "mm_test_image": 6.079673767089844e-05,
      "art_val_caption": 3.055483102798462e-05
    },
    "style__s2": {
      "art_val_image": 5.8963894844055176e-05,
      "art_held_image": 6.0111284255981445e-05,
      "mm_test_image": 6.0424208641052246e-05,
      "art_val_caption": 3.053247928619385e-05
    },
    "style__s3": {
      "art_val_image": 6.0811638832092285e-05,
      "art_held_image": 6.122887134552002e-05,
      "mm_test_image": 6.090104579925537e-05,
      "art_val_caption": 3.053247928619385e-05
    },
    "style__s4": {
      "art_val_image": 5.81592321395874e-05,
      "art_held_image": 6.020069122314453e-05,
      "mm_test_image": 6.0886144638061523e-05,
      "art_val_caption": 3.053247928619385e-05
    },
    "genre__s0": {
      "art_val_image": 5.4076313972473145e-05,
      "art_held_image": 5.9545040130615234e-05,
      "mm_test_image": 5.882978439331055e-05,
      "art_val_caption": 5.519390106201172e-05
    },
    "genre__s1": {
      "art_val_image": 5.908310413360596e-05,
      "art_held_image": 5.926191806793213e-05,
      "mm_test_image": 5.7309865951538086e-05,
      "art_val_caption": 5.21540641784668e-05
    },
    "genre__s2": {
      "art_val_image": 5.8263540267944336e-05,
      "art_held_image": 5.9351325035095215e-05,
      "mm_test_image": 6.07222318649292e-05,
      "art_val_caption": 3.0547380447387695e-05
    },
    "genre__s3": {
      "art_val_image": 5.856156349182129e-05,
      "art_held_image": 5.917251110076904e-05,
      "mm_test_image": 5.9410929679870605e-05,
      "art_val_caption": 6.082653999328613e-05
    },
    "genre__s4": {
      "art_val_image": 5.906820297241211e-05,
      "art_held_image": 5.85019588470459e-05,
      "mm_test_image": 6.002187728881836e-05,
      "art_val_caption": 5.7250261306762695e-05
    }
  },
  "largest": {
    "art_val_image": 6.0811638832092285e-05,
    "art_held_image": 6.122887134552002e-05,
    "art_val_caption": 6.099045276641846e-05,
    "mm_test_image": 6.099045276641846e-05
  },
  "pass": {
    "art": true,
    "mm": true
  }
}
```

### Gate 3: recorded result

```json
{
  "clauses": {
    "core_label": {
      "pass": true,
      "worst": 3.5762786865234375e-07,
      "n": 81
    },
    "late_label": {
      "pass": true,
      "worst": 7.748603820800781e-07,
      "n": 27
    },
    "late_pair": {
      "pass": true,
      "worst": 0.0,
      "n": 18
    },
    "mmart_label": {
      "pass": true,
      "worst": 5.960464477539063e-08,
      "n": 24
    },
    "mmart_pair": {
      "pass": true,
      "worst": 2.9802322387695312e-08,
      "n": 16
    },
    "interval_hand": {
      "pass": true,
      "worst": 1.734723475976807e-17,
      "n": 5
    },
    "interval_tie": {
      "pass": true,
      "worst": 5.551115123125783e-17,
      "n": 12
    },
    "table2": {
      "pass": true,
      "worst": 0.00026545297091074005,
      "n": 36
    }
  },
  "overall": true
}
```

### Gate 4: recorded result

```json
{
  "message": {},
  "art": true,
  "mm": true,
  "sizes": {
    "art_held_paintings": 12281,
    "art_held_captions": 61744,
    "art_held_genre_labelled": 9949,
    "art_val_paintings": 6152,
    "art_val_captions": 30872,
    "mm_test_keys": 16384,
    "mm_val_keys": 5769
  }
}
```

### Gate trace

The investigation compared the saved store, the sampled rerun, repeated sampled batches, a full rerun in store batches, and single-row passes. The trace below preserves every measured value.

```json
{
  "genre_word": {
    "gate_min": 0.9984701871871948,
    "gate_mean": 0.9999514818191528,
    "under_bar": 1,
    "rerun_same_batches_min": 0.9999999403953552,
    "full_rerun_min": 0.9999998807907104,
    "full_rerun_mean": 1.0,
    "full_rerun_under_0_9999": 0,
    "low_rows": [
      113,
      190,
      155,
      107,
      156,
      116
    ],
    "low_cos_gate": [
      0.9984701871871948,
      0.9996297955513,
      0.999647855758667,
      0.9996732473373413,
      0.9996817111968994,
      0.9996905326843262
    ],
    "low_chars": [
      757,
      808,
      809,
      957,
      775,
      766
    ],
    "low_tokens": [
      176,
      186,
      182,
      202,
      178,
      179
    ],
    "low_store_batch_longest": [
      189,
      192,
      192,
      222,
      188,
      188
    ],
    "low_gate_batch_longest": [
      196,
      196,
      196,
      216,
      196,
      196
    ],
    "single_vs_store": [
      0.9998215436935425,
      0.999688982963562,
      0.999704897403717,
      0.9997841119766235,
      0.9998021125793457,
      0.9996843338012695
    ],
    "single_vs_gate": [
      0.9987562298774719,
      0.9997056722640991,
      0.9998265504837036,
      0.999607503414154,
      0.9997550249099731,
      0.9997276663780212
    ],
    "norm_store_low": [
      1.0,
      1.0,
      1.0,
      1.0,
      0.9999999403953552,
      1.0
    ]
  },
  "style_word": {
    "gate_min": 0.9996320605278015,
    "gate_mean": 0.9999620318412781,
    "under_bar": 0,
    "rerun_same_batches_min": 0.9999998807907104,
    "full_rerun_min": 0.9999998807907104,
    "full_rerun_mean": 1.0,
    "full_rerun_under_0_9999": 0,
    "low_rows": [
      185,
      113,
      79,
      58,
      107,
      64
    ],
    "low_cos_gate": [
      0.9996320605278015,
      0.9996399283409119,
      0.9996848106384277,
      0.9997066259384155,
      0.9997115731239319,
      0.9997266530990601
    ],
    "low_chars": [
      740,
      757,
      748,
      763,
      957,
      971
    ],
    "low_tokens": [
      170,
      177,
      165,
      171,
      203,
      206
    ],
    "low_store_batch_longest": [
      185,
      190,
      191,
      186,
      223,
      223
    ],
    "low_gate_batch_longest": [
      197,
      197,
      197,
      197,
      217,
      217
    ],
    "single_vs_store": [
      0.9997314214706421,
      0.9995124340057373,
      0.9997743964195251,
      0.9996592998504639,
      0.9997131824493408,
      0.9997247457504272
    ],
    "single_vs_gate": [
      0.9997861981391907,
      0.9997003078460693,
      0.9997535347938538,
      0.9997943639755249,
      0.9997348785400391,
      0.9997401237487793
    ],
    "norm_store_low": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "emotion_focus": {
    "gate_min": 0.9997557997703552,
    "gate_mean": 0.9999856352806091,
    "under_bar": 0,
    "rerun_same_batches_min": 0.9999998807907104,
    "full_rerun_min": 0.9999998807907104,
    "full_rerun_mean": 1.0,
    "full_rerun_under_0_9999": 0,
    "low_rows": [
      181,
      107,
      102,
      148,
      66,
      57
    ],
    "low_cos_gate": [
      0.9997557997703552,
      0.999799370765686,
      0.999815821647644,
      0.9998178482055664,
      0.9998306632041931,
      0.9998320937156677
    ],
    "low_chars": [
      706,
      957,
      707,
      699,
      690,
      932
    ],
    "low_tokens": [
      171,
      215,
      172,
      175,
      167,
      217
    ],
    "low_store_batch_longest": [
      189,
      235,
      189,
      192,
      188,
      236
    ],
    "low_gate_batch_longest": [
      193,
      229,
      193,
      193,
      193,
      229
    ],
    "single_vs_store": [
      0.9998335242271423,
      0.9997973442077637,
      0.9998325109481812,
      0.9998171329498291,
      0.9998226165771484,
      0.9998139142990112
    ],
    "single_vs_gate": [
      0.9998126029968262,
      0.9997905492782593,
      0.9998379945755005,
      0.999838650226593,
      0.9998219609260559,
      0.9998279809951782
    ],
    "norm_store_low": [
      0.9999999403953552,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
}
```

## Stage 1: frozen picks and seed ranges

Records were committed before held scores at 452e143e. Read-out names: `r0` is raw cosine; `center` subtracts validation means; `center_k10` adds fixed expansion.

### art

| Text | Row | Aspect | Direction | Pick | Seed range | Validation mAP |
|---|---|---|---|---|---|---|
| caption | clip_l14 | emotion | i2t | center_k10 | not seeded | 0.2231619298247371 |
| caption | clip_l14 | emotion | t2i | center_k10 | not seeded | 0.21652553113936787 |
| caption | clip_l14 | emotion | t2t | center | not seeded | 0.20226324922870362 |
| caption | clip_l14 | genre | i2t | center_k10 | not seeded | 0.3568805755063471 |
| caption | clip_l14 | genre | t2i | center_k10 | not seeded | 0.4168202607666191 |
| caption | clip_l14 | genre | t2t | center_k10 | not seeded | 0.25565831653353444 |
| caption | clip_l14 | style | i2t | center_k10 | not seeded | 0.10758613272347273 |
| caption | clip_l14 | style | t2i | center_k10 | not seeded | 0.1515518418491816 |
| caption | clip_l14 | style | t2t | center_k10 | not seeded | 0.09085552606546962 |
| caption | c* | emotion | i2t | center_k10 | 0.0015835392804179338 | 0.28935277183447516 |
| caption | c* | emotion | t2i | center_k10 | 0.0022895336383411125 | 0.2663283333564131 |
| caption | c* | emotion | t2t | center_k10 | 0.004365562632817288 | 0.3485995567334289 |
| caption | c* | genre | i2t | center | 0.006283789432247111 | 0.40010308042393933 |
| caption | c* | genre | t2i | center_k10 | 0.0038192963196493346 | 0.48624334924318513 |
| caption | c* | genre | t2t | center_k10 | 0.013483723260436564 | 0.31270711759221637 |
| caption | c* | style | i2t | center_k10 | 0.0010963156687638365 | 0.1355698450872982 |
| caption | c* | style | t2i | center_k10 | 0.0017992198418700367 | 0.17187890139764556 |
| caption | c* | style | t2t | center_k10 | 0.001419255319081908 | 0.10654915254805901 |
| caption | free variant | emotion | i2t | center_k10 | 0.0033973477355728754 | 0.27799012585905214 |
| caption | free variant | emotion | t2i | center_k10 | 0.0022365710636888525 | 0.258971017154459 |
| caption | free variant | emotion | t2t | center_k10 | 0.005737175147142559 | 0.31733064983326925 |
| caption | free variant | genre | i2t | center_k10 | 0.005193230354662737 | 0.3948690159260596 |
| caption | free variant | genre | t2i | center_k10 | 0.005352554770602258 | 0.4514251607667735 |
| caption | free variant | genre | t2t | center_k10 | 0.005565415977198784 | 0.30254402983248685 |
| caption | free variant | style | i2t | center_k10 | 0.002140453031675052 | 0.110295481020291 |
| caption | free variant | style | t2i | center_k10 | 0.003923339160341377 | 0.14501470112773826 |
| caption | free variant | style | t2t | center_k10 | 0.0003191957392822825 | 0.09208655331028699 |
| caption | late fusion | emotion | i2t | center_k10 | 0.0016773177146722773 | 0.27670263819670804 |
| caption | late fusion | emotion | t2i | center | 0.001025415944786473 | 0.24385913235275836 |
| caption | late fusion | emotion | t2t | center_k10 | 0.0030274943598353476 | 0.3273819039745054 |
| caption | late fusion | genre | i2t | center_k10 | 0.0067654696263132474 | 0.3784159520413396 |
| caption | late fusion | genre | t2i | center_k10 | 0.0015355242858010043 | 0.46739054987458395 |
| caption | late fusion | genre | t2t | center_k10 | 0.006050973775555801 | 0.30154569763262107 |
| caption | late fusion | style | i2t | center | 0.00037488343946663616 | 0.12376116837231957 |
| caption | late fusion | style | t2i | center_k10 | 0.0007592071993523797 | 0.16425307590590738 |
| caption | late fusion | style | t2t | center_k10 | 0.00032851245863634027 | 0.10087108382631974 |
| caption | plain Qwen | emotion | i2t | center_k10 | not seeded | 0.24068182180920236 |
| caption | plain Qwen | emotion | t2i | center | not seeded | 0.22363004631303246 |
| caption | plain Qwen | emotion | t2t | center_k10 | not seeded | 0.25411663906712467 |
| caption | plain Qwen | genre | i2t | center_k10 | not seeded | 0.3390122963814065 |
| caption | plain Qwen | genre | t2i | center_k10 | not seeded | 0.41790763578828766 |
| caption | plain Qwen | genre | t2t | center_k10 | not seeded | 0.26686896358426615 |
| caption | plain Qwen | style | i2t | center | not seeded | 0.10789937642206146 |
| caption | plain Qwen | style | t2i | center_k10 | not seeded | 0.13416177211601826 |
| caption | plain Qwen | style | t2t | center_k10 | not seeded | 0.09401607839160045 |
| caption | roberta | emotion | t2t | r0 | not seeded | 0.30990721198458543 |
| caption | siglip2_so400m | emotion | i2t | center_k10 | not seeded | 0.2214188355438094 |
| caption | siglip2_so400m | emotion | t2i | center_k10 | not seeded | 0.21160814870739209 |
| caption | siglip2_so400m | emotion | t2t | center | not seeded | 0.20291552737730634 |
| caption | siglip2_so400m | genre | i2t | center_k10 | not seeded | 0.3458560326358213 |
| caption | siglip2_so400m | genre | t2i | center_k10 | not seeded | 0.4292648307971472 |
| caption | siglip2_so400m | genre | t2t | center_k10 | not seeded | 0.24473463412838908 |
| caption | siglip2_so400m | style | i2t | center_k10 | not seeded | 0.10557769998675175 |
| caption | siglip2_so400m | style | t2i | center_k10 | not seeded | 0.14945008219764225 |
| caption | siglip2_so400m | style | t2t | center_k10 | not seeded | 0.08927040133132012 |
| caption | siglip_so400m | emotion | i2t | center_k10 | not seeded | 0.22294641681205027 |
| caption | siglip_so400m | emotion | t2i | center_k10 | not seeded | 0.2142692472841581 |
| caption | siglip_so400m | emotion | t2t | center | not seeded | 0.20650706434406169 |
| caption | siglip_so400m | genre | i2t | center_k10 | not seeded | 0.3510613076118452 |
| caption | siglip_so400m | genre | t2i | center_k10 | not seeded | 0.4231494906458192 |
| caption | siglip_so400m | genre | t2t | center_k10 | not seeded | 0.2471199221576725 |
| caption | siglip_so400m | style | i2t | center | not seeded | 0.10647760590465566 |
| caption | siglip_so400m | style | t2i | center_k10 | not seeded | 0.15294346507295292 |
| caption | siglip_so400m | style | t2t | center_k10 | not seeded | 0.08917822436870741 |
| caption | teacher's text | emotion | i2t | center_k10 | not seeded | 0.30363903356678495 |
| caption | teacher's text | emotion | t2i | center_k10 | not seeded | 0.27141332311680516 |
| caption | teacher's text | emotion | t2t | center_k10 | not seeded | 0.378997835090534 |
| caption | teacher's text | genre | i2t | center | not seeded | 0.39316442423470077 |
| caption | teacher's text | genre | t2i | center_k10 | not seeded | 0.4469728372208334 |
| caption | teacher's text | genre | t2t | center_k10 | not seeded | 0.2950969876776589 |
| caption | teacher's text | style | i2t | center | not seeded | 0.12189331406152018 |
| caption | teacher's text | style | t2i | center_k10 | not seeded | 0.15778164910403245 |
| caption | teacher's text | style | t2t | center_k10 | not seeded | 0.09525997388594488 |

Pair picks:

```json
{
  "caption": {
    "i2t": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  }
}
```

### mm

| Text | Row | Aspect | Direction | Pick | Seed range | Validation mAP |
|---|---|---|---|---|---|---|
| emotional | c* | emotion | i2t | center_k10 | 0.004406408577102394 | 0.3569517008904238 |
| emotional | c* | emotion | t2i | center | 0.0032817481206143806 | 0.26573368453674695 |
| emotional | c* | emotion | t2t | center_k10 | 0.009205333212182198 | 0.688656646805861 |
| emotional | c* | genre | i2t | center | 0.014087218124850198 | 0.48121752590068406 |
| emotional | c* | genre | t2i | center_k10 | 0.007254445665006659 | 0.5638126129457668 |
| emotional | c* | genre | t2t | center_k10 | 0.014219057476926256 | 0.4102871597926251 |
| emotional | c* | style | i2t | center_k10 | 0.0027081185519504114 | 0.21518121337187796 |
| emotional | c* | style | t2i | center_k10 | 0.003155936692284589 | 0.25050610382485056 |
| emotional | c* | style | t2t | center_k10 | 0.0019842790512209496 | 0.17772456571996706 |
| emotional | late fusion | emotion | i2t | center_k10 | 0.0008507571518551194 | 0.3387872075754498 |
| emotional | late fusion | emotion | t2i | center | 0.0010169551880392091 | 0.2412952149887341 |
| emotional | late fusion | emotion | t2t | center_k10 | 0.003549932658841626 | 0.6129882418364286 |
| emotional | late fusion | genre | i2t | center | 0.0045909144049603445 | 0.4609506704445055 |
| emotional | late fusion | genre | t2i | center_k10 | 0.0024018016112178353 | 0.5550982067566488 |
| emotional | late fusion | genre | t2t | center_k10 | 0.006899949481667389 | 0.3728229426961375 |
| emotional | late fusion | style | i2t | center | 0.0021347946728390854 | 0.18345668903313234 |
| emotional | late fusion | style | t2i | center_k10 | 0.0008524454236392143 | 0.23370547260757174 |
| emotional | late fusion | style | t2t | center_k10 | 0.0021569220077012063 | 0.14356491651345987 |
| emotional | plain Qwen | emotion | i2t | center_k10 | not seeded | 0.29176348417906184 |
| emotional | plain Qwen | emotion | t2i | center | not seeded | 0.22297142445227014 |
| emotional | plain Qwen | emotion | t2t | center_k10 | not seeded | 0.4830782798604089 |
| emotional | plain Qwen | genre | i2t | center | not seeded | 0.4151243099938424 |
| emotional | plain Qwen | genre | t2i | center_k10 | not seeded | 0.5045727506406537 |
| emotional | plain Qwen | genre | t2t | center_k10 | not seeded | 0.3328665818256346 |
| emotional | plain Qwen | style | i2t | center | not seeded | 0.15660758666251184 |
| emotional | plain Qwen | style | t2i | center_k10 | not seeded | 0.18186361026803435 |
| emotional | plain Qwen | style | t2t | center_k10 | not seeded | 0.13158605324224212 |
| emotional | teacher's text | emotion | i2t | center_k10 | not seeded | 0.36352807412712046 |
| emotional | teacher's text | emotion | t2i | center | not seeded | 0.2652898248395772 |
| emotional | teacher's text | emotion | t2t | center_k10 | not seeded | 0.7102688945999199 |
| emotional | teacher's text | genre | i2t | center | not seeded | 0.48757774759060896 |
| emotional | teacher's text | genre | t2i | center_k10 | not seeded | 0.5122972256522909 |
| emotional | teacher's text | genre | t2t | center_k10 | not seeded | 0.3491495443835654 |
| emotional | teacher's text | style | i2t | center | not seeded | 0.18359450206792083 |
| emotional | teacher's text | style | t2i | center_k10 | not seeded | 0.2153318102357234 |
| emotional | teacher's text | style | t2t | center | not seeded | 0.1289173545734626 |
| formal | c* | emotion | i2t | center_k10 | 0.0005876378797750048 | 0.19148636659014634 |
| formal | c* | emotion | t2i | center_k10 | 0.0008751867693959636 | 0.19409869910276437 |
| formal | c* | emotion | t2t | center_k10 | 0.000544721149624533 | 0.18571706940654434 |
| formal | c* | genre | i2t | center_k10 | 0.01154630906445192 | 0.46872422674451475 |
| formal | c* | genre | t2i | center_k10 | 0.005465611220313749 | 0.5470349893201605 |
| formal | c* | genre | t2t | center_k10 | 0.006946296024877285 | 0.418848662748678 |
| formal | c* | style | i2t | center_k10 | 0.0020184852208695037 | 0.21165262924190525 |
| formal | c* | style | t2i | center_k10 | 0.006586818242535014 | 0.2462867778612594 |
| formal | c* | style | t2t | center_k10 | 0.002037155931735757 | 0.17994645976455498 |
| formal | late fusion | emotion | i2t | r0 | 0.00014512624292004062 | 0.19200028184749238 |
| formal | late fusion | emotion | t2i | center_k10 | 0.0003508295782457538 | 0.1922038410154621 |
| formal | late fusion | emotion | t2t | center_k10 | 0.00017436868413450712 | 0.18575301444066777 |
| formal | late fusion | genre | i2t | center_k10 | 0.0035416787021700924 | 0.4778616314010663 |
| formal | late fusion | genre | t2i | center_k10 | 0.00167652785493011 | 0.5390598207239247 |
| formal | late fusion | genre | t2t | center_k10 | 0.0025877918077536477 | 0.43692602918059953 |
| formal | late fusion | style | i2t | center | 0.0018098005289778618 | 0.18208408134571802 |
| formal | late fusion | style | t2i | center_k10 | 0.0010647067664881216 | 0.22651035061396993 |
| formal | late fusion | style | t2t | center_k10 | 0.001472764399159665 | 0.16380460075422226 |
| formal | plain Qwen | emotion | i2t | r0 | not seeded | 0.1906565912904778 |
| formal | plain Qwen | emotion | t2i | center_k10 | not seeded | 0.18954161768388395 |
| formal | plain Qwen | emotion | t2t | center_k10 | not seeded | 0.18437241066110457 |
| formal | plain Qwen | genre | i2t | center_k10 | not seeded | 0.46342688982899843 |
| formal | plain Qwen | genre | t2i | center_k10 | not seeded | 0.5007182463098315 |
| formal | plain Qwen | genre | t2t | center_k10 | not seeded | 0.42795090288597376 |
| formal | plain Qwen | style | i2t | center | not seeded | 0.1586115666246857 |
| formal | plain Qwen | style | t2i | center_k10 | not seeded | 0.18210937499287747 |
| formal | plain Qwen | style | t2t | center_k10 | not seeded | 0.15438853960334453 |
| formal | teacher's text | emotion | i2t | center_k10 | not seeded | 0.1944739172163491 |
| formal | teacher's text | emotion | t2i | center_k10 | not seeded | 0.19609390306905225 |
| formal | teacher's text | emotion | t2t | center_k10 | not seeded | 0.18760847970061786 |
| formal | teacher's text | genre | i2t | center_k10 | not seeded | 0.507654441863722 |
| formal | teacher's text | genre | t2i | center_k10 | not seeded | 0.5607375864175986 |
| formal | teacher's text | genre | t2t | center_k10 | not seeded | 0.4611888558132688 |
| formal | teacher's text | style | i2t | center | not seeded | 0.21104887045834658 |
| formal | teacher's text | style | t2i | center_k10 | not seeded | 0.24301780788160113 |
| formal | teacher's text | style | t2t | center_k10 | not seeded | 0.17196507884666046 |
| historical | c* | emotion | i2t | center_k10 | 0.0011336765730740328 | 0.18475604334325121 |
| historical | c* | emotion | t2i | center_k10 | 0.001159492519299865 | 0.1861282644420211 |
| historical | c* | emotion | t2t | r0 | 0.0009167607069354156 | 0.180969162155073 |
| historical | c* | genre | i2t | center | 0.017227547341962857 | 0.36155951425328064 |
| historical | c* | genre | t2i | center_k10 | 0.008455417673301546 | 0.4350242522323226 |
| historical | c* | genre | t2t | center_k10 | 0.00995107888888791 | 0.2967976922877523 |
| historical | c* | style | i2t | center_k10 | 0.010113199275683715 | 0.41366448323046223 |
| historical | c* | style | t2i | center_k10 | 0.011578925687017072 | 0.36452346453538415 |
| historical | c* | style | t2t | center_k10 | 0.009212147248342295 | 0.553832940488678 |
| historical | late fusion | emotion | i2t | r0 | 8.111655178080768e-05 | 0.1894856008797415 |
| historical | late fusion | emotion | t2i | center | 0.00034927760081535086 | 0.18677061377451395 |
| historical | late fusion | emotion | t2t | r0 | 0.00024351831297617088 | 0.18419119828243877 |
| historical | late fusion | genre | i2t | center | 0.004791526818143599 | 0.38422585956817584 |
| historical | late fusion | genre | t2i | center_k10 | 0.0021415748936984835 | 0.4534261700698238 |
| historical | late fusion | genre | t2t | center | 0.003552961309917557 | 0.3047846755892242 |
| historical | late fusion | style | i2t | center_k10 | 0.005124027320001556 | 0.4327052463793991 |
| historical | late fusion | style | t2i | center_k10 | 0.0053779024401223174 | 0.34591672189593764 |
| historical | late fusion | style | t2t | center_k10 | 0.0030776406756979346 | 0.6003893544560359 |
| historical | plain Qwen | emotion | i2t | r0 | not seeded | 0.18829167128045607 |
| historical | plain Qwen | emotion | t2i | center | not seeded | 0.18513637105771502 |
| historical | plain Qwen | emotion | t2t | r0 | not seeded | 0.18407432348011518 |
| historical | plain Qwen | genre | i2t | center | not seeded | 0.36728442978168824 |
| historical | plain Qwen | genre | t2i | center_k10 | not seeded | 0.4228355929487968 |
| historical | plain Qwen | genre | t2t | center | not seeded | 0.3037045155812097 |
| historical | plain Qwen | style | i2t | center_k10 | not seeded | 0.39255678513171116 |
| historical | plain Qwen | style | t2i | center | not seeded | 0.29566556239388603 |
| historical | plain Qwen | style | t2t | center_k10 | not seeded | 0.5885361215662457 |
| historical | teacher's text | emotion | i2t | center_k10 | not seeded | 0.18988397480705035 |
| historical | teacher's text | emotion | t2i | center | not seeded | 0.1890519847468167 |
| historical | teacher's text | emotion | t2t | r0 | not seeded | 0.18398874846150304 |
| historical | teacher's text | genre | i2t | center | not seeded | 0.45641994305658823 |
| historical | teacher's text | genre | t2i | center_k10 | not seeded | 0.48118329898349355 |
| historical | teacher's text | genre | t2t | center | not seeded | 0.36206593559340333 |
| historical | teacher's text | style | i2t | center_k10 | not seeded | 0.5069533987322712 |
| historical | teacher's text | style | t2i | center | not seeded | 0.4088772011759051 |
| historical | teacher's text | style | t2t | center_k10 | not seeded | 0.7044617096801632 |
| narrative | c* | emotion | i2t | center_k10 | 0.0006058095905512872 | 0.19767853742602143 |
| narrative | c* | emotion | t2i | r0 | 0.002903689419642308 | 0.20386100968814916 |
| narrative | c* | emotion | t2t | r0 | 0.0010208325714550215 | 0.19606154890932873 |
| narrative | c* | genre | i2t | center_k10 | 0.018906627576475765 | 0.5109833717361318 |
| narrative | c* | genre | t2i | center_k10 | 0.004647868813807077 | 0.5691524517335688 |
| narrative | c* | genre | t2t | center_k10 | 0.02048754465725844 | 0.4627797635262989 |
| narrative | c* | style | i2t | center_k10 | 0.004261283494220863 | 0.2112035091284486 |
| narrative | c* | style | t2i | center_k10 | 0.005443395325543665 | 0.2501290492292588 |
| narrative | c* | style | t2t | center_k10 | 0.003909459720840475 | 0.17808104321224336 |
| narrative | late fusion | emotion | i2t | center_k10 | 0.00037731109838282095 | 0.19506009085833298 |
| narrative | late fusion | emotion | t2i | r0 | 0.0010122620393163972 | 0.20081327043252287 |
| narrative | late fusion | emotion | t2t | r0 | 0.0003961467636762195 | 0.19351152952421793 |
| narrative | late fusion | genre | i2t | center_k10 | 0.006559871496613967 | 0.5122558165777121 |
| narrative | late fusion | genre | t2i | center_k10 | 0.0019391122441096975 | 0.5565781659626392 |
| narrative | late fusion | genre | t2t | center_k10 | 0.0068928521257081155 | 0.4852173869155919 |
| narrative | late fusion | style | i2t | center | 0.0007288980964958169 | 0.18005695906602534 |
| narrative | late fusion | style | t2i | center_k10 | 0.0014326744437641314 | 0.23024501522851554 |
| narrative | late fusion | style | t2t | center_k10 | 0.00035079260669748646 | 0.1583595874132503 |
| narrative | plain Qwen | emotion | i2t | center | not seeded | 0.192022876076765 |
| narrative | plain Qwen | emotion | t2i | r0 | not seeded | 0.195764100243878 |
| narrative | plain Qwen | emotion | t2t | r0 | not seeded | 0.18962168512520725 |
| narrative | plain Qwen | genre | i2t | center_k10 | not seeded | 0.4963986329940534 |
| narrative | plain Qwen | genre | t2i | center_k10 | not seeded | 0.5111559081442859 |
| narrative | plain Qwen | genre | t2t | center_k10 | not seeded | 0.47776584925504606 |
| narrative | plain Qwen | style | i2t | center_k10 | not seeded | 0.16063657853510868 |
| narrative | plain Qwen | style | t2i | center_k10 | not seeded | 0.17790155947850272 |
| narrative | plain Qwen | style | t2t | center_k10 | not seeded | 0.1494567890582591 |
| narrative | teacher's text | emotion | i2t | center_k10 | not seeded | 0.20006503691779567 |
| narrative | teacher's text | emotion | t2i | r0 | not seeded | 0.2047834309589743 |
| narrative | teacher's text | emotion | t2t | center_k10 | not seeded | 0.19724177535773105 |
| narrative | teacher's text | genre | i2t | center_k10 | not seeded | 0.5368091075769326 |
| narrative | teacher's text | genre | t2i | center_k10 | not seeded | 0.5678042720770016 |
| narrative | teacher's text | genre | t2t | center_k10 | not seeded | 0.4876942097823044 |
| narrative | teacher's text | style | i2t | center | not seeded | 0.19090597208251592 |
| narrative | teacher's text | style | t2i | center_k10 | not seeded | 0.2312300606559065 |
| narrative | teacher's text | style | t2t | center_k10 | not seeded | 0.15877680785238515 |
| unified | c* | emotion | i2t | center_k10 | 0.0024137785160015246 | 0.23632651510948016 |
| unified | c* | emotion | t2i | center_k10 | 0.003046754847925437 | 0.2195719875679905 |
| unified | c* | emotion | t2t | center_k10 | 0.006273343489616057 | 0.277275156643424 |
| unified | c* | genre | i2t | center | 0.0110881727401404 | 0.5180069111656221 |
| unified | c* | genre | t2i | center_k10 | 0.0029263785075086535 | 0.584766727541025 |
| unified | c* | genre | t2t | center_k10 | 0.01828810598659042 | 0.46948973519572734 |
| unified | c* | style | i2t | center_k10 | 0.014435176098805402 | 0.339922660954192 |
| unified | c* | style | t2i | center_k10 | 0.011688525251066562 | 0.33627965371050994 |
| unified | c* | style | t2t | center_k10 | 0.01758864247959324 | 0.337251111403725 |
| unified | late fusion | emotion | i2t | center_k10 | 0.0010227738651674978 | 0.21791603386551922 |
| unified | late fusion | emotion | t2i | r0 | 0.001332019574262877 | 0.20776321896723102 |
| unified | late fusion | emotion | t2t | r0 | 0.001744009824665077 | 0.23908193194001287 |
| unified | late fusion | genre | i2t | center_k10 | 0.005443805852575645 | 0.5240465540325039 |
| unified | late fusion | genre | t2i | center_k10 | 0.0015125373049178492 | 0.5736581523216613 |
| unified | late fusion | genre | t2t | center_k10 | 0.005892572707537735 | 0.5018125480963274 |
| unified | late fusion | style | i2t | center_k10 | 0.0073753848066852035 | 0.2870955771272125 |
| unified | late fusion | style | t2i | center_k10 | 0.0035652471922594797 | 0.2923465653031246 |
| unified | late fusion | style | t2t | center_k10 | 0.007859628037500344 | 0.30792956381127645 |
| unified | plain Qwen | emotion | i2t | r0 | not seeded | 0.20344233108241555 |
| unified | plain Qwen | emotion | t2i | center | not seeded | 0.19907375705441446 |
| unified | plain Qwen | emotion | t2t | r0 | not seeded | 0.21129714706554437 |
| unified | plain Qwen | genre | i2t | center_k10 | not seeded | 0.5164306910466112 |
| unified | plain Qwen | genre | t2i | center_k10 | not seeded | 0.5323470847597168 |
| unified | plain Qwen | genre | t2t | center_k10 | not seeded | 0.504432648370175 |
| unified | plain Qwen | style | i2t | center_k10 | not seeded | 0.24781413104651184 |
| unified | plain Qwen | style | t2i | center_k10 | not seeded | 0.22020172244682068 |
| unified | plain Qwen | style | t2t | center_k10 | not seeded | 0.2795583979591655 |
| unified | teacher's text | emotion | i2t | center_k10 | not seeded | 0.25558185106325715 |
| unified | teacher's text | emotion | t2i | center | not seeded | 0.227427654324907 |
| unified | teacher's text | emotion | t2t | center_k10 | not seeded | 0.3257541788732177 |
| unified | teacher's text | genre | i2t | center | not seeded | 0.556693659915001 |
| unified | teacher's text | genre | t2i | center_k10 | not seeded | 0.5801074311768564 |
| unified | teacher's text | genre | t2t | center_k10 | not seeded | 0.5091037826485603 |
| unified | teacher's text | style | i2t | center_k10 | not seeded | 0.39343259734995784 |
| unified | teacher's text | style | t2i | center_k10 | not seeded | 0.36228274656961756 |
| unified | teacher's text | style | t2t | center_k10 | not seeded | 0.49330866233342224 |

Pair picks:

```json
{
  "emotional": {
    "i2t": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  },
  "formal": {
    "i2t": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  },
  "historical": {
    "i2t": {
      "late": {
        "head": "style",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  },
  "narrative": {
    "i2t": {
      "late": {
        "head": "emotion",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "style",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  },
  "unified": {
    "i2t": {
      "late": {
        "head": "style",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    },
    "t2i": {
      "late": {
        "head": "genre",
        "readout": "center"
      },
      "plain": {
        "readout": "center"
      }
    }
  }
}
```

## All bar cells and absolute label mAP

The interval is paired over paintings and model seeds. No multiplicity correction applies. Plain Qwen cells count only WIN.

### SC1

| Cell | Against | Difference [interval] | Word | Range | Read-outs | Paintings/queries | Candidate | Comparator |
|---|---|---|---|---|---|---|---|---|
| emotion t2i | plain Qwen | +0.0417 [+0.0411, +0.0424] | WIN | 0.0023 | center_k10/center | 12281/61744 | 0.2642 | 0.2225 |
| emotion i2t | plain Qwen | +0.0473 [+0.0454, +0.0491] | WIN | 0.0016 | center_k10/center_k10 | 12281/12281 | 0.2870 | 0.2397 |
| emotion t2t | plain Qwen | +0.0923 [+0.0904, +0.0939] | WIN | 0.0044 | center_k10/center_k10 | 12281/61744 | 0.3457 | 0.2534 |
| style t2i | plain Qwen | +0.0381 [+0.0368, +0.0394] | WIN | 0.0018 | center_k10/center_k10 | 12281/61744 | 0.1725 | 0.1344 |
| style i2t | plain Qwen | +0.0284 [+0.0278, +0.0291] | WIN | 0.0011 | center_k10/center | 12281/12281 | 0.1379 | 0.1095 |
| style t2t | plain Qwen | +0.0127 [+0.0121, +0.0133] | WIN | 0.0014 | center_k10/center_k10 | 12281/61744 | 0.1090 | 0.0963 |
| genre t2i | plain Qwen | +0.0657 [+0.0638, +0.0676] | WIN | 0.0038 | center_k10/center_k10 | 9949/49986 | 0.4836 | 0.4179 |
| genre i2t | plain Qwen | +0.0595 [+0.0570, +0.0622] | WIN | 0.0063 | center/center_k10 | 9949/9949 | 0.4028 | 0.3433 |
| genre t2t | plain Qwen | +0.0427 [+0.0383, +0.0475] | WIN | 0.0135 | center_k10/center_k10 | 9949/49986 | 0.3152 | 0.2725 |
| emotion t2i | teacher's text | -0.0036 [-0.0042, -0.0031] | BELOW | 0.0023 | center_k10/center_k10 | 12281/61744 | 0.2642 | 0.2678 |
| emotion i2t | teacher's text | -0.0166 [-0.0187, -0.0145] | BELOW | 0.0016 | center_k10/center_k10 | 12281/12281 | 0.2870 | 0.3035 |
| emotion t2t | teacher's text | -0.0308 [-0.0325, -0.0294] | BELOW | 0.0044 | center_k10/center_k10 | 12281/61744 | 0.3457 | 0.3765 |
| style t2i | teacher's text | +0.0159 [+0.0146, +0.0172] | WIN | 0.0018 | center_k10/center_k10 | 12281/61744 | 0.1725 | 0.1566 |
| style i2t | teacher's text | +0.0137 [+0.0132, +0.0142] | WIN | 0.0011 | center_k10/center | 12281/12281 | 0.1379 | 0.1243 |
| style t2t | teacher's text | +0.0113 [+0.0107, +0.0119] | WIN | 0.0014 | center_k10/center_k10 | 12281/61744 | 0.1090 | 0.0977 |
| genre t2i | teacher's text | +0.0403 [+0.0379, +0.0426] | WIN | 0.0038 | center_k10/center_k10 | 9949/49986 | 0.4836 | 0.4434 |
| genre i2t | teacher's text | +0.0062 [+0.0041, +0.0086] | WIN | 0.0063 | center/center | 9949/9949 | 0.4028 | 0.3966 |
| genre t2t | teacher's text | +0.0162 [+0.0118, +0.0211] | WIN | 0.0135 | center_k10/center_k10 | 9949/49986 | 0.3152 | 0.2990 |

### SC2

| Cell | Against | Difference [interval] | Word | Range | Read-outs | Paintings/queries | Candidate | Comparator |
|---|---|---|---|---|---|---|---|---|
| emotion t2i | plain Qwen | +0.0086 [+0.0074, +0.0097] | WIN | 0.0029 | r0/r0 | 16218/16218 | 0.2037 | 0.1951 |
| emotion i2t | plain Qwen | +0.0057 [+0.0053, +0.0060] | WIN | 0.0006 | center_k10/center | 16218/16218 | 0.1964 | 0.1907 |
| emotion t2t | plain Qwen | +0.0061 [+0.0056, +0.0065] | WIN | 0.0010 | r0/r0 | 16218/16218 | 0.1962 | 0.1902 |
| style t2i | plain Qwen | +0.0684 [+0.0659, +0.0709] | WIN | 0.0054 | center_k10/center_k10 | 16384/16384 | 0.2431 | 0.1747 |
| style i2t | plain Qwen | +0.0496 [+0.0481, +0.0511] | WIN | 0.0043 | center_k10/center_k10 | 16384/16384 | 0.2059 | 0.1563 |
| style t2t | plain Qwen | +0.0273 [+0.0257, +0.0289] | WIN | 0.0039 | center_k10/center_k10 | 16384/16384 | 0.1749 | 0.1476 |
| genre t2i | plain Qwen | +0.0575 [+0.0549, +0.0601] | WIN | 0.0046 | center_k10/center_k10 | 13420/13420 | 0.5595 | 0.5020 |
| genre i2t | plain Qwen | +0.0138 [+0.0092, +0.0194] | WIN | 0.0189 | center_k10/center_k10 | 13420/13420 | 0.5026 | 0.4888 |
| genre t2t | plain Qwen | -0.0137 [-0.0190, -0.0078] | LEVEL | 0.0205 | center_k10/center_k10 | 13420/13420 | 0.4564 | 0.4701 |
| emotion t2i | teacher's text | -0.0021 [-0.0033, -0.0009] | BELOW | 0.0029 | r0/r0 | 16218/16218 | 0.2037 | 0.2057 |
| emotion i2t | teacher's text | -0.0028 [-0.0032, -0.0024] | BELOW | 0.0006 | center_k10/center_k10 | 16218/16218 | 0.1964 | 0.1992 |
| emotion t2t | teacher's text | -0.0012 [-0.0017, -0.0008] | BELOW | 0.0010 | r0/center_k10 | 16218/16218 | 0.1962 | 0.1974 |
| style t2i | teacher's text | +0.0149 [+0.0125, +0.0174] | WIN | 0.0054 | center_k10/center_k10 | 16384/16384 | 0.2431 | 0.2281 |
| style i2t | teacher's text | +0.0183 [+0.0172, +0.0195] | WIN | 0.0043 | center_k10/center | 16384/16384 | 0.2059 | 0.1876 |
| style t2t | teacher's text | +0.0171 [+0.0155, +0.0187] | WIN | 0.0039 | center_k10/center_k10 | 16384/16384 | 0.1749 | 0.1578 |
| genre t2i | teacher's text | +0.0022 [-0.0007, +0.0050] | LEVEL | 0.0046 | center_k10/center_k10 | 13420/13420 | 0.5595 | 0.5574 |
| genre i2t | teacher's text | -0.0324 [-0.0372, -0.0268] | BELOW | 0.0189 | center_k10/center_k10 | 13420/13420 | 0.5026 | 0.5351 |
| genre t2t | teacher's text | -0.0264 [-0.0320, -0.0203] | BELOW | 0.0205 | center_k10/center_k10 | 13420/13420 | 0.4564 | 0.4828 |

### SC3

| Cell | Against | Difference [interval] | Word | Range | Read-outs | Paintings/queries | Candidate | Comparator |
|---|---|---|---|---|---|---|---|---|
| emotion t2i | plain Qwen | +0.0199 [+0.0197, +0.0202] | WIN | 0.0010 | center/center | 12281/61744 | 0.2424 | 0.2225 |
| emotion i2t | plain Qwen | +0.0366 [+0.0355, +0.0377] | WIN | 0.0017 | center_k10/center_k10 | 12281/12281 | 0.2763 | 0.2397 |
| emotion t2t | plain Qwen | +0.0717 [+0.0703, +0.0729] | WIN | 0.0030 | center_k10/center_k10 | 12281/61744 | 0.3251 | 0.2534 |
| style t2i | plain Qwen | +0.0299 [+0.0292, +0.0306] | WIN | 0.0008 | center_k10/center_k10 | 12281/61744 | 0.1643 | 0.1344 |
| style i2t | plain Qwen | +0.0156 [+0.0153, +0.0159] | WIN | 0.0004 | center/center | 12281/12281 | 0.1251 | 0.1095 |
| style t2t | plain Qwen | +0.0066 [+0.0064, +0.0068] | WIN | 0.0003 | center_k10/center_k10 | 12281/61744 | 0.1029 | 0.0963 |
| genre t2i | plain Qwen | +0.0482 [+0.0471, +0.0492] | WIN | 0.0015 | center_k10/center_k10 | 9949/49986 | 0.4661 | 0.4179 |
| genre i2t | plain Qwen | +0.0385 [+0.0365, +0.0409] | WIN | 0.0068 | center_k10/center_k10 | 9949/9949 | 0.3818 | 0.3433 |
| genre t2t | plain Qwen | +0.0333 [+0.0314, +0.0354] | WIN | 0.0061 | center_k10/center_k10 | 9949/49986 | 0.3058 | 0.2725 |
| emotion t2i | teacher's text | -0.0254 [-0.0259, -0.0250] | BELOW | 0.0010 | center/center_k10 | 12281/61744 | 0.2424 | 0.2678 |
| emotion i2t | teacher's text | -0.0272 [-0.0289, -0.0255] | BELOW | 0.0017 | center_k10/center_k10 | 12281/12281 | 0.2763 | 0.3035 |
| emotion t2t | teacher's text | -0.0514 [-0.0527, -0.0502] | BELOW | 0.0030 | center_k10/center_k10 | 12281/61744 | 0.3251 | 0.3765 |
| style t2i | teacher's text | +0.0076 [+0.0066, +0.0087] | WIN | 0.0008 | center_k10/center_k10 | 12281/61744 | 0.1643 | 0.1566 |
| style i2t | teacher's text | +0.0009 [+0.0005, +0.0012] | WIN | 0.0004 | center/center | 12281/12281 | 0.1251 | 0.1243 |
| style t2t | teacher's text | +0.0052 [+0.0050, +0.0055] | WIN | 0.0003 | center_k10/center_k10 | 12281/61744 | 0.1029 | 0.0977 |
| genre t2i | teacher's text | +0.0227 [+0.0206, +0.0248] | WIN | 0.0015 | center_k10/center_k10 | 9949/49986 | 0.4661 | 0.4434 |
| genre i2t | teacher's text | -0.0148 [-0.0171, -0.0122] | BELOW | 0.0068 | center_k10/center | 9949/9949 | 0.3818 | 0.3966 |
| genre t2t | teacher's text | +0.0068 [+0.0048, +0.0090] | WIN | 0.0061 | center_k10/center_k10 | 9949/49986 | 0.3058 | 0.2990 |

### SC4

| Cell | Against | Difference [interval] | Word | Range | Read-outs | Paintings/queries | Candidate | Comparator |
|---|---|---|---|---|---|---|---|---|
| emotion t2i | plain Qwen | +0.0051 [+0.0047, +0.0055] | WIN | 0.0010 | r0/r0 | 16218/16218 | 0.2001 | 0.1951 |
| emotion i2t | plain Qwen | +0.0036 [+0.0034, +0.0038] | WIN | 0.0004 | center_k10/center | 16218/16218 | 0.1943 | 0.1907 |
| emotion t2t | plain Qwen | +0.0038 [+0.0036, +0.0040] | WIN | 0.0004 | r0/r0 | 16218/16218 | 0.1940 | 0.1902 |
| style t2i | plain Qwen | +0.0523 [+0.0512, +0.0535] | WIN | 0.0014 | center_k10/center_k10 | 16384/16384 | 0.2270 | 0.1747 |
| style i2t | plain Qwen | +0.0189 [+0.0183, +0.0195] | WIN | 0.0007 | center/center_k10 | 16384/16384 | 0.1752 | 0.1563 |
| style t2t | plain Qwen | +0.0086 [+0.0084, +0.0088] | WIN | 0.0004 | center_k10/center_k10 | 16384/16384 | 0.1561 | 0.1476 |
| genre t2i | plain Qwen | +0.0458 [+0.0447, +0.0469] | WIN | 0.0019 | center_k10/center_k10 | 13420/13420 | 0.5478 | 0.5020 |
| genre i2t | plain Qwen | +0.0153 [+0.0135, +0.0173] | WIN | 0.0066 | center_k10/center_k10 | 13420/13420 | 0.5041 | 0.4888 |
| genre t2t | plain Qwen | +0.0083 [+0.0063, +0.0104] | WIN | 0.0069 | center_k10/center_k10 | 13420/13420 | 0.4784 | 0.4701 |
| emotion t2i | teacher's text | -0.0056 [-0.0062, -0.0050] | BELOW | 0.0010 | r0/r0 | 16218/16218 | 0.2001 | 0.2057 |
| emotion i2t | teacher's text | -0.0049 [-0.0052, -0.0045] | BELOW | 0.0004 | center_k10/center_k10 | 16218/16218 | 0.1943 | 0.1992 |
| emotion t2t | teacher's text | -0.0035 [-0.0038, -0.0031] | BELOW | 0.0004 | r0/center_k10 | 16218/16218 | 0.1940 | 0.1974 |
| style t2i | teacher's text | -0.0011 [-0.0026, +0.0004] | BELOW | 0.0014 | center_k10/center_k10 | 16384/16384 | 0.2270 | 0.2281 |
| style i2t | teacher's text | -0.0124 [-0.0131, -0.0117] | BELOW | 0.0007 | center/center | 16384/16384 | 0.1752 | 0.1876 |
| style t2t | teacher's text | -0.0016 [-0.0023, -0.0009] | BELOW | 0.0004 | center_k10/center_k10 | 16384/16384 | 0.1561 | 0.1578 |
| genre t2i | teacher's text | -0.0096 [-0.0120, -0.0072] | BELOW | 0.0019 | center_k10/center_k10 | 13420/13420 | 0.5478 | 0.5574 |
| genre i2t | teacher's text | -0.0310 [-0.0336, -0.0283] | BELOW | 0.0066 | center_k10/center_k10 | 13420/13420 | 0.5041 | 0.5351 |
| genre t2t | teacher's text | -0.0044 [-0.0072, -0.0015] | BELOW | 0.0069 | center_k10/center_k10 | 13420/13420 | 0.4784 | 0.4828 |

### Late fusion pair bar

| ID | Direction | Late fusion − plain Qwen pair R@10 | Read-outs | Paintings/queries | Q cell |
|---|---|---|---|---|---|
| SC3 | t2i | +0.0029 [+0.0010, +0.0049] | center/center | 12281/61744 | true |
| SC3 | i2t | -0.0030 [-0.0079, +0.0019] | center/center | 12281/12281 | true |
| SC4 | t2i | -0.0114 [-0.0143, -0.0085] | center/center | 16384/16384 | true |
| SC4 | i2t | -0.0084 [-0.0108, -0.0061] | center/center | 16384/16384 | true |

### Pair rows

The table lists pair metrics at the frozen pair picks.

| Set | Row | Head | Direction | Read-out | R@1 | R@5 | R@10 | Median rank |
|---|---|---|---|---|---|---|---|---|
| art_held | c* | genre | t2i | center | 0.0385 | 0.1053 | 0.1517 | 174.8000 |
| art_held | c* | genre | i2t | center | 0.0573 | 0.1615 | 0.2309 | 61.4000 |
| art_held | plain Qwen | all | t2i | center | 0.0905 | 0.1927 | 0.2512 | 94.0000 |
| art_held | plain Qwen | all | i2t | center | 0.1692 | 0.3289 | 0.4136 | 19.0000 |
| mm_test | c* | emotion | i2t | center | 0.0856 | 0.2107 | 0.2883 | 43.6000 |
| mm_test | c* | style | t2i | center | 0.0075 | 0.0310 | 0.0518 | 492.2000 |
| mm_test | plain Qwen | all | t2i | center | 0.6149 | 0.7936 | 0.8493 | 1.0000 |
| mm_test | plain Qwen | all | i2t | center | 0.6718 | 0.8530 | 0.9006 | 1.0000 |

## Analysis under spec §3.6

MMArt c* fails on genre text to text against plain Qwen: -0.0137 [-0.0190, -0.0078]. Its lower bound misses zero by 0.0190. The spec flagged emotion's validation margin, but genre decides this verdict. The LEVEL word against plain Qwen licenses no claim.

MMArt late fusion fails the teacher count. It has T=0 against at least three for qualified, although P=9 and Q passes. No one cell alone changes this count failure. Its teacher cell closest to WIN is style text to text, whose lower bound is -0.0023 against zero.

The deciding c* cell has a held spread near its validation range. Part 7 records validation range 0.0205, held spread 0.0192, and seed values 0.4479, 0.4568, 0.4671, 0.4545, 0.4556.

Late fusion's fifth ArtELingo teacher WIN has a thin margin. Style image to text is +0.0009 [+0.0005, +0.0012]. C13's closest cell is SC4 / label / late / teacher / genre / t2t, lower -0.0072451194 against -0.0068928519, distance 0.0003522675. This is the LEVEL boundary, not the WIN boundary.

ArtELingo late fusion holds despite a lower prior for holds than for qualified or fails. The other rows take their most likely verdict.

| Verdict | Holds | Qualified | Fails |
|---|---|---|---|
| c*, ArtELingo | 0.60 | 0.32 | 0.08 |
| c*, MMArt | 0.30 | 0.25 | 0.45 |
| Late fusion, ArtELingo | 0.20 | 0.40 | 0.40 |
| Late fusion, MMArt | 0.15 | 0.25 | 0.60 |

Observed outcomes are in the verdict table. The prewritten surprise was: c* not WIN over plain Qwen in any ArtELingo cell. MMArt priors are lower since its text is generated, long and new to
the caption side, which trained on caption means. The ArtELingo surprise did not occur.

Our reading: late fusion retains hits from different query groups. These observations describe the mechanism at the frozen weight; they do not identify a causal reason.

| Set/direction | Both share/rate | Only c* share/rate | Only plain share/rate | Neither share/rate |
|---|---|---|---|---|
| art_held/t2i | 0.1139/0.9999 | 0.0379/0.4407 | 0.1373/0.8124 | 0.7109/0.0169 |
| art_held/i2t | 0.1777/0.9892 | 0.0532/0.3895 | 0.2360/0.8278 | 0.5332/0.0352 |
| mm_test/t2i | 0.0486/1.0000 | 0.0032/0.5009 | 0.8007/0.9681 | 0.1475/0.0850 |
| mm_test/i2t | 0.2832/1.0000 | 0.0051/0.4288 | 0.6175/0.9710 | 0.0943/0.0773 |

The score spreads and rank correlations differ by aspect. The round log gives every sampled cell, label split and curve.

| Set | Spread-ratio range | Spearman-mean range |
|---|---|---|
| art_val | 0.9945 to 2.6125 | 0.1693 to 0.5635 |
| art_held | 1.0066 to 2.6252 | 0.1553 to 0.5572 |
| mm_test | 1.4508 to 3.6345 | 0.3365 to 0.7098 |

The validation curve covers the other seeds with the weight fixed for verdicts.

| Set/direction | At frozen weight | Maximum on grid | Weight at maximum |
|---|---|---|---|
| art_val/i2t | 0.4990 | 0.5114 | 0.9 |
| art_val/t2i | 0.3189 | 0.3206 | 0.8 |
| mm_val/i2t | 0.9464 | 0.9509 | 1.0 |
| mm_val/t2i | 0.9083 | 0.9151 | 0.9 |

Fixed read-outs change descriptive counts. These are counts of positive mean differences, the descriptive counterparts of P and T; stage 3 stores no intervals for these counts. They are not replacement WIN counts or verdicts.

| Set | Row | Fixed read-out | Above plain Qwen | Above teacher |
|---|---|---|---|---|
| art_held | c* | r0 | 9 | 6 |
| art_held | c* | center | 9 | 6 |
| art_held | c* | center_k10 | 9 | 6 |
| art_held | late fusion | r0 | 9 | 4 |
| art_held | late fusion | center | 9 | 2 |
| art_held | late fusion | center_k10 | 9 | 5 |
| mm_test | c* | r0 | 8 | 3 |
| mm_test | c* | center | 8 | 3 |
| mm_test | c* | center_k10 | 8 | 4 |
| mm_test | late fusion | r0 | 9 | 0 |
| mm_test | late fusion | center | 9 | 0 |
| mm_test | late fusion | center_k10 | 9 | 0 |

The risk checks leave the original verdicts in force. Means use validation rows; raw and centering rows, macro mAP, own-painting removal, label-word removal and emotion class AP are in the round log.

| Genre t2t | Row | Raw | Center | Picked | No own painting | No label words |
|---|---|---|---|---|---|---|
| art_held | c* | 0.2786 | 0.2917 | 0.3152 | 0.3149 | 0.3039 |
| art_held | late fusion | 0.2712 | 0.2880 | 0.3058 | 0.3055 | 0.2960 |
| art_held | plain Qwen | 0.2421 | 0.2603 | 0.2725 | 0.2722 | 0.2661 |
| art_held | teacher's text | 0.2756 | 0.2895 | 0.2990 | 0.2987 | 0.2895 |
| mm_test | c* | 0.4151 | 0.4338 | 0.4564 | 0.4564 | 0.4477 |
| mm_test | late fusion | 0.4350 | 0.4528 | 0.4784 | 0.4784 | 0.4698 |
| mm_test | plain Qwen | 0.4237 | 0.4419 | 0.4701 | 0.4701 | 0.4620 |
| mm_test | teacher's text | 0.4503 | 0.4691 | 0.4828 | 0.4828 | 0.4739 |

Truncation affects only the RoBERTa side row: 1 text. All other checked stores have 0 truncated texts. The full token limits and seed spreads are in the round log. MMArt uses generated text and different emotion labels; the spec reports shared-painting agreement, not a new estimate here.

Side rows have numbers without a bar. ArtELingo held label mAP below uses the validation picks; full directions and pair rows are in the round log.

| Row | Emotion t2i | Style t2i | Genre t2i |
|---|---|---|---|
| free variant | 0.2557 | 0.1462 | 0.4502 |
| CLIP L/14 | 0.2137 | 0.1498 | 0.4126 |
| SigLIP 2 | 0.2097 | 0.1489 | 0.4272 |
| SigLIP | 0.2110 | 0.1517 | 0.4188 |

RoBERTa emotion text to text: 0.3118.

MMArt’s other texts are descriptive and follow the verdicts. They change no verdict and cannot replace the narrative verdict.

| Text | Row | Emotion t2i | Style t2i | Genre t2i |
|---|---|---|---|---|
| formal | c* | 0.1924 | 0.2438 | 0.5405 |
| formal | late fusion | 0.1910 | 0.2239 | 0.5338 |
| formal | plain Qwen | 0.1888 | 0.1783 | 0.4948 |
| formal | teacher's text | 0.1953 | 0.2418 | 0.5592 |
| emotional | c* | 0.2587 | 0.2422 | 0.5592 |
| emotional | late fusion | 0.2359 | 0.2292 | 0.5494 |
| emotional | plain Qwen | 0.2186 | 0.1790 | 0.4978 |
| emotional | teacher's text | 0.2587 | 0.2112 | 0.5102 |
| historical | c* | 0.1857 | 0.3623 | 0.4245 |
| historical | late fusion | 0.1869 | 0.3472 | 0.4470 |
| historical | plain Qwen | 0.1856 | 0.2964 | 0.4168 |
| historical | teacher's text | 0.1893 | 0.4090 | 0.4790 |
| unified | c* | 0.2157 | 0.3321 | 0.5731 |
| unified | late fusion | 0.2058 | 0.2895 | 0.5639 |
| unified | plain Qwen | 0.1972 | 0.2164 | 0.5229 |
| unified | teacher's text | 0.2244 | 0.3603 | 0.5730 |

## Required handoff readings

### First handoff: readings a to i

These are build-time findings, read from the handoff. Measured run evidence below supersedes estimates and unfinished checks.



a. **C13's numeric path.** §3.4's 1e-6 per query holds only on one numerical path. Reason: near-tied cosines differ in the last float32 bits, and the GPU's matrix product depends on the matrix shape. A separate pipeline with no `tcl_`, `tx_`, `rt_` or `mt_` imports matched on 9 validation settings, with AP within 1.33e-7 and no first rank difference. Any one departure breaks 1e-6 (worst case 968 queries; mAP moved 4.3e-7; no R@10 flip). So C13 keeps separately written code and re-derived logic, and loses numerical diversity. The effect matters to a verdict only if a lower bound lies within about 1e-6 of a bar. A move to cell level would be an amendment, and it is not needed.
b. **Gate (i)'s parity bar.** 0.999 is ticket 03's, as r15; §3.5 says "compared" and names no bar. Measured here on the run's real rows at batch 64: all 8 checked rows pass, none of 200 under the bar in any, lowest 0.99914 (`genre_word`), 0.99950 (`emotion_focus`), 0.99956 (`style_word`). The margin is thin but the rows and batch size are fixed. MMArt plain rows: lowest 0.99960.
c. **`c13_void`** marks that SC's cells and writes no `cell_claims` entry. §6 names only the verdict. Left on the safe side.
d. **RoBERTa's store** is raw, so gate (i) tests norm above 0 only; `tcl_data` normalises on read. §3.6-only.
e. **A bad score file** in a set not voided crashes `decide`, with no void route. No false verdict results. Trace and fix it.
f. **Plain cells** carry a LEVEL or BELOW word in `cells.json`. It is no claim: only WIN counts, and P counts only WIN.
g. **Unrequested items**, reported at the exploration ceiling: `label_word_share`, the within-query spread, `tcl_review_pair_estimate.py` (a rule-review helper off the run path).
h. **Ticket 01** has no review trace. The numbers review's own interval and AP pass stands in.
i. **Cost:** the report gives measured times. `evaluate` at 61,744 takes 5.2 to 12.6 seconds, `painting_interval` 0.6 to 0.8 seconds, `decide` about 1 minute (the spec says CPU about 1 h; `--est 45m` is only an estimate). `score_art_held` takes about 16 minutes of ranking, row load not measured. Every job stays under 1 hour by these numbers. The 4,002 rankings are by arithmetic.


### Run-2 handoff: readings a to g

These are build-time findings, read from the handoff. Measured run evidence below supersedes estimates and unfinished checks.



a. **C13's inputs.** The keys and painting labels come from the frozen readers `rt_data.names`, `rt_data.labels`, `mt_data.keys`, `mt_data.labels`; the spec treats them as inputs. All joins, caption labels, rows and logic are C13's own code. A fault in those four readers would be shared.
b. **`pass` with a void (agent default, finding F7).** An item with a difference counts as a failure unless every SC that reads it is in `c13_void`. It then goes to `void_differences` and does not fail. With an empty `c13_void` nothing changes.
c. **The chain was stopped once and restarted** for the context cut, during its D13 sleep. No result depends on it.
d. **Measured cost.** At build time, extraction ran 3 to 4 times faster than the estimate (the full run later measured 95.67 minutes against about 1.5 h); D13 still held the `--est` jobs for 19:00. The report gives measured times.
e. **C13's cache** takes about 61 GB on `/data/SSD2` (reviewer's estimate), far over the spec's 2 GB for run outputs. Only C13 uses it; it is on the deletion list.
f. **Weak tests, T1 to T4 of `rereview.md`.** The code is right; these tests would not catch every change of it. Not fixed: they change no number.
g. **Not checked before tonight:** the full held derive, `compare` on real outputs, the real `tcl_forward` path at full size, the AP gap on c*, late fusion and teacher rows at full size.


## C13 final comparison

The final comparison supersedes the build-time incomplete checks.

| Category | Items | Failed | Largest difference | Not bit-identical | Void differences |
|---|---|---|---|---|---|
| ap | 902 | 0 | 1.7881393432617188e-07 | 902 | 0 |
| cell_count | 1 | 0 | 0.0 | 0 | 0 |
| cell_set | 1 | 0 | 0.0 | 0 | 0 |
| cells | 1448 | 0 | 1.05612263379129e-09 | 290 | 0 |
| first_rank | 842 | 0 | 0.0 | 0 | 0 |
| inputs | 5 | 0 | 0.0 | 0 | 0 |
| means | 272 | 0 | 0.0 | 0 | 0 |
| q_pid | 842 | 0 | 0.0 | 0 | 0 |
| rows | 408 | 0 | 0.0 | 0 | 0 |
| stage1 | 338 | 0 | 1.1313608938579023e-09 | 252 | 0 |
| status | 1 | 0 | 0.0 | 0 | 0 |
| verdict_set | 1 | 0 | 0.0 | 0 | 0 |
| verdicts | 16 | 0 | 0.0 | 0 | 0 |
| voided | 1 | 0 | 0.0 | 0 | 0 |

```json
{
  "closest_to_bar": {
    "bar": -0.006892851931043997,
    "distance": 0.00035226750262701365,
    "key": [
      "SC4",
      "label",
      "late",
      "teacher",
      "genre",
      "t2t"
    ],
    "lower": -0.007245119433671011
  },
  "commit": "452e143eeb4f8bace21c3b62b90d76bb50ae96d4",
  "failed_total": 0,
  "failures": [],
  "note": "numerics taken from the run's path, logic derived from the rule; no numerical diversity",
  "pass": true,
  "retired_result": [
    "/data/SSD2/cosir_r23/20261106_text_claim/c13/compare_val.json.previous.1793972961797297829"
  ],
  "sha256": {
    "stage2/cells.json": "f140acea436574399001088d01ce66ff43943e81cc09aa5d2579a77058ee4a2b",
    "stage2/outcomes.json": "9be2a191ef8d7dfad7d5943005d086fdbc31c7fb31a3e16083ee3fd8c9fedfa9"
  },
  "void_differences": [],
  "voided": []
}
```

## Full descriptive tables

The full §3.6 tables are in `OUT/stage3/part1_why_a.json` to `part8_truncation.json` (OUT = `/data/SSD2/cosir_r23/20261106_text_claim`). They are too large for this log. The summary the run wrote, `OUT/stage3/summary.md`, follows.

### Part 1: pair split, mean over seeds 0 to 4 (share / late fusion hit rate)

| set | direction | head | both | only c* | only plain | neither |
|---|---|---|---|---|---|---|
| art_val | t2i | genre | 0.1605 / 0.9999 | 0.0482 / 0.4841 | 0.1498 / 0.8147 | 0.6415 / 0.0212 |
| art_val | i2t | genre | 0.2537 / 0.9899 | 0.0572 / 0.4021 | 0.2531 / 0.8229 | 0.4360 / 0.0364 |
| art_held | t2i | genre | 0.1139 / 0.9999 | 0.0379 / 0.4407 | 0.1373 / 0.8124 | 0.7109 / 0.0169 |
| art_held | i2t | genre | 0.1777 / 0.9892 | 0.0532 / 0.3895 | 0.2360 / 0.8278 | 0.5332 / 0.0352 |
| mm_test | t2i | style | 0.0486 / 1.0000 | 0.0032 / 0.5009 | 0.8007 / 0.9681 | 0.1475 / 0.0850 |
| mm_test | i2t | emotion | 0.2832 / 1.0000 | 0.0051 / 0.4288 | 0.6175 / 0.9710 | 0.0943 / 0.0773 |

### Part 1: a std_plain / ((1 - a) std_cstar), mean over seeds

| set | direction | head | ratio | ratio within query |
|---|---|---|---|---|
| art_val | t2i | emotion | 1.9673 | 1.6511 |
| art_val | i2t | emotion | 1.9842 | 1.9017 |
| art_val | t2t | emotion | 0.9945 | 0.8877 |
| art_val | t2i | style | 2.0916 | 1.7667 |
| art_val | i2t | style | 2.0734 | 2.1276 |
| art_val | t2t | style | 2.6125 | 2.6022 |
| art_val | t2i | genre | 1.6488 | 1.4278 |
| art_val | i2t | genre | 1.6418 | 1.5954 |
| art_val | t2t | genre | 1.7114 | 1.6097 |
| art_held | t2i | emotion | 1.9911 | 1.6330 |
| art_held | i2t | emotion | 1.9765 | 1.8792 |
| art_held | t2t | emotion | 1.0066 | 0.8917 |
| art_held | t2i | style | 2.0984 | 1.7412 |
| art_held | i2t | style | 2.1074 | 2.1512 |
| art_held | t2t | style | 2.6252 | 2.5752 |
| art_held | t2i | genre | 1.6903 | 1.4258 |
| art_held | i2t | genre | 1.6551 | 1.6020 |
| art_held | t2t | genre | 1.7489 | 1.6184 |
| mm_test | t2i | emotion | 2.2908 | 2.2265 |
| mm_test | i2t | emotion | 2.3046 | 2.6032 |
| mm_test | t2t | emotion | 1.4508 | 1.5298 |
| mm_test | t2i | style | 2.1937 | 2.0977 |
| mm_test | i2t | style | 2.1883 | 2.8128 |
| mm_test | t2t | style | 3.6345 | 3.8767 |
| mm_test | t2i | genre | 1.8798 | 1.8884 |
| mm_test | i2t | genre | 1.8855 | 1.8137 |
| mm_test | t2t | genre | 1.6933 | 1.6923 |

### Part 2: pair R@10 of the a curve, mean over seeds 1 to 4

| set | direction | head | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| art_val | t2i | genre | 0.2079 | 0.2309 | 0.2535 | 0.2728 | 0.2893 | 0.3020 | 0.3121 | 0.3189 | 0.3206 | 0.3180 | 0.3102 |
| art_val | i2t | genre | 0.3110 | 0.3513 | 0.3882 | 0.4182 | 0.4436 | 0.4652 | 0.4850 | 0.4990 | 0.5084 | 0.5114 | 0.5068 |
| mm_val | i2t | emotion | 0.4342 | 0.6604 | 0.8011 | 0.8705 | 0.9070 | 0.9265 | 0.9389 | 0.9464 | 0.9484 | 0.9501 | 0.9509 |
| mm_val | t2i | style | 0.1092 | 0.3919 | 0.6408 | 0.7773 | 0.8468 | 0.8792 | 0.8980 | 0.9083 | 0.9130 | 0.9151 | 0.9142 |

self check at a = 0.7: 88 values, largest difference 0.00e+00; rankings run 968

### Part 8: truncated texts

| store | count | truncated | longest tokens |
|---|---|---|---|
| art_held/caption/emotion_focus/teacher | 61744 | 0 | 86 |
| art_held/caption/style_word/teacher | 61744 | 0 | 74 |
| art_held/caption/genre_word/teacher | 61744 | 0 | 73 |
| art_held/caption/none/roberta | 61744 | 1 | 80 |
| mm_test/narrative/default/plain | 16384 | 0 | 139 |
| mm_test/formal/default/plain | 16384 | 0 | 222 |
| mm_test/emotional/default/plain | 16384 | 0 | 132 |
| mm_test/historical/default/plain | 16384 | 0 | 159 |
| mm_test/unified/default/plain | 16384 | 0 | 270 |
| mm_val/narrative/emotion_focus/teacher | 5769 | 0 | 152 |
| mm_val/narrative/style_word/teacher | 5769 | 0 | 140 |
| mm_val/narrative/genre_word/teacher | 5769 | 0 | 139 |
| mm_val/formal/emotion_focus/teacher | 5769 | 0 | 236 |
| mm_val/formal/style_word/teacher | 5769 | 0 | 224 |
| mm_val/formal/genre_word/teacher | 5769 | 0 | 223 |
| mm_val/emotional/emotion_focus/teacher | 5769 | 0 | 146 |
| mm_val/emotional/style_word/teacher | 5769 | 0 | 134 |
| mm_val/emotional/genre_word/teacher | 5769 | 0 | 133 |
| mm_val/historical/emotion_focus/teacher | 5769 | 0 | 173 |
| mm_val/historical/style_word/teacher | 5769 | 0 | 161 |
| mm_val/historical/genre_word/teacher | 5769 | 0 | 160 |
| mm_val/unified/emotion_focus/teacher | 5769 | 0 | 287 |
| mm_val/unified/style_word/teacher | 5769 | 0 | 275 |
| mm_val/unified/genre_word/teacher | 5769 | 0 | 274 |
| mm_test/narrative/emotion_focus/teacher | 16384 | 0 | 156 |
| mm_test/narrative/style_word/teacher | 16384 | 0 | 144 |
| mm_test/narrative/genre_word/teacher | 16384 | 0 | 143 |
| mm_test/formal/emotion_focus/teacher | 16384 | 0 | 239 |
| mm_test/formal/style_word/teacher | 16384 | 0 | 227 |
| mm_test/formal/genre_word/teacher | 16384 | 0 | 226 |
| mm_test/emotional/emotion_focus/teacher | 16384 | 0 | 149 |
| mm_test/emotional/style_word/teacher | 16384 | 0 | 137 |
| mm_test/emotional/genre_word/teacher | 16384 | 0 | 136 |
| mm_test/historical/emotion_focus/teacher | 16384 | 0 | 176 |
| mm_test/historical/style_word/teacher | 16384 | 0 | 164 |
| mm_test/historical/genre_word/teacher | 16384 | 0 | 163 |
| mm_test/unified/emotion_focus/teacher | 16384 | 0 | 287 |
| mm_test/unified/style_word/teacher | 16384 | 0 | 275 |
| mm_test/unified/genre_word/teacher | 16384 | 0 | 274 |

## Storage and provenance

Outputs remain at `/data/SSD2/cosir_r23/20261106_text_claim` (79G), including `stage0`, `stage1`, `stage2`, `stage3`, logs and `c13` (59G). The MMArt store tree is 4.3G. Record files: `verdicts.json`, `c13.json`, `stage1_art.json`, `stage1_mm.json`, `status.json` in `src/test/20261106_text_claim/records/`.

Storage: left on disk and listed on .scratch/pending_deletions.md for the next storage summary; the decision is the user's at the wrap-up.

Git evidence:

```text
60b22ac2 2026-11-06 12:49:49 +0100 r23 run: the user lifts D13's 19:00 wait for this run (tcl_queue.sh switch, spec amendment)
cbe5afce 2026-11-06 13:52:36 +0100 r23 run: gate (i) repeats the store's own batches for an MMArt teacher store under the bar
452e143e 2026-11-06 14:13:19 +0100 r23 stage 1: seed ranges, read-out picks and status (before any held number)
3c435bb9 2026-11-06 14:59:23 +0100 r23 C13: re-derivation passed, 0 failures (record with the outcome hashes)
dcb43313 2026-11-06 14:59:32 +0100 r23 verdicts
```

## Source differences

The handoff estimates C13 cache size above the measured size recorded here. The stage steps record the first failed gate invocation with exit zero, while the chain records a failed gate result. C13’s independently computed closest-boundary values differ from the primary cell by floating-point amounts within its tolerance; the cell table uses the primary cell, and the C13 distance uses C13’s own values. The spec’s planned user-read date differs from the actual path. 
