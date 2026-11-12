# r28 decision lines

One line per result-shaping choice (`~/.claude/references/loop-chats.md`, "Decision lines"):
when · the choice · the main alternative · where · who · could it change a verdict or claim.

| When | Choice | Main alternative | Where | Who | Verdict? |
|---|---|---|---|---|---|
| 2026-11-12 19:40 | Encoder pick: the highest mean of the six validation cells (AMI 9/27/10 and plain-cosine mAP, seed means), as the handoff's agent default | a per-cell vote | handoff §4 item 1 | run chat (default from paper chat) | maybe |
| 2026-11-12 19:35 | Text-pass timing sample: 2,000 of 308,723 caption rows, RandomState(0) | the first 2,000 rows | r28_time.py:main | worker C | no |
| 2026-11-12 19:35 | Text-pass timing includes the tokenizer, as r19's image pass included the processor | forward pass only | r28_time.py:main | worker C | maybe (a slightly lower Qwen3 text time) |
