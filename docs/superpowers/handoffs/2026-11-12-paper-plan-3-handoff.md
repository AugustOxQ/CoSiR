# Handoff: redraw Figure 2 in Figure 1's new style, then fill the tables from r28's numbers sheet as it lands

## Brief
- Done: new names across Sec. 3 and 4 and the tables; Sec. 3 rebuilt in two parts (graphs, aspect adapter) plus test time; genre term and Table 1 moved to 4.1; Figure 1 redrawn with its new caption.
- State: the method text is general (any image encoder g, captions from the MLLM); the tables still show Qwen3 numbers.
- Next: Figure 2 in Figure 1's style, then the tables and 4.2 from r28's `paper_cells.md`.
- Blocked: the tables wait for r28's numbers sheet.

> Written 2026-11-12 20:22 by the "paper plan 2" tab, cut at about 200k tokens at the user's yes. Outside the research
> loop (r28 runs in its own workspace, "CoSiR r28"). PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`;
> REPO means `PF/CVPR2027-CoSiR-Paper/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/notes/paper_plan.md`: every item with its status (updated through this chat).
3. `docs/superpowers/handoffs/2026-11-12-paper-plan-2-handoff.md`, sections 2, 3 and 5: the decisions, numbers and
   pitfalls that still stand (SigLIP 1 as main, two claims and two baselines, the test-time speed claim, r28's order).
4. `REPO/Sections/3_method.tex` (the new structure) and `PF/figures/src/fig1_overview.py` (the style Figure 2 follows).
5. `docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md`, section 4: what r28 produces and where.

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the paper plan 2 handoff stands. New on 2026-11-12 (this chat; PF README rows, plan and trace
`PF/notes/trace/3_method.md`):

- **Sec. 3 in two parts, then test time** (user): 3.1 "Aspect neighbour graphs from instructed embeddings" (the MLLM's
  instructed and plain embeddings, mutual kNN graphs, content correction, the graph-positive loss written for any
  model's aspect embeddings); 3.2 "The aspect adapter" (image tower on g_i = n(g(x_i)) of any frozen image encoder,
  caption tower on the MLLM's plain caption embedding d_i, alignment term, loss, early stopping); 3.3 test time.
  The graph part first, with the MLLM's embeddings in it (user, after seeing the recommendation).
- **4.1 paragraph "Aspect instructions and the genre term"**: Table 1 (the Qwen3 instructions) and the relational term
  (user, after seeing the recommendation). Sec. 3 has one vague template sentence on the instructions (user).
- **Figure 1**: two parts like Sec. 3, "MLLM" boxes, an "any VLM" box for images only, captions from the MLLM under
  p_0, the relational block removed (user; image-only VLM box: user, after seeing the recommendation). Its caption as
  proposed (user: yes). The left crossing (the caption line hops over the painting line) stays for now; the user
  wants it improved later with the freed space.
- **Table rows** (user): "Instructed embedding" in tables with a Backbone column (Tables 2, 3, 7, 8), "Instructed
  Qwen3" where there is none (Tables 4, 6).
- **Caption tower on Qwen3**: the user asked and was told that SigLIP 1 is only the image encoder, and texts pass once
  through Qwen3 (3.62 ms per caption). A SigLIP-only variant would be a new run. The user did not ask for it.

## 3. Where things stand

- REPO at `09c55c5` (pull first: Overleaf). PF at `6f3977e`. Glossary: main repo `e04f1bcc`.
- The method text and Figure 1 are general. Sec. 4 prose and the tables still give the Qwen3 model's numbers and
  "the instructed Qwen3 embedding" wording; they change with r28's numbers.
- The 4.1 genre sentence ("without an extra term, the genre head retrieves paintings of the same genre less well than
  the plain embedding", Tab. 5) rests on the Qwen3 ablation (0.461 against 0.476). Recheck it on SigLIP 1's Table 5.
- r28 at handoff: tier 0 partly done (Qwen3 text pass 3.62 ms per caption, `records/time_text.json`; MMArt records
  started); no `paper_cells.md` yet.

## 4. Open points for the next step

Key:
1. **Figure 2** (`PF/figures/src/fig2_graphs.py`): redraw in Figure 1's style (its caption has a TODO "this borrows
   from earlier figures, need to redraw"). Ask the user what Figure 2 must show before the redraw. Look work goes to an
   Opus `high` worker; check the 300 dpi render, send the PNG, copy on the user's yes.
2. **Tables 2 to 8 and 4.2** from r28's numbers sheet, as it lands: SigLIP 1 rows as CoSiR, comparators matched on
   SigLIP 1, the two claims; the PercepT sentence and SigLIP 1's own style strength (the user: "we discuss this
   later") come up at the 4.2 rewrite. Splitting tables by passes: a TODO, decided later.
3. **Figures 3 and 4**: the user picks from r28's review page.

Minor:
4. Figure 1's left crossing: improve later (user).
5. The introduction's `\enumerate{...}` never closes; the comment beside `\input{Tables/cost}` in 4.5 ("goes to
   supplemental material") is wrong now. Both are the user's lines: mention, do not edit.
6. Sec. 3 keeps emotion's caption graph and the per-aspect correction weights (the user moved only the genre term).
   Raised once, not answered; do not move them unless the user asks.

## 5. Code and pitfalls

- All pitfalls of the earlier paper handoffs stand: pull before editing (Overleaf), tell the user after each push that
  Overleaf needs a pull; stage only our lines; no notes of ours in REPO; paper names only; no semicolons between
  clauses; "test set", never "held-out". Update `PF/notes/trace/<section>.md` and the plan with each change.
- Symbols now: v_i and d_i are the MLLM's plain embeddings, g_i the image encoder's, t^a_i and e_i the instructed
  ones; L_align (`eq:alignloss`), L_graph (`eq:graphloss`), L_rel (`eq:relloss`, in 4.1); labels `sec:graphs`,
  `sec:adapter`, `sec:test`.
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO; build files are gitignored.
- Deadlines: paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 6. State at handoff

- Running: in this chat nothing. Elsewhere: r28 run and r28 main (workspace "CoSiR r28"), r21 run, maybe "paper
  experiments 2".
- Uncommitted: in REPO and PF nothing of ours (PF has the user's `CBMI_IMP_CAMREADY (2).pdf`, left out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 23 MB (renders, the old Figure 1 script, crops); it goes
  with the session.
