# Handoff: r9 build 2: write the stage tickets from the committed rule, then build them

## Brief
- Done: decision rule committed (a443902) after a fresh Opus review and a re-review; SigLIP 2 features of ArtELingo and SemArt; NLLB translation of ArtELingo-28.
- State: ArtELingo-28 episodes per language (option B), agent default after no answer in 30 minutes; nodes idle.
- Next: stage tickets 04 onward (to-tickets), then the build; 0a and 0b gate every use of the chain.
- Blocked: nothing.

> Written 2026-10-13 16:27 by `r9 build` (cut at the context meter). Loop step reached: step 7, the rule committed,
> preparation tickets 01 to 03 merged, stage tickets not yet written. Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. The rule, `src/test/20261202_r9_scope/DECISION_RULE.md` @ a443902 (458 lines; binding where it differs from the
   spec). Its review and fix record: `rule_check/rule_review.md`.
2. The spec from §1: `docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md` (§3.5 stages, §4 decisions).
3. The round log `src/test/20261202_r9_scope/20261202_r9_scope_log.md` (what was done, agent defaults, job tags).
4. The code map `.scratch/r9-scope/notes/aff_chain_map.md` (local): how each AFF part is built today, its CLIP
   assumptions and pins, and the seams for a backbone switch. Tickets index `.scratch/r9-scope/issues/00-index.md`.
5. `~/.claude/references/matt-chain.md` (to-tickets and implement-spec with our changes), `codex-jobs.md`, the
   `cluster-run` skill.

## 2. Decided by the user, not to be reopened

All of spec §4 (D1 to D34). The previous handoff (`2026-10-13-r9-build-handoff.md`) summarizes them.

## 3. Where things stand

- **Rule** a443902: written from a first writer's research note, reviewed fresh on Opus (20 findings, every verdict
  number re-derived equal), all fixed, scoped re-review on Sonnet (one new major, fixed). Integer Holm bounds, the
  two-level test, stage 0a/0b gates and tolerance, pins (`pins_<backbone>.json`, `pins_clip.json`, `pins_a28.json`)
  and provenance checks are in it.
- **Agent defaults taken in this chat** (also in the log): option B for ArtELingo-28 episodes (asked 15:29, no answer
  by 15:59; the user may switch to A by amendment before H8's ledger row; rule §5.1 comment lists the places that
  change); the translation covers all 167,230 non-English rows (only read rows are featurized); heads are fitted and
  pinned locally at 8 threads (narrows D33's "two re-fits" on the nodes; the rest of a re-fit may run on DAS6);
  `other` emotion → −1; IsiNdebele and Emakhuwa translated through proxy codes and flagged, kept in the pooled read.
- **Stores ready:** `/data/SSD2/pre_extract/artelingo_full/siglip2_so400m` (61,402 images, 308,723 captions; row order
  equal to the Qwen store), `/data/SSD2/pre_extract/semart/siglip2_so400m` (21,382), Qwen `artelingo_full` as before;
  translations `/data/SSD2/pre_extract/artelingo28/nllb200_3.3b_en/`. ArtELingo-28 feature stores per backbone: not
  made yet (rule §3.3 fixes the read set and rows; the extraction script needs an `artelingo28` adapter).
- **DAS6:** node406 and node408 idle, reservations to about 2026-10-18 14:40. `/var/scratch/wding` now holds the
  SigLIP 2, NLLB and Qwen-Embedding weights, ArtELingo annotations and row order, the ArtELingo-28 CSV and images,
  SemArt. DATA_MAP gained SemArt and CUB (D24). Job results also sit on the nodes under `/local/wding/res/r9`
  (duplicates after the pulls; cleaning needs the user's yes before release).
- **Code merged on `r9-scope`:** `scripts/extract_features.py` (+ `siglip2_so400m`, node path options),
  `scripts/translate_artelingo28.py`, `scripts/run_r9_extract.sh`, `scripts/run_r9_translate.sh`,
  `scripts/das6_sync_r9.py`.

## 4. Open points for the next step

- Ticket split (agent's call; a suggestion only): a backbone loader that returns the `load_artelingo()` shape with
  CLIP's split passed in (verdict); stage 0a on the generic chain; the recipe runner split into small verdict tickets
  (groupings and banks; heads; A3; reader and τ; baselines, picks and cells; pins and provenance); stage 0b; the
  ArtELingo-28 read set, labels, flags, value sets and stores (verdict) plus its extraction adapter (plumbing); the
  per-language episode builder (verdict); the held runners H6/H7 and H8–H10 with the ledger-first guards; ledger
  amendments (backbone column D17, the 81 seeds); the C13 re-derivation (Codex, separate code); the stage 5 probes.
- The coverage line of hard rule 6 at the end of the ticket list (each SC and D ID with its tickets).

## 5. Code and pitfalls

- The `cluster` on PATH is a graph tool; the cluster CLI is `~/.claude/skills/cluster-run/cluster`. `bash scripts/run_*.sh`
  jobs have unknown results folders, so a pull copies all of `/local/wding/res` on that node (fine while only r9 writes
  there). Run `das6_sync_r9.py` with `/usr/bin/python3`; its `--make-sample-ids` step with the CoSiR env.
- Never amend or force-push a pushed commit (an amend was refused in this chat; fixed with a new commit).
- Codex: three plumbing tickets used 8% of the five-hour window together (ledger rows added).
- Nothing is deleted; `.scratch/pending_deletions.md` has the pulled store copies (`res/r9`, about 1 GB) and the
  ticket worktrees `/project/CoSiR-wt/r9-t01..03` with their branches.

## 6. State at handoff

- Running: nothing (no subagents, no cluster jobs).
- Uncommitted: nothing on `r9-scope` except the untracked round 8 results trees, as before.
- On disk over 1 GB: nothing new on `/data` beyond the stores above (SigLIP 2 ArtELingo 0.9 GB); `res/r9` 0.96 GB on
  the system disk (pending deletion).
