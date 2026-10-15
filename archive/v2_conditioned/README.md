# CoSiR v2, the conditioned line (archived 2026-10-15)

CoSiR v2 tested conditioned image-text similarity on frozen CLIP features: aspect episodes on ArtELingo, where example pairs show the aspect (e.g. affect) and a reader picks it. Its best method, AFF (affect steering), passed a held read over the condition-free bar B′(A0) on CLIP (+0.60 R@1, round 6), Qwen3-VL-Embedding-2B (+0.63) and SigLIP 2 (+0.51) (round 9), but was never shown above the strongest condition-free scorer B′(A1), and lost on style × genre. The supervisors stopped the line on 2026-10-15: too specific and complicated for CVPR; the project went back to a condition-free model as an improvement over PercepT.

## How to restore

- Code as it was: `git worktree add ../CoSiR-v2 archive/v2-conditioned` (annotated tag, commit 05e0efb, the whole tracked state before the archive). The gitignored results (features, checkpoints, caches, cluster pulls) exist only in this folder on the workstation disk, not in git.
- Archived round code cannot run in place: 141 files use `Path(__file__).parents[n]` and 29 hard-code `/project/CoSiR/src/test/...`. Their `from src...` imports need the live code root on `PYTHONPATH`, and the paths need fixing. The tag worktree runs as it was.
- The ArtELingo-28 read set is no longer only in `src/test/20261202_r9_scope/results/`: a copy sits at `/data/SSD2/pre_extract/artelingo28/read_set_a28.parquet`, and `scripts/extract_features.py` reads it from there.

## What is where

This folder mirrors the repo layout: a file that was at `X` is at `archive/v2_conditioned/X`. Tracked files are in git; items marked "disk only" are gitignored or untracked.

| Folder | Contents |
|---|---|
| [docs/reports/reports_sum.md](docs/reports/reports_sum.md) | index of the 52 `auto/v2` reports and the 10 v2 stage reports |
| [docs/reports/auto/v2/](docs/reports/auto/v2/) | 52 automatic reports, 2026-09-28 to 2026-12-04 (candidate A, aspect task, rounds 1 to 9, Leiden, k-shot) |
| [docs/reports/stage/](docs/reports/stage/) | 10 v2 stage and outside-view files (methods report, CVPR readiness, affect-steering line, primer, model card, scope limits) |
| [docs/reports/weekly/drafts/](docs/reports/weekly/drafts/) | 8 weekly drafts of 1 to 7 Oct and 13 to 14 Oct (the finals stayed live in `docs/reports/weekly/`) |
| [docs/reports/assets/](docs/reports/assets/) | figures, figure data and build scripts of the v2 reports |
| [docs/user_read/](docs/user_read/) | user-read reports of the affect-steering line, with figures |
| [docs/superpowers/specs/](docs/superpowers/specs/), [plans/](docs/superpowers/plans/), [handoffs/](docs/superpowers/handoffs/) | v2 designs (19), plans (21) and loop handoffs and start prompts (79) |
| [docs/superpowers/constitution.md](docs/superpowers/constitution.md) | the full v2 constitution C1 to C14 with its amendments |
| [docs/superpowers/direction_log.md](docs/superpowers/direction_log.md), [direction_log_raw_2026-10.md](docs/superpowers/direction_log_raw_2026-10.md) | the conditioned line's decision log and the verbatim round notes of 2 to 7 Oct |
| [GLOSSARY.md](GLOSSARY.md) | terms of the conditioned line (aspect, reader, AFF, B′ scorers, held read, round 9 terms) |
| [src/test/](src/test/) | 63 dated round folders (code, rules, logs, final reviews; results are disk only) and `test_method_repair_decide.py` |
| [scripts/](scripts/) | 19 launchers of v2 rounds (cluster sync, r6 and r9 runs, MLLM probe, ArtELingo-28 translation) |
| `res/cluster_jobs/` | 108 pulled cluster job folders (disk only, 3 GB); `res/r9/a28_translate/` holds the ArtELingo-28 translation shards (disk only) |
| `.scratch/`, `.superpowers/` | scratch folders of rounds 6 to 9 and archived build folders (disk only) |

## Where to find

| Question | File |
|---|---|
| Final round 9 results: AFF on CLIP, Qwen3-VL and SigLIP 2, and ArtELingo-28 | [docs/reports/auto/v2/2026-12-02_aff_scope.md](docs/reports/auto/v2/2026-12-02_aff_scope.md) |
| Round 9 detail tables (held reads H6 to H10) | [src/test/20261202_r9_scope/r9_report_tables.md](src/test/20261202_r9_scope/r9_report_tables.md) |
| Round 9 for the user (plain English) | [docs/user_read/2026-10-14_aff_scope.md](docs/user_read/2026-10-14_aff_scope.md) |
| Round 6 held result on CLIP (+0.60 R@1 over B′(A0)) | [docs/reports/auto/v2/2026-11-25_artelingo_held_test.md](docs/reports/auto/v2/2026-11-25_artelingo_held_test.md) |
| Round 6 held read, user version | [docs/user_read/2026-10-10_artelingo_held_test.md](docs/user_read/2026-10-10_artelingo_held_test.md) |
| Leiden against k-means groupings | [docs/reports/auto/v2/2026-12-04_leiden_groupings.md](docs/reports/auto/v2/2026-12-04_leiden_groupings.md) |
| Leiden groupings, user version | [docs/user_read/2026-10-14_leiden_groupings.md](docs/user_read/2026-10-14_leiden_groupings.md) |
| Number of example pairs (k-shot) check | [docs/reports/auto/v2/2026-12-03_kshot_supports.md](docs/reports/auto/v2/2026-12-03_kshot_supports.md) |
| The reader (round 1 report; its rule and code) | [docs/reports/auto/v2/2026-11-18_reader_fix_csd.md](docs/reports/auto/v2/2026-11-18_reader_fix_csd.md) |
| B′(A0), B′(A1), B, AFF, held read: definitions | [GLOSSARY.md](GLOSSARY.md) |
| Held ledger rows | [docs/superpowers/held_ledger.md](../../docs/superpowers/held_ledger.md) (live) |
| Round 1 (reader fix with CSD): report and folder | [report](docs/reports/auto/v2/2026-11-18_reader_fix_csd.md), [folder](src/test/20261117_reader_fix_csd/) |
| Round 2 (reader fix round 2) | [report](docs/reports/auto/v2/2026-11-19_reader_fix_round2.md), [folder](src/test/20261118_reader_fix_round2/) |
| Round 3 (affect gate, AFF's frozen code) | [report](docs/reports/auto/v2/2026-11-21_round3_affect_gate.md), [folder](src/test/20261121_round3_affect_gate/) |
| Round 4 (AFF vetoes) | [report](docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md), [folder](src/test/20261122_round4_aff_vetoes/) |
| Round 5 (GoEmotions placement, idea 3) | [report](docs/reports/auto/v2/2026-11-23_idea3_goemotions.md), [folder](src/test/20261123_idea3_goemotions/) |
| Round 6 (ArtELingo held test) | [report](docs/reports/auto/v2/2026-11-25_artelingo_held_test.md), [folder](src/test/20261125_artelingo_held_test/) |
| Round 7 (design L, style source) | [report](docs/reports/auto/v2/2026-11-26_design_l_style_source.md), [folder](src/test/20261126_design_l_style_source/) |
| Round 8 (learned grouping) | [report](docs/reports/auto/v2/2026-11-29_learned_grouping.md), [folder](src/test/20261129_r8_grouping/) |
| Round 9 (AFF's scope) | [report](docs/reports/auto/v2/2026-12-02_aff_scope.md), [folder](src/test/20261202_r9_scope/) |
| ARS review of the CVPR plan (paper plan) | [docs/reports/auto/v2/2026-10-27_ars_plan_review.md](docs/reports/auto/v2/2026-10-27_ars_plan_review.md) |
| ARS citation check of the plan | [docs/reports/auto/v2/2026-10-28_citation_check.md](docs/reports/auto/v2/2026-10-28_citation_check.md) |
| The v2 constitution | [docs/superpowers/constitution.md](docs/superpowers/constitution.md) |
| The direction log (decisions, parked ideas) | [docs/superpowers/direction_log.md](docs/superpowers/direction_log.md) |
| The CVPR publication plan (design) | [docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md](docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md) |
| Cluster job pulls (r6, CLIP fine-tuning, r9 translate and extract, MLLM probe) | `res/cluster_jobs/` (disk only); the sync scripts are in [scripts/](scripts/) (`das6_sync_*.py`) |
| The full list of v2 reports | [docs/reports/reports_sum.md](docs/reports/reports_sum.md) |
| Loop chat table and briefs of all loops | [chats.tsv](../../docs/superpowers/handoffs/chats.tsv) and [briefs.md](../../docs/superpowers/handoffs/briefs.md) (live, whole) |
