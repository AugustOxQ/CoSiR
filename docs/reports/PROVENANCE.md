# Report provenance

On 2026-09-30 the repository was consolidated. `main` became the CoSiR v2 line (formerly branch `cosir-v2`). The only other active branch is `experiment/percept_topic_pipeline`. Every other branch was archived as an annotated tag `archive/<branch>` and its branch ref was deleted. All reports from every branch were gathered into this folder.

Many reports here were written on a different branch, and the code they cite is **not on main**. v2 kept only a reusable foundation: `src/{conditional_buddy,data,dataset,eval,model,train,utils}` plus the dated `src/test/2026092[89]*` to `src/test/202610*` folders. Use this table to find the code a report refers to:

| Reports (by filename prefix) | Research line | Where the cited code lives |
|---|---|---|
| `2026-06-09` … `2026-09-03`, `2026-09-15_condition_space_audit` | Conditional buddies publication track (Exp 0–17) | tag `archive/experiment/condition_drift_retrieval_correlation`. Each earlier experiment branch also has its own `archive/<branch>` tag. |
| `2026-08-27_symmetric_conditioning_exp13` | Exp 13 symmetric conditioning | tag `archive/experiment/two_side_conditioning` |
| `2026-09-15_buddy_prototype_conditioning`, `2026-09-16_*`, `2026-09-22_*` | Exp 18 buddy-prototype conditioning and ArtELingo affect/fusion | branch `experiment/percept_topic_pipeline` (active) or tag `archive/experiment/buddy_prototype_conditioning` |
| `2026-09-23_*`, `2026-09-26_*`, `2026-09-27_*`, `2026-09-28_buddy_percept_sweep_handoff` | PercepT topic pipeline and the buddy-percept sweep | branch `experiment/percept_topic_pipeline` (active; folder `/project/CoSiR-buddy_prototype_conditioning`) |
| `2026-09-28_cosir_*` … `2026-10-12_*` | CoSiR v2 redesign | main |

Related specs and plans follow the same split. `docs/superpowers/{specs,plans}/2026-09-28-buddy-percept-sweep*` belong to the percept line. `docs/superpowers/plans/2026-08-27-symmetric-conditioning-exp13.md` and the spec's Experiment 13 section belong to the archived two_side line.

To check out an archived line: `git worktree add ../CoSiR-<name> archive/<branch>`, then `git switch -c <branch>` inside it if you want to continue it.
