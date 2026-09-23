# Review — Attention-h1 baseline provenance slides

## Local verification

- Recompiled and rebuilt the deck successfully with the named builder.
- Reopened the PPTX with `python-pptx`: 13 slides.
- Verified the insertion sequence: existing collapse slide; new fusion/CCA slide; new Attention-h1 architecture slide; existing Fine K-sweep slide.
- Verified `fusion_pareto_frontier.png` is embedded on slide 6 by SHA-256 equality of the PPTX picture payload and source asset.
- Verified the newly generated held-out AMI chart is embedded on slide 7 by SHA-256 equality.
- Ran `git diff --check`: no whitespace errors.

## External review

- Codex read-only analysis corroborated the required values, placement, chart thresholds, and media checks. Its later wrapper review was not used as a completion gate because that wrapper recursively launched nested reviewers in this environment.
- The required Claude backend was invoked for analysis and review, but its wrapper rejects `--dangerously-skip-permissions` when run as root, so it exited before producing findings.

## Findings

- Critical: none found by local validation or the completed independent analysis.
- Warning: none.
- Info: the deck uses the requested `fusion_pareto_frontier.png` only (not the optional CCA image); its slide caption explicitly states that the figure predates the Attention-h1 architecture sweep.
