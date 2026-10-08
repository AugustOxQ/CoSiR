# Consolidated review and verification

Codex and Claude both analyzed and reviewed this standalone utility. Neither review found a Critical issue.

The configured Claude wrapper forced permission bypass and failed under root. Claude analysis and review succeeded through its native read-only plan mode with tools disabled; no permission bypass was used.

## Resolved findings

- Reproduced interruption during cleanup using a real fake-server process. The regression failed with exit 130 before the fix. Cleanup now ignores further termination signals until process-group signaling, reaping, and pipe closure finish.
- Signal the child process group, including launchers' native descendants, and escalate even when the leader has already exited.
- Identify five-hour and weekly windows by their returned durations, rather than primary/secondary position.
- Added coverage for a stubborn server requiring SIGKILL, multiple JSON buckets, unknown JSON fields, missing windows, invalid reset timestamps, premature EOF, malformed JSON, auth/RPC errors, missing CLI, invalid timeout arguments, and termination.
- Verification discovers the repository through its parent directories, so archiving and another checkout do not break the script path.
- Human output is assembled before printing; unusable reset timestamps show unavailable without hiding known usage.
- Executable invocation, help, and live JSON were checked from /tmp.

## Final evidence

- 11 black-box tests passed in 2.900 seconds after the final edits.
- Live `codex-usage` returned 80% remaining for five hours and 81% for the week; percentages are snapshots.
- Live `codex-usage --json` parsed successfully and contained the service rateLimits object.
- Python syntax parsed successfully.
- Source: bin/codex-usage, executable and standalone with Python's standard library.
- Installed command: /root/.local/bin/codex-usage, symlinked to the repository script.
- Existing local Codex config warnings remain on stderr; JSON stdout stays valid.
- Tested on Linux with codex-cli 0.161.0. macOS has not been tested locally.

Spec feedback: no existing project spec applies to this standalone external-tool helper. The process cleanup and framing findings are recorded here for future work; no application conventions were changed.
