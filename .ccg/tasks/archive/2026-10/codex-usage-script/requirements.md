# Shared Codex usage script

- Add a standalone executable `bin/codex-usage`, runnable from any working directory by any agent.
- Depend only on Python 3 and an installed Codex CLI already signed in to ChatGPT.
- Use the verified app-server initialize/initialized/account/rateLimits/read protocol.
- Default output shows five-hour and weekly remaining and used percentages plus local reset times.
- Identify windows by their returned duration, not by hardcoded primary/secondary assumptions.
- `--json` prints the service result for agents; `--help` describes dependencies and invocation.
- Targets Linux and macOS shells. JSON preserves the complete RPC result, including multiple buckets.
- The default timeout is 30 seconds for initialization and the read combined; cleanup may take up to two additional seconds.
- `--timeout` bounds the RPC interaction; all failures have nonzero exit codes and stderr diagnostics.
- Clean up the child process on success, timeout, protocol errors, or interruption.
- Do not start threads or model turns, expose authentication tokens, or change authentication/configuration.
- Keep verification fixtures and reports in this task directory; do not modify unrelated files.
