# Codex usage from a shell

Verified with the installed codex-cli 0.161.0.

- No dedicated usage subcommand appears in CLI help.
- The documented `account/rateLimits/read` app-server method returns ChatGPT account limits.
- A Python subprocess client can initialize, acknowledge initialization, read limits, and terminate without starting a thread or model turn.
- The live read-only query exited successfully and returned primary duration 300 minutes with 10% used, and secondary duration 10080 minutes with 17% used.
- `usedPercent` measures usage, so remaining percentage is `100 - usedPercent`.
- `resetsAt` is a Unix timestamp in seconds.
- Source: https://learn.chatgpt.com/docs/app-server
- No project code changed; no project tests are applicable. Verification was the live RPC query.
- No project specification exists and no project-specific convention needs recording.
