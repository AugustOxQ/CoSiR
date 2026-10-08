1. **Problem Analysis**

The [requirements](/project/CoSiR/.ccg/tasks/codex-usage-script/requirements.md) define a suitably narrow utility. Before implementation, clarify the JSON output contract, supported operating systems, missing-window behavior, and whether cleanup time is included in `--timeout`.

2. **Technical Evaluation**

- **Protocol:** Wait for the matching successful `initialize` response, send `initialized`, then request limits. Match responses by ID; unrelated notifications must not count as success. This follows the [official OpenAI documentation](https://learn.chatgpt.com/docs/app-server#initialization).
- **Timeout:** Blocking `readline()` can hang on a partial line. Use one monotonic deadline across initialization and limits retrieval; notifications must not extend it. Treat premature EOF, malformed JSON, RPC errors, and broken stdin as failures.
- **Stderr:** An undrained pipe can block the child. Drain it concurrently, retaining only a bounded diagnostic tail. Avoid dumping raw protocol or potentially sensitive logs.
- **Cleanup:** Use `finally`: close stdin, allow a short graceful exit, terminate, then kill if needed, and always reap the child. Bound each wait; cover Ctrl-C and SIGTERM.
- **Windows:** Match five-hour and weekly windows using durations **300** and **10080 minutes**. Null or unexpected windows must remain unavailable or explicitly duration-labelled. The service also supports multiple limit buckets; choose the Codex bucket explicitly rather than merging quotas. Reset timestamps are Unix seconds. [Rate-limit schema](https://learn.chatgpt.com/docs/app-server#6-rate-limits-chatgpt)
- **Invocation:** Independence from the working directory requires no repository-relative reads. Bare `codex-usage` also requires installation on `PATH`. “Any agent” still requires access to the signed-in OS user’s Codex environment.

3. **Options**

- **`subprocess` plus selectors:** Small synchronous implementation; appropriate for Linux/macOS. Pipe readiness is not portable to Windows.
- **Standard-library `asyncio` subprocess:** Portable concurrent stream handling and deadline enforcement, with modest additional cancellation complexity.

4. **Recommendation**

Use selectors if the utility explicitly targets Linux/macOS; otherwise use `asyncio`. Keep one process and a small sequential handshake.

Define `--json` as exactly the successful RPC **`result` object**, preserving unknown fields, followed by a newline. Keep stdout free of diagnostics. Validate `--timeout` as finite and positive, document its default and separate cleanup allowance, and let `--help` work without launching Codex.

For human output, show used percentage, remaining percentage derived from it, and local reset time with a UTC offset. Missing data should display `unavailable`, never imply an unused quota.

5. **Action Items**

Clarify those contracts, then verify fixtures for swapped windows, null/unknown durations, multiple buckets, notifications, partial lines, EOF, RPC errors, stderr flooding, timeout, and interruption. Record the verified CLI version and confirm cleanup leaves no child process.

---
SESSION_ID: 01a11cc4-8601-7960-ac45-ca3e34491149
