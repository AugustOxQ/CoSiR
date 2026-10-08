I haven't run anything or read any files, so this is based only on your spec. The points about Codex internals (the Node shim, token rotation) come from memory and should be checked against your installed Codex version.

**Important risks, most serious first**

1. **Concurrent token refresh.** Each run starts a full app-server, which may refresh the ChatGPT token in `~/.codex/auth.json`. Several agents at once, or one beside a live Codex session, can race on that rotating refresh token and log the account out everywhere.
   *Fix:* hold a `flock` on a file under `$CODEX_HOME` for the whole run, and count the time spent waiting for it inside `--timeout`.

2. **Orphaned children.** The npm `codex` command is a Node shim that starts the native binary, so terminate or kill may only reach the shim. Also, a SIGTERM sent to `codex-usage` itself skips your `finally` block.
   *Fix:*
   - Start the child with `start_new_session=True` and signal the whole group with `os.killpg`.
   - Add a SIGTERM/SIGHUP handler that raises `SystemExit`.
   - Close stdin first, then TERM, `wait(2)`, KILL, `wait(2)`.

3. **Framing.** `os.read` returns arbitrary chunks.
   *Fix:*
   - Buffer bytes, split on `\n`, and decode only complete lines.
   - Treat `b""` as an early exit and report the child's exit code.
   - Skip non-JSON lines with a note on stderr.

4. **Response matching.** A response is a message with your `id` and **no** `method`, because a server request can reuse your id. Notifications may arrive before the `initialize` reply. A response carrying `error` must exit nonzero, not print `null`.

5. **Time zone.** "Local" means the process time zone, and this container's clock isn't guaranteed to be Amsterdam (your timestamps rule).
   *Fix:* convert with `datetime.fromtimestamp(ts).astimezone()` so `TZ` is honored, and document `TZ=Europe/Amsterdam codex-usage` for agents.

6. **Schema drift.** The app-server API is still experimental.
   *Fix:*
   - Treat a null or empty `rateLimitsByLimitId.codex` as missing before falling back.
   - Clamp remaining to 0–100 and accept a null `resetsAt`.
   - If a window's duration isn't 300 or 10080, print it with its duration rather than hiding it as "unavailable".

7. **How agents call it.**
   *Fix:*
   - Use distinct exit codes: 2 for bad arguments, 3 for spawn failure, 4 for timeout, 5 for an RPC or auth error.
   - Leave stdout empty on failure.
   - `--json` should write exactly one line to stdout. Callers must not use `2>&1`, because the inherited stderr carries app-server logs.
   - Add a `CODEX_BIN` override, since an agent's PATH may not include the npm/nvm bin.

8. **Deadline details.** `float()` accepts `nan` and `inf`, so reject them and values ≤0 explicitly. Recompute the select timeout as `max(0, deadline − now)` on every loop pass. Cleanup stays outside the deadline but is bounded by the waits in point 2.

**Not a risk:** the messages written to the child are tiny, far below the 64 KB pipe buffer, so writes outside the deadline can't block.

**Minimal set before release:** 1, 2, 4, and the exit codes from 7. The rest is about a dozen lines.
