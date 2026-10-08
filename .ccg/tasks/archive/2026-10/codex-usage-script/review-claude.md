## Review: `bin/codex-usage` and its black-box checks

The main path is correct. The 30 s deadline covers spawn, `initialize` and the read together. Notifications are skipped, and so are server requests that reuse a matching id. Windows are matched by `windowDurationMins`, `--json` dumps the whole result, and failures go to stderr with a nonzero exit. I found no verified Critical defect. "Verified" below means confirmed by tracing the code; I ran nothing.

### Critical
- None verified.

### Warning
- **[unverified, macOS] Cleanup can crash on an exited child (the `os.killpg` calls in `finally`).** Only `ProcessLookupError` is caught. On macOS, signalling a group whose only member is an unreaped zombie may return EPERM. In the `eof` path that would raise `PermissionError` out of `finally`. The error message would change from "closed before replying" to "Operation not permitted", and `wait()` would be skipped.
  - Fix: catch `OSError` around both `killpg` calls, or skip SIGTERM when `process.poll()` is not None. Always reach `wait()`.
  - Run the suite once on macOS.
- **[verified, test gap] The SIGKILL path is never tested.** No fake ignores SIGTERM, so the `TimeoutExpired → SIGKILL` branch and the "at most 2 s extra" promise are untested. Add a mode with `signal.signal(SIGTERM, SIG_IGN)` and assert the child is reaped within timeout + 2 s + slack.
- **[verified, test gap] "Multiple buckets" is untested.** The fixture has one bucket, and the JSON test checks only two fields. Add a second `rateLimitsByLimitId` entry and assert `json.loads(stdout) == fixture_result`.
- **[verified, test gap] Running the file directly is untested.** Every check runs `sys.executable SCRIPT`, so the shebang, the exec bit and running from another working directory are not covered. With `PATH=<tmp>`, `/usr/bin/env python3` cannot find Python, so that test must keep Python's directory on PATH.

### Info
- **[verified] Only the group leader is waited on.** The group gets SIGKILL only if the leader outlives 2 s. A group member that ignores SIGTERM survives once the leader exits. The npm `codex` wrapper (unverified) forwards signals and waits for its native child, so the practical risk looks low.
  - Optional: escalate the group before reaping the leader, using `os.waitid(..., WEXITED | WNOWAIT)`.
  - Add a fake that spawns a grandchild.
- **[verified] A second signal can cut cleanup short.** If SIGINT or SIGTERM arrives during `process.wait(timeout=2)`, it raises out of `finally` and skips SIGKILL. Block those signals during cleanup with `signal.pthread_sigmask`.
- **[verified] `print_summary` does not check types.** A non-dict bucket or a string `resetsAt` ends in an uncaught traceback. An out-of-range weekly `resetsAt` fails after the 5h line is already on stdout. Build all lines first, then print; catch `TypeError` and `AttributeError`.
- **[verified] An early child exit can show up as "[Errno 32] Broken pipe" from `send()`.** Map `BrokenPipeError` to "app-server exited early".
- **[verified, Linux] Huge timeouts pass validation.** A `--timeout` above about 24.8 days is accepted, then epoll raises `OverflowError` after the child has started. Cap the value in `positive_timeout`.
- **[unverified] The real server may write to stderr.** The child's stderr is inherited, so the success test's `stderr == ''` only shows that the fake is quiet. Confirm with the live check.
- **[verified] Some assertions are weak.** `assertIn('reset')` never checks the time itself: set `TZ=UTC` and assert the exact ISO string. `SCRIPT` is also a hard-coded absolute path.
- **Integration note:** in this container, "local" time may be UTC, and the output includes an offset. Agents reporting to the user should run it with `TZ=Europe/Amsterdam`; one line in `--help` would cover it.

### Summary
Approve with changes. Make cleanup safe on macOS and close the four test gaps. The rest is optional hardening.
