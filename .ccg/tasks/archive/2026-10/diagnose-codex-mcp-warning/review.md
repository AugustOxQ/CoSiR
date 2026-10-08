# Review

Removed exactly the two unsupported `type = "stdio"` lines from the user Codex configuration.
Parsed TOML comparison confirms every other setting is unchanged. A private backup was saved beside the configuration.

Verification: `codex mcp list --json` exited 0 and listed both context7 and fast-context. Neither unsupported key remains.
Low-risk, two-line configuration cleanup; self-review found no issues. No repository source files changed.
No new project specification convention needed.
