---
_harness_template: "Plans.md.template"
_harness_version: "4.3.3"
---

# Plans.md - Task Tracking

> **Project**: CoSiR
> **Last updated**: 2026-08-26
> **Updated by**: Claude Code

---

## In Progress

<!-- Add tasks with cc:wip here. -->

(none)

---

## Not Started

<!-- Add tasks with cc:todo or pm:requested here. -->

(none)

---

## Completed

<!-- Add tasks with cc:done or pm:approved here. -->

(none)

---

## Archive

<!-- Move older completed tasks here. -->

---

## Status Marker Legend

These markers are protocol values used by Harness tooling. Keep them unchanged
unless the project has tested parser aliases.

| Marker | Meaning |
|--------|---------|
| `pm:requested` | PM requested work |
| `cc:todo` | Not started by Claude Code |
| `cc:wip` | Claude Code is working |
| `cc:done` | Claude Code completed the task and is awaiting confirmation |
| `pm:approved` | PM confirmed completion |
| `pm:依頼中` | Compatibility alias for `pm:requested` |
| `cc:TODO` | Compatibility alias for `cc:todo` |
| `cc:WIP` | Compatibility alias for `cc:wip` |
| `pm:確認済` | Compatibility alias for `pm:approved` |
| `cursor:依頼中` | Compatibility alias for `pm:requested` |
| `cursor:確認済` | Compatibility alias for `pm:approved` |
| `blocked` | Blocked; include the reason next to the task |

---

## Optional Extended Syntax

For larger plans, you may add task IDs, dependencies, and parallel markers.

### Task ID / Dependency / Parallel Marker

```markdown
- [ ] T001: Authentication `cc:todo`
- [ ] T002: User API `cc:todo` depends:T001
- [ ] T003: Product API `cc:todo` [P]
- [ ] T004: Order API `cc:todo` depends:T001,T003
```

| Syntax | Meaning | Example |
|--------|---------|---------|
| `T001:` | Optional task ID | Used for references and dependencies |
| `depends:ID` | Dependency task | `depends:T001,T002` |
| `[P]` | Parallelizable | Can run at the same time as other ready tasks |

**Note**: Extended syntax is optional. The plain checklist format still works.

---

## Last Update

- **Updated at**: 2026-08-26
- **Last session owner**: Claude Code
- **Branch**: experiment/condition_drift_retrieval_correlation
