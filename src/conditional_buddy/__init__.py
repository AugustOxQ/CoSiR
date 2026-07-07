"""Conditional-buddies condition initialization.

See docs/superpowers/specs/2026-06-09-conditional-buddies-init-design.md and
.claude/conditional_buddies_init.md for the design.
"""

from .compute_buddies import build_buddy_graphs, compute_buddy_init

__all__ = ["compute_buddy_init", "build_buddy_graphs"]
