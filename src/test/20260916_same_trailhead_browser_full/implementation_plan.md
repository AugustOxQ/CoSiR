# Same Trailhead Full Browser Implementation Plan

**Goal:** Build a local live browser for every RedCaps-150k hub-anchored C/D pair.

**Architecture:** A one-time builder reconstructs the established buddy graph and writes parallel NumPy arrays.  A FastAPI process memory-maps the arrays, resolves annotations in feature-row order, and creates thumbnails only for requested A/B/C/D rows.  The browser fetches one selected row at a time.

**Constraints:** All created files live in this dated directory; graph semantics come only from the existing conditional-buddy helpers; individual-pair pull is never calculated or displayed.

## Tasks

- [ ] Write and run failing unit tests for deterministic cache metadata, bucket validation, edge label classification, request bounds, and subreddit extraction.
- [ ] Implement `build_full_index.py` with the existing feature loader, graph helpers, full enumeration, deterministic B choice, full C/D image distances, atomic cache writes, and population sanity check.
- [ ] Implement `server.py` with startup cache/annotation loading, bucket row offsets, live thumbnails, and validated endpoints.
- [ ] Copy/adapt the browser visual design into `browser_full.html`, replacing static data with API navigation and contextual aggregate pull values.
- [ ] Add `serve.sh`, run syntax/unit checks, create the full cache, exercise endpoints, review the diff, and record build/API evidence.
