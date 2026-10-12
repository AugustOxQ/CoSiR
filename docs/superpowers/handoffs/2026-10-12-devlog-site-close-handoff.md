# Handoff: the dev-log site is live; publish each new stage page with `devlog-publish`, on the user's word

## Brief
- Done: PIN-gated dev-log site (Cloudflare Pages) built, reviewed and published with the 2 to 12 Oct stage page; tool `devlog-publish`.
- State: live; every page needs the PIN; personal-information check clean (19 files); fail-open off.
- Next: nothing in this line; a new stage page goes out by `devlog-publish add`, a preview, and `deploy --yes` on the user's word.
- Blocked: nothing. The user rolls the API token (it was pasted into this chat) and shares the PIN with readers.

> Written 2026-10-12 22:00 by the dev-log site chat. Process work beside round 8; not a research-loop step. Replaces
> `2026-10-12-devlog-site-handoff.md`, whose Cloudflare Access plan the user changed to a shared PIN.

## 1. Read in this order

1. `~/.claude/references/stage-page.md`, "Publishing for outside readers": the habit, the tool, credentials, the checks.
2. `/project/cosir-devlog/README.md` (private site repo; the site's address and `deploys.log` are there, never in this
   public repo).
3. `devlog-publish --help` and its row in `/project/tools/README.md`.

## 2. Decided by the user, not to be reopened

- A shared PIN instead of Cloudflare Access with an email list: "this is not top secret" (user, 2026-10-12).
- A basic gate is enough; no personal information on the site before any publish (user, 2026-10-12).
- Credentials in `/project/.secrets/cloudflare.env`, outside every git repo (user, after seeing the recommendation).
- Site name `cosir-devlog` (user, after seeing the recommendation).
- The published copy drops html-plan's comment box for "reply in the Slack thread" (user, after seeing the recommendation).
- A private GitHub backup of the site repo (user, after seeing the recommendation).
- Gated dev-log pages become a global habit (user, after seeing the recommendation); stage-page prose budget about
  1,800 words instead of 1,500, because outside readers need more context (user).
- Publish the 2 to 12 Oct page (user, "yeah publish", 2026-10-12 21:52). The PIN is a random 6-digit code the user
  asked Claude to pick; it is in the credentials file.

## 3. Where things stand

- **Live:** index, "Start here" (from the primer) and one update. Checked from outside: without the PIN every page,
  image and the per-deploy address give 401; only `robots.txt` (disallow all) is open; a wrong PIN gives 401; the right
  PIN gives the page byte for byte as staged, with `X-Robots-Tag: noindex`.
- **The gate** (`functions/_middleware.js`): fails closed without the PIN or the cookie key; cookie = issue time plus
  an HMAC under a server-only key, 30 days; same-origin redirects only; 1 s per wrong PIN. 17 node tests.
- **The tool** (`/project/tools/bin/devlog-publish`, 43 tests): `add`, `start`, `check`, `serve`, `deploy`. `check`
  and every `deploy` scan every file (bytes, entities, percent-encoding, file names) for the forbid list and personal
  information, and stop on a hit. `deploy` forces Pages fail-open off and commits only its own paths.
- **Reviews:** an Opus final review found a cookie that could be guessed offline (a 4-digit PIN in 1.7 s), an open
  redirect, a robots.txt fallback, a possible `.env` commit and scan gaps; all fixed in one wave. A Sonnet re-review
  confirmed each by a run; the controller fixed its two small points (a failed fail-open read now stops `deploy`; the
  commit takes only the tool's paths).
- **Accepted, not fixed:** parallel PIN guessing stays possible (6 digits; the free daily request limit runs out
  first); the forbid word `wding` also matches "crowding" (the deploy stops and names the file).

## 4. Open points for the next step

- Minor, the user: roll the Cloudflare token (API Tokens page, the token's menu, Roll) and paste the new value into the
  credentials file in Cursor.
- Minor, the user: how readers get the PIN (pinned message or direct message).
- The Slack post with the link is `docs/reports/stage/2026-10-12_affect_steering_line_devlog.md` plus the address from
  `deploys.log`; the saved copy keeps a placeholder, because this repo is public.

## 5. Code and pitfalls

- Never write the site's address or the PIN into this public repo; the post template says so.
- Codex jobs: the auto-mode classifier refused a `codex e ...` call that began with `cd`; start the command with
  `codex e --dangerously-bypass-approvals-and-sandbox` (the user's allow rule), or a Claude worker takes the job.
- `/project/tools` main moves under other sessions: build in a worktree, rebase, fast-forward. `test_loop_watch`
  fails on this machine on main as well (38 failures; it calls the live `herdr`).

## 6. State at handoff

- Running: nothing (the local preview on port 8788 is stopped).
- Uncommitted: nothing.
- On disk over 1 GB: nothing from this chat (site repo 205 MB, of which `node_modules` 199 MB, kept for deploys).
