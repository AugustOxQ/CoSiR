# Handoff: set up a gated dev-log site for outside readers, then publish the 2 to 12 October stage page on the user's word

## Brief
- Done: stage report 2 to 12 Oct for outside readers (markdown, HTML page, 10 figures, Slack post and primer), checked and pushed.
- State: AFF +0.60 [+0.46, +0.73] R@1 over B′(A0) on held paintings; the page is in `docs/reports/stage/`.
- Next: build a gated static site (Cloudflare Pages + Access) and a `devlog-publish` tool; publish only when the user says so.
- Blocked: the user makes a Cloudflare account and API token, and names the readers.

> Written 2026-10-12 20:58 by the stage-report chat. Process work beside round 8; not a research-loop step.

## 1. Read in this order

1. This handoff.
2. The page to publish first: `docs/reports/stage/2026-10-12_affect_steering_line.html` (source `.scratch/stage_pages/2026-10-12_affect_steering_line.src.html`), and its report `docs/reports/stage/2026-10-12_affect_steering_line.md`.
3. The Slack texts: `docs/reports/stage/2026-10-12_affect_steering_line_devlog.md` (post #1) and `docs/reports/stage/2026-10-12_devlog_primer.md` (pinned primer).
4. `~/.claude/references/stage-page.md` (how pages are built and packed) and `~/.claude/rules/container.md` ("Showing the user an HTML page"; secrets and what persists).
5. Before writing any rule or reference: `~/.claude/references/maintaining-rules.md`.

## 2. Decided by the user, not to be reopened

- Outside readers get each update as a Slack message with a link to an HTML page; the page is a detailed dev log, so they can follow a week of agent work before the one-hour weekly meeting (user, 2026-10-12).
- Not a Claude artifact (user, 2026-10-12).
- Gated access: Cloudflare Pages with Cloudflare Access (email one-time PIN), not a public or unlisted page (user, after seeing the recommendation, "go", 2026-10-12 20:57).
- Publish only on the user's word, page by page (user, after seeing the recommendation, "go", 2026-10-12 20:57).
- Figures for this stage report were made by Claude, not Codex; this held only for this report (user, 2026-10-12).

## 3. Where things stand

- **The page links images by relative paths** (`../assets/2026-10-12_affect_steering_line/`, `../assets/2026-10-07_weekly/`, `../../user_read/assets/2026-10-12_design_l_style_source/`). A copy with every image inlined, one self-contained file of 6.3 MB, is at `.scratch/stage_pages/2026-10-12_affect_steering_line.html` (gitignored). Publishing either uses such an inlined copy or copies the asset folders beside the page.
- **Double-blind (CVPR, abstract 2026-11-10):** the CoSiR GitHub repo is public. The site must not link it or be linked from it. Use a separate private repo (for example `cosir-devlog`), `noindex` on every page and a `robots.txt` that disallows all.
- **Proposed site layout** (agent default, open to the user): an index (updates newest first: date, one-line outcome, headline figure, link); a fixed "Start here" page from the primer; one page per stage report.
- **Proposed tool** (agent default): `devlog-publish` in `/project/tools` (git, `install.sh`), around `wrangler pages deploy <site> --project-name <name>`. Node is installed (the html-plan pack ran on it).
- **Slack:** a gated page has no link preview, so each post attaches its headline figure as an image.
- **html-plan's comment box:** outside readers cannot send its Respond sheet to Claude. Either keep it (it copies text they can paste into the Slack thread) or replace it with a line "reply in the Slack thread".

## 4. Open points for the next step

- **Key, the user:** who reads (a list of emails, or a whole domain such as the university's); this sets the Access policy.
- **Key, the user:** create the free Cloudflare account and an API token with Pages edit rights, and store it in the container yourself, never in a chat or in git. Decide with the user where it lives so it survives a container rebuild (a file under `/project` outside any git repo, or re-entered after a rebuild).
- **Minor:** the site's name and URL; the index layout above.
- **Minor:** whether "outside readers get gated dev-log pages" becomes a global rule (`habit-scope.md`), since it is a workflow habit, not a CoSiR fact.
- **Minor:** the page has 1,661 words of prose against its 1,500 budget, because of definitions the cold read asked for; the user has not said whether to trim it.
- **Storage:** `/data/SSD2/third_party/ALADIN_weights/aladinVIT_bamfg_64.47.zip` (320 MB, listed by the r8 build chat) stays on `.scratch/pending_deletions.md`; the user did not decide it.
