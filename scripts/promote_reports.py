"""
Gather reports from another branch into this branch's docs/reports/ layout (docs only).

Reports stay on their own branch while work happens there. When a stage report or a comparison
needs them here, one run copies everything that is new or changed since the last run:
  - docs/reports/** on the branch: reports, slide markdown, assets (figures, build scripts)
  - pilot reports and logs from src/test/<dir>/*.md (not *_BRIEF.md, READMEs or plans)
  - local .pptx decks, if the branch is checked out in a worktree (decks are gitignored)
Each file lands in the reports_sum.md layout (auto/<line>/, stage/, weekly/, pptx/, assets/,
auto/<line>/pilots/<dir>/) under its shortened name. Links are re-pointed at the copies; links
to code that is not on this branch are left as written. Each new report gets a row in
reports_sum.md (its title is the description; polish by hand if needed), then
check_reports_sum.py runs. The branch itself is never modified.

docs/reports/.promoted.json records what came from where. A report updated on the branch is
copied again on the next run; a copy edited here since the last run is left alone (--force
overwrites it).

Usage
-----
  python scripts/promote_reports.py experiment/percept_topic_pipeline             # copy + index + check
  python scripts/promote_reports.py experiment/percept_topic_pipeline --dry-run   # show the plan only
  python scripts/promote_reports.py <branch> --line <name> --commit                # other line; commit too
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = "docs/reports"
MANIFEST = f"{R}/.promoted.json"
DEFAULT_LINES = {"experiment/percept_topic_pipeline": "percept"}
LAYOUT = ("auto/", "stage/", "weekly/", "pptx/", "assets/")
PILOT_SKIP = {"README.md", "implementation_plan.md", "CLUSTER_RUN_PLAN.md"}
NOT_COPIED = {f"{R}/reports_sum.md", MANIFEST}
# Docs that moved here after side branches recorded their paths; gathered copies follow the move.
REDIRECTS = {
    "docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md":
        "docs/archive/buddy_publication_plan/2026-08-04-buddy-publication-plan-design.md",
    "docs/proposals/2026-08-04-conditional-buddies-publication-proposal.md":
        "docs/archive/buddy_publication_plan/2026-08-04-conditional-buddies-publication-proposal.md",
}
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
DECK_OUT = re.compile(r'((?:parents\[1\]|ROOT) / )"(\d{4}-\d\d-\d\d_[^"/]+\.pptx)"')


@dataclass
class Item:
    src: str          # path on the branch; "worktree:<path>" for a local deck
    data: bytes
    dest: str = ""
    title: str = ""
    action: str = ""

    @property
    def path(self) -> str:
        return self.src.removeprefix("worktree:")


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # same id git gives the file


def title_of(item: Item) -> str:
    if not item.path.endswith(".md"):
        return ""
    lines = item.data.decode("utf-8", "replace").splitlines()
    heading = next((ln[2:] for ln in lines if ln.startswith("# ")), "")
    heading = re.sub(r"\s+", " ", heading.replace("|", "/").replace("`", "")).strip()
    return heading if len(heading) <= 110 else heading[:107].rstrip() + "..."


def shorten(name: str) -> str:
    name = re.sub(r"^(\d{4}-\d\d-\d\d_)weekly_", r"\1", name)
    name = name.replace("stage_report_", "").replace("_stage_report", "")
    return re.sub(r"_report(?=(_slides)?\.(md|pptx)$)", "", name)


def kind_of(name: str, title: str, line: str) -> str:
    low = title.lower()
    if "weekly" in name or low.startswith("weekly report"):
        return "weekly"
    if "stage_report" in name or "stage report" in low or "progress report" in low:
        return "stage"
    return f"auto/{line}"


def destination(item: Item, line: str) -> str:
    if item.path.startswith("src/test/"):
        _, _, folder, name = item.path.split("/")
        return f"{R}/auto/{line}/pilots/{folder}/{name}"
    rel = item.path[len(R) + 1:]
    if rel.startswith(LAYOUT):
        return item.path                      # the branch already uses this layout
    name = os.path.basename(rel)
    if name.endswith(".pptx"):
        return f"{R}/pptx/{shorten(name)}"
    return f"{R}/{kind_of(name, item.title, line)}/{shorten(name)}"


def worktree_of(root: Path, ref: str) -> Path | None:
    path = None
    for ln in git(root, "worktree", "list", "--porcelain").decode().splitlines():
        if ln.startswith("worktree "):
            path = Path(ln[len("worktree "):])
        elif ln == f"branch refs/heads/{ref}":
            return path
    return None


def branch_items(root: Path, ref: str) -> list[Item]:
    here = set(git(root, "ls-tree", "-r", "--name-only", "HEAD", "--", "src/test").decode().splitlines())
    items = []
    for f in git(root, "ls-tree", "-r", "--name-only", ref, "--", R, "src/test").decode().splitlines():
        if f in NOT_COPIED or f.endswith(".pyc"):
            continue
        if f.startswith("src/test/"):
            parts = f.split("/")
            if (len(parts) != 4 or not f.endswith(".md") or f in here or parts[3] in PILOT_SKIP
                    or parts[3].upper().endswith("BRIEF.MD")):
                continue
        items.append(Item(f, git(root, "show", f"{ref}:{f}")))
    tracked = {i.src for i in items}
    wt = worktree_of(root, ref)
    for deck in sorted((wt / R).rglob("*.pptx")) if wt else []:
        rel = deck.relative_to(wt).as_posix()
        if rel not in tracked:
            items.append(Item(f"worktree:{rel}", deck.read_bytes()))
    return items


def rewrite_links(content: str, item: Item, mapping: dict[str, str], root: Path) -> str:
    def fix(m):
        target = m.group(2)
        if SCHEME.match(target) or target.startswith(("#", "/")):
            return m.group(0)
        path, _, frag = target.partition("#")
        for c in (os.path.normpath(os.path.join(os.path.dirname(item.path), path)), os.path.normpath(path)):
            new = None if c.startswith("..") else mapping.get(c) or (c if (root / c).exists() else None)
            if new:
                rel = os.path.relpath(new, os.path.dirname(item.dest)) + ("/" if path.endswith("/") else "")
                return m.group(1) + rel + (f"#{frag}" if frag else "") + m.group(3)
        return m.group(0)                     # not on this branch: keep as written
    return LINK.sub(fix, content)


def rewrite_names(content: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        content = re.sub(r"(?<![\w./-])" + re.escape(old) + r"(?![\w-])", new, content)
        ob, nb = os.path.basename(old), os.path.basename(new)
        if ob != nb:
            content = re.sub(r"(?<![\w/-])" + re.escape(ob) + r"(?![\w-])", nb, content)
    return DECK_OUT.sub(lambda m: f'{m.group(1)}"pptx" / "{shorten(m.group(2))}"', content)


def render(item: Item, mapping: dict[str, str], renames: list[tuple[str, str]], root: Path) -> bytes:
    if not item.path.endswith((".md", ".py")):
        return item.data
    content = item.data.decode("utf-8")
    if item.path.endswith(".md"):
        content = rewrite_links(content, item, mapping, root)
    return rewrite_names(content, renames).encode("utf-8")


def decide(item: Item, rec: dict | None, root: Path, force: bool) -> str:
    if rec and rec["src_blob"] == blob_id(item.data):
        return "same"
    dest = root / item.dest
    if dest.exists() and not force:
        current = blob_id(dest.read_bytes())
        if rec and current != rec["dest_blob"]:
            return "skip (edited here since the last copy; --force overwrites)"
        if not rec:
            same_bytes = current == blob_id(item.data)
            return "record" if same_bytes else "skip (already here, not copied from this branch)"
    return "update" if rec else "new"


def plan(root: Path, ref: str, line: str, manifest: dict, force: bool) -> list[Item]:
    recs = manifest.get(ref, {})
    items = branch_items(root, ref)
    for i in items:
        i.title = title_of(i)
    kinds = {os.path.basename(i.path): kind_of(os.path.basename(i.path), i.title, line) for i in items}
    for i in items:
        i.dest = recs[i.src]["to"] if i.src in recs else destination(i, line)
        base = os.path.basename(i.path)
        if i.src not in recs and base.endswith("_slides.md") and i.dest.startswith(f"{R}/auto/"):
            parent = kinds.get(base.replace("_slides.md", ".md"), "weekly")
            i.dest = f"{R}/{'weekly' if parent.startswith('auto/') else parent}/{shorten(base)}"
    seen = set()
    for i in items:
        clash = "skip (another file maps to the same place)"
        i.action = clash if i.dest in seen else decide(i, recs.get(i.src), root, force)
        seen.add(i.dest)
    return items


def deck_row(root: Path, base: str, rel: str) -> str:
    stem = base[:-len(".pptx")]
    source = next((p for p in (f"weekly/{stem}.md", f"stage/{stem}.md") if (root / R / p).exists()), "none")
    builds = sorted((root / R / "assets").glob("build_*.py"))
    build = next((f"`{p.name}`" for p in builds if base in p.read_text(encoding="utf-8", errors="ignore")), "none")
    return f"| [{base}]({rel}) | {source} | {build} |"


def row_for(item: Item, root: Path) -> tuple[str, str] | None:
    rel, base = os.path.relpath(item.dest, R), os.path.basename(item.dest)
    date, slug = base[5:10], os.path.splitext(base[11:])[0]
    if rel.startswith("pptx/"):
        return "## pptx", deck_row(root, base, rel)
    if rel.startswith("weekly/"):
        cells = f"| | [{slug}]({rel}) |" if base.endswith("_slides.md") else f"| [{slug}]({rel}) | |"
        return "## weekly", f"| {date} {cells} {item.title} |"
    if rel.startswith(("stage/", "auto/")) and base.endswith(".md"):
        section = "## stage" if rel.startswith("stage/") else "## " + "/".join(rel.split("/")[:2])
        return section, f"| {date} | [{slug}]({rel}) | {item.title} |"
    return None


def index_rows(items: list[Item], ref: str, root: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    new = [i for i in items if i.action == "new"]
    for hit in (row_for(i, root) for i in new if "/pilots/" not in i.dest):
        if hit:
            rows.setdefault(hit[0], []).append(hit[1])
    for folder in sorted({os.path.dirname(i.dest) for i in new if "/pilots/" in i.dest}):
        rel, n = os.path.relpath(folder, R), sum(1 for _ in (root / folder).iterdir())
        rows.setdefault("## Pilots", []).append(f"| [{rel.removeprefix('auto/')}]({rel}/) | {n} | Copied from `{ref}` |")
    return rows


def table_end(lines: list[str], head: int) -> int:
    k = head + 1
    while k < len(lines) and not lines[k].startswith("|"):
        k += 1
    while k < len(lines) and lines[k].startswith("|"):
        k += 1
    return k


def add_rows(text: str, rows: dict[str, list[str]], ref: str) -> str:
    lines = text.split("\n")
    for section, new in rows.items():
        new = [r for r in new if f"]({r.split('](')[1].split(')')[0]})" not in text]
        head = next((k for k, ln in enumerate(lines) if ln.startswith(section)), None)
        if new and head is None:              # a research line seen for the first time
            at = next((k for k, ln in enumerate(lines) if ln.startswith("## Pilots")), len(lines))
            lines[at:at] = [f"{section}: reports gathered from `{ref}`", "", "| Date | Report | What it is |",
                            "|---|---|---|", *new, ""]
        elif new:
            at = table_end(lines, head)
            lines[at:at] = new
    return "\n".join(lines)


def run_checker(root: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("check_reports_sum", Path(__file__).with_name("check_reports_sum.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check(root / R)


def promote(root: Path, ref: str, line: str, dry_run: bool = False, force: bool = False) -> list[Item]:
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    items = plan(root, ref, line, manifest, force)
    if dry_run:
        return items
    mapping = {i.path: i.dest for i in items if not i.action.startswith("skip")}
    mapping.update({os.path.dirname(i.path): os.path.dirname(i.dest) for i in items if "/pilots/" in i.dest})
    mapping.update(REDIRECTS)
    renames = sorted([(i.path, i.dest) for i in items if i.path.startswith(R) and i.path != i.dest]
                     + list(REDIRECTS.items()), key=lambda pair: -len(pair[0]))
    recs = manifest.setdefault(ref, {})
    for i in items:
        if i.action in ("new", "update"):
            (root / i.dest).parent.mkdir(parents=True, exist_ok=True)
            (root / i.dest).write_bytes(render(i, mapping, renames, root))
        if i.action in ("new", "update", "record"):
            dest_blob = blob_id((root / i.dest).read_bytes())
            recs[i.src] = {"to": i.dest, "src_blob": blob_id(i.data), "dest_blob": dest_blob}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    summary = root / R / "reports_sum.md"
    rows = index_rows(items, ref, root)
    summary.write_text(add_rows(summary.read_text(encoding="utf-8"), rows, ref), encoding="utf-8")
    return items


def print_plan(items: list[Item]) -> dict[str, int]:
    for i in (i for i in items if i.action != "same"):
        print(f"{i.action}: {i.src}" if i.action.startswith("skip") else f"{i.action:<7} {i.src} -> {i.dest}")
    counts = {a: sum(i.action.split(" ")[0] == a for i in items) for a in ("new", "update", "record", "skip", "same")}
    print(" | ".join(f"{a}: {n}" for a, n in counts.items()))
    return counts


def commit(root: Path, ref: str, counts: dict[str, int]) -> None:
    git(root, "add", "-A", R)
    sha = git(root, "rev-parse", "--short", ref).decode().strip()
    git(root, "commit", "-q", "-m", f"docs(reports): gather reports from {ref} ({sha})\n\n"
        f"{counts['new']} new, {counts['update']} updated; copied by scripts/promote_reports.py.")
    print("committed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("branch", help="branch or tag to gather reports from")
    ap.add_argument("--line", help="research line for auto/<line>/ (default: known per branch)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be copied, change nothing")
    ap.add_argument("--force", action="store_true", help="overwrite copies that were edited here")
    ap.add_argument("--commit", action="store_true", help="commit docs/reports afterwards")
    args = ap.parse_args()
    line = args.line or DEFAULT_LINES.get(args.branch)
    if not line:
        ap.error(f"no default research line for {args.branch}; pass --line <name>")
    if git(ROOT, "branch", "--show-current").decode().strip() == args.branch:
        ap.error("run this from the branch that should receive the reports, not from the source branch")

    counts = print_plan(promote(ROOT, args.branch, line, args.dry_run, args.force))
    if args.dry_run:
        return 0
    problems = run_checker(ROOT)
    print("\n".join(problems) or "reports_sum.md: OK")
    if problems:
        return 1
    if args.commit and (counts["new"] or counts["update"]):
        commit(ROOT, args.branch, counts)
    elif not args.commit:
        print("review with `git status docs/reports`, polish new rows in reports_sum.md, then commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
