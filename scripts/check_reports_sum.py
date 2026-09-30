"""
Check that docs/reports/reports_sum.md indexes every report.

Run it after adding, moving or promoting a report (see "Adding a report" in reports_sum.md).
It fails when:
  - a report (.md under auto/<line>/, stage/, weekly/) or a deck in pptx/ is not linked from
    reports_sum.md
  - a pilots/<dir>/ folder is not linked from reports_sum.md
  - a relative link in reports_sum.md points at nothing (decks are exempt: *.pptx is
    gitignored, so a fresh clone lacks most of them)
  - a file or folder sits loose at the top of docs/reports/

Usage
-----
  python scripts/check_reports_sum.py            # exit 1 and list problems if any
"""
import os
import re
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "reports"
SUM = "reports_sum.md"
TOP_LEVEL = {SUM, "assets", "auto", "stage", "weekly", "pptx"}
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


def indexed_targets(reports: Path) -> set[str]:
    text = (reports / SUM).read_text(encoding="utf-8")
    return {os.path.normpath(t).rstrip("/") for t in LINK.findall(text) if "://" not in t}


def check(reports: Path = REPORTS) -> list[str]:
    problems = []
    linked = indexed_targets(reports)

    for entry in sorted(p.name for p in reports.iterdir()):
        if entry not in TOP_LEVEL and not entry.startswith("."):
            problems.append(f"loose at top level (move it into a folder): {entry}")

    expected = []
    for sub in ("auto", "stage", "weekly"):
        for md in sorted((reports / sub).rglob("*.md")) if (reports / sub).is_dir() else []:
            if "pilots" not in md.relative_to(reports).parts:
                expected.append(md)
    expected += sorted((reports / "pptx").glob("*.pptx")) if (reports / "pptx").is_dir() else []
    pilot_dirs = sorted(d for d in (reports / "auto").glob("*/pilots/*") if d.is_dir()) if (reports / "auto").is_dir() else []

    for path in expected + pilot_dirs:
        rel = path.relative_to(reports).as_posix()
        if rel not in linked:
            problems.append(f"not indexed in {SUM}: {rel}")

    for target in sorted(linked):
        if not (reports / target).exists() and not target.endswith(".pptx"):
            problems.append(f"broken link in {SUM}: {target}")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(p)
    print(f"{SUM}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
