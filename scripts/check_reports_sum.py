"""
Check that docs/reports/reports_sum.md indexes every report.

Run it after adding, moving or promoting a report (see "Adding a report" in reports_sum.md).
It fails when:
  - a report (.md under auto/<line>/, literature/, stage/, weekly/) or a deck in pptx/ is not linked from
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
TOP_LEVEL = {SUM, "assets", "auto", "literature", "stage", "weekly", "pptx"}
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


def indexed_targets(reports: Path) -> set[str]:
    text = (reports / SUM).read_text(encoding="utf-8")
    return {os.path.normpath(t).rstrip("/") for t in LINK.findall(text) if "://" not in t}


def must_be_indexed(reports: Path) -> list[str]:
    """Reports (outside pilots/), decks and pilot folders, relative to docs/reports."""
    paths = [md for sub in ("auto", "literature", "stage", "weekly") for md in (reports / sub).rglob("*.md")
             if "pilots" not in md.relative_to(reports).parts]
    paths += (reports / "pptx").glob("*.pptx")
    paths += (d for d in (reports / "auto").glob("*/pilots/*") if d.is_dir())
    return sorted(p.relative_to(reports).as_posix() for p in paths)


def check(reports: Path = REPORTS) -> list[str]:
    linked = indexed_targets(reports)
    loose = [e.name for e in reports.iterdir() if e.name not in TOP_LEVEL and not e.name.startswith(".")]
    problems = [f"loose at top level (move it into a folder): {name}" for name in sorted(loose)]
    problems += [f"not indexed in {SUM}: {rel}" for rel in must_be_indexed(reports) if rel not in linked]
    problems += [f"broken link in {SUM}: {t}" for t in sorted(linked)
                 if not (reports / t).exists() and not t.endswith(".pptx")]
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(p)
    print(f"{SUM}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
