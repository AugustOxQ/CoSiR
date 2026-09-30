import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promote_reports = _load("promote_reports")
check_reports_sum = _load("check_reports_sum")

SUM = """# Reports guide

## stage: stage reports

| Date | Report | What it is |
|---|---|---|

## weekly: weekly reports and slides

| Week of | Report | Slides md | Topic |
|---|---|---|---|

## Pilots

| Folder | Files | Contents |
|---|---|---|

## pptx: rendered decks

| Deck | Source | Build script |
|---|---|---|
"""
BUILD = 'OUT = Path(__file__).resolve().parents[1] / "2026-10-01_weekly_topic_slides.pptx"\n'
THING = ("# Thing result\n![f](assets/fig.png) [p](../../src/test/20261003_p/p_report.md) "
         "[code](../../src/test/20261003_p/run.py) [w](2026-10-01_weekly_topic.md)\n")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text) if isinstance(text, bytes) else path.write_text(text)


def _commit_on(repo, branch, files):
    worktree = repo.parent / "wt"             # once the fixture checks the branch out there
    where = worktree if worktree.exists() else repo
    if where is repo:
        _git(repo, "checkout", "-q", branch)
    for rel, text in files.items():
        _write(where, rel, text)
    _git(where, "add", "-A")
    _git(where, "commit", "-q", "-m", "change")
    if where is repo:
        _git(repo, "checkout", "-q", "main")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "docs/reports/reports_sum.md", SUM)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "branch", "feat")
    _commit_on(repo, "feat", {
        "docs/reports/2026-10-01_weekly_topic.md": "# Weekly Report — Topic\n",
        "docs/reports/2026-10-01_weekly_topic_slides.md": "# Topic in slides\n",
        "docs/reports/2026-10-02_x_stage_report.md": "# X stage report\n",
        "docs/reports/2026-10-03_thing_report.md": THING,
        "docs/reports/assets/fig.png": b"\x89PNG fake",
        "docs/reports/assets/build_2026-10-01_weekly_slides.py": BUILD,
        "src/test/20261003_p/p_report.md": "# pilot\n",
        "src/test/20261003_p/P_BRIEF.md": "# brief\n",
        "src/test/20261003_p/run.py": "print()\n",
    })
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "feat")
    _write(tmp_path / "wt", "docs/reports/2026-10-01_weekly_topic_slides.pptx", b"deck")
    return repo


def _actions(items):
    return {i.src: i.action for i in items}


def test_first_run_places_rewrites_and_indexes_everything(repo):
    items = promote_reports.promote(repo, "feat", "lab")
    r = repo / "docs/reports"
    for rel in ("weekly/2026-10-01_topic.md", "weekly/2026-10-01_topic_slides.md", "stage/2026-10-02_x.md",
                "auto/lab/2026-10-03_thing.md", "assets/fig.png", "auto/lab/pilots/20261003_p/p_report.md",
                "pptx/2026-10-01_topic_slides.pptx"):
        assert (r / rel).exists(), rel
    assert not (r / "auto/lab/pilots/20261003_p/P_BRIEF.md").exists()
    thing = (r / "auto/lab/2026-10-03_thing.md").read_text()
    assert "](../../assets/fig.png)" in thing
    assert "](pilots/20261003_p/p_report.md)" in thing
    assert "](../../src/test/20261003_p/run.py)" in thing          # not on this branch: kept as written
    assert "](../../weekly/2026-10-01_topic.md)" in thing
    assert '"pptx" / "2026-10-01_topic_slides.pptx"' in (r / "assets/build_2026-10-01_weekly_slides.py").read_text()
    summary = (r / "reports_sum.md").read_text()
    assert "## auto/lab: reports gathered from `feat`" in summary
    assert "| 10-03 | [thing](auto/lab/2026-10-03_thing.md) | Thing result |" in summary
    assert "`build_2026-10-01_weekly_slides.py`" in summary
    assert check_reports_sum.check(r) == []
    assert set(_actions(items).values()) == {"new"}


def test_second_run_is_a_no_op(repo):
    promote_reports.promote(repo, "feat", "lab")
    before = (repo / "docs/reports/reports_sum.md").read_text()
    items = promote_reports.promote(repo, "feat", "lab")
    assert set(_actions(items).values()) == {"same"}
    assert (repo / "docs/reports/reports_sum.md").read_text() == before


def test_branch_update_is_copied_but_local_edits_are_protected(repo):
    promote_reports.promote(repo, "feat", "lab")
    src = "docs/reports/2026-10-03_thing_report.md"
    _commit_on(repo, "feat", {src: THING + "more\n"})
    items = promote_reports.promote(repo, "feat", "lab")
    assert _actions(items)[src] == "update"
    copy = repo / "docs/reports/auto/lab/2026-10-03_thing.md"
    assert copy.read_text().endswith("more\n")

    copy.write_text(copy.read_text() + "edited on main\n")
    _commit_on(repo, "feat", {src: THING + "even more\n"})
    items = promote_reports.promote(repo, "feat", "lab")
    assert _actions(items)[src].startswith("skip")
    assert "edited on main" in copy.read_text()
    items = promote_reports.promote(repo, "feat", "lab", force=True)
    assert _actions(items)[src] == "update" and copy.read_text().endswith("even more\n")
