import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_reports_sum", ROOT / "scripts" / "check_reports_sum.py")
check_reports_sum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_reports_sum)


def test_repo_reports_sum_indexes_every_report():
    assert check_reports_sum.check() == []


@pytest.fixture
def reports(tmp_path):
    for sub in ("auto/v2/pilots/20261001_x", "stage", "weekly", "pptx", "assets"):
        (tmp_path / sub).mkdir(parents=True)
    (tmp_path / "auto/v2/2026-10-01_a.md").write_text("# a")
    (tmp_path / "auto/v2/pilots/20261001_x/p_report.md").write_text("# p")
    (tmp_path / "pptx/2026-10-01_a_slides.pptx").write_bytes(b"")
    (tmp_path / "reports_sum.md").write_text(
        "[a](auto/v2/2026-10-01_a.md) [p](auto/v2/pilots/20261001_x/) "
        "[deck](pptx/2026-10-01_a_slides.pptx) [gone](pptx/2026-01-01_gitignored.pptx)"
    )
    return tmp_path


def test_clean_index_passes_and_missing_decks_are_exempt(reports):
    assert check_reports_sum.check(reports) == []


def test_unindexed_report_and_broken_link_and_loose_file_are_reported(reports):
    (reports / "stage/2026-10-02_new.md").write_text("# new")
    (reports / "2026-10-03_loose.md").write_text("# loose")
    (reports / "reports_sum.md").write_text((reports / "reports_sum.md").read_text() + " [x](weekly/missing.md)")
    problems = check_reports_sum.check(reports)
    assert "not indexed in reports_sum.md: stage/2026-10-02_new.md" in problems
    assert "not indexed in reports_sum.md: 2026-10-03_loose.md" not in problems  # reported as loose instead
    assert "loose at top level (move it into a folder): 2026-10-03_loose.md" in problems
    assert "broken link in reports_sum.md: weekly/missing.md" in problems
