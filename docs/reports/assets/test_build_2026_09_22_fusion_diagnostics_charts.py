"""Regression checks for the table-driven fusion diagnostics chart builder."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("build_2026-09-22_fusion_diagnostics_charts.py")


def load_builder():
    spec = importlib.util.spec_from_file_location("fusion_diagnostics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusionDiagnosticsDataTests(unittest.TestCase):
    def test_build_data_parses_every_required_source_table(self):
        builder = load_builder()

        data = builder.build_data(ROOT)

        self.assertEqual(len(data["early_fusion"]), 5)
        self.assertEqual(len(data["stage1_trajectory"]), 41)
        self.assertEqual(len(data["stage2_trajectory"]), 41)
        self.assertEqual(len(data["weight_sweep"]), 8)
        self.assertEqual(len(data["cca"]), 10)
        self.assertEqual({row["split"] for row in data["weight_sweep"]}, {"train", "held-out"})
        self.assertIn("hierarchical", data["hierarchical"])
        self.assertIn("Control A", data["hierarchical"])
        self.assertIn("Control B", data["hierarchical"])

    def test_parser_skips_markdown_separator_rows_and_preserves_headers(self):
        builder = load_builder()

        tables = builder.parse_markdown_tables(
            ROOT / "src/test/20260923_artelingo_buddy_analysis/cca_audit_pilot_report.md"
        )
        cca = builder.find_table(tables, "component", "held-out correlation")

        self.assertEqual(len(cca), 10)
        self.assertEqual(cca[0]["component"], "1")
        self.assertNotIn("---", cca[0].values())


if __name__ == "__main__":
    unittest.main()
