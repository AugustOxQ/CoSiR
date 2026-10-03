"""Tests for the joint decision logic (Task 8 of method-repair diagnostics)."""

import importlib.util
import pytest
import sys
from pathlib import Path

# Import the decide function using importlib since module name starts with a digit
diagnostics_path = Path(__file__).resolve().parent / "20261105_method_repair_diagnostics" / "decide.py"
spec = importlib.util.spec_from_file_location("decide", diagnostics_path)
decide_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decide_module)
decide = decide_module.decide


class TestJointDecision:
    """Test the decide() pure function with synthetic data."""

    def test_promising_reading(self):
        """(a) promising H1 → preregister_A3_nested, regardless of H3."""
        pilot = {"A3": {"reading": "promising"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": True},
        }
        result = decide(pilot, h3)
        assert result["h1"] == "promising"
        assert result["h3"] == "ceiling_sufficient"
        assert result["decision"] == "preregister_A3_nested"
        assert "pre-registration" in result["next_step"]
        assert result["h2_bank"] == "MK"

    def test_not_promising_ceiling_sufficient_mk(self):
        """(b) not_promising + ceiling_sufficient + granularity_lever=True → h2_grid with MK."""
        pilot = {"A3": {"reading": "not_promising"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": True},
        }
        result = decide(pilot, h3)
        assert result["h1"] == "not_promising"
        assert result["h3"] == "ceiling_sufficient"
        assert result["decision"] == "h2_grid"
        assert result["h2_bank"] == "MK"
        assert "H2 grid" in result["next_step"]
        assert "{bank}" not in result["next_step"]  # Should be formatted
        assert "MK" in result["next_step"]

    def test_not_promising_ceiling_sufficient_aic(self):
        """(b) not_promising + ceiling_sufficient + granularity_lever=False → h2_grid with AIC."""
        pilot = {"A3": {"reading": "not_promising"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["h1"] == "not_promising"
        assert result["h3"] == "ceiling_sufficient"
        assert result["decision"] == "h2_grid"
        assert result["h2_bank"] == "AIC"
        assert "H2 grid" in result["next_step"]
        assert "AIC" in result["next_step"]

    def test_inconclusive_no_fit_branch_3(self):
        """(c) inconclusive + no_fit → branch_3."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "no_fit",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["h1"] == "inconclusive"
        assert result["h3"] == "no_fit"
        assert result["decision"] == "branch_3"
        assert "branch 3" in result["next_step"]

    def test_inconclusive_ceiling_too_low_branch_3(self):
        """(c) inconclusive + ceiling_too_low → branch_3."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "ceiling_too_low",
            "matched_k": {"granularity_lever": True},
        }
        result = decide(pilot, h3)
        assert result["h1"] == "inconclusive"
        assert result["h3"] == "ceiling_too_low"
        assert result["decision"] == "branch_3"
        assert "branch 3" in result["next_step"]

    def test_null_h3_reading_raises(self):
        """(d) null h3_reading raises ValueError."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": None,
            "matched_k": {"granularity_lever": False},
        }
        with pytest.raises(ValueError, match="h3_reading is null"):
            decide(pilot, h3)

    def test_not_promising_ceiling_too_low_branch_3(self):
        """not_promising + ceiling_too_low → branch_3."""
        pilot = {"A3": {"reading": "not_promising"}}
        h3 = {
            "h3_reading": "ceiling_too_low",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["decision"] == "branch_3"

    def test_result_structure(self):
        """Verify result dict has all required keys."""
        pilot = {"A3": {"reading": "promising"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        required_keys = {"h1", "h3", "decision", "next_step", "h2_bank"}
        assert required_keys.issubset(result.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
