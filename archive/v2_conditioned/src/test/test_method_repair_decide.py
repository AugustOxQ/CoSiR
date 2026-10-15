"""Tests for the joint decision logic (Task 8 of method-repair diagnostics)."""

import importlib.util
import json
import pytest
from pathlib import Path

# Import the decide function and NEXT dict using importlib since module name starts with a digit
diagnostics_path = Path(__file__).resolve().parent / "20261105_method_repair_diagnostics" / "decide.py"
spec = importlib.util.spec_from_file_location("decide", diagnostics_path)
decide_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decide_module)
decide = decide_module.decide
write_decision_atomic = decide_module.write_decision_atomic
NEXT = decide_module.NEXT


class TestJointDecisionTable:
    """Test all 9 (h1, h3) combinations per pre-registered table."""

    @pytest.mark.parametrize("h1_reading,h3_reading,expected_decision", [
        # h1 = "promising" → always preregister_A3_nested (regardless of h3)
        ("promising", "no_fit", "preregister_A3_nested"),
        ("promising", "ceiling_too_low", "preregister_A3_nested"),
        ("promising", "ceiling_sufficient", "preregister_A3_nested"),
        # h1 = "inconclusive" → h2_grid only if ceiling_sufficient, else branch_3
        ("inconclusive", "no_fit", "branch_3"),
        ("inconclusive", "ceiling_too_low", "branch_3"),
        ("inconclusive", "ceiling_sufficient", "h2_grid"),
        # h1 = "not_promising" → h2_grid only if ceiling_sufficient, else branch_3
        ("not_promising", "no_fit", "branch_3"),
        ("not_promising", "ceiling_too_low", "branch_3"),
        ("not_promising", "ceiling_sufficient", "h2_grid"),
    ])
    def test_decision_table_9_combinations(self, h1_reading, h3_reading, expected_decision):
        """Test all 9 (h1, h3) combinations against pre-registered decision table."""
        pilot = {"A3": {"reading": h1_reading}}
        h3 = {
            "h3_reading": h3_reading,
            "matched_k": {"granularity_lever": True},
        }
        result = decide(pilot, h3)
        assert result["h1"] == h1_reading
        assert result["h3"] == h3_reading
        assert result["decision"] == expected_decision

    @pytest.mark.parametrize("granularity_lever,expected_bank", [
        (True, "MK"),
        (False, "AIC"),
    ])
    def test_h2_bank_selection(self, granularity_lever, expected_bank):
        """Test h2_bank is MK when granularity_lever=True, AIC when False."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": granularity_lever},
        }
        result = decide(pilot, h3)
        assert result["h2_bank"] == expected_bank

    def test_next_step_preregister_a3_nested_exact(self):
        """Assert next_step is exact NEXT["preregister_A3_nested"] for promising."""
        pilot = {"A3": {"reading": "promising"}}
        h3 = {
            "h3_reading": "no_fit",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["next_step"] == NEXT["preregister_A3_nested"]

    @pytest.mark.parametrize("granularity_lever,expected_bank", [
        (True, "MK"),
        (False, "AIC"),
    ])
    def test_next_step_h2_grid_exact(self, granularity_lever, expected_bank):
        """Assert next_step equals NEXT["h2_grid"].format(bank=bank) for h2_grid decision."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": granularity_lever},
        }
        result = decide(pilot, h3)
        expected_next_step = NEXT["h2_grid"].format(bank=expected_bank)
        assert result["next_step"] == expected_next_step

    def test_next_step_branch_3_exact(self):
        """Assert next_step is exact NEXT["branch_3"] for branch_3 decision."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "no_fit",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["next_step"] == NEXT["branch_3"]

    def test_failed_mk3_shape_h2_bank_aic(self):
        """Test failed MK3 shape: h3["matched_k"] has status and granularity_lever=False → AIC."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"status": "failed", "granularity_lever": False},
        }
        result = decide(pilot, h3)
        assert result["h2_bank"] == "AIC"
        assert result["decision"] == "h2_grid"

    def test_null_h3_reading_raises(self):
        """null h3_reading raises ValueError before any decision."""
        pilot = {"A3": {"reading": "inconclusive"}}
        h3 = {
            "h3_reading": None,
            "matched_k": {"granularity_lever": False},
        }
        with pytest.raises(ValueError, match="h3_reading is null"):
            decide(pilot, h3)

    def test_result_has_required_keys(self):
        """Verify result dict has all required keys."""
        pilot = {"A3": {"reading": "promising"}}
        h3 = {
            "h3_reading": "ceiling_sufficient",
            "matched_k": {"granularity_lever": False},
        }
        result = decide(pilot, h3)
        required_keys = {"h1", "h3", "decision", "next_step", "h2_bank"}
        assert required_keys.issubset(result.keys())


class TestAtomicWrite:
    """Test atomic write behavior with race condition protection."""

    def test_real_mode_raises_when_target_exists(self, tmp_path):
        """Real mode: FileExistsError raised when decision.json already exists, content unchanged."""
        decision_path = tmp_path / "decision.json"
        original_content = {"existing": "decision"}

        # Pre-create the target file
        with open(decision_path, "w") as f:
            json.dump(original_content, f)

        # Try to write new result
        new_result = {"h1": "promising", "h3": "ceiling_sufficient", "decision": "preregister_A3_nested"}

        # Should raise FileExistsError
        with pytest.raises(FileExistsError, match="decision.json already exists"):
            write_decision_atomic(new_result, decision_path, is_smoke=False)

        # Verify original content is unchanged
        with open(decision_path) as f:
            actual_content = json.load(f)
        assert actual_content == original_content

    def test_smoke_mode_overwrites(self, tmp_path):
        """Smoke mode: overwrites existing file."""
        decision_path = tmp_path / "decision.json"
        original_content = {"old": "decision"}

        # Pre-create the target file
        with open(decision_path, "w") as f:
            json.dump(original_content, f)

        # Write new result in smoke mode
        new_result = {
            "h1": "promising",
            "h3": "ceiling_sufficient",
            "decision": "preregister_A3_nested",
            "next_step": "test",
            "h2_bank": "MK",
        }
        write_decision_atomic(new_result, decision_path, is_smoke=True)

        # Verify new content was written
        with open(decision_path) as f:
            actual_content = json.load(f)
        assert actual_content == new_result

    def test_real_mode_creates_new_file(self, tmp_path):
        """Real mode: creates new file when target doesn't exist."""
        decision_path = tmp_path / "decision.json"

        result = {
            "h1": "promising",
            "h3": "ceiling_sufficient",
            "decision": "preregister_A3_nested",
            "next_step": "test",
            "h2_bank": "MK",
        }
        write_decision_atomic(result, decision_path, is_smoke=False)

        # Verify file was created with correct content
        assert decision_path.exists()
        with open(decision_path) as f:
            actual_content = json.load(f)
        assert actual_content == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
