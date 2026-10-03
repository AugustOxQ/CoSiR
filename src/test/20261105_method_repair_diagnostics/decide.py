"""Joint decision of the method-repair diagnostics (PREREGISTRATION.md §8).

Outputs results/decision.json with the pre-registered table and next step.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Setup path to import from diagnostic and eval modules
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from src.eval.aspect_nested import joint_decision

# Import common module using importlib since directory name starts with a digit
common_path = HERE / "common.py"
spec = importlib.util.spec_from_file_location("common", common_path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


NEXT = {
    "preregister_A3_nested": ("Write the A′ pre-registration: the nested score on A3, spec-§15 cross-fitting on "
                              "seed 42, the GO test once on fresh seed-45 episodes by Oct 12 (H4 only if H3's ceiling "
                              "is sufficient and an H2 model passes its gate and beats A3's pilot by Oct 9)."),
    "h2_grid": ("Run the pre-registered H2 grid (PREREGISTRATION.md §9) on bank {bank} behind its fit gate, re-pilot "
                "each gate-passer with the H1 rule, pre-register A′ by Oct 9 if one is promising, else branch 3."),
    "branch_3": "Stop the repair: branch 3 (analysis paper), per spec §4 and §15.",
}


def decide(pilot: dict, h3: dict) -> dict:
    """Pure decision logic: combines H1 and H3 readings and determines next step.

    Args:
        pilot: dict with at least pilot["A3"]["reading"]
        h3: dict with at least h3["h3_reading"] and h3["matched_k"]["granularity_lever"]

    Returns:
        dict with keys: h1, h3, decision, next_step, h2_bank

    Raises:
        ValueError: if h3_reading is null or unknown readings.
    """
    h1_reading = pilot["A3"]["reading"]
    h3_reading = h3["h3_reading"]

    if h3_reading is None:
        raise ValueError("h3_reading is null (needs_seed43); cannot decide without H3 result")

    # Call the pre-registered joint_decision function
    decision = joint_decision(h1_reading, h3_reading)

    # Determine H2 bank based on granularity_lever
    granularity_lever = h3["matched_k"]["granularity_lever"]
    h2_bank = "MK" if granularity_lever else "AIC"

    # Format the next step message
    if decision == "h2_grid":
        next_step = NEXT[decision].format(bank=h2_bank)
    else:
        next_step = NEXT[decision]

    return {
        "h1": h1_reading,
        "h3": h3_reading,
        "decision": decision,
        "next_step": next_step,
        "h2_bank": h2_bank,
    }


def main():
    parser = argparse.ArgumentParser(description="Joint decision of the method-repair diagnostics")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke data")
    args = parser.parse_args()

    # Get paths
    paths = common.folders(args.smoke)
    res = paths["res"]
    res.mkdir(parents=True, exist_ok=True)

    pilot_path = res / "pilot_seed42.json"
    h3_path = res / "h3.json"
    decision_path = res / "decision.json"

    # Check that input files exist
    if not pilot_path.exists():
        raise FileNotFoundError(f"Pilot results not found: {pilot_path}")
    if not h3_path.exists():
        raise FileNotFoundError(f"H3 results not found: {h3_path}")

    # Refuse to overwrite real results
    if not args.smoke and decision_path.exists():
        raise FileExistsError(f"Real decision.json already exists: {decision_path}. Not overwriting.")

    # Load inputs
    with open(pilot_path) as f:
        pilot = json.load(f)
    with open(h3_path) as f:
        h3 = json.load(f)

    # Make decision
    result = decide(pilot, h3)

    # Compute SHA-256s of inputs
    pilot_sha = common.rg.sha_file(pilot_path)
    h3_sha = common.rg.sha_file(h3_path)

    # Add metadata
    result["inputs_sha256"] = {
        "pilot_seed42.json": pilot_sha,
        "h3.json": h3_sha,
    }
    result["decided_at"] = datetime.now(timezone.utc).isoformat()

    # Write decision
    with open(decision_path, "w") as f:
        json.dump(result, f, indent=1)

    # Print next step
    print(result["next_step"])
    print(f"\nDecision written to {decision_path}")


if __name__ == "__main__":
    main()
