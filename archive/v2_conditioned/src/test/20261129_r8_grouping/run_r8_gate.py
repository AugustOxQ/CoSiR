"""Ticket 10: pending -> C13 -> write-once trigger/verdict, without scoring or GPU work.

CLI: ``pending|finalize --step trigger|gate [--smoke] [--results ROOT]``.
The seed42 index schema is ``{"batches": [{"file": "arrays_main.npz", "sha256": "..."}, ...]}``.
Grouping metadata is ``{"groupings": {ID: M3_record}, "dropped": {ID: reason}}``.
C13 records at rederive/<step>_rederive.json hold ``step``, ``status: "agree"``,
``smoke`` and ``input_sha256`` (absolute file paths -> SHA-256). They must cover
the pending file as well as its inputs; any additional inputs are rehashed too.
These explicit contracts are asserted; no absent scorer is silently dropped.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import r8_gate as G


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    assert path.is_file(), f"missing input {path}"
    before = sha_file(path)
    rec = json.loads(path.read_text())
    assert isinstance(rec, dict), f"{path}: expected JSON object"
    assert sha_file(path) == before, f"{path}: changed during read"
    return rec


def require_rederivation(record, expected_sha256, step, smoke):
    """No verdict unless ticket 11 agrees on this step and these exact inputs."""
    # rule gap: section 7/ticket 11 name the record but omit its JSON schema.
    # This contract uses status='agree', step, smoke and absolute-path input_sha256.
    assert record.get("status") == "agree", "C13 re-derivation did not agree"
    assert record.get("step") == step, "C13 re-derivation is for another step"
    assert record.get("smoke") is smoke, "C13 smoke/real mismatch"
    covered = record.get("input_sha256")
    assert isinstance(covered, dict), "C13 needs input SHA-256s"
    for path, digest in expected_sha256.items():
        assert covered.get(path) == digest, f"C13 does not cover current SHA-256 of {path}"


def publish_checked(pending_path, record_path, verdict_path, write_json_once):
    """File boundary for C13; verifies hashes before calling the write-once writer."""
    pending_path, record_path, verdict_path = map(Path, (pending_path, record_path, verdict_path))
    assert not verdict_path.exists(), f"refusing to overwrite {verdict_path}"
    assert record_path.is_file(), f"missing C13 re-derivation record {record_path}"
    pending_sha, record_sha = sha_file(pending_path), sha_file(record_path)
    pending, record = read_json(pending_path), read_json(record_path)
    step, smoke = pending["step"], pending["smoke"]
    assert step in ("trigger", "gate") and isinstance(smoke, bool)
    expected = {**pending["input_sha256"], str(pending_path.resolve()): pending_sha}
    require_rederivation(record, expected, step, smoke)
    for path, digest in record["input_sha256"].items():
        assert Path(path).is_absolute(), "C13 input paths must be absolute"
        assert sha_file(path) == digest, f"C13 input SHA-256 changed: {path}"
    assert sha_file(pending_path) == pending_sha and sha_file(record_path) == record_sha, "C13 files changed"
    payload = pending["trigger"] if step == "trigger" else pending
    payload = {k: v for k, v in payload.items() if k not in ("rule_sha256", "written")}
    payload.update(smoke=smoke, step=step, input_sha256=expected,
                   rederivation_sha256=record_sha, rederivation_file=str(record_path.resolve()))
    return write_json_once(verdict_path, payload)


def load_batches(root, index_paths):
    """Read every indexed batch, checking hashes and repeated anchor arrays."""
    # rule gap: ticket 09 specifies an index but no filename/schema. Default:
    # seed42/index.json, batches=[{file, sha256}]; --index accepts multiple immutable indices.
    seed42 = (root / "seed42").resolve()
    batches, inputs = [], {}
    for index_path in index_paths:
        index_path = Path(index_path).resolve()
        index_sha = sha_file(index_path)
        index = read_json(index_path)
        entries = index["batches"]
        assert isinstance(entries, list) and entries, "index needs nonempty batches list"
        inputs[str(index_path)] = index_sha
        for entry in entries:
            assert isinstance(entry, dict) and {"file", "sha256"} <= set(entry)
            path = (seed42 / entry["file"]).resolve()
            assert path.parent == seed42 and path.name.startswith("arrays_") and path.suffix == ".npz", "invalid batch path"
            assert str(path) not in inputs, f"batch indexed twice: {path}"
            digest = sha_file(path)
            assert digest == entry["sha256"], f"batch SHA-256 mismatch: {path}"
            with np.load(path, allow_pickle=False) as z:
                batch = {k: z[k] for k in z.files}
            assert sha_file(path) == digest, f"batch changed during read: {path}"
            assert not any(k.startswith(("R1_H", "XAFF_H")) for k in batch), "H<X> may run only after gate_verdict.json"
            batches.append(batch)
            inputs[str(path)] = digest
        assert sha_file(index_path) == index_sha, f"index changed during read: {index_path}"
    return G.merge_batches(batches), inputs


def grouping_metadata(paths):
    """Merge immutable grouping/drop records, including a later NBC collect file."""
    # rule gap: grouping JSON's internal shape is unspecified. Use groupings={ID: record},
    # dropped={ID: reason}; preserve each entire M3 record (including every (r,K)).
    groups, dropped, inputs = {}, {}, {}
    for path in paths:
        path = Path(path).resolve()
        digest = sha_file(path)
        rec = read_json(path)
        assert isinstance(rec.get("groupings"), dict) and isinstance(rec.get("dropped"), dict)
        assert all(isinstance(value, dict) and value for value in rec["groupings"].values()), "grouping records must be nonempty objects"
        for target, source in ((groups, rec["groupings"]), (dropped, rec["dropped"])):
            for x, value in source.items():
                assert x not in target or target[x] == value, f"conflicting grouping record: {x}"
                target[x] = value
        inputs[str(path)] = digest
        assert sha_file(path) == digest, f"grouping metadata changed: {path}"
    assert all(isinstance(reason, str) and reason.strip() for reason in dropped.values()), "drop needs reason"
    return groups, dropped, inputs


def stage_records(root, smoke, rule_sha):
    """Require SC1 before any candidate evaluation; preserve SC2/ESC1 inputs."""
    # rule gap: stages' JSON schemas are unspecified; use R7's top-level pass and smoke.
    records, inputs = {}, {}
    names = ("stage0_regression.json", "stage1_fcmi.json", "stageE_AL.json", "stageE_DI.json", "stageE_SG.json")
    for name in names:
        path = (root / name).resolve()
        digest = sha_file(path)
        rec = read_json(path)
        assert isinstance(rec.get("pass"), bool) and rec.get("smoke") is smoke, f"{name}: pass/smoke missing or mismatched"
        assert rec.get("rule_sha256") == rule_sha, f"{name}: decision rule SHA-256 mismatch"
        records[name] = rec
        inputs[str(path)] = digest
        assert sha_file(path) == digest, f"stage record changed: {name}"
    assert records["stage0_regression.json"]["pass"], "SC1 failure stops both tracks before candidate numbers"
    return records, inputs


def prepare_pending(root, step, smoke, index_paths, grouping_paths, common):
    """File-level wiring only. The verdict calculations remain in r8_gate."""
    common.assert_rule()
    common.assert_inputs()
    assert step in ("trigger", "gate") and isinstance(smoke, bool)
    # rule gap: ticket 10 names gate_pending but does not name the pre-trigger
    # pending file. Use trigger_pending.json, then gate_pending.json after NBC,
    # so both steps retain immutable pending inputs for their own C13 check.
    pending_path = root / f"{step}_pending.json"
    assert not pending_path.exists(), f"refusing to overwrite {pending_path}"
    rule_path = Path(common.RULE).resolve()
    rule_sha = sha_file(rule_path)
    stages, inputs = stage_records(root, smoke, rule_sha)
    arr, batch_inputs = load_batches(root, index_paths)
    inputs.update(batch_inputs)
    groups, dropped, group_inputs = grouping_metadata(grouping_paths)
    inputs.update(group_inputs)
    inputs[str(rule_path)] = rule_sha
    if not smoke:
        assert len(arr["cl"]) == 12288, "M15 requires 12,288 pooled episodes"
        # C6/M14: every real batch's anchor/painting order and frozen baseline
        # are the R7 seed-42 arrays that SC1 proved. Never reconstruct an ID join.
        old = (Path(common.MAIN) / "src/test/20261126_design_l_style_source/results/seed42/arrays.npz").resolve()
        old_sha = sha_file(old)
        with np.load(old, allow_pickle=False) as z:
            keys = [k for k in z.files if k in ("cl", "pair_index", "parity")
                    or k.startswith(("AFF__", "AFF_cf__", "B__", "B0__", "B1__"))]
            for key in keys:
                assert key in arr and np.array_equal(arr[key], z[key]), f"SC1 baseline/anchor alignment mismatch: {key}"
        assert sha_file(old) == old_sha, "R7 baseline changed during read"
        inputs[str(old)] = old_sha
    if not stages["stage1_fcmi.json"]["pass"]:
        assert set(f"FC{i}" for i in range(1, 7)) <= set(dropped), "SC2 failure must drop FC1-FC6"
    for x in G.SCREEN:
        assert stages[f"stageE_{x}.json"]["pass"] or x in dropped, f"ESC1 failure must drop {x}"
    probe = None
    include_nbc = False
    if step == "gate":
        probe_path = (root / "probe_trigger.json").resolve()
        probe = read_json(probe_path)
        assert probe.get("smoke") is smoke and probe.get("rule_sha256") == rule_sha
        assert probe.get("step") == "trigger" and isinstance(probe.get("triggered"), bool)
        for path, digest in probe["input_sha256"].items():
            assert sha_file(path) == digest, f"trigger input SHA-256 changed: {path}"
        include_nbc = probe["triggered"]
        inputs[str(probe_path)] = sha_file(probe_path)
    nbc_present = any(k.startswith(("R1_NBC__", "XAFF_NBC__")) for k in arr)
    assert not nbc_present or include_nbc, "NBC arrays before a checked trigger"
    required = (*(G.MAIN if include_nbc else G.MAIN[:-1]), *G.SCREEN, "R0own", "NRK", "TP")
    for x in required:
        assert x in groups or x in dropped, f"missing grouping metadata: {x}"
    stats = G.gate_stats(arr, dropped=dropped, include_nbc=include_nbc)
    if probe is not None:
        assert all(stats["trigger"][k] == probe[k] for k in stats["trigger"]), "frozen NBC trigger changed"
    payload = {**stats, "step": step, "smoke": smoke, "groupings": groups, "dropped": dropped,
               "stage_sha256": {Path(p).name: sha for p, sha in inputs.items() if Path(p).name.startswith("stage")},
               "probe_trigger_sha256": inputs.get(str((root / "probe_trigger.json").resolve())),
               "input_sha256": inputs}
    # Labels/AMIs never enter the gate. Descriptive report mechanics from rule section 7.
    payload["report_flags"] = {"AL": "ALADIN-ViT from StyleBabel (M10)", "group_counts": "final K in groupings",
                               "V_drops": {x: dropped[x] for x in ("V100", "V67", "V33") if x in dropped},
                               "FCMI": "unit shared-encoder representation; GPU training (M7)"}
    for path, digest in inputs.items():
        assert sha_file(path) == digest, f"input SHA-256 changed during gate evaluation: {path}"
    return common.write_json_once(pending_path, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("pending", "finalize"))
    p.add_argument("--step", choices=("trigger", "gate"), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--results", type=Path)
    p.add_argument("--index", type=Path, nargs="+", help="one or more immutable seed42 indices")
    p.add_argument("--groupings", type=Path, nargs="+", help="one or more immutable grouping metadata files")
    args = p.parse_args()
    import r8_common as common
    common.assert_rule()
    root = (args.results or (Path(common.SMOKE) / "gate" if args.smoke else common.RESULTS)).resolve()
    allowed = Path(common.SMOKE if args.smoke else common.RESULTS).resolve()
    assert root.is_relative_to(allowed), f"outputs must stay under {allowed}"
    if not args.smoke:
        assert not root.is_relative_to(Path(common.SMOKE).resolve()), "real verdict cannot use smoke results"
    if args.action == "pending":
        prepare_pending(root, args.step, args.smoke, args.index or [root / "seed42" / "index.json"],
                        args.groupings or [root / "groupings" / "r8_groupings.json"], common)
        print(f"{args.step}: pending written; run ticket 11 before finalize")
    else:
        pending = root / f"{args.step}_pending.json"
        rec = read_json(pending)
        assert rec["step"] == args.step and rec["smoke"] is args.smoke
        assert rec["input_sha256"].get(str(Path(common.RULE).resolve())) == sha_file(common.RULE), "pending decision rule changed"
        verdict = root / ("probe_trigger.json" if args.step == "trigger" else "gate_verdict.json")
        publish_checked(pending, root / "rederive" / f"{args.step}_rederive.json", verdict, common.write_json_once)
        print(f"{args.step}: C13 agrees; {verdict.name} written")


if __name__ == "__main__":
    main()
