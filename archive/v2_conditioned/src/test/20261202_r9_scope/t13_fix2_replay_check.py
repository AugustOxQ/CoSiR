"""Read-only N1/N2 check on real CLIP inputs, without scoring or writing receipts.

The current stage0b certificate predates r9_pins.py. Report certificate
mismatches separately, then isolate the input/provenance checks as in the
re-review. This is not a certified stage 2 pass or the CLIP dry run.
"""
import json
from pathlib import Path
from unittest.mock import patch

import run_r9_stage2 as D


def main():
    replay = Path('/project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32')
    gate = D.read_json('/project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json')
    modules = D.recipe_module_shas()
    mismatches = [module for module, sha in modules.items()
                  if gate.get('module_sha256', {}).get(module) != sha]
    paths = {**D.BB.CLIP_PARTS, **D.S0B.replay_files(replay)}
    inputs = {role: D._input_files(D.read_json(str(paths[role]) + '.record.json'), paths, replay=True)
              for role in sorted(D.BB.RECIPE_ROLES)}
    # Only this diagnostic bypasses module certification, explicitly reported
    # below. Producer scripts and all frozen, array and input checks still run.
    with patch.object(D, 'assert_modules', return_value=modules):
        prov = D.provenance('clip_b32', paths, gate, replay=True)
    assert len(inputs) == len(prov) == 20
    assert set(inputs) == set(prov) == D.BB.RECIPE_ROLES
    print(json.dumps({'replay': str(replay), 'input_files_passed': len(inputs),
                      'provenance_passed': len(prov), 'roles': sorted(prov),
                      'skipped_self_input_sha256': {r: p['skipped_self_input_sha256']
                                                    for r, p in prov.items() if p['skipped_self_input_sha256']},
                      'module_certificate_mismatches': mismatches,
                      'module_check_isolated_for_read_only_diagnostic': True}, indent=2))


if __name__ == '__main__':
    main()
