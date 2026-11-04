"""Record-writing CLI guard: OM_OUTPUTS and OM_SMOKE=1 must be used together, away from DEFAULT_OUT."""
import os
from pathlib import Path

import om_paths as op


def check_record_environment():
    """Refuse mixed real/smoke environments before any CLI writes a record."""
    outputs = os.environ.get('OM_OUTPUTS')
    smoke = os.environ.get('OM_SMOKE') == '1'
    if outputs is not None and not smoke:
        raise SystemExit('OM_OUTPUTS requires OM_SMOKE=1 for record-writing commands')
    if smoke and (not outputs or Path(outputs).resolve() == op.DEFAULT_OUT.resolve()):
        raise SystemExit('OM_SMOKE=1 requires OM_OUTPUTS at a scratch path different from DEFAULT_OUT')
