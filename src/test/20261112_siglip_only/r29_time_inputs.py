"""K* reading and caption-form helpers for r29_time.py. They live apart so they are CPU-testable without r28's timing
imports: r28_time imports r19's pp_paths, whose guard refuses once r14's bs_paths is loaded (as in a whole-suite run)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402

PICK = P.RECORDS / 'prefix_pick.json'
FORMS = {'siglip1': 'plain', 'siglip1_prefix': 'prefixed'}       # K*'s arm -> its caption form


def read_kstar(path=PICK):
    """K* and its caption form from `records/prefix_pick.json` (ticket 05), which must be committed unchanged (spec §3.5 order
    rule; the other K* readers refuse the same way). Returns (kstar, kstar_form); the keys and their consistency are asserted."""
    path = Path(path)
    P.require_committed(path, 'Table 8 times K*; the pick is written and committed before it (spec §3.4, §3.5)')
    doc = json.loads(path.read_text())
    assert {'kstar', 'kstar_form', 'p1_form', 'flag'} <= doc.keys(), f'{path}: missing shared interface keys ({sorted(doc)})'
    assert doc['kstar'] in FORMS, f'{path}: kstar {doc["kstar"]!r} not in {sorted(FORMS)}'
    assert doc['kstar_form'] == FORMS[doc['kstar']], f'{path}: kstar_form {doc["kstar_form"]!r} does not match {doc["kstar"]!r}'
    return doc['kstar'], doc['kstar_form']


def caption_texts(captions, form):
    """The texts the SigLIP 1 text pass reads, in K*'s caption form. Prefixed: exactly as ticket 02's prefixed store builds them
    (`r29_extract._encode_chunk`: `prefix + caption`, one concatenation, before the encoder's own lowercasing)."""
    assert form in ('plain', 'prefixed'), form
    captions = list(captions)
    return captions if form == 'plain' else [P.PREFIX + caption for caption in captions]
