"""r19 fix wave A: which inputs a stage 2 scorer may skip, from the committed `records/status.json`.

Spec §3.5 (voids table): a failed path voids only its own cells and the others go on. A scorer therefore writes every cell its
inputs allow, and drops a cell only when the committed status says the run or gate behind it failed. The mapping is
`pp_outcome.Voids`'s own rules (mode 'matched': SC3, SC9, SC13; mode 'teacher': SC6, SC10, SC14), read from there, so the two
cannot drift. An input that is absent while the status says nothing still raises: a bug stays loud.

  committed_status()          the status record as stage 2 reads it (`pp_paths.read_committed_record`: refuses unless equal
                              to git HEAD). Outside smoke a missing file raises. Under smoke a missing file means "no status":
                              None, which voids nothing (smoke and tests that run a scorer alone).
  matched_void(status, enc, kind)   True when row (v) voids the matched arm's cells of `enc` on `kind` (runs.<enc>.matched or
                              gate (v) for `enc` failed; kind 'artelingo' | 'mmart' | 'semart').
  teacher_void(status, enc, kind)   True when a failed gate voids the teacher cells of `enc` on `kind` (gates.i_semart_genre_word:
                              SC14 of every encoder).
"""
import pp_paths as PP
import pp_outcome


def committed_status():
    """The committed status, checked (`pp_outcome.check_status`); None only under smoke when no status file exists."""
    if PP.SMOKE and not PP.record_path('status.json').exists():
        return None
    return pp_outcome.check_status(PP.read_committed_record('status.json'))


def resolve(status):
    """`status` as given (a dict, checked) or, when None, the committed one (`committed_status`)."""
    return committed_status() if status is None else pp_outcome.check_status(status)


def _voided(status, mode, enc, kind):
    if status is None:
        return False
    return any(m == mode and (kind, enc) in scope for _, _, scope, m in pp_outcome.Voids(status).rules)


def matched_void(status, enc, kind):
    """The matched arm of `enc` on `kind` is voided by the status (row (v)): its cells SC3, SC9 or SC13 are not written."""
    assert kind in ('artelingo', 'mmart', 'semart'), kind
    return _voided(status, 'matched', enc, kind)


def teacher_void(status, enc, kind='semart'):
    """The teacher cells of `enc` on `kind` are voided by the status (row (i), genre_word): SC14 is not written."""
    assert kind in ('artelingo', 'photos', 'mmart', 'semart'), kind
    return _voided(status, 'teacher', enc, kind)
