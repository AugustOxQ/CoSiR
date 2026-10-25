"""The seven encoders of the round and their arm IDs (rule §2.1, §2.2, §7), and the rb_ maps extended to them."""
from dataclasses import dataclass

import eg_paths  # noqa: F401  (sys.path)


@dataclass(frozen=True)
class Encoder:
    store: str        # backbone folder under /data/SSD2/pre_extract/{artelingo_full,artelingo28}/
    checkpoint: str
    width: int        # projected image embedding width
    student: str      # K: R6-5 student on this encoder's default image
    own: str          # own PercepT arm: the matched arm M, or B1 for Qwen (rule §2.2)
    frozen: str       # F: the frozen default image
    new: bool         # extracted, trained and inferred in this round
    texts: bool       # the store holds text rows (spec D10)


# Stage ii's order (rule §2.2).
ENCODERS = (
    Encoder('qwen3vl_emb_2b', 'Qwen/Qwen3-VL-Embedding-2B', 2048, 'K4', 'B1', 'F4', False, True),
    Encoder('siglip2_so400m', 'google/siglip2-so400m-patch14-384', 1152, 'K2', 'M2', 'F2', False, True),
    Encoder('clip_b32', 'openai/clip-vit-base-patch32', 512, 'K3', 'M3', 'F3', False, True),
    Encoder('clip_b16', 'openai/clip-vit-base-patch16', 512, 'K6', 'M6', 'F6', True, True),
    Encoder('clip_l14', 'openai/clip-vit-large-patch14', 768, 'K7', 'M7', 'F7', True, True),
    Encoder('siglip_so400m', 'google/siglip-so400m-patch14-384', 1152, 'K8', 'M8', 'F8', True, True),
    Encoder('dinov2_large', 'facebook/dinov2-large', 1024, 'K5', 'M5', 'F5', True, False),
)
BY_STORE = {e.store: e for e in ENCODERS}
BY_STUDENT = {e.student: e for e in ENCODERS}
BY_OWN = {e.own: e for e in ENCODERS}
CANDIDATES = ('K6', 'K7', 'K8', 'K5')        # stage i's order (rule §2.1)
DEFERRAL_ORDER = ('K5', 'K8', 'K7', 'K6')    # deferral for time only, never dropped
NEW_STORES = tuple(e.store for e in ENCODERS if e.new)
STAGE_II = tuple(e.student for e in ENCODERS)   # K4, K2, K3, K6, K7, K8, K5
RECIPE = 'R6-5'


def _extend(table, new):
    """Add entries; an existing key must already hold the same value."""
    for key, value in new.items():
        assert table.get(key, value) == value, f'{key}: {table[key]} != {value}'
        table[key] = value


def extend_rb_maps():
    """Extend the rb_ modules' encoder and arm maps in this process to K5 to K8 and M5 to M8 (idempotent).

    Only adds entries; K1 to K5, M2, M3, M5, B1 and B6 keep their 12-24 values. Functions whose bodies hard-code the
    old lists (rb_score.arms_for, rb_score.output_path, rb_run, rb_a28.default_models, the CLIs) are not covered:
    the eg_ module that needs them copies them (code map §2).
    """
    import rb_a28
    import rb_encoders
    import rb_matched
    import rb_part2_report
    import rb_student
    import rb_v2
    new = [e for e in ENCODERS if e.new]
    _extend(rb_encoders.WIDTH, {e.store: e.width for e in ENCODERS})
    _extend(rb_student.ARM, {e.store: e.student for e in ENCODERS})
    _extend(rb_student.ENCODER, {e.student: e.store for e in ENCODERS})
    _extend(rb_matched.MATCHED, {e.own: e.store for e in new})
    _extend(rb_matched.CANDIDATE, {e.own: e.student for e in new})
    _extend(rb_a28.PERCEPT_FILE, {e.own: 'P3' for e in new})
    _extend(rb_a28.MATCHED, {e.student: e.own for e in ENCODERS})
    _extend(rb_a28.ENCODER, {e.student: e.store for e in ENCODERS})
    _extend(rb_part2_report.BACKBONE, {e.student: e.store for e in ENCODERS})
    _extend(rb_part2_report.FROZEN, {e.student: e.frozen for e in ENCODERS})
    _extend(rb_v2.MATCHED, {e.student: e.own for e in new})
