"""The round's arms, switches and orders (rule §2, §4)."""
BASE = 'R6-5'
TEACHER_WEIGHT = 0.25                    # R6-5's direct genre term weight
TAU = 0.05                               # the student's row temperature, always; the teacher's unless GS-tau
N_ANCHORS = 4096                         # GQ-8's teacher anchors (rule §2.1)

# Every arm: m (1 keeps the direct term at 0.25, 0 removes it), lam (relational weight), tau_t (teacher row
# temperature), anchors (0: in-batch rows; 4096: teacher anchors).
ARMS = {
    'R6-5': dict(m=1, lam=0, tau_t=TAU, anchors=0),
    **{f'GR-{lam}': dict(m=1, lam=lam, tau_t=TAU, anchors=0) for lam in (2, 4, 8, 16, 32, 64, 128, 256)},
    'GN-8': dict(m=0, lam=8, tau_t=TAU, anchors=0),
    'GS-0.02': dict(m=1, lam=8, tau_t=0.02, anchors=0),
    'GS-0.01': dict(m=1, lam=8, tau_t=0.01, anchors=0),
    'GQ-8': dict(m=1, lam=8, tau_t=TAU, anchors=N_ANCHORS),
}
LOSS_SCALE = 'GR-1'                      # the loss-scale run only (spec §3.6): lam = 1, deciding nothing
LOSS_SCALE_SWITCHES = dict(m=1, lam=1, tau_t=TAU, anchors=0)

# Stage A (rule §4.1)
LADDER = ('GR-8', 'GR-16', 'GR-32', 'GR-64', 'GR-128', 'GR-256')   # GR-8 frozen; 16 and 32 first; then the climb
A_FIRST = ('GR-16', 'GR-32')
CLIMB = ('GR-64', 'GR-128', 'GR-256')    # at most three more rungs, each double the last
A_OTHER = ('GN-8', 'GS-0.02', 'GS-0.01', 'GQ-8')
A_TIE_ORDER = LADDER + A_OTHER           # v*_A on equal A_genre: GR-8 first, then this order
# Stage B (rule §4.2)
FRESH_SEEDS = (1, 2, 3)
B_NEW = ('GR-2', 'GR-4')                 # trained on seeds 1 to 3, with v*_A when A-NEW
B_TIE_ORDER = ('GR-8', 'GR-2', 'GR-4')   # then v*_A
# Stage C (rule §2.3, §4.3)
FULL_SEEDS = (0, 1, 2, 3, 4)

T_G = 0.4989161752444698                 # the instructed Qwen genre image's AMI on P (12-25, frozen)
PROBE_GENRE = 0.504009                   # the ceiling probe on P, reported only
RANGES = {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}
X = RANGES['genre']                      # stage C's margin
G_HALF = 0.0263670169701253              # rule §3.2: from 12-24's reported_stage4.json; recomputed within 1e-6
TOL = 1e-6


def switches(arm):
    """The genre-loss switches of an arm (a copy)."""
    assert arm in ARMS, arm
    return dict(ARMS[arm])


def is_off(sw):
    """Every switch off: the recipe R6-5 itself."""
    return sw == ARMS[BASE]


def eg_form(sw):
    """12-25's switches {m, lam, w} when the arm is one 12-25's trainer computes (R6-5, GR-lam), else None."""
    if sw['m'] == 1 and sw['tau_t'] == TAU and sw['anchors'] == 0:
        return dict(m=1, lam=sw['lam'], w=0)
    return None


assert all(a in ARMS for a in A_TIE_ORDER + B_NEW + B_TIE_ORDER)
assert eg_form(ARMS['GR-8']) == dict(m=1, lam=8, w=0) and eg_form(ARMS[BASE]) == dict(m=1, lam=0, w=0)
