"""Round 5 shared constants and helpers (rule DECISION_RULE.md of this folder: D12, §4 item 3, §8 T1, §10 list A item 1).

Imports round 4's modules by path (r4_common, r4_bundle, r4_stats; round 4's folder goes on sys.path, and r4_common
brings round 3's, round 1's and round 2's in) and the quick-check, told-oracle and go/no-go modules through round 3's
by-path loader, and checks each resolved to the file of the earlier round. On import it sets R3.TEST_SEEDS to this
round's fresh seeds (rule §4 item 3), which round 3's seed guard reads at call time, and asserts round 4's equal.
No round-5 module may share a name with a module of rounds 1 to 4 (prefix r5_, rule §10).

GOEMO_FILE_SHA and GE_POST_SHA are None until one-line commits set them (rule D2 item 5, D4); r5_guard refuses a GE
posterior file while GE_POST_SHA is None.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST = ROOT / "src/test"
R4_DIR = TEST / "20261122_round4_aff_vetoes"
R3_DIR = TEST / "20261121_round3_affect_gate"
CACHE = HERE / "cache"
RESULTS = HERE / "results"
SMOKE = RESULTS / "smoke"
for p in (str(R4_DIR), str(R3_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_common as R4C  # noqa: E402  (puts rounds 1 to 3 and the repo root on sys.path; sets R3.TEST_SEEDS)
import r4_bundle as RB4  # noqa: E402
import r4_stats as RS4  # noqa: E402

R3 = R4C.R3
RB3, RF3, RS3 = R4C.RB3, R4C.RF3, R4C.RS3
C = R3.C
rc_core = R3.K
_M = RB3.modules()                       # run_checks, run_n6, run_told_oracle, run_gonogo, each one shared instance
RCHK, N6, RTO, RG = _M.rc, _M.n6, _M.rto, _M.rg

for _m in (R4C, RB4, RS4):
    if Path(_m.__file__).resolve().parent != R4_DIR:
        raise ImportError(f"'{_m.__name__}' resolved to {_m.__file__}, not round 4's")
for _m in (R3, RB3, RF3, RS3):
    if Path(_m.__file__).resolve().parent != R3_DIR:
        raise ImportError(f"'{_m.__name__}' resolved to {_m.__file__}, not round 3's")

now_ams = R3.now_ams
git_head = R3.git_head
sha256_file = R3.sha_file

TEST_SEEDS = (52, 53, 54)                                  # rule §4 item 3
SMOKE_SEEDS = (9001, 9002, 9003)                           # rule §10
EARLIER_SEEDS = (42, 43, 45, 47, 48, 49, 50, 51)           # rule §6.2 hash check
R3.TEST_SEEDS = TEST_SEEDS                                 # round 3's seed guard reads this at call time
if tuple(R4C.TEST_SEEDS) != TEST_SEEDS:
    raise ImportError("round 4's TEST_SEEDS differs from this round's")
if tuple(R3.SMOKE_SEEDS) != SMOKE_SEEDS:
    raise ImportError("round 3's SMOKE_SEEDS differs from this round's")

RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "19e59fc7220c05b630f4773a94578aa3858d29853dbf7d438455e1ee973d735e"

# ---------------------------------------------------------------- constants
A0 = R3.A0
CANDIDATES = ("G-T", "G-TF")
TIE_BAND_UNITS = 24                                        # rule §5 item 7
GOEMO_BATCH = 256                                          # rule D2 item 2
GOEMO_MAXLEN = 64
REG_SAMPLE = (5, 2048)                                     # rule D3: (rng seed, sample size) of the scorer-train sample
SPOT_SAMPLE = (6, 1024)                                    # rule §8: the re-derivation's spot check of selection rows
GOEMO_TOL = 1e-4                                           # rule D3
GE_MAX_ITER = 300                                          # rule D4 (the recipe's own cap)
GE_FALLBACK_MAX_ITER = 3000
N_ROWS = 308_723                                           # rows of the whole dataset (rule D4)
N_SELECTION = 32_413                                       # selection rows (rule §1)
N_CLASSES = 41                                             # partition_L communities
N_GOEMO = 28

GOEMO_FILE_SHA = "f8372a89a808421772e19cce9413dbab9b83dcef5cfd28da8232d77d729e18a8"  # SHA-256 of cache/r5_goemotions_selection.npz; set by a one-line commit after D2's run
GE_POST_SHA = None       # SHA-256 of cache/r5_ge_posterior.npz; set by a one-line commit after D4's run

# rule §5 item 3: told_oracle.json arm L's CLIP-head record (the placement function must reproduce it)
CLIP_HEAD_RECORD = {
    "n_classes": 41,
    "draw_rows_sha256": "7be956c09bf716547df20264435388bd3645ae963df728636774e80359cdef5c",
    "heldout_accuracy": {"img": 9.81, "txt": 35.72},
    "check_majority_share": 6.41,
    "uniform": 2.4390243902439024,
}
CLIP_N_ITER = {"txt": 176, "img": 154}                     # descriptive (rule §5 item 3)
PAIR_RATIOS = {"ratio_same_over_diff": 1.1445184466303795, "emotionxstyle": 1.070827615209179,
               "emotionxgenre": 1.0375160447836334}        # rule §5 item 4, pair_stats_heads (affect, CLIP)
AUC_AFF = 0.7870951145887375                               # rule §5 item 4 and diagnostic (a): AFF's detection AUC
GROUP_LIFT = 2.7111312041209863                            # rule diagnostic (c): told_oracle.json arms.L.pairs.groups.lift

# rule §5 item 1: AFF as round 3 tested it (seed 42, full precision); same keys as round 3's AFF_BRAINSTORM
AFF_CELLS = {"fused": (39, 119), "cf": (149, 10), "sigma": (0.0, 0.0)}
AFF_ITEM1 = {
    "fused_r1": 19.136555989583336, "cf_r1": 18.39599609375, "comparator": "B_prime",
    "bar": (0.6998697916666667, 0.4598852740816973, 0.9371680126852968),
    "margin": (0.7405598958333333, 0.5196896694963071, 0.9598857494832738),
    "gain_statistic": (3.110758463541667, 2.780005709854805, 3.4559584315470384),
    "either": -1.629638671875,
    "per_pair_bar": {"emotion__style": 0.9765625, "emotion__genre": 1.45263671875, "style__genre": -0.32958984375},
    "aff_minus_r1_fused": (0.21769205729166666, 0.06425880757348419, 0.3709597330984391),
    "aff_minus_r1_bar": (0.25634765625, 0.04280778303598444, 0.46195041633015954),
    "open_tau0_counts": {"a": 9941, "b": 3627},
}
BPRIME_A1_MEAN_42 = R4C.BPA1_MEAN_42                       # 18.804931640625
# rule §5 item 1 (round 4's dev_seed42.json): AFF minus B'(A1), point and 95% interval
AFF_MINUS_BP1 = (0.33162434895833337, (0.048231414333532084, 0.6246158772581268))
AFF_EITHER_PER_GAIN = 0.5238718116415958                   # rule diagnostic (d): 1.629638671875 / 3.110758463541667

# ---------------------------------------------------------------- D12 inputs
SNAP = ("/data/SSD2/HF_home/hub/models--SamLowe--roberta-base-go_emotions/snapshots/"
        "d75048347613a25d77de8cf6412eaae9fa7b26be/")
# rule D12 (this round's additional inputs): path under the repository root (or absolute) -> SHA-256. Earlier rounds'
# tables stay in R3.INPUTS / R4C.INPUTS (paths relative to src/test/) and are checked through them.
INPUTS = {
    "docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md": "bc7384027d856102ab5efc8eb24a1a7831a4ce33e660f356a66628a2dad184ee",
    "src/test/20261122_round4_aff_vetoes/DECISION_RULE.md": "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b",
    "src/test/20261122_round4_aff_vetoes/r4_common.py": "d1d4b819868aaa21603438fc7121151cd06639ec0318b610959340a989518fbb",
    "src/test/20261122_round4_aff_vetoes/r4_bundle.py": "2fc452800a14768bbb81efbb0728113ce10aa4f281a4c5910252eb5c4bd634e2",
    "src/test/20261122_round4_aff_vetoes/r4_stats.py": "b308ca43afc3e8accd35894a3c555f551c5c17db0e90d330bab8669b68f2d271",
    "src/test/20261122_round4_aff_vetoes/r4_fusion.py": "9b9b12e388dd41a01db5c6aea4277af4710d4cd4990adfefc918b3e0ccbd961f",
    "src/test/20261122_round4_aff_vetoes/run_r4_seed42.py": "74f6ebe0a97f83322fd2858ee61a95ac9033edd33693bea192db1c6de6149eda",
    "src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz": "72fb827fa9b44360d237dd3a3d555403825cf3847282976d24073dcc12ee0c75",
    "src/test/20261122_round4_aff_vetoes/results/dev_seed42.json": "fd7b3f480d5997284f9d8cfe29fe25275ff3edd38c1ada0a9392d8396ba9e01f",
    "src/test/20261122_round4_aff_vetoes/.gitignore": "65c8bc8600f811509b0d8325f4aec4d53bd0c35639e6400c460e2445bddbdbc9",
    "src/test/20261122_round4_aff_vetoes/rule_check/opus_rule_check.md": "8c550feae32f30b954531abc563bf21cc486be7d094b38d30bc55aa3782e113c",
    "src/test/20261122_round4_aff_vetoes/final_review/final_review.md": "5c68260241011d9e2d8137bf8d5069f6f32f50b9c30a024f8f64573df400224b",
    "docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md": "53ecffe87f9f8927f46c8755bd44053d6aa34feb34bf372c886bec52b68d5334",
    "docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md": "0aeb88506c6aa988a82776bd9aea5261e544a92f4b203ebc0e056142ce72fac2",
    "src/data/affect.py": "d37e306c9b8673068f6d74e74d7eff61f649ebdb53088f0d8ad7f592e17f3d33",
    "src/data/artelingo.py": "623b7b02eb03b7a81ecf246b120bb7f49e1b929194830faeebcd0d64fa3c89e1",
    "src/data/artelingo_splits.py": "f130950355487b8537ba0c0b1e35230a08e2e07d3d0fddd4c3e5aad240886c13",
    "/data/PDD/artelingo/artelingo_train.json": "6a4e5b17feecc2c3b54cd166416b75f1bbffd56bec1523f4173b245edb8e894d",
    SNAP + "model.safetensors": "84d6d338b4cf63f0ed3c990a0ce748d32d1d2965c072f4645accaa71af3888c0",
    SNAP + "config.json": "3d4ef8e1465958e169761e2eb09d6e2c8d8806216973691ac40e405c97339d5c",
    SNAP + "tokenizer.json": "90e2336a1cdacffe5d4328ab323aa9e5c33889026e4e4881323bebdeeb0e179d",
    SNAP + "tokenizer_config.json": "6735f2f38dc5399eb76a2c20dcba3ef27a9b2fbba0d05b6e2966038f28aefcf9",
    SNAP + "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    SNAP + "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
    SNAP + "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
    "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz": "e25d2dcadadc23b33ac94659fb44c13e220110e64ce463def07b04343bfc4f2e",
    "src/test/20261018_affect_factor_learning/cache/affect_prepare.json": "8734bdac0ec49b07ddd53caa5aa088dd4d51d0daba9c50730b359c98395316d1",
    "src/test/20261018_affect_factor_learning/run_affect.py": "ef04c7c24810d7550a791c8191162de0664d241d5414450f849e28a847f88c65",
    "src/test/20261018_affect_factor_learning/run_posthoc_affect.py": "07da2767706964f36cdf0ca282e2298b61414e6e86ed1afbaaa18862a0fb12d8",
    "src/test/20261111_community_told_oracle/run_told_oracle.py": "b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d",
    "src/test/20261108_new_method_quick_checks/run_checks.py": "5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610",
    "src/test/20261108_new_method_quick_checks/run_n6.py": "57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0",
    "src/test/20261101_aspect_factor_gonogo/run_gonogo.py": "8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc",
    "src/data/wikiart_genre.py": "aef1d35978305b9bc00f84897a4d8a999b729dac8bc85a3237dc01b5e00c5448",
    "src/test/20261120_r1_levers_brainstorm/bs_09_direction.py": "74f3fa4c88588832cd681b1cbb793b7bf6982748477df9147acdf6141346de1f",
    "src/test/20261120_r1_levers_brainstorm/results/bs_09_direction.json": "0882262340518347d59e825f9e5444d5042f75698d2b9668f36eac2163e7fc8e",
    "src/test/20261120_r1_levers_brainstorm/bs_cache.py": "62b6d9baa534a04a7fc5365e7a2fea8f47e8719ec29890d9cafe7ae24fd3b07e",
    "src/test/20261118_reader_fix_round2/results/cand_R1_A0.npz": "c707af6a101bf3c0e49ee6ce509cf31d82bfa54ee33597dbece2ec6261941407",
}
_CHECKED = {}

# rule §8 T1: every earlier-round module file this round imports (and the round-1/round-2 modules they pull in), path
# relative to src/test/ -> SHA-256. Values are those of round 3's D15 table (R3.INPUTS), round 4's D11 table
# (R4C.INPUTS) or this rule's D12 (INPUTS); assert_modules cross-checks every value against whichever list it.
MODULE_SHA = {
    "20261117_reader_fix_csd/common.py": "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
    "20261117_reader_fix_csd/rc_core.py": "e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c",
    "20261117_reader_fix_csd/rb_build.py": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "20261117_reader_fix_csd/rb_eval.py": "1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14",
    "20261117_reader_fix_csd/rb_features.py": "1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf",
    "20261118_reader_fix_round2/r2_fusion.py": "1406e8e71629e74c468f8b009a110e49aa05c481ed52cc1b8d99b59f87016027",
    "20261121_round3_affect_gate/r3_common.py": "b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86",
    "20261121_round3_affect_gate/r3_bundle.py": "bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d",
    "20261121_round3_affect_gate/r3_fusion.py": "ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637",
    "20261121_round3_affect_gate/r3_stats.py": "846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d",
    "20261122_round4_aff_vetoes/r4_common.py": "d1d4b819868aaa21603438fc7121151cd06639ec0318b610959340a989518fbb",
    "20261122_round4_aff_vetoes/r4_bundle.py": "2fc452800a14768bbb81efbb0728113ce10aa4f281a4c5910252eb5c4bd634e2",
    "20261122_round4_aff_vetoes/r4_stats.py": "b308ca43afc3e8accd35894a3c555f551c5c17db0e90d330bab8669b68f2d271",
    "20261111_community_told_oracle/run_told_oracle.py": "b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d",
    "20261108_new_method_quick_checks/run_checks.py": "5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610",
    "20261108_new_method_quick_checks/run_n6.py": "57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0",
    "20261101_aspect_factor_gonogo/run_gonogo.py": "8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc",
}
_MODULE_OBJECTS = {   # module name -> (module, path relative to src/test/): the loaded file must be the hashed one
    "common": (C, "20261117_reader_fix_csd/common.py"), "rc_core": (rc_core, "20261117_reader_fix_csd/rc_core.py"),
    "rb_build": (R3.rb, "20261117_reader_fix_csd/rb_build.py"), "rb_eval": (R3.rbe, "20261117_reader_fix_csd/rb_eval.py"),
    "rb_features": (R3.rf, "20261117_reader_fix_csd/rb_features.py"),
    "r2_fusion": (R3.F, "20261118_reader_fix_round2/r2_fusion.py"),
    "r3_common": (R3, "20261121_round3_affect_gate/r3_common.py"),
    "r3_bundle": (RB3, "20261121_round3_affect_gate/r3_bundle.py"),
    "r3_fusion": (RF3, "20261121_round3_affect_gate/r3_fusion.py"),
    "r3_stats": (RS3, "20261121_round3_affect_gate/r3_stats.py"),
    "r4_common": (R4C, "20261122_round4_aff_vetoes/r4_common.py"),
    "r4_bundle": (RB4, "20261122_round4_aff_vetoes/r4_bundle.py"),
    "r4_stats": (RS4, "20261122_round4_aff_vetoes/r4_stats.py"),
    "run_told_oracle": (RTO, "20261111_community_told_oracle/run_told_oracle.py"),
    "run_checks": (RCHK, "20261108_new_method_quick_checks/run_checks.py"),
    "run_n6": (N6, "20261108_new_method_quick_checks/run_n6.py"),
    "run_gonogo": (RG, "20261101_aspect_factor_gonogo/run_gonogo.py"),
}


def assert_rule():
    """Every script calls this first: this rule's, round 4's and round 3's rule SHA-256 must be the committed ones."""
    got = sha256_file(RULE)
    if got != RULE_SHA:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA}")
    R4C.assert_rule()          # round 4's rule, then round 3's


def input_path(name) -> Path:
    if name in INPUTS:
        return Path(name) if name.startswith("/") else ROOT / name
    raise KeyError(f"{name} is not a D12 input of this round")


def assert_inputs(names) -> dict:
    """SHA-256 of each named input; a mismatch stops the run. Names are this round's D12 keys (repo-root-relative or
    absolute) or earlier rounds' tables' keys (relative to src/test/, checked through R4C.assert_inputs)."""
    names = list(names)
    own = [n for n in names if n in INPUTS]
    out = dict(R4C.assert_inputs([n for n in names if n not in INPUTS]))   # KeyError for a name no table lists
    for n in own:
        if n not in _CHECKED:
            got = sha256_file(input_path(n))
            if got != INPUTS[n]:
                raise SystemExit(f"input {n}: SHA-256 {got} differs from the rule's {INPUTS[n]}")
            _CHECKED[n] = got
        out[n] = _CHECKED[n]
    return out


def assert_modules() -> dict:
    """Rule §8 T1: the SHA-256 of every imported module file of rounds 1 to 4 equals MODULE_SHA, MODULE_SHA agrees with
    every table that lists the file, and each module object in this process was loaded from that very file. Every
    runner calls this before its first bundle call; a mismatch stops the run."""
    for rel, sha in MODULE_SHA.items():
        for table, key in ((R3.INPUTS, rel), (R4C.INPUTS, rel), (INPUTS, "src/test/" + rel)):
            if key in table and table[key] != sha:
                raise SystemExit(f"module {rel}: MODULE_SHA {sha} differs from the rule table's {table[key]}")
    for name, (mod, rel) in _MODULE_OBJECTS.items():
        if Path(mod.__file__).resolve() != (TEST / rel).resolve():
            raise SystemExit(f"module {name} was loaded from {mod.__file__}, not {rel}")
        if rel not in MODULE_SHA:
            raise SystemExit(f"module {name}: {rel} has no MODULE_SHA entry")
    out = {}
    for rel, sha in MODULE_SHA.items():
        got = sha256_file(TEST / rel)
        if got != sha:
            raise SystemExit(f"module {rel}: SHA-256 {got} differs from the rule's {sha}")
        out[rel] = got
    return out


def check_seed(seed, smoke):
    """Rule §4 item 3: admits 42, 52, 53, 54 (smoke False) and 9001 to 9003 (smoke True); refuses everything else."""
    if tuple(R3.TEST_SEEDS) != TEST_SEEDS:
        raise ValueError("round 3's seed guard does not hold this round's seeds")
    RB4._check_seed(int(seed), bool(smoke))


def res_dir(smoke) -> Path:
    d = RESULTS / "smoke" if smoke else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance(smoke) -> dict:
    return {"rule_sha256": RULE_SHA, "written": now_ams(), "smoke": bool(smoke), "git_head": git_head()}


def refuse_existing(paths, smoke):
    """Non-smoke outputs are never overwritten."""
    if smoke:
        return
    busy = [str(p) for p in paths if Path(p).exists()]
    if busy:
        raise SystemExit(f"outputs exist ({', '.join(busy)}); refusing to overwrite")


def write_json_once(path, rec, smoke):
    """A results JSON with this round's provenance (non-smoke: never overwritten; must be finite)."""
    path = Path(path)
    refuse_existing([path], smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**C.jsonable(rec), "provenance": provenance(smoke)}
    C.assert_finite_tree(rec)
    path.write_text(json.dumps(rec, indent=1))
    return rec
