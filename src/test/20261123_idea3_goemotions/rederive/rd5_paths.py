"""Round 5 re-derivation: paths, SHA-256 constants (from the rules' text), hashing and the allowed imports.

Independence: this module imports only what DECISION_RULE.md §8 "What the re-derivation may import" lists. Every
imported module file is hashed and recorded (`module_shas`); files whose SHA-256 the rules state are asserted.
"""
import hashlib
import sys
import time
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                      # .../20261123_idea3_goemotions/rederive
R5 = HERE.parent
ROOT = R5.parents[2]
T = ROOT / "src/test"
RESULTS = HERE / "results"

# ---------------------------------------------------------------- folders of the allowed modules
GONOGO = T / "20261101_aspect_factor_gonogo"
QUICK = T / "20261108_new_method_quick_checks"
ORACLE = T / "20261111_community_told_oracle"
ROUND1 = T / "20261117_reader_fix_csd"
for p in (ROOT, GONOGO, QUICK, ORACLE, ROUND1):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------- SHA-256s stated by the rules
RULES = {
    R5 / "DECISION_RULE.md": "19e59fc7220c05b630f4773a94578aa3858d29853dbf7d438455e1ee973d735e",
    T / "20261122_round4_aff_vetoes/DECISION_RULE.md": "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b",
    T / "20261121_round3_affect_gate/DECISION_RULE.md": "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925",
}
P = {
    "seed42_arrays": T / "20261122_round4_aff_vetoes/results/seed42_arrays.npz",
    "cand_rc": ROUND1 / "results/cand_Rc_Rb_expected_A0.npz",
    "rc_tau": ROUND1 / "results/rc_tau.json",
    "reader_pkl": ROUND1 / "results/rb_reader_A0.pkl",
    "reader_json": ROUND1 / "results/rb_reader_A0.json",
    "per_anchor_seed42": T / "20261030_aspect_baselines/results/per_anchor_seed42.npz",
    "episodes_seed42": T / "20261030_aspect_baselines/results/episodes_seed42.npz",
    "baselines_seed42": T / "20261030_aspect_baselines/results/baselines_seed42.json",
    "told_npz": ORACLE / "results/per_anchor_told_oracle.npz",
    "told_json": ORACLE / "results/told_oracle.json",
    "partitions": T / "20261031_pseudo_partitions/results/partitions.npz",
    "n6_posteriors": QUICK / "results/n6_posteriors.npz",
    "a3_ckpt": GONOGO / "checkpoints/A3_seed42.pt",
    "affect_npz": T / "20261018_affect_factor_learning/cache/affect_prepare.npz",
    "affect_json": T / "20261018_affect_factor_learning/cache/affect_prepare.json",
    "annotations": Path("/data/PDD/artelingo/artelingo_train.json"),
    "src_affect": ROOT / "src/data/affect.py",
    "src_artelingo": ROOT / "src/data/artelingo.py",
    "src_splits": ROOT / "src/data/artelingo_splits.py",
    "mod_told_oracle": ORACLE / "run_told_oracle.py",
    "mod_run_checks": QUICK / "run_checks.py",
    "mod_run_n6": QUICK / "run_n6.py",
    "mod_run_gonogo": GONOGO / "run_gonogo.py",
    "mod_rb_build": ROUND1 / "rb_build.py",
    "mod_common": ROUND1 / "common.py",
}
SHA = {
    "seed42_arrays": "72fb827fa9b44360d237dd3a3d555403825cf3847282976d24073dcc12ee0c75",
    "cand_rc": "628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981",
    "rc_tau": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    "reader_pkl": "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "reader_json": "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    "per_anchor_seed42": "a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d",
    "episodes_seed42": "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    "baselines_seed42": "ce42c81e8eec256496454e88fc07dc4fcfb02e2d5f1b043f2274ba6008564ce6",
    "told_npz": "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    "told_json": "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2",
    "partitions": "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "n6_posteriors": "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "a3_ckpt": "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    "affect_npz": "e25d2dcadadc23b33ac94659fb44c13e220110e64ce463def07b04343bfc4f2e",
    "affect_json": "8734bdac0ec49b07ddd53caa5aa088dd4d51d0daba9c50730b359c98395316d1",
    "annotations": "6a4e5b17feecc2c3b54cd166416b75f1bbffd56bec1523f4173b245edb8e894d",
    "src_affect": "d37e306c9b8673068f6d74e74d7eff61f649ebdb53088f0d8ad7f592e17f3d33",
    "src_artelingo": "623b7b02eb03b7a81ecf246b120bb7f49e1b929194830faeebcd0d64fa3c89e1",
    "src_splits": "f130950355487b8537ba0c0b1e35230a08e2e07d3d0fddd4c3e5aad240886c13",
    "mod_told_oracle": "b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d",
    "mod_run_checks": "5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610",
    "mod_run_n6": "57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0",
    "mod_run_gonogo": "8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc",
    "mod_rb_build": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "mod_common": "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
}
# Library modules this code imports from (no SHA-256 stated in the rules; recorded, not asserted).
RECORDED = {
    "src/eval/aspect_quick_checks.py": ROOT / "src/eval/aspect_quick_checks.py",
    "src/eval/aspect_metrics.py": ROOT / "src/eval/aspect_metrics.py",
    "src/model/aspect_rule.py": ROOT / "src/model/aspect_rule.py",
}
# The GoEmotions model (D12), used by the stage-B spot check only.
HF_SNAPSHOT = Path("/data/SSD2/HF_home/hub/models--SamLowe--roberta-base-go_emotions")
HF_REV = "d75048347613a25d77de8cf6412eaae9fa7b26be"
HF_FILES = {
    "model.safetensors": "84d6d338b4cf63f0ed3c990a0ce748d32d1d2965c072f4645accaa71af3888c0",
    "config.json": "3d4ef8e1465958e169761e2eb09d6e2c8d8806216973691ac40e405c97339d5c",
    "tokenizer.json": "90e2336a1cdacffe5d4328ab323aa9e5c33889026e4e4881323bebdeeb0e179d",
    "tokenizer_config.json": "6735f2f38dc5399eb76a2c20dcba3ef27a9b2fbba0d05b6e2966038f28aefcf9",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
    "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
}
# This round's own files (stage B), their SHA-256 read from the GoEmotions record and the run-log line.
GE_FILE = R5 / "cache/r5_goemotions_selection.npz"
GE_RECORD = R5 / "cache/r5_goemotions_selection.json"
RUN_LOG = R5 / "20261123_idea3_goemotions_log.md"


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now_ams() -> str:
    """Amsterdam local time, 'YYYY-MM-DD HH:MM'."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Amsterdam"
    time.tzset()
    out = time.strftime("%Y-%m-%d %H:%M")
    if old is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = old
    time.tzset()
    return out


def assert_inputs(keys=None) -> dict:
    """Assert the rules' SHA-256s and those of the stored inputs this code reads; return them."""
    out = {}
    for path, want in RULES.items():
        got = sha_file(path)
        if got != want:
            raise SystemExit(f"{path}: SHA-256 {got} != {want}")
        out[str(path.relative_to(ROOT))] = got
    for k in (keys or SHA):
        got = sha_file(P[k])
        if got != SHA[k]:
            raise SystemExit(f"{P[k]}: SHA-256 {got} != {SHA[k]}")
        out[k] = got
    return out


def module_shas() -> dict:
    """SHA-256 of every imported repository module file (allowed list) and of the library files used."""
    out = {k: sha_file(P[k]) for k in SHA if k.startswith("mod_") or k.startswith("src_")}
    out.update({k: sha_file(v) for k, v in RECORDED.items()})
    return out


# ---------------------------------------------------------------- the allowed imports (rule §8), nothing else
def allowed():
    """Import the allowed functions lazily (they load data and torch)."""
    import run_gonogo                                            # EvalContext
    import run_checks                                            # model_inputs
    import run_n6                                                # load_posteriors
    import run_told_oracle                                       # fit_one_head (CLIP heads only)
    import rb_build                                              # load_readers("A0", False)
    from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores
    from src.eval.aspect_metrics import cluster_bootstrap
    from src.model.aspect_rule import zscore_rows
    from src.data.artelingo_splits import artelingo_splits
    return {"EvalContext": run_gonogo.EvalContext, "model_inputs": run_checks.model_inputs,
            "load_posteriors": run_n6.load_posteriors, "fit_one_head": run_told_oracle.fit_one_head,
            "load_readers": rb_build.load_readers, "centered_term": centered_term,
            "crossfit_condition_free": crossfit_condition_free, "uniform_probe_scores": uniform_probe_scores,
            "cluster_bootstrap": cluster_bootstrap, "zscore_rows": zscore_rows, "artelingo_splits": artelingo_splits}


def load_bundle_round1():
    """Round 1's common.load_bundle (seed 42 only; allowed by rule §8)."""
    import common
    return common.load_bundle()
