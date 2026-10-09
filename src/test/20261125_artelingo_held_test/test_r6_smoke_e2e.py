"""The full local smoke of run_r6_smoke.py with crafted GPU outputs (ticket 15; rule section 6 item 7): no GPU, real
selection rows, smoke seeds 9001 to 9003 at 64 episodes per pair.

The run's own setup() runs on the real data (heads refit and checked bit for bit, PM fits, readers); refit_check.json
is written from it by run_r6_refit.record (the real record); the other seed-42 records (picks, regression,
sensitivity, the DTS stages) are stand-ins with the current module SHA-256s (test_r6_held.write_records,
test_r6_smoke.write_dts42), since the real ones are the run chat's. The three stages run in order: gpu-inputs (the held
smoke and the round-1 job folders written by r6_gpu_inputs), list-input (after crafted verbaliser outputs of the four
wordings on every smoke seed, in the jobs' formats and keyed to their inputs) and finish (after crafted listing,
reranker and FT outputs): the DTS stages on seed 9001 with the real CLIP value encoder on the CPU,
external_sources.json, the smoke agreement, the rule application, the descriptive pass with every external row
present, the wiring mutation in a subprocess (it must fire), the leak check, the record. A control run of the
mutation's subprocess on an unchanged copy of r6_score.py must pass. About 10 minutes on CPU (three setups: the
run's, the mutation's, the control's):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_smoke_e2e.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_dts as DT  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_external as XT  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_ft_features as FF  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import r6_gpu_listing as L  # noqa: E402
import run_r6_apply_rule as AR  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_refit as RR  # noqa: E402
import run_r6_smoke as SM  # noqa: E402
import test_r6_dts as TDTS  # noqa: E402  (crafted answers and listings of ticket 11's end to end)
import test_r6_external as TX  # noqa: E402  (planted reranker scores, FT features)
import test_r6_held as TH  # noqa: E402
import test_r6_smoke as TS  # noqa: E402

import numpy as np  # noqa: E402

DECIMAL = re.compile(r"\d*\.\d+")
SETTINGS, SETTINGS_SHA = DT.load_settings()


def scripts(*names) -> dict:
    return {n: R.sha256_file(HERE / n) for n in names}


def shas(folder, files) -> dict:
    return {f: G.sha256_file(Path(folder) / f) for f in files}


def provenance(d, fp, runs=()):
    G.write_json(Path(d) / "provenance.json", {"fingerprint": fp, "runs": list(runs)})


def write_vout(d, recs, job_shas):
    """phrases_<W>.jsonl per wording and provenance.json as r6_gpu_verbalise writes them (its fingerprint's fields)."""
    d.mkdir(parents=True)
    by_w = defaultdict(list)
    for r in recs:
        by_w[r["wording"]].append(r)
    for w, rs in by_w.items():
        st = G.KeyedJsonl(d / f"phrases_{w}.jsonl", DT.VERBALISE_FIELDS, DT.VERBALISE_FIELDS[:4])
        for r in rs:
            st.add(dict(r))
        st.save()
    provenance(d, {"job": "r6_gpu_verbalise", "model_id": SETTINGS["model"]["id"],
                   "snapshot": SETTINGS["model"]["snapshot"], "settings_sha256": SETTINGS_SHA,
                   "scripts_sha256": scripts("r6_gpu_verbalise.py", "r6_gpu_common.py"), "inputs_sha256": job_shas})
    return d


def write_lout(d, answers):
    d.mkdir(parents=True)
    st = G.KeyedJsonl(d / "listings.jsonl", DT.LISTING_FIELDS, DT.LISTING_FIELDS[:2])
    for (p, K), a in answers.items():
        st.add({"phrase": p, "K": int(K), "answer": a})
    st.save()
    provenance(d, {"job": "r6_gpu_listing", "model_id": SETTINGS["model"]["id"],
                   "snapshot": SETTINGS["model"]["snapshot"], "settings_sha256": SETTINGS_SHA,
                   "scripts_sha256": scripts("r6_gpu_listing.py", "r6_gpu_common.py")})
    return d


def write_rr_out(d, job_shas, index, shown):
    d.mkdir(parents=True)
    with open(d / "scores.npz", "wb") as f:
        np.savez(f, episode_index=np.asarray(index, dtype=np.int64), scores_shown=np.asarray(shown, np.float32))
    provenance(d, {"job": "r6_gpu_rerank", **XT.rerank_model(), "inputs_sha256": job_shas,
                   "scripts_sha256": scripts("r6_gpu_rerank.py", "r6_gpu_common.py")})
    return d


def write_ft_out(d, job_shas, rows, rng):
    d.mkdir(parents=True)
    done = {}
    for v in XT.FT_VARIANTS:
        _, img, txt = TX.ft_features(rows, rng)
        with open(d / f"features_{v}.npz", "wb") as f:
            np.savez(f, rows=rows, img=img, txt=txt)
        done[v] = G.sha256_file(d / f"features_{v}.npz")
    provenance(d, {"job": "r6_gpu_ft_features", "variants": list(XT.FT_VARIANTS), "selected": FF.SELECTED,
                   "ckpt_sha256": {v: XT.FT_CKPTS[v]["sha256"] for v in XT.FT_VARIANTS},
                   "scripts_sha256": scripts("r6_gpu_ft_features.py", "r6_gpu_common.py"),
                   "inputs_sha256": job_shas, "image_source": "r6_ft_cache"},
               [{"status": "complete", "features_sha256": done}])
    return d


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    env = RH.setup()                                                   # real data, heads, check, PM, readers
    base = tmp_path_factory.mktemp("smoke_e2e")
    res = TH.write_records(base / "results", coef=env.coef_sha256, smoke_names=())
    (res / RH.REFIT_NAME).write_text(json.dumps(RR.record(env.head_check, env.heads, env.inputs)))
    TS.write_dts42(res)
    (base / "folder").mkdir()
    return SimpleNamespace(env=env, base=base, res=res, folder=base / "folder", staging=base / "staging",
                           paths=SM.paths_of("smoke", res, base / "folder"), stdout=[])


def run(w, capsys, stage, outs=None):
    code = SM.run(stage, outs=outs, results=w.res, folder=w.folder, env=w.env, staging=w.staging)
    out = capsys.readouterr().out
    w.stdout.append(out)
    return code, out


def test_full_local_smoke_with_crafted_gpu_outputs(world, capsys):
    w, p = world, world.paths
    assert w.env.head_check["passed"] is True

    # gpu-inputs: step 1, the held smoke, the round-1 job folders, the commands
    code, out = run(w, capsys, "gpu-inputs")
    assert code == 0, out[-3000:]
    state = json.loads(p.state.read_text())
    assert set(state["jobs"]["verbalise"]) == {str(s) for s in R.SMOKE_SEEDS}
    assert "cluster launch --node <node> -- bash scripts/run_r6_verbalise.sh smoke_verbalise_9001 --wordings " \
           "W1,W2,W3,W4" in out and "--stage list-input" in out
    for s in R.SMOKE_SEEDS:
        assert (p.smoke / f"held_episodes_seed{s}.npz").is_file()
    assert (p.smoke / "held_pass.json").is_file() and not (p.smoke / "held_verdict.json").exists()

    # round 1, crafted: the verbaliser (four wordings, every seed; seed 9001 in two shards), the reranker (two shards),
    # the FT features
    rng = np.random.default_rng(15)
    vouts = []
    for s, job in sorted(state["jobs"]["verbalise"].items()):
        jd, s = p.gpu / job, int(s)
        n = E.load_episodes(p.smoke / RH.episodes_name(s)).n
        recs = TDTS.records_for(s, range(n), DT.WORDINGS, TDTS.craft_answer)
        js = shas(jd, ("rows_manifest.npz", "verbalise_input.npz"))
        if s == SM.DTS_SEED:
            half = len(recs) // 2
            vouts += [write_vout(w.base / f"out/v{s}a", recs[:half + 1], js),
                      write_vout(w.base / f"out/v{s}b", recs[half:], js)]
        else:
            vouts.append(write_vout(w.base / f"out/v{s}", recs, js))
    rj = p.gpu / state["jobs"]["rerank"]
    with np.load(I.perms_path(rj)) as z:
        perms = z["perms"]
    shown = TX.planted(perms, rng)
    shown[1::3] = rng.standard_normal(shown[1::3].shape).astype(np.float32)
    rjs, cut = shas(rj, ("rows_manifest.npz", "rerank_input.npz")), len(perms) // 2
    routs = [write_rr_out(w.base / "out/rr0", rjs, np.arange(cut), shown[:cut]),
             write_rr_out(w.base / "out/rr1", rjs, np.arange(cut, len(perms)), shown[cut:])]
    fj = p.gpu / state["jobs"]["ft"]
    with np.load(fj / "ft_rows.npz") as z:
        rows = z["rows"]
    fouts = [write_ft_out(w.base / "out/ft", shas(fj, ("rows_manifest.npz", "ft_rows.npz")), rows, rng)]

    # list-input: the verbaliser outputs checked, the one listing job of round 2
    code, out = run(w, capsys, "list-input", {"verbalise": vouts})
    assert code == 0, out[-3000:]
    state = json.loads(p.state.read_text())
    items = L.load_listing_input(p.gpu / state["listing_job"] / "listing_input.jsonl", SETTINGS)
    assert {(n, K) for n in DT.TARGET_NAME.values() for K in DT.KS} <= set(items)
    names = R.value_names(w.env.data)
    louts = [write_lout(w.base / "out/listing", {(q, K): TDTS.craft_listing(q, K, names) for q, K in items})]

    # finish
    outs = {"verbalise": vouts, "listing": louts, "rerank": routs, "ft": fouts}
    code, out = run(w, capsys, "finish", outs)
    rec = json.loads(p.record.read_text())
    assert code == 0, (out[-3000:], {k: v for k, v in rec.items() if k in ("steps", "mutation", "leak_check")})
    assert rec["passed"] is True and p.record.read_bytes() == (w.folder / "smoke_record.json").read_bytes()
    assert rec["module_sha256"] == R.r6_module_shas() and rec["same_bytes_as_the_smoke_pass"] is True
    assert rec["runner_sha256"] == R.sha256_file(HERE / "run_r6_held.py")
    assert rec["dts_settings_sha256"] == R.sha256_file(HERE / "dts_settings.json")
    assert all(s["passed"] for s in rec["steps"].values()), rec["steps"]
    assert set(rec["steps"]["dts_stages"]["stages"]) == {"sanity", "tune", "chosen", "stop"}
    assert rec["mutation"]["fired"] is True and rec["mutation"]["exit"] != 0 and rec["mutation"]["message_found"]
    assert rec["leak_check"]["passed"] is True and len(rec["leak_check"]["files"]) >= 10
    assert set(rec["external_sources"]) == set(XT.SOURCE_KEYS)
    assert not any("missing" in v for v in rec["external_sources"].values()), rec["external_sources"]
    assert rec["external_rows"] == {n: "present" for n in XT.NAMES}, rec["external_rows"]
    assert rec["steps"]["external_rows"] == {"passed": True, "missing_although_given": []}
    src = json.loads((p.smoke / XT.SOURCES_NAME).read_text())
    assert src == rec["external_sources"] and src["dts"]["record"].endswith("dts/dts_seed9001.json")
    stop = json.loads((p.dts / "dts_stop.json").read_text())
    assert stop["stop"] is False and stop["built"] is True
    agr = json.loads((p.smoke / "rederive_agreement.json").read_text())
    assert agr["smoke"] is True and agr["held_pass_sha256"] == R.sha256_file(p.smoke / "held_pass.json")
    verdict = json.loads((p.smoke / "held_verdict.json").read_text())
    assert verdict["mode"] == "smoke" and verdict["agreement_sha256"] == R.sha256_file(
        p.smoke / "rederive_agreement.json")
    desc = json.loads((p.smoke / "descriptive.json").read_text())
    assert desc["mode"] == "smoke" and desc["verdict"]["sha256"] == R.sha256_file(p.smoke / "held_verdict.json")
    mlog = (R.MAIN / rec["mutation"]["log"]) if not Path(rec["mutation"]["log"]).is_absolute() \
        else Path(rec["mutation"]["log"])
    text = mlog.read_text()
    assert SM.WIRING_MESSAGE in text and "the CF term of cell" in text and "r6_score.py from the copy" in text
    assert SM.WIRING_TO.strip() in (p.smoke / "mutation/code/r6_score.py").read_text()

    # the smoke prints no metric: no decimal number in its stdout, in any stage
    allout = "".join(w.stdout)
    assert not DECIMAL.search(allout), DECIMAL.findall(allout)[:5]

    # the record's consumers read it (keys, location, name): the held runner, the apply step, the descriptive pass
    assert RH.smoke_guard(w.res, "held")["name"] == "smoke_record.json"
    assert AR.smoke_guard(w.res, False)["name"] == "smoke_record.json"
    assert RD.smoke_guard(w.res, False, False)["file"] == "smoke_record.json"


def test_a_control_run_without_the_mutation_passes(world):
    """The mutation's subprocess on an unchanged copy of r6_score.py: the smoke scoring goes through (exit 0, the
    smoke pass written), so the mutation's failure is the mutation's."""
    out = SM.wiring_mutation(world.paths, HERE, control=True)
    assert out["passed"] is True and out["exit"] == 0 and out["message_found"] is False, out
