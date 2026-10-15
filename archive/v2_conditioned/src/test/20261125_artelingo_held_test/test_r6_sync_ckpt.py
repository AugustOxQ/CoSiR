"""das6_sync_r6.py --ckpt: the plan, the refusals and the remote path, with the cluster helpers stubbed (no network)."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
SCRIPT = CHECKOUT / "scripts" / "das6_sync_r6.py"
pytestmark = pytest.mark.skipif(not Path("/root/.claude/skills/cluster-run/cluster.py").is_file(),
                                reason="the cluster-run skill is missing")


@pytest.fixture
def sync(monkeypatch):
    spec = importlib.util.spec_from_file_location("das6_sync_r6_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    def boom(*a, **k):
        raise AssertionError("the cluster must not be contacted")
    for name in ("remote", "require_remote", "require_local"):
        monkeypatch.setattr(mod.cluster, name, boom)
    monkeypatch.setattr(mod.cluster, "load_config", lambda *a, **k: {"DATA_MAX_GB": "5"})
    monkeypatch.setattr(mod.cluster, "validate_node", lambda n, cfg: n)
    monkeypatch.setattr(mod.cluster, "run_data_sync",
                        lambda cfg, node, acts: calls.append((node, acts)) or [{"key": a["key"]} for a in acts])
    mod.calls = calls
    return mod


def ckpt(tmp_path, folder="a", data=b"weights"):
    d = tmp_path / folder
    d.mkdir()
    (d / "best_params.pt").write_bytes(data)
    return d / "best_params.pt"


def main(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["das6_sync_r6.py", "--node", "node401", *argv])
    mod.main()


def test_plan_prints_sha_size_and_remote_without_copying(sync, monkeypatch, tmp_path, capsys):
    a, b = ckpt(tmp_path, "a", b"lb-bytes"), ckpt(tmp_path, "b", b"lora")
    main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={a}", "--ckpt", f"LoRA_lr1e-4={b}")
    out = capsys.readouterr().out
    assert hashlib.sha256(b"lb-bytes").hexdigest() in out and "8 bytes" in out
    assert hashlib.sha256(b"lora").hexdigest() in out
    assert "-> /local/wding/r6_jobs/ckpt/LB_lr3e-5.pt" in out and "-> /local/wding/r6_jobs/ckpt/LoRA_lr1e-4.pt" in out
    assert "Plan only" in out and sync.calls == []


def test_run_hands_file_actions_to_the_cluster_sync(sync, monkeypatch, tmp_path):
    a = ckpt(tmp_path)
    main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={a}", "--run")
    (node, acts), = sync.calls
    act, = acts
    assert node == "node401" and act["kind"] == "file" and act["local"] == str(a.resolve())
    assert act["remote"] == "/local/wding/r6_jobs/ckpt/LB_lr3e-5.pt"
    argv = sync.cluster.rsync_argv("node401", act)
    assert argv[-2:] == [str(a.resolve()), "node401:/local/wding/r6_jobs/ckpt/LB_lr3e-5.pt"]


def test_run_refuses_a_file_changed_after_the_plan(sync, monkeypatch, tmp_path):
    a = ckpt(tmp_path)
    acts = sync.ckpt_actions([f"LB_lr3e-5={a}"])
    a.write_bytes(b"changed")
    monkeypatch.setattr(sync, "build_actions", lambda cfg, args: (acts, []))
    with pytest.raises(SystemExit, match="changed since the plan"):
        main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={a}", "--run")
    assert sync.calls == []


@pytest.mark.parametrize("spec", ["Other_name={p}", "LB_lr3e-5", "lb_lr3e-5={p}", "={p}"])
def test_refuses_an_unknown_name(sync, monkeypatch, tmp_path, spec):
    p = ckpt(tmp_path)
    with pytest.raises(SystemExit, match="REFUSING"):
        main(sync, monkeypatch, "--ckpt", spec.format(p=p))


def test_refuses_a_path_that_is_not_best_params(sync, monkeypatch, tmp_path):
    other = tmp_path / "model.pt"
    other.write_bytes(b"x")
    real = ckpt(tmp_path)
    link = tmp_path / "link" / "best_params.pt"
    link.parent.mkdir()
    link.symlink_to(real)
    for bad in (other, tmp_path / "a", tmp_path / "missing" / "best_params.pt", link):
        with pytest.raises(SystemExit, match="REFUSING"):
            main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={bad}")


def test_refuses_a_name_twice(sync, monkeypatch, tmp_path):
    p = ckpt(tmp_path)
    with pytest.raises(SystemExit, match="twice"):
        main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={p}", "--ckpt", f"LB_lr3e-5={p}")


def test_remote_paths_must_lie_under_local_wding(sync, monkeypatch, tmp_path):
    p = ckpt(tmp_path)
    monkeypatch.setattr(sync, "CKPT_REMOTE", "/tmp/ckpt")
    with pytest.raises(SystemExit, match="outside /local/wding/"):
        main(sync, monkeypatch, "--ckpt", f"LB_lr3e-5={p}")


def test_ckpt_combines_with_job_dir(sync, monkeypatch, tmp_path, capsys):
    p = ckpt(tmp_path)
    monkeypatch.setattr(sync, "style_folders", lambda *a: ["Baroque"])
    job = tmp_path / "jobx"
    job.mkdir()
    (job / "listing_input.jsonl").write_text('{"id": 0}\n')
    main(sync, monkeypatch, "--job-dir", str(job), "--ckpt", f"LB_lr3e-5={p}")
    out = capsys.readouterr().out
    assert "r6_job_jobx: dir" in out and "r6_ckpt_LB_lr3e-5: file" in out and "(2 actions)" in out


def test_ckpt_alone_is_enough(sync, monkeypatch, tmp_path, capsys):
    p = ckpt(tmp_path)
    main(sync, monkeypatch, "--ckpt", f"LoRA_lr1e-4={p}")
    assert "(1 actions)" in capsys.readouterr().out
