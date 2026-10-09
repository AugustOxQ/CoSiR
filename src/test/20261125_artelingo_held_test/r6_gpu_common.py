"""Round 6 GPU job helpers (DECISION_RULE.md of this folder: section 7 items 1 and 2, section 8 item 3, section 10 item
2; contracts section 9): the DTS settings file, the job-input files, the keyed jsonl outputs with checkpoint and resume,
provenance, the pinned model snapshot and greedy generation.

Where it runs. On a DAS6 node inside a job worktree, or here under the GPU lock. It imports neither r6_common nor
anything from src: r6_common resolves the main checkout and imports earlier rounds' folders and inputs, which a job
worktree need not hold, and a GPU job needs none of them. torch and transformers are imported only inside the functions
that load or run the model, so the CPU paths (argument parsing, file checks, message building, the tests) run without
them.

What a job sees. Its job folder only: the row manifest (row id, sample id, image path, caption) and its own input file.
Never the episode file, labels, aspect names (DTS-N listings excepted) or candidate order. A job writes raw outputs
keyed by (seed, episode index, condition, wording) or by (phrase, K), and computes and prints no metric.

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import hashlib
import json
import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE / "dts_settings.json"
CONDITIONS = ("a", "b")
NUM_PAIRS = 4
CHECKPOINT_EVERY = 50
MANIFEST_KEYS = ("rows", "sample_id", "image_relpath", "caption")
PAIR_FIELDS = ("pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
VERBALISE_KEYS = ("seed", "episode_index", *PAIR_FIELDS)
WORDING_IDS = ("W1", "W2", "W3", "W4")
HEX40 = re.compile(r"[0-9a-f]{40}")
SNAPSHOT_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json", "preprocessor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "chat_template.json")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def sha256_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def amsterdam_now() -> str:
    """'YYYY-MM-DD HH:MM:SS' in Europe/Amsterdam; a node without tz data gets UTC, marked ' UTC'."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001  (no tz database on the node)
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------- settings (rule section 7)

def load_settings(path=None):
    """(settings dict, SHA-256 of the file's bytes). The structure the jobs rely on is asserted."""
    path = Path(path or SETTINGS_PATH)
    raw = path.read_bytes()
    s = json.loads(raw.decode("utf-8"))
    gen, ver, lst = s["generation"], s["verbaliser"], s["listing"]
    kw = gen["generate_kwargs"]
    _require(kw.get("do_sample") is False and kw.get("num_beams") == 1
             and all(kw.get(k, 0) is None for k in ("temperature", "top_p", "top_k")),
             f"{path}: generation is not greedy: {kw}")  # guard:settings_greedy
    _require(gen["verbaliser"]["batch_size"] == 1 and gen["verbaliser"]["max_new_tokens"] == 32
             and gen["listing"]["max_new_tokens"] == 128 and int(gen["listing"]["batch_size"]) >= 1
             and gen["listing"]["padding_side"] == "left",
             f"{path}: token limits or batch sizes differ from the rule's 32 / 128 and batch size 1")
    _require(tuple(sorted(ver["wordings"])) == WORDING_IDS
             and all(isinstance(w, str) and w.strip() == w and w for w in ver["wordings"].values()),
             f"{path}: the wordings must be W1 to W4, non-empty")
    _require(ver["conditions"] == {"a": {"group_a": "pairs_a", "group_b": "pairs_b"},
                                   "b": {"group_a": "pairs_b", "group_b": "pairs_a"}},
             f"{path}: condition b must swap the groups of condition a")  # guard:settings_swap
    _require(lst["K"] == [8, 16] and "{K}" in lst["prompt"] and "{phrase}" in lst["prompt"],
             f"{path}: listing K or prompt placeholders differ")
    _require(HEX40.fullmatch(s["model"]["snapshot"]) is not None, f"{path}: the snapshot is not a 40-hex revision")
    return s, sha256_bytes(raw)


# ---------------------------------------------------------------- job inputs (contracts section 9)

class Manifest:
    """rows_manifest.npz: global row id -> (sample id, image path relative to the WikiArt root, caption)."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            _require(sorted(z.files) == sorted(MANIFEST_KEYS),
                     f"{self.path}: keys {sorted(z.files)} differ from {sorted(MANIFEST_KEYS)}")  # guard:manifest_keys
            d = {k: z[k] for k in MANIFEST_KEYS}
        self.rows, self.sample_id = d["rows"], d["sample_id"]
        self.image_relpath, self.caption = d["image_relpath"], d["caption"]
        n = len(self.rows)
        _require(self.rows.dtype == np.int64 and self.sample_id.dtype == np.int64, f"{self.path}: ids must be int64")
        _require(self.image_relpath.dtype.kind == "U" and self.caption.dtype.kind == "U",
                 f"{self.path}: image_relpath and caption must be unicode arrays")
        _require(all(x.shape == (n,) for x in d.values()) and n > 0, f"{self.path}: arrays of unequal length")
        _require(bool((np.diff(self.rows) > 0).all()), f"{self.path}: rows must be sorted and unique")
        _require(len(np.unique(self.sample_id)) == n, f"{self.path}: sample ids must be unique")
        for rel in self.image_relpath.tolist():
            _require(rel and not rel.startswith("/") and ".." not in Path(rel).parts,
                     f"{self.path}: image path {rel!r} is not a plain relative path")

    def index(self, rows) -> np.ndarray:
        """Positions of ``rows`` in the manifest; every row must be listed."""
        rows = np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self.rows, rows)
        pos_c = np.minimum(pos, len(self.rows) - 1)
        _require(bool((self.rows[pos_c] == rows).all()),
                 f"{self.path}: {int((self.rows[pos_c] != rows).sum())} rows are not in the manifest")  # guard:manifest_rows
        return pos_c

    def items(self, img_rows, txt_rows, image_root):
        """[(image path, caption)] of pairs whose image is row img_rows[k] and caption row txt_rows[k]."""
        ii, tt = self.index(img_rows), self.index(txt_rows)
        root = Path(image_root)
        return [(str(root / self.image_relpath[i]), str(self.caption[t])) for i, t in zip(ii.tolist(), tt.tolist())]


def load_verbalise_input(path, manifest=None) -> dict:
    """verbalise_input.npz: seed, episode_index (n,), pairs_{a,b}_{img,txt} (n, 4) global row ids, all int64."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        _require(sorted(z.files) == sorted(VERBALISE_KEYS),
                 f"{path}: keys {sorted(z.files)} differ from {sorted(VERBALISE_KEYS)}")  # guard:input_keys
        d = {k: z[k] for k in VERBALISE_KEYS}
    n = len(d["episode_index"])
    _require(all(d[k].dtype == np.int64 for k in VERBALISE_KEYS), f"{path}: every array must be int64")
    _require(d["seed"].shape == () and d["episode_index"].shape == (n,) and n > 0, f"{path}: seed or index shape")
    _require(all(d[k].shape == (n, NUM_PAIRS) for k in PAIR_FIELDS), f"{path}: pair arrays must be (n, 4)")
    _require(bool((np.diff(d["episode_index"]) > 0).all()) and int(d["episode_index"][0]) >= 0,
             f"{path}: episode_index must be increasing and non-negative")
    if manifest is not None:
        for k in PAIR_FIELDS:
            manifest.index(d[k].ravel())
    d["seed"] = int(d["seed"])
    return d


# ---------------------------------------------------------------- keyed jsonl outputs with checkpoints

class KeyedJsonl:
    """Records with exactly ``fields`` (in that order), unique by ``key_fields``. The file is rewritten whole at each
    checkpoint (temporary file, fsync, os.replace), so on disk it always holds whole records; a resumed run reads it,
    validates every record and skips the keys already present."""

    def __init__(self, path, fields, key_fields):
        self.path, self.fields, self.key_fields = Path(path), tuple(fields), tuple(key_fields)
        self._records, self._keys, self.dirty = [], set(), False

    def key(self, rec) -> tuple:
        return tuple(rec[f] for f in self.key_fields)

    def _check(self, rec, allowed):
        _require(isinstance(rec, dict) and tuple(rec) == self.fields,
                 f"{self.path}: record fields {list(rec) if isinstance(rec, dict) else rec!r} differ from "
                 f"{list(self.fields)}")
        k = self.key(rec)
        _require(k not in self._keys, f"{self.path}: duplicate record {k}")  # guard:jsonl_unique
        _require(allowed is None or k in allowed, f"{self.path}: record {k} is not one this run plans")  # guard:jsonl_planned
        return k

    def load(self, allowed=None) -> int:
        """Read an existing file (each line one record); ``allowed`` (a set of keys) refuses foreign records."""
        if not self.path.exists():
            return 0
        text = self.path.read_text(encoding="utf-8")
        _require(text == "" or text.endswith("\n"), f"{self.path}: the last record is incomplete")
        for line in text.splitlines():
            rec = json.loads(line)
            self._keys.add(self._check(rec, allowed))
            self._records.append(rec)
        return len(self._records)

    def add(self, rec, allowed=None):
        rec = {f: rec[f] for f in self.fields} if set(rec) == set(self.fields) else rec
        self._keys.add(self._check(rec, allowed))
        self._records.append(rec)
        self.dirty = True

    def __contains__(self, key):
        return key in self._keys

    def __len__(self):
        return len(self._records)

    def records(self) -> list:
        return list(self._records)

    def save(self):
        if not self.dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.dirty = False


# ---------------------------------------------------------------- provenance

def script_shas(*paths) -> dict:
    return {Path(p).name: sha256_file(p) for p in paths}


def versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "transformers"):
        mod = sys.modules.get(name)
        out[name] = getattr(mod, "__version__", None) if mod is not None else None
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cuda"] = torch.version.cuda
    return out


def begin_provenance(out_dir, fingerprint: dict, run: dict) -> dict:
    """Open provenance.json of ``out_dir``: a new file, or an existing one whose fingerprint (model, snapshot,
    settings, scripts, inputs) equals this run's, so a resumed run never mixes outputs of other code or inputs."""
    path = Path(out_dir) / "provenance.json"
    if path.exists():
        prov = json.loads(path.read_text())
        _require(prov.get("fingerprint") == fingerprint,
                 f"{path}: written by another model, settings, script or input; use a new output folder "
                 f"(differs in {sorted(k for k in set(fingerprint) | set(prov.get('fingerprint', {})) if prov.get('fingerprint', {}).get(k) != fingerprint.get(k))})")  # guard:fingerprint
    else:
        prov = {"fingerprint": fingerprint, "runs": []}
    prov["runs"].append(dict(run, start=amsterdam_now(), host=socket.gethostname(), status="running"))
    write_json(path, prov)
    return prov


def end_provenance(out_dir, prov: dict, **fields) -> dict:
    prov["runs"][-1].update(fields, end=amsterdam_now())
    write_json(Path(out_dir) / "provenance.json", prov)
    return prov


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- the pinned model snapshot

def hub_cache(arg=None) -> Path:
    """--hub-cache, else $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub."""
    if arg:
        return Path(arg)
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def snapshot_dir(hub, model_id, snapshot) -> Path:
    return Path(hub) / f"models--{model_id.replace('/', '--')}" / "snapshots" / snapshot


def check_snapshot(path) -> dict:
    """The pinned snapshot directory is complete: the config, processor and tokenizer files, every weight shard the
    index names, and no entry that points at a missing blob."""
    path = Path(path)
    _require(path.is_dir(), f"model snapshot missing: {path}")
    dangling = [p.name for p in path.iterdir() if not p.exists()]
    _require(not dangling, f"{path}: entries point at missing blobs: {dangling[:5]}")
    missing = [f for f in SNAPSHOT_FILES if not (path / f).is_file()]
    _require(not missing, f"{path}: missing {missing}")
    shards = sorted(set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values()))
    absent = [s for s in shards if not (path / s).is_file()]
    _require(shards and not absent, f"{path}: weight shards missing: {absent}")  # guard:snapshot_shards
    refs = path.parent.parent / "refs" / "main"
    return {"dir": str(path), "files": len(list(path.iterdir())), "shards": len(shards),
            "refs_main": refs.read_text().strip() if refs.is_file() else None,
            "chat_template_sha256": sha256_file(path / "chat_template.json"),
            "generation_config_sha256": sha256_file(path / "generation_config.json")}


def load_model(snap, settings):
    """Processor and model as src/eval/mllm_reranker.py loads them, from the pinned snapshot directory."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    _require(torch.cuda.is_available(), "no CUDA device: the model runs on the GPU only", RuntimeError)
    m = settings["model"]
    _require(m["dtype"] == "bfloat16" and m["device"] == "cuda", "settings: model dtype/device")
    processor = AutoProcessor.from_pretrained(str(snap), **m["processor_kwargs"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(snap), torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    return processor, model


# ---------------------------------------------------------------- greedy generation

def check_greedy(new_tokens, argmax_steps, eos_ids, pad_id):
    """new_tokens (B, T): tokens generated after the prompt; argmax_steps (T, B): the argmax of each step's logits.
    Up to and including a row's first end-of-sequence token each token must be its step's argmax; after it, padding."""
    new_tokens, am = np.asarray(new_tokens), np.asarray(argmax_steps)
    eos = {int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])}
    _require(new_tokens.ndim == 2 and am.shape == new_tokens.shape[::-1],
             f"argmax steps {am.shape} do not match the generated tokens {new_tokens.shape}")
    for b in range(new_tokens.shape[0]):
        done = False
        for t in range(new_tokens.shape[1]):
            tok = int(new_tokens[b, t])
            want = int(pad_id) if done else int(am[t, b])
            _require(tok == want, f"row {b}, step {t}: token {tok} is not the greedy choice {want}",
                     RuntimeError)  # guard:greedy
            done = done or tok in eos
    return True


def generate_texts(model, processor, inputs, max_new_tokens, settings) -> list:
    """Greedy generation of a batch: the decoded new text of each row (special tokens skipped, raw, unnormalised),
    every token checked by check_greedy."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class ArgmaxRecorder(LogitsProcessor):
        def __init__(self):
            self.steps = []

        def __call__(self, input_ids, logits):
            self.steps.append(logits.argmax(dim=-1))
            return logits

    rec = ArgmaxRecorder()
    kwargs = dict(settings["generation"]["generate_kwargs"])
    with torch.no_grad():
        seq = model.generate(**inputs, max_new_tokens=int(max_new_tokens), logits_processor=LogitsProcessorList([rec]),
                             **kwargs)
    new = seq[:, inputs["input_ids"].shape[1]:].cpu().numpy()
    am = torch.stack(rec.steps).cpu().numpy() if rec.steps else np.zeros((0, new.shape[0]), dtype=np.int64)
    gc = model.generation_config
    check_greedy(new, am, gc.eos_token_id, gc.pad_token_id)
    tok = processor.tokenizer
    return [tok.decode(row.tolist(), skip_special_tokens=True) for row in new]
