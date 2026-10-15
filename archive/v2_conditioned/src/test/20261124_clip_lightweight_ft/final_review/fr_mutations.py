"""Final review: mutation checks of the guards that protect the claim, each on a scratch copy of the code and tests.

Usage: python fr_mutations.py <scratch_dir>   (copies the six files per mutation, applies one exact-string edit,
runs pytest on the named test file, records whether the tests catch it). Never edits the repo's files.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]
FILES = ["ft_data.py", "ft_train.py", "ft_eval.py", "test_ft_data.py", "test_ft_train.py", "test_ft_eval.py"]
M = [
 # (id, guard, file, old, new, test file)
 ("H1", "held: check_rows never raises", "ft_train.py",
  "    bad = ~np.isin(np.asarray(rows, dtype=np.int64), allowed)\n",
  "    bad = np.zeros(len(np.asarray(rows)), dtype=bool)\n", "test_ft_train.py"),
 ("H2", "held: split_index held-row check removed", "ft_data.py",
  "    if np.intersect1d(allrows, held).size:\n        raise ValueError(\"a non-held split contains held rows\")\n", "", "test_ft_data.py"),
 ("H3", "held: cache builder painting-level held check removed", "ft_data.py",
  "    if held_p & set(uniq.tolist()):\n        raise ValueError(\"a painting of the cache also has held rows\")\n", "", "test_ft_data.py"),
 ("H4", "held: trainer never calls check_rows on its row sets", "ft_train.py",
  "    for k in ft_data.SPLIT_NAMES:\n        check_rows(use[k], idx, k)\n", "", "test_ft_train.py"),
 ("S1", "sampler: every train row instead of one per painting", "ft_train.py",
  "    return pick[rng.permutation(len(pick))].astype(np.int64)\n",
  "    return rng.permutation(len(pos)).astype(np.int64)\n", "test_ft_train.py"),
 ("S2", "sampler: same caption every epoch (rng ignores epoch)", "ft_train.py",
  "    rng = np.random.default_rng((seed, epoch))\n", "    rng = np.random.default_rng((seed, 0))\n", "test_ft_train.py"),
 ("S3", "sampler: runtime same-painting batch assertion removed", "ft_train.py",
  "            if len(np.unique(train.pos[b])) != len(b):\n                raise AssertionError(\"a batch holds two captions of one painting\")\n", "", "test_ft_train.py"),
 ("P1", "trainable set: LB without the projections", "ft_train.py",
  "\"text_model.final_layer_norm.\", \"visual_projection.\", \"text_projection.\")", "\"text_model.final_layer_norm.\")", "test_ft_train.py"),
 ("P2", "trainable set: LB trains the second-to-last layer too", "ft_train.py",
  "    lv = clip.config.vision_config.num_hidden_layers - 1\n", "    lv = clip.config.vision_config.num_hidden_layers - 2\n", "test_ft_train.py"),
 ("P3", "trainable set: LoRA without out_proj", "ft_train.py",
  "\"target_modules\": [\"q_proj\", \"k_proj\", \"v_proj\", \"out_proj\"]", "\"target_modules\": [\"q_proj\", \"k_proj\", \"v_proj\"]", "test_ft_train.py"),
 ("P4", "trainable set: LoRA temperature frozen", "ft_train.py",
  "        clip.logit_scale.requires_grad_(True)\n", "", "test_ft_train.py"),
 ("P5", "trainable set: LoRA rank 8", "ft_train.py", "LORA = {\"r\": 16,", "LORA = {\"r\": 8,", "test_ft_train.py"),
 ("T1", "tie rule: earlier epoch before smaller lr", "ft_train.py",
  "min((-e[\"selection\"], float(r[\"lr\"]), e[\"epoch\"])", "min((-e[\"selection\"], e[\"epoch\"], float(r[\"lr\"]))", "test_ft_train.py"),
 ("T2", "tie rule: larger lr wins a tie", "ft_train.py",
  "min((-e[\"selection\"], float(r[\"lr\"]), e[\"epoch\"])", "min((-e[\"selection\"], -float(r[\"lr\"]), e[\"epoch\"])", "test_ft_train.py"),
 ("T3", "tie rule: best_epoch ties to the later epoch", "ft_train.py",
  "key=lambda e: (-e[\"selection\"], e[\"epoch\"])", "key=lambda e: (-e[\"selection\"], -e[\"epoch\"])", "test_ft_train.py"),
 ("T4", "selection: epoch 0 selectable", "ft_train.py",
  "    cands = [e for e in epochs if e[\"epoch\"] >= 1]\n", "    cands = [e for e in epochs if e[\"epoch\"] >= 0]\n", "test_ft_train.py"),
 ("F1", "placement (eval): rows sorted before writing values", "ft_eval.py",
  "        full[rows[keep]] = values[keep]\n", "        full[np.sort(rows[keep])] = values[keep]\n", "test_ft_eval.py"),
 ("F2", "placement (trainer): features not reordered with rows", "ft_train.py",
  "    img = torch.cat([f[0] for f in feats]).numpy()[order].astype(np.float32)\n",
  "    img = torch.cat([f[0] for f in feats]).numpy().astype(np.float32)\n", "test_ft_train.py"),
 ("F3", "placement (eval): val rows kept beside selection", "ft_eval.py",
  "    keep = np.isin(rows, selection)\n", "    keep = np.ones(len(rows), dtype=bool)\n", "test_ft_eval.py"),
 ("F4", "pairs: training image by painting position, not cache row", "ft_train.py",
  "    return TrainPairs(cache_dir, train.cache_rows[train.pos], train.ids, train.mask)\n",
  "    return TrainPairs(cache_dir, train.pos, train.ids, train.mask)\n", "test_ft_train.py"),
 ("F5", "pairs: captions by feature row, not sample_ids", "ft_train.py",
  "ft_data.captions(np.asarray(data.sample_ids)[rows], annotations)", "ft_data.captions(rows, annotations)", "test_ft_train.py"),
 ("F6", "eval: image per painting from row's own painting broken (LB/LoRA eval uses pos of first painting)", "ft_train.py",
  "    return torch.cat(img_p)[torch.from_numpy(rs.pos)], torch.cat(txt)\n",
  "    return torch.cat(img_p)[torch.from_numpy(np.sort(rs.pos)[::-1].copy())], torch.cat(txt)\n", "test_ft_train.py"),
 ("E1", "either = other-aspect rate", "ft_eval.py",
  "        return (np.asarray(per_anchor_dict[\"r1\"], dtype=np.float64)\n                + np.asarray(per_anchor_dict[\"other\"], dtype=np.float64))\n",
  "        return np.asarray(per_anchor_dict[\"other\"], dtype=np.float64)\n", "test_ft_eval.py"),
 ("E2", "either = 2 x R@1 (right only for condition-free scorers)", "ft_eval.py",
  "        return (np.asarray(per_anchor_dict[\"r1\"], dtype=np.float64)\n                + np.asarray(per_anchor_dict[\"other\"], dtype=np.float64))\n",
  "        return 2 * np.asarray(per_anchor_dict[\"r1\"], dtype=np.float64)\n", "test_ft_eval.py"),
 ("E3", "comparison sign: plain minus ft", "ft_eval.py",
  "\"ft_minus_plain\": (\"ft\", \"plain\")", "\"ft_minus_plain\": (\"plain\", \"ft\")", "test_ft_eval.py"),
]

def main(scratch):
    scratch = Path(scratch)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", PYTHONPATH="/project/CoSiR")
    res = []
    for mid, what, f, old, new, test in M:
        d = scratch / mid
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for x in FILES:
            shutil.copy(SRC / x, d / x)
        txt = (d / f).read_text()
        n = txt.count(old)
        if n != 1:
            res.append({"id": mid, "guard": what, "status": f"NOT APPLIED (pattern count {n})"})
            print(mid, "NOT APPLIED", n, flush=True)
            continue
        (d / f).write_text(txt.replace(old, new))
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore", test],
                           cwd=d, env=env, capture_output=True, text=True, timeout=900)
        tail = [l for l in p.stdout.splitlines() if l.strip()][-1:]
        failed = [l.split(" ")[1] if l.startswith("FAILED") else l for l in p.stdout.splitlines() if l.startswith("FAILED")]
        status = "CAUGHT" if p.returncode != 0 else "SURVIVED"
        res.append({"id": mid, "guard": what, "file": f, "test_file": test, "status": status, "failing": failed[:3], "tail": tail})
        print(mid, status, failed[:2], tail, flush=True)
        shutil.rmtree(d)
    (Path(__file__).resolve().parent / "fr_mutations.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
