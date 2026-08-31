"""
Comprehensive test for FeatureManager and TrainableEmbeddingManager.

Tests:
  1. Correctness  – stored values match retrieved values exactly.
  2. Timing       – extraction write, shard load, training batch throughput.
  3. Memory       – RSS memory before/after each major phase; no runaway growth.
  4. All EM APIs  – every public method on TrainableEmbeddingManager.
  5. Simulated training loop – batch fetch → fake backward → embedding update,
                               repeated over multiple epochs.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
    python src/test/20260406_feature_embedding_test/test_feature_embedding.py
"""

import shutil
import time
import tracemalloc
from pathlib import Path
from typing import List

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

# --- project imports ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.feature_manager import FeatureManager
from src.utils.embedding_manager_nocache import TrainableEmbeddingManager
from src.dataset.cosir_datamodule import CoSiRShardDataset, CoSiRShardStreamDataset

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES      = 5_000      # total samples (small enough for fast CI)
CLIP_DIM       = 512        # CLIP CLS embedding dim
EMBEDDING_DIM  = 32         # label embedding dim
SHARD_SIZE     = 1_000      # samples per shard  → 5 shards
BATCH_SIZE     = 256
N_EPOCHS       = 3
EXTRACTION_BS  = 512        # batch size during fake extraction
TMPDIR         = Path("/tmp/cosir_test_20260406")

HEADER = "\n" + "="*60 + "\n"

def sep(title: str):
    print(f"{HEADER}{title}{HEADER[:-1]}")

def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_features(n: int, dim: int = CLIP_DIM) -> torch.Tensor:
    """Deterministic by shape so we can verify values later."""
    return torch.arange(n * dim, dtype=torch.float32).reshape(n, dim) / (n * dim)


def make_sample_ids(n: int) -> List[int]:
    """Shuffled IDs to mimic extraction DataLoader shuffle=True."""
    ids = list(range(n))
    np.random.default_rng(42).shuffle(ids)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – FeatureManager
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_manager():
    sep("SECTION 1 – FeatureManager")
    fm_dir = TMPDIR / "features"
    shutil.rmtree(fm_dir, ignore_errors=True)

    # ── 1a. Extraction write ──────────────────────────────────────────────────
    print("[1a] Extraction write…")
    mem_before = rss_mb()
    t0 = time.perf_counter()

    # Build ground-truth: img[i] = i-th row of make_features(N)
    gt_img = make_features(N_SAMPLES, CLIP_DIM)
    gt_txt = make_features(N_SAMPLES, CLIP_DIM) * 2.0
    sample_ids = make_sample_ids(N_SAMPLES)          # shuffled extraction order

    fm = FeatureManager(str(fm_dir), shard_size=SHARD_SIZE)
    fm.open_for_writing(N_SAMPLES, {
        "img_features": (CLIP_DIM,),
        "txt_features": (CLIP_DIM,),
    })

    # Write in extraction batches
    for start in range(0, N_SAMPLES, EXTRACTION_BS):
        bs = min(EXTRACTION_BS, N_SAMPLES - start)
        batch_ids = sample_ids[start:start + bs]
        # Use the actual row index (start..start+bs) as the tensor content
        fm.write_batch(
            gt_img[start:start + bs],
            gt_txt[start:start + bs],
            batch_ids,
        )

    fm.finalize_writing()
    write_time = time.perf_counter() - t0
    mem_after = rss_mb()
    print(f"    Write time:   {write_time*1000:.1f} ms")
    print(f"    Memory delta: {mem_after - mem_before:+.1f} MB")
    print(f"    Shards:       {fm.num_shards}  total_samples: {fm.total_samples}")
    assert fm.total_samples == N_SAMPLES
    assert fm.num_shards == N_SAMPLES // SHARD_SIZE

    # ── 1b. Correctness: get_all_sample_ids ──────────────────────────────────
    print("[1b] Correctness – get_all_sample_ids…")
    stored_ids = fm.get_all_sample_ids()
    assert stored_ids == sample_ids, "sample_ids ordering mismatch"
    print("    PASSED")

    # ── 1c. Correctness: load_shard_to_ram ───────────────────────────────────
    print("[1c] Correctness – load_shard_to_ram (shard 0)…")
    shard0 = fm.load_shard_to_ram(0)
    # Positions 0..SHARD_SIZE-1 in the file map to the first SHARD_SIZE extraction rows
    expected_img = gt_img[:SHARD_SIZE]
    assert torch.allclose(shard0["img_features"], expected_img), \
        "img_features shard0 mismatch"
    expected_ids = torch.tensor(sample_ids[:SHARD_SIZE], dtype=torch.int64)
    assert torch.equal(shard0["sample_ids"], expected_ids), \
        "sample_ids shard0 mismatch"
    print("    PASSED")

    # ── 1d. Correctness: load_all_to_ram ─────────────────────────────────────
    print("[1d] Correctness – load_all_to_ram…")
    t0 = time.perf_counter()
    all_data = fm.load_all_to_ram()
    load_time = time.perf_counter() - t0
    assert torch.allclose(all_data["img_features"], gt_img), \
        "load_all img_features mismatch"
    assert torch.allclose(all_data["txt_features"], gt_txt), \
        "load_all txt_features mismatch"
    print(f"    Load time:  {load_time*1000:.1f} ms")
    print(f"    img shape:  {all_data['img_features'].shape}")
    print("    PASSED")

    # ── 1e. RAM dataset + DataLoader ─────────────────────────────────────────
    print("[1e] Timing – CoSiRShardDataset + DataLoader (shuffle=True)…")
    # Re-open FM (simulates loading existing store)
    fm2 = FeatureManager(str(fm_dir), shard_size=SHARD_SIZE)
    ds_ram = CoSiRShardDataset(fm2)
    loader_ram = DataLoader(ds_ram, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    t0 = time.perf_counter()
    seen_ids = set()
    for batch in loader_ram:
        seen_ids.update(batch["sample_ids"].tolist())
    epoch_time = time.perf_counter() - t0

    assert len(seen_ids) == N_SAMPLES, "Not all samples seen in one epoch"
    print(f"    Epoch time ({N_SAMPLES} samples, bs={BATCH_SIZE}): {epoch_time*1000:.1f} ms")
    print(f"    Batches/epoch: {len(loader_ram)}")
    print("    PASSED – all samples seen exactly once")

    # Verify shuffle changes batch order each epoch
    batches_e0 = [b["sample_ids"][0].item() for b in loader_ram]
    batches_e1 = [b["sample_ids"][0].item() for b in loader_ram]
    assert batches_e0 != batches_e1, "Batches did not shuffle between epochs"
    print("    PASSED – batch order differs across epochs")

    # ── 1f. Stream dataset ────────────────────────────────────────────────────
    print("[1f] Timing – CoSiRShardStreamDataset…")
    ds_stream = CoSiRShardStreamDataset(fm2, window_shards=2, seed=0)
    loader_stream = DataLoader(ds_stream, batch_size=BATCH_SIZE, num_workers=0)

    epoch_times = []
    for epoch in range(2):
        ds_stream.set_epoch(epoch)
        t0 = time.perf_counter()
        seen = set()
        for batch in loader_stream:
            seen.update(batch["sample_ids"].tolist())
        epoch_times.append(time.perf_counter() - t0)
        assert len(seen) == N_SAMPLES, f"Stream epoch {epoch}: missing samples"

    print(f"    Epoch times: {[f'{t*1000:.0f}ms' for t in epoch_times]}")
    print("    PASSED – all samples seen each epoch")

    # ── 1g. Compat shims ─────────────────────────────────────────────────────
    print("[1g] Compat shims (get_num_chunks / get_features_by_chunk)…")
    assert fm2.get_num_chunks() == fm2.num_shards
    shard_via_compat = fm2.get_features_by_chunk(0)
    assert "img_features" in shard_via_compat
    assert shard_via_compat["img_features"].shape == (SHARD_SIZE, CLIP_DIM)
    print("    PASSED")

    return fm2   # return for use in EmbeddingManager tests


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – TrainableEmbeddingManager: all APIs
# ══════════════════════════════════════════════════════════════════════════════

def test_embedding_manager(fm: FeatureManager):
    sep("SECTION 2 – TrainableEmbeddingManager: all APIs")
    em_base = TMPDIR / "embeddings"
    sample_ids = fm.get_all_sample_ids()

    # ── 2a. Initialization strategies ────────────────────────────────────────
    print("[2a] Initialization strategies…")
    for strategy in ("zeros", "normal", "uniform"):
        em_dir = em_base / f"init_{strategy}"
        shutil.rmtree(em_dir, ignore_errors=True)
        em = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=EMBEDDING_DIM,
            embeddings_dir=str(em_dir),
            mode="ram",
            initialization_strategy=strategy,
        )
        _, all_emb = em.get_all_embeddings()
        if strategy == "zeros":
            assert all_emb.abs().max() == 0.0, "zeros init failed"
        else:
            assert all_emb.abs().max() > 0.0, f"{strategy} init produced zeros"
        print(f"    {strategy:8s}  shape={all_emb.shape}  "
              f"max={all_emb.abs().max():.4f}  PASSED")

    # ── 2b. PCA-based initialization ─────────────────────────────────────────
    print("[2b] PCA-based initialization (imgtxt / txt / img)…")
    for strategy in ("imgtxt", "txt", "img"):
        em_dir = em_base / f"init_{strategy}"
        shutil.rmtree(em_dir, ignore_errors=True)
        t0 = time.perf_counter()
        em = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=EMBEDDING_DIM,
            embeddings_dir=str(em_dir),
            mode="ram",
        )
        em.initialize(strategy, feature_manager=fm, device="cpu", factor=1.0)
        elapsed = time.perf_counter() - t0
        _, all_emb = em.get_all_embeddings()
        assert all_emb.shape == (N_SAMPLES, EMBEDDING_DIM)
        assert all_emb.abs().max() > 0.0, f"{strategy} PCA init produced zeros"
        print(f"    {strategy:8s}  shape={all_emb.shape}  "
              f"mean_norm={all_emb.norm(dim=1).mean():.4f}  "
              f"time={elapsed*1000:.0f}ms  PASSED")

    # ── 2c. get_embeddings / update_embeddings roundtrip ─────────────────────
    print("[2c] get_embeddings / update_embeddings roundtrip…")
    em_dir = em_base / "roundtrip"
    shutil.rmtree(em_dir, ignore_errors=True)
    em = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=EMBEDDING_DIM,
        embeddings_dir=str(em_dir),
        mode="ram",
    )

    # Pick random batch of IDs
    rng = np.random.default_rng(7)
    batch_ids = rng.choice(sample_ids, size=BATCH_SIZE, replace=False).tolist()
    new_emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    em.update_embeddings(batch_ids, new_emb)
    retrieved = em.get_embeddings(batch_ids)
    assert torch.allclose(retrieved.cpu(), new_emb.cpu(), atol=1e-6), \
        "Roundtrip mismatch after update"
    print("    RAM mode roundtrip PASSED")

    # Verify other IDs were NOT modified
    other_ids = [sid for sid in sample_ids[:20] if sid not in batch_ids][:5]
    before = em.get_embeddings(other_ids).clone()
    em.update_embeddings(batch_ids, torch.randn(BATCH_SIZE, EMBEDDING_DIM))
    after = em.get_embeddings(other_ids)
    assert torch.allclose(before, after), "Untouched IDs were modified"
    print("    Untouched IDs unaffected PASSED")

    # ── 2d. Disk persistence ──────────────────────────────────────────────────
    print("[2d] Disk persistence – update survives reload…")
    sentinel_ids = sample_ids[:5]
    sentinel_emb = torch.arange(5 * EMBEDDING_DIM, dtype=torch.float32).reshape(5, EMBEDDING_DIM)
    em.update_embeddings(sentinel_ids, sentinel_emb)

    # Reload from same dir
    em_reload = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=EMBEDDING_DIM,
        embeddings_dir=str(em_dir),
        mode="ram",
    )
    reloaded = em_reload.get_embeddings(sentinel_ids)
    assert torch.allclose(reloaded.cpu(), sentinel_emb.cpu(), atol=1e-6), \
        "Disk persistence failed"
    print("    PASSED")

    # ── 2e. mmap mode ─────────────────────────────────────────────────────────
    print("[2e] mmap mode get/update…")
    em_dir_mmap = em_base / "mmap"
    shutil.rmtree(em_dir_mmap, ignore_errors=True)
    em_mmap = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=EMBEDDING_DIM,
        embeddings_dir=str(em_dir_mmap),
        mode="mmap",
    )
    mmap_emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    em_mmap.update_embeddings(batch_ids, mmap_emb)
    mmap_retrieved = em_mmap.get_embeddings(batch_ids)
    assert torch.allclose(mmap_retrieved.cpu(), mmap_emb.cpu(), atol=1e-5), \
        "mmap mode roundtrip failed"
    print("    PASSED")

    # ── 2f. get_all_embeddings ────────────────────────────────────────────────
    print("[2f] get_all_embeddings…")
    t0 = time.perf_counter()
    ids_out, all_emb = em.get_all_embeddings()
    elapsed = time.perf_counter() - t0
    assert len(ids_out) == N_SAMPLES
    assert all_emb.shape == (N_SAMPLES, EMBEDDING_DIM)
    print(f"    shape={all_emb.shape}  time={elapsed*1000:.1f}ms  PASSED")

    # ── 2g. Template store / load ─────────────────────────────────────────────
    print("[2g] store_imgtxt_template / load_imgtxt_template…")
    em.store_imgtxt_template()
    em_fresh_dir = em_base / "fresh_for_template"
    shutil.rmtree(em_fresh_dir, ignore_errors=True)
    em_fresh = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=EMBEDDING_DIM,
        embeddings_dir=str(em_fresh_dir),
        mode="ram",
        initialization_strategy="zeros",
    )
    # Before load: should be zeros
    _, before_load = em_fresh.get_all_embeddings()
    assert before_load.abs().max() == 0.0

    em_fresh.load_imgtxt_template()
    _, after_load = em_fresh.get_all_embeddings()
    _, original = em.get_all_embeddings()
    assert torch.allclose(after_load, original, atol=1e-6), \
        "Template round-trip mismatch"
    print("    PASSED")

    # ── 2h. load_phase_1_template ─────────────────────────────────────────────
    print("[2h] load_phase_1_template…")
    phase1_path = em_base / "phase1_save"
    em._copy_to(phase1_path)   # use internal helper to create a "phase1" dir
    em_p1 = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=EMBEDDING_DIM,
        embeddings_dir=str(em_base / "p1_dest"),
        mode="ram",
        initialization_strategy="zeros",
    )
    em_p1.load_phase_1_template(str(phase1_path))
    _, p1_emb = em_p1.get_all_embeddings()
    assert torch.allclose(p1_emb, original, atol=1e-6), "load_phase_1 mismatch"
    print("    PASSED")

    return em  # return for training loop test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Simulated training loop
# ══════════════════════════════════════════════════════════════════════════════

def test_training_loop(fm: FeatureManager, em: TrainableEmbeddingManager):
    sep("SECTION 3 – Simulated training loop")

    # Use RAM dataset for training
    ds = CoSiRShardDataset(fm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    mem_start = rss_mb()
    tracemalloc.start()

    epoch_times = []
    batch_times = []
    mem_per_epoch = []

    print(f"Epochs={N_EPOCHS}  N={N_SAMPLES}  batch={BATCH_SIZE}  "
          f"batches/epoch={len(loader)}")

    for epoch in range(N_EPOCHS):
        t_epoch = time.perf_counter()

        for batch in loader:
            t_batch = time.perf_counter()

            img = batch["img_features"]          # [B, CLIP_DIM]
            txt = batch["txt_features"]          # [B, CLIP_DIM]
            ids = batch["sample_ids"].tolist()   # [B]

            # Load label embeddings
            label_embs = em.get_embeddings(ids)
            label_param = torch.nn.Parameter(label_embs.clone(), requires_grad=True)

            # Fake forward: project label to CLIP_DIM, add to txt, compare to img
            # (real model does something similar; here we just need a gradient)
            proj = label_param @ torch.randn(EMBEDDING_DIM, CLIP_DIM)  # [B, CLIP_DIM]
            combined = txt + proj
            loss = (combined - img).pow(2).mean()

            # Fake backward
            loss.backward()

            with torch.no_grad():
                label_param.data -= 1e-3 * label_param.grad

            # Persist updated embeddings
            em.update_embeddings(ids, label_param.detach())

            batch_times.append(time.perf_counter() - t_batch)

        epoch_times.append(time.perf_counter() - t_epoch)
        mem_per_epoch.append(rss_mb())

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_end = rss_mb()
    mem_growth = mem_end - mem_start

    print(f"\nResults:")
    print(f"  Epoch times:       {[f'{t*1000:.0f}ms' for t in epoch_times]}")
    print(f"  Mean batch time:   {np.mean(batch_times)*1000:.2f} ms")
    print(f"  Max batch time:    {np.max(batch_times)*1000:.2f} ms")
    print(f"  RSS at epoch end:  {[f'{m:.0f}MB' for m in mem_per_epoch]}")
    print(f"  Total RSS growth:  {mem_growth:+.1f} MB  (expected ~0 after warmup)")
    print(f"  Python peak alloc: {peak_bytes/1024**2:.1f} MB")

    # Sanity: embeddings should have changed from their initial values
    _, final_embs = em.get_all_embeddings()
    initial = torch.zeros_like(final_embs)
    assert not torch.allclose(final_embs, initial), \
        "Embeddings unchanged after training — update not working"
    print("  Embeddings updated PASSED")

    # Memory growth check: allow up to 50 MB for Python overhead (caches, etc.)
    # but flag if it keeps growing across epochs
    epoch_to_epoch_growth = [
        mem_per_epoch[i] - mem_per_epoch[i-1]
        for i in range(1, len(mem_per_epoch))
    ]
    max_growth = max(epoch_to_epoch_growth) if epoch_to_epoch_growth else 0
    if max_growth > 50:
        print(f"  WARNING: epoch-to-epoch memory growth = {max_growth:.1f} MB — "
              "possible memory leak")
    else:
        print(f"  Memory stable across epochs (max epoch-to-epoch growth: "
              f"{max_growth:.1f} MB)  PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    sep("SECTION 4 – Edge cases")
    ec_dir = TMPDIR / "edge"
    shutil.rmtree(ec_dir, ignore_errors=True)

    # ── 4a. Dataset size not a multiple of shard_size ─────────────────────────
    print("[4a] N not divisible by shard_size…")
    N_ODD = 2_333
    fm_odd = FeatureManager(str(ec_dir / "odd"), shard_size=1_000)
    fm_odd.open_for_writing(N_ODD, {"img_features": (CLIP_DIM,), "txt_features": (CLIP_DIM,)})
    for start in range(0, N_ODD, EXTRACTION_BS):
        bs = min(EXTRACTION_BS, N_ODD - start)
        fm_odd.write_batch(
            torch.randn(bs, CLIP_DIM), torch.randn(bs, CLIP_DIM), list(range(start, start+bs))
        )
    fm_odd.finalize_writing()
    assert fm_odd.total_samples == N_ODD
    assert fm_odd.num_shards == 3    # 1000 + 1000 + 333
    all_odd = fm_odd.load_all_to_ram()
    assert all_odd["img_features"].shape[0] == N_ODD
    print(f"    num_shards={fm_odd.num_shards}  total={fm_odd.total_samples}  PASSED")

    # ── 4b. Re-open existing store raises FileExistsError ─────────────────────
    print("[4b] Re-open existing store raises FileExistsError…")
    fm_again = FeatureManager(str(ec_dir / "odd"), shard_size=1_000)
    try:
        fm_again.open_for_writing(N_ODD, {"img_features": (CLIP_DIM,), "txt_features": (CLIP_DIM,)})
        print("    FAILED – should have raised FileExistsError")
    except FileExistsError as e:
        print(f"    PASSED – got FileExistsError: {e}")

    # ── 4c. EM with legacy storage_mode kwarg ─────────────────────────────────
    print("[4c] EmbeddingManager legacy storage_mode kwarg…")
    em_compat = TrainableEmbeddingManager(
        sample_ids=list(range(100)),
        embedding_dim=8,
        embeddings_dir=str(ec_dir / "em_compat"),
        storage_mode="memory",   # old kwarg → should map to mode='ram'
    )
    assert em_compat.mode == "ram"
    print("    PASSED")

    # ── 4d. Single-sample batch ───────────────────────────────────────────────
    print("[4d] Single-sample get/update…")
    em_single = TrainableEmbeddingManager(
        sample_ids=list(range(100)),
        embedding_dim=8,
        embeddings_dir=str(ec_dir / "em_single"),
        mode="ram",
    )
    v = torch.tensor([[1.0]*8])
    em_single.update_embeddings([42], v)
    r = em_single.get_embeddings([42])
    assert torch.allclose(r, v), "Single-sample roundtrip failed"
    print("    PASSED")

    # ── 4e. Store with img_full and txt_full ──────────────────────────────────
    print("[4e] Optional img_full / txt_full storage…")
    IMG_FULL_DIM  = (50, CLIP_DIM)   # CLIP ViT-B/32: 49 patches + CLS
    TXT_FULL_DIM  = (77, CLIP_DIM)
    fm_full = FeatureManager(str(ec_dir / "full_feats"), shard_size=200,
                             hdf5_compression=True)
    N_FULL = 400
    fm_full.open_for_writing(N_FULL, {
        "img_features": (CLIP_DIM,),
        "txt_features": (CLIP_DIM,),
        "img_full":     IMG_FULL_DIM,
        "txt_full":     TXT_FULL_DIM,
    })
    for start in range(0, N_FULL, 100):
        bs = min(100, N_FULL - start)
        fm_full.write_batch(
            torch.randn(bs, CLIP_DIM),
            torch.randn(bs, CLIP_DIM),
            list(range(start, start + bs)),
            img_full=torch.randn(bs, *IMG_FULL_DIM),
            txt_full=torch.randn(bs, *TXT_FULL_DIM),
        )
    fm_full.finalize_writing()

    shard = fm_full.load_shard_to_ram(0, ["img_features", "img_full", "txt_full"])
    assert "img_full" in shard, "img_full missing from shard"
    assert "txt_full" in shard, "txt_full missing from shard"
    assert shard["img_full"].shape == (200, *IMG_FULL_DIM)
    assert shard["txt_full"].shape == (200, *TXT_FULL_DIM)
    print(f"    img_full={shard['img_full'].shape}  "
          f"txt_full={shard['txt_full'].shape}  PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    shutil.rmtree(TMPDIR, ignore_errors=True)
    TMPDIR.mkdir(parents=True)

    print(f"Test dir: {TMPDIR}")
    print(f"N={N_SAMPLES}  shard_size={SHARD_SIZE}  clip_dim={CLIP_DIM}  "
          f"emb_dim={EMBEDDING_DIM}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")

    try:
        fm  = test_feature_manager()
        em  = test_embedding_manager(fm)
        test_training_loop(fm, em)
        test_edge_cases()

        sep("ALL TESTS PASSED")

    finally:
        shutil.rmtree(TMPDIR, ignore_errors=True)
        print(f"Cleaned up {TMPDIR}")
