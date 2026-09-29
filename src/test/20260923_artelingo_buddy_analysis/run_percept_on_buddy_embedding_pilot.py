"""Screen faithful PercepT Variant A on frozen, aligned Attention-h1 vectors.

This script is intentionally a single seed-42 GPU pilot. It writes its report
and snapshot only when explicitly run by the experimenter.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.metrics import silhouette_score


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PERCEPT_DIR = os.path.abspath(os.path.join(
    OUT_DIR, "..", "20260922_percept_topic_pipeline"
))
TEMPLATE_PATH = os.path.join(
    PERCEPT_DIR, "run_percept_stage1_faithful_recipe_snapshot_pilot.py"
)
BUDDY_PATH = os.path.join(OUT_DIR, "attention_h1_embedding_snapshot.npz")
REPORT_PATH = os.path.join(OUT_DIR, "percept_on_buddy_embedding_pilot_report.md")
SNAPSHOT_PATH = os.path.join(OUT_DIR, "percept_on_buddy_embedding_snapshot.npz")

# The template imports the faithful recipe and its base pilot as siblings.
# Importing it does not run either pilot's main function.


def load_template():
    spec = importlib.util.spec_from_file_location(
        "percept_faithful_snapshot_for_buddy_input", TEMPLATE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import snapshot template from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


template = load_template()
faithful = template.faithful
base = faithful.base
SEED = faithful.SEED
VARIANT = "A"
N_INITIAL_CLUSTERS = faithful.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = faithful.N_SURVIVING_CLUSTERS
EMOTION_PARETO_BAR = faithful.EMOTION_PARETO_BAR
GENRE_PARETO_BAR = faithful.GENRE_PARETO_BAR

# Cited seed-42 rows, not the four-seed mean. See the reports named below.
BUDDY_EMOTION_AMI = 0.1249
BUDDY_GENRE_AMI = 0.2404
BUDDY_NATIVE_SILHOUETTE = 0.0377
ORIGINAL_EMOTION_AMI = 0.1092
ORIGINAL_GENRE_AMI = 0.3288
ORIGINAL_LATENT_SILHOUETTE = 0.5120
ORIGINAL_FINAL_PRETRAIN = 0.000045
ORIGINAL_FINAL_RECONSTRUCTION = 0.000080
ORIGINAL_FINAL_KL = 0.391414
ORIGINAL_HELDOUT_MAX = 1251
ORIGINAL_HELDOUT_MEDIAN = 13.0


def align_snapshot_split(snapshot, canonical_paintings, split, expected_count):
    """Check the population with the template helper, then reorder all arrays."""
    saved_paintings = snapshot[f"{split}_paintings"]
    template.assert_matching_paintings(
        canonical_paintings, saved_paintings, split, expected_count
    )
    saved_index = {painting: index for index, painting in enumerate(saved_paintings)}
    order = np.fromiter(
        (saved_index[painting] for painting in canonical_paintings),
        dtype=np.intp,
        count=expected_count,
    )
    embedding = np.asarray(snapshot[f"{split}_embedding_post"][order])
    if embedding.shape != (expected_count, 32) or embedding.dtype != np.float32:
        raise RuntimeError(
            f"{split} embedding must be float32 ({expected_count}, 32); "
            f"got {embedding.shape} {embedding.dtype}."
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError(f"{split} embedding contains non-finite values.")
    norms = np.linalg.norm(embedding, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"{split} embedding is not unit-normalized.")
    return {
        "order": order,
        "embedding": np.ascontiguousarray(embedding),
        "community": np.asarray(snapshot[f"{split}_community_post"][order]),
        "emotion": np.asarray(snapshot[f"{split}_emotion"][order]),
        "genre": np.asarray(snapshot[f"{split}_genre"][order]),
    }


def sampled_native_silhouette(embedding, assignments):
    """Apply the buddy pilots' seeded 6,000-then-4,000 convention."""
    idx = np.random.default_rng(42).choice(
        len(embedding), size=min(6000, len(embedding)), replace=False
    )
    if len(np.unique(assignments[idx])) < 2:
        return float("nan")
    try:
        return float(silhouette_score(
            embedding[idx], assignments[idx],
            sample_size=min(4000, len(idx)), random_state=42,
        ))
    except ValueError as exc:
        # A heavily collapsed partition can lose its second label in the
        # inner 4,000-point sample, making silhouette undefined.
        if "Number of labels" not in str(exc):
            raise
        return float("nan")


def occupancy_text(sizes, threshold_n, fixed_centers=True):
    below = int(np.sum(sizes < 0.01 * threshold_n))
    zeros = int(np.sum(sizes == 0))
    collapsed = below > len(sizes) / 2 if fixed_centers else None
    text = (
        f"min {int(sizes.min()):,}; max {int(sizes.max()):,}; "
        f"median {float(np.median(sizes)):.1f}; zero {zeros}; "
        f"below 1% {below}/{len(sizes)}"
    )
    if fixed_centers:
        text += f"; {'collapsed' if collapsed else 'not collapsed'}"
    else:
        text += "; Leiden (67-center rule n/a)"
    return text


def format_silhouette(value):
    return f"{value:.4f}" if np.isfinite(value) else "n/a (one cluster)"


def write_report(result, buddy_occupancy, buddy_community_count):
    """Write the complete comparison after the single GPU fit."""
    train_metrics = result["train_metrics"]
    heldout = result["heldout_metrics"]
    native_silhouette = result["native_silhouette"]
    pretrain_final = result["pretrain_losses"][-1]["reconstruction"]
    dec_final = result["dec_losses"][-1]
    recon = dec_final["reconstruction"]
    kl = dec_final["kl"]
    ratio = recon / kl if np.isfinite(kl) and kl > 0 else float("nan")
    original_ratio = ORIGINAL_FINAL_RECONSTRUCTION / ORIGINAL_FINAL_KL
    factor = ratio / original_ratio
    if not np.isfinite(factor):
        loss_shift = "undefined because the new loss ratio is non-finite"
    elif factor >= 3 or factor <= 1 / 3:
        loss_shift = "shifted substantially"
    else:
        loss_shift = "similar in scale"
    sizes = heldout["cluster_sizes"]
    train_sizes = train_metrics["cluster_sizes"]
    new_occupancy = occupancy_text(sizes, len(result["heldout_h"]))
    train_occupancy = occupancy_text(train_sizes, len(result["train_h"]))
    original_occupancy = (
        f"min 0; max {ORIGINAL_HELDOUT_MAX:,}; "
        f"median {ORIGINAL_HELDOUT_MEDIAN:.1f}; zero n/a; "
        "below 1% 50/67; collapsed"
    )
    emotion = heldout["emotion"]["AMI"]
    genre = heldout["genre"]["AMI"]
    clears_bar = emotion > EMOTION_PARETO_BAR and genre > GENRE_PARETO_BAR
    native_beats = np.isfinite(native_silhouette) and native_silhouette > BUDDY_NATIVE_SILHOUETTE
    latent_separable = (
        np.isfinite(heldout["silhouette"]) and heldout["silhouette"] > 0
    )
    useful_topics = not heldout["collapsed"] and clears_bar and latent_separable
    beats_buddy = (
        not heldout["collapsed"] and native_beats
        and emotion >= BUDDY_EMOTION_AMI and genre >= BUDDY_GENRE_AMI
    )
    approaches_original = (
        emotion >= 0.9 * ORIGINAL_EMOTION_AMI
        and genre >= 0.9 * ORIGINAL_GENRE_AMI
        and np.isfinite(heldout["silhouette"])
        and heldout["silhouette"] >= 0.9 * ORIGINAL_LATENT_SILHOUETTE
    )
    lines = [
        "# PercepT faithful Variant A on frozen buddy embeddings\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "Seed 42 only. Canonical train and held-out painting orders came from "
        "the respective `pipeline.load_dedup_features()` calls. The painting "
        "sets and uniqueness were checked against the buddy snapshot with "
        "the faithful snapshot template's `assert_matching_paintings`, then "
        "all saved buddy arrays were reindexed to those orders. For the "
        "native-space sampled silhouette, DEC labels were mapped back "
        "to the snapshot's saved held-out order so the seed-42 indices "
        "match the cited buddy baseline. The frozen "
        "float32, unit-normalized 32-D `embedding_post` vectors replaced "
        "PercepT's 2,816-D fused input h entirely; no buddy training or "
        "affect extraction was repeated.\n\n",
        "The faithful recipe's own seed-threaded pretraining, K-means "
        "initialization, joint DEC training, and assignment evaluation "
        "functions were called unmodified. Variant A uses noise 0.1 in "
        f"pretraining only (noise={faithful.NOISE_STD:g}), "
        f"lambda_R={faithful.LAMBDA_RECONSTRUCTION:g}, "
        "100 initial centers and 67 "
        "highest-norm survivors. The unchanged autoencoder maps "
        "32→500→500→2000→128→2000→500→500→32. Its 128-D latent is "
        "overcomplete relative to the 32-D input. Raw per-coordinate MSE "
        "and the pretraining noise-to-signal scale can therefore differ; "
        "lambda_R=1 is audited "
        "below, not presumed equivalent to the original run.\n\n",
        "Point (a) uses the seed-42 buddy Leiden AMIs from "
        "`attention_h1_embedding_snapshot_pilot_report.md` and the "
        "matching seed-42 sampled silhouette from "
        "`attention_h1_baseline_seed_stress_pilot_report.md`; buddy occupancy "
        "is counted from that snapshot's saved held-out labels. "
        "Point (b) uses the faithful full-held-out 128-D latent evaluator. "
        "Point (c) keeps exactly the labels from (b), but scores silhouette "
        "on original 32-D held-out vectors with a seeded draw of at most "
        "6,000 and `silhouette_score(sample_size=min(4000, len(idx)), "
        "random_state=42)`. AMI and occupancy are identical in (b) and (c). "
        "The original reference uses the 2,816-D Variant A held-out row "
        "of `percept_stage1_faithful_recipe_pilot_report.md`. Full latent "
        "silhouettes and sampled native silhouettes have different "
        "protocols and spaces.\n\n",
        f"Held-out Pareto bar: emotion AMI > {EMOTION_PARETO_BAR:.4f} "
        f"and genre AMI > {GENRE_PARETO_BAR:.4f}. The verdict below calls "
        "topics useful only when both bars clear, 67-center occupancy is "
        "not collapsed, and the full 128-D latent silhouette is positive. "
        "Whether the partition improves buddy's native geometry is a "
        "separate verdict below. "
        "For the numerical 'approaches original' comparison, all three "
        "128-D metrics must reach at least 90% of the original row; this "
        "is an exploratory screen threshold.\n\n",
        "## Training and cluster-size trajectories\n\n",
        f"### Variant A ({faithful.VARIANTS[VARIANT]}), seed {SEED}\n\n",
        f"- Pretraining reconstruction and LR: "
        f"{faithful.format_pretrain_losses(result['pretrain_losses'])}.\n",
        f"- DEC stopped via **{result['stop_reason']}** at epoch "
        f"**{result['stop_epoch']}** (threshold "
        f"`fraction_changed < {faithful.STABILITY_THRESHOLD:.3f}`, ceiling "
        f"{faithful.MAX_DEC_EPOCHS}).\n",
        f"- Joint DEC losses and LR: "
        f"{faithful.format_dec_losses(result['dec_losses'])}.\n",
        "- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): "
        f"{faithful.format_trajectory(result['trajectory'])}.\n\n",
        "## Loss-scale audit\n\n",
        "MSE is mean per input coordinate. The ratio is final DEC "
        "reconstruction / final DEC KL; the original cited losses were "
        "from epoch 154 and rounded to six decimal places. For unit-norm inputs, "
        "MSE × input dimension is reconstruction squared error relative "
        "to the input's unit squared norm.\n\n",
        "| input | final pretrain reconstruction | final DEC reconstruction "
        "| final DEC KL | reconstruction / KL | final DEC MSE × dim |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| buddy 32-D | {pretrain_final:.8f} | {recon:.8f} | "
        f"{kl:.8f} | {ratio:.6g} | {recon * 32:.6f} |\n",
        f"| original 2,816-D | {ORIGINAL_FINAL_PRETRAIN:.6f} | "
        f"{ORIGINAL_FINAL_RECONSTRUCTION:.6f} | "
        f"{ORIGINAL_FINAL_KL:.6f} | {original_ratio:.6g} | "
        f"{ORIGINAL_FINAL_RECONSTRUCTION * 2816:.6f} |\n\n",
        f"The new ratio is {factor:.2f}× the original ratio: "
        f"**{loss_shift}** under a predeclared threefold scale check. "
        "These loss values alone do not measure gradient strength.\n\n",
        "## Held-out comparison\n\n",
        "| evaluation point (seed 42) | emotion AMI | genre AMI | silhouette "
        "| occupancy |\n",
        "|---|---:|---:|---:|---|\n",
        f"| (a) buddy Leiden, native 32-D | {BUDDY_EMOTION_AMI:.4f} | "
        f"{BUDDY_GENRE_AMI:.4f} | {BUDDY_NATIVE_SILHOUETTE:.4f} "
        f"(sampled) | {buddy_occupancy} |\n",
        f"| (b) DEC labels, PercepT 128-D latent | {emotion:.4f} | "
        f"{genre:.4f} | {format_silhouette(heldout['silhouette'])} "
        f"(full) | {new_occupancy} |\n",
        f"| (c) same DEC labels, buddy native 32-D | {emotion:.4f} | "
        f"{genre:.4f} | {format_silhouette(native_silhouette)} "
        f"(sampled) | {new_occupancy} |\n",
        f"| original 2,816-D PercepT Variant A, 128-D latent | "
        f"{ORIGINAL_EMOTION_AMI:.4f} | {ORIGINAL_GENRE_AMI:.4f} | "
        f"{ORIGINAL_LATENT_SILHOUETTE:.4f} (full) | "
        f"{original_occupancy} |\n\n",
        f"New DEC train occupancy: {train_occupancy}. Held-out genre "
        f"n={heldout['genre_count']:,}. Original zero count was not "
        "reported; its minimum of zero does establish at least one empty "
        f"center. Buddy's own Leiden labels use {buddy_community_count} "
        "observed held-out "
        "communities, so the 67-center collapse criterion does not apply.\n\n",
        "## Verdict\n\n",
        f"- Compressed-input signal: **{'yes' if useful_topics else 'no'}** "
        f"by the stated screen. DEC held-out AMI={emotion:.4f}/{genre:.4f}, "
        f"128-D silhouette={format_silhouette(heldout['silhouette'])}, and "
        f"{heldout['small_clusters']}/{N_SURVIVING_CLUSTERS} centers "
        f"below 1% ({'collapsed' if heldout['collapsed'] else 'not collapsed'}).\n",
        f"- Beats buddy's own Leiden partition in native space: "
        f"**{'yes' if beats_buddy else 'no'}** by noncollapse, both AMIs "
        f"at least matching {BUDDY_EMOTION_AMI:.4f}/{BUDDY_GENRE_AMI:.4f}, "
        f"and native silhouette exceeding {BUDDY_NATIVE_SILHOUETTE:.4f}. "
        f"Observed AMI differences={emotion - BUDDY_EMOTION_AMI:+.4f}/"
        f"{genre - BUDDY_GENRE_AMI:+.4f}; native silhouette "
        f"difference={native_silhouette - BUDDY_NATIVE_SILHOUETTE:+.4f}.\n",
        f"- Approaches original 2,816-D result numerically: "
        f"**{'yes' if approaches_original else 'no; falls short' }** "
        f"on the 90%-of-each-metric screen. Differences in held-out "
        f"AMI={emotion - ORIGINAL_EMOTION_AMI:+.4f}/"
        f"{genre - ORIGINAL_GENRE_AMI:+.4f}; full 128-D silhouette "
        f"difference={heldout['silhouette'] - ORIGINAL_LATENT_SILHOUETTE:+.4f}. "
        "The original itself was collapsed (50/67 below 1%).\n",
    ]
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    if SEED != 42 or VARIANT not in faithful.VARIANTS or VARIANT != "A":
        raise RuntimeError("This screen is specified for Variant A, seed 42 only.")
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_buddy_train", base.PIPELINE_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_buddy_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log
    paintings, _img, _txt, emotion_counts = pipeline.load_dedup_features()
    heldout_paintings, _heldout_img, _heldout_txt, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    genre_map = pipeline.load_genre_map()
    with np.load(BUDDY_PATH, allow_pickle=True) as snapshot:
        if int(snapshot["seed"]) != SEED:
            raise RuntimeError(f"Buddy snapshot seed is not {SEED}.")
        train = align_snapshot_split(snapshot, paintings, "train", 61402)
        heldout = align_snapshot_split(snapshot, heldout_paintings, "heldout", 9365)
    for name, canonical, ids, aligned in (
        ("train", train_emotions, paintings, train),
        ("heldout", heldout_emotions, heldout_paintings, heldout),
    ):
        if not np.array_equal(aligned["emotion"], np.asarray(canonical, dtype=object)):
            raise RuntimeError(f"{name} snapshot emotions disagree with canonical labels.")
        expected_genres = np.asarray([genre_map.get(p, "") for p in ids], dtype=object)
        if not np.array_equal(aligned["genre"], expected_genres):
            raise RuntimeError(f"{name} snapshot genres disagree with canonical labels.")
    train_h = train["embedding"]
    heldout_h = heldout["embedding"]
    log("Aligned seed-42 buddy embeddings to 61,402 train and 9,365 held-out paintings.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for faithful Variant A on frozen 32-D buddy inputs.")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    pretrain_losses = faithful.pretrain_autoencoder(
        encoder, decoder, train_inputs, device, log, SEED
    )
    centers = faithful.initialize_cluster_centers(encoder, train_inputs, device, SEED)
    dec_losses, trajectory, stop_reason, stop_epoch = faithful.train_dec_until_stable(
        encoder, decoder, centers, train_inputs, device, log, SEED, VARIANT
    )
    surviving_centers, surviving_indices = base.prune_centers(centers)
    log(
        f"DEC stopped at epoch {stop_epoch} via {stop_reason}; retained "
        f"{N_SURVIVING_CLUSTERS}/{N_INITIAL_CLUSTERS} centers."
    )
    encoder.eval()
    with torch.no_grad():
        train_latent = encoder(train_inputs.to(device))
        train_assignments = base.soft_assignments(
            train_latent, surviving_centers
        ).argmax(dim=1).cpu().numpy()
        heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
        heldout_assignments = base.soft_assignments(
            heldout_latent, surviving_centers
        ).argmax(dim=1).cpu().numpy()
    train_latent = train_latent.cpu().numpy()
    heldout_latent = heldout_latent.cpu().numpy()
    train_metrics = faithful.evaluate_assignments(
        pipeline, paintings, train_emotions, train_latent, train_assignments
    )
    heldout_metrics = faithful.evaluate_assignments(
        heldout_pipeline, heldout_paintings, heldout_emotions,
        heldout_latent, heldout_assignments,
    )
    # Reverse the canonical reindexing for the native-space score. This keeps
    # the seeded sample indices identical to the buddy baseline's NPZ order.
    snapshot_to_canonical = np.argsort(heldout["order"])
    native_silhouette = sampled_native_silhouette(
        heldout_h[snapshot_to_canonical],
        heldout_assignments[snapshot_to_canonical],
    )
    buddy_sizes = np.unique(heldout["community"], return_counts=True)[1]
    buddy_community_count = len(buddy_sizes)
    buddy_occupancy = occupancy_text(buddy_sizes, len(heldout_h), fixed_centers=False)
    final = dec_losses[-1]
    original_ratio = ORIGINAL_FINAL_RECONSTRUCTION / ORIGINAL_FINAL_KL
    new_ratio = (
        final["reconstruction"] / final["kl"]
        if np.isfinite(final["kl"]) and final["kl"] > 0 else float("nan")
    )
    log(
        f"Loss audit: pretrain recon={pretrain_losses[-1]['reconstruction']:.8f}; "
        f"DEC recon={final['reconstruction']:.8f}, KL={final['kl']:.8f}, "
        f"recon/KL={new_ratio:.6g}; original pretrain recon="
        f"{ORIGINAL_FINAL_PRETRAIN:.6f}, DEC recon="
        f"{ORIGINAL_FINAL_RECONSTRUCTION:.6f}, KL={ORIGINAL_FINAL_KL:.6f}, "
        f"recon/KL={original_ratio:.6g}."
    )
    ratio_factor = new_ratio / original_ratio
    if np.isfinite(ratio_factor):
        log(f"Loss audit: new reconstruction/KL ratio is {ratio_factor:.2f}x original.")
    else:
        log("Loss audit: new reconstruction/KL ratio is undefined or non-finite.")
    result = {
        "pretrain_losses": pretrain_losses, "dec_losses": dec_losses,
        "trajectory": trajectory, "stop_reason": stop_reason,
        "stop_epoch": stop_epoch, "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics, "native_silhouette": native_silhouette,
        "train_h": train_h, "heldout_h": heldout_h,
    }
    np.savez_compressed(
        SNAPSHOT_PATH,
        train_paintings=np.asarray(paintings, dtype=object),
        train_input=train_h,
        train_latent=train_latent,
        train_topic=train_assignments,
        heldout_paintings=np.asarray(heldout_paintings, dtype=object),
        heldout_input=heldout_h,
        heldout_latent=heldout_latent,
        heldout_topic=heldout_assignments,
        surviving_indices=surviving_indices,
        seed=SEED,
        variant=VARIANT,
        heldout_native_silhouette=native_silhouette,
        stop_epoch=stop_epoch,
        stop_reason=stop_reason,
        pretrain_losses=np.asarray(pretrain_losses, dtype=object),
        dec_losses=np.asarray(dec_losses, dtype=object),
        cluster_trajectory=np.asarray(trajectory, dtype=object),
        train_metrics=np.asarray(train_metrics, dtype=object),
        heldout_metrics=np.asarray(heldout_metrics, dtype=object),
    )
    write_report(result, buddy_occupancy, buddy_community_count)
    log(f"Wrote {REPORT_PATH} and {SNAPSHOT_PATH}.")


if __name__ == "__main__":
    main()
