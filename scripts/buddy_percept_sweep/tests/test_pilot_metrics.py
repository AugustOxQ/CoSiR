from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from scripts.buddy_percept_sweep.pilot_metrics import (
    PilotModules, ami_emotion_genre, independent_partition,
)


def test_ami_emotion_genre_perfect_agreement():
    labels = np.array([0, 0, 1, 1, 2, 2])
    emotion = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    genre = np.array(["x", "x", "y", "y", "z", "z"], dtype=object)
    emo, gen = ami_emotion_genre(labels, emotion, genre)
    assert emo == pytest.approx(1.0)
    assert gen == pytest.approx(1.0)


def test_ami_emotion_genre_excludes_empty_genre_rows():
    labels = np.array([0, 0, 1, 1, 0, 1])
    emotion = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
    # The two rows with "" genre disagree with the partition; if they were
    # counted, genre AMI could not be 1.0.
    genre = np.array(["x", "x", "y", "y", "", ""], dtype=object)
    emo, gen = ami_emotion_genre(labels, emotion, genre)
    assert emo == pytest.approx(1.0)
    assert gen == pytest.approx(1.0)


def test_ami_emotion_genre_single_label_among_genre_rows_gives_zero():
    labels = np.array([0, 0, 0, 1, 1, 1])
    emotion = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    genre = np.array(["x", "x", "x", "", "", ""], dtype=object)
    emo, gen = ami_emotion_genre(labels, emotion, genre)
    assert emo == pytest.approx(1.0)
    assert gen == 0.0


def _stub_modules(seen):
    def build_single_modality_graph(name, embedding, pipeline, affect_pilot, device, expected_nodes):
        seen.append((name, pipeline, expected_nodes, device))
        clique = np.ones((4, 4)) - np.eye(4)
        return csr_matrix(block_diag([clique, clique]).toarray())

    return PilotModules(
        pipeline="train-pipeline", heldout_pipeline="heldout-pipeline", affect_pilot="affect",
        single_modality=SimpleNamespace(build_single_modality_graph=build_single_modality_graph),
    )


@pytest.mark.parametrize("split,pipeline_name", [("train", "train-pipeline"), ("heldout", "heldout-pipeline")])
def test_independent_partition_finds_two_cliques(split, pipeline_name):
    seen = []
    labels = independent_partition(np.zeros((8, 3), dtype=np.float32), _stub_modules(seen), split, 42, "cpu")
    assert labels.dtype == np.int64
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:4].tolist())) == 1 and len(set(labels[4:].tolist())) == 1
    assert seen == [(f"{split}-independent", pipeline_name, 8, "cpu")]


def test_independent_partition_rejects_bad_split():
    with pytest.raises(ValueError):
        independent_partition(np.zeros((8, 3)), _stub_modules([]), "val", 42, "cpu")
