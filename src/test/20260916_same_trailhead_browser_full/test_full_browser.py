"""Focused contract tests for the full-population browser modules."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException


HERE = Path(__file__).parent


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_edge_labels_follow_typed_graph_keys():
    builder = load_module("build_full_index")
    typed = {
        "keys": np.array([1, 2, 7, 19]),
        "txt_only": np.array([False, True, False, False]),
        "img_only": np.array([False, False, True, False]),
        "both": np.array([False, False, False, True]),
        "repair": np.array([True, False, False, False]),
    }
    labels = builder._edge_type_codes(typed, 5, np.array([0, 0, 1, 3, 0]), np.array([4, 2, 2, 4, 3]))
    assert labels.tolist() == [0, 1, 2, 3, 0]


def test_subreddit_is_extracted_from_redcaps_image_path():
    server = load_module("server")
    assert server._subreddit({"image": "redcaps/images2020/antiques/kkbn86.jpg"}) == "antiques"


@pytest.mark.parametrize("bucket,index,message", [
    ("missing", 0, "unknown bucket"),
    ("both", -1, "index must be within"),
    ("both", 2, "index must be within"),
])
def test_row_lookup_rejects_invalid_bucket_or_index(bucket, index, message):
    server = load_module("server")
    server.BUCKET_ROWS = {"both": np.array([9, 11], dtype=np.int64)}
    with pytest.raises(ValueError, match=message):
        server._global_row(bucket, index)


def test_random_endpoint_rejects_unknown_bucket_clearly():
    server = load_module("server")
    server.BUCKET_ROWS = {"both": np.array([9], dtype=np.int64)}
    with pytest.raises(HTTPException, match="unknown bucket"):
        server.random_example("missing")


def test_cache_validation_rejects_unknown_edge_code():
    server = load_module("server")
    arrays = {name: np.zeros(2, dtype=np.int32) for name in server.ARRAY_NAMES}
    arrays["edge_type_code"] = np.array([0, 4], dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown edge-type codes"):
        server._validate_cache(arrays, {"row_count": 2, "buckets": []})
