"""Small provenance and atomic-output helpers, usable in either environment."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def array_sha(array):
    """Hash dtype and shape as well as the contiguous array bytes."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with tmp.open('xb' if binary else 'x') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'))
