"""JSON output regressions using temporary files only."""
import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402


def test_write_json_preserves_content_and_normal_mode(tmp_path):
    obj = {'z': [1, {'b': True, 'a': None}], 'a': 'value'}
    previous = tmp_path / 'previous.json'
    previous.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    target = tmp_path / 'new' / 'target.json'
    P.write_json(target, obj)
    assert target.read_bytes() == previous.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(previous.stat().st_mode)
    assert list(target.parent.iterdir()) == [target]


def test_write_json_parallel_calls_replace_from_unique_sibling_files(tmp_path, monkeypatch):
    target = tmp_path / 'target.json'
    payloads = [dict(call=i, text='x' * 2048) for i in range(12)]
    paths = []
    replace = os.replace

    def checked_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.parent == target.parent and source != destination == target
        assert json.loads(source.read_text()) in payloads
        paths.append(source)
        replace(source, destination)

    monkeypatch.setattr(P.os, 'replace', checked_replace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda obj: P.write_json(target, obj), payloads))
    assert len(paths) == len(set(paths)) == len(payloads)
    assert json.loads(target.read_text()) in payloads
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('failure', ['dumps', 'replace'])
def test_write_json_failure_keeps_old_file_and_cleans_temp(tmp_path, monkeypatch, failure):
    target = tmp_path / 'target.json'
    target.write_text('old content\n')

    def fail(*args, **kwargs):
        raise ValueError('synthetic write failure')

    monkeypatch.setattr(P.json if failure == 'dumps' else P.os,
                        'dumps' if failure == 'dumps' else 'replace', fail)
    with pytest.raises(ValueError, match='synthetic write failure'):
        P.write_json(target, {'new': 'content'})
    assert target.read_text() == 'old content\n'
    assert list(tmp_path.iterdir()) == [target]
