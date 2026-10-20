"""Local safety checks for sync/fetch; every cluster transport is mocked.

Fixture files are retained under the authorized notes folder (no deletions).
"""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import das6_sync_ct2 as sync

FIXTURES = Path('/project/CoSiR/.scratch/r10-claim-test-s2/node/plumbing_test_fixture')


class Plumbing(unittest.TestCase):
    def test_remote_path_guard(self):
        for path in ('/etc/passwd', '../outside', 'outputs/../../outside'):
            with self.assertRaises(ValueError):
                sync.remote_path(path)
        self.assertEqual(sync.remote_path('env'), sync.NR + '/env')

    def test_exclusions_match_transfer(self):
        base = FIXTURES / 'inputs'
        for name in ('keep/file', 'stale/file', 'stage2/file',
                     'lib/python3.11/site-packages/cupy/file',
                     'lib/python3.11/site-packages/numpy/file'):
            p = base / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')
        item = dict(local=str(base), exclude=['stale','stage2'], exclude_env=True)
        expected = sync.files(item)
        self.assertEqual(set(expected), {'keep/file', 'lib/python3.11/site-packages/numpy/file'})
        for name in ('keep/file','stale/file','stage2/file',
                     'lib/python3.11/site-packages/cupy/file'):
            self.assertEqual(sync.included(item, name), name in expected)
        self.assertIn('/stale/***', sync.exclusions(item))
        self.assertIn('lib/python*/site-packages/cupy/***', sync.exclusions(item))

    def test_fetch_conflicts_never_transfer(self):
        root = FIXTURES / 'conflict'
        target = root / 'src/test/20261020_claim_test/outputs/stage2/a.txt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('local data')
        remote = {'a.txt': {'bytes': 6, 'sha256': hashlib.sha256(b'remote').hexdigest()}}
        with patch.object(sync, 'ROOT', root), patch.object(sync, 'inventory', return_value=remote), \
             patch.object(sync.cluster, 'require_local') as transfer:
            with self.assertRaisesRegex(SystemExit, 'REFUSING.*differing local files'):
                sync.fetch('node408')
            transfer.assert_not_called()
        self.assertEqual(target.read_text(), 'local data')

    def test_fetch_equal_file_is_protected_and_verified(self):
        root = FIXTURES / 'equal'
        target = root / 'src/test/20261020_claim_test/outputs/stage2/a.txt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('same')
        remote = {'a.txt': {'bytes': 4, 'sha256': hashlib.sha256(b'same').hexdigest()}}
        with patch.object(sync, 'ROOT', root), patch.object(sync, 'inventory', return_value=remote), \
             patch.object(sync.cluster, 'require_local') as transfer, contextlib.redirect_stdout(io.StringIO()):
            sync.fetch('node408')
            argv = transfer.call_args.args[0]
            self.assertIn('--ignore-existing', argv)
            self.assertEqual(argv[-2], 'node408:' + sync.NR + '/outputs/stage2/')
            self.assertEqual(target.read_text(), 'same')

    def test_fetch_traversal_refused(self):
        remote = {'../escape': {'bytes': 0, 'sha256': ''}}
        with patch.object(sync, 'inventory', return_value=remote), \
             patch.object(sync.cluster, 'require_local') as transfer:
            with self.assertRaisesRegex(ValueError, 'unsafe fetch path'):
                sync.fetch('node408')
            transfer.assert_not_called()

    def test_plan_never_uses_transport(self):
        for mode in ('--inputs','--fetch','--verify'):
            with self.subTest(mode=mode), patch.object(sys, 'argv', ['sync','--node','node408',mode]), \
                 patch.object(sync, 'actions', return_value=[]), \
                 patch.object(sync.cluster, 'remote', side_effect=AssertionError('node contact')), \
                 patch.object(sync.cluster, 'require_local', side_effect=AssertionError('transfer')), \
                 contextlib.redirect_stdout(io.StringIO()):
                sync.main()


if __name__ == '__main__':
    unittest.main()
