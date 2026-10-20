"""Local safety checks for sync/fetch; every cluster transport is mocked.

Fixture files are retained under the authorized notes folder (no deletions).
"""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
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
            self.assertEqual(transfer.call_args.args[2], 6 * 3600)
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

    def test_possible_inputs_require_opt_in(self):
        items = [dict(key=certainty, certainty=certainty, files_count=1, bytes=7,
                      remote=sync.remote_path(certainty)) for certainty in ('certain', 'possible')]
        for mode in ('--inputs', '--verify'):
            for with_possible in (False, True):
                for run in (False, True):
                    argv = ['sync', '--node', 'node408', mode]
                    if with_possible:
                        argv.append('--with-possible')
                    if run:
                        argv.append('--run')
                    output = io.StringIO()
                    with self.subTest(mode=mode, with_possible=with_possible, run=run), \
                         patch.object(sys, 'argv', argv), patch.object(sync, 'actions', return_value=items), \
                         patch.object(sync, 'sync') as copy, \
                         patch.object(sync, 'inventory', return_value={'x': {'bytes': 7}}) as inventory, \
                         patch.object(sync, 'files', return_value={'x': 7}), \
                         patch.object(sync.cluster, 'remote', side_effect=AssertionError('node contact')), \
                         contextlib.redirect_stdout(output):
                        sync.main()
                        text = output.getvalue()
                        self.assertEqual('possible: skipped (possible), 7 bytes' in text, not with_possible)
                        self.assertIn(f'Total: {14 if with_possible else 7} bytes', text)
                        expected = 2 if with_possible else 1
                        self.assertEqual(copy.call_count, expected if run and mode == '--inputs' else 0)
                        self.assertEqual(inventory.call_count, expected if run and mode == '--verify' else 0)

    def test_sync_allows_long_transfer(self):
        item = dict(key='env', kind='dir', remote=sync.remote_path('env'), local='/unused')
        with patch.object(sync.cluster, 'remote'), patch.object(sync.cluster, 'require_remote'), \
             patch.object(sync.cluster, 'require_local') as transfer:
            sync.sync('node408', item)
            self.assertEqual(transfer.call_args.args[2], 6 * 3600)

    def test_launcher_has_no_cuda_visible_devices_assignment(self):
        script = Path(__file__).with_name('run_ct2_node.sh').read_text()
        active = '\n'.join(line for line in script.splitlines() if not line.lstrip().startswith('#'))
        assignment = re.compile(r'(^|&&|\|\||;|then|do)\s*(export\s+|env\s+)?CUDA_VISIBLE_DEVICES=', re.M)
        self.assertIsNone(assignment.search(active))

    def test_run_gpu_guard_before_namespace(self):
        root = Path(__file__).resolve().parents[1]
        for device in ('', '1', '0,1', '0'):
            for mode in ('run', 'check', 'smoke'):
                with self.subTest(mode=mode, device=device):
                    result = subprocess.run(['bash', 'scripts/run_ct2_node.sh', mode, '--run'],
                                            cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 1)
                    if mode == 'run' and device != '0':
                        self.assertIn('the frozen queue pins GPU 0; launch when slot 0 is free', result.stderr)
                    else:
                        self.assertIn('requires /local/wding/jobs/<tag>/code', result.stderr)

    def test_missing_possible_bind_is_skipped(self):
        script = Path(__file__).with_name('run_ct2_node.sh').read_text()
        function = 'bind_path() {' + script.split('bind_path() {', 1)[1].split('\n}', 1)[0] + '\n}'
        for certainty in ('possible', 'certain'):
            with self.subTest(certainty=certainty):
                result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + function +
                                         '\nbind_path "$1" "$2" "$3"', 'test',
                                         str(FIXTURES / 'absent-bind-source'), '/optional-target', certainty],
                                        capture_output=True, text=True, timeout=10)
                if certainty == 'possible':
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.strip(),
                                     'skip possible bind: /optional-target (source absent)')
                else:
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('Missing bind source:', result.stderr)


if __name__ == '__main__':
    unittest.main()
