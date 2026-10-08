"""Black-box checks for bin/codex-usage using a local app-server double."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

REPOSITORY = next(parent for parent in Path(__file__).resolve().parents
                  if (parent / '.git').exists())
SCRIPT = REPOSITORY / 'bin' / 'codex-usage'
FAKE_SERVER = r'''
import json, os, signal, sys, time
from pathlib import Path
Path(os.environ['FAKE_PID_FILE']).write_text(str(os.getpid()))
mode = os.environ.get('FAKE_MODE', 'success')
if mode == 'stubborn':
    signal.signal(signal.SIGTERM, lambda *_: Path(os.environ['FAKE_PID_FILE'] + '.cleanup').write_text('started'))
def emit(message):
    print(json.dumps(message), flush=True)
request = json.loads(sys.stdin.readline())
assert request['method'] == 'initialize'
if mode == 'init_error':
    emit({'id': request['id'], 'error': {'code': -1, 'message': 'initialization rejected'}})
else:
    emit({'method': 'notice', 'params': {}})
    emit({'method': 'notice', 'id': request['id'], 'params': {}})
    emit({'id': request['id'], 'result': {}})
    assert json.loads(sys.stdin.readline())['method'] == 'initialized'
    request = json.loads(sys.stdin.readline())
    assert request['method'] == 'account/rateLimits/read'
    if mode == 'timeout':
        time.sleep(10)
    elif mode == 'eof':
        sys.exit(0)
    elif mode == 'malformed':
        print('not JSON', flush=True)
    elif mode == 'rpc_error':
        emit({'id': request['id'], 'error': {'code': -1, 'message': 'ChatGPT login required'}})
    else:
        windows = {'primary': {'usedPercent': 17, 'windowDurationMins': 10080, 'resetsAt': 1791959730}, 'secondary': {'usedPercent': 10, 'windowDurationMins': 300, 'resetsAt': 1791488183}}
        if mode == 'invalid_reset':
            windows['primary']['resetsAt'] = 'not-a-timestamp'
        if mode == 'missing':
            windows = {'primary': None, 'secondary': None}
        result = {'rateLimits': windows, 'rateLimitsByLimitId': {'codex': windows, 'codex_other': {'primary': None, 'secondary': None}}, 'futureField': {'preserved': True}}
        emit({'id': request['id'] + 99, 'result': {'wrong': True}})
        emit({'method': 'notice', 'id': request['id'], 'params': {}})
        emit({'id': request['id'], 'result': result})
time.sleep(10)
'''


class UsageChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='codex-usage-check-')
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        fake = root / 'codex'
        fake.write_text('#!' + sys.executable + '\n' + textwrap.dedent(FAKE_SERVER))
        fake.chmod(0o755)
        self.pid_file = root / 'pid'
        self.environment = dict(os.environ, PATH=str(root), FAKE_PID_FILE=str(self.pid_file))

    def run_script(self, *args, mode='success'):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            env=dict(self.environment, FAKE_MODE=mode),
            capture_output=True, text=True, timeout=5,
        )
        if self.pid_file.exists():
            pid = int(self.pid_file.read_text())
            with self.assertRaises(ProcessLookupError, msg='app-server must be reaped'):
                os.kill(pid, 0)
        return result

    def test_json_preserves_account_result_and_ignores_notifications(self):
        result = self.run_script('--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['rateLimits']['secondary']['usedPercent'], 10)
        self.assertEqual(data['rateLimits']['primary']['windowDurationMins'], 10080)
        self.assertEqual(data['rateLimitsByLimitId']['codex_other'], {'primary': None, 'secondary': None})
        self.assertEqual(data['futureField'], {'preserved': True})
        self.assertEqual(result.stderr, '')

    def test_human_output_uses_durations_even_when_windows_are_swapped(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r'5h.*90% remaining.*10% used')
        self.assertRegex(result.stdout, r'Weekly.*83% remaining.*17% used')
        self.assertIn('reset', result.stdout)

    def test_missing_windows_are_unavailable_not_zero_usage(self):
        result = self.run_script(mode='missing')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('unavailable'), 2)

    def test_unusable_reset_time_does_not_hide_known_usage(self):
        result = self.run_script(mode='invalid_reset')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r'Weekly.*83% remaining.*reset unavailable')

    def test_failures_are_nonzero_and_stay_off_stdout(self):
        for mode, expected in [('init_error', 'initialization rejected'), ('rpc_error', 'ChatGPT login required'), ('eof', 'closed'), ('malformed', 'JSON')]:
            with self.subTest(mode=mode):
                result = self.run_script('--json', mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertIn(expected, result.stderr)

    def test_timeout_is_bounded_and_reaps_child(self):
        started = time.monotonic()
        result = self.run_script('--timeout', '0.2', mode='timeout')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('timed out', result.stderr)
        self.assertLess(time.monotonic() - started, 3)

    def test_missing_cli_has_actionable_error(self):
        self.environment['PATH'] = '/nonexistent'
        result = self.run_script('--json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('install', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_timeout_rejects_nonpositive_and_nonfinite_values(self):
        for value in ['0', '-1', 'nan', 'inf']:
            with self.subTest(value=value):
                result = self.run_script('--timeout', value)
                self.assertEqual(result.returncode, 2)
                self.assertIn('finite', result.stderr)

    def test_help_does_not_start_server(self):
        result = self.run_script('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--json', result.stdout)
        self.assertFalse(self.pid_file.exists())

    def test_termination_signal_reaps_server(self):
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT)],
            env=dict(self.environment, FAKE_MODE='timeout'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 2
            while not self.pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(self.pid_file.exists())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130)
            self.assertEqual(stdout, '')
            self.assertIn('interrupted', stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(self.pid_file.read_text()), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_second_signal_during_cleanup_does_not_skip_reaping(self):
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT)],
            env=dict(self.environment, FAKE_MODE='stubborn'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            marker = Path(str(self.pid_file) + '.cleanup')
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=4)
            self.assertEqual(process.returncode, 0)
            stdout, stderr = process.communicate(timeout=4)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn('90% remaining', stdout)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(self.pid_file.read_text()), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if self.pid_file.exists():
                try:
                    os.killpg(int(self.pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.stdout.close()
            process.stderr.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
