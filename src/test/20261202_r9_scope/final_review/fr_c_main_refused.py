"""Final review C: run_r9_held.py executed as a script (its own namespace, like __main__) and the launch gate.

preread_guard raises run_r9_held.Refused (the imported module's class); run_held's `except Refused`
names the script namespace's class. Shows what the CLI does on a refused gate. No file is written.
"""
import importlib.util, sys, tempfile
from pathlib import Path
from types import SimpleNamespace as NS

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
spec = importlib.util.spec_from_file_location('as_script', F / 'run_r9_held.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
g = vars(m)
g['refuse_or_go'] = lambda *a: NS(names=g['read_names']('H8', 'held'), attempts=[], want={}, ledger={})
with tempfile.TemporaryDirectory(dir=sys.argv[1]) as d:
    try:
        code = g['run_held']('clip_b32', 'H8', results=Path(d))
        print('returned', code, '(EXIT_REFUSE =', g['EXIT_REFUSE'], ')')
    except Exception as e:
        print('uncaught', type(e).__module__ + '.' + type(e).__name__, '-', e)
