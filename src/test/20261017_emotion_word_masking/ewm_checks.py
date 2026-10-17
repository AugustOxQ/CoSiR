"""CPU-only masking units, import safety and fake-encoder extraction contract checks."""
import argparse
import contextlib
import importlib
import io
import runpy
import sys
import time
from types import ModuleType
from unittest.mock import patch
import numpy as np
from ewm_masks import (HERE, LABEL_FORMS, WORDS, EMOTION_NAMES, build_masks, mask_text,
                       word_statistics)


@contextlib.contextmanager
def encoder_module(module):
    # Restore only the substituted entry. Restoring all of sys.modules would unload
    # Python wrappers around Torch native modules and break a subsequent import.
    name = 'src.data.feature_extract'
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def units():
    assert mask_text('contentment content contents discontent', 'label') == '[MASK] [MASK] contents discontent'
    assert mask_text('sadly sadler SAD SaDnEsS', 'label') == '[MASK] sadler [MASK] [MASK]'
    assert mask_text('An AWE-INSPIRING view.', 'label') == 'An [MASK] view.'
    assert mask_text('An AWE-INSPIRING view.', 'broad') == 'An [MASK] view.'
    assert mask_text('sad-looking, sadness; sadly!', 'label') == '[MASK]-looking, [MASK]; [MASK]!'
    original = '  A blue sky.\t\nCafé: paint; contents discontent sadler.  '
    for mask in WORDS:
        assert mask_text(original, mask).encode('utf-8') == original.encode('utf-8')
        for word in WORDS[mask]:
            assert mask_text(word.upper(), mask) == '[MASK]', (mask, word)
    assert mask_text('feel feeling feelings feels felt', 'broad') == ' '.join(['[MASK]'] * 5)
    assert mask_text('feel feeling feelings feels felt', 'label') == 'feel feeling feelings feels felt'
    b = build_masks(['sad SAD awe-inspiring', 'angry', 'quiet sky', 'feel calm'])
    assert b['masked']['label'][0] == '[MASK] [MASK] [MASK]'
    assert b['label_word_counts'].shape == (4, 9)
    assert b['label_word_counts'][0, EMOTION_NAMES.index('sadness')] == 2
    assert b['label_word_counts'][0, EMOTION_NAMES.index('awe')] == 1
    assert b['counters']['broad']['awe-inspiring'] == 1 and b['counters']['broad']['inspiring'] == 0
    # Nonempty each class for the real statistics helper; check the conditional agreement explicitly.
    b = build_masks(['sad SAD awe-inspiring', 'angry'] + ['quiet sky'] * 9)
    stats = word_statistics(b, np.asarray([7, 0] + list(range(9))))
    assert stats['label_agreement_share'] == .5
    assert len(set(word for words in LABEL_FORMS.values() for word in words)) == len(WORDS['label'])
    print(f'masking units: PASS ({len(WORDS["label"])} label words, {len(WORDS["broad"])} broad words)', flush=True)


def imports_and_help():
    sentinel = ModuleType('src.data.feature_extract')
    def forbidden(*args, **kwargs):
        raise AssertionError('Qwen construction attempted during import/help')
    sentinel.Qwen3VLEmb = forbidden
    with encoder_module(sentinel):
        for name in ('ewm_masks', 'ewm_extract', 'ewm_score', 'ewm_checks'):
            importlib.import_module(name)
            with patch.object(sys, 'argv', [name + '.py', '--help']), contextlib.redirect_stdout(io.StringIO()):
                try:
                    runpy.run_path(str(HERE / (name + '.py')), run_name='__main__')
                except SystemExit as error:
                    assert error.code == 0
                else:
                    raise AssertionError(f'{name}: --help did not exit')
            print(f'{name}: import/help PASS; Qwen constructor blocked', flush=True)


def extraction_contract():
    """Fake embeddings live only in a clearly separate /tmp fixture directory."""
    import ewm_extract as ex
    from pav_prompts import PROMPTS
    from ewm_masks import KEYS
    n = 256
    s = dict(sample_ids=np.arange(n, dtype=np.int64), painting_ids=np.asarray(['fake'] * n),
             subset_painting_ids=np.asarray(['fake']), subset_image_indices=np.asarray([0]))
    texts = ['A sad view, and a feel of calm.'] * n
    stored = np.zeros((256, 2048), dtype=np.float16)
    stored[:, 0] = 1
    constructions, calls = [], []
    class FakeEncoder:
        dim = 2048
        def __init__(self, device, instruction):
            assert device == 'cpu'
            self.instruction = instruction
            constructions.append(instruction)
        def encode_texts(self, values, batch_size):
            assert batch_size == 128
            calls.append((self.instruction, len(values), values[0]))
            out = np.zeros((len(values), 2048), dtype=np.float32)
            out[:, 0] = 1
            return out
    sentinel = ModuleType('src.data.feature_extract')
    sentinel.Qwen3VLEmb = FakeEncoder
    # A fresh retained fixture per check avoids deleting or overwriting any older seal.
    destination = ex.SMOKE_OUTPUT / ('fake_contract_' + str(time.time_ns()))
    with encoder_module(sentinel), \
            patch.object(ex, 'SMOKE_OUTPUT', destination), \
            patch.object(ex, 'load_frozen', return_value=s), \
            patch.object(ex, 'load_captions', return_value=texts), \
            patch.object(ex, 'validate_key', return_value=(None, stored)), \
            patch.object(sys, 'argv', ['ewm_extract.py', '--smoke', '--device', 'cpu']):
        ex.main()
        assert len(constructions) == 1 and len(calls) == 4
        assert [c[1] for c in calls] == [256, 256, 64, 64]
        assert [c[0] for c in calls[2:]] == [PROMPTS[key] for key in KEYS]
        assert all(c[2] == 'A [MASK] view, and a feel of calm.' for c in calls[2:])
        ex.main()
        assert len(constructions) == 1 and len(calls) == 4, 'sealed extraction was not skipped'
    print(f'fake extraction contract: PASS (one model, instruction switches, reproduction, seals, resume); {destination}', flush=True)


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', action='store_true', help='also check extraction with a fake CPU encoder in /tmp')
    args = p.parse_args()
    units()
    imports_and_help()
    if args.contract:
        extraction_contract()
    print(f'Checks finished in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
