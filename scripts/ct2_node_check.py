"""Read-only CT2 readiness check; run inside the node namespace after data sync.

--imports-only supports the authorized local CPU check without reading providers.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'src/test/20261020_claim_test'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROUND))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports-only', action='store_true')
    args = parser.parse_args()
    for path in sorted(ROUND.glob('ct2_*.py')):
        importlib.import_module(path.stem)
        print('import', path.stem, 'PASS', flush=True)
    if args.imports_only:
        return
    for item in json.loads((ROOT / 'scripts/ct2_node_inputs.json').read_text()):
        path = Path(item['bind'])
        if not path.exists() and item['certainty'] == 'possible':
            print('input', item['key'], 'absent (possible)', flush=True)
            continue
        assert path.exists(), f'missing {item["certainty"]} input: {path}'
        print('input', item['key'], 'present', flush=True)
    import numpy as np
    import ct2_data as data
    import ct2_percept as c2
    stores = [data.ASPECT / name for name in
              ('emotion_focus_s2img', 'style_word_s2img', 'genre_word_s2img', 'emotion_focus_s2txt')]
    stores.append(data.ROBERTA.parent / 'roberta_go_emotions_s2')
    for path in stores:
        done = data.check_s2_store(path)
        print(f'store {path.name}: PASS count {done["count"]} parity {done["parity_gate"]}', flush=True)
    for mode in ('sample', 'full'):
        s = data.split2(mode)
        print(f'split2 {mode}: {len(s["all_painting_ids"])} paintings, '
              f'{len(s["sample_ids"])} captions, fit {len(s["fit"])} dev {len(s["dev"])}', flush=True)
        if mode == 'sample':
            continue
        n_cap, n_img = len(s['sample_ids']), len(s['all_painting_ids'])
        def line(name, provider, rows):
            tick = time.time()
            x = provider()
            assert x.shape[0] == rows and np.isfinite(x).all(), name
            print(f'{name}: shape {x.shape} finite, {time.time()-tick:.0f} s', flush=True)
        for kind in ('qwen_default', 'clip_image'):
            line(f'image {kind}', lambda: c2.image_features(s, kind), n_img)
        for key in ('emotion_focus', 'style_word', 'genre_word'):
            line(f'instructed image {key}', lambda: data.instructed_images(s, key), n_img)
        for kind in ('qwen_default', 'qwen_emotion_focus', 'roberta', 'clip_text'):
            line(f'caption {kind}', lambda: c2.caption_features(s, kind), n_cap)
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
