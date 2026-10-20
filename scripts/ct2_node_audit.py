"""Write the local static import/input audit to the explicitly supplied notes folder."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from das6_sync_ct2 import EXCLUDED, actions, files


def import_tree():
    # Dated helper modules have unique stems; qualified src/scripts modules use
    # their actual package paths. Follow imports inside functions as well.
    index = {}
    for p in (ROOT / 'src/test').rglob('*.py'):
        index.setdefault(p.stem, []).append(p)
    todo = list((ROOT / 'src/test/20261020_claim_test').glob('ct*.py'))
    seen = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = p.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    stem = parent / (node.module or '').replace('.', '/')
                    for candidate in [stem.with_suffix('.py'), stem / '__init__.py']:
                        if candidate.is_file():
                            todo.append(candidate)
                    if not node.module:
                        for a in node.names:
                            candidate = parent / (a.name + '.py')
                            if candidate.is_file():
                                todo.append(candidate)
                else:
                    modules = [node.module or '']
            for module in modules:
                parts = module.split('.')
                for n in range(1, len(parts) + 1):
                    candidate = ROOT.joinpath(*parts[:n])
                    if candidate.with_suffix('.py').is_file():
                        todo.append(candidate.with_suffix('.py'))
                    if (candidate / '__init__.py').is_file():
                        todo.append(candidate / '__init__.py')
                if module in index:
                    todo.extend(index[module])
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notes', type=Path)
    args = parser.parse_args()
    allowed = Path('/project/CoSiR/.scratch/r10-claim-test-s2/node')
    assert args.notes.resolve() == allowed
    sources = import_tree()
    buddy = Path('/project/CoSiR-buddy_prototype_conditioning/src/conditional_buddy/buddy_graph.py')
    sources.append(buddy)
    pattern = re.compile(r'^\s*(?:from|import)\s+(?:' + '|'.join(EXCLUDED) + r')(?:[.\s,]|$)', re.M)
    hits = [str(p) for p in sources if pattern.search(p.read_text())]
    assert not hits, hits
    imported = '\n'.join(str(p) for p in sources) + '\n'
    (allowed / 'import_tree.txt').write_text(imported)
    absolute = []
    for p in sources:
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if any(x in line for x in ('/project/', '/data/', '/tmp/', '/root/', 'HF_HOME', 'from_pretrained')):
                absolute.append(f'{p}:{n}: {line.strip()}')
    (allowed / 'path_evidence.txt').write_text('\n'.join(absolute) + '\n')
    manifest = json.loads((ROOT / 'scripts/ct2_node_inputs.json').read_text())
    action_map = {a['key']: a for a in actions()}
    text = ['# CT2 node inputs', '',
            'Static audit at HEAD ' + subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            '', f'Followed {len(sources)} source files, including function-local imports and package initializers; '
            'see import_tree.txt and path_evidence.txt. All paths below are local source paths.', '',
            '| Certainty | Input | du -sh | Transfer bytes (dereferenced) | Delivery |',
            '|---|---|---:|---:|---|']
    for item in manifest:
        du = subprocess.check_output(['du','-sh',item['local']],text=True).split()[0]
        listing = files(item)
        volume = sum(listing.values())
        text.append(f'| {item["certainty"]} | `{item["local"]}` | {du} | {volume:,} | '
                    + ('controller' if item.get('controller') else 'das6_sync_ct2 --inputs') + ' |')
    certain = sum(a['bytes'] for a in action_map.values() if a['certainty'] == 'certain')
    possible = sum(a['bytes'] for a in action_map.values() if a['certainty'] == 'possible')
    text += ['', f'Helper transfer: certain {certain:,} bytes ({certain/1024**3:.3f} GiB), '
             f'possible {possible:,} bytes ({possible/1024**3:.3f} GiB), total {(certain+possible)/1024**3:.3f} GiB.', '',
             'The helper copies possible inputs too, so missing-store extraction can run unchanged. '
             'Possible means read only if a stage 2 store is missing: pinned Qwen and RoBERTa model directories '
             '(including snapshots and refs) and WikiArt RGB images. All five stage 2 stores currently have done.json. '
             'Model snapshots are copied with symlinks dereferenced; the duplicate blobs directory is excluded.', '',
             'The controller copies both whole pre-extract roots. Within artelingo_full the queue reads '
             'qwen3vl_emb_2b, qwen3vl_emb_2b_aspect_prompts/{default,emotion_focus,style_word,genre_word}, '
             'the *_trainimg stores, *_held diagnostic stores, emotion_focus_s2img, style_word_s2img, '
             'genre_word_s2img, emotion_focus_s2txt, roberta_go_emotions and roberta_go_emotions_s2. '
             'artelingo/features is the CLIP FeatureManager store (metadata, sample IDs and feature shards). '
             'The separate artelingo_percept_patch_features/train_patch_features.pt plus its metadata are certain '
             'via pc_data.patch_tokens -> ct_held.clip_patches.', '',
             'sample.npz freezes roles and labels. Frozen AGS graphs feed sample stage 2a. '
             'Stage 1 outputs supply D2 seed-0 NPZ/JSON, PercepT core and paper stage1 and stage2 '
             'NPZ/JSON/PT and percept/rcol_stage1.json for regression and replay checks. '
             'The complete current outputs tree is copied except top-level stale/ and stage2/. '
             'No model-sheet or percept-core outputs are used by stage 2a: its reference is claim-test outputs. '
             'Their defaults occur only in the unused ct_stage0.check path. '
             'stage0.json, verdict_stage1.json and diagnostics reports in the round root are tracked at HEAD; '
             'they travel with code. No .scratch file is needed at runtime.', '',
             'Genre CSVs genre_train.csv and genre_val.csv are certain via split2(full) and painting_labels. '
             'buddy_graph.py is the only buddy file loaded: ct_diag.september_graph_helpers dynamically loads it; '
             'it imports numpy, torch and scipy only. No buddy checkout, weights or other outputs are needed.', '',
             'Excluded site-packages directories: ' + ', '.join(EXCLUDED) + '. '
             f'Grep over import statements in all {len(sources)} files found zero imports of them. '
             'This is a project import-tree check; third-party optional imports are not an EL8 runtime guarantee. '
             'The env exclusion glob is lib/python*/site-packages/<name>/*** (including the python3.1 alias). '
             'Package metadata is retained. Symlinks are dereferenced for portable inputs; logical transfer bytes '
             'can exceed du, especially the env python3.1 -> python3.11 alias.', '',
             'Writes outside the round: certain job-local /tmp, TMPDIR, MPLCONFIGDIR, HOME, XDG_CACHE_HOME, '
             'CT_GPU_LOCK, namespace root mount points and smoke output. Possible extraction writes new '
             '*.pending and sealed stage 2 stores under the controller pre-extract root; model caches are offline '
             'under NR/hf_home. Namespace mount points in the job code copy are plumbing writes. '
             'Round-root stage2a.json, verdict_stage2.json and diag_stage2.{json,md} are copied into '
             'outputs/stage2/reports by the wrapper for fetch. Queue logs/status and trained artifacts are under '
             'NR/outputs/stage2. Imported historical smoke defaults under /tmp are not invoked by either queue; '
             'the entire /tmp maps to job-local storage anyway. No W&B run or other model download is started.', '',
             'Runtime OS libraries, git worktree metadata and device/proc/sys paths come from the probe host binds. '
             'The CLI worktree .git points into its node clone under /local/wding, visible through /local. '
             'They are platform dependencies, not repository input transfers.', '',
             'The frozen feature_extract.py CUDA-library hint uses the absent env/lib/python3.10/'
             'site-packages/nvidia/cu13/lib. The launcher binds the copied env/lib/python3.11/'
             'site-packages/nvidia/cu13/lib to that exact legacy path. It needs no additional input '
             'transfer and creates only empty mountpoint directories under NR/env. This bind is '
             'untested on the node.', '',
             '## Exact non-repo project and buddy transfer contents', '']
    for key in ('sample','graphs','outputs','buddy_graph','genre','patches'):
        item = action_map[key]
        text += [f'### {key}', '']
        text += [f'- `{name}` — {size:,} bytes' for name,size in files(item).items()]
        text.append('')
    (allowed / 'inputs.md').write_text('\n'.join(text) + '\n')
    print(f'Audit: {len(sources)} sources; excluded imports 0; certain {certain/1024**3:.3f} GiB; '
          f'possible {possible/1024**3:.3f} GiB')


if __name__ == '__main__':
    main()
