"""r28 SigLIP 1 knob/removal rows through r14's trainer and r10's patch layer.

train --keep all retains r14's weights, train/<row>/R6-5 validation exports and
run records. Run `export --kind held` separately after validation selection is
complete. `ref` retains r14's k11=40 arm and loss; use its existing checkpoints.
No older source file is edited. Context patches unwind on success or failure.
"""
import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261102_backbones_sweep'))
import bs_paths as P  # noqa: E402

# r14 points its r10 import at the main checkout. Select this worktree's code
# for the import only; frozen artifacts keep their original read-only paths.
with patch.object(P, 'R10_ROUND', TEST / '20261027_recipe_sweep'):
    import bs_cstar as B  # noqa: E402
import r28_trainer_loss as L  # noqa: E402

S, K, G, V = B.S, B.rs_knobs, B.rs_graphs, B.V
OUTPUTS = Path('/data/SSD2/cosir_r28/20261112_siglip_main')
SMOKE = OUTPUTS / 'smoke'
KNOBS = ('k11=10', 'k11=20', 'k11=80', 'k6=0.0005')
REMOVALS = ('instructed_graphs', 'content_correction', 'buddy', 'relational')
BASE_CONTEXT = B.training_context
BASE_RECORD = B._record
BASE_REQUIRE_GATE = G.require_gate


def row_config(row):
    if row in REMOVALS:
        return S.config({'k11': B.STAR['k11']}, remove=(row,))
    return B.row_config(row)


def check_root(root, smoke=False):
    root = B.check_root(root, smoke)
    assert root.is_relative_to(OUTPUTS.resolve()), f'outputs must be under {OUTPUTS}'
    assert smoke == root.is_relative_to(SMOKE.resolve()), 'smoke and full outputs must be separate'
    return root


def graph_gate(_outputs, name, item=None):
    """Read r10's code-bound graph gates, never write or manufacture a gate."""
    return BASE_REQUIRE_GATE(P.R10_OUT, name, item)


def graph_source(k, variant, outputs):
    """Reuse code/input-verified older stores read-only; build missing ones in r28."""
    for root in (P.R10_OUT, P.DEFAULT_OUT):
        store = G.store_root(root, 'full', k, variant)
        if (store / 'd2/graphs/manifest.json').exists():
            # _materialize also checks input identity, source and archive hashes.
            G._require_own_store(store, 'full', k, variant)
            return root
    root = Path(outputs).resolve()
    assert root.is_relative_to(OUTPUTS.resolve()), root
    return root


def graph_data(scale, k, outputs, variant='aspect'):
    """r14's graph seam opened to r10's two removal variants, on Qwen data."""
    assert scale == 'full' and (k, variant) != (S.G['k11'], 'aspect')
    B.require_graph_gate()
    if variant != 'aspect':
        graph_gate(outputs, 'ablation_graphs', variant)
    source = graph_source(k, variant, outputs)
    return G._materialize(scale, k, variant, source)[1]


@contextmanager
def training_context(enc, row=B.REF, *, export=False, graph_out=None):
    cfg = row_config(row)
    assert enc in ('qwen', 'siglip1'), enc
    assert enc == 'qwen' or row in (B.REF, *KNOBS, *REMOVALS), row
    graph_out = Path(graph_out) if graph_out is not None else OUTPUTS / 'trainer'
    with ExitStack() as stack:
        stack.enter_context(patch.object(K, 'store_training_data', graph_data))
        stack.enter_context(patch.object(G, 'require_gate', graph_gate))
        if row == B.REF or (enc == 'qwen' and row not in REMOVALS):
            # Preserve r14's function objects, model creation and patch order.
            stack.enter_context(patch.object(B, 'graph_data', graph_data))
            yield stack.enter_context(BASE_CONTEXT(enc, row, export=export, graph_out=graph_out))
            return
        name, width = B.backbone(enc), B.width(enc)
        if enc != 'qwen':
            stack.enter_context(patch.object(B.rg_full.rg_loss, 'switched_head_loss',
                                             B.bs_cstar_loss.switched_head_loss))
            stack.enter_context(patch.object(K, 'genre_loss', L.genre_loss))
        stack.enter_context(patch.object(B.rg_full, 'ENCODER', name))
        stack.enter_context(patch.dict(B.rb_encoders.WIDTH, {name: width}))
        # Exactly r10's patch plan (including effective k3=0 and buddy wrapper).
        knobs = K.patches(dict(cfg, k11=S.G['k11']) if export else cfg, stack,
                          scale='full', outputs=graph_out)
        stack.enter_context(B.rs_full.arm_alias(row))
        yield B.Context(cfg, row, knobs, name, width)


def code_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(L.__file__))}


def run_record(*args, **kwargs):
    record = BASE_RECORD(*args, **kwargs)
    record['r28_code_sha256'] = code_hashes()
    record['r28_graph_gate_root'] = str(P.R10_OUT)
    return record


@contextmanager
def trainer_patches(row):
    """Extend r14 only for this call; its train/export implementations are reused."""
    variant = B.rs_full.graph_choice(row_config(row))['variant']
    with ExitStack() as stack:
        stack.enter_context(patch.object(B, 'training_context', training_context))
        stack.enter_context(patch.object(B, '_record', run_record))
        stack.enter_context(patch.object(B, 'GRAPH_ROWS', G.row_names(variant)))
        yield


def train(enc, row, seeds, device, root, *, keep='all', smoke=False):
    assert keep == 'all', 'r28 retains all artifacts; no scratch deletion'
    assert not (enc == 'siglip1' and row == B.REF), 'reuse r14 SigLIP 1 ref checkpoints; do not retrain'
    root = check_root(root, smoke)
    seeds = list(seeds)
    for seed in seeds:
        path = B.record_path(root, B.row_arm(row), seed)
        if path.exists():
            recorded = json.loads(path.read_text())
            assert recorded.get('r28_code_sha256') == code_hashes(), f'{path}: r28 training code changed'
    with trainer_patches(row):
        return B.train(enc, row, seeds, device, root, keep=keep, smoke=smoke, graph_out=root)


def export(enc, row, seeds, root, *, kinds=('held',), device=None, out_root=None, smoke=False):
    # A source root may be r14's existing ref; all destinations remain in r28.
    destination = check_root(out_root if out_root is not None else root, smoke)
    with trainer_patches(row):
        for kind in kinds:
            assert kind in ('held', 'val'), kind
            fn = B.export_held if kind == 'held' else B.export_validation
            fn(enc, row, seeds, root, device=device, out_root=destination, smoke=smoke)


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'export'):
        cmd = commands.add_parser(command)
        cmd.add_argument('--encoder', choices=('siglip1', 'qwen'), required=True)
        cmd.add_argument('--row', required=True)
        cmd.add_argument('--seeds', nargs='+', type=int, choices=B.SEEDS, default=list(B.SEEDS))
        cmd.add_argument('--root', type=Path, required=True)
        cmd.add_argument('--device', default='cuda' if command == 'train' else None)
        cmd.add_argument('--smoke', action='store_true')
        if command == 'train':
            cmd.add_argument('--keep', choices=('all',), default='all')
        else:
            cmd.add_argument('--kind', nargs='+', choices=('held', 'val'), default=['held'])
            cmd.add_argument('--out-root', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'train':
        runs = train(args.encoder, args.row, args.seeds, args.device, args.root,
                     keep=args.keep, smoke=args.smoke)
        for seed, run in runs.items():
            print(f'R28 train {args.encoder} {args.row} s{seed} {run.record["status"]} '
                  f'seconds={run.record["seconds"]:.2f}', flush=True)
    else:
        export(args.encoder, args.row, args.seeds, args.root, kinds=args.kind,
               device=args.device, out_root=args.out_root, smoke=args.smoke)
        print(f'R28 export {args.encoder} {args.row}: {args.kind}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
