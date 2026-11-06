"""Part c CLI: extraction, label-only scoring, gates, and fixed-pick outcomes."""
import argparse
import os

import numpy as np

from mc_common import OUT, P, ROUND, record, saved, write
import mc_data as D
from mc_score import context, means, run as score
from mc_metrics import evaluate
from tcl_score import query_gallery_means
import tcl_data
import tcl_forward
import mt_data


def extract(fr, device, smoke):
    from tcl_extract import completed, encode_qwen_store, load_qwen
    from pav_prompts import PROMPTS
    instructions = ('default', *P.INSTRUCTIONS.values())
    todo = [i for i in instructions if not completed(D.store(i), PROMPTS[i])]
    if not todo:
        print('extract: four stores already complete')
        return
    texts = D.narrative(fr['keys'])
    encoder = load_qwen(device)
    old_smoke = P.SMOKE
    try:
        P.SMOKE = smoke
        for i in todo:
            encode_qwen_store(encoder, texts, D.store(i), i, 64, keys=fr['keys'])
    finally:
        P.SMOKE = old_smoke


def image_gate(fr):
    n = min(200, fr['n_i'])
    art = tcl_data.image('art_held', 'plain')[fr['image_rows'][:n]]
    mm = mt_data.features('qwen_default', fr['keys'][:n])
    cos = np.sum(art * mm, axis=1)
    assert cos.min() >= .999, float(cos.min())
    return dict(n=n, min_cosine=float(cos.min()), passed=True)


def rows_gate(fr, device, smoke):
    """Text A rows of the new path against r23's own forward, cut by the same row lists. Skipped under smoke."""
    if smoke:
        return dict(skipped='smoke')
    rows = D.Rows('A', fr)
    out = []
    for row, head, seed in (('cstar', 'genre', 0), ('late', 'genre', 0), ('plain', 'all', 'x'), ('teacher', 'genre', 'x')):
        got_t, got_i = rows.rows(row, head, seed, device)
        ref_t, ref_i = tcl_forward.rows('art_held', 'caption', row, head, seed, device)
        error = max(float(np.abs(got_t - ref_t[fr['text_rows']]).max()), float(np.abs(got_i - ref_i[fr['image_rows']]).max()))
        out.append(dict(row=row, max_error=error, passed=error <= 1e-6))
    assert all(r['passed'] for r in out), out
    return out


def gates(frames, device, block, smoke):
    result = dict(smoke=smoke, image=image_gate(frames['M']), rows=rows_gate(frames['A'], device, smoke), regression=[])
    for set_name, kind, text in (('art_held', 'caption', 'A'), ('mm_test', 'narrative', 'M')):
        fr = tcl_data.frame(set_name, kind)
        for row, head, seed in (('cstar', 'genre', 0), ('plain', 'all', 'x')):
            t, i = tcl_forward.rows(set_name, kind, row, head, seed, device)
            vectors, labels = dict(t=t, i=i), fr['labels']['genre']
            for d in P.DIRECTIONS:
                ro = record(P.GROUP[set_name])['picks'][kind][row]['genre'][d]
                pid, ref = saved(set_name, kind, row, head, seed, d, ro, 'genre')
                assert np.array_equal(pid, fr['tp'] if d[0] == 't' else fr['ip'])
                qrows = np.flatnonzero(pid < smoke) if smoke else np.arange(len(pid))
                q, g = vectors[d[0]], vectors[d[2]]
                ap = evaluate(q[qrows], g, {'genre': dict(q=labels[d[0]][qrows], g=labels[d[2]])}, d == 't2t', ro, device,
                              block, query_gallery_means(d, *means(text, row, head, seed)),
                              self_indices=qrows if d == 't2t' else None)['genre']
                ref = ref[qrows]
                assert np.array_equal(np.isfinite(ap), np.isfinite(ref))
                mask = np.isfinite(ref)
                assert mask.any()
                error = float(np.max(np.abs(ap[mask].astype(np.float64) - ref[mask])))
                result['regression'].append(dict(set=set_name, row=row, head=head, seed=seed, direction=d, readout=ro, n_query=len(qrows),
                                                 n_finite=int(mask.sum()), max_error=error, passed=error <= 1e-5))
            del t, i
    result['passed'] = all(r['passed'] for r in result['regression'])
    write(OUT / 'gates.json', result)
    assert result['passed'], result['regression']
    print(f"gates: image cosine {result['image']['min_cosine']:.8f}; {len(result['regression'])} regression units pass")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=('extract', 'score', 'gates', 'outcomes'))
    p.add_argument('--text', choices=('M', 'A'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', type=int, default=0, metavar='N')
    p.add_argument('--block', type=int, default=1024)
    a = p.parse_args()
    assert a.smoke >= 0 and a.block > 0
    assert not os.environ.get('TCL_SMOKE') and not os.environ.get('TCL_STORES'), 'use --smoke and MC_OUTPUTS only'
    if a.smoke:
        assert os.environ.get('MC_OUTPUTS'), '--smoke requires MC_OUTPUTS'
        a.device = 'cpu'
    # Prevent mixing smoke sizes, full runs, or painting orders in resumable outputs.
    frames = D.shared(a.smoke)
    manifest = OUT / 'shared_manifest.json'
    spec = dict(smoke=a.smoke, keys=frames['M']['keys'], stores=str(OUT / 'stores'))
    if manifest.exists():
        import json
        assert json.loads(manifest.read_text()) == spec, 'output belongs to another run'
    else:
        write(manifest, spec)
    if a.job == 'extract':
        extract(frames['M'], a.device, a.smoke)
    elif a.job == 'score':
        assert a.text, 'score requires --text M or A'
        score(a.text, frames[a.text], D.Rows(a.text, frames[a.text]), a.device, a.block)
    elif a.job == 'gates':
        gates(frames, a.device, a.block, a.smoke)
    else:
        from mc_outcomes import run
        run(frames, a.smoke)


if __name__ == '__main__':
    main()
