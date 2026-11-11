"""Write the paper's other-encoder table and photo table from recorded cells (no new scoring).

Sources (all test sets, means over seeds 0 to 4 for trained models):
- R17 = src/test/20261103_new_scores/records/verdicts.json: AMI at the class count (SC1 plain, SC2 matched PercepT,
  SC3 teacher on ArtELingo; SC4 Affection and EmoSet), mAP at the validation-picked read-out (SC7 plain, SC8 matched
  PercepT and c*).
- B02 = src/test/20261102_backbones_sweep/records/verdicts.json: photo mAP (SC5 Affection per encoder, SC8 and SC10
  EmoSet Qwen).
- R13 = /data/SSD2/cosir_r13/20261031_photo_finetune/outputs/stage3/verdict.json: Qwen photo, plain and teacher mAP on
  Affection (rows.<arm>.map_picked.map).
- Teacher mAP on ArtELingo: Table 3's teacher row (r11 stage 5, map.instructed.<aspect>.mean); equal to B02 SC6
  comparator values.

Usage: python ept_make.py <repo Tables dir>   (writes encoders.tex, photos.tex there and records/cells.json here)
"""
import json
import sys
from pathlib import Path

ROOT = Path('/project/CoSiR')
HERE = Path(__file__).resolve().parent
R17 = ROOT / 'src/test/20261103_new_scores/records/verdicts.json'
B02 = ROOT / 'src/test/20261102_backbones_sweep/records/verdicts.json'
R13 = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/stage3/verdict.json')
S5 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json')

ASPECTS = ('emotion', 'style', 'genre')
K = {'emotion': 9, 'style': 27, 'genre': 10}
ENCODERS = (('qwen', 'Qwen3'), ('siglip2', 'SigLIP 2'), ('siglip1', 'SigLIP 1'), ('clip_l14', 'CLIP L/14'))

record = {}


def put(key, value, source):
    record[key] = {'value': value, 'source': source}
    return value


def r17_cells():
    return {c['id']: c for c in json.load(open(R17))['cells']}


def b02_cells():
    out = {}
    for c in json.load(open(B02))['cells']:
        out[(c['sc'], c['encoder'], c['task'], c.get('data'), c['candidate'], c['comparator'])] = c
    return out


def fmt(v):
    return f'{v:.3f}'


def bold_columns(rows, ncols):
    """rows: list of (label_cells, values or None). Bold the max per column at printed precision (ties all bold)."""
    best = []
    for j in range(ncols):
        vals = [float(fmt(v[j])) for _, v in rows if v is not None and v[j] is not None]
        best.append(max(vals) if vals else None)
    lines = []
    for label, v in rows:
        if v is None:
            lines.append(label)
            continue
        cells = []
        for j, x in enumerate(v):
            if x is None:
                cells.append('--')
            elif float(fmt(x)) == best[j]:
                cells.append(r'\textbf{' + fmt(x) + '}')
            else:
                cells.append(fmt(x))
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    return lines


def encoders_table():
    c17 = r17_cells()
    s5 = json.load(open(S5))
    rows = []
    teacher = [put(f'enc/teacher/{a}/ami', c17[f'SC3/artelingo/qwen/{a}/ami/{K[a]}/teacher']['comp'],
                   f'R17 SC3/artelingo/qwen/{a}/ami/{K[a]}/teacher comp') for a in ASPECTS]
    teacher += [put(f'enc/teacher/{a}/map', s5['map']['instructed'][a]['mean'], f'S5 map.instructed.{a}.mean')
                for a in ASPECTS]
    for i, (enc, name) in enumerate(ENCODERS):
        rule = '' if i == 0 else r'\midrule' + '\n    '
        rows.append((rule + r'\multicolumn{7}{@{}l}{\emph{' + name + r'}} \\', None))
        if enc == 'qwen':  # the teacher is Qwen3's for every encoder
            rows.append((r'Teacher (instructed)', teacher))
        plain = [put(f'enc/{enc}/plain/{a}/ami', c17[f'SC1/artelingo/{enc}/{a}/ami/{K[a]}/plain']['comp'],
                     f'R17 SC1/artelingo/{enc}/{a}/ami/{K[a]}/plain comp') for a in ASPECTS]
        plain += [put(f'enc/{enc}/plain/{a}/map', c17[f'SC7/artelingo/{enc}/{a}/map/-/plain']['comp'],
                      f'R17 SC7/artelingo/{enc}/{a}/map/-/plain comp') for a in ASPECTS]
        pt = [put(f'enc/{enc}/percept/{a}/ami', c17[f'SC2/artelingo/{enc}/{a}/ami/{K[a]}/matched']['comp'],
                  f'R17 SC2/artelingo/{enc}/{a}/ami/{K[a]}/matched comp') for a in ASPECTS]
        pt += [put(f'enc/{enc}/percept/{a}/map', c17[f'SC8/artelingo/{enc}/{a}/map/-/matched']['comp'],
                   f'R17 SC8/artelingo/{enc}/{a}/map/-/matched comp') for a in ASPECTS]
        ours = [put(f'enc/{enc}/cstar/{a}/ami', c17[f'SC1/artelingo/{enc}/{a}/ami/{K[a]}/plain']['cand'],
                    f'R17 SC1/artelingo/{enc}/{a}/ami/{K[a]}/plain cand') for a in ASPECTS]
        ours += [put(f'enc/{enc}/cstar/{a}/map', c17[f'SC8/artelingo/{enc}/{a}/map/-/matched']['cand'],
                     f'R17 SC8/artelingo/{enc}/{a}/map/-/matched cand') for a in ASPECTS]
        # c* on one encoder must be the same model in SC1, SC2 and SC3
        for a in ASPECTS:
            ids = [f'SC{n}/artelingo/{enc}/{a}/ami/{K[a]}/{arm}' for n, arm in ((2, 'matched'), (3, 'teacher'))]
            assert all(abs(c17[i]['cand'] - record[f'enc/{enc}/cstar/{a}/ami']['value']) < 1e-12 for i in ids)
        rows.append(('Plain embedding', plain))
        rows.append((r'PercepT~\cite{mohamed2026percept}', pt))
        rows.append((r'\method (ours)', ours))
    body = bold_columns(rows, 6)
    return r"""\begin{table}[t]
  \centering
  \small
  \setlength{\tabcolsep}{1.7pt}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    & \multicolumn{3}{c}{Clustering (AMI)} & \multicolumn{3}{c}{Retrieval (mAP)} \\
    \cmidrule(lr){2-4} \cmidrule(l){5-7}
    Method & Emo. & Style & Genre & Emo. & Style & Genre \\
    \midrule
""" + '\n'.join('    ' + line for line in body) + r"""
    \bottomrule
  \end{tabular}
  \caption{Other image encoders on the ArtELingo test set. On each encoder, \method and PercepT read that encoder's plain image embedding, while the aspect graphs, the teacher and PercepT's topics stay those of Qwen3, so the teacher row applies to every encoder. Clustering is the AMI at each aspect's class count, as in \cref{tab:clustering}, and retrieval is the label mAP with the read-out that each method picked on the validation split, as in \cref{tab:retrieval}. \method and PercepT are means over five random seeds. The best result per column is in bold.}
  \label{tab:encoders}
\end{table}
"""


def photos_table():
    c17 = r17_cells()
    b02 = b02_cells()
    r13 = json.load(open(R13))['rows']
    rows = []

    def r17v(set_, enc, arm, field):
        k = 8 if set_ == 'emoset' else 9
        cid = f'SC4/{set_}/{enc}/emotion/ami/{k}/{arm}'
        return put(f'photo/{set_}/{enc}/{arm if field == "comp" else "photo"}/ami', c17[cid][field], f'R17 {cid} {field}')

    # teacher (Qwen3 instructed for emotion)
    t = [r17v('affection', 'qwen', 'teacher', 'comp'),
         put('photo/affection/teacher/map', r13['teacher_artwork']['map_picked']['map'],
             'R13 rows.teacher_artwork.map_picked.map'),
         r17v('emoset', 'qwen', 'teacher', 'comp'),
         put('photo/emoset/teacher/map', b02[('SC10', 'qwen', 'ret', 'emoset_test', 'photo_qwen', 'teacher_emotion_focus')]
             ['comparator_value'], 'B02 SC10 qwen ret comparator_value')]
    rows.append((r'Teacher (instructed) & Qwen3', t))
    for enc, name in ENCODERS:
        rows.append((r'\midrule', None))
        if enc == 'qwen':
            plain = [r17v('affection', enc, 'plain', 'comp'),
                     put('photo/affection/qwen/plain/map', r13['plain']['map_picked']['map'],
                         'R13 rows.plain.map_picked.map'),
                     r17v('emoset', enc, 'plain', 'comp'),
                     put('photo/emoset/qwen/plain/map',
                         b02[('SC8', 'qwen', 'ret', 'emoset_test', 'photo_qwen', 'plain_qwen')]['comparator_value'],
                         'B02 SC8 qwen ret comparator_value')]
            ours = [r17v('affection', enc, 'plain', 'cand'),
                    put('photo/affection/qwen/photo/map', r13['photo']['map_picked']['map'],
                        'R13 rows.photo.map_picked.map'),
                    r17v('emoset', enc, 'plain', 'cand'),
                    put('photo/emoset/qwen/photo/map',
                        b02[('SC8', 'qwen', 'ret', 'emoset_test', 'photo_qwen', 'plain_qwen')]['candidate_value'],
                        'B02 SC8 qwen ret candidate_value')]
            assert abs(b02[('SC10', 'qwen', 'ret', 'emoset_test', 'photo_qwen', 'teacher_emotion_focus')]['candidate_value']
                       - ours[3]) < 1e-12
        else:
            cell = b02[('SC5', enc, 'ret', 'affection_test', f'photo_{enc}', f'frozen_{enc}')]
            plain = [r17v('affection', enc, 'plain', 'comp'),
                     put(f'photo/affection/{enc}/plain/map', cell['comparator_value'], f'B02 SC5 {enc} comparator_value'),
                     None, None]
            ours = [r17v('affection', enc, 'plain', 'cand'),
                    put(f'photo/affection/{enc}/photo/map', cell['candidate_value'], f'B02 SC5 {enc} candidate_value'),
                    None, None]
        # the photo model on one encoder must be the same in the plain and teacher cells
        a, b = (c17[f'SC4/affection/{enc}/emotion/ami/9/{arm}']['cand'] for arm in ('plain', 'teacher'))
        assert abs(a - b) < 1e-12
        rows.append((f'Plain embedding & {name}', plain))
        rows.append((rf'\method (ours) & {name}', ours))
    body = bold_columns(rows, 4)
    return r"""\begin{table}[t]
  \centering
  \small
  \setlength{\tabcolsep}{2.7pt}
  \begin{tabular}{@{}llcccc@{}}
    \toprule
    & & \multicolumn{2}{c}{Affection} & \multicolumn{2}{c}{EmoSet} \\
    \cmidrule(lr){3-4} \cmidrule(l){5-6}
    Method & Backbone & AMI & mAP & AMI & mAP \\
    \midrule
""" + '\n'.join('    ' + line for line in body) + r"""
    \bottomrule
  \end{tabular}
  \caption{Emotion on photos: the Affection~\cite{achlioptas2023affection} test set (8,501 photos) and the EmoSet~\cite{yang2023emoset} test set (17,348 photos, without near copies of Affection's training photos). \method is trained on ArtELingo paintings, then fine-tuned on Affection's training photos and their explanations without emotion labels, and is not trained on EmoSet. Clustering is the AMI at the class count (9 for Affection, 8 for EmoSet), and retrieval is the label mAP with the read-out that each method picked on the validation split of each dataset. EmoSet is scored on Qwen3 only. \method is a mean over five random seeds. The best result per column is in bold.}
  \label{tab:photos}
\end{table}
"""


if __name__ == '__main__':
    out = Path(sys.argv[1])
    (out / 'encoders.tex').write_text(encoders_table())
    (out / 'photos.tex').write_text(photos_table())
    (HERE / 'records').mkdir(exist_ok=True)
    (HERE / 'records' / 'cells.json').write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f'wrote {out}/encoders.tex, {out}/photos.tex, {len(record)} cells')
