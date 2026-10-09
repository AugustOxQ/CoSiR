"""Render the weekly draft figures from the controller's transcribed facts.

Run without arguments. Times are local wall-clock labels, as supplied in facts.md.
Approximate events use the controller-specified nominal times and hollow markers.
Writes timeline.png, flowchart.png, flowchart.dot and flowchart.svg beside this file.
"""

# id, date, supplied time, lane, status, short label, table event, table outcome,
# source paths relative to docs/reports/weekly/drafts/ (empty means session notes).
EVENTS = [
    ('L1', '30 Sep', '15:15 (commit)', 1, 'diag', 'Factor headroom probe', 'Factor headroom probe (lead-in).', 'The label oracle reached 49.8% versus repaired factors at 20.5%.', ['../../auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md']),
    ('L2', '30 Sep', '17:45 to 18:19 (commits)', 1, 'fail', 'Factor selection stopped', 'Factor learning: painting-level agreement crossed with CLIP-image-cluster condition episodes (lead-in).', 'No cell qualified: style episodes failed the emotion guard and sparsity gate.', ['../../auto/v2/2026-10-16_candidate_a_factor_learning_selection.md']),
    ('L3', '1 Oct', '00:40 to 02:06 (commits)', 1, 'pass', 'Affect factors confirmed (lead-in)', 'Affect-signal factor learning (SE): GoEmotions affect clusters plus CLIP image clusters (lead-in).', 'Held emotion improved +2.08 and style +0.65 over the matched control.', ['../../auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md', '../../auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md']),
    ('1', '2 Oct', '02:53 (commit)', 0, 'diag', 'Previous weekly report committed', 'Last weekly report committed.', 'The previous weekly report was recorded.', ['../../weekly/2026-09-30_percept_buddy_to_v2.md']),
    ('2', '2 Oct', '04:02 (commit)', 0, 'diag', 'Value episodes resemble recognition', 'Support-set baseline spike.', 'The query-free prototype scored 22.83; adding the query gave only +0.57.', ['../../auto/v2/2026-10-22_support_baseline_spike.md']),
    ('3', '2 Oct', '04:04 (commit)', 0, 'diag', 'CVPR literature review completed', 'CVPR literature review.', 'No prior paper defined the task to our knowledge; label-free condition discovery already existed.', ['../../literature/2026-10-21_cvpr_literature_review.md']),
    ('4', '2 Oct', '04:39 (commit)', 1, 'diag', 'Aspect spike shows headroom', 'Aspect-episode spike.', 'No factor model selected the aspect: affect factors scored 11.32 versus CLIP at 11.13.', ['../../auto/v2/2026-10-23_aspect_episode_spike.md']),
    ('5', '2 Oct', '05:01 (commit)', 0, 'diag', 'Task novelty checked', 'Novelty check.', 'The task and combination were new to our knowledge; the individual ingredients existed.', ['../../literature/2026-10-24_aspect_task_novelty_check.md']),
    ('6', '2 Oct', 'day (brainstorm, progress note 16:29)', 3, 'decision', 'User: aspect, not value', '**User:** The condition specifies an aspect, not a value; choose method A, ArtELingo primary, plus CUB, SemArt and GeneCIS.', 'The task became aspect episodes.', ['../../../superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md']),
    ('7', '2 Oct', '18:59 (commit)', 1, 'diag', 'Backbones show modality split', 'Frozen-backbone check.', 'Emotion lived mainly in captions and style in images; weak-side probes stayed flat.', ['../../auto/v2/2026-10-25_backbone_check.md']),
    ('8', '2 Oct', '19:22 (commit)', 0, 'diag', 'CVPR publication plan written', 'CVPR publication plan: claims, baselines, GO bar and schedule.', 'The plan was written.', ['../../../superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md']),
    ('9', '2 Oct', '21:57 (commit)', 0, 'diag', 'Plan review: Major Revision', 'ARS plan review: Major Revision.', 'Condition gain became co-primary, with a condition-free control and stronger evaluation requirements.', ['../../auto/v2/2026-10-27_ars_plan_review.md']),
    ('10', '3 Oct', '00:31 (commit)', 0, 'diag', 'Publication plan revised', 'Plan revision 2.', 'The plan was revised.', ['../../../superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md']),
    ('11', '3 Oct', '02:04 to 04:51 (commits, overnight)', 1, 'fail', 'E3: method A NO-GO', 'Baselines, label-free partitions and method A built and run (E0 to E3); held-out-aspect and in-context probes failed.', 'Method A lost 2.96 R@1 to its own uniform-weight control.', ['../../auto/v2/2026-10-30_aspect_baselines.md', '../../auto/v2/2026-10-31_pseudo_partitions.md', '../../auto/v2/2026-11-01_aspect_factor_gonogo.md']),
    ('12', '3 Oct', '13:54 (commit)', 3, 'decision', 'User: repair method A', '**User:** Repair method A (A′) before giving up.', 'A repair was requested.', []),
    ('13', '3 Oct', '15:30 to 17:07 (commits)', 1, 'fail', 'Nested repair reaches stop', 'ARS repair-order review; nested score (A′) and label-trained factors as a diagnostic.', 'The nested score reached 16.52 versus its control at 16.55; the declared rule stopped repair.', ['../../auto/v2/2026-11-04_ars_repair_order_review.md', '../../auto/v2/2026-11-05_method_repair_diagnostics.md']),
    ('14', '3 Oct', '20:51 (commit)', 1, 'fail', 'Larger in-context probe fails', 'In-context probe with Qwen3-VL-8B.', 'It gained +1.07 R@1 over cosine, but condition gain was only +0.21.', ['../../stage/2026-10-04_aspect_conditioned_similarity_methods.md']),
    ('15', '3 Oct', 'evening', 3, 'decision', 'User: try new methods', '**User:** Not branch 3 yet; try new methods first.', 'Candidates N1 to N6 were drafted and checked against the literature.', ['../../stage/2026-10-04_aspect_conditioned_similarity_methods.md']),
    ('16', '4 Oct', '03:09 to 05:08 (commits, overnight, fully automate)', 2, 'fail', 'Quick checks: fusion fails', 'Label-probe check (D0), centered rule (N1), find then select (N2), partition-head readers (N6, N6c).', 'The partition reader reached condition gain 4.41, but fusion beat its control by only +0.23 and failed.', ['../../auto/v2/2026-11-08_new_method_quick_checks.md']),
    ('17', '4 Oct', '14:50 to 15:23 (commits)', 2, 'diag', 'Stage report locates bottleneck', 'Stage report final-reviewed: readers lost either rate as they gained condition gain.', 'Told the grouping, the heads gained +1.14 over their counterpart; the reader gained +0.14.', ['../../stage/2026-10-04_aspect_conditioned_similarity_methods.md']),
    ('18', '5 Oct', '03:04 (commit)', 0, 'diag', 'Stage report Q&A revision', 'Stage report revised after a design Q&A.', 'No number changed.', ['../../stage/2026-10-04_aspect_conditioned_similarity_methods.md']),
    ('19', '5 Oct', '12:57 to 15:01 (commits)', 2, 'pass', 'Leiden affect communities kept', 'Grouping quality: partition profile, Leiden affect communities and grouping sweep.', 'Leiden gained +1.64 when told the grouping versus k-means at +1.10.', ['../../auto/v2/2026-11-12_partition_quality_leiden_communities.md']),
    ('20', '5 Oct', 'by 15:01 (recorded in the draft report)', 3, 'decision', 'User: redesign the groupings', '**User:** The grouping component is not well shaped; redesign it before returning to bars and readers.', 'Grouping redesign became the next step.', ['../../auto/v2/2026-11-12_partition_quality_leiden_communities.md']),
    ('21', '5 Oct', '15:01 to 21:44 (commits)', 2, 'pass', 'Style grouping raises ceiling', 'Redesign: literature synthesis, initial checks and style groupings; placeability adopted, other additions dropped.', 'The CSD style grouping reached told gain +2.23, but the reader fell to +0.06.', ['../../auto/v2/2026-11-12_partition_quality_leiden_communities.md']),
    ('22', '6 Oct', 'night (before 01:20)', 3, 'decision', 'User: fix reader with CSD', '**User:** Plan (a), fix the reader with the CSD grouping in the set.', 'Reader repair was chosen.', []),
    ('23', '6 Oct', '01:20; 02:25 (~, memory)', 3, 'decision', 'User adopts methodology fixes', 'ARS methodology review: Major Revision; **User:** Adopt all fixes.', 'The methodology fixes were adopted.', ['../../auto/v2/2026-11-17_ars_reader_fix_plan_review.md']),
    ('24', '6 Oct', '02:51 to 03:43 (run)', 2, 'fail', 'Reader round 1 near miss', 'Reader round 1: noise-scaled, learned and confidence-gated readers, with and without CSD.', 'No candidate cleared: the best margin was +0.444 versus the +0.5 bar.', ['../../auto/v2/2026-11-18_reader_fix_csd.md']),
    ('25', '6 Oct', '07:59 to 15:53 (commits)', 0, 'diag', 'Reader briefing final reviewed', 'Round 1 report final-reviewed; user-read briefing written and checked.', 'The report and briefing were reviewed.', ['../../../user_read/2026-10-06_reader_fix.md']),
    ('26', '6 Oct', '15:43 to 17:01 (run; rule applied 16:57)', 2, 'fail', 'Reader round 2 fails', 'Reader round 2: adaptation, retraining on impure practice episodes and top-k restriction.', 'No candidate cleared: the unchanged gated reader reached +0.472 versus the +0.5 bar.', ['../../auto/v2/2026-11-19_reader_fix_round2.md']),
    ('27', '6 Oct', '18:11 (~, memory)', 3, 'decision', 'User: keep improving R1', '**User:** Not design L now; keep improving the gated reader (R1).', 'Work continued on the gated reader.', []),
    ('28', '6 Oct', '18:18 (commit)', 2, 'diag', 'No-caption spike loses margin', 'Exploratory no-caption spike (A1c): affect, image and CSD groupings.', 'The pooled margin fell to +0.256 from +0.444.', []),
    ('29', '6 Oct', '18:16 to 18:46', 2, 'diag', 'Brainstorm: steer affect only', 'Exploratory gated-reader levers brainstorm; one-sided affect steering (AFF) emerged.', 'Development margin reached +0.700 on seed 42, inflated by selection.', ['../../auto/v2/2026-11-20_r1_levers_brainstorm.md']),
    ('30', '6 Oct', '19:10', 3, 'decision', 'User: fresh-test affect steering', '**User:** Send one-sided affect steering straight to fresh seeds, with the gated reader beside it.', 'A fresh test was requested.', []),
    ('31', '6 Oct', '19:10 to 21:17 (run; verdict 21:16)', 2, 'pass', 'Round 3: AFF GO', 'Round 3: one-sided affect steering on fresh episode draws, with the gated reader beside it.', 'GO: affect steering reached 18.88 versus the strongest condition-free scorer at 18.29.', ['../../auto/v2/2026-11-21_round3_affect_gate.md']),
    ('32', '6 Oct', '22:29 to 23:34 (commits)', 0, 'diag', 'Affect steering review confirmed', 'Round 3 final review confirmed with fixes; user-read briefing covered rounds 1 to 3.', 'The result was confirmed with fixes.', ['../../../user_read/2026-10-06_reader_fix_affect_steering.md']),
    ('33', '7 Oct', '00:02 (~, memory)', 3, 'decision', 'User: improve before held test', '**User:** Option B, improve the method before the held-split paper test.', 'The held test was deferred for method improvements.', []),
    ('34', '7 Oct', '00:17 to 02:36 (run)', 2, 'fail', 'Round 4: vetoes killed', 'Round 4: image-agreement abstention, a CSD-informed second reader and both together.', 'No veto beat one-sided affect steering; the CSD-informed vetoes also missed the stronger bar.', ['../../auto/v2/2026-11-22_round4_aff_vetoes.md']),
    ('35', '7 Oct', '03:39 to 03:55 (commits)', 0, 'diag', 'Veto report final reviewed', 'Round 4 report final-reviewed; user-read briefing written.', 'The veto report and briefing were reviewed.', ['../../../user_read/2026-10-07_round4_vetoes.md']),
    ('36', '7 Oct', 'before 07:43', 3, 'decision', 'User: caption placement first', '**User:** Idea 3, GoEmotions placement of captions, before the held test, to keep both held reads.', 'Caption placement was prioritized.', []),
    ('37', '7 Oct', '07:56 (commit)', 0, 'diag', 'CVPR readiness memo written', 'CVPR readiness memo: the plan still needs held splits; CUB and SemArt had not started.', 'Contributions ranked task and protocol, then analysis, then method.', ['../../stage/2026-10-07_cvpr_readiness.md']),
    ('38', '7 Oct', '~08:00', 3, 'decision', 'User: add fine-tuned comparator', '**User:** Hold paper framing; adopt a lightweight fine-tuned CLIP comparator using linear probe, last block and LoRA, with no full fine-tuning.', 'The lightweight comparator was adopted.', ['../../stage/2026-10-07_cvpr_readiness.md', '../../../superpowers/specs/2026-10-07-clip-lightweight-ft-design.md']),
    ('39', '7 Oct', '08:05 to 08:16 (commits)', 2, 'running', 'Caption placement and comparator underway', 'Caption-placement spec committed; CLIP fine-tuning spec, plan and evaluation script in progress.', 'No results yet.', ['../../../superpowers/specs/2026-10-07-idea3-goemotions-design.md', '../../../superpowers/specs/2026-10-07-clip-lightweight-ft-design.md']),
]

# Exact quoted labels from facts.md section 2; current best is a separate badge.
NODES = [
    ('N0', 'base', "Last week's end: repaired factors R3 (30 Sep)"),
    ('N1', 'fail', 'Factor learning 2×2 (painting agreement, CLIP-cluster episodes): stopped'),
    ('N2', 'pass', 'Affect factor learning (SE): confirmed on held value episodes'),
    ('N3', 'reframe', 'Value episodes = recognition → task redefined as aspect episodes (2 Oct)'),
    ('N4', 'reframe', 'CVPR plan + ARS review: condition gain, matched controls'),
    ('N5', 'diag', 'Metric-from-pairs baselines (E1): none separates aspects; RCA = GO bar'),
    ('N6', 'fail', 'In-context MLLM (Qwen3-VL 2B, 8B): does not work'),
    ('N7', 'fail', 'Method A: factors on pseudo-aspect episodes (E3): NO-GO'),
    ('N8', 'fail', 'Repair A′ (nested score) + label-trained factors: stopped'),
    ('N9', 'fail', 'Centered rule (N1): fails matched control'),
    ('N10', 'fail', 'Find then select (N2): fails'),
    ('N11', 'diag', 'Label-probe check (D0): task is learnable'),
    ('N12', 'fail', 'Partition-head reader (N6, N6c): biggest gain, fusion fails'),
    ('N13', 'diag', "Stage report: told grouping +1.14 vs reader +0.14 → the reader's pick blocks"),
    ('N14', 'pass', 'Leiden affect communities: kept'),
    ('N15', 'fail', 'Sibling-aware agreement, Gram style, Leiden image/caption: dropped'),
    ('N16', 'pass', 'CSD style grouping: highest told ceiling, kept as ablation (A1)'),
    ('N17', 'fail', 'Reader round 1 (noise-scaled, learned, gated): near miss +0.444'),
    ('N18', 'fail', 'Reader round 2 (adapted, retrained, top-k): fails'),
    ('N19', 'diag', 'No-caption spike (A1c): exploratory'),
    ('N20', 'diag', 'Brainstorm: the margin lives on the emotion side'),
    ('N21', 'pass', 'One-sided affect steering (AFF): GO on fresh seeds, +0.59 R@1'),
    ('N22', 'fail', 'Round 4: three vetoes on AFF: killed'),
    ('N23', 'running', 'Idea 3: GoEmotions placement of captions'),
    ('N24', 'running', 'Fine-tuned CLIP comparator (linear probe, last block, LoRA)'),
    ('N25', 'running', 'Held-split paper test (0 of 2 reads used)'),
    ('NB', 'diag', 'Condition-free bar B: 12.96 → 18.34 (B′(A1) 18.81)'),
]

# source, target, style, surviving ingredient (only dotted edges have labels).
EDGES = [
    ('N0', 'N1', 'solid', ''), ('N0', 'N2', 'solid', ''),
    ('N2', 'N3', 'solid', ''), ('N3', 'N4', 'solid', ''),
    ('N4', 'N5', 'solid', ''), ('N4', 'N6', 'solid', ''),
    ('N4', 'N7', 'solid', ''), ('N7', 'N8', 'solid', ''),
    ('N7', 'N9', 'solid', ''), ('N7', 'N10', 'solid', ''),
    ('N4', 'N11', 'solid', ''), ('N11', 'N12', 'solid', ''),
    ('N12', 'N13', 'solid', ''), ('N13', 'N14', 'solid', ''),
    ('N14', 'N15', 'solid', ''), ('N14', 'N16', 'solid', ''),
    ('N13', 'N17', 'solid', ''), ('N14', 'N17', 'solid', ''),
    ('N16', 'N17', 'solid', ''), ('N17', 'N18', 'solid', ''),
    ('N18', 'N19', 'solid', ''), ('N18', 'N20', 'solid', ''),
    ('N20', 'N21', 'solid', ''), ('N21', 'N22', 'solid', ''),
    ('N16', 'N22', 'solid', ''), ('N21', 'N23', 'solid', ''),
    ('N21', 'N24', 'solid', ''), ('N23', 'N25', 'solid', ''),
    ('N24', 'N25', 'solid', ''),
    ('N9', 'NB', 'dotted', 'centered term'),
    ('N12', 'NB', 'dotted', 'averaged heads'),
    ('N16', 'NB', 'dotted', 'B′(A1)'),
]

from collections import Counter
from html import escape
from pathlib import Path
import re
import subprocess
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

OUT = Path(__file__).resolve().parent
TEAL, ORANGE, SLATE = '#1B8A7A', '#D9822B', '#5B6C8F'
STATUS = {
    'base': ('#EEEEEE', '#525252', '#202020'),
    'reframe': ('#FFF4CC', '#514A32', '#202020'),
    'pass': (TEAL, '#14685D', 'white'),
    'fail': ('#F2C6A0', '#82502F', '#202020'),
    'diag': ('#DCE3F0', SLATE, '#202020'),
    'running': ('white', '#808080', '#333333'),
}


# Controller-specified display nominals, not additional factual timestamps:
# row 6 day -> 12:00; 15 evening -> 20:00; 20 by 15:01 -> 15:01;
# 22 night (before 01:20) -> 00:30; 23 -> 01:20 and 02:25;
# 27 -> 18:11; 33 -> 00:02; 36 before 07:43 -> 07:30; 38 ~08:00 -> 08:00.
NOMINALS = {'6': ['12:00'], '15': ['20:00'], '20': ['15:01'],
            '22': ['00:30'], '23': ['01:20', '02:25'], '27': ['18:11'],
            '33': ['00:02'], '36': ['07:30'], '38': ['08:00']}
SHAPES = {
    'o': '1 2 3 4 5 7 8 9 10 17 18 25 32 35 37'.split(),
    's': 'L1 L2 L3 11 13 14 16'.split(),
    '^': '19 21 24 26 28 29 31 34 39'.split(),
    'D': '6 12 15 20 22 23 27 30 33 36 38'.split(),
}


def clock_hour(clock):
    hour, minute = map(int, clock.split(':'))
    return hour + minute / 60


def timeline():
    # The requested date list contains eight days, including the lead-in.
    days = ['30 Sep'] + [f'{day} Oct' for day in range(1, 8)]
    fig, axes = plt.subplots(len(days), 1, figsize=(18, 13), dpi=200, sharex=True)
    fig.subplots_adjust(left=.12, right=.98, top=.92, bottom=.14, hspace=.10)
    fig.suptitle("CoSiR v2, 30 Sep to 7 Oct: from method A's NO-GO to one-sided affect steering",
                 fontsize=17, fontweight='bold', y=.975)
    marker_by_id = {event_id: shape for shape, ids in SHAPES.items() for event_id in ids}
    assert set(marker_by_id) == {event[0] for event in EVENTS}
    annotations = []
    for ax, day in zip(axes, days):
        ax.set_xlim(0, 24)
        ax.set_ylim(-3.8, 3.8)
        ax.set_facecolor('#EEEEEE' if day == '30 Sep' else 'white')
        ax.axhline(0, color='#B9C1CA', linewidth=.7, zorder=1)
        ax.set_yticks([])
        ax.set_ylabel('30 Sep (lead-in)' if day == '30 Sep' else day,
                      rotation=0, ha='right', va='center', labelpad=12,
                      fontsize=10, fontweight='bold')
        ax.set_xticks(range(0, 25, 3))
        ax.grid(axis='x', color='#E1E5EA', linewidth=.5)
        ax.tick_params(axis='x', length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        records = []
        for event in (event for event in EVENTS if event[1] == day):
            event_id, _, supplied, _, _, label, *_ = event
            clocks = re.findall(r'\d{2}:\d{2}', supplied)
            starts = NOMINALS.get(event_id, clocks[:1])
            end = clock_hour(clocks[1]) if ' to ' in supplied else None
            for index, start in enumerate(starts):
                short = label
                if event_id == '6':
                    short = 'User: aspect task (time approx.)'
                elif event_id == '23':
                    short = ['Methodology review: Major Revision', 'User adopts methodology fixes'][index]
                elif event_id == '31':
                    short = 'Round 3: AFF GO on fresh seeds'
                records.append((clock_hour(start), event_id, short, end))
        records.sort(key=lambda record: record[0])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        row_labels = []
        level_boxes = {}
        for index, (start, event_id, label, end) in enumerate(records):
            shape = marker_by_id[event_id]
            color = ('#111111' if shape == 'D' else TEAL if event_id in {'L3', '31'}
                     else ORANGE if event_id in {'L2', '11', '13', '14', '16', '24', '26', '34'}
                     else '#888888' if event_id == '39' else SLATE)
            if end is not None:
                ax.plot([start, end], [0, 0], color=color, lw=3, alpha=.45, zorder=2)
            ax.scatter([start], [0], marker=shape, s=28,
                       facecolor='white' if event_id in NOMINALS else color,
                       edgecolor=color, linewidth=1.2,
                       linestyle='--' if event_id == '39' else '-', zorder=5)
            # Alternate sides; try up to three levels on that side before the other.
            side = 1 if index % 2 == 0 else -1
            candidates = [(side, level) for level in range(1, 4)]
            candidates += [(-side, level) for level in range(1, 4)]
            text = ax.text(start, 0, label, ha='center', va='center', fontsize=9,
                           color=color, fontweight='bold' if event_id in {'11', '31', '34'} else 'normal',
                           bbox=dict(facecolor=ax.get_facecolor(), edgecolor='none', pad=1), zorder=6)
            # Keep the whole label inside the shared 00:00 to 24:00 axis.
            width = text.get_window_extent(renderer).width
            x_pixel = ax.transData.transform((start, 0))[0]
            x_pixel = max(ax.bbox.x0 + width / 2 + 3,
                          min(ax.bbox.x1 - width / 2 - 3, x_pixel))
            label_x = ax.transData.inverted().transform((x_pixel, 0))[0]
            for sign, level in candidates:
                text.set_position((label_x, sign * level))
                bounds = text.get_window_extent(renderer).expanded(1.025, 1.18)
                previous = level_boxes.get((sign, level))
                if previous is not None and bounds.overlaps(previous):
                    continue
                if any(bounds.overlaps(box) for _, box in row_labels):
                    continue
                level_boxes[(sign, level)] = bounds
                row_labels.append((text, bounds))
                break
            else:
                raise AssertionError(f'No collision-free label level: {day}, {event_id}')
            ax.annotate('', xy=(start, 0), xytext=text.get_position(),
                        arrowprops=dict(arrowstyle='-', color=color, lw=.5, alpha=.65), zorder=3)
        annotations.append((day, [text for text, _ in row_labels]))
    axes[-1].set_xticklabels([f'{hour:02d}:00' for hour in range(0, 25, 3)])
    axes[-1].set_xlabel('Amsterdam time', fontsize=10, labelpad=9)
    shapes = [('o', 'Task / plan / review / report'), ('s', 'Method / probe / quick check'),
              ('^', 'Grouping / reader'), ('D', 'User decision')]
    shape_handles = [Line2D([], [], marker=shape, color='#444444', linestyle='none', label=label)
                     for shape, label in shapes]
    color_handles = [Patch(facecolor=color, edgecolor=color, label=label)
                     for color, label in [(TEAL, 'Worked / kept / GO'), (ORANGE, 'Failed / stopped / killed'),
                                          (SLATE, 'Diagnostic / insight / report / review')]]
    color_handles += [Patch(facecolor='#BBBBBB', edgecolor='#888888', linestyle='--', label='Work in progress'),
                      Patch(facecolor='#111111', label='User decision'),
                      Line2D([], [], marker='o', markerfacecolor='white', color='#444444',
                             linestyle='none', label='hollow: approximate time')]
    fig.legend(handles=shape_handles, loc='lower center', bbox_to_anchor=(.55, .059),
               ncol=4, frameon=False, fontsize=9)
    fig.legend(handles=color_handles, loc='lower center', bbox_to_anchor=(.55, .005),
               ncol=3, frameon=False, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    total_overlaps = 0
    for day, labels in annotations:
        boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in labels]
        overlaps = sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
        total_overlaps += overlaps
        print(f'Timeline {day}: {len(labels)} labels, {overlaps} overlaps')
    print(f'Timeline overlap count: {total_overlaps}')
    assert total_overlaps == 0
    fig.savefig(OUT / 'timeline.png', dpi=200, facecolor='white')
    plt.close(fig)
    with Image.open(OUT / 'timeline.png') as rendered:
        print(f'timeline.png: {rendered.size}')


def wrapped_label(label):
    return textwrap.fill(label, width=40, break_long_words=False, break_on_hyphens=False)


def mermaid():
    lines = ['flowchart TD']
    for node, status, label in NODES:
        label = wrapped_label(label).replace('\n', '<br/>')
        if node == 'N21':
            label += '<br/>current best'
        lines.append(f'    {node}["{label}"]:::{status}')
    for start, end, style, label in EDGES:
        if style == 'dotted':
            lines.append(f'    {start} -.->|"{label}"| {end}')
        else:
            lines.append(f'    {start} --> {end}')
    for status, (fill, border, text) in STATUS.items():
        extra = ',stroke-dasharray:5 4' if status == 'running' else ''
        lines.append(f'    classDef {status} fill:{fill},stroke:{border},color:{text}{extra};')
    lines.append('    style N21 stroke:#104C44,stroke-width:4px;')
    return '\n'.join(lines) + '\n'


def flowchart():
    spine = 'N0 N2 N3 N4 N11 N12 N13 N17 N18 N20 N21 N25'.split()
    spine_edges = set(zip(spine, spine[1:]))
    lines = ['digraph history {',
             '  graph [rankdir=TB, nodesep=0.35, ranksep=0.45, splines=true, newrank=true,',
             '         fontname="DejaVu Sans", fontsize=22, labelloc=t,',
             '         label="Experiment history, 30 Sep to 7 Oct"];',
             '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=13, margin="0.12,0.09"];',
             '  edge [color="#555555", arrowsize=0.7, fontname="DejaVu Sans", fontsize=10];']
    for node, status, label in NODES:
        fill, border, foreground = STATUS[status]
        # Two or three content lines, in addition to the small node identifier.
        width = min(36, max(18, (len(label) + 1) // 2))
        content = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
        while len(content) > 3:
            width += 1
            content = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
        if node == 'N0':
            content = ["Last week's end:", 'repaired factors R3', '(30 Sep)']
        elif node == 'NB':
            content = ['Condition-free bar B:', '12.96 → 18.34 (B′(A1) 18.81)']
        id_color = '#D8F0EC' if status == 'pass' else '#666666'
        html = f'<FONT POINT-SIZE="10" COLOR="{id_color}">{node}</FONT><BR/>'
        html += '<BR/>'.join(escape(part) for part in content)
        if node == 'NB':
            html += ('<BR/><FONT POINT-SIZE="10">from: centered term (N9), '
                     'averaged heads (N12), style grouping (N16)</FONT>')
        if node == 'N21':
            html += '<BR/><B>current best</B>'
        attrs = [f'label=<{html}>', f'fillcolor="{fill}"', f'color="{border}"',
                 f'fontcolor="{foreground}"']
        if node in spine:
            attrs.append('group="spine"')
        if status == 'running':
            attrs.append('style="rounded,dashed"')
        if node == 'N21':
            attrs.append('penwidth=3.5')
        lines.append(f'  {node} [{", ".join(attrs)}];')
    lines += ['  { rank=same; N1; N2; }',
              '  { rank=same; N5; N7; N11; N6; }',
              '  { rank=same; N8; N9; N10; N12; }',
              '  { rank=same; N13; NB; }',
              '  { rank=same; N15; N16; }',
              '  { rank=same; N19; N20; }',
              '  { rank=same; N21; N22; }',
              '  { rank=same; N23; N24; }']
    # Outgoing order pins the failed branches to their requested sides.
    lines += ['  N4 [ordering=out];', '  N18 [ordering=out];']
    edge_order = {('N4', target): index for index, target in enumerate(['N5', 'N7', 'N11', 'N6'])}
    ordered_edges = sorted(EDGES, key=lambda edge: (int(edge[0][1:]), edge_order.get(edge[:2], 0)))
    for source, target, style, label in ordered_edges:
        attrs = [f'style={style}']
        if style == 'dotted':
            attrs += [f'color="{SLATE}"', f'fontcolor="{SLATE}"',
                      'constraint=false']
        if (source, target) == ('N14', 'N17'):
            attrs += ['tailport=w', 'headport=n']
        if (source, target) == ('N16', 'NB'):
            attrs += ['tailport=se', 'headport=e']
        if (source, target) in spine_edges:
            attrs.append('weight=10')
        lines.append(f'  {source} -> {target} [{", ".join(attrs)}];')
    labels = {'base': 'Starting point', 'reframe': 'Task / plan reframe', 'pass': 'Worked / kept',
              'fail': 'Failed / stopped', 'diag': 'Diagnostic / insight', 'running': 'In progress / pending'}
    cells = []
    for status, label in labels.items():
        fill, border, _ = STATUS[status]
        cells.append(f'<TD BGCOLOR="{fill}" COLOR="{border}" BORDER="1">  </TD><TD ALIGN="LEFT">{label}</TD>')
    rows = ['<TR>' + ''.join(cells[i:i + 3]) + '</TR>' for i in range(0, 6, 3)]
    rows += ['<TR><TD COLSPAN="6" ALIGN="LEFT">solid: led to</TD></TR>',
             '<TR><TD COLSPAN="6" ALIGN="LEFT">dotted: an ingredient survived into the condition-free bar</TD></TR>']
    lines += ['  subgraph cluster_legend {', '    label=""; color="#DDDDDD"; margin=10;',
              '    legend [shape=plain, style="", fontsize=10, label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="5">'
              + ''.join(rows) + '</TABLE>>];', '  }',
              '  { rank=sink; legend; }',
              '  // Invisible constraints are layout only, not experiment-history edges.',
              '  N13 -> NB [style=invis, weight=100];',
              '  N15 -> N16 [style=invis, weight=100];',
              '  N19 -> N20 [style=invis, weight=100];',
              '  N25 -> legend [style=invis, weight=10];', '}']
    dot = '\n'.join(lines) + '\n'
    (OUT / 'flowchart.dot').write_text(dot, encoding='utf-8')
    facts = (OUT.parents[3] / '.ccg/tasks/weekly-summary-draft/facts.md').read_text()
    tree = facts.split('## 2.')[1].split('## 3.')[0].split('```')[1]
    expected = Counter(re.findall(r'(N(?:\d+|B))\s+(?:->|-\.->)\s+(N(?:\d+|B))', tree))
    history_edges = '\n'.join(line for line in dot.splitlines() if 'style=invis' not in line)
    actual = Counter(re.findall(r'(N(?:\d+|B))\s+->\s+(N(?:\d+|B))', history_edges))
    print(f'DOT edge count: {sum(actual.values())}; facts: {sum(expected.values())}; '
          f'missing: {sum((expected - actual).values())}; extra: {sum((actual - expected).values())}; '
          f'layout-only invisible edges: {dot.count("style=invis")}')
    assert actual == expected and all(count == 1 for count in actual.values())
    for format_, options in [('png', ['-Gdpi=200']), ('svg', [])]:
        subprocess.run(['/root/miniconda3/bin/dot', f'-T{format_}', *options,
                        str(OUT / 'flowchart.dot'), '-o', str(OUT / f'flowchart.{format_}')], check=True)
    with Image.open(OUT / 'flowchart.png') as rendered:
        width, height = rendered.size
        print(f'flowchart.png: {rendered.size}')
        assert width < height and 2500 <= height <= 4500


if __name__ == '__main__':
    timeline()
    flowchart()
