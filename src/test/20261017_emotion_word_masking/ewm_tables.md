# Emotion word masking

Train-side check. AMI: mean ± population sd across seeds 0/1/2. Probe: five grouped folds.
Labels enter scoring only. Missing views are skipped. Unmasked probes are copied from pav results.json.

## Word statistics

| Mask | Captions matched | Share |
|---|---:|---:|
| label | 13195 / 99999 | 0.131951 |
| broad | 40586 / 99999 | 0.405864 |

Among captions with label matches, at least one matched word maps to the own label.
Agreement: 11615 captions; share 0.880258.

| Own emotion | n | Label share | Broad share | Own-label agreement among label matches |
|---|---:|---:|---:|---:|
| amusement | 10180 | 0.112377 | 0.360904 | 0.851399 |
| anger | 1458 | 0.285322 | 0.419753 | 0.913462 |
| awe | 16237 | 0.079633 | 0.279670 | 0.851508 |
| contentment | 28366 | 0.072199 | 0.456004 | 0.885742 |
| disgust | 4820 | 0.101867 | 0.273029 | 0.743381 |
| excitement | 8429 | 0.185075 | 0.386523 | 0.947436 |
| fear | 9100 | 0.267912 | 0.464176 | 0.897867 |
| sadness | 10485 | 0.331235 | 0.518932 | 0.954218 |
| something else | 10924 | 0.030392 | 0.419718 | 0.000000 |

### label: top 30 matches (occurrences, lowercased)

| Word | Count |
|---|---:|
| sad | 3105 |
| content | 1396 |
| scary | 1095 |
| awe | 937 |
| sadness | 756 |
| exciting | 742 |
| fear | 645 |
| excited | 595 |
| angry | 560 |
| contentment | 512 |
| amusing | 502 |
| excitement | 378 |
| amused | 357 |
| fearful | 234 |
| scared | 229 |
| anger | 229 |
| awesome | 191 |
| afraid | 182 |
| disgusting | 163 |
| amusement | 151 |
| frightening | 139 |
| disgust | 127 |
| disgusted | 103 |
| contented | 97 |
| terrifying | 87 |
| awe-inspiring | 56 |
| scares | 53 |
| amuses | 51 |
| excites | 39 |
| awed | 36 |

### broad: top 30 matches (occurrences, lowercased)

| Word | Count |
|---|---:|
| feel | 8956 |
| love | 3177 |
| sad | 3105 |
| peaceful | 2554 |
| happy | 2353 |
| feeling | 2093 |
| calm | 1993 |
| feels | 1624 |
| content | 1396 |
| relaxing | 1111 |
| scary | 1095 |
| fun | 1070 |
| calming | 1063 |
| awe | 937 |
| funny | 936 |
| lovely | 909 |
| relaxed | 865 |
| sadness | 756 |
| exciting | 742 |
| peace | 679 |
| creepy | 655 |
| fear | 645 |
| serene | 603 |
| excited | 595 |
| pleasant | 576 |
| angry | 560 |
| gloomy | 554 |
| boring | 548 |
| contentment | 512 |
| amusing | 502 |

## Full and no-word label distributions

| Group | n | amusement | anger | awe | contentment | disgust | excitement | fear | sadness | something else |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 99999 | 10180 (0.1018) | 1458 (0.0146) | 16237 (0.1624) | 28366 (0.2837) | 4820 (0.0482) | 8429 (0.0843) | 9100 (0.0910) | 10485 (0.1049) | 10924 (0.1092) |
| no_label_word | 86804 | 9036 (0.1041) | 1042 (0.0120) | 14944 (0.1722) | 26318 (0.3032) | 4329 (0.0499) | 6869 (0.0791) | 6662 (0.0767) | 7012 (0.0808) | 10592 (0.1220) |
| no_broad_word | 59413 | 6506 (0.1095) | 846 (0.0142) | 11696 (0.1969) | 15431 (0.2597) | 3504 (0.0590) | 5171 (0.0870) | 4876 (0.0821) | 5044 (0.0849) | 6339 (0.1067) |

## Caption emotion AMI

| View | k=9 | k=20 | k=40 |
|---|---:|---:|---:|
| default/unmasked | 0.098293 ± 0.008553 | 0.121728 ± 0.006921 | 0.128728 ± 0.004409 |
| default/masked_label | 0.078521 ± 0.004721 | 0.095310 ± 0.003654 | 0.108796 ± 0.002891 |
| default/masked_broad | 0.058510 ± 0.006350 | 0.069409 ± 0.003447 | 0.082379 ± 0.003351 |
| emotion_focus/unmasked | 0.304819 ± 0.007292 | 0.278238 ± 0.003696 | 0.263681 ± 0.001109 |
| emotion_focus/masked_label | 0.238731 ± 0.006813 | 0.230735 ± 0.001878 | 0.221703 ± 0.000807 |
| emotion_focus/masked_broad | 0.187844 ± 0.009760 | 0.187355 ± 0.003445 | 0.183643 ± 0.001450 |

## Mean-caption painting AMI (5,000 paintings, k=20)

| View | Emotion | Genre | Style |
|---|---:|---:|---:|
| default/unmasked | 0.107478 ± 0.006510 | 0.337556 ± 0.004927 | 0.120231 ± 0.003478 |
| default/masked_label | 0.103263 ± 0.005108 | 0.331112 ± 0.015470 | 0.118756 ± 0.004147 |
| default/masked_broad | 0.093914 ± 0.005816 | 0.337859 ± 0.006193 | 0.120183 ± 0.003261 |
| emotion_focus/unmasked | 0.217173 ± 0.003044 | 0.243924 ± 0.015294 | 0.100889 ± 0.003607 |
| emotion_focus/masked_label | 0.189384 ± 0.002985 | 0.252366 ± 0.006926 | 0.102677 ± 0.001289 |
| emotion_focus/masked_broad | 0.166416 ± 0.002518 | 0.282093 ± 0.008161 | 0.110254 ± 0.001925 |

## Caption emotion probe (subset captions, grouped by painting)

| View | Accuracy | Majority rate | Source |
|---|---:|---:|---|
| default/unmasked | 0.564189 ± 0.003204 | 0.283148 | copied from pav results.json: views[key].probes[caption/emotion] |
| default/masked_label | 0.527266 ± 0.003326 | 0.283148 | computed with pav_score.probe |
| default/masked_broad | 0.491216 ± 0.005052 | 0.283148 | computed with pav_score.probe |
| emotion_focus/unmasked | 0.586010 ± 0.004270 | 0.283148 | copied from pav results.json: views[key].probes[caption/emotion] |
| emotion_focus/masked_label | 0.546900 ± 0.003309 | 0.283148 | computed with pav_score.probe |
| emotion_focus/masked_broad | 0.513275 ± 0.004732 | 0.283148 | computed with pav_score.probe |

## No-word subgroups: unmasked caption emotion AMI

| Group | Key | n | k=9 | k=20 | k=40 |
|---|---|---:|---:|---:|---:|
| no_broad_word | default | 59413 | 0.055716 ± 0.003773 | 0.070158 ± 0.011681 | 0.084551 ± 0.001565 |
| no_broad_word | emotion_focus | 59413 | 0.203497 ± 0.007189 | 0.196916 ± 0.002960 | 0.191046 ± 0.000588 |
| no_label_word | default | 86804 | 0.086874 ± 0.009353 | 0.102332 ± 0.007904 | 0.104353 ± 0.002735 |
| no_label_word | emotion_focus | 86804 | 0.237538 ± 0.012683 | 0.226385 ± 0.003343 | 0.220045 ± 0.001289 |

## Unmasked consistency gates (tolerance 0.002 for mean and sd)

| Metric | Status | Current | Earlier |
|---|---|---:|---:|
| default/unmasked/caption_ami/9 | PASS | 0.098293 ± 0.008553 | 0.098293 ± 0.008553 |
| default/unmasked/caption_ami/20 | PASS | 0.121728 ± 0.006921 | 0.121728 ± 0.006921 |
| default/unmasked/caption_ami/40 | PASS | 0.128728 ± 0.004409 | 0.128728 ± 0.004409 |
| default/unmasked/mean_caption/emotion | PASS | 0.107478 ± 0.006510 | 0.107478 ± 0.006510 |
| default/unmasked/mean_caption/genre | PASS | 0.337556 ± 0.004927 | 0.337556 ± 0.004927 |
| default/unmasked/mean_caption/style | PASS | 0.120231 ± 0.003478 | 0.120231 ± 0.003478 |
| emotion_focus/unmasked/caption_ami/9 | PASS | 0.304819 ± 0.007292 | 0.304819 ± 0.007292 |
| emotion_focus/unmasked/caption_ami/20 | PASS | 0.278238 ± 0.003696 | 0.278238 ± 0.003696 |
| emotion_focus/unmasked/caption_ami/40 | PASS | 0.263681 ± 0.001109 | 0.263681 ± 0.001109 |
| emotion_focus/unmasked/mean_caption/emotion | PASS | 0.217173 ± 0.003044 | 0.217173 ± 0.003044 |
| emotion_focus/unmasked/mean_caption/genre | PASS | 0.243924 ± 0.015294 | 0.243924 ± 0.015294 |
| emotion_focus/unmasked/mean_caption/style | PASS | 0.100889 ± 0.003607 | 0.100889 ± 0.003607 |

## Availability

- default/unmasked: verified
- default/masked_label: verified
- default/masked_broad: verified
- emotion_focus/unmasked: verified
- emotion_focus/masked_label: verified
- emotion_focus/masked_broad: verified
