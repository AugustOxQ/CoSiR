# Prompted aspect views

CPU scores; missing views are omitted. AMI and accuracy: mean ± population sd.

## Caption emotion AMI

| View | k=9 | k=20 | k=40 |
|---|---:|---:|---:|
| clip_text | 0.03700 ± 0.00139 | 0.04666 ± 0.00162 | 0.05196 ± 0.00251 |
| goemotions | 0.19656 ± 0.00183 | 0.20930 ± 0.00080 | 0.19937 ± 0.00173 |
| default | 0.09829 ± 0.00855 | 0.12173 ± 0.00692 | 0.12873 ± 0.00441 |
| emotion_focus | 0.30482 ± 0.00729 | 0.27824 ± 0.00370 | 0.26368 ± 0.00111 |
| emotion_word | 0.29245 ± 0.00297 | 0.27380 ± 0.00263 | 0.26069 ± 0.00045 |
| emotion_neg | 0.24877 ± 0.01517 | 0.25437 ± 0.00068 | 0.24343 ± 0.00225 |
| genre_focus | 0.18392 ± 0.01029 | 0.18780 ± 0.01238 | 0.18491 ± 0.00340 |
| genre_word | 0.14867 ± 0.00307 | 0.16937 ± 0.00614 | 0.15916 ± 0.00271 |
| genre_neg | 0.18942 ± 0.01338 | 0.19792 ± 0.00362 | 0.19502 ± 0.00168 |
| style_focus | 0.18038 ± 0.00935 | 0.17598 ± 0.00333 | 0.17399 ± 0.00236 |
| style_word | 0.23613 ± 0.00886 | 0.21544 ± 0.00286 | 0.20900 ± 0.00192 |
| style_neg | 0.18853 ± 0.02795 | 0.18948 ± 0.00478 | 0.18326 ± 0.00075 |

## Shift: cosine to default

| View | Image | Caption |
|---|---:|---:|
| default | 1.0000000575468206 | 1.0000000579876718 |
| emotion_focus | 0.4301193786418839 | 0.6372683837221178 |
| emotion_word | 0.4104940228451589 | 0.6324640141225877 |
| emotion_neg | 0.6874870804243199 | 0.7358837412108866 |
| genre_focus | 0.7752700700071591 | 0.793300371047057 |
| genre_word | 0.4403524857849605 | 0.6305485910449637 |
| genre_neg | 0.7785251895987078 | 0.7889112777497795 |
| style_focus | 0.8177298088441902 | 0.7873009263928692 |
| style_word | 0.4602442124816652 | 0.64330590504396 |
| style_neg | 0.7530833843876442 | 0.7263432773509185 |

## Painting view × label AMI (k=20)

| View | Emotion | Genre | Style |
|---|---:|---:|---:|
| clip_text/mean_caption | 0.06672 ± 0.00334 | 0.33045 ± 0.01954 | 0.10992 ± 0.00169 |
| clip_image/image | 0.06474 ± 0.00221 | 0.41158 ± 0.00977 | 0.30001 ± 0.00274 |
| csd_style/image | 0.04403 ± 0.00065 | 0.25377 ± 0.00357 | 0.31717 ± 0.00307 |
| vgg_gram/image | 0.03111 ± 0.00111 | 0.15531 ± 0.00694 | 0.12415 ± 0.00190 |
| default/image | 0.06857 ± 0.00110 | 0.44861 ± 0.00424 | 0.22160 ± 0.00988 |
| default/mean_caption | 0.10748 ± 0.00651 | 0.33756 ± 0.00493 | 0.12023 ± 0.00348 |
| emotion_focus/image | 0.09288 ± 0.00174 | 0.40963 ± 0.00887 | 0.16782 ± 0.01024 |
| emotion_focus/mean_caption | 0.21717 ± 0.00304 | 0.24392 ± 0.01529 | 0.10089 ± 0.00361 |
| emotion_word/image | 0.09352 ± 0.00232 | 0.41428 ± 0.00273 | 0.18031 ± 0.00510 |
| emotion_word/mean_caption | 0.21146 ± 0.00542 | 0.24663 ± 0.00614 | 0.10583 ± 0.00105 |
| emotion_neg/image | 0.07660 ± 0.00094 | 0.44525 ± 0.00301 | 0.18756 ± 0.00466 |
| emotion_neg/mean_caption | 0.18482 ± 0.00276 | 0.29016 ± 0.00536 | 0.11725 ± 0.00183 |
| genre_focus/image | 0.06886 ± 0.00179 | 0.48183 ± 0.01131 | 0.25727 ± 0.00722 |
| genre_focus/mean_caption | 0.13917 ± 0.00447 | 0.33578 ± 0.00668 | 0.13015 ± 0.00685 |
| genre_word/image | 0.07011 ± 0.00046 | 0.49563 ± 0.00363 | 0.26439 ± 0.00395 |
| genre_word/mean_caption | 0.13802 ± 0.00220 | 0.35310 ± 0.00494 | 0.13967 ± 0.00149 |
| genre_neg/image | 0.06897 ± 0.00131 | 0.47421 ± 0.00745 | 0.25122 ± 0.00746 |
| genre_neg/mean_caption | 0.14129 ± 0.00552 | 0.34435 ± 0.00456 | 0.13297 ± 0.00220 |
| style_focus/image | 0.06226 ± 0.00054 | 0.41029 ± 0.01256 | 0.35561 ± 0.01382 |
| style_focus/mean_caption | 0.14276 ± 0.00579 | 0.33873 ± 0.00655 | 0.13354 ± 0.00360 |
| style_word/image | 0.06300 ± 0.00114 | 0.34896 ± 0.01351 | 0.39515 ± 0.00281 |
| style_word/mean_caption | 0.17753 ± 0.00349 | 0.30983 ± 0.00651 | 0.13549 ± 0.00252 |
| style_neg/image | 0.06244 ± 0.00159 | 0.40527 ± 0.00462 | 0.35648 ± 0.00135 |
| style_neg/mean_caption | 0.13786 ± 0.00168 | 0.33239 ± 0.00992 | 0.13036 ± 0.00082 |

## Probes

| View | Task | Accuracy | Majority rate |
|---|---|---:|---:|
| clip_text | mean_caption/emotion | 0.52140 ± 0.01198 | 0.35160 |
| clip_text | mean_caption/genre | 0.66518 ± 0.01731 | 0.21786 |
| clip_text | mean_caption/style | 0.27120 ± 0.01332 | 0.15120 |
| clip_text | caption/emotion | 0.55147 ± 0.00854 | 0.28315 |
| clip_image | image/emotion | 0.39160 ± 0.00948 | 0.35160 |
| clip_image | image/genre | 0.76289 ± 0.00998 | 0.21786 |
| clip_image | image/style | 0.54560 ± 0.01261 | 0.15120 |
| csd_style | image/emotion | 0.39540 ± 0.01351 | 0.35160 |
| csd_style | image/genre | 0.78633 ± 0.01018 | 0.21786 |
| csd_style | image/style | 0.61960 ± 0.02053 | 0.15120 |
| vgg_gram | image/emotion | 0.33000 ± 0.01468 | 0.35160 |
| vgg_gram | image/genre | 0.62250 ± 0.01078 | 0.21786 |
| vgg_gram | image/style | 0.42920 ± 0.00870 | 0.15120 |
| goemotions | caption/emotion | 0.51578 ± 0.00660 | 0.28315 |
| default | image/emotion | 0.39160 ± 0.01643 | 0.35160 |
| default | image/genre | 0.80903 ± 0.01916 | 0.21786 |
| default | image/style | 0.57800 ± 0.00593 | 0.15120 |
| default | mean_caption/emotion | 0.53480 ± 0.01504 | 0.35160 |
| default | mean_caption/genre | 0.66617 ± 0.01477 | 0.21786 |
| default | mean_caption/style | 0.28040 ± 0.00880 | 0.15120 |
| default | caption/emotion | 0.56419 ± 0.00320 | 0.28315 |
| emotion_focus | image/emotion | 0.40920 ± 0.01348 | 0.35160 |
| emotion_focus | image/genre | 0.79497 ± 0.01101 | 0.21786 |
| emotion_focus | image/style | 0.53820 ± 0.01477 | 0.15120 |
| emotion_focus | mean_caption/emotion | 0.56080 ± 0.00870 | 0.35160 |
| emotion_focus | mean_caption/genre | 0.65259 ± 0.03229 | 0.21786 |
| emotion_focus | mean_caption/style | 0.26720 ± 0.00634 | 0.15120 |
| emotion_focus | caption/emotion | 0.58601 ± 0.00427 | 0.28315 |
| emotion_word | image/emotion | 0.40440 ± 0.00916 | 0.35160 |
| emotion_word | image/genre | 0.79817 ± 0.01201 | 0.21786 |
| emotion_word | image/style | 0.54180 ± 0.00882 | 0.15120 |
| emotion_word | mean_caption/emotion | 0.56100 ± 0.01761 | 0.35160 |
| emotion_word | mean_caption/genre | 0.65259 ± 0.02723 | 0.21786 |
| emotion_word | mean_caption/style | 0.26440 ± 0.01378 | 0.15120 |
| emotion_word | caption/emotion | 0.58581 ± 0.00481 | 0.28315 |
| emotion_neg | image/emotion | 0.39880 ± 0.01317 | 0.35160 |
| emotion_neg | image/genre | 0.79645 ± 0.02071 | 0.21786 |
| emotion_neg | image/style | 0.55880 ± 0.01406 | 0.15120 |
| emotion_neg | mean_caption/emotion | 0.55340 ± 0.00873 | 0.35160 |
| emotion_neg | mean_caption/genre | 0.65654 ± 0.01956 | 0.21786 |
| emotion_neg | mean_caption/style | 0.27040 ± 0.01326 | 0.15120 |
| emotion_neg | caption/emotion | 0.58104 ± 0.00512 | 0.28315 |
| genre_focus | image/emotion | 0.40240 ± 0.00206 | 0.35160 |
| genre_focus | image/genre | 0.81174 ± 0.01409 | 0.21786 |
| genre_focus | image/style | 0.60080 ± 0.01564 | 0.15120 |
| genre_focus | mean_caption/emotion | 0.54480 ± 0.00943 | 0.35160 |
| genre_focus | mean_caption/genre | 0.67036 ± 0.01938 | 0.21786 |
| genre_focus | mean_caption/style | 0.27220 ± 0.02156 | 0.15120 |
| genre_focus | caption/emotion | 0.56665 ± 0.00872 | 0.28315 |
| genre_word | image/emotion | 0.39760 ± 0.00546 | 0.35160 |
| genre_word | image/genre | 0.82236 ± 0.00705 | 0.21786 |
| genre_word | image/style | 0.60320 ± 0.01408 | 0.15120 |
| genre_word | mean_caption/emotion | 0.54800 ± 0.00846 | 0.35160 |
| genre_word | mean_caption/genre | 0.66938 ± 0.01505 | 0.21786 |
| genre_word | mean_caption/style | 0.27600 ± 0.01037 | 0.15120 |
| genre_word | caption/emotion | 0.55823 ± 0.00665 | 0.28315 |
| genre_neg | image/emotion | 0.38580 ± 0.00935 | 0.35160 |
| genre_neg | image/genre | 0.80508 ± 0.01072 | 0.21786 |
| genre_neg | image/style | 0.59460 ± 0.01460 | 0.15120 |
| genre_neg | mean_caption/emotion | 0.54220 ± 0.00471 | 0.35160 |
| genre_neg | mean_caption/genre | 0.66962 ± 0.01595 | 0.21786 |
| genre_neg | mean_caption/style | 0.27400 ± 0.01527 | 0.15120 |
| genre_neg | caption/emotion | 0.56638 ± 0.00856 | 0.28315 |
| style_focus | image/emotion | 0.39900 ± 0.01016 | 0.35160 |
| style_focus | image/genre | 0.81076 ± 0.01063 | 0.21786 |
| style_focus | image/style | 0.61440 ± 0.00882 | 0.15120 |
| style_focus | mean_caption/emotion | 0.53300 ± 0.00363 | 0.35160 |
| style_focus | mean_caption/genre | 0.67678 ± 0.02032 | 0.21786 |
| style_focus | mean_caption/style | 0.27640 ± 0.02173 | 0.15120 |
| style_focus | caption/emotion | 0.56407 ± 0.00524 | 0.28315 |
| style_word | image/emotion | 0.39700 ± 0.01122 | 0.35160 |
| style_word | image/genre | 0.80779 ± 0.00833 | 0.21786 |
| style_word | image/style | 0.62920 ± 0.00880 | 0.15120 |
| style_word | mean_caption/emotion | 0.56300 ± 0.01301 | 0.35160 |
| style_word | mean_caption/genre | 0.66690 ± 0.02317 | 0.21786 |
| style_word | mean_caption/style | 0.28240 ± 0.01250 | 0.15120 |
| style_word | caption/emotion | 0.57083 ± 0.00447 | 0.28315 |
| style_neg | image/emotion | 0.39280 ± 0.01102 | 0.35160 |
| style_neg | image/genre | 0.80829 ± 0.01440 | 0.21786 |
| style_neg | image/style | 0.60820 ± 0.01617 | 0.15120 |
| style_neg | mean_caption/emotion | 0.53720 ± 0.01468 | 0.35160 |
| style_neg | mean_caption/genre | 0.67282 ± 0.02087 | 0.21786 |
| style_neg | mean_caption/style | 0.27900 ± 0.01691 | 0.15120 |
| style_neg | caption/emotion | 0.56049 ± 0.00656 | 0.28315 |

## image partition AMI (seed 0)

| View | clip_image | csd_style | vgg_gram | default | emotion_focus | emotion_word | emotion_neg | genre_focus | genre_word | genre_neg | style_focus | style_word | style_neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_image | 1.0000 | 0.3399 | 0.1898 | 0.4515 | 0.3751 | 0.4026 | 0.4142 | 0.4804 | 0.4749 | 0.4761 | 0.4883 | 0.4327 | 0.4981 |
| csd_style | 0.3399 | 1.0000 | 0.2666 | 0.2432 | 0.1819 | 0.2005 | 0.2050 | 0.2671 | 0.3045 | 0.2711 | 0.3891 | 0.4356 | 0.4332 |
| vgg_gram | 0.1898 | 0.2666 | 1.0000 | 0.1546 | 0.1231 | 0.1362 | 0.1349 | 0.1628 | 0.1895 | 0.1581 | 0.2141 | 0.2157 | 0.2352 |
| default | 0.4515 | 0.2432 | 0.1546 | 1.0000 | 0.4657 | 0.4828 | 0.5718 | 0.6395 | 0.5171 | 0.5919 | 0.4767 | 0.3251 | 0.4244 |
| emotion_focus | 0.3751 | 0.1819 | 0.1231 | 0.4657 | 1.0000 | 0.5872 | 0.5208 | 0.4502 | 0.4172 | 0.4442 | 0.3558 | 0.2694 | 0.3386 |
| emotion_word | 0.4026 | 0.2005 | 0.1362 | 0.4828 | 0.5872 | 1.0000 | 0.5453 | 0.4759 | 0.4506 | 0.4726 | 0.3845 | 0.2946 | 0.3614 |
| emotion_neg | 0.4142 | 0.2050 | 0.1349 | 0.5718 | 0.5208 | 0.5453 | 1.0000 | 0.5487 | 0.4870 | 0.5408 | 0.4180 | 0.2972 | 0.3782 |
| genre_focus | 0.4804 | 0.2671 | 0.1628 | 0.6395 | 0.4502 | 0.4759 | 0.5487 | 1.0000 | 0.5831 | 0.6213 | 0.5253 | 0.3866 | 0.4762 |
| genre_word | 0.4749 | 0.3045 | 0.1895 | 0.5171 | 0.4172 | 0.4506 | 0.4870 | 0.5831 | 1.0000 | 0.5936 | 0.5299 | 0.4423 | 0.4968 |
| genre_neg | 0.4761 | 0.2711 | 0.1581 | 0.5919 | 0.4442 | 0.4726 | 0.5408 | 0.6213 | 0.5936 | 1.0000 | 0.5203 | 0.3804 | 0.4785 |
| style_focus | 0.4883 | 0.3891 | 0.2141 | 0.4767 | 0.3558 | 0.3845 | 0.4180 | 0.5253 | 0.5299 | 0.5203 | 1.0000 | 0.5359 | 0.6033 |
| style_word | 0.4327 | 0.4356 | 0.2157 | 0.3251 | 0.2694 | 0.2946 | 0.2972 | 0.3866 | 0.4423 | 0.3804 | 0.5359 | 1.0000 | 0.5567 |
| style_neg | 0.4981 | 0.4332 | 0.2352 | 0.4244 | 0.3386 | 0.3614 | 0.3782 | 0.4762 | 0.4968 | 0.4785 | 0.6033 | 0.5567 | 1.0000 |

| View | Seed 0 vs 1 |
|---|---:|
| clip_image | 0.6218 |
| csd_style | 0.5455 |
| vgg_gram | 0.5689 |
| default | 0.6077 |
| emotion_focus | 0.6118 |
| emotion_word | 0.6177 |
| emotion_neg | 0.6294 |
| genre_focus | 0.6549 |
| genre_word | 0.6941 |
| genre_neg | 0.6199 |
| style_focus | 0.6550 |
| style_word | 0.6679 |
| style_neg | 0.6855 |

## mean_caption partition AMI (seed 0)

| View | clip_text | default | emotion_focus | emotion_word | emotion_neg | genre_focus | genre_word | genre_neg | style_focus | style_word | style_neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_text | 1.0000 | 0.5359 | 0.2506 | 0.2848 | 0.3861 | 0.4791 | 0.4264 | 0.4729 | 0.4489 | 0.3663 | 0.4735 |
| default | 0.5359 | 1.0000 | 0.3443 | 0.3795 | 0.4840 | 0.6055 | 0.5053 | 0.5736 | 0.5355 | 0.4592 | 0.5828 |
| emotion_focus | 0.2506 | 0.3443 | 1.0000 | 0.5393 | 0.5315 | 0.3928 | 0.3803 | 0.3890 | 0.3918 | 0.4489 | 0.3868 |
| emotion_word | 0.2848 | 0.3795 | 0.5393 | 1.0000 | 0.5313 | 0.4228 | 0.4329 | 0.4235 | 0.4403 | 0.5178 | 0.4310 |
| emotion_neg | 0.3861 | 0.4840 | 0.5315 | 0.5313 | 1.0000 | 0.5163 | 0.4818 | 0.5118 | 0.5197 | 0.5197 | 0.5138 |
| genre_focus | 0.4791 | 0.6055 | 0.3928 | 0.4228 | 0.5163 | 1.0000 | 0.5377 | 0.5996 | 0.5800 | 0.5075 | 0.6176 |
| genre_word | 0.4264 | 0.5053 | 0.3803 | 0.4329 | 0.4818 | 0.5377 | 1.0000 | 0.5876 | 0.5559 | 0.5424 | 0.5419 |
| genre_neg | 0.4729 | 0.5736 | 0.3890 | 0.4235 | 0.5118 | 0.5996 | 0.5876 | 1.0000 | 0.5636 | 0.5261 | 0.5917 |
| style_focus | 0.4489 | 0.5355 | 0.3918 | 0.4403 | 0.5197 | 0.5800 | 0.5559 | 0.5636 | 1.0000 | 0.5299 | 0.5741 |
| style_word | 0.3663 | 0.4592 | 0.4489 | 0.5178 | 0.5197 | 0.5075 | 0.5424 | 0.5261 | 0.5299 | 1.0000 | 0.5359 |
| style_neg | 0.4735 | 0.5828 | 0.3868 | 0.4310 | 0.5138 | 0.6176 | 0.5419 | 0.5917 | 0.5741 | 0.5359 | 1.0000 |

| View | Seed 0 vs 1 |
|---|---:|
| clip_text | 0.6116 |
| default | 0.6296 |
| emotion_focus | 0.5962 |
| emotion_word | 0.5964 |
| emotion_neg | 0.5808 |
| genre_focus | 0.5970 |
| genre_word | 0.6253 |
| genre_neg | 0.6341 |
| style_focus | 0.5880 |
| style_word | 0.5914 |
| style_neg | 0.6537 |

## Buddy graphs (mutual cosine top-20)

| Key | Graph | Pairs | Emotion lift | Genre lift | Style lift |
|---|---|---:|---:|---:|---:|
| clip_image | image | 23334 | 1.4670 | 5.0024 | 5.5329 |
| clip_image | mean_caption | 12419 | 1.8051 | 4.1533 | 2.3280 |
| clip_image | strict | 1049 | 1.9751 | 5.6923 | 6.4927 |
| clip_image | union | 34704 | 1.5727 | 4.6868 | 4.3570 |
| default | image | 27702 | 1.5536 | 5.0647 | 4.6026 |
| default | mean_caption | 19391 | 2.0642 | 4.1720 | 2.4056 |
| default | strict | 2984 | 2.1667 | 5.4028 | 4.8802 |
| default | union | 44109 | 1.7366 | 4.6637 | 3.6180 |
| emotion_focus | image | 25488 | 1.6367 | 4.9364 | 3.6928 |
| emotion_focus | mean_caption | 23289 | 2.3658 | 3.6771 | 2.1026 |
| emotion_focus | strict | 1790 | 2.2985 | 5.2368 | 4.3515 |
| emotion_focus | union | 46987 | 1.9729 | 4.3204 | 2.8796 |
| emotion_word | image | 25869 | 1.6514 | 4.9467 | 3.6515 |
| emotion_word | mean_caption | 23693 | 2.3550 | 3.7887 | 2.1361 |
| emotion_word | strict | 2058 | 2.3276 | 5.2972 | 4.3273 |
| emotion_word | union | 47504 | 1.9731 | 4.3686 | 2.8664 |
| emotion_neg | image | 24999 | 1.6293 | 5.0495 | 3.9332 |
| emotion_neg | mean_caption | 22369 | 2.2720 | 4.0049 | 2.3298 |
| emotion_neg | strict | 2418 | 2.2667 | 5.2765 | 4.6427 |
| emotion_neg | union | 44950 | 1.9148 | 4.5314 | 3.0971 |
| genre_focus | image | 26695 | 1.5818 | 5.1583 | 5.1676 |
| genre_focus | mean_caption | 21103 | 2.1743 | 4.3025 | 2.4613 |
| genre_focus | strict | 2839 | 2.2532 | 5.5384 | 5.4917 |
| genre_focus | union | 44959 | 1.8175 | 4.7423 | 3.8768 |
| genre_word | image | 24273 | 1.5778 | 5.2561 | 5.4083 |
| genre_word | mean_caption | 22402 | 2.1645 | 4.2551 | 2.4182 |
| genre_word | strict | 2236 | 2.2102 | 5.5930 | 5.6657 |
| genre_word | union | 44439 | 1.8417 | 4.7383 | 3.8880 |
| genre_neg | image | 25364 | 1.5796 | 5.1986 | 5.1104 |
| genre_neg | mean_caption | 20803 | 2.2087 | 4.2943 | 2.4311 |
| genre_neg | strict | 2579 | 2.2315 | 5.6265 | 5.3985 |
| genre_neg | union | 43588 | 1.8413 | 4.7523 | 3.8146 |
| style_focus | image | 27014 | 1.5170 | 5.0776 | 6.2413 |
| style_focus | mean_caption | 21235 | 2.1446 | 4.2836 | 2.5080 |
| style_focus | strict | 2520 | 2.1885 | 5.6484 | 6.5851 |
| style_focus | union | 45729 | 1.7715 | 4.6856 | 4.4888 |
| style_word | image | 26921 | 1.5328 | 4.8759 | 6.6086 |
| style_word | mean_caption | 23234 | 2.2774 | 4.0940 | 2.4142 |
| style_word | strict | 1604 | 2.2566 | 5.5889 | 7.5618 |
| style_word | union | 48551 | 1.8652 | 4.4823 | 4.5699 |
| style_neg | image | 27313 | 1.5126 | 5.0305 | 6.1124 |
| style_neg | mean_caption | 20673 | 2.1620 | 4.2630 | 2.4834 |
| style_neg | strict | 2146 | 2.1797 | 5.6767 | 6.5930 |
| style_neg | union | 45840 | 1.7742 | 4.6597 | 4.4533 |
