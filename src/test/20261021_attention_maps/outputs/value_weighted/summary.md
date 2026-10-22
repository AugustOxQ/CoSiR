# Value-weighted attention against raw attention

Image only, 304 development paintings (random, seed 0, plus the sheet paintings). Cells: median [10th, 90th percentile] over paintings. raw = head-mean attention (the stored maps); w = ||sum_h a_h W_O^h v_h|| (Kobayashi et al. 2020); w2 = sum_h a_h ||W_O^h v_h||.

## layers 16-27

| View | First-token share raw | w | Image share raw | w | Moran raw | w | w2 | shuffled w | Moran w > raw | Spearman raw-w | Top-5% mass raw | w | Border-hottest raw | w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.623 [0.607, 0.638] | 0.122 [0.112, 0.131] | 0.216 [0.202, 0.240] | 0.638 [0.610, 0.670] | 0.134 [0.061, 0.220] | 0.111 [0.049, 0.194] | 0.113 [0.049, 0.202] | -0.010 [-0.058, 0.043] | 0.28 | 0.957 [0.914, 0.975] | 0.197 [0.152, 0.251] | 0.210 [0.166, 0.264] | 0.25 | 0.15 |
| emotion_focus | 0.647 [0.629, 0.662] | 0.130 [0.118, 0.141] | 0.123 [0.104, 0.146] | 0.371 [0.308, 0.424] | 0.064 [0.006, 0.178] | 0.094 [0.025, 0.220] | 0.085 [0.018, 0.206] | -0.005 [-0.043, 0.054] | 0.83 | 0.965 [0.942, 0.976] | 0.385 [0.287, 0.471] | 0.348 [0.257, 0.427] | 0.60 | 0.41 |
| style_word | 0.669 [0.657, 0.685] | 0.127 [0.118, 0.137] | 0.121 [0.103, 0.136] | 0.369 [0.321, 0.420] | 0.088 [0.034, 0.159] | 0.066 [0.011, 0.141] | 0.067 [0.009, 0.140] | -0.009 [-0.040, 0.043] | 0.14 | 0.967 [0.948, 0.977] | 0.372 [0.275, 0.478] | 0.407 [0.306, 0.542] | 0.47 | 0.31 |
| genre_word | 0.674 [0.656, 0.688] | 0.116 [0.106, 0.126] | 0.119 [0.103, 0.137] | 0.359 [0.316, 0.407] | 0.082 [0.026, 0.162] | 0.064 [0.010, 0.136] | 0.064 [0.014, 0.139] | -0.009 [-0.046, 0.039] | 0.20 | 0.971 [0.957, 0.978] | 0.401 [0.296, 0.500] | 0.429 [0.322, 0.542] | 0.40 | 0.23 |

L1 to the default map (rounding floor 0.09 in the main report):

| View | raw | w | w2 | w, shuffled |
|---|---|---|---|---|
| emotion_focus | 0.676 [0.571, 0.800] | 0.654 [0.551, 0.774] | 0.672 [0.565, 0.793] | 1.179 [0.951, 1.364] |
| style_word | 0.788 [0.658, 0.914] | 0.871 [0.732, 1.031] | 0.875 [0.731, 1.046] | 1.269 [1.007, 1.435] |
| genre_word | 0.803 [0.677, 0.924] | 0.885 [0.749, 1.027] | 0.881 [0.746, 1.027] | 1.307 [1.078, 1.475] |

## last layer

| View | First-token share raw | w | Image share raw | w | Moran raw | w | w2 | shuffled w | Moran w > raw | Spearman raw-w | Top-5% mass raw | w | Border-hottest raw | w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.675 [0.645, 0.699] | 0.241 [0.225, 0.259] | 0.074 [0.054, 0.093] | 0.337 [0.262, 0.412] | 0.093 [0.024, 0.177] | 0.078 [0.012, 0.174] | 0.082 [0.014, 0.172] | -0.007 [-0.054, 0.041] | 0.33 | 0.980 [0.963, 0.987] | 0.331 [0.236, 0.407] | 0.320 [0.230, 0.395] | 0.60 | 0.51 |
| emotion_focus | 0.634 [0.610, 0.656] | 0.196 [0.180, 0.213] | 0.035 [0.018, 0.056] | 0.135 [0.071, 0.222] | 0.075 [0.013, 0.177] | 0.071 [0.006, 0.159] | 0.074 [0.009, 0.163] | -0.014 [-0.047, 0.041] | 0.34 | 0.982 [0.975, 0.987] | 0.488 [0.356, 0.604] | 0.477 [0.341, 0.584] | 0.67 | 0.54 |
| style_word | 0.656 [0.641, 0.675] | 0.181 [0.166, 0.197] | 0.037 [0.022, 0.057] | 0.134 [0.081, 0.206] | 0.086 [0.020, 0.174] | 0.073 [0.005, 0.149] | 0.077 [0.009, 0.159] | -0.008 [-0.050, 0.044] | 0.25 | 0.980 [0.971, 0.986] | 0.430 [0.310, 0.520] | 0.452 [0.316, 0.540] | 0.63 | 0.56 |
| genre_word | 0.657 [0.635, 0.673] | 0.157 [0.146, 0.173] | 0.016 [0.009, 0.037] | 0.051 [0.027, 0.127] | 0.074 [0.011, 0.167] | 0.056 [-0.003, 0.143] | 0.061 [0.004, 0.152] | -0.011 [-0.043, 0.052] | 0.21 | 0.970 [0.956, 0.980] | 0.411 [0.299, 0.525] | 0.464 [0.348, 0.588] | 0.59 | 0.58 |

L1 to the default map (rounding floor 0.09 in the main report):

| View | raw | w | w2 | w, shuffled |
|---|---|---|---|---|
| emotion_focus | 0.512 [0.404, 0.678] | 0.523 [0.423, 0.685] | 0.492 [0.394, 0.629] | 1.455 [1.211, 1.597] |
| style_word | 0.469 [0.364, 0.563] | 0.512 [0.398, 0.624] | 0.477 [0.369, 0.581] | 1.442 [1.179, 1.563] |
| genre_word | 0.545 [0.401, 0.711] | 0.626 [0.461, 0.866] | 0.555 [0.409, 0.744] | 1.404 [1.221, 1.550] |

## all layers

| View | First-token share raw | w | Image share raw | w | Moran raw | w | w2 | shuffled w | Moran w > raw | Spearman raw-w | Top-5% mass raw | w | Border-hottest raw | w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.639 [0.632, 0.646] | 0.134 [0.125, 0.143] | 0.141 [0.133, 0.148] | 0.610 [0.583, 0.639] | 0.231 [0.175, 0.294] | 0.112 [0.051, 0.198] | 0.114 [0.051, 0.202] | -0.008 [-0.052, 0.044] | 0.01 | 0.847 [0.811, 0.875] | 0.211 [0.182, 0.241] | 0.207 [0.165, 0.259] | 0.99 | 0.15 |
| emotion_focus | 0.642 [0.634, 0.648] | 0.142 [0.129, 0.152] | 0.095 [0.081, 0.107] | 0.356 [0.296, 0.406] | 0.207 [0.124, 0.280] | 0.096 [0.028, 0.223] | 0.087 [0.020, 0.207] | -0.009 [-0.050, 0.043] | 0.15 | 0.803 [0.754, 0.848] | 0.316 [0.261, 0.363] | 0.339 [0.251, 0.415] | 0.84 | 0.42 |
| style_word | 0.653 [0.647, 0.660] | 0.138 [0.129, 0.149] | 0.097 [0.082, 0.105] | 0.356 [0.310, 0.405] | 0.223 [0.140, 0.300] | 0.068 [0.013, 0.144] | 0.069 [0.011, 0.144] | -0.010 [-0.049, 0.045] | 0.00 | 0.830 [0.794, 0.866] | 0.316 [0.252, 0.370] | 0.396 [0.297, 0.526] | 0.73 | 0.31 |
| genre_word | 0.655 [0.647, 0.661] | 0.128 [0.117, 0.137] | 0.096 [0.084, 0.106] | 0.347 [0.305, 0.393] | 0.202 [0.137, 0.266] | 0.068 [0.011, 0.138] | 0.068 [0.014, 0.140] | -0.007 [-0.042, 0.052] | 0.01 | 0.831 [0.792, 0.863] | 0.319 [0.250, 0.367] | 0.418 [0.315, 0.526] | 0.77 | 0.23 |

L1 to the default map (rounding floor 0.09 in the main report):

| View | raw | w | w2 | w, shuffled |
|---|---|---|---|---|
| emotion_focus | 0.427 [0.359, 0.508] | 0.633 [0.533, 0.753] | 0.649 [0.547, 0.770] | 1.155 [0.938, 1.333] |
| style_word | 0.501 [0.429, 0.575] | 0.845 [0.713, 1.000] | 0.848 [0.709, 1.016] | 1.241 [0.984, 1.392] |
| genre_word | 0.492 [0.419, 0.564] | 0.858 [0.725, 0.995] | 0.852 [0.721, 0.992] | 1.275 [1.047, 1.443] |

## First token (<|im_start|>)

value_size_ratio: mean over heads of ||W_O^h v|| of token 0 / median over image tokens, per attention layer 0..27; maxabs/norm ratio: token 0 / median over the other tokens, hidden outputs of layers 0..26. Medians over paintings.

- default: value size ratio, layers 0/4/8/16/20/24/27: 1.638, 0.153, 0.180, 0.246, 0.117, 0.052, 0.071; median over layers 16-27 0.067. Max-abs ratio, hidden 0/2/4/8/16/26: 0.4, 353.5, 275.0, 291.3, 92.4, 18.9. Norm ratio, same layers: 0.6, 219.0, 197.6, 200.4, 73.0, 10.1.
- emotion_focus: value size ratio, layers 0/4/8/16/20/24/27: 1.638, 0.153, 0.181, 0.246, 0.117, 0.051, 0.067; median over layers 16-27 0.065. Max-abs ratio, hidden 0/2/4/8/16/26: 0.4, 365.3, 285.7, 301.7, 96.6, 21.3. Norm ratio, same layers: 0.6, 223.7, 202.9, 206.5, 75.2, 10.7.
- style_word: value size ratio, layers 0/4/8/16/20/24/27: 1.638, 0.154, 0.182, 0.247, 0.118, 0.052, 0.069; median over layers 16-27 0.066. Max-abs ratio, hidden 0/2/4/8/16/26: 0.4, 357.3, 277.6, 294.2, 96.1, 20.4. Norm ratio, same layers: 0.6, 220.4, 198.9, 202.5, 75.3, 10.6.
- genre_word: value size ratio, layers 0/4/8/16/20/24/27: 1.638, 0.154, 0.183, 0.248, 0.119, 0.052, 0.067; median over layers 16-27 0.066. Max-abs ratio, hidden 0/2/4/8/16/26: 0.4, 356.2, 276.3, 294.2, 96.6, 21.3. Norm ratio, same layers: 0.6, 219.9, 198.7, 202.5, 76.0, 10.8.

## Captions only (25160 captions, layers 16-27, tokens with >= 50 occurrences)

### default
- first-token share raw 0.654 [0.635, 0.670], w 0.156 [0.139, 0.174]; caption share raw 0.135 [0.101, 0.171], w 0.451 [0.356, 0.536]
- Spearman of token means raw vs w: 0.968; top-20 overlap 0.85; eligible tokens 834
- top raw: horses 0.236, statue 0.228, beach 0.221, vase 0.219, woods 0.217, chair 0.215, horse 0.212, garden 0.209, sunset 0.208, flowers 0.205, nude 0.204, church 0.204
- top w: horses 0.286, statue 0.27, chair 0.256, ships 0.256, sketch 0.256, horse 0.255, beach 0.254, lake 0.248, boats 0.246, boat 0.244, rainbow 0.241, angels 0.24

### emotion_focus
- first-token share raw 0.679 [0.660, 0.694], w 0.150 [0.136, 0.162]; caption share raw 0.077 [0.059, 0.102], w 0.241 [0.187, 0.311]
- Spearman of token means raw vs w: 0.979; top-20 overlap 0.85; eligible tokens 834
- top raw: bored 0.404, anger 0.348, lonely 0.335, amused 0.331, angry 0.324, sadness 0.322, excited 0.319, awe 0.315, boring 0.314, fearful 0.312, confused 0.306, comforting 0.301
- rise over default, raw: bored +0.266, awe +0.247, amused +0.247, excited +0.231, fearful +0.211, confused +0.205, anger +0.202, excitement +0.197, proud +0.197, lonely +0.197, amusing +0.196, boring +0.195
- top w: bored 0.454, awe 0.401, amused 0.386, excited 0.373, confused 0.364, proud 0.356, comforting 0.356, anger 0.356, laugh 0.354, excitement 0.354, fearful 0.352, amusing 0.345
- rise over default, w: awe +0.317, bored +0.302, amused +0.287, excited +0.268, amusing +0.266, excitement +0.243, comforting +0.242, exciting +0.241, confused +0.241, fearful +0.235, proud +0.234, curious +0.231

### style_word
- first-token share raw 0.706 [0.692, 0.717], w 0.169 [0.158, 0.179]; caption share raw 0.058 [0.042, 0.080], w 0.170 [0.116, 0.234]
- Spearman of token means raw vs w: 0.982; top-20 overlap 0.75; eligible tokens 834
- top raw: cartoon 0.258, abstract 0.256, rainbow 0.245, sketch 0.243, woods 0.225, nude 0.223, angels 0.222, portrait 0.22, distorted 0.219, beach 0.217, garden 0.214, flowers 0.212
- rise over default, raw: abstract +0.131, cartoon +0.112, realistic +0.107, cute +0.092, distorted +0.089, ical +0.083, disturbing +0.081, creepy +0.081, religious +0.08, funny +0.073, evil +0.07, scary +0.066
- top w: sketch 0.317, rainbow 0.302, cartoon 0.301, woods 0.298, cat 0.291, beach 0.283, dog 0.281, horses 0.279, garden 0.275, abstract 0.262, angels 0.256, forest 0.255
- rise over default, w: cartoon +0.129, abstract +0.117, distorted +0.112, realistic +0.102, cute +0.087, creepy +0.083, disturbing +0.083, ical +0.081, . +0.074, funny +0.073, oding +0.068, evil +0.067

### genre_word
- first-token share raw 0.700 [0.684, 0.714], w 0.158 [0.146, 0.169]; caption share raw 0.056 [0.040, 0.076], w 0.172 [0.122, 0.233]
- Spearman of token means raw vs w: 0.981; top-20 overlap 0.75; eligible tokens 834
- top raw: portrait 0.32, abstract 0.294, statue 0.274, garden 0.271, beach 0.27, horses 0.268, sketch 0.267, flowers 0.261, nude 0.258, landscape 0.256, angels 0.253, cartoon 0.252
- rise over default, raw: abstract +0.169, portrait +0.136, adventure +0.115, religious +0.114, cartoon +0.107, landscape +0.092, battle +0.087, winter +0.08, war +0.078, fruit +0.074, scenery +0.074, party +0.073
- top w: sketch 0.363, horses 0.351, portrait 0.342, statue 0.338, beach 0.332, garden 0.325, cat 0.316, angels 0.311, dog 0.311, abstract 0.309, rainbow 0.309, woods 0.306
- rise over default, w: abstract +0.165, portrait +0.144, adventure +0.118, cartoon +0.115, sketch +0.107, winter +0.09, dream +0.09, garden +0.089, battle +0.089, park +0.089, party +0.088, landscape +0.088

