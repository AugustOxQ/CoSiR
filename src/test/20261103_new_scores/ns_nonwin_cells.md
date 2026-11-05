# r17: every claim cell that is not a WIN

Generated from `records/verdicts.json` (C13 passed). 412 cells; the others are WIN. d = candidate minus comparator; the
interval is the 95% paired bootstrap interval; r is the seed range of that cell. WIN: lower bound above 0. LEVEL: not WIN
and lower bound above -r. BELOW: otherwise. "Claim" is the sentence the cell licenses under the spec's section 6.

|Family|Set|Encoder|Aspect|Metric|K|Candidate|Comparator|d [95% interval]|r|Outcome|Bar|Claim|
|---|---|---|---|---|---:|---:|---:|---|---:|---|---|---|
|SC2|ArtELingo held|SigLIP 1|emotion|ACC|9|0.2479|0.2391 (matched PercepT)|+0.0088 [-0.0002, +0.0178]|0.0162|LEVEL|WIN|none|
|SC2|MMArt test|SigLIP 2|emotion|ACC|9|0.1994|0.1963 (matched PercepT)|+0.0031 [-0.0042, +0.0096]|0.0274|LEVEL|WIN|none|
|SC2|MMArt test|CLIP L/14|emotion|ACC|9|0.2009|0.1966 (matched PercepT)|+0.0043 [-0.0022, +0.0119]|0.0091|LEVEL|WIN|none|
|SC3|ArtELingo held|Qwen|emotion|ARI|9|0.0605|0.0726 (teacher)|-0.0121 [-0.0177, -0.0056]|0.0130|BELOW|LEVEL or WIN|none|
|SC3|ArtELingo held|Qwen|emotion|ACC|9|0.2525|0.2554 (teacher)|-0.0029 [-0.0125, +0.0064]|0.0204|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|Qwen|style|ACC|27|0.3246|0.3203 (teacher)|+0.0043 [-0.0047, +0.0121]|0.0089|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|Qwen|genre|ARI|10|0.4497|0.4474 (teacher)|+0.0023 [-0.0095, +0.0137]|0.0528|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|Qwen|genre|ACC|10|0.6102|0.6071 (teacher)|+0.0032 [-0.0073, +0.0128]|0.0346|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|SigLIP 2|emotion|ARI|9|0.0615|0.0726 (teacher)|-0.0111 [-0.0150, -0.0071]|0.0293|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|SigLIP 2|emotion|ACC|9|0.2474|0.2554 (teacher)|-0.0080 [-0.0149, -0.0003]|0.0493|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|SigLIP 1|emotion|ARI|9|0.0625|0.0726 (teacher)|-0.0101 [-0.0147, -0.0046]|0.0074|BELOW|LEVEL or WIN|none|
|SC3|ArtELingo held|SigLIP 1|emotion|ACC|9|0.2479|0.2554 (teacher)|-0.0075 [-0.0149, -0.0003]|0.0162|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|SigLIP 1|genre|ARI|10|0.4607|0.4474 (teacher)|+0.0134 [-0.0061, +0.0324]|0.0316|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|CLIP L/14|emotion|ARI|9|0.0582|0.0726 (teacher)|-0.0144 [-0.0182, -0.0108]|0.0253|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|CLIP L/14|emotion|ACC|9|0.2513|0.2554 (teacher)|-0.0040 [-0.0100, +0.0019]|0.0253|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|CLIP L/14|style|ACC|27|0.3255|0.3203 (teacher)|+0.0052 [-0.0025, +0.0126]|0.0237|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|CLIP L/14|genre|ARI|10|0.4303|0.4474 (teacher)|-0.0171 [-0.0244, -0.0099]|0.0384|LEVEL|LEVEL or WIN|level with or better than|
|SC3|ArtELingo held|CLIP L/14|genre|ACC|10|0.5980|0.6071 (teacher)|-0.0091 [-0.0150, -0.0031]|0.0480|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|Qwen|emotion|ARI|9|0.0311|0.0309 (teacher)|+0.0002 [-0.0021, +0.0024]|0.0028|LEVEL|WIN|level with or better than|
|SC3|MMArt test|Qwen|emotion|ACC|9|0.2060|0.2087 (teacher)|-0.0026 [-0.0087, +0.0032]|0.0184|LEVEL|WIN|level with or better than|
|SC3|MMArt test|Qwen|genre|AMI|10|0.5306|0.5256 (teacher)|+0.0050 [-0.0007, +0.0105]|0.0141|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|Qwen|genre|NMI|10|0.5312|0.5262 (teacher)|+0.0050 [-0.0007, +0.0105]|0.0140|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|Qwen|genre|ARI|10|0.4581|0.4600 (teacher)|-0.0019 [-0.0110, +0.0069]|0.0455|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|Qwen|genre|ACC|10|0.6230|0.6206 (teacher)|+0.0024 [-0.0125, +0.0178]|0.0582|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|SigLIP 2|emotion|ARI|9|0.0297|0.0309 (teacher)|-0.0012 [-0.0037, +0.0014]|0.0079|LEVEL|WIN|level with or better than|
|SC3|MMArt test|SigLIP 2|emotion|ACC|9|0.1994|0.2087 (teacher)|-0.0092 [-0.0153, -0.0031]|0.0274|LEVEL|WIN|level with or better than|
|SC3|MMArt test|SigLIP 2|genre|ARI|10|0.4732|0.4600 (teacher)|+0.0132 [-0.0033, +0.0294]|0.0236|LEVEL|LEVEL or WIN|level with or better than|
|SC3|MMArt test|SigLIP 1|emotion|ARI|9|0.0307|0.0309 (teacher)|-0.0002 [-0.0021, +0.0017]|0.0015|BELOW|WIN|none|
|SC3|MMArt test|SigLIP 1|emotion|ACC|9|0.2016|0.2087 (teacher)|-0.0070 [-0.0118, -0.0016]|0.0050|BELOW|WIN|none|
|SC3|MMArt test|CLIP L/14|emotion|ARI|9|0.0309|0.0309 (teacher)|+0.0000 [-0.0020, +0.0020]|0.0032|LEVEL|WIN|level with or better than|
|SC3|MMArt test|CLIP L/14|emotion|ACC|9|0.2009|0.2087 (teacher)|-0.0078 [-0.0137, -0.0018]|0.0091|BELOW|WIN|none|
|SC4|Affection test|Qwen|emotion|ARI|9|0.0888|0.0859 (teacher)|+0.0030 [-0.0039, +0.0092]|0.0176|LEVEL|LEVEL or WIN|level with or better than|
|SC4|Affection test|Qwen|emotion|ACC|9|0.2932|0.2845 (teacher)|+0.0087 [-0.0020, +0.0195]|0.0272|LEVEL|LEVEL or WIN|level with or better than|
|SC5|ArtELingo held|CLIP L/14|emotion|ARI|9|0.0582|0.0642 (CRL on CLIP L/14)|-0.0060 [-0.0101, -0.0020]|0.0253|LEVEL|WIN|none|
|SC5|ArtELingo held|CLIP L/14|emotion|ACC|9|0.2513|0.2506 (CRL on CLIP L/14)|+0.0007 [-0.0060, +0.0072]|0.0253|LEVEL|WIN|none|
|SC5|MMArt test|CLIP L/14|emotion|ACC|9|0.2009|0.1985 (CRL on CLIP L/14)|+0.0024 [-0.0029, +0.0085]|0.0091|LEVEL|WIN|none|
|SC5|MMArt test|CLIP L/14|style|ACC|20|0.3518|0.3542 (CRL on CLIP L/14)|-0.0023 [-0.0138, +0.0074]|0.0081|BELOW|WIN|none|
|SC7|ArtELingo held|Qwen|genre|mAP||0.5376|0.5361 (plain image)|+0.0015 [-0.0015, +0.0047]|0.0038|LEVEL|WIN|none|
|SC7|ArtELingo held|SigLIP 1|style|mAP||0.2877|0.2962 (plain image)|-0.0085 [-0.0101, -0.0069]|0.0047|BELOW|WIN|none|
