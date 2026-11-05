# r19 full tables

Rendered by `pp_report_tables.py` from `records/outcomes.json`. Qwen = Qwen3-VL-Embedding-2B.

## Verdicts

|Encoder|ArtELingo held|Affection test|MMArt test|SemArt new artists|Cost|
|---|---|---|---|---|---|
|Qwen|not in this round|not in this round|not in this round|holds|holds|
|SigLIP 2|not in this round|not in this round|not in this round|holds|holds|
|SigLIP 1|not in this round|not in this round|not in this round|holds|holds|
|CLIP L/14|not in this round|not in this round|not in this round|holds|holds|
|DINOv2|holds|holds|holds|holds|holds|
|CLIP B/32|holds|holds|holds|holds|holds|
|CLIP B/16|holds|holds|holds|holds|holds|

## Cells

### ArtELingo held (12,281 paintings)

Entries: **c*/comparator; d [95%]**; every entry is a WIN unless marked.

|Encoder, aspect|SC1|SC2|SC3|
|---|---|---|---|
|DINOv2, emotion|0.0859/0.0602; +0.0257 [+0.0228, +0.0285]|0.2549/0.2368; +0.0181 [+0.0172, +0.0191]|0.0859/0.0728; +0.0131 [+0.0083, +0.0137]|
|DINOv2, style|0.3444/0.2354; +0.1090 [+0.1035, +0.1133]|0.2657/0.1791; +0.0866 [+0.0843, +0.0890]|0.3444/0.1970; +0.1474 [+0.1373, +0.1496]|
|DINOv2, genre|0.4866/0.4209; +0.0657 [+0.0600, +0.0715]|0.6007/0.5051; +0.0956 [+0.0931, +0.0980]|0.4866/0.4218; +0.0648 [+0.0567, +0.0722]|
|CLIP B/32, emotion|0.0919/0.0642; +0.0277 [+0.0246, +0.0307]|0.2592/0.2472; +0.0120 [+0.0109, +0.0130]|0.0919/0.0716; +0.0203 [+0.0151, +0.0214]|
|CLIP B/32, style|0.3743/0.2978; +0.0764 [+0.0717, +0.0810]|0.2928/0.2348; +0.0580 [+0.0563, +0.0597]|0.3743/0.2161; +0.1582 [+0.1473, +0.1610]|
|CLIP B/32, genre|0.4892/0.4085; +0.0807 [+0.0749, +0.0868]|0.5981/0.4905; +0.1076 [+0.1051, +0.1101]|0.4892/0.4286; +0.0606 [+0.0524, +0.0687]|
|CLIP B/16, emotion|0.0968/0.0631; +0.0337 [+0.0304, +0.0368]|0.2624/0.2469; +0.0155 [+0.0146, +0.0164]|0.0968/0.0749; +0.0219 [+0.0169, +0.0230]|
|CLIP B/16, style|0.3922/0.3066; +0.0855 [+0.0800, +0.0906]|0.3086/0.2457; +0.0629 [+0.0606, +0.0653]|0.3922/0.2228; +0.1694 [+0.1585, +0.1726]|
|CLIP B/16, genre|0.4952/0.4119; +0.0834 [+0.0768, +0.0901]|0.6081/0.5040; +0.1040 [+0.1015, +0.1067]|0.4952/0.4309; +0.0643 [+0.0550, +0.0737]|

### Affection test, photo model

Entries: **c*/comparator; d [95%]**; every entry is a WIN unless marked.

|Encoder, aspect|SC4|SC5|
|---|---|---|
|DINOv2, emotion|0.1547/0.0875; +0.0672 [+0.0616, +0.0730]|0.2870/0.2273; +0.0597 [+0.0579, +0.0614]|
|CLIP B/32, emotion|0.1711/0.1116; +0.0595 [+0.0543, +0.0643]|0.3049/0.2527; +0.0522 [+0.0500, +0.0542]|
|CLIP B/16, emotion|0.1808/0.1147; +0.0661 [+0.0609, +0.0713]|0.3125/0.2559; +0.0565 [+0.0539, +0.0589]|

### MMArt test (16,384 paintings)

Entries: **c*/comparator; d [95%]**; every entry is a WIN unless marked.

|Encoder, aspect|SC7|SC8|SC9|
|---|---|---|---|
|DINOv2, emotion|0.0510/0.0367; +0.0143 [+0.0123, +0.0163]|0.1939/0.1890; +0.0049 [+0.0045, +0.0053]|0.0510/0.0437; +0.0073 [+0.0038, +0.0073]|
|DINOv2, style|0.3682/0.2510; +0.1173 [+0.1120, +0.1216]|0.2924/0.1929; +0.0994 [+0.0971, +0.1017]|0.3682/0.2012; +0.1670 [+0.1584, +0.1702]|
|DINOv2, genre|0.4978/0.4255; +0.0724 [+0.0662, +0.0788]|0.6078/0.4632; +0.1446 [+0.1420, +0.1471]|0.4978/0.4299; +0.0679 [+0.0602, +0.0760]|
|CLIP B/32, emotion|0.0517/0.0356; +0.0161 [+0.0140, +0.0181]|0.1958/0.1909; +0.0050 [+0.0045, +0.0054]|0.0517/0.0415; +0.0102 [+0.0065, +0.0106]|
|CLIP B/32, style|0.3916/0.3110; +0.0807 [+0.0768, +0.0840]|0.3188/0.2566; +0.0622 [+0.0606, +0.0639]|0.3916/0.2215; +0.1702 [+0.1619, +0.1734]|
|CLIP B/32, genre|0.4924/0.4127; +0.0796 [+0.0745, +0.0845]|0.6037/0.5004; +0.1033 [+0.1010, +0.1055]|0.4924/0.4284; +0.0639 [+0.0561, +0.0715]|
|CLIP B/16, emotion|0.0552/0.0374; +0.0179 [+0.0157, +0.0200]|0.1978/0.1908; +0.0070 [+0.0066, +0.0074]|0.0552/0.0432; +0.0120 [+0.0083, +0.0125]|
|CLIP B/16, style|0.4139/0.3207; +0.0932 [+0.0885, +0.0976]|0.3344/0.2685; +0.0658 [+0.0640, +0.0677]|0.4139/0.2274; +0.1866 [+0.1766, +0.1911]|
|CLIP B/16, genre|0.5051/0.4409; +0.0642 [+0.0590, +0.0697]|0.6143/0.5143; +0.0999 [+0.0976, +0.1023]|0.5051/0.4378; +0.0672 [+0.0593, +0.0752]|

### SemArt new artists (11,159 paintings)

Entries: **c*/comparator; d [95%]**; every entry is a WIN unless marked.

|Encoder, aspect|SC11|SC12|SC13|
|---|---|---|---|
|Qwen, genre|0.4972/0.4369; +0.0603 [+0.0550, +0.0656]|0.7075/0.5941; +0.1135 [+0.1114, +0.1156]|0.4972/0.3765; +0.1207 [+0.1111, +0.1295]|
|SigLIP 2, genre|0.4983/0.4630; +0.0353 [+0.0310, +0.0398]|0.7057/0.6005; +0.1052 [+0.1032, +0.1073]|0.4983/0.4022; +0.0962 [+0.0832, +0.1077]|
|SigLIP 1, genre|0.4980/0.4667; +0.0312 [+0.0263, +0.0364]|0.7077/0.6035; +0.1042 [+0.1020, +0.1064]|0.4980/0.4168; +0.0812 [+0.0675, +0.0936]|
|CLIP L/14, genre|0.4873/0.4577; +0.0296 [+0.0246, +0.0350]|0.6951/0.6106; +0.0845 [+0.0822, +0.0868]|0.4873/0.4085; +0.0788 [+0.0690, +0.0880]|
|DINOv2, genre|0.4638/0.4070; +0.0568 [+0.0517, +0.0620]|0.6642/0.5580; +0.1063 [+0.1032, +0.1095]|0.4638/0.3763; +0.0875 [+0.0792, +0.0949]|
|CLIP B/32, genre|0.4547/0.4262; +0.0285 [+0.0240, +0.0334]|0.6581/0.5819; +0.0762 [+0.0742, +0.0783]|0.4547/0.3744; +0.0803 [+0.0705, +0.0894]|
|CLIP B/16, genre|0.4741/0.4509; +0.0231 [+0.0178, +0.0286]|0.6795/0.5885; +0.0910 [+0.0887, +0.0934]|0.4741/0.3847; +0.0894 [+0.0776, +0.1009]|

## Teacher cells

B = "better than"; L = "level with or better than"; None = no claim.

### ArtELingo held and Affection test

|Encoder|Data|Aspect|Task|c*/teacher|d [95%]|r|Outcome|Set for "better than"|Claim|
|---|---|---|---|---|---|---|---|---|---|
|DINOv2|affection_test|emotion|AMI|0.1547/0.1666|-0.0119 [-0.0179, -0.0056]|0.0057|BELOW|no|None|
|DINOv2|affection_test|emotion|mAP|0.2870/0.2945|-0.0075 [-0.0094, -0.0057]|0.0021|BELOW|no|None|
|DINOv2|artelingo_held|emotion|AMI|0.0859/0.0921|-0.0061 [-0.0092, -0.0029]|0.0043|BELOW|no|None|
|DINOv2|artelingo_held|emotion|mAP|0.2549/0.2684|-0.0135 [-0.0146, -0.0124]|0.0032|BELOW|no|None|
|DINOv2|artelingo_held|style|AMI|0.3444/0.4026|-0.0581 [-0.0622, -0.0525]|0.0110|BELOW|no|None|
|DINOv2|artelingo_held|style|mAP|0.2657/0.3217|-0.0560 [-0.0585, -0.0535]|0.0032|BELOW|no|None|
|DINOv2|artelingo_held|genre|AMI|0.4866/0.5062|-0.0195 [-0.0250, -0.0138]|0.0116|BELOW|no|None|
|DINOv2|artelingo_held|genre|mAP|0.6007/0.6124|-0.0118 [-0.0140, -0.0096]|0.0044|BELOW|no|None|
|CLIP B/32|affection_test|emotion|AMI|0.1711/0.1666|+0.0045 [-0.0010, +0.0099]|0.0121|LEVEL|no|L|
|CLIP B/32|affection_test|emotion|mAP|0.3049/0.2945|+0.0104 [+0.0081, +0.0125]|0.0051|WIN|no|L|
|CLIP B/32|artelingo_held|emotion|AMI|0.0919/0.0921|-0.0002 [-0.0036, +0.0032]|0.0063|LEVEL|no|L|
|CLIP B/32|artelingo_held|emotion|mAP|0.2592/0.2684|-0.0092 [-0.0103, -0.0082]|0.0019|BELOW|no|None|
|CLIP B/32|artelingo_held|style|AMI|0.3743/0.4026|-0.0283 [-0.0328, -0.0228]|0.0091|BELOW|no|None|
|CLIP B/32|artelingo_held|style|mAP|0.2928/0.3217|-0.0289 [-0.0308, -0.0270]|0.0030|BELOW|no|None|
|CLIP B/32|artelingo_held|genre|AMI|0.4892/0.5062|-0.0170 [-0.0232, -0.0111]|0.0108|BELOW|no|None|
|CLIP B/32|artelingo_held|genre|mAP|0.5981/0.6124|-0.0144 [-0.0167, -0.0121]|0.0025|BELOW|no|None|
|CLIP B/16|affection_test|emotion|AMI|0.1808/0.1666|+0.0142 [+0.0089, +0.0191]|0.0049|WIN|no|L|
|CLIP B/16|affection_test|emotion|mAP|0.3125/0.2945|+0.0179 [+0.0153, +0.0203]|0.0062|WIN|no|L|
|CLIP B/16|artelingo_held|emotion|AMI|0.0968/0.0921|+0.0048 [+0.0017, +0.0080]|0.0031|WIN|no|L|
|CLIP B/16|artelingo_held|emotion|mAP|0.2624/0.2684|-0.0060 [-0.0069, -0.0051]|0.0017|BELOW|no|None|
|CLIP B/16|artelingo_held|style|AMI|0.3922/0.4026|-0.0104 [-0.0150, -0.0054]|0.0138|BELOW|no|None|
|CLIP B/16|artelingo_held|style|mAP|0.3086/0.3217|-0.0131 [-0.0154, -0.0108]|0.0037|BELOW|no|None|
|CLIP B/16|artelingo_held|genre|AMI|0.4952/0.5062|-0.0109 [-0.0171, -0.0047]|0.0121|BELOW|no|None|
|CLIP B/16|artelingo_held|genre|mAP|0.6081/0.6124|-0.0044 [-0.0065, -0.0021]|0.0027|BELOW|no|None|

### MMArt test

|Encoder|Data|Aspect|Task|c*/teacher|d [95%]|r|Outcome|Set for "better than"|Claim|
|---|---|---|---|---|---|---|---|---|---|
|DINOv2|mmart_test|emotion|AMI|0.0510/0.0566|-0.0056 [-0.0077, -0.0034]|0.0021|BELOW|no|None|
|DINOv2|mmart_test|emotion|mAP|0.1939/0.2021|-0.0082 [-0.0086, -0.0077]|0.0009|BELOW|no|None|
|DINOv2|mmart_test|style|AMI|0.3682/0.4158|-0.0476 [-0.0518, -0.0427]|0.0040|BELOW|no|None|
|DINOv2|mmart_test|style|mAP|0.2924/0.3476|-0.0552 [-0.0574, -0.0531]|0.0032|BELOW|no|None|
|DINOv2|mmart_test|genre|AMI|0.4978/0.5071|-0.0092 [-0.0151, -0.0030]|0.0112|BELOW|no|None|
|DINOv2|mmart_test|genre|mAP|0.6078/0.6172|-0.0095 [-0.0113, -0.0076]|0.0042|BELOW|no|None|
|CLIP B/32|mmart_test|emotion|AMI|0.0517/0.0566|-0.0049 [-0.0070, -0.0025]|0.0037|BELOW|no|None|
|CLIP B/32|mmart_test|emotion|mAP|0.1958/0.2021|-0.0062 [-0.0068, -0.0057]|0.0010|BELOW|no|None|
|CLIP B/32|mmart_test|style|AMI|0.3916/0.4158|-0.0242 [-0.0280, -0.0200]|0.0122|BELOW|no|None|
|CLIP B/32|mmart_test|style|mAP|0.3188/0.3476|-0.0287 [-0.0306, -0.0269]|0.0038|BELOW|no|None|
|CLIP B/32|mmart_test|genre|AMI|0.4924/0.5071|-0.0147 [-0.0204, -0.0094]|0.0147|BELOW|no|None|
|CLIP B/32|mmart_test|genre|mAP|0.6037/0.6172|-0.0135 [-0.0157, -0.0114]|0.0025|BELOW|no|None|
|CLIP B/16|mmart_test|emotion|AMI|0.0552/0.0566|-0.0013 [-0.0035, +0.0009]|0.0053|LEVEL|no|L|
|CLIP B/16|mmart_test|emotion|mAP|0.1978/0.2021|-0.0043 [-0.0047, -0.0038]|0.0012|BELOW|no|None|
|CLIP B/16|mmart_test|style|AMI|0.4139/0.4158|-0.0019 [-0.0062, +0.0027]|0.0135|LEVEL|no|L|
|CLIP B/16|mmart_test|style|mAP|0.3344/0.3476|-0.0132 [-0.0150, -0.0114]|0.0037|BELOW|no|None|
|CLIP B/16|mmart_test|genre|AMI|0.5051/0.5071|-0.0020 [-0.0074, +0.0033]|0.0059|BELOW|no|None|
|CLIP B/16|mmart_test|genre|mAP|0.6143/0.6172|-0.0029 [-0.0050, -0.0009]|0.0027|BELOW|no|None|

### SemArt new artists

|Encoder|Data|Aspect|Task|c*/teacher|d [95%]|r|Outcome|Set for "better than"|Claim|
|---|---|---|---|---|---|---|---|---|---|
|Qwen|semart_new|genre|AMI|0.4972/0.4968|+0.0004 [-0.0035, +0.0041]|0.0139|LEVEL|no|L|
|Qwen|semart_new|genre|mAP|0.7075/0.7085|-0.0009 [-0.0019, +0.0001]|0.0038|LEVEL|yes|L|
|SigLIP 2|semart_new|genre|AMI|0.4983/0.4968|+0.0015 [-0.0027, +0.0052]|0.0070|LEVEL|no|L|
|SigLIP 2|semart_new|genre|mAP|0.7057/0.7085|-0.0028 [-0.0042, -0.0013]|0.0043|LEVEL|yes|L|
|SigLIP 1|semart_new|genre|AMI|0.4980/0.4968|+0.0011 [-0.0034, +0.0058]|0.0115|LEVEL|no|L|
|SigLIP 1|semart_new|genre|mAP|0.7077/0.7085|-0.0008 [-0.0023, +0.0007]|0.0041|LEVEL|yes|L|
|CLIP L/14|semart_new|genre|AMI|0.4873/0.4968|-0.0095 [-0.0149, -0.0047]|0.0159|LEVEL|no|L|
|CLIP L/14|semart_new|genre|mAP|0.6951/0.7085|-0.0133 [-0.0154, -0.0114]|0.0017|BELOW|yes|None|
|DINOv2|semart_new|genre|AMI|0.4638/0.4968|-0.0330 [-0.0389, -0.0275]|0.0116|BELOW|no|None|
|DINOv2|semart_new|genre|mAP|0.6642/0.7085|-0.0442 [-0.0473, -0.0411]|0.0044|BELOW|no|None|
|CLIP B/32|semart_new|genre|AMI|0.4547/0.4968|-0.0422 [-0.0477, -0.0366]|0.0108|BELOW|no|None|
|CLIP B/32|semart_new|genre|mAP|0.6581/0.7085|-0.0504 [-0.0527, -0.0481]|0.0025|BELOW|no|None|
|CLIP B/16|semart_new|genre|AMI|0.4741/0.4968|-0.0228 [-0.0284, -0.0172]|0.0121|BELOW|no|None|
|CLIP B/16|semart_new|genre|mAP|0.6795/0.7085|-0.0289 [-0.0310, -0.0269]|0.0027|BELOW|no|None|

## Cost (ms per image, model time, 2,000 MMArt paintings, batch 32)

|Encoder|c* mean|c* slowest|Teacher mean|Teacher fastest|Ratio of means|c* with decoding|Teacher with decoding|
|---|---|---|---|---|---|---|---|
|Qwen|88.6|88.8|266.8|266.5|3.0|88.5|267.6|
|SigLIP 2|19.4|19.8|266.8|266.5|13.8|19.0|267.6|
|SigLIP 1|19.5|20.0|266.8|266.5|13.7|19.4|267.6|
|CLIP L/14|7.9|8.1|266.8|266.5|33.8|8.6|267.6|
|DINOv2|8.5|8.7|266.8|266.5|31.3|9.4|267.6|
|CLIP B/32|5.2|5.2|266.8|266.5|51.4|6.9|267.6|
|CLIP B/16|5.8|5.8|266.8|266.5|46.3|7.3|267.6|
