# Deep Stage 1 / Stage 2 analysis: buddy vs. fixed PercepT

Generated 2026-09-27 20:15:59. Companion to [`docs/reports/auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md`](../../2026-09-26_artelingo_buddy_vs_percept_stage1.md) §6b. Buddy has 19 held-out topics (k=20 transfer); PercepT (bug-fixed, K=60/40) has 40 surviving held-out topics.

Buddy Stage-2 macro AUC in this run: 0.5978 (cross-check against `buddy_stage2_pilot_report.md`'s 0.5978 -- minor differences are expected from a fresh mapper-init draw at the same seed if any upstream randomness source differs; treat close agreement as a sanity pass).

## 1. Topic characterization

Majority emotion/genre and share are of the topic's held-out members; genre share is only over the genre-labelled subset (thin sample, see `genre_n`). Entropy is the Shannon entropy (bits) of the topic's emotion-label distribution -- lower means more emotionally homogeneous. Captions are a seed-42 sample of up to 3 held-out members' first English caption.

### Buddy (19 topics)

| topic | size | majority emotion | share | majority genre | genre share | genre n | entropy | example captions |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 0 | 1033 | contentment | 0.43 | landscape | 0.67 | 27 | 2.44 | It looks like the canals of Venice. really like the details, the beautiful sky,  / This is a very detailed drawing of castle ramparts and entrance but the tone is  / it makes you feel satisfied on your current work |
| 1 | 811 | contentment | 0.23 | genre_painting | 0.38 | 16 | 2.78 | i feel confused.  I have no idea what this is meant to depict and what the text  / The soft colors are soothing. The mother has a serene look on her face and the c / It almost seems like the subjects here are underwater, and I am very confused at |
| 2 | 917 | something else | 0.37 | abstract_painting | 0.50 | 12 | 2.69 | The brush strokes are loose, subject matter comfortable, the colors sedate / Im confused because the words are foreign. / Annoyed because I have a phobia about things this simple |
| 3 | 795 | fear | 0.52 | illustration | 0.50 | 4 | 2.31 | A man is standing on the grass while a woman is walking close to him with two ki / This looks like scribbles and it angers me when this kind of thing is said to be / The sharp diagonal slashes give off a sense of danger |
| 4 | 945 | sadness | 0.54 | portrait | 0.50 | 16 | 2.05 | the expression of the figure seems regretful / It looks as if it is a person begging the other person to stay. / The colours, smudging and lack of detail about the person makes me feel down. |
| 5 | 620 | contentment | 0.72 | landscape | 0.40 | 10 | 1.24 | a lush tree with leaves full of birds / I find the colors in painting to be soothing. / Very calm colors and the animals in the painting seem content. |
| 6 | 526 | contentment | 0.48 | religious_painting | 0.33 | 12 | 2.32 | The woman appears quite loving as she observes the children laying on the floor. / The man holds his child in a loving way. / The town square filled with people makes for a quaint scene to me. |
| 7 | 594 | amusement | 0.50 | portrait | 0.80 | 5 | 2.35 | Such a plump little guy. And his knees! You just have to smile or laugh at the p / I feel amusement because the man has a happy expression even though he is all dr / The man seems to have a sock coming out of his head |
| 8 | 424 | contentment | 0.36 | genre_painting | 0.33 | 6 | 2.40 | The yellow haze of light in the room makes it appear warm and inviting. / Different peoples with horse ridding in a function or any else looks colourful / The experience the best colors of fall, it is the most beautiful time to be outs |
| 9 | 558 | disgust | 0.26 | portrait | 0.50 | 8 | 2.89 | This makes me feel agitated because the woman's expression and pose look aggress / Doesn't feel good about this painting. / A middle aged man from the 1940s. |
| 10 | 523 | contentment | 0.23 | portrait | 0.71 | 14 | 2.88 | There is something about this portrait that makes me smile. Maybe the facial fea / The disagreeable look on the man's face, the way his hands are placed and the we / the man seems to portray an angry stern man |
| 11 | 367 | contentment | 0.47 | portrait | 0.67 | 6 | 2.19 | the ladie in the background looks to have beef with the girl sitting on the grou / the head and hair has not been finished therefore it looks very awkward / Childs seem to be satisfied with their reading activity. |
| 12 | 278 | contentment | 0.73 | landscape | 0.60 | 5 | 1.26 | This makes me feel content because the weather and the landscape look pleasant. / the abstract colors actually work well here, its beautiful. / The scene looks inviting, and the details in the blades of grass brings life to  |
| 13 | 240 | contentment | 0.66 | still_life | 1.00 | 4 | 1.79 | One of the flowers has fallen off to the side of the vase / Bright colorful flowers make it feel like a warm spring day. / looks like a slab of mashed potatoes topped with a salad |
| 14 | 230 | awe | 0.55 | religious_painting | 1.00 | 3 | 2.00 | i feel this way because of the way the figures are pointed. / Not sure if that mantle is part of the painting or a real mantle. Very realistic / Everyone is bowing and kneeling down to Jesus, showing deference to him |
| 15 | 283 | contentment | 0.30 | cityscape | 0.50 | 8 | 2.89 | the fountain looks like it has water sprouting from her breast. / The landscape looks like a mirage as it blurs from view. / The simple mountain scene has a good sense of scale when compared to the horse r |
| 16 | 87 | contentment | 0.68 | landscape | 1.00 | 2 | 1.55 | The vertical lines in the top right look like an impending storm. / This looks like a real room and a nice relaxing place to sit. / The surreal blurring and warm colors give this painting something like a boozy n |
| 17 | 69 | contentment | 0.52 | n/a | nan | 0 | 2.03 | It looks like a forest where a murder took place, because it is very dark and dr / Since the ballet dancer is probably performing, she does a really great job of d / the painting has an air of innocence, as if the father is teaching his daughter  |
| 18 | 65 | contentment | 0.62 | cityscape | 1.00 | 1 | 1.76 | I love the city skyline and people walking peacefully / The colors are vibrant and the street scene is busy in spite of the recent rain. / The muted tones of the painting give it a calm feel. |

### PercepT, bug-fixed (40 topics)

| topic | size | majority emotion | share | majority genre | genre share | genre n | entropy | example captions |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 0 | 13 | awe | 0.31 | n/a | nan | 0 | 2.57 | An interesting painting of the human body which makes me feel I would rather see / The bright colors add to the jovial nature of the women dancing / Several men playing near a river all naked |
| 1 | 64 | contentment | 0.70 | n/a | nan | 0 | 1.58 | The clouds look like white sheets hanging from the sky / the tree tops look like green waves crashing onto the shore / This painting shows a forest of trees with a blue and white sky. |
| 2 | 30 | contentment | 0.73 | genre_painting | 1.00 | 1 | 1.44 | a woman talking to a young shepherd boy while pointing across a river / The child isn't helping with any of the chores / It places one in a early afternoon trip to the market state of mind. |
| 3 | 37 | contentment | 0.54 | landscape | 0.50 | 2 | 1.99 | Interest is formed with the lines of the buildings and a whimsical feel in how t / A great blend of colors that makes the painting look magical along with a great  / The richness of color and overall hustle and bustle are inspiring |
| 4 | 80 | contentment | 0.59 | still_life | 1.00 | 2 | 1.95 | I love fruit, but these just look sad and unappetizing. They're dull, and lack c / This reminds me of home cooking. / i feel very nuetral when i look at this. i am not sure why |
| 5 | 113 | contentment | 0.72 | portrait | 1.00 | 1 | 1.59 | This man looks peaceful holding his cat and it shows comfort.  I am curious what / The hand holding gesture between the man and woman suggests romantic comraderie. / very nice brush strokes here, they add an element of movement to the painting. |
| 6 | 79 | contentment | 0.63 | genre_painting | 0.33 | 3 | 1.87 | I only enjoy the sharpness of the white triangles above the green, however the s / This image is very stark and lonely and depressing. / Looks like an alleyway that would be a nice place to get away for some privacy. |
| 7 | 187 | contentment | 0.88 | landscape | 1.00 | 1 | 0.79 | A quiet, cool, peaceful fall day walking in the park. / There is much tranquility between the pale colors of the building and the little / A family gathering for a picnic, by the water. The daughter looks bored. |
| 8 | 147 | contentment | 0.76 | still_life | 1.00 | 2 | 1.33 | The soft bright flowers, and the light being reflected off of the vase, contrast / The swirling green of the upper background is as lively as the flowers in the va / The deep orange is immediately eye catching and serves as a nice background for  |
| 9 | 62 | excitement | 0.50 | genre_painting | 1.00 | 1 | 2.23 | the people harmonizing together in the park is a lovely thing to see / A beach party with friends, looks like so much fun! / I feel like i want to go to the beach. |
| 10 | 88 | contentment | 0.49 | n/a | nan | 0 | 2.09 | the lady appears to be prepared to attend a party / The woman in the painting is praying makes me feel contentment / The insect-like wings seem very twee. It is interesting that the figures are sit |
| 11 | 153 | contentment | 0.46 | portrait | 1.00 | 2 | 2.26 | the womans pink dress compliments the green tone background. / Honestly this reminds me of myself as a young teenager. I had a similar haircut. / The eyes are striking, they show interest and perhaps satisfaction. |
| 12 | 107 | contentment | 0.37 | landscape | 1.00 | 1 | 1.95 | The sun slowly setting reminds me of relaxation after a day's work. / The sun is rising over the ocean, providing opportunity and warmth / The lighting does a good job on putting focus on the angel on the right. |
| 13 | 203 | contentment | 0.38 | landscape | 0.50 | 6 | 2.41 | wow, I want to be in these woods walking with these animals in this sunshine / The branches of this tree is sticking out in all directions like it had a bad ha / Nothing in particular is felt here. The image is very blurry even though it is i |
| 14 | 58 | something else | 0.31 | n/a | nan | 0 | 2.79 | Confusion. I don't know what this cut up painting is supposed to be so it makes  / The jagged grey shapes look ready to devour the happy colors above them-very sha / I feel sad because the horse looks hurt and weak and trapped. |
| 15 | 77 | awe | 0.42 | landscape | 1.00 | 2 | 1.94 | The purple and orange with the looming dark mountain and (what appears to be) a  / the landscape looks dangerous and unsafe, like it would be scary to walk through / The clouds in the sky echo the shape of the mountains but are softer and rounded |
| 16 | 477 | contentment | 0.73 | landscape | 0.67 | 9 | 1.30 | The colorful, bright colors make me feel at peace and at home. / A nice day outside with no one to bother me. / A sad setting that depresses one self, has a very damp and closed-off atmosphere |
| 17 | 160 | contentment | 0.55 | landscape | 0.60 | 5 | 1.60 | I enjoy how bright the sky is in comparison to the rest of the piece. / The grey sky in the background is a bit scary because you can't tell what lies b / The yellow water goes very well with the green rocks. |
| 18 | 131 | contentment | 0.24 | n/a | nan | 0 | 2.87 | I like the sketching of the nude lady and her shaved vagina. / Too much nudity. Meh about the image. / The statue has a nice butt - but it's missing a head and limbs. |
| 19 | 187 | contentment | 0.72 | landscape | 0.60 | 5 | 1.36 | Impressive! The use of distance/perspective, with more detail in the foreground  / It's a simpler time. I feel calm. Even the lack of bright color gives a sense of / The water looks peaceful and calm. The boats are large and they look like fishin |
| 20 | 182 | contentment | 0.43 | landscape | 0.86 | 7 | 2.11 | The sparseness of what appears to be a section of a larget painting is rather du / i love the peacefulness of this painting, feels light and calm / This makes me feel calm. I can sense the plodding pace of the horse. |
| 21 | 210 | contentment | 0.50 | genre_painting | 0.67 | 3 | 2.12 | I enjoy the realistic look of this man's portrait / Contentment, everyone including the animals look so warm and relaxed. / There is a lot happening in this painting. The attention to detail and color is  |
| 22 | 100 | contentment | 0.34 | sketch_and_study | 0.50 | 2 | 2.55 | Painting reminds me of something I would see in a book when I was younger and br / The level of detail is striking and the use of black and white makes it pop out / There is an element of sadness, as though they realize that they are stuck on th |
| 23 | 265 | fear | 0.30 | sketch_and_study | 0.40 | 5 | 2.74 | it seems like a man is raping a woman / The art style and use of straight lines give me a sense of wonder. / Gloomy because it is shaded dark. There are faces in the top half that I don't k |
| 24 | 136 | contentment | 0.53 | cityscape | 1.00 | 3 | 2.06 | I like how the light brightens the building fronts.  i wish the foreground was c / a happy frenzy with a building that could be the Louvre is exciting / What a wonderfully colorful event taking place, a little darkness in the sky mig |
| 25 | 85 | awe | 0.34 | portrait | 0.50 | 2 | 2.46 | I feel indifferent because I like the angels and some of the color but the rest  / The cool pastel blue and portrait as if on real currency is interesting. / This photo makes me feel bored. There is no real effort here. It's really just a |
| 26 | 192 | something else | 0.35 | genre_painting | 0.33 | 3 | 2.72 | Nice rendering of the coloring and curving drawing / the faces in the painting are distorted, giving them a malicious, near evil look / The person's large size and the way the instument is held so lovingly makes a st |
| 27 | 541 | contentment | 0.48 | portrait | 0.64 | 14 | 2.18 | The way the artist captured the texture and translucence of the layered dress / The white highlights on the horse's nose make the fur look buttery soft. And the / Confusion. For a second there, I could not find this girl's nipples and was wond |
| 28 | 860 | sadness | 0.25 | portrait | 0.40 | 10 | 2.86 | The people appear to be totally comfortable and relaxed being nude around each o / his blank eye sockets where the eyes should be are haunting / A black cloud fills the sky as if a terrible storm approaches. |
| 29 | 150 | fear | 0.23 | portrait | 1.00 | 2 | 2.90 | The paint tones and the faerie-like beings in the painting leave me with a sense / Black and white paintings make me so content.It is calm and takes you back to wh / This makes me feel disgusted because of what is going on in the picture. I know  |
| 30 | 400 | fear | 0.21 | illustration | 0.40 | 5 | 2.91 | The subjects look wearily resigned / This painting looks like it is a hopefully religious painting from the renaissan / This scene looks as if all involved parties are up to no good. I don't feel anyt |
| 31 | 351 | contentment | 0.26 | abstract_painting | 1.00 | 3 | 2.57 | The red and black contrast nicely. Not sure what it is, so dont know how to reac / The colors make it look like I'm watching an old cartoon. / The objects look like claws. |
| 32 | 165 | awe | 0.51 | religious_painting | 1.00 | 1 | 1.91 | The strong nature of the building makes me appreciate those who built it and tho / I feel inspired, as if I were actually in Venice or some other busy port town. / It captures the busy scene wonderfully well. |
| 33 | 453 | sadness | 0.50 | portrait | 0.75 | 8 | 2.13 | This looks like a woman who would have been very popular and would have had mone / the look on the subjects face does not look very good, is scary / The darkness of the drawing and the sunken features make are unsettling. |
| 34 | 282 | amusement | 0.54 | sketch_and_study | 0.50 | 2 | 2.24 | I feel conflicted.  The lamb is so sweet, but he is goinf to be someone.s dinner / Their facial expressions are just hysterical, especially hers!~! / It looks like this is meant to show hard times and how numb people are look at t |
| 35 | 327 | contentment | 0.36 | religious_painting | 0.88 | 8 | 2.32 | The woman appears to have small bloody cuts on her face / Two Madonnas and Child floating in the heavens. The colors are vibrant and vivid / I have a serious and studious feeling when I see this man who is poised against  |
| 36 | 314 | awe | 0.31 | portrait | 0.89 | 9 | 2.71 | The warrior looks ferocious but the outfit is beautiful. / This guy looks aristocratic, yet cheery. I feel nothing looking at this. / A proud and confident black man from a different era. |
| 37 | 753 | something else | 0.27 | abstract_painting | 0.56 | 9 | 2.89 | I feel rather indifferent, as the yellow block isnt especially stimulating. / mix of color to show a cup of wine, a knife and a bowl of fruit / I feel like this is a great representation of a man's thoughts at any given time |
| 38 | 518 | awe | 0.25 | religious_painting | 0.58 | 12 | 2.84 | The person in the photo made me feel gloomy because of their pose. / He wonders why does his king wage a war which he must fight. / What kind of weird imagry was the painter going for? I find it funny he is makin |
| 39 | 628 | sadness | 0.27 | cityscape | 0.38 | 8 | 2.67 | Looks very dark and foreboding, the shadowy figure is scary. / The dark is uncomfortable and the crouched person looks in pain. / The horses look like they're being whipped into submission |

## 2. Per-topic AUC vs. occupancy

Buddy: Pearson r=0.531 (p=0.019), Spearman r=0.493 (p=0.032), n=19 topics. ![buddy AUC vs occupancy](assets/buddy_auc_vs_occupancy.png)

PercepT: Pearson r=0.363 (p=0.021), Spearman r=0.294 (p=0.066), n=40 topics. ![percept AUC vs occupancy](assets/percept_auc_vs_occupancy.png)

## 3. Cross-system topic alignment

Normalized mutual information (buddy 19-way vs. PercepT 40-way, same 9365 held-out paintings): **NMI = 0.2947**.

Top PercepT topics each buddy community maps onto most heavily (topic: shared-painting count):

- Buddy 0 -> P39:300, P20:144, P16:74
- Buddy 1 -> P38:99, P28:74, P30:67
- Buddy 2 -> P37:420, P31:208, P26:85
- Buddy 3 -> P39:105, P38:99, P23:84
- Buddy 4 -> P33:285, P28:241, P38:92
- Buddy 5 -> P16:177, P7:90, P19:63
- Buddy 6 -> P35:108, P27:54, P37:36
- Buddy 7 -> P34:146, P27:68, P28:54
- Buddy 8 -> P21:60, P9:38, P31:33
- Buddy 9 -> P28:129, P18:68, P37:54
- Buddy 10 -> P28:139, P36:136, P27:49
- Buddy 11 -> P27:191, P11:66, P28:32
- Buddy 12 -> P16:140, P19:33, P17:26
- Buddy 13 -> P8:123, P4:62, P16:11
- Buddy 14 -> P38:56, P32:50, P35:44
- Buddy 15 -> P28:53, P13:51, P30:44
- Buddy 16 -> P16:15, P24:10, P19:8
- Buddy 17 -> P38:7, P12:5, P27:5
- Buddy 18 -> P24:49, P19:7, P16:3

Top buddy communities each PercepT topic maps onto most heavily:

- PercepT 0 -> B7:3, B8:3, B9:2
- PercepT 1 -> B0:19, B5:17, B6:4
- PercepT 2 -> B5:8, B6:5, B4:4
- PercepT 3 -> B0:17, B5:9, B2:3
- PercepT 4 -> B13:62, B15:5, B1:2
- PercepT 5 -> B6:20, B5:20, B8:18
- PercepT 6 -> B0:20, B8:18, B5:14
- PercepT 7 -> B5:90, B0:38, B12:19
- PercepT 8 -> B13:123, B8:7, B6:4
- PercepT 9 -> B8:38, B1:5, B3:5
- PercepT 10 -> B8:20, B7:16, B6:15
- PercepT 11 -> B11:66, B10:24, B4:14
- PercepT 12 -> B0:35, B3:22, B8:13
- PercepT 13 -> B15:51, B0:41, B3:29
- PercepT 14 -> B2:30, B1:8, B3:6
- PercepT 15 -> B5:22, B0:20, B12:15
- PercepT 16 -> B5:177, B12:140, B0:74
- PercepT 17 -> B5:56, B0:46, B12:26
- PercepT 18 -> B9:68, B1:18, B11:16
- PercepT 19 -> B5:63, B0:39, B12:33
- PercepT 20 -> B0:144, B4:14, B3:9
- PercepT 21 -> B8:60, B7:46, B1:30
- PercepT 22 -> B0:65, B2:10, B3:10
- PercepT 23 -> B3:84, B1:62, B9:27
- PercepT 24 -> B18:49, B0:38, B8:13
- PercepT 25 -> B1:20, B6:12, B3:9
- PercepT 26 -> B2:85, B9:42, B3:15
- PercepT 27 -> B11:191, B7:68, B1:64
- PercepT 28 -> B4:241, B10:139, B9:129
- PercepT 29 -> B3:55, B1:37, B9:34
- PercepT 30 -> B3:69, B1:67, B4:52
- PercepT 31 -> B2:208, B8:33, B7:31
- PercepT 32 -> B14:50, B0:36, B5:18
- PercepT 33 -> B4:285, B1:39, B6:28
- PercepT 34 -> B7:146, B9:37, B1:26
- PercepT 35 -> B6:108, B4:65, B14:44
- PercepT 36 -> B10:136, B1:36, B14:29
- PercepT 37 -> B2:420, B3:67, B9:54
- PercepT 38 -> B1:99, B3:99, B4:92
- PercepT 39 -> B0:300, B3:105, B4:64

## 4. Emotion-conditioned error analysis

Top-1 accuracy = predicted topic (argmax mapper score) equals the painting's true hard-assigned topic. Full held-out set (9,365 paintings), grouped by ground-truth majority emotion.

| emotion | buddy top-1 acc | buddy n | PercepT top-1 acc | PercepT n |
|---|---:|---:|---:|---:|
| amusement | 0.117 | 881 | 0.198 | 881 |
| anger | 0.158 | 95 | 0.326 | 95 |
| awe | 0.189 | 1328 | 0.158 | 1328 |
| contentment | 0.181 | 3031 | 0.210 | 3031 |
| disgust | 0.087 | 416 | 0.329 | 416 |
| excitement | 0.120 | 532 | 0.194 | 532 |
| fear | 0.145 | 948 | 0.192 | 948 |
| sadness | 0.203 | 1099 | 0.202 | 1099 |
| something else | 0.230 | 1035 | 0.261 | 1035 |

### Genre-conditioned (thin sample, indicative only -- ~159 genre-labelled held-out paintings)

| genre | buddy top-1 acc | buddy n | PercepT top-1 acc | PercepT n |
|---|---:|---:|---:|---:|
| abstract_painting | 0.111 | 9 | 0.556 | 9 |
| cityscape | 0.182 | 11 | 0.182 | 11 |
| genre_painting | 0.333 | 21 | 0.048 | 21 |
| illustration | 0.000 | 5 | 0.200 | 5 |
| landscape | 0.545 | 33 | 0.242 | 33 |
| portrait | 0.103 | 39 | 0.205 | 39 |
| religious_painting | 0.000 | 18 | 0.000 | 18 |
| sketch_and_study | 0.533 | 15 | 0.200 | 15 |
| still_life | 0.000 | 8 | 0.125 | 8 |

## 5. Synthesis -- concrete next ideas

**Finding A: buddy's macro-AUC is significantly correlated with topic size
(Pearson r=0.531, p=0.019); PercepT's is too, more weakly (r=0.363,
p=0.021).** Buddy's three smallest held-out topics (16/17/18: 87/69/65
paintings) are almost certainly dragging macro AUC down disproportionately.
**Idea 1: give buddy's Stage 2 mapper minimum-occupancy handling** — either
merge/drop train communities below a size floor before freezing Stage 2
targets, or reweight the BCE loss to stop tiny topics from being
under-trained relative to their share of the macro-AUC average. This is
new, directly evidenced motivation for the "higher K" and "mapper
sweep" candidates already on the list — it specifically targets *why*
buddy's macro AUC might be capped, not just generic tuning.

**Finding B: buddy's top-1 accuracy is worse than PercepT's on 7 of 9
emotion categories, despite buddy's macro AUC edging PercepT's overall.**
Disgust is the starkest gap (buddy 0.087 vs. PercepT 0.329, n=416 each).
Buddy has fewer classes (19 vs. 40), so its random-guess floor is easier,
which makes this gap more concerning, not less. AUC (ranking quality) and
top-1 accuracy (exact-match) are different things, and buddy's near-tie on
the former does not carry over to the latter — the "buddy edges PercepT"
framing should not be read as buddy being uniformly better at Stage 2.
**Idea 2: this strengthens the case for a K sweep on buddy's own topic
count** (already a candidate) — PercepT's smaller, more numerous topics
look more content-homogeneous in the topic-characterization table (§1;
e.g. PercepT topic 39 is dominated by dark/fearful captions, topic 7 by
calm/peaceful ones), which plausibly explains its top-1 accuracy edge.
A finer buddy partition might close this specific gap.

**Finding C: cross-system NMI = 0.2947 — real but partial agreement.**
The per-topic mapping table (§3) shows buddy's coarser communities are
largely coherent *unions* of several PercepT topics along recognizable
genre/emotion lines (e.g. buddy 2, majority genre abstract_painting/
emotion "something else", maps onto PercepT 37/31/26 — all abstract- or
ambiguous-themed). This is reassuring: both systems are finding real,
overlapping structure, not unrelated noise. It also means buddy is not
simply missing what PercepT finds — it's finding the same structure at a
coarser grain, which is consistent with Finding B's suggestion that more
granularity (idea 2) could help.

**Finding D: buddy's topics are overwhelmingly "contentment"-majority (14
of 19) despite covering visually distinct genres, with uniformly high
emotion entropy (2.0-2.9 bits, near the ~3.17-bit max for 9 categories).**
This suggests buddy's graph is discovering mostly *content/genre*
structure, with emotion riding along only weakly — consistent with this
investigation's long-standing pattern of genre AMI being easier than
emotion AMI for buddy. PercepT's topics show more emotion-majority
variety (fear, sadness, awe, excitement each win several topics), which
may partly explain Finding B. **Idea 3 (more speculative, lower
priority): upweight the affect/emotion signal specifically in buddy's
graph construction**, rather than relying on it falling out as a
byproduct of genre-driven clustering. This is a deeper architectural
change, not a quick Stage 2 tweak, and should rank behind ideas 1-2 and
the existing candidates.

**Updated candidate list for the next implementation round** (supersedes
the earlier, pre-analysis list):

1. **Minimum-occupancy handling for buddy's Stage 2 targets** (new,
   Finding A) — cheapest, most directly evidenced.
2. **Buddy Stage 2 mapper LR/capacity sweep** (existing candidate,
   unchanged priority) — cheap, mirrors what worked for PercepT.
3. **Buddy topic-count (K) sweep** (existing candidate, priority raised by
   Finding B/C) — now has direct evidence it could close a real,
   documented top-1-accuracy gap, not just a speculative "worth trying."
4. **Richer multi-label Stage 2 targets for buddy** (existing candidate,
   unchanged priority) — still worth trying, now with mild extra support
   from the high within-topic emotion entropy in Finding D.
5. **Seed-stress buddy's Stage 2** (existing candidate, unchanged
   priority) — still needed for any of the above to be reported credibly,
   given the current +0.0053 margin is single-seed.
6. **Upweight affect signal in buddy's graph construction** (new, Finding
   D) — real but more speculative and costly; lowest priority, a
   follow-up if 1-5 don't close the gap.
