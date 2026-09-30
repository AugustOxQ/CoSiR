# PercepT topic qualitative check

Generated automatically, 2026-09-23 03:28:13.

## Stage-1 K=60/40 seed-42 reproduction

| metric | established | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1238 | 0.1238 | 0.0000 | reproduced |
| held-out genre AMI | 0.2617 | 0.2617 | 0.0000 | reproduced |

The target is the frozen Stage-2 rule: argmax topic plus every topic whose DEC assignment satisfies `q > 1.2/40`. The five topics below are sampled without replacement with `random.seed(42)`: 7, 1, 17, 15, 14.

## Topic 7

- Single-labeled core members: **237**
- Secondary memberships: **4,163**

### Single-labeled example captions

- `abidin-dino_unknown-title-2`: I feel like I'm walking down a soothing road.
- `ad-reinhardt_abstract-painting-1966-1`: The dark blue reminds me of depression and there is no end in sight where there are no other colors.
- `adolf-hitler_still-life-with-bottle-and-fruit`: The painting is a bit dull. It needs more colors to liven it up.
- `agnes-martin_mountain-i-1966`: The soft green color makes me feel very calm.
- `agnes-martin_untitled-1977-2`: annoying, make  me state hard to see the shades

### Secondary-membership example captions and primary-topic comparisons

#### Secondary example 1: topic 7 with primary topic 20

- `a.y.-jackson_grey-day-laurentians-1928`: There is a dynamic to the way the hills and sky are painted.

Primary topic 20's single-labeled comparison examples (population: 400):

- `abraham-manievich_unknown-title-1`: winter trail thru snow covered trees, wow!
- `adolphe-joseph-thomas-monticelli_farmyard-with-donkeys-and-roosters`: It looks like an abandoned cabin out in the woods.

#### Secondary example 2: topic 7 with primary topic 8

- `aaron-siskind_acolman-1-1955`: I like the woods and as this looks like a graveyard, it soothes.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 3: topic 7 with primary topic 8

- `aaron-siskind_chicago-6-1961`: The black sharp lines running throughout the composition is jarring and uneasy.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 4: topic 7 with primary topic 8

- `aaron-siskind_new-york-2-1948`: i dont eeven know what that is supposed to represent

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 5: topic 7 with primary topic 8

- `aaron-siskind_new-york-city-w-1-1947`: Dark pitch black background with jagged gray

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

## Topic 1

- Single-labeled core members: **292**
- Secondary memberships: **2,704**

### Single-labeled example captions

- `adriaen-van-de-venne_emblem`: Dark colors gives a sad feeling. Person checking on her from the door makes me think she is alone.
- `adriaen-van-ostade_cutting-the-feather`: He looks mad and uninviting.
- `agnolo-bronzino_piero-de-medici-il-gottoso`: the face is not interesting has a bland expression
- `agnolo-bronzino_portrait-of-a-gentleman`: The glance off to the side and figurine in the background give me a feeling of unease.
- `agnolo-bronzino_portrait-of-a-sculptor`: confusion, angel's position makes me wonder in opposing viewpoints

### Secondary-membership example captions and primary-topic comparisons

#### Secondary example 1: topic 1 with primary topic 3

- `agnolo-bronzino_garcia-de-medici`: The kid looks rough in the picture

Primary topic 3's single-labeled comparison examples (population: 365):

- `adam-baltatu_still-life-with-travel-props`: The objects that are resting on a lighter tan table which makes me feel content.
- `adam-baltatu_the-needlewoman`: The room is well-lit. The woman appears to be sewing on a garment. I feel contentment because even if it's a chore she seems relaxed.

#### Secondary example 2: topic 1 with primary topic 3

- `agnolo-bronzino_pietro-de-medici-1`: The face looks very sad and mournful.

Primary topic 3's single-labeled comparison examples (population: 365):

- `adam-baltatu_still-life-with-travel-props`: The objects that are resting on a lighter tan table which makes me feel content.
- `adam-baltatu_the-needlewoman`: The room is well-lit. The woman appears to be sewing on a garment. I feel contentment because even if it's a chore she seems relaxed.

#### Secondary example 3: topic 1 with primary topic 0

- `agnolo-bronzino_pope-leo-x`: Expression on face looks very determined

Primary topic 0's single-labeled comparison examples (population: 135):

- `adriaen-brouwer_the-smoker`: A lot of work in the painting, looks real.
- `adriaen-van-de-venne_a-man-carrying-a-sack`: because i like art with detail. Interesting to try and understand.

#### Secondary example 4: topic 1 with primary topic 11

- `agnolo-bronzino_portrait-of-a-young-man`: The way the figure's left hand is gripping his leg suggests apprehension.

Primary topic 11's single-labeled comparison examples (population: 201):

- `adriaen-brouwer_fumatore`: I'm left feeling amused by the funny expression the man is shown having in the picture.
- `agnolo-bronzino_cosimo-de-medici`: He is someone important, and love the red clothing and details

#### Secondary example 5: topic 1 with primary topic 0

- `agnolo-bronzino_portrait-of-a-young-man-with-book`: Man holding a book confidently with his hand on his side with a smug look on his face

Primary topic 0's single-labeled comparison examples (population: 135):

- `adriaen-brouwer_the-smoker`: A lot of work in the painting, looks real.
- `adriaen-van-de-venne_a-man-carrying-a-sack`: because i like art with detail. Interesting to try and understand.

## Topic 17

- Single-labeled core members: **258**
- Secondary memberships: **4,071**

### Single-labeled example captions

- `ad-reinhardt_ho-to-look-at-an-artist`: I feel like this piece is supposed to evoke happiness or laughter. It appears to be a comical piece.
- `albert-gleizes_femmes-cousant-1913`: The people are working and innovating with the past behind them and a way forward.
- `albert-gleizes_football-players-1912`: It takes close examintion to see the humans. Interesting.
- `albert-gleizes_two-women-seated-by-a-window-1914`: It's as if the picture is hidden. There is a lot of intrigue to it.
- `albert-marquet_carnival-on-the-beach-1906`: How people are painted in black without face

### Secondary-membership example captions and primary-topic comparisons

#### Secondary example 1: topic 17 with primary topic 8

- `aaron-siskind_chicago-6-1961`: The black sharp lines running throughout the composition is jarring and uneasy.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 2: topic 17 with primary topic 7

- `aaron-siskind_gloucester-16a-1944`: It makes me feel uncomfortable because I can't tell if it is a face or something else.

Primary topic 7's single-labeled comparison examples (population: 237):

- `abidin-dino_unknown-title-2`: I feel like I'm walking down a soothing road.
- `ad-reinhardt_abstract-painting-1966-1`: The dark blue reminds me of depression and there is no end in sight where there are no other colors.

#### Secondary example 3: topic 17 with primary topic 7

- `aaron-siskind_jerome-arizona-1949`: looks like pealing paint. YUCK

Primary topic 7's single-labeled comparison examples (population: 237):

- `abidin-dino_unknown-title-2`: I feel like I'm walking down a soothing road.
- `ad-reinhardt_abstract-painting-1966-1`: The dark blue reminds me of depression and there is no end in sight where there are no other colors.

#### Secondary example 4: topic 17 with primary topic 7

- `aaron-siskind_kentucky-4-1951`: looks like an old sign so i do not consider this art

Primary topic 7's single-labeled comparison examples (population: 237):

- `abidin-dino_unknown-title-2`: I feel like I'm walking down a soothing road.
- `ad-reinhardt_abstract-painting-1966-1`: The dark blue reminds me of depression and there is no end in sight where there are no other colors.

#### Secondary example 5: topic 17 with primary topic 7

- `aaron-siskind_new-york-1951`: The darkness of the majority of the painting makes the other parts of it stand out in a disorienting way

Primary topic 7's single-labeled comparison examples (population: 237):

- `abidin-dino_unknown-title-2`: I feel like I'm walking down a soothing road.
- `ad-reinhardt_abstract-painting-1966-1`: The dark blue reminds me of depression and there is no end in sight where there are no other colors.

## Topic 15

- Single-labeled core members: **291**
- Secondary memberships: **4,120**

### Single-labeled example captions

- `adam-baltatu_dobrujan-landscape`: It looks like their hiding from something coming from over the horizon.
- `albrecht-durer_apostle-bartholomew`: I feel excitement as I think the subject matter is interesting and want to know the artist
- `albrecht-durer_coat-of-arms-with-open-man-behind`: The person in the castle looks like he's daydreaming and content even though the big bird looks he's carrying him away.
- `albrecht-durer_handestudien`: These hands look powerful like they can create or fix anything.
- `albrecht-durer_madonna-and-child`: I love the detail on the lower half of the drawing, setting it off from the top half.

### Secondary-membership example captions and primary-topic comparisons

#### Secondary example 1: topic 15 with primary topic 8

- `aaron-siskind_new-york-2-1948`: i dont eeven know what that is supposed to represent

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 2: topic 15 with primary topic 8

- `aaron-siskind_new-york-24-1988`: the hand prints are creepy and horrifying.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 3: topic 15 with primary topic 8

- `aaron-siskind_new-york-city-w-1-1947`: Dark pitch black background with jagged gray

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 4: topic 15 with primary topic 8

- `abidin-dino_abstract-composition`: Indifference. It looks faded, muted. Maybe just a bad photo of it. Looks like a T-shirt.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

#### Secondary example 5: topic 15 with primary topic 8

- `abidin-dino_drawing-pain-self-portrait-1967`: The painting appears to depict a hospital patient.

Primary topic 8's single-labeled comparison examples (population: 327):

- `adolphe-joseph-thomas-monticelli_the-adoration-of-the-magi`: the primary figure appears to be melting into his surroundings
- `albert-gleizes_man-on-a-balcony-portrait-of-dr-th-o-morinaud-1912`: this man seems to have so much power that it upsets him

## Topic 14

- Single-labeled core members: **130**
- Secondary memberships: **2,398**

### Single-labeled example captions

- `abraham-manievich_still-life-with-fruits`: i think it shows the appreciation of apples and a fruit bowl
- `albert-gleizes_portrait-de-jacques-nayral-1911`: he looks like a business man ready to take on the world.
- `albrecht-durer_lion`: I feel indifferent to this picture.  It looks like it would be great if it were finished, but it looks undone.
- `alexandre-benois_self-portrait(2)`: Curiosity.  This picture makes me wonder what is he thinking about at this exact moment.  What could he be pondering and who is he looking at, at this exact moment.
- `amedeo-modigliani_portrait-of-celso-lagar-1915`: I like the abstrac and impressionisim.  The face looks like he is curious about what is just out of range of the picture.

### Secondary-membership example captions and primary-topic comparisons

#### Secondary example 1: topic 14 with primary topic 11

- `agnolo-bronzino_alessandro-de-medici`: The color are easy to look at in this picture and that makes me feel calm.

Primary topic 11's single-labeled comparison examples (population: 201):

- `adriaen-brouwer_fumatore`: I'm left feeling amused by the funny expression the man is shown having in the picture.
- `agnolo-bronzino_cosimo-de-medici`: He is someone important, and love the red clothing and details

#### Secondary example 2: topic 14 with primary topic 0

- `agnolo-bronzino_pope-leo-x`: Expression on face looks very determined

Primary topic 0's single-labeled comparison examples (population: 135):

- `adriaen-brouwer_the-smoker`: A lot of work in the painting, looks real.
- `adriaen-van-de-venne_a-man-carrying-a-sack`: because i like art with detail. Interesting to try and understand.

#### Secondary example 3: topic 14 with primary topic 0

- `agnolo-bronzino_portrait-of-a-young-man-with-book`: Man holding a book confidently with his hand on his side with a smug look on his face

Primary topic 0's single-labeled comparison examples (population: 135):

- `adriaen-brouwer_the-smoker`: A lot of work in the painting, looks real.
- `adriaen-van-de-venne_a-man-carrying-a-sack`: because i like art with detail. Interesting to try and understand.

#### Secondary example 4: topic 14 with primary topic 1

- `agnolo-bronzino_portrait-of-cosimo-i-de-medici-1`: The man seems to glare at something in disgust as he looks up and to the side.

Primary topic 1's single-labeled comparison examples (population: 292):

- `adriaen-van-de-venne_emblem`: Dark colors gives a sad feeling. Person checking on her from the door makes me think she is alone.
- `adriaen-van-ostade_cutting-the-feather`: He looks mad and uninviting.

#### Secondary example 5: topic 14 with primary topic 0

- `agostino-carracci_italian-scientist-ulisse-aldrovandi`: The man eludes intelligence, gives a feeling of great wisdom.

Primary topic 0's single-labeled comparison examples (population: 135):

- `adriaen-brouwer_the-smoker`: A lot of work in the painting, looks real.
- `adriaen-van-de-venne_a-man-carrying-a-sack`: because i like art with detail. Interesting to try and understand.

## Qualitative reading

This five-topic sample is a caption-level observation, not a statistical test. The paired examples should be read for recurring shared subject matter, style, or mood language between each secondary member and the topic's single-labeled core. A mixed or unclear pattern is itself the honest outcome here: neither apparent overlap nor apparent loose affinity in this small sample establishes that all threshold-derived memberships are semantically meaningful.
