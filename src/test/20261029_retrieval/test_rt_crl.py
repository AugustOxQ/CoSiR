import rt_crl

NUMBERED = """1. Joy
2) Calm

3. joy
 4.  Deep Sadness
10: Awe
"""
BULLETED = """- Baroque
* BAROQUE
• Pop Art

  - Cubism
"""


def test_parse_numbered():
    v, note = rt_crl.parse(NUMBERED)
    assert v == ['joy', 'calm', 'deep sadness', 'awe'] and 'fewer than 40' in note


def test_parse_bulleted_and_cap():
    assert rt_crl.parse(BULLETED)[0] == ['baroque', 'pop art', 'cubism']
    v, note = rt_crl.parse('\n'.join(f'{i}. v{i}' for i in range(1, 61)))
    assert len(v) == 50 and v[-1] == 'v50' and note == ''
