"""Small CPU checks against only the prescribed validation stores."""
import numpy as np

import tx_data as d
import tx_paths as p
import rt_data


def test_caption_count_and_pid_join():
    c = d.val_captions()
    assert len(c['sample_id']) == p.N_CAPTIONS
    assert (np.diff(c['sample_id']) > 0).all()
    assert np.array_equal(np.unique(c['pid']), np.arange(p.N_PAINTINGS))
    index = d.read_json(d.FULL / 'qwen3vl_emb_2b/index.json')
    names = rt_data.names('val')
    for i in range(0, len(c['pid']), 211):
        row = index['texts'][c['store_row'][i]]
        assert row['sample_id'] == c['sample_id'][i] and row['painting'] == names[c['pid'][i]]


def test_own_emotion_plurality_and_inherited_labels():
    c, captions, paintings = d.val_captions(), d.caption_labels(), d.painting_labels()
    for pid in range(0, p.N_PAINTINGS, 47):
        own = c['emotion'][c['pid'] == pid]
        counts = np.bincount(own, minlength=9)
        if (counts == counts.max()).sum() == 1:
            assert counts.argmax() == paintings['emotion'][pid]
    assert np.array_equal(captions['emotion'], c['emotion'])
    for a in ('style', 'genre'):
        np.testing.assert_array_equal(captions[a], paintings[a][c['pid']])


def test_tables_words_and_leaks():
    for a, n in [('emotion', 8), ('style', 27), ('genre', 10)]:
        assert len(d.class_words(a)) == n
        flag = d.names_label(a)
        assert flag.shape == (p.N_CAPTIONS,) and flag.dtype == bool
    assert set(d.label_names('style')) == set(d.STYLE_PATTERNS)
    assert set(d.label_names('genre')) == set(d.GENRE_PATTERNS)
    assert d.emotion_named('This is scary and sad.', d.label_names('emotion').index('sadness'))
    assert not d.emotion_named('This is scary and sad.', d.label_names('emotion').index('something else'))


def test_real_store_rows_are_joined_unit_float32():
    for name in ('plain', 'clip_l14', 'siglip2_so400m', 'siglip_so400m', 'roberta'):
        x = d.text_rows(name)
        assert x.shape[0] == p.N_CAPTIONS and x.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-6)
    x = d.text_rows('teacher', 'emotion')
    assert x.shape == (p.N_CAPTIONS, 2048)


def test_caption_mean_preserves_raw_row_weights():
    x = np.asarray([[4, 0], [0, 1], [1, 1]], dtype=np.float32)
    pid = np.asarray([0, 0, 1], dtype=np.int64)
    got = d.caption_mean(x, pid, 2)
    expected = d.unit(np.asarray([[2, 0.5], [1, 1]], dtype=np.float32))
    np.testing.assert_allclose(got, expected)
    assert not np.allclose(got[0], d.unit(x[:2]).mean(0) / np.linalg.norm(d.unit(x[:2]).mean(0)))
