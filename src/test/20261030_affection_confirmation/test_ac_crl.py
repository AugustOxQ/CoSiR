import json

import numpy as np
import pytest

import ac_crl
import ac_paths


def test_vectors_against_float64_reference():
    img = np.array([[1, 0, 0], [0.6, 0.8, 0]], np.float32)
    text = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], np.float32)
    # by hand: row 0 dots (1, 0, 0.6), row 1 dots (0.6, 0.8, 0.36 + 0.64 = 1.0 within float32 rounding of the inputs)
    ref = np.array([[1, 0, 0.6], [0.6, 0.8, 1.0]], np.float64)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    got = ac_crl.vectors(img, text)
    assert got.dtype == np.float32 and got.shape == (2, 3)
    assert np.allclose(got, ref, atol=1e-6)
    assert np.allclose(np.linalg.norm(got.astype(np.float64), axis=1), 1, atol=1e-6)


def test_vectors_is_r11_form_bitwise():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(7, 16)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    text = rng.normal(size=(5, 16)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    x = img.astype(np.float64) @ text.astype(np.float64).T          # rt_data.raw('crl')
    x = x.astype(np.float64)                                        # rt_data.embed: float64 normalize, one cast
    r11 = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    assert ac_crl.vectors(img, text).tobytes() == r11.tobytes()


def test_vectors_rejects_width_mismatch():
    with pytest.raises(AssertionError):
        ac_crl.vectors(np.zeros((2, 4), np.float32), np.zeros((3, 5), np.float32))


def test_template_rendering():
    assert ac_crl.template('painting') == 'a painting with the {aspect} of {value}'
    assert ac_crl.template('photo') == 'a photo with the {aspect} of {value}'
    assert ac_crl.render('photo', 'emotion', ['joy', 'dread']) == [
        'a photo with the emotion of joy', 'a photo with the emotion of dread']
    with pytest.raises(AssertionError):
        ac_crl.template('sketch')


@pytest.mark.real_data
def test_painting_rendering_matches_r11_sidecar_and_ac_data_check():
    import rt_crl
    for aspect in ac_paths.ASPECTS:
        side = json.loads((ac_paths.R11_OUT / f'crl/{aspect}_text.json').read_text())
        assert ac_crl.render('painting', aspect, ac_crl.frozen_values(aspect)) == side['texts']
        assert ac_crl.template('painting') == rt_crl.TEMPLATE
