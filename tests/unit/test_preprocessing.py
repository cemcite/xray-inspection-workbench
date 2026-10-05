import numpy as np
import pytest

from xray_workbench.vision.detector import PreprocessingMode
from xray_workbench.vision.preprocessing import preprocess_image


def test_raw_mode_returns_the_original_image() -> None:
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))

    processed = preprocess_image(image, PreprocessingMode.RAW)

    assert processed is image


@pytest.mark.parametrize(
    "mode",
    [PreprocessingMode.CLAHE, PreprocessingMode.DENOISE_CLAHE],
)
@pytest.mark.parametrize("channels", [None, 3])
def test_enhancement_preserves_dimensions_and_uint8(mode, channels) -> None:
    shape = (32, 32) if channels is None else (32, 32, channels)
    image = np.random.default_rng(7).integers(0, 256, size=shape, dtype=np.uint8)

    processed = preprocess_image(image, mode)

    assert processed.shape == image.shape
    assert processed.dtype == np.uint8


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.empty((0, 0), dtype=np.uint8), "cannot be empty"),
        (np.zeros((4, 4), dtype=np.float32), "uint8"),
        (np.zeros((4, 4, 4), dtype=np.uint8), "grayscale or a three-channel"),
    ],
)
def test_rejects_unsupported_image_input(image, message) -> None:
    with pytest.raises(ValueError, match=message):
        preprocess_image(image, PreprocessingMode.CLAHE)
