"""Opt-in image preprocessing experiments; raw input remains the default."""

from typing import cast

import cv2
import numpy as np

from xray_workbench.vision.detector import ImageArray, PreprocessingMode


def preprocess_image(image: ImageArray, mode: PreprocessingMode) -> ImageArray:
    """Apply enhancement to uint8 grayscale or BGR image data."""
    if image.size == 0:
        raise ValueError("Image cannot be empty")
    if image.dtype != np.uint8:
        raise ValueError("Image must use uint8 pixel values")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Image must be grayscale or a three-channel BGR image")
    if mode is PreprocessingMode.RAW:
        return image

    enhanced = image
    if mode is PreprocessingMode.DENOISE_CLAHE:
        if enhanced.ndim == 2:
            enhanced = cast(ImageArray, cv2.fastNlMeansDenoising(enhanced))
        else:
            enhanced = cast(ImageArray, cv2.fastNlMeansDenoisingColored(enhanced))
    if enhanced.ndim == 2:
        return _apply_clahe(enhanced)

    lab = cast(ImageArray, cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB))
    lightness, a_channel, b_channel = cv2.split(lab)
    adjusted = cast(
        ImageArray,
        cv2.merge((_apply_clahe(cast(ImageArray, lightness)), a_channel, b_channel)),
    )
    return cast(ImageArray, cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR))


def _apply_clahe(channel: ImageArray) -> ImageArray:
    return cast(
        ImageArray,
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(channel),
    )
