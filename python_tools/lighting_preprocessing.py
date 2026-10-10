"""Experimental geometry-preserving inputs; no automatic model promotion."""
import numpy as np


def green_channel_rgb(frame):
    """Replicate decoded BGR's green channel into three equal channels.

    This produces a grayscale green-channel diagnostic input, not a green tint.
    It preserves source geometry, and cannot restore clipped/missing detail.
    Existing RGB models were not necessarily trained for this distribution.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]:
        raise ValueError('Nonempty uint8 BGR source image required')
    return np.ascontiguousarray(np.repeat(frame[:, :, 1:2], 3, axis=2))
