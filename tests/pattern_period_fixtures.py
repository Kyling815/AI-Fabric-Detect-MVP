"""Deterministic synthetic inputs, never sourced from or written to locked data."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def pattern(kind: str) -> np.ndarray:
    y, x = np.indices((288, 320))
    checker = 40 + 180 * ((x // 8 + y // 12) % 2)
    if kind == "checker":
        values = checker
    elif kind == "stripes":
        values = 40 + 180 * ((x // 8) % 2)
    elif kind == "horizontal_stripes":
        values = 40 + 180 * ((y // 12) % 2)
    elif kind == "dots":
        values = 40 + 180 * ((x % 20 < 5) & (y % 24 < 5))
    elif kind == "uniform":
        values = np.full_like(x, 150)
    elif kind == "noise":
        values = np.random.default_rng(17).integers(0, 256, size=x.shape)
    elif kind == "stretch":
        values = cv2.resize(checker.astype(np.uint8), (480, 360), interpolation=cv2.INTER_NEAREST)
    elif kind == "lighting":
        values = np.clip(checker + x * 0.06 + y * 0.04, 0, 255)
    elif kind == "rotate":
        values = cv2.warpAffine(checker.astype(np.uint8), cv2.getRotationMatrix2D((160, 144), 12, 1),
                                (320, 288), borderMode=cv2.BORDER_REFLECT)
    else:
        raise ValueError(kind)
    return values.astype(np.uint8)


def png_bytes(kind: str = "checker") -> bytes:
    buffer = BytesIO()
    Image.fromarray(pattern(kind)).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
