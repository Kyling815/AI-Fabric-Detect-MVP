"""Synthetic capture fixtures; no production dataset/model dependencies."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def capture_pixels(kind: str, width: int = 128, height: int = 96) -> np.ndarray:
    yy, xx = np.indices((height, width))
    gray = (((xx // 4 + yy // 4) % 2) * 100 + 80).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    if kind == "black":
        image[:] = 0
    elif kind == "white":
        image[:] = 255
    elif kind == "blurred":
        image = cv2.GaussianBlur(image, (31, 31), 6)
    elif kind != "valid":
        raise ValueError(kind)
    return image


def png_bytes(kind: str, width: int = 128, height: int = 96) -> bytes:
    buffer = BytesIO()
    Image.fromarray(capture_pixels(kind, width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def write_capture(root: Path, kind: str, width: int = 128, height: int = 96) -> Path:
    path = root / f"{kind}_{width}_{height}.png"
    path.write_bytes(png_bytes(kind, width, height))
    return path


def corrupt_png_checksum(content: bytes) -> bytes:
    payload = bytearray(content)
    kind_offset = payload.index(b"IDAT")
    chunk_size = int.from_bytes(payload[kind_offset - 4:kind_offset], "big")
    payload[kind_offset + 4 + chunk_size] ^= 1
    return bytes(payload)
