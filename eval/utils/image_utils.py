import math
import base64
from io import BytesIO
from PIL import Image, ImageOps
from pathlib import Path
from typing import Union


ImageLike = Union[str, Path, Image.Image, bytes, bytearray, memoryview]


def _to_pil(img: ImageLike) -> Image.Image:
    """Convert path, PIL.Image, or raw bytes to RGB PIL.Image with EXIF orientation handling."""
    if isinstance(img, Image.Image):
        im = img
    elif isinstance(img, (bytes, bytearray, memoryview)):
        im = Image.open(BytesIO(bytes(img)))
    elif isinstance(img, (str, Path)):
        s = str(img)
        if s.startswith("data:image/"):
            b64 = s.split(",", 1)[1]
            im = Image.open(BytesIO(base64.b64decode(b64)))
        else:
            im = Image.open(s)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    im.load()
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def check_and_resize_image(
    img: ImageLike,
    max_pixels: int = 512 * 512,
    min_pixels: int = 338 * 338,
) -> Image.Image:
    """Resize image to fit within [min_pixels, max_pixels] while preserving aspect ratio."""
    im = _to_pil(img)
    w, h = im.width, im.height
    pixels = w * h

    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        w, h = im.width, im.height
        pixels = w * h

    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    return im


def encode_image(img: ImageLike) -> str:
    """Encode image to base64 JPEG string."""
    im = check_and_resize_image(img)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
