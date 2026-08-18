"""Composite a background plate with character cutouts into a single first-frame image.

Used by :mod:`ltx_pipelines.character_scene_i2vid` to anchor multi-character identity: pasting
the actual character pixels into the frame-0 conditioning image locks their appearance in place,
instead of relying on the text prompt alone to keep it consistent for the whole clip.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageOps

DEFAULT_CHARACTER_SCALE = 0.75


@dataclass
class CharacterPlacement:
    path: str
    x_frac: float
    """Horizontal position of the character's center, as a fraction of canvas width (0-1)."""
    scale: float = DEFAULT_CHARACTER_SCALE
    """Character height as a fraction of canvas height."""


def evenly_spaced_x_fracs(count: int) -> list[float]:
    """Default horizontal centers for *count* characters, spread evenly across the frame."""
    return [(i + 1) / (count + 1) for i in range(count)]


def _cover_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize + center-crop to exactly (width, height), preserving aspect ratio (cover fit)."""
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return image.crop((left, top, left + width, top + height))


def _character_layer(image: Image.Image, target_height: int) -> Image.Image:
    """Return an RGBA cutout of *image* scaled to *target_height*.

    An image that already carries real transparency (e.g. exported from a background-removal
    tool) is used as-is -- this gives the cleanest composite and the most faithful identity
    anchor. An opaque image is treated as a full-frame photo and gets a feathered rectangular
    mask instead, so the paste blends into the background rather than showing a hard seam; a
    real cutout will still look better.
    """
    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in ("RGBA", "LA") and image.convert("RGBA").getchannel("A").getextrema() != (255, 255)
    image = image.convert("RGBA")
    src_w, src_h = image.size
    scale = target_height / src_h
    image = image.resize((max(1, round(src_w * scale)), target_height), Image.LANCZOS)

    if not has_alpha:
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        inset_x, inset_y = round(w * 0.04), round(h * 0.04)
        ImageDraw.Draw(mask).rectangle((inset_x, inset_y, w - inset_x, h - inset_y), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.02))
        image.putalpha(mask)

    return image


def compose_character_scene(
    background_path: str,
    characters: list[CharacterPlacement],
    width: int,
    height: int,
    output_path: str,
) -> str:
    """Composite a background plate with one or more character cutouts into a single image at
    exactly (width, height). Characters are bottom-anchored (feet on the background's ground
    plane) and horizontally centered at their ``x_frac``. The result is meant to be used as the
    frame-0 image conditioning input for an image-to-video pipeline.
    """
    canvas = _cover_resize(Image.open(background_path).convert("RGB"), width, height).convert("RGBA")

    for char in characters:
        target_h = max(1, round(height * char.scale))
        layer = _character_layer(Image.open(char.path), target_h)
        x = round(width * char.x_frac - layer.width / 2)
        x = max(0, min(x, width - layer.width)) if layer.width <= width else (width - layer.width) // 2
        y = height - layer.height
        canvas.alpha_composite(layer, (x, y))

    canvas.convert("RGB").save(output_path)
    return output_path
