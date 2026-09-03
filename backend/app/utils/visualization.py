from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend.app.ai.bbox import MergedBox


def save_annotated_image(source_path: Path, boxes: list[MergedBox], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in boxes:
        x1 = box.x
        y1 = box.y
        x2 = box.x + box.w
        y2 = box.y + box.h
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=4)
        label = f"{box.label} {box.score:.2f} ({box.patch_count})"
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=(255, 40, 40))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    canvas.save(output_path, quality=95)
