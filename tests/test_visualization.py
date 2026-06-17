from pathlib import Path

from PIL import Image

from backend.app.ai.bbox import MergedBox
from backend.app.utils.visualization import save_annotated_image


def test_save_annotated_image_writes_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    output_path = tmp_path / "annotated.jpg"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(source_path)

    save_annotated_image(
        source_path,
        [MergedBox(x=10, y=10, w=40, h=30, score=0.8)],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
