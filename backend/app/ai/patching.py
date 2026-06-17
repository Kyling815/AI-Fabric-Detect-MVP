from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class PatchBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class PatchGrid:
    image_width: int
    image_height: int
    patch_size: int
    patch_stride: int
    boxes: tuple[PatchBox, ...]

    @property
    def rows(self) -> int:
        return len({box.y for box in self.boxes})

    @property
    def cols(self) -> int:
        return len({box.x for box in self.boxes})

    @property
    def patch_count(self) -> int:
        return len(self.boxes)

    @property
    def coverage_ratio(self) -> float:
        if self.image_width <= 0 or self.image_height <= 0:
            return 0.0
        covered_x = {x for box in self.boxes for x in range(box.x, box.x + box.w)}
        covered_y = {y for box in self.boxes for y in range(box.y, box.y + box.h)}
        return (len(covered_x) * len(covered_y)) / (self.image_width * self.image_height)


@dataclass(frozen=True)
class ImagePatch:
    box: PatchBox
    image: Image.Image


def build_patch_grid(
    image_width: int,
    image_height: int,
    patch_size: int,
    patch_stride: int,
    include_edges: bool = True,
) -> PatchGrid:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")
    if patch_stride <= 0:
        raise ValueError("Patch stride must be positive.")

    effective_width = min(patch_size, image_width)
    effective_height = min(patch_size, image_height)
    x_positions = _positions(image_width, effective_width, patch_stride, include_edges)
    y_positions = _positions(image_height, effective_height, patch_stride, include_edges)
    boxes = tuple(
        PatchBox(x=x, y=y, w=effective_width, h=effective_height)
        for y in y_positions
        for x in x_positions
    )
    return PatchGrid(
        image_width=image_width,
        image_height=image_height,
        patch_size=effective_width,
        patch_stride=patch_stride,
        boxes=boxes,
    )


def extract_image_patches(image: Image.Image, grid: PatchGrid) -> list[ImagePatch]:
    if image.width != grid.image_width or image.height != grid.image_height:
        raise ValueError("Image size does not match patch grid dimensions.")

    patches: list[ImagePatch] = []
    for box in grid.boxes:
        crop = image.crop((box.x, box.y, box.x + box.w, box.y + box.h))
        patches.append(ImagePatch(box=box, image=crop))
    return patches


def _positions(length: int, window: int, stride: int, include_edges: bool) -> list[int]:
    if length <= window:
        return [0]

    positions = list(range(0, length - window + 1, stride))
    edge_position = length - window
    if include_edges and positions[-1] != edge_position:
        positions.append(edge_position)
    return positions
