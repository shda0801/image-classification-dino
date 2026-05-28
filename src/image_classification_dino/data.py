from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def discover_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    if not paths:
        raise ValueError(f"No supported images found under: {root}")
    return paths


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Iterable[Path]) -> None:
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Path, Image.Image]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            return path, image.convert("RGB")


def collate_images(batch: list[tuple[Path, Image.Image]]) -> tuple[list[Path], list[Image.Image]]:
    paths, images = zip(*batch, strict=True)
    return list(paths), list(images)
