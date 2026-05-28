from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def load_dino_model(model_dir: str | Path, device: str = "auto") -> tuple[AutoImageProcessor, AutoModel, torch.device]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    resolved_device = resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device
