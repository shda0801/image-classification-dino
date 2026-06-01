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


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype | None:
    if dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be one of: auto, float32, float16, bfloat16")


def load_dino_model(
    model_dir: str | Path,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[AutoImageProcessor, AutoModel, torch.device]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model_kwargs = {"local_files_only": True}
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device
