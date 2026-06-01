from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_classification_dino.data import ImagePathDataset, collate_images, discover_images
from image_classification_dino.model_loader import load_dino_model


def _select_embedding(outputs: object, strategy: str) -> torch.Tensor:
    if strategy == "cls":
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]
    if strategy == "patch_mean":
        return outputs.last_hidden_state[:, 1:].mean(dim=1)
    if strategy == "cls_patch_mean":
        cls = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0]
        patch_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
        return torch.cat([cls, patch_mean], dim=1)
    raise ValueError("embedding_strategy must be one of: cls, patch_mean, cls_patch_mean")


def embed_images(
    model_dir: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    embedding_strategy: str = "cls",
    normalize: bool = True,
) -> Path:
    image_paths = discover_images(image_dir)
    processor, model, resolved_device = load_dino_model(model_dir, device, dtype)

    dataset = ImagePathDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
        pin_memory=resolved_device.type == "cuda",
    )

    all_paths: list[str] = []
    all_embeddings: list[np.ndarray] = []

    for paths, images in tqdm(loader, desc="Embedding images"):
        inputs = processor(images=images, return_tensors="pt").to(resolved_device)
        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = _select_embedding(outputs, embedding_strategy)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        all_paths.extend(str(path) for path in paths)
        all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))

    matrix = np.concatenate(all_embeddings, axis=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=matrix,
        paths=np.array(all_paths, dtype=object),
        embedding_strategy=np.array(embedding_strategy),
        normalized=np.array(normalize),
    )

    metadata_output = output.with_suffix(".csv")
    pd.DataFrame({"image_path": all_paths, "embedding_index": range(len(all_paths))}).to_csv(
        metadata_output,
        index=False,
    )
    return output
