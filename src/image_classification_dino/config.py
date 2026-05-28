from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClusterConfig:
    method: str = "kmeans"
    n_clusters: int = 8
    random_state: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    model_dir: Path
    image_dir: Path
    embeddings_output: Path = Path("outputs/embeddings.npz")
    clusters_output: Path = Path("outputs/clusters.csv")
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 0
    embedding_strategy: str = "cls"
    normalize: bool = True
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cluster_raw = raw.pop("cluster", {}) or {}
    raw["cluster"] = ClusterConfig(**cluster_raw)

    for key in ("model_dir", "image_dir", "embeddings_output", "clusters_output"):
        if key in raw:
            raw[key] = Path(raw[key])

    return PipelineConfig(**raw)


def merge_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    values = config.__dict__.copy()
    cluster = values.pop("cluster")
    cluster_values = cluster.__dict__.copy()

    for key, value in overrides.items():
        if value is None:
            continue
        if key.startswith("cluster."):
            cluster_values[key.split(".", 1)[1]] = value
        else:
            values[key] = value

    values["cluster"] = ClusterConfig(**cluster_values)
    return PipelineConfig(**values)
