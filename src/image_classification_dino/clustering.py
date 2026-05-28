from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score


def load_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(Path(path), allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    paths = [str(path) for path in data["paths"].tolist()]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    if len(paths) != embeddings.shape[0]:
        raise ValueError("Number of paths does not match number of embeddings.")
    return embeddings, paths


def cluster_embeddings(
    embeddings_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "kmeans",
    n_clusters: int = 8,
    random_state: int = 42,
) -> Path:
    embeddings, paths = load_embeddings(embeddings_path)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "minibatch-kmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN()
    else:
        raise ValueError("method must be one of: kmeans, minibatch-kmeans, agglomerative, dbscan")

    labels = model.fit_predict(embeddings)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "image_path": paths,
            "cluster": labels,
        }
    )
    frame.to_csv(output, index=False)

    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) > 1 and -1 not in unique_labels:
        score = silhouette_score(embeddings, labels, metric="cosine")
        summary = pd.DataFrame(
            [{"method": method, "n_clusters": len(unique_labels), "silhouette_cosine": score}]
        )
        summary.to_csv(output.with_suffix(".summary.csv"), index=False)

    return output
