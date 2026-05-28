# Image Classification DINO

Local-first image embedding and clustering pipeline for DINOv3 feature extractors.

The default workflow is:

1. Put a downloaded Hugging Face model snapshot on your machine.
2. Convert unlabeled images to embeddings.
3. Cluster embeddings and export a CSV.

No model or data path is hard-coded. The loader uses `local_files_only=True`, so it will not download model files at runtime.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For CUDA, install the PyTorch build that matches your driver/CUDA environment before installing this package.

## Expected Local Model Layout

Keep the Hugging Face snapshot structure:

```text
/secure/models/dinov3-vitb16-pretrain-lvd1689m/
  config.json
  model.safetensors
  preprocessor_config.json
  ...
```

Do not commit this directory.

## Embed Images

```bash
dino-images embed \
  --model-dir /secure/models/dinov3-vitb16-pretrain-lvd1689m \
  --image-dir /secure/data/timeseries_images \
  --output outputs/embeddings.npz \
  --batch-size 8 \
  --device auto
```

Outputs:

- `outputs/embeddings.npz`: embedding matrix and image paths
- `outputs/embeddings.csv`: path-level metadata

## Cluster Embeddings

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters.csv \
  --method kmeans \
  --n-clusters 8
```

For larger datasets:

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters.csv \
  --method minibatch-kmeans \
  --n-clusters 32
```

### Clustering Methods

`dino-images cluster` currently supports four methods.

| Method | CLI value | When to use | Required choice |
| --- | --- | --- | --- |
| KMeans | `kmeans` | Good default when you roughly know the number of groups. | `--n-clusters` |
| MiniBatch KMeans | `minibatch-kmeans` | Better for larger datasets where standard KMeans is slow. | `--n-clusters` |
| Agglomerative | `agglomerative` | Useful for smaller datasets when hierarchical structure matters. | `--n-clusters` |
| DBSCAN | `dbscan` | Useful when clusters may have irregular shapes or outliers. | No cluster count |

Examples:

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters_kmeans.csv \
  --method kmeans \
  --n-clusters 8
```

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters_minibatch.csv \
  --method minibatch-kmeans \
  --n-clusters 32
```

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters_agglomerative.csv \
  --method agglomerative \
  --n-clusters 8
```

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters_dbscan.csv \
  --method dbscan
```

For `kmeans`, `minibatch-kmeans`, and `agglomerative`, try several values of `--n-clusters` and compare the exported `*.summary.csv` silhouette score. The score is only a rough guide; final cluster quality should be checked by visually inspecting sample images from each cluster.

### CLI Reference

Embed command:

```bash
dino-images embed \
  --model-dir MODEL_DIR \
  --image-dir IMAGE_DIR \
  --embeddings-output outputs/embeddings.npz \
  --device auto \
  --batch-size 8 \
  --num-workers 0 \
  --embedding-strategy cls
```

Important embed options:

| Option | Meaning |
| --- | --- |
| `--model-dir` | Local Hugging Face model snapshot directory. |
| `--image-dir` | Directory containing unlabeled images. Searched recursively. |
| `--embeddings-output` / `--output` | Output `.npz` embedding file. A sidecar `.csv` is also written. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size` | Increase on stronger GPUs; lower it on weak CPU/GPU machines. |
| `--num-workers` | DataLoader workers. Start with `0`; increase on HPC if image loading is slow. |
| `--embedding-strategy` | `cls`, `patch_mean`, or `cls_patch_mean`. Start with `cls`. |
| `--no-normalize` | Disable L2 normalization. Default is normalized embeddings. |

Cluster command:

```bash
dino-images cluster \
  --embeddings outputs/embeddings.npz \
  --output outputs/clusters.csv \
  --method kmeans \
  --n-clusters 8 \
  --random-state 42
```

Important cluster options:

| Option | Meaning |
| --- | --- |
| `--embeddings` | Input `.npz` file created by `dino-images embed`. |
| `--output` | Output cluster assignment CSV. |
| `--method` | `kmeans`, `minibatch-kmeans`, `agglomerative`, or `dbscan`. |
| `--n-clusters` | Number of clusters for methods that require it. |
| `--random-state` | Reproducibility seed for supported methods. |

## One-Shot Pipeline

```bash
dino-images run \
  --model-dir /secure/models/dinov3-vitb16-pretrain-lvd1689m \
  --image-dir /secure/data/timeseries_images \
  --embeddings-output outputs/embeddings.npz \
  --clusters-output outputs/clusters.csv \
  --n-clusters 8
```

## Hardware Notes

- Low-end CPU/GPU: start with `--batch-size 1` or `--batch-size 4`.
- NVIDIA GPU: use `--device cuda` or `--device auto`.
- Supercomputer/HPC GPU: keep the same commands and adjust `--batch-size`, `--num-workers`, and scheduler paths.
