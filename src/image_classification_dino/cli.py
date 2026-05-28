from __future__ import annotations

import argparse
from pathlib import Path


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--embedding-strategy",
        choices=["cls", "patch_mean", "cls_patch_mean"],
        default=None,
    )
    parser.add_argument("--no-normalize", action="store_true")


def _config_from_args(args: argparse.Namespace):
    from image_classification_dino.config import PipelineConfig, load_config, merge_overrides

    if args.config:
        config = load_config(args.config)
    else:
        if not args.model_dir or not args.image_dir:
            raise SystemExit("--model-dir and --image-dir are required when --config is not provided.")
        config = PipelineConfig(model_dir=Path(args.model_dir), image_dir=Path(args.image_dir))

    overrides = {
        "model_dir": Path(args.model_dir) if getattr(args, "model_dir", None) else None,
        "image_dir": Path(args.image_dir) if getattr(args, "image_dir", None) else None,
        "embeddings_output": Path(args.embeddings_output) if getattr(args, "embeddings_output", None) else None,
        "clusters_output": Path(args.clusters_output) if getattr(args, "clusters_output", None) else None,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "embedding_strategy": args.embedding_strategy,
        "normalize": False if args.no_normalize else None,
        "cluster.method": getattr(args, "method", None),
        "cluster.n_clusters": getattr(args, "n_clusters", None),
        "cluster.random_state": getattr(args, "random_state", None),
    }
    return merge_overrides(config, overrides)


def embed_command(args: argparse.Namespace) -> None:
    from image_classification_dino.embedder import embed_images

    config = _config_from_args(args)
    output = embed_images(
        config.model_dir,
        config.image_dir,
        config.embeddings_output,
        device=config.device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        embedding_strategy=config.embedding_strategy,
        normalize=config.normalize,
    )
    print(f"Embeddings saved: {output}")


def cluster_command(args: argparse.Namespace) -> None:
    from image_classification_dino.clustering import cluster_embeddings

    output = cluster_embeddings(
        args.embeddings,
        args.output,
        method=args.method,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    print(f"Clusters saved: {output}")


def run_command(args: argparse.Namespace) -> None:
    from image_classification_dino.clustering import cluster_embeddings
    from image_classification_dino.embedder import embed_images

    config = _config_from_args(args)
    embeddings_output = embed_images(
        config.model_dir,
        config.image_dir,
        config.embeddings_output,
        device=config.device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        embedding_strategy=config.embedding_strategy,
        normalize=config.normalize,
    )
    clusters_output = cluster_embeddings(
        embeddings_output,
        config.clusters_output,
        method=config.cluster.method,
        n_clusters=config.cluster.n_clusters,
        random_state=config.cluster.random_state,
    )
    print(f"Embeddings saved: {embeddings_output}")
    print(f"Clusters saved: {clusters_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dino-images",
        description="Local DINOv3 image embedding and clustering CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Convert images to DINO embeddings.")
    embed.add_argument("--config", default=None)
    embed.add_argument("--model-dir", default=None)
    embed.add_argument("--image-dir", default=None)
    embed.add_argument("--embeddings-output", "--output", dest="embeddings_output", default=None)
    _add_common_runtime_args(embed)
    embed.set_defaults(func=embed_command)

    cluster = subparsers.add_parser("cluster", help="Cluster saved embeddings.")
    cluster.add_argument("--embeddings", required=True)
    cluster.add_argument("--output", required=True)
    cluster.add_argument(
        "--method",
        choices=["kmeans", "minibatch-kmeans", "agglomerative", "dbscan"],
        default="kmeans",
    )
    cluster.add_argument("--n-clusters", type=int, default=8)
    cluster.add_argument("--random-state", type=int, default=42)
    cluster.set_defaults(func=cluster_command)

    run = subparsers.add_parser("run", help="Embed images and cluster them.")
    run.add_argument("--config", default=None)
    run.add_argument("--model-dir", default=None)
    run.add_argument("--image-dir", default=None)
    run.add_argument("--embeddings-output", default=None)
    run.add_argument("--clusters-output", default=None)
    run.add_argument(
        "--method",
        choices=["kmeans", "minibatch-kmeans", "agglomerative", "dbscan"],
        default=None,
    )
    run.add_argument("--n-clusters", type=int, default=None)
    run.add_argument("--random-state", type=int, default=None)
    _add_common_runtime_args(run)
    run.set_defaults(func=run_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
