from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def _load_cluster_frame(clusters_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(clusters_csv)
    required = {"image_path", "cluster"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in cluster CSV: {sorted(missing)}")
    return frame


def _sample_frame(frame: pd.DataFrame, samples_per_cluster: int) -> pd.DataFrame:
    return (
        frame.sort_values(["cluster", "image_path"])
        .groupby("cluster", group_keys=False)
        .head(samples_per_cluster)
        .reset_index(drop=True)
    )


def export_cluster_html(
    clusters_csv: str | Path,
    output_path: str | Path,
    *,
    samples_per_cluster: int = 24,
) -> Path:
    frame = _load_cluster_frame(clusters_csv)
    sampled = _sample_frame(frame, samples_per_cluster)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cluster_counts = frame.groupby("cluster").size().sort_index()
    sections: list[str] = []
    for cluster, count in cluster_counts.items():
        rows = sampled[sampled["cluster"] == cluster]
        items = []
        for image_path in rows["image_path"]:
            path = Path(image_path)
            escaped_path = html.escape(str(path))
            uri = path.resolve().as_uri()
            items.append(
                "\n".join(
                    [
                        '<figure class="item">',
                        f'  <a href="{uri}"><img src="{uri}" loading="lazy" alt="{escaped_path}"></a>',
                        f"  <figcaption title=\"{escaped_path}\">{html.escape(path.name)}</figcaption>",
                        "</figure>",
                    ]
                )
            )
        sections.append(
            "\n".join(
                [
                    '<section class="cluster">',
                    f"<h2>Cluster {html.escape(str(cluster))} <span>{count} images</span></h2>",
                    '<div class="grid">',
                    *items,
                    "</div>",
                    "</section>",
                ]
            )
        )

    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>DINO Cluster Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f8;color:#222}",
            "h1{font-size:24px;margin:0 0 16px}",
            ".cluster{margin:0 0 32px}",
            "h2{font-size:18px;margin:0 0 12px}",
            "h2 span{font-size:13px;color:#666;font-weight:400;margin-left:8px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px}",
            ".item{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}",
            ".item img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#fff;display:block}",
            ".item figcaption{font-size:12px;line-height:1.3;padding:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>DINO Cluster Report</h1>",
            *sections,
            "</body>",
            "</html>",
        ]
    )
    output.write_text(document, encoding="utf-8")
    return output


def export_cluster_contact_sheet(
    clusters_csv: str | Path,
    output_dir: str | Path,
    *,
    samples_per_cluster: int = 24,
    thumb_size: int = 160,
    columns: int = 6,
) -> list[Path]:
    frame = _load_cluster_frame(clusters_csv)
    sampled = _sample_frame(frame, samples_per_cluster)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for cluster, rows in sampled.groupby("cluster", sort=True):
        images = []
        for image_path in rows["image_path"]:
            try:
                with Image.open(image_path) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (thumb_size, thumb_size))
            except OSError:
                continue
            tile = Image.new("RGB", (thumb_size, thumb_size), "white")
            x = (thumb_size - thumb.width) // 2
            y = (thumb_size - thumb.height) // 2
            tile.paste(thumb, (x, y))
            images.append(tile)

        if not images:
            continue

        rows_count = (len(images) + columns - 1) // columns
        header_height = 32
        sheet = Image.new("RGB", (columns * thumb_size, rows_count * thumb_size + header_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"Cluster {cluster} - {len(images)} sample images", fill=(20, 20, 20))

        for index, image in enumerate(images):
            col = index % columns
            row = index // columns
            sheet.paste(image, (col * thumb_size, header_height + row * thumb_size))

        output = output_root / f"cluster_{cluster}.jpg"
        sheet.save(output, quality=92)
        outputs.append(output)

    return outputs
