from __future__ import annotations

import math
from io import BytesIO
from typing import Dict, Iterable, Set, Tuple


def _calc_extent(points: Iterable[Dict]) -> tuple[float, float, float, float]:
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    pad_lat = max(0.02, (lat_max - lat_min) * 0.15)
    pad_lon = max(0.02, (lon_max - lon_min) * 0.15)
    return (
        lon_min - pad_lon,
        lon_max + pad_lon,
        lat_min - pad_lat,
        lat_max + pad_lat,
    )


def render_dataset_map(dataset: Dict, selected: Set[str], highlight: str | None) -> Tuple[BytesIO, str]:
    """Render map (PNG) for dataset points, destacando seleção e ponto focal."""
    points = dataset["points"]
    extent = _calc_extent(points)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    try:
        import cartopy.crs as ccrs  # type: ignore
        has_cartopy = True
    except Exception:
        has_cartopy = False
        ccrs = None  # type: ignore

    if not has_cartopy:
        # Fallback robusto: gera SVG para evitar problemas de backend no container.
        svg_buf = _render_svg(points, selected, highlight, extent)
        return svg_buf, "image/svg+xml"

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6.5, 6.5), dpi=200)
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color="#cbd5e1", x_inline=False, y_inline=False)

    ax.set_facecolor("#ffffff")

    default_color = "#ef4444"  # vermelho vivo
    selected_color = "#0ea5e9"  # ciano
    highlight_color = "#22c55e"  # verde

    point_count = 0
    for point in points:
        point_id = point["id"]
        color = highlight_color if highlight and point_id == highlight else (
            selected_color if point_id in selected else default_color
        )
        ax.scatter(
            point["lon"],
            point["lat"],
            s=160,
            color=color,
            edgecolors="#0f172a",
            linewidths=1.3,
            marker="o",
            alpha=0.98,
            transform=ccrs.PlateCarree() if has_cartopy else None,
            zorder=5,
        )
        # draw a cross overlay to guarantee visibility
        ax.plot(
            [point["lon"]],
            [point["lat"]],
            marker="+",
            markersize=9,
            color="#0f172a",
            transform=ccrs.PlateCarree() if has_cartopy else None,
            zorder=6,
        )
        point_count += 1

    ax.text(
        0.01,
        0.01,
        f"Pontos: {point_count}",
        transform=ax.transAxes,
        fontsize=8,
        color="#111827",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="#e5e7eb"),
        zorder=10,
    )

    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_title(dataset["name"], fontsize=10, pad=10)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf, "image/png"


def _render_svg(points: Iterable[Dict], selected: Set[str], highlight: str | None, extent: tuple[float, float, float, float]) -> BytesIO:
    """Fallback renderer (no matplotlib) that returns a simple SVG scatter."""
    min_lon, max_lon, min_lat, max_lat = extent
    width, height = 600, 600

    def project(lat: float, lon: float) -> tuple[float, float]:
        x = (lon - min_lon) / (max_lon - min_lon + 1e-6) * (width - 40) + 20
        y = (max_lat - lat) / (max_lat - min_lat + 1e-6) * (height - 40) + 20
        return x, y

    default_color = "#0f172a"
    selected_color = "#0ea5e9"
    highlight_color = "#2563eb"

    circles = []
    for point in points:
        px, py = project(point["lat"], point["lon"])
        point_id = point["id"]
        color = highlight_color if highlight and point_id == highlight else (
            selected_color if point_id in selected else default_color
        )
        circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1" />')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#f9fafb">
      <rect width="100%" height="100%" fill="#f9fafb"/>
      {''.join(circles)}
    </svg>"""
    buf = BytesIO(svg.encode("utf-8"))
    buf.seek(0)
    return buf
