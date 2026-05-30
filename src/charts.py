"""Generate simple PNG charts from summary CSV without external dependencies."""

from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path


WIDTH = 1000
HEIGHT = 560
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 40
MARGIN_BOTTOM = 70

WHITE = (255, 255, 255)
BLACK = (28, 28, 28)
GRID = (224, 228, 232)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
RED = (220, 38, 38)


def generate_charts_from_summary(summary_csv: Path, output_dir: Path) -> dict[str, Path]:
    rows = _read_summary(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "assets_curve": output_dir / "assets_curve.png",
        "cumulative_profit": output_dir / "cumulative_profit.png",
        "max_drawdown": output_dir / "max_drawdown.png",
    }
    _write_line_chart(rows, "day", "total_assets", charts["assets_curve"], BLUE)
    _write_line_chart(rows, "day", "cumulative_profit", charts["cumulative_profit"], GREEN)
    _write_line_chart(rows, "day", "max_drawdown", charts["max_drawdown"], RED)
    return charts


def _read_summary(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {
                "day": float(row["day"]),
                "total_assets": float(row["total_assets"]),
                "cumulative_profit": float(row["cumulative_profit"]),
                "max_drawdown": float(row["max_drawdown"]),
            }
            for row in reader
        ]


def _write_line_chart(rows: list[dict[str, float]], x_key: str, y_key: str, path: Path, color: tuple[int, int, int]) -> None:
    image = [[WHITE for _ in range(WIDTH)] for _ in range(HEIGHT)]
    _draw_grid(image)

    if rows:
        points = _scale_points(rows, x_key, y_key)
        _draw_polyline(image, points, color)
        for point in points:
            _draw_circle(image, point[0], point[1], 4, color)

    _write_png(path, image)


def _draw_grid(image: list[list[tuple[int, int, int]]]) -> None:
    x0 = MARGIN_LEFT
    x1 = WIDTH - MARGIN_RIGHT
    y0 = MARGIN_TOP
    y1 = HEIGHT - MARGIN_BOTTOM

    for index in range(6):
        y = round(y0 + (y1 - y0) * index / 5)
        _draw_line(image, x0, y, x1, y, GRID)
    for index in range(6):
        x = round(x0 + (x1 - x0) * index / 5)
        _draw_line(image, x, y0, x, y1, GRID)

    _draw_line(image, x0, y0, x0, y1, BLACK)
    _draw_line(image, x0, y1, x1, y1, BLACK)


def _scale_points(rows: list[dict[str, float]], x_key: str, y_key: str) -> list[tuple[int, int]]:
    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        padding = abs(min_y) * 0.05 or 1
        min_y -= padding
        max_y += padding
    else:
        padding = (max_y - min_y) * 0.08
        min_y -= padding
        max_y += padding

    plot_left = MARGIN_LEFT
    plot_right = WIDTH - MARGIN_RIGHT
    plot_top = MARGIN_TOP
    plot_bottom = HEIGHT - MARGIN_BOTTOM

    points = []
    for row in rows:
        x = plot_left + (row[x_key] - min_x) / (max_x - min_x) * (plot_right - plot_left)
        y = plot_bottom - (row[y_key] - min_y) / (max_y - min_y) * (plot_bottom - plot_top)
        points.append((round(x), round(y)))
    return points


def _draw_polyline(image: list[list[tuple[int, int, int]]], points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for start, end in zip(points, points[1:]):
        _draw_line(image, start[0], start[1], end[0], end[1], color)


def _draw_line(
    image: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_pixel(image, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def _draw_circle(image: list[list[tuple[int, int, int]]], cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                _set_pixel(image, x, y, color)


def _set_pixel(image: list[list[tuple[int, int, int]]], x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        image[y][x] = color


def _write_png(path: Path, image: list[list[tuple[int, int, int]]]) -> None:
    raw_rows = []
    for row in image:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    raw = b"".join(raw_rows)
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, 9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
