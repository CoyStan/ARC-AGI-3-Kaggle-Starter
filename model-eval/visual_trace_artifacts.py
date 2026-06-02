"""Visual trace artifacts for VLM/rule-hypothesis ARC-AGI-3 evals.

This module is intentionally stdlib-first. It writes portable PPM images so the
research harness can generate visual evidence without adding a submitted-runtime
or network/model dependency. Optional PNG conversion can be added later behind an
eval-only dependency gate.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

SCHEMA_TRACE_PACK = "mini-palari.visual-trace-pack.v0.1"

# Stable high-contrast palette for ARC-ish integer colors. Values beyond the
# palette are deterministically wrapped.
PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0 black/background
    (30, 144, 255),   # 1 blue
    (220, 20, 60),    # 2 red
    (50, 205, 50),    # 3 green
    (255, 215, 0),    # 4 yellow
    (169, 169, 169),  # 5 gray
    (255, 140, 0),    # 6 orange
    (138, 43, 226),   # 7 purple
    (0, 206, 209),    # 8 cyan
    (160, 82, 45),    # 9 brown
    (255, 105, 180),  # 10 pink
    (124, 252, 0),    # 11 lime
    (70, 130, 180),   # 12 steel
    (255, 255, 255),  # 13 white
    (128, 0, 0),      # 14 maroon
    (0, 128, 128),    # 15 teal
]


def _unwrap_frame_object(frame_obj: Any) -> Any:
    if isinstance(frame_obj, dict):
        for key in ("frame", "data", "array", "grid"):
            if key in frame_obj:
                return frame_obj[key]
    if hasattr(frame_obj, "frame"):
        return getattr(frame_obj, "frame")
    return frame_obj


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def normalize_frame(frame_obj: Any) -> list[list[int]]:
    """Normalize ARC frame-like payloads to a rectangular 2D integer matrix.

    Handles plain 2D lists, singleton 3D wrappers like ``[[[...]]]``, objects or
    dicts with a ``frame`` field, and numpy-like objects exposing ``tolist``.
    """

    raw = _to_list(_unwrap_frame_object(frame_obj))
    raw = _to_list(raw)

    # Common singleton channel/batch wrapper: [[[row...], ...]] or
    # FrameDataRaw.frame shaped as [numpy_array_2d].
    while isinstance(raw, list) and len(raw) == 1:
        first = _to_list(raw[0])
        if isinstance(first, list) and first and isinstance(_to_list(first[0]), list):
            raw = first
            continue
        break

    if not isinstance(raw, list):
        return []
    if not raw:
        return []

    matrix: list[list[int]] = []
    for row in raw:
        row = _to_list(row)
        if not isinstance(row, list):
            return []
        normalized_row: list[int] = []
        for cell in row:
            try:
                normalized_row.append(int(cell))
            except Exception:
                normalized_row.append(0)
        matrix.append(normalized_row)

    if not matrix:
        return []
    width = max(len(row) for row in matrix)
    if width == 0:
        return []
    return [row + [0] * (width - len(row)) for row in matrix]


def _neighbors(x: int, y: int, width: int, height: int) -> Iterable[tuple[int, int]]:
    if x > 0:
        yield x - 1, y
    if x + 1 < width:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < height:
        yield x, y + 1


def connected_components(frame_obj: Any, *, include_zero: bool = False) -> list[dict[str, Any]]:
    """Return deterministic same-color 4-connected components with bboxes."""

    grid = normalize_frame(frame_obj)
    if not grid:
        return []
    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in visited:
                continue
            color = grid[y][x]
            if color == 0 and not include_zero:
                visited.add((x, y))
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            points: list[tuple[int, int]] = []
            while queue:
                px, py = queue.popleft()
                points.append((px, py))
                for nx, ny in _neighbors(px, py, width, height):
                    if (nx, ny) in visited:
                        continue
                    if grid[ny][nx] != color:
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            area = len(points)
            components.append(
                {
                    "id": f"c{len(components)}",
                    "color": int(color),
                    "area": area,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "centroid": [round(sum(xs) / area, 3), round(sum(ys) / area, 3)],
                }
            )
    return components


def frame_delta(before_obj: Any, after_obj: Any) -> dict[str, Any]:
    before = normalize_frame(before_obj)
    after = normalize_frame(after_obj)
    height = max(len(before), len(after))
    width = max([0] + [len(row) for row in before] + [len(row) for row in after])
    changed: list[list[int]] = []
    for y in range(height):
        for x in range(width):
            b = before[y][x] if y < len(before) and x < len(before[y]) else 0
            a = after[y][x] if y < len(after) and x < len(after[y]) else 0
            if b != a:
                changed.append([x, y])
    if changed:
        xs = [p[0] for p in changed]
        ys = [p[1] for p in changed]
        bbox: list[int] | None = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bbox = None
    return {"changed_cells": len(changed), "bbox": bbox, "changed_points": changed[:200]}


def _rgb_for_color(color: int) -> tuple[int, int, int]:
    return PALETTE[int(color) % len(PALETTE)]


def write_ppm(frame_obj: Any, path: str | Path, *, scale: int = 8) -> Path:
    """Write a scaled PPM image for a frame and return the path."""

    grid = normalize_frame(frame_obj)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not grid:
        grid = [[0]]
    height = len(grid)
    width = len(grid[0])
    scale = max(1, int(scale))
    with out.open("w", encoding="ascii") as f:
        f.write(f"P3\n{width * scale} {height * scale}\n255\n")
        for y in range(height):
            expanded_row: list[str] = []
            for x in range(width):
                rgb = _rgb_for_color(grid[y][x])
                expanded_row.extend([f"{rgb[0]} {rgb[1]} {rgb[2]}"] * scale)
            line = " ".join(expanded_row)
            for _ in range(scale):
                f.write(line + "\n")
    return out


def write_trace_pack(
    *,
    output_dir: str | Path,
    game_id: str,
    frames: list[Any],
    actions: list[int] | None = None,
    available_actions: list[int] | None = None,
    title: str | None = None,
    image_scale: int = 8,
) -> dict[str, Any]:
    """Write visual artifacts plus replayable JSON metadata for a frame trace."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions = [int(a) for a in (actions or [])]
    available_actions = [int(a) for a in (available_actions or [])]
    normalized_frames = [normalize_frame(frame) for frame in frames]

    frame_records: list[dict[str, Any]] = []
    for idx, frame in enumerate(normalized_frames):
        image_path = write_ppm(frame, out / f"frame_{idx:03d}.ppm", scale=image_scale)
        frame_records.append(
            {
                "index": idx,
                "image_path": str(image_path),
                "shape": [len(frame), len(frame[0]) if frame else 0],
                "components": connected_components(frame),
            }
        )

    deltas = [
        {"from": idx, "to": idx + 1, **frame_delta(normalized_frames[idx], normalized_frames[idx + 1])}
        for idx in range(max(0, len(normalized_frames) - 1))
    ]

    pack = {
        "schema": SCHEMA_TRACE_PACK,
        "title": title or f"visual trace for {game_id}",
        "game_id": game_id,
        "actions": actions,
        "available_actions": available_actions,
        "frames": frame_records,
        "deltas": deltas,
        "authority": "evidence_only_no_runtime_authority",
    }
    pack_path = out / "trace_pack.json"
    pack["trace_pack_path"] = str(pack_path)
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
    return pack


def load_trace_pack(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
