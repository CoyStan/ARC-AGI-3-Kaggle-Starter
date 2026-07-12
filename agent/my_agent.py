"""Dependency-light ARC-AGI-3 submission policy helpers.

This module intentionally avoids importing the official ARC runtime.  The same
logic can be unit-tested inside Mini-Palari and copied into the Kaggle starter's
single-file `agent/my_agent.py` submission entrypoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:  # Available inside the ARC-AGI-3 starter/Kaggle runtime.
    from arcengine import FrameData, GameAction  # type: ignore
    from agents.agent import Agent  # type: ignore
except Exception:  # Mini-Palari unit tests exercise the policy without starter deps.
    FrameData = Any  # type: ignore
    GameAction = Any  # type: ignore

    class Agent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.game_id = kwargs.get("game_id", "generic")

        @property
        def name(self) -> str:
            return self.__class__.__name__

# The Kaggle starter loads this file with importlib without first registering
# the module in sys.modules. dataclasses consult sys.modules[__name__] while
# processing postponed annotations, so keep the self-contained export robust.
sys.modules.setdefault(__name__, types.ModuleType(__name__))

ACTION_BY_NUMBER = {index: f"ACTION{index}" for index in range(1, 8)} | {0: "RESET"}
RESET_STATES = frozenset({"NOT_PLAYED", "GAME_OVER"})
WIN_STATE = "WIN"
SUBMISSION_HYPOTHESIS_SCHEMA = "mini-palari.submission-hypothesis.v0.1"
SUBMISSION_LINE_RECORD_SCHEMA = "mini-palari.submission-line-record.v0.1"
LOCAL_SIDECAR_PROPOSAL_SCHEMA = "mini-palari.local-sidecar-proposal.v0.1"
LOCAL_PROPOSAL_VERIFICATION_SCHEMA = "mini-palari.local-proposal-verification.v0.1"
DEFAULT_SIMPLE_PROBE_SEQUENCE = ("ACTION4", "ACTION1", "ACTION2", "ACTION3", "ACTION5", "ACTION7")
TERMINAL_CONTACT_PROGRESS_SIGNAL = "candidate_plan_terminal_contact_observed"
CANDIDATE_PROGRESS_SIGNALS = frozenset({"candidate_target_distance_decreased", TERMINAL_CONTACT_PROGRESS_SIGNAL})
OBJECTIVE_PROGRESS_SIGNALS = frozenset({"level_completion_increased"}) | CANDIDATE_PROGRESS_SIGNALS
REPEAT_AUTHORIZING_PROGRESS_SIGNALS = OBJECTIVE_PROGRESS_SIGNALS - frozenset({TERMINAL_CONTACT_PROGRESS_SIGNAL})


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _state_name(value: Any) -> str:
    raw = _field(value, "name", value)
    if raw is None:
        return ""
    text = str(raw).strip().upper()
    if text.startswith("GAMESTATE."):
        text = text.rsplit(".", 1)[-1]
    return text


def action_name_from_value(value: Any) -> str:
    """Return a normalized ARC-AGI-3 action name from enum/int/string values."""

    raw = _field(value, "name", value)
    if isinstance(raw, bool):
        raise ValueError("action identifiers cannot be booleans")
    if isinstance(raw, int):
        if raw in ACTION_BY_NUMBER:
            return ACTION_BY_NUMBER[raw]
        raise ValueError(f"unsupported action number: {raw!r}")
    text = str(raw).strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.rsplit(".", 1)[-1]
    if text.isdecimal():
        return action_name_from_value(int(text))
    if text == "RESET" or text in ACTION_BY_NUMBER.values():
        return text
    raise ValueError(f"unsupported action identifier: {value!r}")


def _is_complex_action(action: Any) -> bool:
    if action is None:
        return False
    checker = _field(action, "is_complex")
    if callable(checker):
        try:
            return bool(checker())
        except TypeError:
            pass
    return bool(_field(action, "complex_action", False) or _field(action, "requires_args", False))


def _clean_grid(grid_payload: Any) -> list[list[int]]:
    if not isinstance(grid_payload, Sequence) or isinstance(grid_payload, (str, bytes, bytearray)) or not grid_payload:
        return [[0]]
    rows: list[list[int]] = []
    for row in grid_payload:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            continue
        clean_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                clean_row.append(0)
            else:
                clean_row.append(max(0, min(15, int(cell))))
        if clean_row:
            rows.append(clean_row)
    return rows or [[0]]


def _looks_like_grid_row(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(cell, int) and not isinstance(cell, bool) for cell in value)
    )


def _latest_grid(frame_payload: Any) -> list[list[int]]:
    if not isinstance(frame_payload, Sequence) or isinstance(frame_payload, (str, bytes, bytearray)) or not frame_payload:
        return [[0]]
    # ARC/starter payloads have appeared in both shapes:
    #   2-D grid: [[0, 1], [2, 0]]
    #   sequence of grids: [[[0, 1], [2, 0]], ...]
    # Treat a sequence of integer rows as an already-current 2-D grid instead
    # of accidentally selecting its last row as the whole grid.
    if all(_looks_like_grid_row(row) for row in frame_payload):
        return _clean_grid(frame_payload)
    return _clean_grid(frame_payload[-1])


def frame_delta_count(previous_frame_payload: Any, current_frame_payload: Any) -> int:
    """Return a bounded count of changed cells between latest grids."""

    previous = _latest_grid(previous_frame_payload)
    current = _latest_grid(current_frame_payload)
    height = max(len(previous), len(current))
    width = max(max((len(row) for row in previous), default=0), max((len(row) for row in current), default=0))
    changed = 0
    for y in range(height):
        previous_row = previous[y] if y < len(previous) else []
        current_row = current[y] if y < len(current) else []
        for x in range(width):
            if (previous_row[x] if x < len(previous_row) else 0) != (current_row[x] if x < len(current_row) else 0):
                changed += 1
    return changed


def salient_coordinate(frame_payload: Any) -> dict[str, int]:
    """Pick a deterministic bounded ACTION6 coordinate from non-zero cells."""

    return salient_coordinate_candidates(frame_payload, limit=1)[0]


def salient_coordinate_candidates(frame_payload: Any, limit: int = 16) -> list[dict[str, int]]:
    """Rank bounded ACTION6 probe coordinates from public frame salience only.

    This deterministic Brain-aligned coordinate-probe generator does not claim
    objective authority. It gives click/selection-like games more coverage than
    repeatedly clicking one centroid forever.
    """

    grid = _latest_grid(frame_payload)
    height = len(grid)
    width = max((len(row) for row in grid), default=1)
    points: list[tuple[int, int, int]] = []
    by_color: dict[int, list[tuple[int, int, int]]] = {}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell:
                point = (x, y, int(cell))
                points.append(point)
                by_color.setdefault(int(cell), []).append(point)

    candidates: list[dict[str, int]] = []

    def add(x: float | int, y: float | int) -> None:
        coord = {"x": min(63, max(0, int(round(x)))), "y": min(63, max(0, int(round(y))))}
        if coord not in candidates:
            candidates.append(coord)

    def add_if_in_frame(x: int, y: int) -> None:
        if 0 <= y < height and 0 <= x < len(grid[y]):
            add(x, y)

    components: list[dict[str, Any]] = []
    visited: set[tuple[int, int]] = set()
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if not cell or (x, y) in visited:
                continue
            color = int(cell)
            stack = [(x, y)]
            visited.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= len(grid[ny]):
                        continue
                    if (nx, ny) in visited or int(grid[ny][nx]) != color:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))
            xs = [cx for cx, _cy in cells]
            ys = [cy for _cx, cy in cells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "center_x": (min_x + max_x) / 2,
                    "center_y": (min_y + max_y) / 2,
                }
            )

    components_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        components_by_color.setdefault(int(component["color"]), []).append(component)

    def component_sort_key(component: Mapping[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(component["area"]),
            int(component["color"]),
            int(component["min_y"]),
            int(component["min_x"]),
        )

    def add_visual_role_candidates() -> None:
        # Hollow boxes/rings and dotted paths often mark click targets or route
        # hints. Put their centers/endpoints before singleton noise pixels.
        hollow_targets: list[Mapping[str, Any]] = []
        for component in sorted(components, key=component_sort_key):
            comp_width = int(component["width"])
            comp_height = int(component["height"])
            if comp_width < 3 or comp_height < 3:
                continue
            cells = set(component["cells"])
            min_x = int(component["min_x"])
            max_x = int(component["max_x"])
            min_y = int(component["min_y"])
            max_y = int(component["max_y"])
            touches_all_edges = (
                any(y == min_y for _x, y in cells)
                and any(y == max_y for _x, y in cells)
                and any(x == min_x for x, _y in cells)
                and any(x == max_x for x, _y in cells)
            )
            hollowish = int(component["area"]) < comp_width * comp_height
            if touches_all_edges and hollowish:
                hollow_targets.append(component)

        def add_path_ring_candidates() -> None:
            if not hollow_targets:
                return
            for target in hollow_targets:
                target_cx = float(target["center_x"])
                target_cy = float(target["center_y"])
                path_groups: list[list[Mapping[str, Any]]] = []
                for color, color_components in sorted(components_by_color.items()):
                    if color == int(target["color"]):
                        continue
                    small = [component for component in color_components if int(component["area"]) <= 4]
                    if len(small) >= 3:
                        path_groups.append(small)
                for path_group in path_groups:
                    path_group = sorted(path_group, key=lambda component: (float(component["center_x"]), float(component["center_y"])))
                    source_end = max(
                        path_group,
                        key=lambda component: (
                            abs(float(component["center_x"]) - target_cx) + abs(float(component["center_y"]) - target_cy),
                            float(component["center_y"]),
                            -float(component["center_x"]),
                        ),
                    )
                    target_end = min(
                        path_group,
                        key=lambda component: (
                            abs(float(component["center_x"]) - target_cx) + abs(float(component["center_y"]) - target_cy),
                            float(component["center_y"]),
                            float(component["center_x"]),
                        ),
                    )
                    source_x = float(source_end["center_x"])
                    source_y = float(source_end["center_y"])
                    source_markers = [
                        component
                        for component in components
                        if component not in path_group
                        and component is not target
                        and int(component["color"]) not in {int(source_end["color"]), int(target["color"])}
                        and int(component["area"]) <= 4
                        and abs(float(component["center_x"]) - source_x) <= 2.0
                        and abs(float(component["center_y"]) - source_y) <= 2.0
                    ]
                    if source_markers:
                        marker_xs = [float(component["center_x"]) for component in source_markers]
                        marker_ys = [float(component["center_y"]) for component in source_markers]
                        add(sum(marker_xs) / len(marker_xs), sum(marker_ys) / len(marker_ys))
                    add(float(source_end["center_x"]), float(source_end["center_y"]))
                    add(float(target_end["center_x"]), float(target_end["center_y"]))

        horizontal_controls = [
            component
            for component in components
            if int(component["width"]) >= 4
            and int(component["width"]) >= max(2, int(component["height"]) * 2)
        ]

        def central_control_between(upper: Mapping[str, Any], lower: Mapping[str, Any]) -> Mapping[str, Any] | None:
            upper_y = float(upper["center_y"])
            lower_y = float(lower["center_y"])
            if lower_y <= upper_y:
                return None
            between = [
                component
                for component in horizontal_controls
                if upper_y < float(component["center_y"]) < lower_y
                and int(component["color"]) not in {int(upper["color"]), int(lower["color"])}
            ]
            if not between:
                return None
            midpoint_y = (upper_y + lower_y) / 2
            return max(
                between,
                key=lambda component: (
                    int(component["width"]),
                    -abs(float(component["center_y"]) - midpoint_y),
                    -int(component["min_y"]),
                    -int(component["min_x"]),
                ),
            )

        for color, color_components in sorted(components_by_color.items()):
            small = [component for component in color_components if 1 < int(component["area"]) <= 16]
            pairs: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
            for lower in small:
                lower_cx = float(lower["center_x"])
                lower_cy = float(lower["center_y"])
                above = [
                    component
                    for component in small
                    if float(component["center_y"]) < lower_cy
                    and abs(float(component["center_x"]) - lower_cx) <= 1.0
                    and lower_cy - float(component["center_y"]) >= max(3.0, height / 3)
                ]
                if not above:
                    continue
                upper = min(above, key=lambda component: (abs(float(component["center_x"]) - lower_cx), float(component["center_y"])))
                pairs.append((lower_cx, lower, upper))
            for _lower_cx, lower, upper in sorted(pairs, key=lambda item: (item[0], float(item[1]["center_y"]), color)):
                add(float(lower["center_x"]), float(lower["center_y"]))
                add(float(upper["center_x"]), float(upper["center_y"]))
                central = central_control_between(upper, lower)
                if central is not None:
                    add(float(central["center_x"]), float(central["center_y"]))

        for target in hollow_targets:
            add(float(target["center_x"]), float(target["center_y"]))
        add_path_ring_candidates()

        for color, color_components in sorted(components_by_color.items()):
            small = [component for component in color_components if int(component["area"]) <= 4]
            if len(small) < 3:
                continue
            centers = [
                (float(component["center_x"]), float(component["center_y"]), component)
                for component in small
            ]
            for ordered in (
                sorted(centers, key=lambda item: (item[0], item[1])),
                sorted(centers, key=lambda item: (item[1], item[0])),
            ):
                if not ordered:
                    continue
                for index in (-1, 0, -2, 1):
                    if -len(ordered) <= index < len(ordered):
                        add(ordered[index][0], ordered[index][1])

        for component in sorted(components, key=component_sort_key):
            comp_width = int(component["width"])
            comp_height = int(component["height"])
            min_x = int(component["min_x"])
            max_x = int(component["max_x"])
            min_y = int(component["min_y"])
            max_y = int(component["max_y"])
            center_x = float(component["center_x"])
            center_y = float(component["center_y"])
            if comp_width >= 4 and comp_width >= max(2, comp_height * 2):
                add(center_x, center_y)
                add(min_x, center_y)
                add(max_x, center_y)
            if comp_height >= 4 and comp_height >= max(2, comp_width * 2):
                add(center_x, center_y)
                add(center_x, min_y)
                add(center_x, max_y)

        for component in sorted(components, key=component_sort_key):
            if int(component["area"]) > 1:
                add(float(component["center_x"]), float(component["center_y"]))

    if points:
        xs = [x for x, _y, _cell in points]
        ys = [y for _x, y, _cell in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        add_visual_role_candidates()
        add(sum(xs) / len(xs), sum(ys) / len(ys))
        for _count, color in sorted((len(color_points), color) for color, color_points in by_color.items()):
            color_points = by_color[color]
            add(sum(x for x, _y, _cell in color_points) / len(color_points), sum(y for _x, y, _cell in color_points) / len(color_points))
        add((min_x + max_x) / 2, (min_y + max_y) / 2)
        add(min_x, min_y)
        add(max_x, min_y)
        add(min_x, max_y)
        add(max_x, max_y)
        # Sample individual salient cells from rare colors first. This helps
        # button/target games where the actionable pixel is not the centroid.
        for _count, color in sorted((len(color_points), color) for color, color_points in by_color.items()):
            for x, y, _cell in sorted(by_color[color], key=lambda p: (abs(p[0] - width / 2) + abs(p[1] - height / 2), p[1], p[0]))[:4]:
                add(x, y)
        # Rare markers often act as buttons, sockets, or anchors whose useful
        # click is adjacent to the colored cell rather than on its centroid.
        for _count, color in sorted((len(color_points), color) for color, color_points in by_color.items()):
            for x, y, _cell in sorted(by_color[color], key=lambda p: (abs(p[0] - width / 2) + abs(p[1] - height / 2), p[1], p[0]))[:4]:
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    add_if_in_frame(x + dx, y + dy)
    else:
        add(width // 2, height // 2)

    # Generic fallback coverage for empty/sparse or misleading frames. The hard
    # cap keeps runtime bounded and deterministic.
    add(width // 2, height // 2)
    add(width // 4, height // 4)
    add((3 * width) // 4, height // 4)
    add(width // 4, (3 * height) // 4)
    add((3 * width) // 4, (3 * height) // 4)
    return candidates[: max(1, limit)]

def available_action_names(latest_frame: Any, actions_by_name: Mapping[str, Any]) -> list[str]:
    raw_available = _field(latest_frame, "available_actions")
    if raw_available is None:
        action_input = _field(latest_frame, "action_input")
        raw_available = _field(action_input, "available_actions")
    if raw_available is None:
        raw_available = [name for name in actions_by_name if name != "RESET"]
    names: list[str] = []
    for item in raw_available:
        try:
            name = action_name_from_value(item)
        except ValueError:
            continue
        if name in actions_by_name and name not in names:
            names.append(name)
    return names


def _attach_reasoning(action: Any, reasoning: Any) -> Any:
    try:
        action.reasoning = reasoning
    except Exception:
        pass
    return action


def _trace_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_trace_value(item) for item in value)
    return value


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_trace_value(payload), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _hypothesis_confidence(positive: int, negative: int) -> float:
    # Conservative Beta(1, 1) posterior mean: one positive is enough to repeat a
    # probe, but not enough to claim solved/action authority.
    return round((1 + positive) / (2 + positive + negative), 6)


def _inert_authority_record() -> dict[str, bool]:
    return {
        "action_authority_created": False,
        "executable_action_args_created": False,
        "model_call_created": False,
        "llm_model_proposer_used": False,
        "network_used": False,
    }


def _submission_line_record(
    *,
    game_id: str,
    turn_index: int,
    line_id: int,
    line_name: str,
    authority: str,
    payload: Mapping[str, Any],
    source_record_refs: Sequence[str] = (),
) -> dict[str, Any]:
    base = {
        "game_id": game_id,
        "turn_index": turn_index,
        "line_id": line_id,
        "line_name": line_name,
        "payload": _trace_value(payload),
        "source_record_refs": list(source_record_refs),
    }
    return {
        "schema": SUBMISSION_LINE_RECORD_SCHEMA,
        "record_id": _stable_id(f"subline{line_id}", base),
        "line_id": line_id,
        "line_name": line_name,
        "authority": authority,
        "source_record_refs": list(source_record_refs),
        "authority_flags": _inert_authority_record(),
        "payload": dict(_trace_value(payload)),
    }


def _scene_summary(frame_payload: Any) -> dict[str, Any]:
    grid = _latest_grid(frame_payload)
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    points: list[tuple[int, int, int]] = []
    colors: dict[int, int] = {}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell:
                value = int(cell)
                points.append((x, y, value))
                colors[value] = colors.get(value, 0) + 1
    if points:
        xs = [x for x, _y, _value in points]
        ys = [y for _x, y, _value in points]
        bbox: dict[str, int] | None = {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}
    else:
        bbox = None
    return {
        "grid_width": width,
        "grid_height": height,
        "nonzero_cells": len(points),
        "color_histogram": {str(color): count for color, count in sorted(colors.items())},
        "bounding_box": bbox,
        "public_only": True,
    }


def _nonzero_points(frame_payload: Any) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for y, row in enumerate(_latest_grid(frame_payload)):
        for x, cell in enumerate(row):
            if cell:
                points.append((x, y, int(cell)))
    return points


def _centroid(points: Sequence[tuple[int, int, int]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (sum(x for x, _y, _cell in points) / len(points), sum(y for _x, y, _cell in points) / len(points))


def _movement_label(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _delta_signature(previous_frame_payload: Any, current_frame_payload: Any) -> dict[str, Any]:
    previous = _latest_grid(previous_frame_payload)
    current = _latest_grid(current_frame_payload)
    height = max(len(previous), len(current))
    width = max(max((len(row) for row in previous), default=0), max((len(row) for row in current), default=0))
    changed = 0
    color_changes: dict[str, int] = {}
    changed_before_points: list[tuple[int, int, int]] = []
    changed_after_points: list[tuple[int, int, int]] = []
    for y in range(height):
        previous_row = previous[y] if y < len(previous) else []
        current_row = current[y] if y < len(current) else []
        for x in range(width):
            before = previous_row[x] if x < len(previous_row) else 0
            after = current_row[x] if x < len(current_row) else 0
            if before != after:
                changed += 1
                key = f"{before}->{after}"
                color_changes[key] = color_changes.get(key, 0) + 1
                if before:
                    changed_before_points.append((x, y, int(before)))
                if after:
                    changed_after_points.append((x, y, int(after)))
    before_centroid = _centroid(_nonzero_points(previous_frame_payload))
    after_centroid = _centroid(_nonzero_points(current_frame_payload))
    before_changed_centroid = _centroid(changed_before_points)
    after_changed_centroid = _centroid(changed_after_points)
    bbox_shift = None
    if before_changed_centroid is not None and after_changed_centroid is not None:
        bbox_shift = {"dx": int(round(after_changed_centroid[0] - before_changed_centroid[0])), "dy": int(round(after_changed_centroid[1] - before_changed_centroid[1]))}
    elif before_centroid is not None and after_centroid is not None:
        bbox_shift = {"dx": int(round(after_centroid[0] - before_centroid[0])), "dy": int(round(after_centroid[1] - before_centroid[1]))}
    return {
        "changed_cells": changed,
        "bbox_shift": bbox_shift,
        "before_centroid": {"x": round(before_centroid[0], 3), "y": round(before_centroid[1], 3)} if before_centroid is not None else None,
        "after_centroid": {"x": round(after_centroid[0], 3), "y": round(after_centroid[1], 3)} if after_centroid is not None else None,
        "before_changed_centroid": {"x": round(before_changed_centroid[0], 3), "y": round(before_changed_centroid[1], 3)} if before_changed_centroid is not None else None,
        "after_changed_centroid": {"x": round(after_changed_centroid[0], 3), "y": round(after_changed_centroid[1], 3)} if after_changed_centroid is not None else None,
        "color_changes": {key: color_changes[key] for key in sorted(color_changes)},
    }


def _local_sidecar_verification(
    proposal_id: str,
    evidence_refs: Sequence[str],
    *,
    positive_evidence: int = 0,
    negative_evidence: int = 0,
) -> dict[str, Any]:
    if positive_evidence > negative_evidence:
        decision = "accepted_candidate"
    elif negative_evidence > positive_evidence:
        decision = "rejected_counter_evidence"
    elif positive_evidence or negative_evidence:
        decision = "inconclusive"
    else:
        decision = "blocked_needs_probe_evidence"
    return {
        "schema": LOCAL_PROPOSAL_VERIFICATION_SCHEMA,
        "proposal_id": proposal_id,
        "decision": decision,
        "evidence_refs": list(evidence_refs),
        "counter_evidence_refs": [ref for ref in evidence_refs if negative_evidence > positive_evidence],
        "runtime_authority_granted": False,
    }


def _local_sidecar_proposals(
    *,
    game_id: str,
    turn_index: int,
    scene: Mapping[str, Any],
    available: Sequence[str],
    non_reset: Sequence[str],
    positive_evidence: int = 0,
    negative_evidence: int = 0,
) -> list[dict[str, Any]]:
    """Emit deterministic local proposals shaped like future LLM sidecar output.

    This is deliberately not an LLM call. It gives the runtime a structured,
    proposal-only interface that a future offline model can target, while the
    submission path remains deterministic, stdlib-only, and no-network.
    """
    nonzero_cells = int(scene.get("nonzero_cells", 0) or 0)
    color_histogram = scene.get("color_histogram", {})
    action6_available = "ACTION6" in non_reset
    if action6_available:
        candidate = "click_selection_or_targeting"
        rationale = "ACTION6 is available, so a coordinate/selection mechanic is a plausible family to test."
        expected = ["coordinate probes may change a localized object, target, or selection state"]
        tests = ["compare visible delta after bounded salient ACTION6 coordinate probes"]
    elif nonzero_cells <= 3 and len(color_histogram) <= 2:
        candidate = "grid_navigation_or_control"
        rationale = "Sparse public frame suggests a small controlled object or navigation-style mechanic."
        expected = ["directional probes may move or alter a small nonzero region"]
        tests = ["probe available directional actions and compare one-step visible deltas"]
    else:
        candidate = "object_transformation_or_toggle"
        rationale = "Multiple visible cells/colors suggest transformation, toggle, or object-interaction mechanics."
        expected = ["some probes may transform color, topology, or local object state"]
        tests = ["probe distinct action classes and compare changed-cell/count/color signatures"]
    payload = {
        "schema": LOCAL_SIDECAR_PROPOSAL_SCHEMA,
        "game_id": game_id,
        "turn_index": turn_index,
        "kind": "game_family",
        "candidate": candidate,
        "rationale": rationale,
        "expected_observations": expected,
        "tests_to_run": tests,
        "forbidden_authority": ["live_action", "policy_rank", "memory_write", "objective_claim"],
        "source_trace_refs": ["line9:public_scene_summary"],
        "authority": _inert_authority_record(),
        "available_actions": list(available),
        "non_reset_actions": list(non_reset),
    }
    proposal_id = _stable_id("localsidecar", payload)
    payload["proposal_id"] = proposal_id
    payload["verification"] = _local_sidecar_verification(
        proposal_id,
        payload["source_trace_refs"],
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
    )
    return [payload]


@dataclass
class MiniPalariArcAgi3Policy:
    """Small deterministic policy suitable as a crash-free submission baseline."""

    game_id: str
    simple_probe_sequence: tuple[str, ...] = DEFAULT_SIMPLE_PROBE_SEQUENCE
    turn_index: int = 0
    trace_notes: list[dict[str, Any]] = field(default_factory=list)
    last_probe_action: str | None = None
    last_action_snapshot: dict[str, Any] | None = None
    hypothesis_bank: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_hypotheses: int = 32
    enable_sidecar_proposals: bool = True
    enable_objective_progress: bool = True
    enable_bounded_planning: bool = False
    enable_visual_rule_candidate: bool = False
    active_plan: dict[str, Any] | None = None
    pending_plan_continuation: dict[str, Any] | None = None
    last_probe_progress_label: str | None = None
    probed_actions: set[str] = field(default_factory=set)
    action6_probe_index: int = 0
    action6_tried_coordinates: set[tuple[int, int]] = field(default_factory=set)
    action6_no_delta_coordinates: set[tuple[int, int]] = field(default_factory=set)
    last_action6_coordinate: tuple[int, int] | None = None
    action_no_progress_repeats: dict[str, int] = field(default_factory=dict)
    max_no_progress_action_repeats: int = 4
    objective_no_progress_repeats: dict[str, set[str]] = field(default_factory=dict)
    exhausted_objective_targets: set[str] = field(default_factory=set)
    phase_breaker_resets_without_progress: int = 0
    pending_phase_breaker_recovery: bool = False

    @property
    def normalized_game_id(self) -> str:
        return self.game_id.lower().split("-", 1)[0]

    def is_done(self, latest_frame: Any) -> bool:
        return _state_name(_field(latest_frame, "state")) == WIN_STATE

    def _snapshot_for_result_tracking(self, latest_frame: Any) -> dict[str, Any]:
        return {
            "frame": _trace_value(_field(latest_frame, "frame")),
            "levels_completed": int(_field(latest_frame, "levels_completed", 0) or 0),
        }

    def _status_from_support_counts(self, record: Mapping[str, Any]) -> tuple[str, int, int, float]:
        support = record.get("support", {})
        positive = int(support.get("positive", 0) or 0) if isinstance(support, Mapping) else 0
        negative = int(support.get("negative", 0) or 0) if isinstance(support, Mapping) else 0
        if negative >= 3 and positive == 0:
            status = "retired"
        elif negative > positive:
            status = "contradicted"
        elif positive > negative:
            status = "supported"
        else:
            status = "needs_probe"
        return status, positive, negative, _hypothesis_confidence(positive, negative)

    def _rebind_trace_summary(self, rebound: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for item in rebound:
            status = str(item.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "rebound_count": len(rebound),
            "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        }

    def _rebind_stale_action_semantics_after_boundary(self, boundary: str) -> None:
        rebound: list[dict[str, Any]] = []
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "action_semantics" or record.get("status") != "stale_no_progress":
                continue
            status, positive, negative, confidence = self._status_from_support_counts(record)
            record["status"] = status
            record["confidence"] = confidence
            rebound.append(
                {
                    "action": record.get("action"),
                    "status": status,
                    "positive": positive,
                    "negative": negative,
                }
            )
        if rebound:
            self.trace_notes.append(
                {
                    "policy": "mini-palari boundary action-family rebind",
                    **self._rebind_trace_summary(rebound),
                    "boundary": boundary,
                    "rebound_action_semantics": list(_trace_value(rebound)),
                    "decision": "restore_phase_local_stale_action_semantics_from_support_counts",
                    "claim_boundary": "boundary_rebind_is_not_level_success_claim",
                }
            )

    def _rebind_stale_objective_bindings_after_boundary(self, boundary: str) -> None:
        rebound: list[dict[str, Any]] = []
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "objective_binding" or record.get("status") != "stale_no_progress":
                continue
            status, positive, negative, confidence = self._status_from_support_counts(record)
            record["status"] = status
            record["confidence"] = confidence
            rebound.append(
                {
                    "action": record.get("action"),
                    "target_key": self._objective_target_key(record.get("target_candidate")),
                    "status": status,
                    "positive": positive,
                    "negative": negative,
                }
            )
        if rebound:
            self.trace_notes.append(
                {
                    "policy": "mini-palari boundary objective-binding rebind",
                    **self._rebind_trace_summary(rebound),
                    "boundary": boundary,
                    "rebound_objective_bindings": list(_trace_value(rebound)),
                    "decision": "restore_phase_local_stale_objective_bindings_from_support_counts",
                    "claim_boundary": "boundary_rebind_is_not_level_success_claim",
                }
            )

    def _remember_emitted_action(
        self,
        action_name: str,
        latest_frame: Any,
        *,
        preserve_stale_evidence: bool = False,
        count_phase_breaker_reset: bool = True,
    ) -> None:
        if action_name == "RESET":
            self.last_probe_action = None
            self.last_action_snapshot = None
            self.last_probe_progress_label = None
            self.last_action6_coordinate = None
            self.active_plan = None
            self.pending_plan_continuation = None
            if preserve_stale_evidence:
                if count_phase_breaker_reset:
                    self.phase_breaker_resets_without_progress += 1
                self.pending_phase_breaker_recovery = True
            else:
                self.action_no_progress_repeats = {}
                self.objective_no_progress_repeats = {}
                self.exhausted_objective_targets = set()
                self.action6_no_delta_coordinates = set()
                self.phase_breaker_resets_without_progress = 0
                self.pending_phase_breaker_recovery = False
                self._rebind_stale_action_semantics_after_boundary("reset")
                self._rebind_stale_objective_bindings_after_boundary("reset")
            return
        self.pending_phase_breaker_recovery = False
        if action_name != "ACTION6":
            self.last_action6_coordinate = None
        self.last_probe_action = action_name
        self.last_action_snapshot = self._snapshot_for_result_tracking(latest_frame)

    def _upsert_hypothesis(
        self,
        *,
        kind: str,
        label: str,
        action: str | None,
        positive_delta: bool,
        evidence: Mapping[str, Any],
        scope: Sequence[str],
        extra_fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        hyp_id = _stable_id(
            "subhyp",
            {
                "game_id": self.normalized_game_id,
                "kind": kind,
                "action": action,
                "scope": list(scope),
            },
        )
        existing = self.hypothesis_bank.get(hyp_id)
        support = dict(existing.get("support", {})) if existing else {"positive": 0, "negative": 0}
        counter_evidence = list(existing.get("counter_evidence", [])) if existing else []
        if positive_delta:
            support["positive"] = int(support.get("positive", 0)) + 1
        else:
            support["negative"] = int(support.get("negative", 0)) + 1
            counter_evidence.append(
                {
                    "kind": "no_visible_delta",
                    "action": action,
                    "changed_cells": int(evidence.get("changed_cells", 0) or 0),
                    "trace_ref": f"turn:{self.turn_index}:action:{action}",
                }
            )
        positive = int(support["positive"])
        negative = int(support["negative"])
        confidence = _hypothesis_confidence(positive, negative)
        if negative >= 3 and positive == 0:
            status = "retired"
        elif negative > positive:
            status = "contradicted"
        elif positive > negative:
            status = "supported"
        else:
            status = "needs_probe"
        record = {
            "schema": SUBMISSION_HYPOTHESIS_SCHEMA,
            "id": hyp_id,
            "kind": kind,
            "label": label,
            "action": action,
            "confidence": confidence,
            "support": support,
            "counter_evidence": counter_evidence,
            "evidence": dict(_trace_value(evidence)),
            "scope": list(scope),
            "status": status,
            "authority": _inert_authority_record(),
            "source_flags": ["submission_hypothesis_layer", "frame_delta_probe"],
        }
        if extra_fields:
            record.update(dict(_trace_value(extra_fields)))
        self.hypothesis_bank[hyp_id] = record
        if len(self.hypothesis_bank) > self.max_hypotheses:
            ranked_ids = sorted(
                self.hypothesis_bank,
                key=lambda item_id: (-float(self.hypothesis_bank[item_id]["confidence"]), item_id),
            )
            self.hypothesis_bank = {item_id: self.hypothesis_bank[item_id] for item_id in ranked_ids[: self.max_hypotheses]}
        return record

    def _prediction_for_action(self, action: str) -> str:
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "action_semantics" or record.get("action") != action:
                continue
            support = record.get("support", {})
            positive = int(support.get("positive", 0))
            negative = int(support.get("negative", 0))
            if positive > negative:
                return "changed_from_prior_evidence"
            if negative > positive:
                return "no_visible_delta_from_prior_evidence"
        return "unknown_no_prior_evidence"

    def _sidecar_evidence_counts(self) -> tuple[int, int]:
        positive = 0
        negative = 0
        for record in self.hypothesis_bank.values():
            if record.get("kind") in {"action_semantics", "game_family", "objective_binding", "progress_signal"}:
                support = record.get("support", {})
                positive += int(support.get("positive", 0) or 0)
                negative += int(support.get("negative", 0) or 0)
        return positive, negative

    def _candidate_plan_for_action(self, action: str) -> dict[str, Any] | None:
        if not self.enable_bounded_planning:
            return None
        if self._action_hypothesis_status(action) != "supported":
            return None
        prediction = self._prediction_for_action(action)
        if prediction != "changed_from_prior_evidence":
            return None
        supporting_refs = [
            record["id"]
            for record in self.hypothesis_bank.values()
            if record.get("kind") == "action_semantics"
            and record.get("action") == action
            and record.get("status") == "supported"
        ]
        if not supporting_refs:
            return None
        payload = {
            "actions": [action, action, action],
            "hypothesis_refs": supporting_refs,
            "predicted_outcomes": [prediction, prediction, prediction],
            "max_depth": 3,
            "authority": "candidate_only",
        }
        payload["schema"] = "mini-palari.submission-plan-candidate.v0.1"
        payload["plan_id"] = _stable_id("subplan", {"game_id": self.normalized_game_id, "turn_index": self.turn_index, **payload})
        return payload

    def _supported_motion_vectors(self, non_reset: Sequence[str]) -> dict[str, dict[str, int]]:
        vectors: dict[str, dict[str, int]] = {}
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "action_semantics" or record.get("status") != "supported":
                continue
            action = str(record.get("action"))
            if action not in non_reset or self.action_no_progress_repeats.get(action, 0) >= self.max_no_progress_action_repeats:
                continue
            signature = record.get("delta_signature", {})
            shift = signature.get("bbox_shift") if isinstance(signature, Mapping) else None
            if not isinstance(shift, Mapping):
                continue
            dx = int(shift.get("dx", 0) or 0)
            dy = int(shift.get("dy", 0) or 0)
            if dx or dy:
                vectors[action] = {"dx": dx, "dy": dy}
        return vectors

    def _objective_target_key(self, target: Mapping[str, Any] | None) -> str | None:
        if not isinstance(target, Mapping):
            return None
        try:
            return f"{int(target.get('x', -1) or -1)}:{int(target.get('y', -1) or -1)}:{int(target.get('color', -1) or -1)}"
        except (TypeError, ValueError):
            return None

    def _candidate_target_from_points(
        self,
        points: Sequence[tuple[int, int, int]],
        *,
        object_x: float,
        object_y: float,
        min_distance: int = 1,
    ) -> dict[str, Any] | None:
        target_points = [(x, y, cell) for x, y, cell in points if abs(x - object_x) + abs(y - object_y) >= min_distance]
        if not target_points:
            return None
        color_counts: dict[int, int] = {}
        for _x, _y, cell in points:
            color_counts[cell] = color_counts.get(cell, 0) + 1
        for target_x, target_y, target_color in sorted(
            target_points,
            key=lambda item: (color_counts.get(item[2], 0), -(abs(item[0] - object_x) + abs(item[1] - object_y)), item[1], item[0]),
        ):
            candidate = {
                "x": int(target_x),
                "y": int(target_y),
                "color": int(target_color),
                "selection_basis": "rare_color_then_far_from_controlled_object_candidate",
            }
            if self._objective_target_key(candidate) not in self.exhausted_objective_targets:
                return candidate
        return None

    def _supported_objective_target(
        self,
        points: Sequence[tuple[int, int, int]],
        *,
        object_x: float,
        object_y: float,
    ) -> dict[str, Any] | None:
        visible = {(x, y, color) for x, y, color in points}
        self._rebind_target_scoped_stale_objective_bindings_for_visible_targets(
            points,
            object_x=object_x,
            object_y=object_y,
        )
        supported: list[tuple[float, str, dict[str, Any]]] = []
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "objective_binding" or record.get("status") != "supported":
                continue
            target = record.get("target_candidate")
            if not isinstance(target, Mapping):
                continue
            x = int(target.get("x", -1) or -1)
            y = int(target.get("y", -1) or -1)
            color = int(target.get("color", -1) or -1)
            if self._objective_target_key(target) in self.exhausted_objective_targets:
                continue
            if (x, y, color) not in visible or abs(x - object_x) + abs(y - object_y) < 1:
                continue
            supported.append((float(record.get("confidence", 0.0) or 0.0), str(record.get("id", "")), dict(_trace_value(target))))
        if not supported:
            return None
        return sorted(supported, key=lambda item: (-item[0], item[1]))[0][2]

    def _navigation_object_anchor(self, points: Sequence[tuple[int, int, int]]) -> tuple[float, float, str] | None:
        visible_cells = {(x, y) for x, y, _color in points}
        historical: list[tuple[float, float]] = []
        for record in self.hypothesis_bank.values():
            signature = record.get("delta_signature", {})
            if not isinstance(signature, Mapping):
                continue
            shift = signature.get("bbox_shift")
            if not isinstance(shift, Mapping):
                continue
            if not (int(shift.get("dx", 0) or 0) or int(shift.get("dy", 0) or 0)):
                continue
            after = signature.get("after_changed_centroid") or signature.get("after_centroid")
            if isinstance(after, Mapping):
                historical.append((float(after.get("x", 0.0) or 0.0), float(after.get("y", 0.0) or 0.0)))

        stale_historical = 0
        for anchor_x, anchor_y in reversed(historical):
            rounded_anchor = (int(round(anchor_x)), int(round(anchor_y)))
            if rounded_anchor in visible_cells:
                return anchor_x, anchor_y, "visible_historical_changed_region"
            stale_historical += 1

        center = _centroid(points)
        if center is None:
            return None
        source = "current_frame_centroid"
        if stale_historical:
            source = "current_frame_centroid_after_stale_historical_anchor"
        return center[0], center[1], source

    def _target_excluded_navigation_anchor(
        self,
        points: Sequence[tuple[int, int, int]],
        target: Mapping[str, Any],
    ) -> tuple[float, float, str] | None:
        try:
            target_cell = (int(target.get("x", -1) or -1), int(target.get("y", -1) or -1))
        except (TypeError, ValueError):
            return None
        non_target_points = [(x, y, color) for x, y, color in points if (x, y) != target_cell]
        if not non_target_points:
            return None
        visible_non_target = {(x, y) for x, y, _color in non_target_points}
        historical: list[tuple[float, float]] = []
        for record in self.hypothesis_bank.values():
            signature = record.get("delta_signature", {})
            if not isinstance(signature, Mapping):
                continue
            shift = signature.get("bbox_shift")
            if not isinstance(shift, Mapping):
                continue
            if not (int(shift.get("dx", 0) or 0) or int(shift.get("dy", 0) or 0)):
                continue
            after = signature.get("after_changed_centroid") or signature.get("after_centroid")
            if isinstance(after, Mapping):
                historical.append((float(after.get("x", 0.0) or 0.0), float(after.get("y", 0.0) or 0.0)))

        for anchor_x, anchor_y in reversed(historical):
            rounded_anchor = (int(round(anchor_x)), int(round(anchor_y)))
            if rounded_anchor in visible_non_target:
                return anchor_x, anchor_y, "visible_historical_changed_region"

        if historical:
            anchor_x, anchor_y = historical[-1]
            nearest_x, nearest_y, _color = sorted(
                non_target_points,
                key=lambda point: (abs(point[0] - anchor_x) + abs(point[1] - anchor_y), point[1], point[0]),
            )[0]
            return float(nearest_x), float(nearest_y), "nearest_visible_non_target_after_stale_historical_anchor"

        center = _centroid(non_target_points)
        if center is None:
            return None
        return center[0], center[1], "current_frame_non_target_centroid"

    def _target_distance_progress(self, previous_frame: Any, latest_frame: Any) -> dict[str, Any] | None:
        signature = _delta_signature(_field(previous_frame, "frame"), _field(latest_frame, "frame"))
        before_changed = signature.get("before_changed_centroid") if isinstance(signature, Mapping) else None
        after_changed = signature.get("after_changed_centroid") if isinstance(signature, Mapping) else None
        if not isinstance(before_changed, Mapping) or not isinstance(after_changed, Mapping):
            return None
        before_x = float(before_changed.get("x", 0.0) or 0.0)
        before_y = float(before_changed.get("y", 0.0) or 0.0)
        after_x = float(after_changed.get("x", 0.0) or 0.0)
        after_y = float(after_changed.get("y", 0.0) or 0.0)
        points = _nonzero_points(_field(latest_frame, "frame"))
        if len(points) < 2:
            return None
        target = self._supported_objective_target(points, object_x=after_x, object_y=after_y) or self._candidate_target_from_points(points, object_x=after_x, object_y=after_y)
        if target is None:
            return None
        target_x = int(target["x"])
        target_y = int(target["y"])
        before_distance = abs(target_x - before_x) + abs(target_y - before_y)
        after_distance = abs(target_x - after_x) + abs(target_y - after_y)
        if after_distance < before_distance:
            label = "candidate_target_distance_decreased"
        elif after_distance > before_distance:
            label = "candidate_target_distance_increased"
        else:
            label = "candidate_target_distance_unchanged"
        return {
            "progress_signal": label,
            "target_candidate": dict(_trace_value(target)),
            "distance_before": before_distance,
            "distance_after": after_distance,
            "object_centroid_before": {"x": round(before_x, 3), "y": round(before_y, 3)},
            "object_centroid_after": {"x": round(after_x, 3), "y": round(after_y, 3)},
            "claim_boundary": "candidate_only_no_success_claim",
        }

    def _record_objective_exhaustion_if_needed(
        self,
        action: str,
        progress_label: str,
        distance_progress: Mapping[str, Any] | None,
        available: Sequence[str],
    ) -> None:
        if progress_label == "level_completion_increased":
            self.objective_no_progress_repeats = {}
            self.exhausted_objective_targets = set()
            return
        if progress_label != "candidate_target_distance_decreased":
            return
        if self.action_no_progress_repeats.get(action, 0) < self.max_no_progress_action_repeats:
            return
        target = distance_progress.get("target_candidate") if isinstance(distance_progress, Mapping) else None
        target_key = self._objective_target_key(target)
        if target_key is None:
            return
        exhausted_actions = self.objective_no_progress_repeats.setdefault(target_key, set())
        exhausted_actions.add(action)
        objective_records = [
            record
            for record in self.hypothesis_bank.values()
            if record.get("kind") == "objective_binding" and self._objective_target_key(record.get("target_candidate")) == target_key
        ]
        if not objective_records:
            return
        candidate_actions = {
            str(record.get("action"))
            for record in objective_records
            if record.get("action") in available and record.get("action") is not None
        } or {action}
        stale_actions = {
            name
            for name in candidate_actions
            if self.action_no_progress_repeats.get(name, 0) >= self.max_no_progress_action_repeats
        }
        if not candidate_actions.issubset(stale_actions):
            return
        self.exhausted_objective_targets.add(target_key)
        counter = {
            "kind": "objective_no_level_progress",
            "action": action,
            "progress_signal": progress_label,
            "target_key": target_key,
            "exhausted_actions": sorted(stale_actions),
            "claim_boundary": "candidate_distance_progress_is_not_level_success",
            "trace_ref": f"turn:{self.turn_index}:action:{action}",
        }
        for record in objective_records:
            support = dict(record.get("support", {}))
            support["negative"] = int(support.get("negative", 0) or 0) + 1
            record["support"] = support
            record["counter_evidence"] = list(record.get("counter_evidence", [])) + [dict(counter)]
            record["status"] = "stale_no_progress"
            record["confidence"] = _hypothesis_confidence(
                int(support.get("positive", 0) or 0), int(support.get("negative", 0) or 0)
            )
        action_counter = {
            "kind": "objective_action_without_level_completion",
            "trigger_action": action,
            "progress_signal": progress_label,
            "target_key": target_key,
            "exhausted_actions": sorted(stale_actions),
            "claim_boundary": "candidate_distance_progress_is_not_level_success",
            "trace_ref": f"turn:{self.turn_index}:action:{action}",
        }
        action_records = [
            record
            for record in self.hypothesis_bank.values()
            if record.get("kind") == "action_semantics" and record.get("action") in stale_actions
        ]
        for record in action_records:
            support = dict(record.get("support", {}))
            support["negative"] = int(support.get("negative", 0) or 0) + 1
            record["support"] = support
            record_counter = dict(action_counter)
            record_counter["action"] = record.get("action")
            record["counter_evidence"] = list(record.get("counter_evidence", [])) + [record_counter]
            record["status"] = "stale_no_progress"
            record["confidence"] = _hypothesis_confidence(
                int(support.get("positive", 0) or 0), int(support.get("negative", 0) or 0)
            )
        self.trace_notes.append(
            {
                "policy": "mini-palari objective action-family transfer",
                "action": action,
                "target_key": target_key,
                "exhausted_actions": sorted(stale_actions),
                "affected_action_semantics": len(action_records),
                "decision": "retire_objective_action_semantics_without_level_completion",
                "claim_boundary": "candidate_distance_progress_is_not_level_success",
            }
        )
        self.trace_notes.append(
            {
                "policy": "mini-palari objective authority exhaustion",
                "action": action,
                "target_key": target_key,
                "exhausted_actions": sorted(stale_actions),
                "decision": "demote_objective_binding_without_level_completion",
                "claim_boundary": "candidate_distance_progress_is_not_level_success",
            }
        )

    def _record_terminal_contact_target_rebinding_if_needed(
        self, action: str, plan_progress_evidence: Mapping[str, Any] | None
    ) -> None:
        if not isinstance(plan_progress_evidence, Mapping):
            return
        target = plan_progress_evidence.get("target_candidate")
        target_key = self._objective_target_key(target)
        if target_key is None:
            return
        was_exhausted = target_key in self.exhausted_objective_targets
        self.exhausted_objective_targets.add(target_key)
        counter = {
            "kind": "terminal_contact_without_level_completion",
            "action": action,
            "progress_signal": TERMINAL_CONTACT_PROGRESS_SIGNAL,
            "target_key": target_key,
            "plan_id": plan_progress_evidence.get("plan_id"),
            "parent_plan_id": plan_progress_evidence.get("parent_plan_id"),
            "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
            "trace_ref": f"turn:{self.turn_index}:action:{action}",
        }
        objective_records = [
            record
            for record in self.hypothesis_bank.values()
            if (
                record.get("kind") == "objective_binding"
                and self._objective_target_key(record.get("target_candidate")) == target_key
            )
        ]
        for record in objective_records:
            support = dict(record.get("support", {}))
            support["negative"] = int(support.get("negative", 0) or 0) + 1
            record["support"] = support
            record["counter_evidence"] = list(record.get("counter_evidence", [])) + [dict(counter)]
            record["status"] = "stale_no_progress"
            record["confidence"] = _hypothesis_confidence(
                int(support.get("positive", 0) or 0), int(support.get("negative", 0) or 0)
            )
        self.trace_notes.append(
            {
                "policy": "mini-palari terminal contact target rebinding",
                "action": action,
                "target_key": target_key,
                "target_candidate": dict(_trace_value(target)) if isinstance(target, Mapping) else None,
                "affected_objective_bindings": len(objective_records),
                "decision": "keep_contacted_candidate_target_exhausted_without_level_completion"
                if was_exhausted
                else "retire_contacted_candidate_target_without_level_completion",
                "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
            }
        )

    def _record_terminal_contact_action_family_transfer_if_needed(
        self, action: str, plan_progress_evidence: Mapping[str, Any] | None
    ) -> None:
        if not isinstance(plan_progress_evidence, Mapping):
            return
        self.action_no_progress_repeats[action] = max(
            self.action_no_progress_repeats.get(action, 0), self.max_no_progress_action_repeats
        )
        target = plan_progress_evidence.get("target_candidate")
        target_key = self._objective_target_key(target)
        counter = {
            "kind": "terminal_contact_action_without_level_completion",
            "action": action,
            "progress_signal": TERMINAL_CONTACT_PROGRESS_SIGNAL,
            "target_key": target_key,
            "plan_id": plan_progress_evidence.get("plan_id"),
            "parent_plan_id": plan_progress_evidence.get("parent_plan_id"),
            "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
            "trace_ref": f"turn:{self.turn_index}:action:{action}",
        }
        action_records = [
            record
            for record in self.hypothesis_bank.values()
            if record.get("kind") == "action_semantics" and record.get("action") == action
        ]
        for record in action_records:
            support = dict(record.get("support", {}))
            support["negative"] = int(support.get("negative", 0) or 0) + 1
            record["support"] = support
            record["counter_evidence"] = list(record.get("counter_evidence", [])) + [dict(counter)]
            record["status"] = "stale_no_progress"
            record["confidence"] = _hypothesis_confidence(
                int(support.get("positive", 0) or 0), int(support.get("negative", 0) or 0)
            )
        self.trace_notes.append(
            {
                "policy": "mini-palari terminal contact action-family transfer",
                "action": action,
                "target_key": target_key,
                "target_candidate": dict(_trace_value(target)) if isinstance(target, Mapping) else None,
                "affected_action_semantics": len(action_records),
                "repeat_count": self.action_no_progress_repeats.get(action, 0),
                "decision": "retire_contact_action_semantics_without_level_completion",
                "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
            }
        )

    def _record_inverse_action_pairs(self, action: str, dx: int, dy: int, signature: Mapping[str, Any]) -> None:
        if not (dx or dy):
            return
        for record in list(self.hypothesis_bank.values()):
            other_action = record.get("action")
            if not other_action or other_action == action:
                continue
            movement = record.get("movement_vector")
            if not isinstance(movement, Mapping):
                other_signature = record.get("delta_signature", {})
                shift = other_signature.get("bbox_shift") if isinstance(other_signature, Mapping) else None
                movement = shift if isinstance(shift, Mapping) else None
            if not isinstance(movement, Mapping):
                continue
            other_dx = int(movement.get("dx", 0) or 0)
            other_dy = int(movement.get("dy", 0) or 0)
            if other_dx == -dx and other_dy == -dy:
                self._upsert_hypothesis(
                    kind="action_semantics",
                    label="inverse_action_pair",
                    action=action,
                    positive_delta=True,
                    evidence={
                        "action": action,
                        "inverse_action": str(other_action),
                        "movement_vector": {"dx": dx, "dy": dy},
                        "inverse_movement_vector": {"dx": other_dx, "dy": other_dy},
                        "delta_signature": signature,
                    },
                    scope=[f"action_pair:{action}:{other_action}", "inverse_motion_vectors"],
                    extra_fields={
                        "paired_action": str(other_action),
                        "movement_vector": {"dx": dx, "dy": dy},
                        "inverse_movement_vector": {"dx": other_dx, "dy": other_dy},
                    },
                )
                return

    def _rebind_target_scoped_stale_objective_bindings_for_visible_targets(
        self,
        points: Sequence[tuple[int, int, int]],
        *,
        object_x: float,
        object_y: float,
    ) -> None:
        visible_target_keys = {
            f"{x}:{y}:{color}"
            for x, y, color in points
            if abs(x - object_x) + abs(y - object_y) >= 1
        }
        if not visible_target_keys:
            return
        rebound: list[dict[str, Any]] = []
        target_scoped_kinds = {
            "objective_no_level_progress",
            "terminal_contact_without_level_completion",
        }
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "objective_binding" or record.get("status") != "stale_no_progress":
                continue
            target = record.get("target_candidate")
            target_key = self._objective_target_key(target)
            if target_key is None or target_key not in visible_target_keys or target_key in self.exhausted_objective_targets:
                continue
            raw_counters = record.get("counter_evidence", [])
            counters = raw_counters if isinstance(raw_counters, Sequence) and not isinstance(raw_counters, (str, bytes)) else []
            stale_target_keys = {
                str(counter.get("target_key"))
                for counter in counters
                if isinstance(counter, Mapping)
                and counter.get("kind") in target_scoped_kinds
                and counter.get("target_key") is not None
            }
            if not stale_target_keys or target_key in stale_target_keys:
                continue
            status, positive, negative, confidence = self._status_from_support_counts(record)
            record["status"] = status
            record["confidence"] = confidence
            rebound.append(
                {
                    "action": record.get("action"),
                    "target_key": target_key,
                    "target_candidate": dict(_trace_value(target)) if isinstance(target, Mapping) else None,
                    "status": status,
                    "positive": positive,
                    "negative": negative,
                    "stale_target_keys": sorted(stale_target_keys),
                }
            )
        if rebound:
            self.trace_notes.append(
                {
                    "policy": "mini-palari cross-target objective-binding rebind",
                    **self._rebind_trace_summary(rebound),
                    "visible_target_keys": sorted(visible_target_keys),
                    "rebound_objective_bindings": list(_trace_value(rebound)),
                    "decision": "restore_target_scoped_stale_objective_bindings_for_visible_targets",
                    "claim_boundary": "cross_target_rebind_is_not_level_success_claim",
                }
            )

    def _rebind_target_scoped_stale_actions_for_target(self, target: Mapping[str, Any]) -> None:
        target_key = self._objective_target_key(target)
        if target_key is None:
            return
        rebound: list[dict[str, Any]] = []
        target_scoped_kinds = {
            "objective_action_without_level_completion",
            "terminal_contact_action_without_level_completion",
        }
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "action_semantics" or record.get("status") != "stale_no_progress":
                continue
            raw_counters = record.get("counter_evidence", [])
            counters = raw_counters if isinstance(raw_counters, Sequence) and not isinstance(raw_counters, (str, bytes)) else []
            stale_target_keys = {
                str(counter.get("target_key"))
                for counter in counters
                if isinstance(counter, Mapping)
                and counter.get("kind") in target_scoped_kinds
                and counter.get("target_key") is not None
            }
            if not stale_target_keys or target_key in stale_target_keys:
                continue
            status, positive, negative, confidence = self._status_from_support_counts(record)
            record["status"] = status
            record["confidence"] = confidence
            action = record.get("action")
            if action is not None:
                self.action_no_progress_repeats.pop(str(action), None)
            rebound.append(
                {
                    "action": action,
                    "status": status,
                    "positive": positive,
                    "negative": negative,
                    "stale_target_keys": sorted(stale_target_keys),
                    "current_target_key": target_key,
                }
            )
        if rebound:
            self.trace_notes.append(
                {
                    "policy": "mini-palari cross-target action-family rebind",
                    **self._rebind_trace_summary(rebound),
                    "target_key": target_key,
                    "target_candidate": dict(_trace_value(target)),
                    "rebound_action_semantics": list(_trace_value(rebound)),
                    "decision": "restore_target_scoped_stale_action_semantics_for_new_target",
                    "claim_boundary": "cross_target_rebind_is_not_level_success_claim",
                }
            )

    def _navigation_target_action(self, latest_frame: Any, non_reset: Sequence[str]) -> tuple[str, dict[str, Any]] | None:
        """Choose a supported non-click action that moves a candidate object toward a visible target.

        This is a tiny deterministic controller, not a game-id script. It only
        acts after live probe evidence has supported one or more motion vectors,
        and it treats the objective as candidate-only unless level progress is
        later observed.
        """

        points = _nonzero_points(_field(latest_frame, "frame"))
        if len(points) < 2:
            return None
        anchor = self._navigation_object_anchor(points)
        if anchor is None:
            return None
        object_x, object_y, object_anchor_source = anchor
        # Prefer a supported objective hypothesis when one remains visible; if
        # none exists yet, create a candidate-only target from the public frame.
        target = self._supported_objective_target(points, object_x=object_x, object_y=object_y) or self._candidate_target_from_points(points, object_x=object_x, object_y=object_y)
        if target is None:
            return None
        self._rebind_target_scoped_stale_actions_for_target(target)
        vectors = self._supported_motion_vectors(non_reset)
        if not vectors:
            return None
        target_x = int(target["x"])
        target_y = int(target["y"])
        target_color = int(target["color"])
        refined_anchor = self._target_excluded_navigation_anchor(points, target)
        if refined_anchor is not None:
            object_x, object_y, object_anchor_source = refined_anchor
        before_distance = abs(target_x - object_x) + abs(target_y - object_y)
        ranked: list[tuple[float, str, dict[str, int], float]] = []
        for action, vector in vectors.items():
            after_x = object_x + vector["dx"]
            after_y = object_y + vector["dy"]
            after_distance = abs(target_x - after_x) + abs(target_y - after_y)
            if after_distance < before_distance:
                ranked.append((after_distance, action, vector, before_distance))
        if not ranked:
            return None
        _distance, action, vector, before = sorted(ranked, key=lambda item: (item[0], item[1]))[0]
        explanation = {
            "policy": "mini-palari bounded navigation/object-target controller",
            "action": action,
            "object_centroid": {"x": round(object_x, 3), "y": round(object_y, 3)},
            "object_anchor_source": object_anchor_source,
            "target_candidate": {"x": int(target_x), "y": int(target_y), "color": int(target_color)},
            "motion_vector": dict(vector),
            "distance_before": before,
            "distance_after": _distance,
            "authority": "evidence_gated_policy_branch",
            "claim_boundary": "candidate_target_only_until_progress_signal",
        }
        return action, explanation

    def _short_horizon_navigation_action(
        self,
        latest_frame: Any,
        non_reset: Sequence[str],
        *,
        avoid_first_actions: set[str] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Choose the first step of a bounded two-action vector plan.

        The planner is candidate-only: it uses already-supported motion vectors
        and a visible target candidate, emits only the first action, and records
        the proposed two-step path without claiming objective success.
        """

        points = _nonzero_points(_field(latest_frame, "frame"))
        if len(points) < 2:
            return None
        anchor = self._navigation_object_anchor(points)
        if anchor is None:
            return None
        object_x, object_y, object_anchor_source = anchor
        target = self._supported_objective_target(points, object_x=object_x, object_y=object_y) or self._candidate_target_from_points(points, object_x=object_x, object_y=object_y)
        if target is None:
            return None
        self._rebind_target_scoped_stale_actions_for_target(target)
        vectors = self._supported_motion_vectors(non_reset)
        if len(vectors) < 2:
            return None
        refined_anchor = self._target_excluded_navigation_anchor(points, target)
        if refined_anchor is not None:
            object_x, object_y, object_anchor_source = refined_anchor
        target_x = int(target["x"])
        target_y = int(target["y"])
        target_color = int(target["color"])
        before_distance = abs(target_x - object_x) + abs(target_y - object_y)
        blocked_first = avoid_first_actions or set()

        # Preserve the single-step controller's priority. This branch only
        # handles cases where no supported individual motion vector improves
        # the candidate-target distance.
        for vector in vectors.values():
            after_x = object_x + vector["dx"]
            after_y = object_y + vector["dy"]
            if abs(target_x - after_x) + abs(target_y - after_y) < before_distance:
                return None

        ranked: list[tuple[float, float, str, str, dict[str, int], dict[str, int]]] = []
        for first_action, first_vector in vectors.items():
            if first_action in blocked_first:
                continue
            first_x = object_x + first_vector["dx"]
            first_y = object_y + first_vector["dy"]
            first_distance = abs(target_x - first_x) + abs(target_y - first_y)
            for second_action, second_vector in vectors.items():
                final_x = first_x + second_vector["dx"]
                final_y = first_y + second_vector["dy"]
                final_distance = abs(target_x - final_x) + abs(target_y - final_y)
                if final_distance < before_distance:
                    ranked.append((final_distance, first_distance, first_action, second_action, dict(first_vector), dict(second_vector)))
        if not ranked:
            return None

        final_distance, first_distance, first_action, second_action, first_vector, second_vector = sorted(
            ranked,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )[0]
        reasoning = {
            "policy": "mini-palari short-horizon vector planner",
            "action": first_action,
            "planned_actions": [first_action, second_action],
            "object_centroid": {"x": round(object_x, 3), "y": round(object_y, 3)},
            "object_anchor_source": object_anchor_source,
            "target_candidate": {"x": target_x, "y": target_y, "color": target_color},
            "motion_vectors": [first_vector, second_vector],
            "distance_before": before_distance,
            "distance_after_first": first_distance,
            "distance_after_plan": final_distance,
            "max_depth": 2,
            "authority": "candidate_only_evidence_gated_policy_branch",
            "claim_boundary": "two_step_candidate_plan_is_not_level_success",
        }
        reasoning["plan_id"] = _stable_id(
            "subplan",
            {
                "game_id": self.normalized_game_id,
                "turn_index": self.turn_index,
                "planned_actions": reasoning["planned_actions"],
                "target_candidate": reasoning["target_candidate"],
                "motion_vectors": reasoning["motion_vectors"],
                "max_depth": reasoning["max_depth"],
            },
        )
        return first_action, reasoning

    def _activate_short_horizon_plan(self, reasoning: Mapping[str, Any]) -> None:
        planned_actions = reasoning.get("planned_actions")
        motion_vectors = reasoning.get("motion_vectors")
        if not isinstance(planned_actions, Sequence) or isinstance(planned_actions, (str, bytes)) or not planned_actions:
            return
        if not isinstance(motion_vectors, Sequence) or isinstance(motion_vectors, (str, bytes)) or not motion_vectors:
            return
        first_vector = motion_vectors[0]
        if not isinstance(first_vector, Mapping):
            return
        self.pending_plan_continuation = None
        self.active_plan = {
            "schema": "mini-palari.short-horizon-vector-plan.v0.1",
            "plan_kind": "short_horizon_vector_navigation",
            "plan_id": str(reasoning.get("plan_id") or _stable_id("subplan", dict(_trace_value(reasoning)))),
            "actions": [str(action) for action in planned_actions],
            "predicted_outcomes": ["changed_from_prior_evidence" for _ in planned_actions],
            "motion_vectors": [dict(_trace_value(vector)) for vector in motion_vectors if isinstance(vector, Mapping)],
            "expected_first_motion_vector": dict(_trace_value(first_vector)),
            "expected_distance_before": reasoning.get("distance_before"),
            "expected_distance_after_first": reasoning.get("distance_after_first"),
            "expected_distance_after_plan": reasoning.get("distance_after_plan"),
            "target_candidate": dict(_trace_value(reasoning.get("target_candidate", {}))),
            "authority": "candidate_only",
            "status": "first_step_emitted",
            "claim_boundary": "two_step_candidate_plan_is_not_level_success",
        }

    def _record_short_horizon_plan_result_if_needed(
        self,
        action: str,
        signature: Mapping[str, Any],
        distance_progress: Mapping[str, Any] | None,
        plan: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(plan, Mapping) or plan.get("plan_kind") != "short_horizon_vector_navigation":
            return None
        planned_actions = plan.get("actions")
        if not isinstance(planned_actions, Sequence) or isinstance(planned_actions, (str, bytes)) or not planned_actions:
            return None
        planned_action_names = [str(planned_action) for planned_action in planned_actions]
        if planned_action_names[0] != action:
            return None

        expected_raw = plan.get("expected_first_motion_vector")
        expected_motion = (
            {"dx": int(expected_raw.get("dx", 0) or 0), "dy": int(expected_raw.get("dy", 0) or 0)}
            if isinstance(expected_raw, Mapping)
            else None
        )
        observed_raw = signature.get("bbox_shift")
        observed_motion = (
            {"dx": int(observed_raw.get("dx", 0) or 0), "dy": int(observed_raw.get("dy", 0) or 0)}
            if isinstance(observed_raw, Mapping)
            else None
        )
        expected_distance = plan.get("expected_distance_after_first")
        observed_distance = distance_progress.get("distance_after") if isinstance(distance_progress, Mapping) else None
        distance_matches = True
        if isinstance(expected_distance, (int, float)) and isinstance(observed_distance, (int, float)):
            distance_matches = abs(float(expected_distance) - float(observed_distance)) <= 1e-6
        vector_matches = observed_motion == expected_motion
        is_continuation_step = bool(plan.get("parent_plan_id")) or plan.get("status") == "continuation_step_emitted"
        observed_status = "continuation_step_observed" if is_continuation_step else "first_step_observed"
        mismatch_status = "aborted_continuation_step_mismatch" if is_continuation_step else "aborted_first_step_mismatch"
        plan_status = observed_status if vector_matches and distance_matches else mismatch_status
        continuation_candidate = None
        if plan_status == "continuation_step_observed":
            decision = "clear_completed_continuation_plan"
        elif plan_status == "first_step_observed":
            decision = "clear_observed_candidate_plan"
        elif plan_status == "aborted_continuation_step_mismatch":
            decision = "abort_continuation_candidate_after_step_mismatch"
        else:
            decision = "abort_candidate_plan_after_first_step_mismatch"
        if plan_status == "first_step_observed" and len(planned_action_names) > 1:
            remaining_actions = planned_action_names[1:]
            raw_vectors = plan.get("motion_vectors")
            remaining_vectors: list[dict[str, int]] = []
            if isinstance(raw_vectors, Sequence) and not isinstance(raw_vectors, (str, bytes)):
                for raw_vector in list(raw_vectors)[1:]:
                    if isinstance(raw_vector, Mapping):
                        remaining_vectors.append({"dx": int(raw_vector.get("dx", 0) or 0), "dy": int(raw_vector.get("dy", 0) or 0)})
            expected_next_motion = remaining_vectors[0] if remaining_vectors else None
            if expected_next_motion is not None:
                continuation_candidate = {
                    "schema": "mini-palari.short-horizon-vector-continuation.v0.1",
                    "plan_kind": "short_horizon_vector_navigation_continuation",
                    "plan_id": _stable_id(
                        "subplan-continuation",
                        {
                            "parent_plan_id": plan.get("plan_id"),
                            "remaining_actions": remaining_actions,
                            "remaining_motion_vectors": remaining_vectors,
                        },
                    ),
                    "parent_plan_id": plan.get("plan_id"),
                    "action": remaining_actions[0],
                    "remaining_actions": remaining_actions,
                    "remaining_motion_vectors": remaining_vectors,
                    "expected_motion_vector": expected_next_motion,
                    "expected_distance_before": observed_distance if isinstance(observed_distance, (int, float)) else expected_distance,
                    "expected_distance_after": plan.get("expected_distance_after_plan"),
                    "target_candidate": dict(_trace_value(plan.get("target_candidate", {}))),
                    "authority": "candidate_only_after_observed_first_step",
                    "status": "ready_after_first_step_observed",
                    "claim_boundary": "continued_candidate_plan_is_not_level_success",
                }
                self.pending_plan_continuation = continuation_candidate
                decision = "stage_continuation_candidate_after_observed_first_step"
            else:
                self.pending_plan_continuation = None
        elif plan_status != "first_step_observed":
            self.pending_plan_continuation = None
        note = {
            "policy": "mini-palari short-horizon vector planner result gate",
            "plan_id": plan.get("plan_id"),
            "parent_plan_id": plan.get("parent_plan_id"),
            "plan_step": "continuation" if is_continuation_step else "first_step",
            "action": action,
            "planned_actions": list(_trace_value(planned_action_names)),
            "expected_motion_vector": expected_motion,
            "observed_motion_vector": observed_motion,
            "expected_distance_after_first": expected_distance,
            "observed_distance_after": observed_distance,
            "target_candidate": dict(_trace_value(plan.get("target_candidate", {}))),
            "plan_status": plan_status,
            "decision": decision,
            "claim_boundary": "planner_result_evidence_is_not_level_success",
        }
        if continuation_candidate is not None:
            note["continuation_candidate"] = dict(_trace_value(continuation_candidate))
        self.trace_notes.append(note)
        self.active_plan = None
        return note

    def _target_candidate_visible(self, latest_frame: Any, target: Any) -> bool:
        if not isinstance(target, Mapping):
            return False
        try:
            x = int(target.get("x", -1) or -1)
            y = int(target.get("y", -1) or -1)
            color = int(target.get("color", -1) or -1)
        except (TypeError, ValueError):
            return False
        grid = _latest_grid(_field(latest_frame, "frame"))
        if y < 0 or y >= len(grid):
            return False
        row = grid[y]
        if x < 0 or x >= len(row):
            return False
        return int(row[x] or 0) == color

    def _clear_pending_plan_continuation(self, action: str | None, decision: str) -> None:
        continuation = self.pending_plan_continuation
        if not isinstance(continuation, Mapping):
            self.pending_plan_continuation = None
            return
        self.trace_notes.append(
            {
                "policy": "mini-palari short-horizon vector planner continuation",
                "plan_id": continuation.get("plan_id"),
                "parent_plan_id": continuation.get("parent_plan_id"),
                "action": action or continuation.get("action"),
                "decision": decision,
                "claim_boundary": "continued_candidate_plan_is_not_level_success",
            }
        )
        self.pending_plan_continuation = None

    def _short_horizon_continuation_action(
        self,
        latest_frame: Any,
        non_reset: Sequence[str],
        actions_by_name: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        continuation = self.pending_plan_continuation
        if not isinstance(continuation, Mapping):
            return None
        action_name = str(continuation.get("action") or "")
        if not action_name or action_name not in non_reset or action_name not in actions_by_name:
            self._clear_pending_plan_continuation(action_name or None, "clear_unavailable_continuation_action")
            return None
        if _is_complex_action(actions_by_name.get(action_name)):
            self._clear_pending_plan_continuation(action_name, "clear_complex_continuation_action_without_payload_model")
            return None
        target_candidate = continuation.get("target_candidate")
        if not self._target_candidate_visible(latest_frame, target_candidate):
            self._clear_pending_plan_continuation(action_name, "clear_continuation_after_target_candidate_not_visible")
            return None
        raw_motion = continuation.get("expected_motion_vector")
        if not isinstance(raw_motion, Mapping):
            self._clear_pending_plan_continuation(action_name, "clear_continuation_missing_motion_vector")
            return None
        expected_motion = {"dx": int(raw_motion.get("dx", 0) or 0), "dy": int(raw_motion.get("dy", 0) or 0)}
        expected_after = continuation.get("expected_distance_after")
        self.active_plan = {
            "schema": "mini-palari.short-horizon-vector-plan.v0.1",
            "plan_kind": "short_horizon_vector_navigation",
            "plan_id": str(continuation.get("plan_id") or _stable_id("subplan-continuation", dict(_trace_value(continuation)))),
            "parent_plan_id": continuation.get("parent_plan_id"),
            "actions": [action_name],
            "predicted_outcomes": ["changed_from_prior_evidence"],
            "motion_vectors": [expected_motion],
            "expected_first_motion_vector": expected_motion,
            "expected_distance_before": continuation.get("expected_distance_before"),
            "expected_distance_after_first": expected_after,
            "expected_distance_after_plan": expected_after,
            "target_candidate": dict(_trace_value(target_candidate)),
            "authority": "candidate_only_after_observed_first_step",
            "status": "continuation_step_emitted",
            "claim_boundary": "continued_candidate_plan_is_not_level_success",
        }
        self.pending_plan_continuation = None
        reasoning = {
            "policy": "mini-palari short-horizon vector planner continuation",
            "action": action_name,
            "plan_id": self.active_plan["plan_id"],
            "parent_plan_id": self.active_plan.get("parent_plan_id"),
            "planned_actions": [action_name],
            "motion_vector": expected_motion,
            "target_candidate": dict(_trace_value(target_candidate)),
            "distance_before": continuation.get("expected_distance_before"),
            "distance_after_plan": expected_after,
            "authority": "candidate_only_after_observed_first_step",
            "claim_boundary": "continued_candidate_plan_is_not_level_success",
        }
        return action_name, reasoning

    def _visual_rule_candidate_status(self, non_reset: Sequence[str]) -> str:
        if not self.enable_visual_rule_candidate:
            return "disabled"
        if not self._supported_motion_vectors(non_reset):
            return "enabled_no_supported_predicate_yet"
        return "enabled_supported_predicates_available"

    def _visual_rule_candidate_action(self, latest_frame: Any, non_reset: Sequence[str]) -> tuple[str, dict[str, Any]] | None:
        """Small deterministic candidate distilled from visual-rule predicate work.

        Disabled by default. When enabled for ablation, it uses only already-supported
        local predicates: a motion vector, a supported visible target candidate,
        and a generic contact check. It never reads sidecar artifacts or grants
        sidecar/runtime authority.
        """

        if not self.enable_visual_rule_candidate:
            return None
        vectors = self._supported_motion_vectors(non_reset)
        if not vectors:
            return None
        points = _nonzero_points(_field(latest_frame, "frame"))
        if len(points) < 2:
            return None
        after_centroids: list[tuple[float, float]] = []
        for record in self.hypothesis_bank.values():
            if record.get("kind") not in {"action_semantics", "transition_model"}:
                continue
            signature = record.get("delta_signature", {})
            after = signature.get("after_changed_centroid") if isinstance(signature, Mapping) else None
            if isinstance(after, Mapping):
                after_centroids.append((float(after.get("x", 0.0) or 0.0), float(after.get("y", 0.0) or 0.0)))
        if after_centroids:
            object_x, object_y = after_centroids[-1]
        else:
            center = _centroid(points)
            if center is None:
                return None
            object_x, object_y = center
        target = self._supported_objective_target(points, object_x=object_x, object_y=object_y)
        if target is None:
            return None
        target_x = int(target["x"])
        target_y = int(target["y"])
        target_color = int(target["color"])
        target_tuple = (target_x, target_y, target_color)
        visible_non_target_cells = {(x, y) for x, y, color in points if (x, y, color) != target_tuple}
        before_distance = abs(target_x - object_x) + abs(target_y - object_y)
        supported: list[tuple[float, str, dict[str, int], tuple[int, int]]] = []
        vetoed: list[str] = []
        for action, vector in vectors.items():
            predicted_x = int(round(object_x + vector["dx"]))
            predicted_y = int(round(object_y + vector["dy"]))
            if (predicted_x, predicted_y) in visible_non_target_cells:
                vetoed.append(action)
                continue
            after_distance = abs(target_x - predicted_x) + abs(target_y - predicted_y)
            supported.append((after_distance, action, dict(vector), (predicted_x, predicted_y)))
        if not supported:
            return None
        contact = [item for item in supported if item[0] == 0]
        if contact:
            after_distance, action, vector, predicted = sorted(contact, key=lambda item: (item[1], item[3]))[0]
            return action, {
                "policy": "mini-palari visual-rule candidate target-contact preference",
                "predicate": "target_contact",
                "action": action,
                "object_centroid": {"x": round(object_x, 3), "y": round(object_y, 3)},
                "target_candidate": {"x": target_x, "y": target_y, "color": target_color},
                "predicted_contact_cell": {"x": predicted[0], "y": predicted[1]},
                "motion_vector": vector,
                "distance_before": before_distance,
                "distance_after": after_distance,
                "runtime_authority_granted": False,
                "authority": "evidence_gated_policy_branch",
                "claim_boundary": "candidate_target_contact_is_not_level_success",
            }
        if vetoed:
            after_distance, action, vector, predicted = sorted(supported, key=lambda item: (item[0], item[1]))[0]
            return action, {
                "policy": "mini-palari visual-rule candidate hazard/contact veto",
                "predicate": "hazard_contact",
                "action": action,
                "vetoed_actions": sorted(vetoed),
                "object_centroid": {"x": round(object_x, 3), "y": round(object_y, 3)},
                "target_candidate": {"x": target_x, "y": target_y, "color": target_color},
                "predicted_safe_cell": {"x": predicted[0], "y": predicted[1]},
                "motion_vector": vector,
                "distance_before": before_distance,
                "distance_after": after_distance,
                "runtime_authority_granted": False,
                "authority": "evidence_gated_policy_branch",
                "claim_boundary": "hazard_veto_is_candidate_only_until_score_evidence",
            }
        return None

    def _transition_model_loop_escape(self, previous_frame: Any, latest_frame: Any, action: str) -> dict[str, Any] | None:
        """Retire a supported motion action when its observed transition moves away from the visible target.

        This is still bounded/local: it uses only the last public frame delta plus
        candidate-only target evidence, and it does not claim objective success.
        """

        vector = None
        for record in self.hypothesis_bank.values():
            if record.get("kind") not in {"transition_model", "action_semantics"} or record.get("action") != action:
                continue
            movement = record.get("movement_vector")
            if isinstance(movement, Mapping):
                vector = {"dx": int(movement.get("dx", 0) or 0), "dy": int(movement.get("dy", 0) or 0)}
                break
            signature = record.get("delta_signature", {})
            shift = signature.get("bbox_shift") if isinstance(signature, Mapping) else None
            if isinstance(shift, Mapping):
                vector = {"dx": int(shift.get("dx", 0) or 0), "dy": int(shift.get("dy", 0) or 0)}
                break
        if not vector or not (vector["dx"] or vector["dy"]):
            return None
        previous_points = _nonzero_points(_field(previous_frame, "frame"))
        latest_points = _nonzero_points(_field(latest_frame, "frame"))
        if len(previous_points) < 2 or len(latest_points) < 2:
            return None
        signature = _delta_signature(_field(previous_frame, "frame"), _field(latest_frame, "frame"))
        before_changed = signature.get("before_changed_centroid") if isinstance(signature, Mapping) else None
        after_changed = signature.get("after_changed_centroid") if isinstance(signature, Mapping) else None
        if isinstance(before_changed, Mapping) and isinstance(after_changed, Mapping):
            before_x = float(before_changed.get("x", 0.0) or 0.0)
            before_y = float(before_changed.get("y", 0.0) or 0.0)
            after_x = float(after_changed.get("x", 0.0) or 0.0)
            after_y = float(after_changed.get("y", 0.0) or 0.0)
        else:
            before_center = _centroid(previous_points)
            after_center = _centroid(latest_points)
            if before_center is None or after_center is None:
                return None
            before_x, before_y = before_center
            after_x, after_y = after_center
        target = self._supported_objective_target(latest_points, object_x=after_x, object_y=after_y) or self._candidate_target_from_points(latest_points, object_x=after_x, object_y=after_y)
        if target is None:
            return None
        target_x = int(target["x"])
        target_y = int(target["y"])
        before_distance = abs(target_x - before_x) + abs(target_y - before_y)
        after_distance = abs(target_x - after_x) + abs(target_y - after_y)
        if after_distance <= before_distance:
            return None
        self.action_no_progress_repeats[action] = max(
            self.action_no_progress_repeats.get(action, 0),
            self.max_no_progress_action_repeats,
        )
        return {
            "policy": "mini-palari transition model loop escape",
            "action": action,
            "movement_vector": vector,
            "target_candidate": dict(_trace_value(target)),
            "distance_before": before_distance,
            "distance_after": after_distance,
            "decision": "retire_non_improving_supported_action_and_probe_next",
            "authority": "evidence_gated_policy_branch",
            "claim_boundary": "candidate_target_only_until_progress_signal",
        }

    def _mark_plan_mismatch_if_needed(self, action: str, changed: bool) -> None:
        if not self.active_plan or not self.active_plan.get("actions"):
            return
        expected = self.active_plan.get("predicted_outcomes", ["unknown_no_prior_evidence"])[0]
        expected_changed = expected == "changed_from_prior_evidence"
        if expected_changed != changed:
            self.trace_notes.append(
                {
                    "policy": "mini-palari bounded planner",
                    "action": action,
                    "plan_id": self.active_plan.get("plan_id"),
                    "plan_status": "aborted_prediction_mismatch",
                    "expected": expected,
                    "observed_visible_delta": "changed" if changed else "unchanged",
                }
            )
            self.active_plan = None

    def _action_hypothesis_status(self, action: str) -> str:
        if self.action_no_progress_repeats.get(action, 0) >= self.max_no_progress_action_repeats:
            return "stale_no_progress"
        statuses = [
            str(record.get("status", "needs_probe"))
            for record in self.hypothesis_bank.values()
            if record.get("kind") == "action_semantics" and record.get("action") == action
        ]
        if "retired" in statuses:
            return "retired"
        if "contradicted" in statuses:
            return "contradicted"
        if "supported" in statuses:
            return "supported"
        if statuses:
            return statuses[0]
        return "untested"

    def _remember_action6_coordinate(self, coordinate: Mapping[str, Any] | None) -> None:
        if not isinstance(coordinate, Mapping):
            self.last_action6_coordinate = None
            return
        try:
            self.last_action6_coordinate = (int(coordinate["x"]), int(coordinate["y"]))
        except (KeyError, TypeError, ValueError):
            self.last_action6_coordinate = None

    def _action6_coordinate_transform_candidates(
        self,
        coordinate: Mapping[str, Any],
        frame_payload: Any,
    ) -> list[dict[str, Any]]:
        try:
            base_x = int(coordinate["x"])
            base_y = int(coordinate["y"])
        except (KeyError, TypeError, ValueError):
            return []
        grid = _latest_grid(frame_payload)
        height = max(1, len(grid))
        width = max((len(row) for row in grid), default=1)
        variants: list[dict[str, Any]] = []

        def add(transform: str, x: float | int, y: float | int) -> None:
            transformed = {"x": min(63, max(0, int(round(x)))), "y": min(63, max(0, int(round(y))))}
            if transformed == {"x": base_x, "y": base_y}:
                return
            if any(item["coordinate"] == transformed for item in variants):
                return
            variants.append({"transform": transform, "coordinate": transformed})

        add("swapped_axes", base_y, base_x)
        if width > 1 or height > 1:
            scaled_x = base_x * 63 / max(1, width - 1)
            scaled_y = base_y * 63 / max(1, height - 1)
            add("public_envelope_scaled", scaled_x, scaled_y)
            add("swapped_axes_public_envelope_scaled", scaled_y, scaled_x)
        return variants

    def _next_action6_transform_coordinate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        frame_payload: Any,
    ) -> dict[str, int] | None:
        failed_candidates: list[dict[str, int]] = []
        seen: set[tuple[int, int]] = set()
        for candidate in candidates:
            try:
                key = (int(candidate["x"]), int(candidate["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in self.action6_no_delta_coordinates and key not in seen:
                failed_candidates.append({"x": key[0], "y": key[1]})
                seen.add(key)
        for key in sorted(self.action6_no_delta_coordinates):
            if key not in seen:
                failed_candidates.append({"x": key[0], "y": key[1]})
                seen.add(key)
        for failed in failed_candidates:
            for variant in self._action6_coordinate_transform_candidates(failed, frame_payload):
                coordinate = dict(variant["coordinate"])
                key = (int(coordinate["x"]), int(coordinate["y"]))
                if key in self.action6_tried_coordinates or key in self.action6_no_delta_coordinates:
                    continue
                self.action6_tried_coordinates.add(key)
                self._remember_action6_coordinate(coordinate)
                self.trace_notes.append(
                    {
                        "policy": "mini-palari action6 coordinate transform failover",
                        "base_coordinate": dict(failed),
                        "coordinate": dict(coordinate),
                        "transform": variant["transform"],
                        "decision": "probe_bounded_transform_after_base_coordinate_no_delta_exhaustion",
                        "claim_boundary": "coordinate_transform_probe_is_not_action6_semantics_claim",
                    }
                )
                return coordinate
        return None

    def _action6_coordinate_exhaustion_summary(self, frame_payload: Any) -> dict[str, Any]:
        candidates = salient_coordinate_candidates(frame_payload)
        base_keys: set[tuple[int, int]] = set()
        transform_keys: set[tuple[int, int]] = set()
        expected_order: list[tuple[int, int]] = []

        def remember_expected(key: tuple[int, int]) -> None:
            if key not in expected_order:
                expected_order.append(key)

        for candidate in candidates:
            try:
                key = (int(candidate["x"]), int(candidate["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            base_keys.add(key)
            remember_expected(key)
            for variant in self._action6_coordinate_transform_candidates(candidate, frame_payload):
                coordinate = variant.get("coordinate", {})
                try:
                    transform_key = (int(coordinate["x"]), int(coordinate["y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                transform_keys.add(transform_key)
                remember_expected(transform_key)
        expected_keys = set(expected_order)
        missing_no_delta = [key for key in expected_order if key not in self.action6_no_delta_coordinates]
        next_unresolved = missing_no_delta[0] if missing_no_delta else None
        return {
            "base_candidate_count": len(base_keys),
            "transform_candidate_count": len(transform_keys),
            "no_delta_coordinate_count": len(expected_keys & self.action6_no_delta_coordinates),
            "missing_no_delta_coordinate_count": len(missing_no_delta),
            "next_unresolved_coordinate": (
                {"x": next_unresolved[0], "y": next_unresolved[1]} if next_unresolved is not None else None
            ),
            "exhausted": bool(base_keys) and not missing_no_delta,
        }

    def _next_action6_coordinate(self, frame_payload: Any) -> dict[str, int]:
        candidates = salient_coordinate_candidates(frame_payload)
        if not candidates:
            coordinate = {"x": 0, "y": 0}
            self._remember_action6_coordinate(coordinate)
            return coordinate
        for offset in range(len(candidates)):
            idx = (self.action6_probe_index + offset) % len(candidates)
            candidate = candidates[idx]
            key = (int(candidate["x"]), int(candidate["y"]))
            if key not in self.action6_tried_coordinates:
                self.action6_probe_index = idx + 1
                self.action6_tried_coordinates.add(key)
                coordinate = dict(candidate)
                self._remember_action6_coordinate(coordinate)
                return coordinate
        for offset in range(len(candidates)):
            idx = (self.action6_probe_index + offset) % len(candidates)
            candidate = candidates[idx]
            key = (int(candidate["x"]), int(candidate["y"]))
            if key not in self.action6_no_delta_coordinates:
                self.action6_probe_index = idx + 1
                self.action6_tried_coordinates.add(key)
                coordinate = dict(candidate)
                self._remember_action6_coordinate(coordinate)
                return coordinate
        transformed_coordinate = self._next_action6_transform_coordinate(candidates, frame_payload)
        if transformed_coordinate is not None:
            return transformed_coordinate
        candidate = candidates[self.action6_probe_index % len(candidates)]
        self.action6_probe_index += 1
        self.action6_tried_coordinates.add((int(candidate["x"]), int(candidate["y"])))
        coordinate = dict(candidate)
        self._remember_action6_coordinate(coordinate)
        return coordinate

    def _immediate_inverse_actions(self) -> set[str]:
        if self.last_probe_action is None:
            return set()
        inverses: set[str] = set()
        for record in self.hypothesis_bank.values():
            if record.get("kind") != "action_semantics" or record.get("label") != "inverse_action_pair":
                continue
            if str(record.get("paired_action")) == self.last_probe_action:
                action = record.get("action")
                if action is not None:
                    inverses.add(str(action))
        return inverses

    def _ordered_probe_names(
        self,
        non_reset: Sequence[str],
        avoid_actions: Sequence[str] = (),
        prefer_action6: bool = False,
    ) -> list[str]:
        ordered: list[str] = []
        for name in self.simple_probe_sequence:
            if name in non_reset and name not in ordered:
                ordered.append(name)
        for name in non_reset:
            if name not in ordered:
                ordered.append(name)
        avoid_penalty = set(avoid_actions)
        inverse_penalty = self._immediate_inverse_actions()
        action6_status = self._action_hypothesis_status("ACTION6") if "ACTION6" in non_reset else "unavailable"
        action6_preferred = prefer_action6 and action6_status not in {"retired", "stale_no_progress"}
        rank = {"untested": 0, "needs_probe": 1, "contradicted": 2, "supported": 3, "already_probed": 3, "stale_no_progress": 4, "retired": 5}
        return sorted(
            ordered,
            key=lambda name: (
                not (action6_preferred and name == "ACTION6"),
                name in avoid_penalty,
                name in inverse_penalty,
                rank.get("already_probed" if name in self.probed_actions and self._action_hypothesis_status(name) == "untested" else self._action_hypothesis_status(name), 1),
                ordered.index(name),
            ),
        )

    def _probe_selection_basis(self, selected: str) -> list[str]:
        basis = ["available_action_filter", "uncertainty_first_probe_order"]
        if any(self._action_hypothesis_status(name) == "retired" for name in self.simple_probe_sequence):
            basis.append("retired_action_penalty")
        if self._immediate_inverse_actions():
            basis.append("immediate_inverse_undo_penalty")
        if self._action_hypothesis_status(selected) == "untested":
            basis.append("untested_action_priority")
        return basis

    def _phase_breaker_recovery_probe_name(self, non_reset: Sequence[str]) -> str | None:
        if "ACTION6" in non_reset:
            return "ACTION6"
        ordered = self._ordered_probe_names(non_reset)
        if not ordered:
            return None
        offset = max(self.phase_breaker_resets_without_progress - 1, 0) % len(ordered)
        return ordered[offset]

    def _candidate_actions(self, non_reset: Sequence[str]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for name in self._ordered_probe_names(non_reset):
            status = self._action_hypothesis_status(name)
            candidates.append(
                {
                    "action": name,
                    "role": "fallback_probe_candidate" if name not in self.simple_probe_sequence else "probe_candidate",
                    "predicted_visible_delta": self._prediction_for_action(name),
                    "hypothesis_status": status,
                    "information_value": "low_retired" if status == "retired" else ("high_untested" if status == "untested" else "medium_retest"),
                    "executable_args_created": False,
                }
            )
        return candidates[:8]

    def _distilled_planning_records(
        self,
        *,
        latest_frame: Any,
        available: Sequence[str],
        non_reset: Sequence[str],
        planned_action: str,
    ) -> list[dict[str, Any]]:
        scene = _scene_summary(_field(latest_frame, "frame"))
        candidates = self._candidate_actions(non_reset)
        sidecar_positive, sidecar_negative = self._sidecar_evidence_counts()
        sidecar_proposals = []
        if self.enable_sidecar_proposals:
            sidecar_proposals = _local_sidecar_proposals(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                scene=scene,
                available=available,
                non_reset=non_reset,
                positive_evidence=sidecar_positive,
                negative_evidence=sidecar_negative,
            )
        predicted_visible_delta = self._prediction_for_action(planned_action)
        candidate_plan = self._candidate_plan_for_action(planned_action)
        return [
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=5,
                line_name="Budget / Risk",
                authority="budget_gate_only",
                payload={
                    "max_hypotheses": self.max_hypotheses,
                    "hypotheses_active": len(self.hypothesis_bank),
                    "available_actions": list(available),
                    "non_reset_actions": list(non_reset),
                    "risk_veto": False,
                    "selected_probe": planned_action,
                    "probe_selection_basis": self._probe_selection_basis(planned_action),
                },
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=8,
                line_name="Authority / Readiness Gate",
                authority="readiness_gate_only",
                payload={
                    "submission_runtime_local_only": True,
                    "model_or_network_required": False,
                    "planned_action_has_executable_args": False,
                    "sidecar_authority": "proposal_only" if self.enable_sidecar_proposals else "disabled_for_ablation",
                    "sidecar_runtime_authority_granted": False,
                    "readiness_decision": "bounded_probe_candidate_only",
                },
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=9,
                line_name="Spatial Abstraction / Scene Graph",
                authority="evidence_only",
                payload=scene,
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=10,
                line_name="Predictive World Model",
                authority="proposal_only",
                payload={
                    "planned_action": planned_action,
                    "predicted_visible_delta": predicted_visible_delta,
                    "prediction_scope": "one_step_visible_delta_only",
                    "public_anchor": scene,
                },
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=11,
                line_name="Model-Predictive Planning Adapter",
                authority="policy_candidate_only",
                payload={
                    "candidate_actions": candidates,
                    "sidecar_proposals": sidecar_proposals,
                    "candidate_plan": candidate_plan,
                    "visual_rule_candidate": self._visual_rule_candidate_status(non_reset),
                    "top_candidate": planned_action,
                    "ranking_basis": ["simple_probe_order", "available_action_filter", "prior_visible_delta_evidence"],
                    "chosen_next_action_authority_created": False,
                },
            ),
        ]

    def _record_distilled_planning_trace(
        self,
        *,
        action: str,
        latest_frame: Any,
        available: Sequence[str],
        non_reset: Sequence[str],
    ) -> None:
        line_records = self._distilled_planning_records(
            latest_frame=latest_frame,
            available=available,
            non_reset=non_reset,
            planned_action=action,
        )
        candidate_plan = None
        for record in line_records:
            if record.get("line_id") == 11:
                candidate_plan = record.get("payload", {}).get("candidate_plan")
        if candidate_plan:
            self.active_plan = candidate_plan
        self.trace_notes.append(
            {
                "policy": "mini-palari distilled line planner",
                "action": action,
                "candidate_plan": candidate_plan,
                "line_records": line_records,
            }
        )

    def _probe_result_line_records(
        self,
        *,
        latest_frame: Any,
        action: str,
        changed_cells: int,
        available: Sequence[str],
        previous_frame: Any | None = None,
        progress_signal_override: str | None = None,
    ) -> list[dict[str, Any]]:
        changed = changed_cells > 0
        observed = "changed" if changed else "unchanged"
        prior_prediction = self._prediction_for_action(action)
        if prior_prediction == "unknown_no_prior_evidence":
            prediction_match = f"unknown_prediction_observed_{observed}"
        elif (prior_prediction == "changed_from_prior_evidence") == changed:
            prediction_match = "matched_prior_visible_delta_prediction"
        else:
            prediction_match = "prediction_error_visible_delta_mismatch"
        scene = _scene_summary(_field(latest_frame, "frame"))
        signature = _delta_signature(_field(previous_frame, "frame"), _field(latest_frame, "frame")) if previous_frame is not None else {"changed_cells": changed_cells, "bbox_shift": None, "color_changes": {}}
        previous_levels = int(_field(previous_frame, "levels_completed", 0) or 0) if previous_frame is not None else int(_field(latest_frame, "levels_completed", 0) or 0)
        current_levels = int(_field(latest_frame, "levels_completed", 0) or 0)
        distance_progress = self._target_distance_progress(previous_frame, latest_frame) if previous_frame is not None and self.enable_objective_progress else None
        if progress_signal_override is not None:
            progress_signal = progress_signal_override
            objective_progress_claimed = False
        elif not self.enable_objective_progress:
            progress_signal = "disabled_for_ablation"
            objective_progress_claimed = False
        elif current_levels > previous_levels:
            progress_signal = "level_completion_increased"
            objective_progress_claimed = True
        elif isinstance(distance_progress, Mapping) and distance_progress.get("progress_signal") == "candidate_target_distance_decreased":
            progress_signal = "candidate_target_distance_decreased"
            objective_progress_claimed = False
        elif isinstance(distance_progress, Mapping) and distance_progress.get("progress_signal") == "candidate_target_distance_increased":
            progress_signal = "candidate_target_distance_increased"
            objective_progress_claimed = False
        elif changed:
            progress_signal = "visible_change_not_objective_progress"
            objective_progress_claimed = False
        else:
            progress_signal = "no_visible_change_no_objective_progress"
            objective_progress_claimed = False
        return [
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=2,
                line_name="Objective Progress",
                authority="evidence_only",
                payload={
                    "observed_action": action,
                    "changed_cells": changed_cells,
                    "progress_signal": progress_signal,
                    "objective_progress_claimed": objective_progress_claimed,
                },
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=3,
                line_name="Action Semantics",
                authority="evidence_only",
                payload={
                    "observed_action": action,
                    "visible_delta_evidence": observed,
                    "delta_signature": signature,
                    "available_actions": list(available),
                },
            ),
            _submission_line_record(
                game_id=self.normalized_game_id,
                turn_index=self.turn_index,
                line_id=10,
                line_name="Predictive World Model",
                authority="proposal_only",
                payload={
                    "observed_action": action,
                    "predicted_visible_delta": prior_prediction,
                    "observed_visible_delta": observed,
                    "prediction_match": prediction_match,
                    "public_anchor": scene,
                },
            ),
        ]

    def _record_probe_result(self, action: str, changed_cells: int, available: Sequence[str], latest_frame: Any, previous_frame: Any | None = None) -> None:
        changed = changed_cells > 0
        active_plan_before_result = dict(self.active_plan) if isinstance(self.active_plan, Mapping) else None
        signature = _delta_signature(_field(previous_frame, "frame"), _field(latest_frame, "frame")) if previous_frame is not None else {"changed_cells": changed_cells, "bbox_shift": None, "color_changes": {}}
        self._mark_plan_mismatch_if_needed(action, changed)
        previous_levels = int(_field(previous_frame, "levels_completed", 0) or 0) if previous_frame is not None else int(_field(latest_frame, "levels_completed", 0) or 0)
        current_levels = int(_field(latest_frame, "levels_completed", 0) or 0)
        distance_progress = self._target_distance_progress(previous_frame, latest_frame) if previous_frame is not None and self.enable_objective_progress else None
        if not self.enable_objective_progress:
            progress_label = "disabled_for_ablation"
        elif current_levels > previous_levels:
            progress_label = "level_completion_increased"
        elif isinstance(distance_progress, Mapping) and distance_progress.get("progress_signal") == "candidate_target_distance_decreased":
            progress_label = "candidate_target_distance_decreased"
        elif isinstance(distance_progress, Mapping) and distance_progress.get("progress_signal") == "candidate_target_distance_increased":
            progress_label = "candidate_target_distance_increased"
        elif changed:
            progress_label = "visible_change_not_objective_progress"
        else:
            progress_label = "no_visible_change_no_objective_progress"
        plan_result_note = self._record_short_horizon_plan_result_if_needed(action, signature, distance_progress, active_plan_before_result)
        plan_progress_evidence = None
        expected_plan_distance = plan_result_note.get("expected_distance_after_first") if isinstance(plan_result_note, Mapping) else None
        if (
            isinstance(plan_result_note, Mapping)
            and plan_result_note.get("plan_status") == "continuation_step_observed"
            and progress_label == "visible_change_not_objective_progress"
            and isinstance(expected_plan_distance, (int, float))
            and abs(float(expected_plan_distance)) <= 1e-6
        ):
            progress_label = TERMINAL_CONTACT_PROGRESS_SIGNAL
            plan_progress_evidence = {
                "plan_id": plan_result_note.get("plan_id"),
                "parent_plan_id": plan_result_note.get("parent_plan_id"),
                "plan_step": plan_result_note.get("plan_step"),
                "action": action,
                "expected_distance_after_first": expected_plan_distance,
                "observed_motion_vector": plan_result_note.get("observed_motion_vector"),
                "target_candidate": dict(_trace_value(plan_result_note.get("target_candidate", {}))),
                "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
            }
            self.trace_notes.append(
                {
                    "policy": "mini-palari continuation result consumption",
                    "action": action,
                    "plan_result_progress": dict(_trace_value(plan_progress_evidence)),
                    "progress_signal": progress_label,
                    "decision": "treat_terminal_continuation_as_candidate_progress",
                    "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
                }
            )
        label = "action_causes_frame_delta" if changed else "action_has_no_visible_delta_yet"
        line_records = self._probe_result_line_records(
            latest_frame=latest_frame,
            action=action,
            changed_cells=changed_cells,
            available=available,
            previous_frame=previous_frame,
            progress_signal_override=progress_label if plan_progress_evidence is not None else None,
        )
        self._upsert_hypothesis(
            kind="action_semantics",
            label=label,
            action=action,
            positive_delta=changed,
            evidence={
                "observed_action": action,
                "changed_cells": changed_cells,
                "available_actions": list(available),
                "delta_signature": signature,
            },
            scope=[f"action:{action}", "visible_frame_delta"],
            extra_fields={"delta_signature": signature},
        )
        if changed:
            self._upsert_hypothesis(
                kind="game_family",
                label="responsive_grid_control_candidate",
                action=None,
                positive_delta=True,
                evidence={
                    "trigger_action": action,
                    "changed_cells": changed_cells,
                    "available_non_reset_actions": [name for name in available if name != "RESET"],
                },
                scope=["game_family:responsive_grid_control", "visible_frame_delta"],
            )
            if signature.get("bbox_shift") is not None:
                shift = signature.get("bbox_shift")
                dx = int(shift.get("dx", 0) or 0) if isinstance(shift, Mapping) else 0
                dy = int(shift.get("dy", 0) or 0) if isinstance(shift, Mapping) else 0
                self._upsert_hypothesis(
                    kind="object_identity",
                    label="controlled_object_candidate_from_moving_region",
                    action=action,
                    positive_delta=True,
                    evidence={
                        "candidate": "moving_nonzero_region",
                        "delta_signature": signature,
                        "trigger_action": action,
                    },
                    scope=["object_identity:moving_nonzero_region", f"action:{action}"],
                    extra_fields={"candidate": "moving_nonzero_region", "delta_signature": signature},
                )
                self._upsert_hypothesis(
                    kind="transition_model",
                    label="one_step_changed_region_motion",
                    action=action,
                    positive_delta=bool(dx or dy),
                    evidence={
                        "trigger_action": action,
                        "changed_cells": changed_cells,
                        "delta_signature": signature,
                        "prediction_scope": "one_step_changed_region_delta",
                    },
                    scope=[f"transition_model:action:{action}", "one_step_changed_region_delta"],
                    extra_fields={
                        "movement_vector": {"dx": dx, "dy": dy},
                        "movement_label": _movement_label(dx, dy),
                        "delta_signature": signature,
                    },
                )
                self._record_inverse_action_pairs(action, dx, dy, signature)
            scene = _scene_summary(_field(latest_frame, "frame"))
            objective_kind = "reach_target_or_transform_region" if scene.get("nonzero_cells", 0) >= 2 else "unknown_objective"
            target_candidate = None
            after_centroid = None
            if isinstance(signature, Mapping):
                after_centroid = signature.get("after_changed_centroid") or signature.get("after_centroid")
            if isinstance(after_centroid, Mapping):
                target_candidate = self._candidate_target_from_points(
                    _nonzero_points(_field(latest_frame, "frame")),
                    object_x=float(after_centroid.get("x", 0.0) or 0.0),
                    object_y=float(after_centroid.get("y", 0.0) or 0.0),
                )
            objective_evidence = {
                "objective_kind": objective_kind,
                "scene_summary": scene,
                "progress_signal": progress_label,
                "trigger_action": action,
                "claim_boundary": "candidate_only_no_success_claim",
            }
            objective_extra: dict[str, Any] = {"objective_kind": objective_kind}
            if isinstance(distance_progress, Mapping):
                objective_evidence["distance_progress"] = dict(_trace_value(distance_progress))
                objective_extra["distance_progress"] = dict(_trace_value(distance_progress))
            objective_scope = [f"objective:{objective_kind}", f"action:{action}"]
            if target_candidate is not None:
                objective_evidence["target_candidate"] = target_candidate
                objective_extra["target_candidate"] = target_candidate
                target_key = self._objective_target_key(target_candidate)
                if target_key is not None:
                    objective_scope.append(f"target:{target_key}")
            self._upsert_hypothesis(
                kind="objective_binding",
                label=objective_kind,
                action=action,
                positive_delta=True,
                evidence=objective_evidence,
                scope=objective_scope,
                extra_fields=objective_extra,
            )
        if action == "ACTION6" and self.last_action6_coordinate is not None:
            if progress_label == "level_completion_increased" or changed:
                self.action6_no_delta_coordinates.discard(self.last_action6_coordinate)
            else:
                self.action6_no_delta_coordinates.add(self.last_action6_coordinate)
                self.trace_notes.append(
                    {
                        "policy": "mini-palari action6 coordinate no-delta evidence",
                        "coordinate": {
                            "x": self.last_action6_coordinate[0],
                            "y": self.last_action6_coordinate[1],
                        },
                        "progress_signal": progress_label,
                        "decision": "retire_coordinate_from_exhausted_fallback_rotation",
                        "claim_boundary": "coordinate_no_delta_is_not_task_failure_claim",
                    }
                )

        if progress_label == "level_completion_increased":
            self.action_no_progress_repeats = {}
            self.objective_no_progress_repeats = {}
            self.exhausted_objective_targets = set()
            self.action6_probe_index = 0
            self.action6_tried_coordinates = set()
            self.action6_no_delta_coordinates = set()
            self.last_action6_coordinate = None
            self.phase_breaker_resets_without_progress = 0
            self.pending_phase_breaker_recovery = False
            self._rebind_stale_action_semantics_after_boundary("level_completion")
            self._rebind_stale_objective_bindings_after_boundary("level_completion")
        elif changed:
            self.action_no_progress_repeats[action] = self.action_no_progress_repeats.get(action, 0) + 1
            if (
                progress_label in CANDIDATE_PROGRESS_SIGNALS
                and self.action_no_progress_repeats.get(action, 0) >= self.max_no_progress_action_repeats
            ):
                self.trace_notes.append(
                    {
                        "policy": "mini-palari weak-progress lock guard",
                        "action": action,
                        "repeat_count": self.action_no_progress_repeats.get(action, 0),
                        "progress_signal": progress_label,
                        "decision": "downgrade_candidate_progress_action_without_level_completion",
                        "claim_boundary": "candidate_distance_progress_is_not_level_success",
                    }
                )
        else:
            self.action_no_progress_repeats[action] = self.action_no_progress_repeats.get(action, 0) + 1
            if self.action_no_progress_repeats.get(action, 0) >= self.max_no_progress_action_repeats:
                self.trace_notes.append(
                    {
                        "policy": "mini-palari no-change lock guard",
                        "action": action,
                        "repeat_count": self.action_no_progress_repeats.get(action, 0),
                        "progress_signal": progress_label,
                        "decision": "downgrade_repeated_unchanged_action_without_level_completion",
                    }
                )
        if progress_label == TERMINAL_CONTACT_PROGRESS_SIGNAL:
            self._record_terminal_contact_target_rebinding_if_needed(action, plan_progress_evidence)
            self._record_terminal_contact_action_family_transfer_if_needed(action, plan_progress_evidence)
        self._record_objective_exhaustion_if_needed(action, progress_label, distance_progress, available)
        if progress_label in OBJECTIVE_PROGRESS_SIGNALS:
            progress_evidence = {
                "previous_levels_completed": previous_levels,
                "current_levels_completed": current_levels,
                "trigger_action": action,
                "progress_signal": progress_label,
            }
            progress_extra: dict[str, Any] = {"progress_signal": progress_label}
            if isinstance(distance_progress, Mapping):
                progress_evidence["distance_progress"] = dict(_trace_value(distance_progress))
                progress_extra["distance_progress"] = dict(_trace_value(distance_progress))
            if isinstance(plan_progress_evidence, Mapping):
                progress_evidence["plan_result_progress"] = dict(_trace_value(plan_progress_evidence))
                progress_extra["plan_result_progress"] = dict(_trace_value(plan_progress_evidence))
            self._upsert_hypothesis(
                kind="progress_signal",
                label=progress_label,
                action=action,
                positive_delta=True,
                evidence=progress_evidence,
                scope=[f"progress_signal:{progress_label}", f"action:{action}"],
                extra_fields=progress_extra,
            )
        self.last_probe_progress_label = progress_label
        self.trace_notes.append(
            {
                "policy": "mini-palari hypothesis layer",
                "action": action,
                "evidence": "frame_delta" if changed else "no_frame_delta",
                "changed_cells": changed_cells,
                "hypotheses_active": len(self.hypothesis_bank),
                "line_records": line_records,
            }
        )

    def choose_action(self, frames: Sequence[Any], latest_frame: Any, actions_by_name: Mapping[str, Any]) -> Any:
        state = _state_name(_field(latest_frame, "state"))
        if state in RESET_STATES:
            action = actions_by_name["RESET"]
            preserve_phase_breaker = self.pending_phase_breaker_recovery
            self._remember_emitted_action(
                "RESET",
                latest_frame,
                preserve_stale_evidence=preserve_phase_breaker,
                count_phase_breaker_reset=False,
            )
            if preserve_phase_breaker:
                reasoning = {
                    "policy": "mini-palari reset-state phase-breaker bridge",
                    "state": state,
                    "reset_count": self.phase_breaker_resets_without_progress,
                    "decision": "preserve_stale_evidence_across_reset_state",
                    "claim_boundary": "reset_state_bridge_is_not_success_claim",
                }
                self.trace_notes.append(reasoning)
                return _attach_reasoning(action, reasoning)
            return _attach_reasoning(action, f"mini-palari reset: state={state}")

        available = available_action_names(latest_frame, actions_by_name)
        non_reset = [name for name in available if name != "RESET"]
        recorded_probe_result = False
        changed = 0
        previous_snapshot = self.last_action_snapshot
        previous_action = self.last_probe_action
        fallback_avoid_actions: set[str] = set()
        fallback_prefer_action6 = False
        if previous_action in non_reset and previous_snapshot is not None:
            changed = frame_delta_count(_field(previous_snapshot, "frame"), _field(latest_frame, "frame"))
            self._record_probe_result(previous_action, changed, available, latest_frame, previous_snapshot)
            recorded_probe_result = True
        elif previous_action in non_reset and frames:
            # Legacy fallback for objects created before this field existed in
            # tests/reloads. Normal operation uses last_action_snapshot above.
            changed = frame_delta_count(_field(frames[-1], "frame"), _field(latest_frame, "frame"))
            self._record_probe_result(previous_action, changed, available, latest_frame, frames[-1])
            recorded_probe_result = True

        continuation_choice = self._short_horizon_continuation_action(latest_frame, non_reset, actions_by_name)
        if continuation_choice is not None:
            continuation_action_name, continuation_reasoning = continuation_choice
            action = actions_by_name[continuation_action_name]
            self.probed_actions.add(continuation_action_name)
            self.turn_index += 1
            self._remember_emitted_action(continuation_action_name, latest_frame)
            self.trace_notes.append(
                {
                    "policy": "mini-palari short-horizon vector planner continuation",
                    "action": continuation_action_name,
                    "evidence": "observed_first_step_confirmed_candidate_plan",
                    "reasoning": continuation_reasoning,
                    "hypotheses_active": len(self.hypothesis_bank),
                }
            )
            return _attach_reasoning(action, continuation_reasoning)

        if (
            self.pending_phase_breaker_recovery
            and non_reset
            and all(self._action_hypothesis_status(name) == "stale_no_progress" for name in non_reset)
        ):
            recovery_name = self._phase_breaker_recovery_probe_name(non_reset)
            if recovery_name is not None and recovery_name in actions_by_name:
                action = actions_by_name[recovery_name]
                coordinate = None
                if recovery_name == "ACTION6" and hasattr(action, "set_data"):
                    coordinate = self._next_action6_coordinate(_field(latest_frame, "frame"))
                    action.set_data(coordinate)
                stale_actions = list(non_reset)
                self.turn_index += 1
                self.probed_actions.add(recovery_name)
                reasoning = {
                    "policy": "mini-palari phase-breaker recovery probe",
                    "action": recovery_name,
                    "stale_actions": stale_actions,
                    "reset_count": self.phase_breaker_resets_without_progress,
                    "coordinate": coordinate,
                    "decision": "probe_after_preserved_stale_evidence_reset",
                    "claim_boundary": "recovery_probe_is_not_success_claim",
                }
                self.trace_notes.append(
                    {
                        "policy": "mini-palari phase-breaker recovery probe",
                        "action": recovery_name,
                        "stale_actions": stale_actions,
                        "reset_count": self.phase_breaker_resets_without_progress,
                        "coordinate": coordinate,
                        "decision": "probe_after_preserved_stale_evidence_reset",
                    }
                )
                self._remember_emitted_action(recovery_name, latest_frame)
                return _attach_reasoning(action, reasoning)

        if (
            non_reset
            and len(non_reset) == 1
            and self._action_hypothesis_status(non_reset[0]) == "stale_no_progress"
            and "RESET" in actions_by_name
            and not _is_complex_action(actions_by_name.get(non_reset[0]))
        ):
            stale_action = non_reset[0]
            action = actions_by_name["RESET"]
            self.turn_index += 1
            self.trace_notes.append(
                {
                    "policy": "mini-palari single-action-stale phase breaker",
                    "stale_action": stale_action,
                    "repeat_count": self.action_no_progress_repeats.get(stale_action, 0),
                    "decision": "reset_after_repeated_single_simple_action_without_level_progress",
                    "claim_boundary": "reset_is_probe_phase_break_not_success_claim",
                }
            )
            self._remember_emitted_action("RESET", latest_frame, preserve_stale_evidence=True)
            return _attach_reasoning(
                action,
                {
                    "policy": "mini-palari single-action-stale phase breaker",
                    "stale_action": stale_action,
                    "decision": "reset_after_repeated_single_simple_action_without_level_progress",
                },
            )

        if "ACTION6" in available and len(available) == 1:
            exhaustion = self._action6_coordinate_exhaustion_summary(_field(latest_frame, "frame"))
            action6_stale = self._action_hypothesis_status("ACTION6") == "stale_no_progress"
            if exhaustion.get("exhausted") and "RESET" in actions_by_name:
                action = actions_by_name["RESET"]
                self.turn_index += 1
                reasoning = {
                    "policy": "mini-palari action6 coordinate-exhaustion phase breaker",
                    **dict(exhaustion),
                    "action6_probe_index": self.action6_probe_index,
                    "tried_coordinate_count": len(self.action6_tried_coordinates),
                    "preserved_search_state": True,
                    "decision": "reset_after_exhausting_action6_coordinate_interpretations",
                    "claim_boundary": "coordinate_exhaustion_reset_is_not_success_claim",
                }
                self.trace_notes.append(reasoning)
                self._remember_emitted_action("RESET", latest_frame, preserve_stale_evidence=True)
                return _attach_reasoning(action, reasoning)
            if (
                action6_stale
                and "RESET" in actions_by_name
                and int(exhaustion.get("missing_no_delta_coordinate_count", 0) or 0)
            ):
                self.trace_notes.append(
                    {
                        "policy": "mini-palari action6 coordinate-exhaustion phase breaker",
                        **dict(exhaustion),
                        "decision": "delay_reset_unresolved_action6_coordinate_evidence",
                        "claim_boundary": "unresolved_coordinate_probe_is_not_success_claim",
                    }
                )
            action = actions_by_name["ACTION6"]
            coordinate = self._next_action6_coordinate(_field(latest_frame, "frame"))
            if hasattr(action, "set_data"):
                action.set_data(coordinate)
            self.turn_index += 1
            self._remember_emitted_action("ACTION6", latest_frame)
            return _attach_reasoning(
                action,
                {
                    "policy": "mini-palari action6 coordinate-probe controller",
                    "coordinate": coordinate,
                    "candidate_count": len(salient_coordinate_candidates(_field(latest_frame, "frame"))),
                    "frame_count": len(frames),
                    "authority": "bounded_probe_action_payload",
                },
            )

        if recorded_probe_result and previous_action is not None:
            if changed:
                escape_note = self._transition_model_loop_escape(previous_snapshot or frames[-1], latest_frame, previous_action)
                if escape_note is not None:
                    self.trace_notes.append(escape_note)
                    navigation_choice = self._navigation_target_action(latest_frame, non_reset)
                    if navigation_choice is not None:
                        nav_action_name, nav_reasoning = navigation_choice
                        action = actions_by_name[nav_action_name]
                        self.probed_actions.add(nav_action_name)
                        self.turn_index += 1
                        self._remember_emitted_action(nav_action_name, latest_frame)
                        corrective_reasoning = dict(nav_reasoning)
                        corrective_reasoning["policy"] = "mini-palari transition model corrective navigation"
                        corrective_reasoning["retired_action"] = previous_action
                        corrective_reasoning["escape_decision"] = escape_note.get("decision")
                        self.trace_notes.append(
                            {
                                "policy": "mini-palari transition model corrective navigation",
                                "action": nav_action_name,
                                "retired_action": previous_action,
                                "evidence": "supported_motion_vector_reduces_candidate_target_after_escape",
                                "reasoning": corrective_reasoning,
                                "hypotheses_active": len(self.hypothesis_bank),
                            }
                        )
                        return _attach_reasoning(action, corrective_reasoning)
                else:
                    visual_choice = self._visual_rule_candidate_action(latest_frame, non_reset)
                    if visual_choice is not None:
                        visual_action_name, visual_reasoning = visual_choice
                        action = actions_by_name[visual_action_name]
                        self.probed_actions.add(visual_action_name)
                        self.turn_index += 1
                        self._remember_emitted_action(visual_action_name, latest_frame)
                        self.trace_notes.append(
                            {
                                "policy": visual_reasoning["policy"],
                                "action": visual_action_name,
                                "evidence": "supported_visual_predicate_candidate",
                                "reasoning": visual_reasoning,
                                "hypotheses_active": len(self.hypothesis_bank),
                            }
                        )
                        return _attach_reasoning(action, visual_reasoning)
                    navigation_choice = self._navigation_target_action(latest_frame, non_reset)
                    if navigation_choice is not None:
                        nav_action_name, nav_reasoning = navigation_choice
                        action = actions_by_name[nav_action_name]
                        self.probed_actions.add(nav_action_name)
                        self.turn_index += 1
                        self._remember_emitted_action(nav_action_name, latest_frame)
                        self.trace_notes.append(
                            {
                                "policy": "mini-palari bounded navigation/object-target controller",
                                "action": nav_action_name,
                                "evidence": "supported_motion_vector_reduces_candidate_target_distance",
                                "reasoning": nav_reasoning,
                                "hypotheses_active": len(self.hypothesis_bank),
                            }
                        )
                        return _attach_reasoning(action, nav_reasoning)
                    avoid_first_actions = set()
                    if self.last_probe_progress_label not in REPEAT_AUTHORIZING_PROGRESS_SIGNALS:
                        avoid_first_actions.add(previous_action)
                    plan_choice = self._short_horizon_navigation_action(
                        latest_frame,
                        non_reset,
                        avoid_first_actions=avoid_first_actions,
                    )
                    if plan_choice is not None:
                        plan_action_name, plan_reasoning = plan_choice
                        action = actions_by_name[plan_action_name]
                        self.probed_actions.add(plan_action_name)
                        self.turn_index += 1
                        self._activate_short_horizon_plan(plan_reasoning)
                        self._remember_emitted_action(plan_action_name, latest_frame)
                        self.trace_notes.append(
                            {
                                "policy": "mini-palari short-horizon vector planner",
                                "action": plan_action_name,
                                "evidence": "two_supported_motion_vectors_reduce_candidate_target_distance",
                                "reasoning": plan_reasoning,
                                "hypotheses_active": len(self.hypothesis_bank),
                            }
                        )
                        return _attach_reasoning(action, plan_reasoning)
                terminal_contact_followup = self.last_probe_progress_label == TERMINAL_CONTACT_PROGRESS_SIGNAL
                objective_repeat_allowed = self.last_probe_progress_label in REPEAT_AUTHORIZING_PROGRESS_SIGNALS
                if terminal_contact_followup:
                    fallback_avoid_actions.add(previous_action)
                    self.trace_notes.append(
                        {
                            "policy": "mini-palari terminal continuation follow-up gate",
                            "action": previous_action,
                            "progress_signal": self.last_probe_progress_label,
                            "decision": "advance_probe_after_terminal_contact_without_level_completion",
                            "claim_boundary": "terminal_candidate_plan_contact_is_not_level_success",
                        }
                    )
                elif self.enable_bounded_planning and not objective_repeat_allowed:
                    fallback_avoid_actions.add(previous_action)
                    action6_status = self._action_hypothesis_status("ACTION6") if "ACTION6" in non_reset else "unavailable"
                    fallback_prefer_action6 = (
                        "ACTION6" in non_reset
                        and action6_status not in {"retired", "stale_no_progress"}
                        and all(name == "ACTION6" or name in self.probed_actions for name in non_reset)
                    )
                    self.trace_notes.append(
                        {
                            "policy": "mini-palari visible-delta objective-progress gate",
                            "action": previous_action,
                            "progress_signal": self.last_probe_progress_label,
                            "decision": "advance_probe_after_visible_delta_without_objective_progress",
                            "prefer_action6_fallback": fallback_prefer_action6,
                            "claim_boundary": "visible_delta_is_not_objective_progress",
                        }
                    )
                elif self.action_no_progress_repeats.get(previous_action, 0) >= self.max_no_progress_action_repeats:
                    self.trace_notes.append(
                        {
                            "policy": "mini-palari no-progress repeat limiter",
                            "action": previous_action,
                            "repeat_count": self.action_no_progress_repeats.get(previous_action, 0),
                            "decision": "advance_to_next_probe_after_visible_delta_without_objective_progress",
                        }
                    )
                else:
                    candidate_plan = self._candidate_plan_for_action(previous_action)
                    if candidate_plan:
                        self.active_plan = candidate_plan
                        self.trace_notes.append(
                            {
                                "policy": "mini-palari bounded planner",
                                "action": previous_action,
                                "candidate_plan": candidate_plan,
                                "plan_status": "candidate_from_supported_action_semantics",
                            }
                        )
                    action = actions_by_name[previous_action]
                    coordinate = None
                    if previous_action == "ACTION6" and hasattr(action, "set_data"):
                        coordinate = self._next_action6_coordinate(_field(latest_frame, "frame"))
                        action.set_data(coordinate)
                    self.turn_index += 1
                    self._remember_emitted_action(previous_action, latest_frame)
                    return _attach_reasoning(
                        action,
                        {
                            "policy": "mini-palari hypothesis layer",
                            "action": previous_action,
                            "hypothesis": "action_causes_frame_delta",
                            "evidence": "repeat after frame delta",
                            "changed_cells": changed,
                            "hypotheses_active": len(self.hypothesis_bank),
                            "coordinate": coordinate,
                        },
                    )

        if non_reset and all(self._action_hypothesis_status(name) == "stale_no_progress" for name in non_reset) and "RESET" in actions_by_name:
            action = actions_by_name["RESET"]
            stale_actions = list(non_reset)
            self.turn_index += 1
            self.trace_notes.append(
                {
                    "policy": "mini-palari all-actions-stale phase breaker",
                    "stale_actions": stale_actions,
                    "decision": "reset_after_exhausting_non_reset_action_authority",
                    "claim_boundary": "reset_is_probe_phase_break_not_success_claim",
                }
            )
            self._remember_emitted_action("RESET", latest_frame, preserve_stale_evidence=True)
            return _attach_reasoning(
                action,
                {
                    "policy": "mini-palari all-actions-stale phase breaker",
                    "stale_actions": stale_actions,
                    "decision": "reset_after_exhausting_non_reset_action_authority",
                },
            )

        visual_choice = self._visual_rule_candidate_action(latest_frame, non_reset)
        if visual_choice is not None:
            visual_action_name, visual_reasoning = visual_choice
            action = actions_by_name[visual_action_name]
            self.probed_actions.add(visual_action_name)
            self.turn_index += 1
            self._remember_emitted_action(visual_action_name, latest_frame)
            self.trace_notes.append(
                {
                    "policy": visual_reasoning["policy"],
                    "action": visual_action_name,
                    "evidence": "supported_visual_predicate_candidate",
                    "reasoning": visual_reasoning,
                    "hypotheses_active": len(self.hypothesis_bank),
                }
            )
            return _attach_reasoning(action, visual_reasoning)

        plan_choice = self._short_horizon_navigation_action(latest_frame, non_reset)
        if plan_choice is not None:
            plan_action_name, plan_reasoning = plan_choice
            action = actions_by_name[plan_action_name]
            self.probed_actions.add(plan_action_name)
            self.turn_index += 1
            self._activate_short_horizon_plan(plan_reasoning)
            self._remember_emitted_action(plan_action_name, latest_frame)
            self.trace_notes.append(
                {
                    "policy": "mini-palari short-horizon vector planner",
                    "action": plan_action_name,
                    "evidence": "two_supported_motion_vectors_reduce_candidate_target_distance",
                    "reasoning": plan_reasoning,
                    "hypotheses_active": len(self.hypothesis_bank),
                }
            )
            return _attach_reasoning(action, plan_reasoning)

        for name in self._ordered_probe_names(
            non_reset,
            avoid_actions=fallback_avoid_actions,
            prefer_action6=fallback_prefer_action6,
        ):
            self.turn_index += 1
            if not recorded_probe_result:
                self._record_distilled_planning_trace(
                    action=name,
                    latest_frame=latest_frame,
                    available=available,
                    non_reset=non_reset,
                )
            self.probed_actions.add(name)
            action = actions_by_name[name]
            coordinate = None
            if name == "ACTION6" and hasattr(action, "set_data"):
                coordinate = self._next_action6_coordinate(_field(latest_frame, "frame"))
                action.set_data(coordinate)
            self._remember_emitted_action(name, latest_frame)
            reasoning: Any = f"mini-palari deterministic high-information probe: {name}; game={self.game_id}"
            if coordinate is not None:
                reasoning = {
                    "policy": "mini-palari deterministic high-information coordinate probe",
                    "action": name,
                    "coordinate": coordinate,
                    "game_id": self.game_id,
                }
            return _attach_reasoning(
                action,
                reasoning,
            )

        if "ACTION6" in non_reset:
            action = actions_by_name["ACTION6"]
            coordinate = self._next_action6_coordinate(_field(latest_frame, "frame"))
            if hasattr(action, "set_data"):
                action.set_data(coordinate)
            self.turn_index += 1
            self._remember_emitted_action("ACTION6", latest_frame)
            return _attach_reasoning(
                action,
                {"policy": "mini-palari action6 coordinate fallback", "coordinate": coordinate},
            )

        action = actions_by_name["RESET"]
        self._remember_emitted_action("RESET", latest_frame)
        return _attach_reasoning(action, "mini-palari fallback reset: no usable non-reset action")


class MyAgent(Agent):  # type: ignore[misc,valid-type]
    """Deterministic no-network Mini-Palari submission agent."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy = MiniPalariArcAgi3Policy(
            game_id=getattr(self, "game_id", "generic"),
            enable_bounded_planning=True,
            enable_visual_rule_candidate=os.environ.get("MINI_PALARI_ENABLE_VISUAL_RULE_CANDIDATE") == "1",
        )

    @property
    def name(self) -> str:
        base_name = getattr(super(), "name", self.__class__.__name__)
        return f"{base_name}.mini-palari-v0.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return self.policy.is_done(latest_frame)

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        actions_by_name = {action.name: action for action in GameAction}  # type: ignore[union-attr]
        return self.policy.choose_action(frames, latest_frame, actions_by_name)
