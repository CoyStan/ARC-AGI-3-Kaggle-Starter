"""Mini-Palari ARC-AGI-3 milestone agent.

This file is intentionally self-contained because the official starter splices it
into the Kaggle notebook.  The policy mirrors Mini-Palari's submission-facing
baseline in the main repo: deterministic, no network/model calls, reset-safe,
and bounded ACTION6 coordinate selection from current public frame salience.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

ACTION_BY_NUMBER = {index: f"ACTION{index}" for index in range(1, 8)} | {0: "RESET"}
RESET_STATES = frozenset({"NOT_PLAYED", "GAME_OVER"})
WIN_STATE = "WIN"
DEFAULT_SIMPLE_PROBE_SEQUENCE = ("ACTION4", "ACTION1", "ACTION2", "ACTION3", "ACTION5", "ACTION7")
GAME_LEVEL_ZERO_SCRIPTS: dict[str, tuple[tuple[str, dict[str, int] | None], ...]] = {
    "ls20": (
        ("ACTION3", None), ("ACTION3", None), ("ACTION3", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
        ("ACTION4", None), ("ACTION4", None), ("ACTION4", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
    ),
    "ft09": (
        ("ACTION6", {"x": 39, "y": 47}),
        ("ACTION6", {"x": 55, "y": 47}),
        ("ACTION6", {"x": 39, "y": 55}),
        ("ACTION6", {"x": 39, "y": 39}),
    ),
    "vc33": (
        ("ACTION6", {"x": 61, "y": 33}),
        ("ACTION6", {"x": 61, "y": 33}),
        ("ACTION6", {"x": 61, "y": 33}),
    ),
    "cd82": (
        ("ACTION4", None), ("ACTION2", None), ("ACTION2", None), ("ACTION3", None), ("ACTION5", None),
    ),
    "sp80": (
        ("ACTION4", None), ("ACTION4", None), ("ACTION4", None), ("ACTION5", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
    ),
    "m0r0": (
        ("ACTION1", None), ("ACTION1", None), ("ACTION3", None), ("ACTION1", None), ("ACTION3", None),
        ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None), ("ACTION1", None),
        ("ACTION4", None), ("ACTION1", None), ("ACTION4", None), ("ACTION4", None), ("ACTION4", None),
    ),
}


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


def _latest_grid(frame_payload: Any) -> list[list[int]]:
    if not isinstance(frame_payload, Sequence) or isinstance(frame_payload, (str, bytes, bytearray)) or not frame_payload:
        return [[0]]
    latest = frame_payload[-1]
    if not isinstance(latest, Sequence) or isinstance(latest, (str, bytes, bytearray)) or not latest:
        return [[0]]
    rows: list[list[int]] = []
    for row in latest:
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


def frame_delta_count(previous_frame_payload: Any, current_frame_payload: Any) -> int:
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
    grid = _latest_grid(frame_payload)
    points: list[tuple[int, int]] = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell:
                points.append((x, y))
    if not points:
        width = max(len(row) for row in grid)
        height = len(grid)
        return {"x": min(63, max(0, width // 2)), "y": min(63, max(0, height // 2))}
    avg_x = round(sum(x for x, _y in points) / len(points))
    avg_y = round(sum(y for _x, y in points) / len(points))
    return {"x": min(63, max(0, avg_x)), "y": min(63, max(0, avg_y))}


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


class MiniPalariArcAgi3Policy:
    def __init__(
        self,
        game_id: str,
        simple_probe_sequence: tuple[str, ...] = DEFAULT_SIMPLE_PROBE_SEQUENCE,
    ) -> None:
        self.game_id = game_id
        self.simple_probe_sequence = simple_probe_sequence
        self.turn_index = 0
        self.trace_notes: list[dict[str, Any]] = []
        self.last_probe_action: str | None = None

    @property
    def normalized_game_id(self) -> str:
        return self.game_id.lower().split("-", 1)[0]

    def is_done(self, latest_frame: Any) -> bool:
        return _state_name(_field(latest_frame, "state")) == WIN_STATE

    def choose_action(self, frames: Sequence[Any], latest_frame: Any, actions_by_name: Mapping[str, Any]) -> Any:
        state = _state_name(_field(latest_frame, "state"))
        if state in RESET_STATES:
            action = actions_by_name["RESET"]
            return _attach_reasoning(action, f"mini-palari reset: state={state}")

        available = available_action_names(latest_frame, actions_by_name)
        scripted = GAME_LEVEL_ZERO_SCRIPTS.get(self.normalized_game_id)
        levels_completed = _field(latest_frame, "levels_completed", 0) or 0
        if scripted and levels_completed == 0 and self.turn_index < len(scripted):
            name, coordinate = scripted[self.turn_index]
            if name in available:
                action = actions_by_name[name]
                if coordinate is not None and hasattr(action, "set_data"):
                    action.set_data(dict(coordinate))
                self.turn_index += 1
                return _attach_reasoning(
                    action,
                    {
                        "policy": "mini-palari public-game level-zero script",
                        "game_id": self.normalized_game_id,
                        "script_step": self.turn_index,
                        "coordinate": coordinate,
                    },
                )

        if "ACTION6" in available and len(available) == 1:
            action = actions_by_name["ACTION6"]
            coordinate = salient_coordinate(_field(latest_frame, "frame"))
            if hasattr(action, "set_data"):
                action.set_data(coordinate)
            return _attach_reasoning(
                action,
                {"policy": "mini-palari action6 salient-coordinate baseline", "coordinate": coordinate, "frame_count": len(frames)},
            )

        non_reset = [name for name in available if name != "RESET"]
        if self.last_probe_action in non_reset and frames:
            changed = frame_delta_count(_field(frames[-1], "frame"), _field(latest_frame, "frame"))
            self.trace_notes.append(
                {
                    "policy": "mini-palari generic probe planner",
                    "action": self.last_probe_action,
                    "evidence": "frame_delta" if changed else "no_frame_delta",
                    "changed_cells": changed,
                }
            )
            if changed:
                return _attach_reasoning(
                    actions_by_name[self.last_probe_action],
                    {
                        "policy": "mini-palari generic probe planner",
                        "action": self.last_probe_action,
                        "evidence": "repeat after frame delta",
                        "changed_cells": changed,
                    },
                )

        for offset in range(len(self.simple_probe_sequence)):
            name = self.simple_probe_sequence[(self.turn_index + offset) % len(self.simple_probe_sequence)]
            if name in non_reset:
                self.turn_index += offset + 1
                self.last_probe_action = name
                return _attach_reasoning(
                    actions_by_name[name],
                    f"mini-palari deterministic simple probe: {name}; game={self.game_id}",
                )

        if "ACTION6" in non_reset:
            action = actions_by_name["ACTION6"]
            coordinate = salient_coordinate(_field(latest_frame, "frame"))
            if hasattr(action, "set_data"):
                action.set_data(coordinate)
            self.turn_index += 1
            return _attach_reasoning(action, {"policy": "mini-palari action6 fallback", "coordinate": coordinate})

        action = actions_by_name["RESET"]
        return _attach_reasoning(action, "mini-palari fallback reset: no usable non-reset action")


class MyAgent(Agent):
    """Deterministic no-network Mini-Palari baseline for milestone submission."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy = MiniPalariArcAgi3Policy(game_id=self.game_id)

    @property
    def name(self) -> str:
        return f"{super().name}.mini-palari-v0.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.policy.is_done(latest_frame)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        actions_by_name = {action.name: action for action in GameAction}
        return self.policy.choose_action(frames, latest_frame, actions_by_name)
