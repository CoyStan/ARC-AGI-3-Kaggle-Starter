"""Thin ARC-AGI-3 adapter that routes turns through the full Mini-Palari Brain.

This file intentionally demotes the old ARC-specific controller.  ARC is now only
an IO shell: convert the current ARC frame into a MiniPalariAgent observation,
call the real Brain, then convert the Brain's action back to ARC's GameAction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # ARC starter / Kaggle runtime imports.
    from arcengine import FrameData, GameAction  # type: ignore
    from agents.agent import Agent  # type: ignore
except Exception:  # Mini-Palari unit/local smoke imports.
    FrameData = Any  # type: ignore
    GameAction = Any  # type: ignore

    class Agent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.game_id = kwargs.get("game_id", "generic")

        @property
        def name(self) -> str:
            return self.__class__.__name__


def _ensure_full_brain_import_path() -> None:
    """Prefer the real Mini-Palari package over a self-contained ARC shim.

    The local starter lives beside the Mini-Palari repo.  If this file is copied
    into the starter, add the sibling source tree so ARC still calls the full
    Brain instead of reimplementing it here.
    """

    candidates = [
        Path(__file__).resolve().parents[3] / "src",  # inside mini-palari repo
        Path("/home/quetza/projects/mini-palari/src"),  # local ARC starter sibling
    ]
    for candidate in candidates:
        if (candidate / "mini_palari" / "agent.py").exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
            return


_ensure_full_brain_import_path()

from mini_palari.agent import MiniPalariAgent  # noqa: E402
try:  # noqa: E402 - primitive deltas are Brain evidence, not pixel-count scoring.
    from mini_palari.primitive_binding import detect_canonical_primitive_inventory_from_observations  # type: ignore
except Exception:  # pragma: no cover - fallback keeps adapter importable in constrained exports.
    detect_canonical_primitive_inventory_from_observations = None  # type: ignore[assignment]

ACTION_BY_NUMBER = {index: f"ACTION{index}" for index in range(1, 8)} | {0: "RESET"}
RESET_STATES = frozenset({"NOT_PLAYED", "GAME_OVER"})
WIN_STATE = "WIN"
DEFAULT_SIMPLE_PROBE_SEQUENCE = ("ACTION4", "ACTION1", "ACTION2", "ACTION3", "ACTION5", "ACTION7")


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


def _looks_like_grid_row(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(cell, int) and not isinstance(cell, bool) for cell in value)
    )


def _clean_grid(grid_payload: Any) -> list[list[int]]:
    if not isinstance(grid_payload, Sequence) or isinstance(grid_payload, (str, bytes, bytearray)) or not grid_payload:
        return [[0]]
    rows: list[list[int]] = []
    for row in grid_payload:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            continue
        clean_row: list[int] = []
        for cell in row:
            clean_row.append(int(cell) if isinstance(cell, int) and not isinstance(cell, bool) else 0)
        if clean_row:
            rows.append(clean_row)
    return rows or [[0]]


def _latest_grid(frame_payload: Any) -> list[list[int]]:
    """Return the current 2-D grid from ARC's 2-D or sequence-of-grids shape."""

    if not isinstance(frame_payload, Sequence) or isinstance(frame_payload, (str, bytes, bytearray)) or not frame_payload:
        return [[0]]
    if all(_looks_like_grid_row(row) for row in frame_payload):
        return _clean_grid(frame_payload)
    return _clean_grid(frame_payload[-1])


def _nonzero_points(frame_payload: Any) -> list[tuple[int, int, int]]:
    grid = _latest_grid(frame_payload)
    points: list[tuple[int, int, int]] = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell:
                points.append((x, y, int(cell)))
    return points


def primitive_delta_summary(
    previous_frame_payload: Any,
    current_frame_payload: Any,
    *,
    game_id: str = "generic",
) -> dict[str, Any]:
    """Summarize before/after change in primitive/component/relation terms.

    This intentionally avoids raw changed-pixel counts.  If the core primitive
    detector is unavailable, return an explicit unavailable record rather than
    silently falling back to pixel arithmetic.
    """

    if detect_canonical_primitive_inventory_from_observations is None:
        return {
            "basis": "primitive_delta_unavailable_no_pixel_fallback",
            "primitive_delta_record_count": 0,
            "meaningful_primitive_delta_count": 0,
            "delta_label_counts": {},
            "primitive_delta_records": [],
        }
    try:
        inventory = detect_canonical_primitive_inventory_from_observations(
            [
                {"game_id": f"{game_id}:before", "frame": _latest_grid(previous_frame_payload)},
                {"game_id": f"{game_id}:after", "frame": _latest_grid(current_frame_payload)},
            ],
            delta_observation_pairs=[
                {
                    "before_observation_id": f"{game_id}:before",
                    "after_observation_id": f"{game_id}:after",
                }
            ],
        )
    except Exception as exc:  # pragma: no cover - defensive trace path.
        return {
            "basis": "primitive_delta_error_no_pixel_fallback",
            "error": exc.__class__.__name__,
            "primitive_delta_record_count": 0,
            "meaningful_primitive_delta_count": 0,
            "delta_label_counts": {},
            "primitive_delta_records": [],
        }
    records = [record for record in inventory.get("primitive_delta_records", []) if isinstance(record, Mapping)]
    meaningful = [record for record in records if str(record.get("delta_label")) != "primitive_persisted"]
    return {
        "basis": "primitive_delta_records_not_raw_changed_pixel_count",
        "primitive_delta_record_count": len(records),
        "meaningful_primitive_delta_count": len(meaningful),
        "delta_label_counts": dict(inventory.get("delta_label_counts") or {}),
        "primitive_delta_records": [dict(record) for record in records[:8]],
    }


def salient_coordinate_candidates(frame_payload: Any, limit: int = 16) -> list[dict[str, int]]:
    """Small Brain-adapter fallback for ACTION6 payloads.

    The Brain chooses whether ACTION6 is useful; this helper only supplies the
    required coordinate payload if ARC demands one.
    """

    points = _nonzero_points(frame_payload)
    candidates: list[dict[str, int]] = []

    def add(x: float | int, y: float | int) -> None:
        coord = {"x": min(63, max(0, int(round(x)))), "y": min(63, max(0, int(round(y))))}
        if coord not in candidates:
            candidates.append(coord)

    if points:
        xs = [x for x, _y, _c in points]
        ys = [y for _x, y, _c in points]
        add(sum(xs) / len(xs), sum(ys) / len(ys))
        by_color: dict[int, list[tuple[int, int, int]]] = {}
        for point in points:
            by_color.setdefault(point[2], []).append(point)
        for color in sorted(by_color, key=lambda c: (-len(by_color[c]), c)):
            color_points = by_color[color]
            add(sum(p[0] for p in color_points) / len(color_points), sum(p[1] for p in color_points) / len(color_points))
        for x, y, _c in points[: max(0, limit - len(candidates))]:
            add(x, y)
    else:
        add(0, 0)
    return candidates[: max(1, int(limit))]


def salient_coordinate(frame_payload: Any) -> dict[str, int]:
    return salient_coordinate_candidates(frame_payload, limit=1)[0]


def _attach_reasoning(action: Any, reasoning: Any) -> Any:
    try:
        setattr(action, "reasoning", reasoning)
    except Exception:
        pass
    return action


def _attach_data(action: Any, data: dict[str, int]) -> Any:
    setter = getattr(action, "set_data", None)
    if callable(setter):
        setter(dict(data))
    else:
        try:
            setattr(action, "data", dict(data))
        except Exception:
            pass
    return action


def _is_complex_action(action: Any) -> bool:
    checker = _field(action, "is_complex")
    if callable(checker):
        try:
            return bool(checker())
        except TypeError:
            pass
    return bool(_field(action, "complex_action", False) or _field(action, "requires_args", False))


class MiniPalariArcAgi3Policy:
    """Compatibility name; now this is the full-Brain ARC adapter."""

    def __init__(
        self,
        game_id: str = "generic",
        seed: int = 0,
        *,
        enable_bounded_planning: bool = False,
        **_legacy_ignored: Any,
    ) -> None:
        self.game_id = str(game_id or "generic")
        self.brain = MiniPalariAgent(seed=seed)
        self.enable_bounded_planning = bool(enable_bounded_planning)
        self.trace_notes: list[dict[str, Any]] = []
        self.last_frame: list[list[int]] | None = None
        self.last_action: str | None = None
        self.last_action_snapshot: dict[str, Any] | None = None
        self.last_probe_action: str | None = None
        self.last_probe_progress_label: str | None = None
        self.probed_actions: set[str] = set()
        self.action_no_progress_repeats: dict[str, int] = {}
        self.max_no_progress_action_repeats = 2
        self.active_plan: dict[str, Any] | None = None
        self.action6_probe_index = 0
        self.turn_index = 0

    def is_done(self, latest_frame: Any) -> bool:
        return _state_name(_field(latest_frame, "state")) == WIN_STATE

    def _available_action_names(self, latest_frame: Any, actions_by_name: Mapping[str, Any]) -> list[str]:
        raw_available = _field(latest_frame, "available_actions", None)
        if raw_available is None:
            names = [name for name in actions_by_name if name != "RESET"]
        else:
            names = []
            for value in raw_available:
                try:
                    name = action_name_from_value(value)
                except ValueError:
                    continue
                if name in actions_by_name and name != "RESET":
                    names.append(name)
        # Keep the Brain's default probe order stable while still respecting ARC.
        ordered = [name for name in DEFAULT_SIMPLE_PROBE_SEQUENCE if name in names]
        ordered.extend(name for name in names if name not in ordered)
        return ordered

    def _brain_candidates(self, latest_frame: Any, names: Sequence[str]) -> list[dict[str, Any]]:
        grid = _latest_grid(_field(latest_frame, "frame", [[0]]))
        points = _nonzero_points(grid)
        dominant_color = None
        if points:
            counts: dict[int, int] = {}
            for _x, _y, color in points:
                counts[color] = counts.get(color, 0) + 1
            dominant_color = sorted(counts, key=lambda color: (-counts[color], color))[0]
        candidates: list[dict[str, Any]] = []
        for name in names:
            candidate: dict[str, Any] = {"action": name, "predicted_effect": {"kind": "arc_probe", "action": name}}
            if dominant_color is not None:
                candidate["target_color"] = dominant_color
                candidate["predicted_effect"].update(
                    {
                        "target_color": dominant_color,
                        "focus_type": "object",
                        "relation_type": "object_identity",
                    }
                )
            if name == "ACTION6":
                candidate["args"] = salient_coordinate(grid)
                candidate["predicted_effect"]["kind"] = "coordinate_probe"
            candidates.append(candidate)
        return candidates

    def _observation_from_arc(self, frames: Sequence[Any], latest_frame: Any, available_names: Sequence[str]) -> dict[str, Any]:
        current_grid = _latest_grid(_field(latest_frame, "frame", [[0]]))
        previous_grid = self.last_frame
        if previous_grid is None and frames:
            previous_grid = _latest_grid(_field(frames[-1], "frame", [[0]]))
        return {
            "game_id": self.game_id,
            "frame": current_grid,
            "previous_frame": previous_grid,
            "available_actions": self._brain_candidates(latest_frame, available_names),
            "last_action": self.last_action,
            "level_state": {
                "level": int(_field(latest_frame, "levels_completed", 0) or 0),
                "level_completed": False,
                "win_signal": self.is_done(latest_frame),
            },
        }

    def _current_snapshot(self, latest_frame: Any) -> dict[str, Any]:
        return {
            "frame": _latest_grid(_field(latest_frame, "frame", [[0]])),
            "levels_completed": int(_field(latest_frame, "levels_completed", 0) or 0),
        }

    def _record_emitted_action(self, action_name: str, latest_frame: Any) -> None:
        if action_name == "RESET":
            self.last_action_snapshot = None
            self.last_frame = None
        else:
            snapshot = self._current_snapshot(latest_frame)
            self.last_action_snapshot = snapshot
            self.last_frame = list(snapshot["frame"])
            self.probed_actions.add(action_name)
            self.last_probe_action = action_name
        self.last_action = action_name
        self.turn_index += 1

    def _next_probe_name(self, available_names: Sequence[str], *, avoid: str | None = None) -> str | None:
        for name in available_names:
            if name != avoid and name not in self.probed_actions:
                return name
        for name in available_names:
            if name != avoid:
                return name
        return None

    def _choose_direct_probe(
        self,
        action_name: str,
        latest_frame: Any,
        actions_by_name: Mapping[str, Any],
        reasoning: Mapping[str, Any] | str,
    ) -> Any:
        action = actions_by_name[action_name]
        if action_name == "ACTION6" or _is_complex_action(action):
            _attach_data(action, salient_coordinate(_field(latest_frame, "frame", [[0]])))
        self._record_emitted_action(action_name, latest_frame)
        if isinstance(reasoning, Mapping):
            feedback = reasoning.get("primitive_feedback")
            primitive_delta = feedback.get("primitive_delta", {}) if isinstance(feedback, Mapping) else {}
            compact_reasoning: Mapping[str, Any] | str = {
                "decision": reasoning.get("decision"),
                "plan_status": reasoning.get("plan_status"),
                "next_action": action_name,
                "progress_label": feedback.get("progress_label") if isinstance(feedback, Mapping) else None,
                "primitive_delta_basis": primitive_delta.get("basis") if isinstance(primitive_delta, Mapping) else None,
                "meaningful_primitive_delta_count": primitive_delta.get("meaningful_primitive_delta_count") if isinstance(primitive_delta, Mapping) else None,
                "delta_label_counts": primitive_delta.get("delta_label_counts") if isinstance(primitive_delta, Mapping) else None,
            }
        else:
            compact_reasoning = reasoning
        return _attach_reasoning(action, compact_reasoning)

    def _primitive_feedback_from_last_action(self, latest_frame: Any) -> dict[str, Any] | None:
        snapshot = self.last_action_snapshot
        if not snapshot or not self.last_probe_action:
            return None
        summary = primitive_delta_summary(
            snapshot.get("frame"),
            _field(latest_frame, "frame", [[0]]),
            game_id=self.game_id,
        )
        levels_before = int(snapshot.get("levels_completed", 0) or 0)
        levels_after = int(_field(latest_frame, "levels_completed", 0) or 0)
        objective_progress = levels_after > levels_before
        primitive_changed = int(summary.get("meaningful_primitive_delta_count", 0) or 0) > 0
        if objective_progress:
            label = "objective_progress"
            self.action_no_progress_repeats[self.last_probe_action] = 0
        elif primitive_changed:
            label = "visible_change_not_objective_progress"
            self.action_no_progress_repeats[self.last_probe_action] = self.action_no_progress_repeats.get(self.last_probe_action, 0) + 1
        else:
            label = "no_primitive_change"
            self.action_no_progress_repeats[self.last_probe_action] = self.action_no_progress_repeats.get(self.last_probe_action, 0) + 1
        self.last_probe_progress_label = label
        return {
            "schema": "mini-palari.arc-feedback.primitive-result.v0.1",
            "basis": summary.get("basis"),
            "last_probe_action": self.last_probe_action,
            "progress_label": label,
            "objective_progress": objective_progress,
            "primitive_changed": primitive_changed,
            "primitive_delta": summary,
        }

    def _bounded_planning_override(
        self,
        available_names: Sequence[str],
        latest_frame: Any,
        actions_by_name: Mapping[str, Any],
    ) -> Any | None:
        if not self.enable_bounded_planning:
            return None

        simple_names = [name for name in available_names if name in actions_by_name and name != "ACTION6" and not _is_complex_action(actions_by_name[name])]
        if (
            len(simple_names) == 1
            and "RESET" in actions_by_name
            and self.action_no_progress_repeats.get(simple_names[0], 0) >= self.max_no_progress_action_repeats
        ):
            self.trace_notes.append(
                {
                    "decision": "reset_after_repeated_single_simple_action_without_level_progress",
                    "action": simple_names[0],
                    "basis": "bounded_phase_breaker_not_success_claim",
                }
            )
            self._record_emitted_action("RESET", latest_frame)
            return _attach_reasoning(actions_by_name["RESET"], self.trace_notes[-1])

        feedback = self._primitive_feedback_from_last_action(latest_frame)
        if feedback is None:
            return None

        if self.active_plan and feedback["progress_label"] == "no_primitive_change":
            plan = self.active_plan
            self.active_plan = None
            next_name = self._next_probe_name(available_names, avoid=feedback["last_probe_action"])
            self.trace_notes.append(
                {
                    "plan_status": "aborted_prediction_mismatch",
                    "decision": "advance_probe_after_plan_prediction_mismatch",
                    "aborted_plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
                    "primitive_feedback": feedback,
                    "next_action": next_name,
                }
            )
            if next_name:
                return self._choose_direct_probe(next_name, latest_frame, actions_by_name, self.trace_notes[-1])

        if feedback["progress_label"] == "visible_change_not_objective_progress":
            next_name = self._next_probe_name(available_names, avoid=feedback["last_probe_action"])
            if next_name:
                self.trace_notes.append(
                    {
                        "decision": "advance_probe_after_visible_delta_without_objective_progress",
                        "primitive_feedback": feedback,
                        "next_action": next_name,
                    }
                )
                return self._choose_direct_probe(next_name, latest_frame, actions_by_name, self.trace_notes[-1])
        return None

    def choose_action(self, frames: Sequence[Any], latest_frame: Any, actions_by_name: Mapping[str, Any]) -> Any:
        state = _state_name(_field(latest_frame, "state"))
        if state in RESET_STATES and "RESET" in actions_by_name:
            self._record_emitted_action("RESET", latest_frame)
            chosen = actions_by_name["RESET"]
            return _attach_reasoning(chosen, "mini-palari full-brain adapter reset: state=" + state)

        available_names = self._available_action_names(latest_frame, actions_by_name)
        if not available_names:
            chosen = actions_by_name.get("RESET") or next(iter(actions_by_name.values()))
            return _attach_reasoning(chosen, "mini-palari full-brain adapter fallback: no usable action")

        bounded_override = self._bounded_planning_override(available_names, latest_frame, actions_by_name)
        if bounded_override is not None:
            return bounded_override

        observation = self._observation_from_arc(frames, latest_frame, available_names)
        decision = self.brain.decide(observation)
        action_name = str(decision.get("action") or available_names[0])
        if action_name not in actions_by_name or action_name == "RESET":
            action_name = available_names[0]
        action = actions_by_name[action_name]
        args = dict(decision.get("args") or {})
        if action_name == "ACTION6" or _is_complex_action(action):
            args = args or salient_coordinate(_field(latest_frame, "frame", [[0]]))
            _attach_data(action, args)

        trace = decision.get("trace", {})
        self.trace_notes.append(
            {
                "policy": "mini-palari full-brain arc adapter",
                "action": action_name,
                "args": args,
                "brain_turn": self.brain.turn_id - 1,
                "trace": trace,
            }
        )
        self._record_emitted_action(action_name, latest_frame)
        return _attach_reasoning(
            action,
            {
                "policy": "mini-palari full-brain arc adapter",
                "decision": "action_chosen_by_MiniPalariAgent.decide",
                "brain_turn": self.brain.turn_id - 1,
                "trace_schema": trace.get("schema") if isinstance(trace, Mapping) else None,
            },
        )


class MyAgent(Agent):  # type: ignore[misc,valid-type]
    """ARC shell around the full Mini-Palari Brain."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(os.environ.get("MINI_PALARI_SEED", "0"))
        self.policy = MiniPalariArcAgi3Policy(
            game_id=getattr(self, "game_id", "generic"),
            seed=seed,
            enable_bounded_planning=True,
        )

    @property
    def name(self) -> str:
        base_name = getattr(super(), "name", self.__class__.__name__)
        return f"{base_name}.mini-palari-full-brain-v0.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return self.policy.is_done(latest_frame)

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        actions_by_name = {action.name: action for action in GameAction}  # type: ignore[union-attr]
        return self.policy.choose_action(frames, latest_frame, actions_by_name)
