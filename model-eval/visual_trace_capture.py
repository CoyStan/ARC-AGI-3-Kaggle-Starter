"""Capture multi-frame visual trace packs from local ARC-AGI-3 gameplay.

Eval-only helper for Task 6b. It runs deterministic short action probes against
local/public ARC environments when available and writes trace packs plus an
auditable capture manifest. It does not change submitted runtime behavior and
never grants sidecar/runtime authority.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_CAPTURE_MANIFEST = "mini-palari.visual-trace-capture-manifest.v0.1"
DEFAULT_PROBE_ACTIONS = [1, 2, 3, 4, 5, 7]


def _load_trace_module():
    module_path = Path(__file__).with_name("visual_trace_artifacts.py")
    spec = importlib.util.spec_from_file_location("visual_trace_artifacts", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vta = _load_trace_module()
write_trace_pack = _vta.write_trace_pack
normalize_frame = _vta.normalize_frame


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _safe_id(game_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(game_id)) or "game"


def choose_probe_actions(available_actions: list[int] | None, *, max_steps: int = 5) -> list[int]:
    """Choose deterministic simple probe actions from available action IDs.

    RESET (0) and complex coordinate action ACTION6 are excluded because this
    capture lane is evidence gathering, not policy/runtime control. The order of
    available actions is preserved so the trace is transparent and replayable.
    """

    if not available_actions:
        return []
    out: list[int] = []
    for action in available_actions:
        action_id = int(action)
        if action_id in (0, 6):
            continue
        if action_id not in out:
            out.append(action_id)
        if len(out) >= max(0, int(max_steps)):
            break
    return out


def write_capture_manifest(
    *,
    output_root: str | Path,
    requested_game_ids: list[str],
    captures: list[dict[str, Any]],
    blocker: str | None = None,
) -> dict[str, Any]:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA_CAPTURE_MANIFEST,
        "created_at": _now_iso(),
        "decision": "blocked" if blocker else "completed",
        "blocker": blocker,
        "requested_game_ids": list(requested_game_ids),
        "capture_count": len(captures),
        "captures": captures,
        "claims": {
            "generalization": "not_claimed",
            "performance": "not_claimed",
            "scope": "local_public_trace_capture_only",
        },
        "runtime_authority_granted": False,
    }
    path = out / "visual_trace_capture_manifest.json"
    manifest["manifest_path"] = str(path)
    path.write_text(_stable_json(manifest) + "\n", encoding="utf-8")
    return manifest


def write_trace_capture_from_frames(
    *,
    game_id: str,
    frames: list[Any],
    actions: list[int],
    available_actions: list[int] | None,
    output_root: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Write one trace pack and a one-capture manifest from explicit frames."""

    out = Path(output_root)
    trace_dir = out / _safe_id(game_id) / "probe"
    pack = write_trace_pack(
        output_dir=trace_dir,
        game_id=game_id,
        frames=frames,
        actions=actions,
        available_actions=available_actions or [],
        title=title or f"Task 6b probe trace for {game_id}",
        image_scale=8,
    )
    capture = {
        "game_id": game_id,
        "status": "captured",
        "trace_pack_path": pack["trace_pack_path"],
        "frame_count": len(pack.get("frames") or []),
        "delta_count": len(pack.get("deltas") or []),
        "actions": [int(a) for a in actions],
        "available_actions": [int(a) for a in (available_actions or [])],
        "runtime_authority_granted": False,
    }
    manifest = write_capture_manifest(output_root=out, requested_game_ids=[game_id], captures=[capture])
    capture["manifest_path"] = manifest["manifest_path"]
    return capture


def _raw_to_frame(raw: Any) -> Any:
    if raw is None:
        return []
    frame = getattr(raw, "frame", raw)
    if isinstance(frame, dict) and "frame" in frame:
        frame = frame["frame"]
    if isinstance(frame, list) and len(frame) == 1:
        inner = frame[0]
        if hasattr(inner, "tolist"):
            return inner.tolist()
    tolist = getattr(frame, "tolist", None)
    if callable(tolist):
        return tolist()
    return frame


def _raw_available_actions(raw: Any) -> list[int]:
    actions = getattr(raw, "available_actions", None)
    if actions is None and isinstance(raw, dict):
        actions = raw.get("available_actions")
    out: list[int] = []
    for action in actions or []:
        try:
            out.append(int(action))
        except Exception:
            value = getattr(action, "value", None)
            if value is not None:
                out.append(int(value))
    return out


def capture_public_game_traces(
    game_ids: list[str],
    *,
    output_root: str | Path,
    max_steps: int = 5,
    operation_mode: str = "normal",
) -> dict[str, Any]:
    """Capture short deterministic traces from local ARC public games if available."""

    out = Path(output_root)
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor" / "ARC-AGI-3-Agents"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    try:
        import arc_agi  # type: ignore
        from arc_agi import OperationMode  # type: ignore
        from arcengine import GameAction  # type: ignore
    except Exception as exc:
        return write_capture_manifest(
            output_root=out,
            requested_game_ids=game_ids,
            captures=[],
            blocker=f"arc_env_unavailable:{type(exc).__name__}:{exc}",
        )

    mode_name = operation_mode.upper()
    try:
        mode = getattr(OperationMode, mode_name)
    except AttributeError:
        mode = OperationMode.NORMAL

    try:
        arcade = arc_agi.Arcade(operation_mode=mode)
        env_infos = arcade.get_environments()
    except Exception as exc:
        return write_capture_manifest(
            output_root=out,
            requested_game_ids=game_ids,
            captures=[],
            blocker=f"arc_env_list_failed:{type(exc).__name__}:{exc}",
        )

    available_ids = {str(info.game_id).split("-")[0]: info for info in env_infos}
    captures: list[dict[str, Any]] = []
    for requested in game_ids:
        short_id = str(requested).split("-")[0]
        if short_id not in available_ids:
            captures.append({"game_id": short_id, "status": "missing_game", "runtime_authority_granted": False})
            continue
        try:
            env = arcade.make(short_id, render_mode=None)
            raw = env.observation_space
            frames = [_raw_to_frame(raw)]
            initial_available = _raw_available_actions(raw) or list(DEFAULT_PROBE_ACTIONS)
            actions = choose_probe_actions(initial_available, max_steps=max_steps)
            applied: list[int] = []
            latest_available = initial_available
            for action_id in actions:
                action = GameAction.from_id(int(action_id))
                data = action.action_data.model_dump()
                data["game_id"] = short_id
                action.set_data(data)
                raw = env.step(action, data=data, reasoning={"text": "Task 6b deterministic visual trace probe"})
                frames.append(_raw_to_frame(raw))
                applied.append(int(action_id))
                latest_available = _raw_available_actions(raw) or latest_available
            capture = write_trace_capture_from_frames(
                game_id=short_id,
                frames=frames,
                actions=applied,
                available_actions=latest_available,
                output_root=out,
                title=f"Task 6b local public probe trace for {short_id}",
            )
            captures.append(capture)
        except Exception as exc:
            captures.append(
                {
                    "game_id": short_id,
                    "status": "capture_failed",
                    "error": f"{type(exc).__name__}:{exc}",
                    "runtime_authority_granted": False,
                }
            )

    return write_capture_manifest(output_root=out, requested_game_ids=game_ids, captures=captures)
