from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "play_llm_proposal_only.py"
spec = importlib.util.spec_from_file_location("proposal_only_structured_events_under_test", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _blank(width: int = 12, height: int = 12, fill: int = 0) -> list[list[int]]:
    return [[fill for _ in range(width)] for _ in range(height)]


def _stamp(grid: list[list[int]], shape: list[str], *, x0: int, y0: int, color: int = 3) -> list[list[int]]:
    out = [row[:] for row in grid]
    for y, row in enumerate(shape):
        for x, ch in enumerate(row):
            if ch == "#":
                out[y0 + y][x0 + x] = color
    return out


ASYMMETRIC_L = [
    "#..",
    "#..",
    "###",
]

ASYMMETRIC_L_ROT90 = [
    "###",
    "#..",
    "#..",
]

SYMMETRIC_BLOCK = [
    "##",
    "##",
]


def test_large_asymmetric_shape_rotation_emits_trace_only_pose_change_event():
    before = _stamp(_blank(), ASYMMETRIC_L, x0=4, y0=4, color=3)
    after = _stamp(_blank(), ASYMMETRIC_L_ROT90, x0=4, y0=4, color=3)

    summary = mod.structured_visual_event_summary(
        game_id="fixture", step=7, action="ACTION2", before_grid=before, after_grid=after
    )

    assert summary["schema"] == "mini-palari.arc-live.structured-visual-events.v0.1"
    assert summary["authority"] == "trace_only_non_authoritative"
    assert summary["runtime_authority_granted"] is False
    events = summary["events"]
    assert events
    event_text = str(events).lower()
    assert "rotation_candidate" in event_text
    assert "large_structured_component" in event_text
    assert "action2" in event_text
    best = events[0]
    assert best["support"]["best_transform"] in {"rotate90", "rotate270"}
    assert best["support"]["transformed_similarity"] > best["support"]["unchanged_similarity"]


def test_translation_is_not_reported_as_rotation_candidate():
    before = _stamp(_blank(), ASYMMETRIC_L, x0=3, y0=4, color=3)
    after = _stamp(_blank(), ASYMMETRIC_L, x0=6, y0=4, color=3)

    summary = mod.structured_visual_event_summary(
        game_id="fixture", step=1, action="ACTION4", before_grid=before, after_grid=after
    )

    assert not [event for event in summary["events"] if event["event"] == "rotation_candidate"]


def test_symmetric_shape_rotation_is_ambiguous_not_confident_rotation():
    before = _stamp(_blank(), SYMMETRIC_BLOCK, x0=4, y0=4, color=3)
    after = _stamp(_blank(), SYMMETRIC_BLOCK, x0=4, y0=4, color=3)

    summary = mod.structured_visual_event_summary(
        game_id="fixture", step=2, action="ACTION2", before_grid=before, after_grid=after
    )

    assert not [event for event in summary["events"] if event["event"] == "rotation_candidate"]


def test_occluded_area_change_blocks_confident_rotation():
    before = _stamp(_blank(), ASYMMETRIC_L, x0=4, y0=4, color=3)
    after = _stamp(_blank(), ["###", "#.."], x0=4, y0=4, color=3)

    summary = mod.structured_visual_event_summary(
        game_id="fixture", step=3, action="ACTION1", before_grid=before, after_grid=after
    )

    assert not [event for event in summary["events"] if event["event"] == "rotation_candidate"]
    assert any(event["event"] == "pose_change_blocked_area_mismatch" for event in summary["events"])
