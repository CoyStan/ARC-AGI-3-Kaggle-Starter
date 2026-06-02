from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "play_llm_proposal_only.py"
spec = importlib.util.spec_from_file_location("proposal_only_under_test", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_failed_obvious_target_promotes_unique_fixed_cross_marker_candidate():
    grid = [[4 for _ in range(8)] for _ in range(8)]
    # Tiny unusual blue diagonal embedded in black/background neighborhood.
    grid[3][3] = 1
    grid[4][4] = 1
    for x, y in [(3, 2), (2, 3), (4, 3), (3, 4), (4, 5), (5, 4)]:
        grid[y][x] = 0
    # A larger object that action probes moved toward the initial/top guess but did not solve.
    grid[1][5] = 12
    grid[1][6] = 12

    history = [
        {
            "action": "ACTION1",
            "levels_before": 0,
            "levels_after": 0,
            "event_summary": "Moved the obvious top target candidate but the game did not end.",
            "rules_broken": ["obvious top target as terminal goal weakened by no level completion"],
            "primitive_change": {"delta_label_counts": {"primitive_moved": 2, "primitive_persisted": 8}},
        }
    ]

    hints = mod.objective_candidate_hints(grid=grid, history=history)

    assert hints
    hint_text = str(hints).lower()
    assert "color-1" in hint_text
    assert "cross" in hint_text or "marker" in hint_text
    assert "fixed" in hint_text
    assert "promoted_after_failed_obvious_target" in hint_text
    assert "visit" in hint_text or "contact" in hint_text or "align" in hint_text
