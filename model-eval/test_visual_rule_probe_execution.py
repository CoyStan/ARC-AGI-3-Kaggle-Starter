from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_probe_execution.py")
spec = importlib.util.spec_from_file_location("visual_rule_probe_execution", MODULE_PATH)
vrx = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(vrx)


def test_bind_target_intent_to_connected_component_center():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 9, 9, 0, 0],
        [0, 9, 9, 0, 0],
        [0, 0, 0, 4, 4],
        [0, 0, 0, 4, 4],
    ]

    bindings = vrx.bind_target_intents(grid, ["target_ring.center", "bottom_swatches.centers"])

    assert bindings["target_ring.center"][0]["coordinate"] == {"x": 1.5, "y": 1.5}
    assert bindings["target_ring.center"][0]["component_id"].startswith("c9_")
    assert bindings["bottom_swatches.centers"][0]["coordinate"] == {"x": 3.5, "y": 3.5}


def test_replay_probe_sequence_marks_supported_visible_delta_and_no_completion():
    events = [
        {"step": 0, "action": "ACTION1", "levels": 0, "state": "GameState.NOT_FINISHED", "grid": [[0, 1]], "next_grid": [[0, 2]], "changed": 1},
        {"step": 1, "action": "ACTION2", "levels": 0, "state": "GameState.NOT_FINISHED", "grid": [[0, 2]], "next_grid": [[0, 2]], "changed": 0},
    ]
    probe = {
        "experiment_id": "p1",
        "action_sequence": [1, 2],
        "predicates_to_verify": [
            {"relation": "component_count_changed", "subjects": []},
            {"relation": "level_completed", "subjects": []},
        ],
    }

    result = vrx.replay_probe_on_events(probe, events)

    assert result["execution_status"] == "replayed"
    assert result["visible_delta_steps"] == 1
    decisions = {r["relation"]: r["decision"] for r in result["predicate_results"]}
    assert decisions["component_count_changed"] == "supported"
    assert decisions["level_completed"] == "rejected"
    assert result["survivor_status"] == "accepted_partial"


def test_compare_variants_prefers_more_bound_and_verified_long_evidence():
    reports = {
        "short": {"variant": "short", "summary": {"bound_probe_count": 1, "accepted_survivors": 1, "predicate_supported": 1, "predicate_rejected": 0}},
        "long": {"variant": "long", "summary": {"bound_probe_count": 2, "accepted_survivors": 1, "predicate_supported": 3, "predicate_rejected": 1}},
    }

    comparison = vrx.compare_variant_reports(reports)

    assert comparison["best_by_evidence_depth"] == "long"
    assert comparison["variants"]["long"]["evidence_depth_score"] > comparison["variants"]["short"]["evidence_depth_score"]
    assert comparison["claims"]["performance"] == "not_claimed"
