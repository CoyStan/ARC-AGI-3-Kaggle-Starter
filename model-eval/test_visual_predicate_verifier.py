import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_predicate_verifier.py")
spec = importlib.util.spec_from_file_location("visual_predicate_verifier", MODULE_PATH)
assert spec and spec.loader
visual_predicate_verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual_predicate_verifier)

verify_predicate = visual_predicate_verifier.verify_predicate
verify_predicates = visual_predicate_verifier.verify_predicates


def comp(cid, bbox, color=1):
    x0, y0, x1, y1 = bbox
    return {
        "id": cid,
        "bbox": [x0, y0, x1, y1],
        "centroid": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        "area": (x1 - x0 + 1) * (y1 - y0 + 1),
        "color": color,
    }


def trace(frames):
    return {"frames": [{"frame_index": i, "components": cs} for i, cs in enumerate(frames)]}


def trace_with_meta(frames):
    packed = []
    for i, frame in enumerate(frames):
        item = {"frame_index": i, "components": frame.get("components", [])}
        item.update({k: v for k, v in frame.items() if k != "components"})
        packed.append(item)
    return {"frames": packed}


def test_frame_pair_motion_predicates_support_and_reject():
    pack = trace([
        [comp("a", [5, 5, 5, 5]), comp("b", [10, 5, 10, 5])],
        [comp("a", [4, 5, 4, 5]), comp("b", [11, 5, 11, 5])],
    ])

    assert verify_predicate({"predicate_id": "p1", "relation": "moved", "subjects": ["a"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p2", "relation": "stayed", "subjects": ["a"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "rejected"
    assert verify_predicate({"predicate_id": "p3", "relation": "delta_x_sign", "subjects": ["a"], "value": "negative", "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p4", "relation": "delta_y_sign", "subjects": ["a"], "value": "zero", "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p5", "relation": "opposite_delta_x", "subjects": ["a", "b"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p6", "relation": "same_delta_x", "subjects": ["a", "b"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "rejected"


def test_overlap_inside_and_all_time_invariant_predicates():
    pack = trace([
        [comp("avatar", [0, 0, 1, 1]), comp("lava", [5, 5, 6, 6]), comp("room", [0, 0, 10, 10])],
        [comp("avatar", [2, 2, 3, 3]), comp("lava", [5, 5, 6, 6]), comp("room", [0, 0, 10, 10])],
        [comp("avatar", [5, 5, 5, 5]), comp("lava", [5, 5, 6, 6]), comp("room", [0, 0, 10, 10])],
    ])

    assert verify_predicate({"predicate_id": "p1", "relation": "inside_bbox", "subjects": ["avatar", "room"], "time": {"at": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p2", "relation": "outside_bbox", "subjects": ["lava", "room"], "time": {"at": 1}}, pack)["decision"] == "rejected"
    assert verify_predicate({"predicate_id": "p3", "relation": "no_overlap", "subjects": ["avatar", "lava"], "time": {"at": "all"}}, pack)["decision"] == "rejected"
    assert verify_predicate({"predicate_id": "p4", "relation": "overlap", "subjects": ["avatar", "lava"], "time": {"at": 2}}, pack)["decision"] == "supported"


def test_distance_and_midpoint_geometry_predicates():
    pack = trace([
        [comp("a", [0, 0, 0, 0]), comp("b", [10, 0, 10, 0]), comp("body", [4, 0, 4, 0])],
        [comp("a", [2, 0, 2, 0]), comp("b", [8, 0, 8, 0]), comp("body", [5, 0, 5, 0])],
    ])

    assert verify_predicate({"predicate_id": "p1", "relation": "distance_decreased", "subjects": ["a", "b"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p2", "relation": "distance_increased", "subjects": ["a", "b"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "rejected"
    assert verify_predicate({"predicate_id": "p3", "relation": "midpoint_matches", "subjects": ["a", "b", "body"], "time": {"at": 1}}, pack)["decision"] == "supported"


def test_structural_predicates_count_color_appearance():
    pack = trace([
        [comp("a", [0, 0, 0, 0], color=1), comp("b", [2, 0, 2, 0], color=2)],
        [comp("a", [0, 0, 0, 0], color=1), comp("c", [4, 0, 4, 0], color=1), comp("d", [6, 0, 6, 0], color=3)],
    ])

    assert verify_predicate({"predicate_id": "p1", "relation": "component_appeared", "subjects": ["c"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p2", "relation": "component_disappeared", "subjects": ["b"], "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p3", "relation": "component_count_changed", "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"
    assert verify_predicate({"predicate_id": "p4", "relation": "color_count_changed", "value": 1, "time": {"from": 0, "to": 1}}, pack)["decision"] == "supported"


def test_entity_bindings_and_batch_verification_are_supported():
    pack = trace([
        [comp("c14", [5, 5, 5, 5]), comp("c15", [10, 5, 10, 5])],
        [comp("c14", [4, 5, 4, 5]), comp("c15", [11, 5, 11, 5])],
    ])
    entities = [
        {"entity_id": "left_square", "component_refs": ["c14"]},
        {"entity_id": "right_square", "component_refs": ["c15"]},
    ]

    results = verify_predicates([
        {"predicate_id": "p1", "relation": "opposite_delta_x", "subjects": ["left_square", "right_square"], "time": {"from": 0, "to": 1}},
        {"predicate_id": "p2", "relation": "same_delta_x", "subjects": ["left_square", "right_square"], "time": {"from": 0, "to": 1}},
    ], pack, entity_bindings=entities)

    assert [r["decision"] for r in results] == ["supported", "rejected"]
    assert results[0]["runtime_authority_granted"] is False


def test_semantic_progress_predicates_distinguish_objective_progress_from_motion():
    pack = trace([
        [comp("avatar", [0, 0, 0, 0]), comp("goal", [5, 0, 6, 1]), comp("hazard", [0, 5, 1, 6])],
        [comp("avatar", [5, 0, 5, 0]), comp("goal", [5, 0, 6, 1]), comp("hazard", [0, 5, 1, 6])],
        [comp("avatar", [0, 5, 0, 5]), comp("goal", [5, 0, 6, 1]), comp("hazard", [0, 5, 1, 6])],
    ])

    target_contact = verify_predicate({"predicate_id": "goal_touch", "relation": "target_contact", "subjects": ["avatar", "goal"], "time": {"at": 1}}, pack)
    hazard_contact = verify_predicate({"predicate_id": "hazard_touch", "relation": "hazard_contact", "subjects": ["avatar", "hazard"], "time": {"at": 2}}, pack)
    entered_goal = verify_predicate({"predicate_id": "entered", "relation": "object_entered_goal_region", "subjects": ["avatar", "goal"], "time": {"from": 0, "to": 1}}, pack)
    left_goal = verify_predicate({"predicate_id": "left", "relation": "object_left_goal_region", "subjects": ["avatar", "goal"], "time": {"from": 1, "to": 2}}, pack)
    progress = verify_predicate({"predicate_id": "closer", "relation": "progress_toward_static_target", "subjects": ["avatar", "goal"], "time": {"from": 0, "to": 1}}, pack)

    assert target_contact["decision"] == "supported"
    assert hazard_contact["decision"] == "supported"
    assert entered_goal["decision"] == "supported"
    assert left_goal["decision"] == "supported"
    assert progress["decision"] == "supported"
    assert progress["observed"]["target_delta"] == [0.0, 0.0]
    assert progress["runtime_authority_granted"] is False


def test_semantic_progress_metadata_predicates_support_and_reject():
    pack = trace_with_meta([
        {"components": [comp("avatar", [0, 0, 0, 0])], "score_proxy": 1, "level_completed": False},
        {"components": [comp("avatar", [1, 0, 1, 0])], "score_proxy": 3, "level_completed": True},
        {"components": [comp("avatar", [0, 0, 0, 0])], "reset": True, "failure": True, "score_proxy": 0},
    ])

    completed = verify_predicate({"predicate_id": "done", "relation": "level_completed", "time": {"at": 1}}, pack)
    reset = verify_predicate({"predicate_id": "fail", "relation": "reset_or_failure", "time": {"at": 2}}, pack)
    score_up = verify_predicate({"predicate_id": "score", "relation": "score_proxy_improved", "time": {"from": 0, "to": 1}}, pack)
    score_not_up = verify_predicate({"predicate_id": "score_bad", "relation": "score_proxy_improved", "time": {"from": 1, "to": 2}}, pack)

    assert completed["decision"] == "supported"
    assert reset["decision"] == "supported"
    assert score_up["decision"] == "supported"
    assert score_not_up["decision"] == "rejected"
    assert score_up["observed"] == {"score_before": 1.0, "score_after": 3.0, "delta": 2.0}


def test_missing_semantic_metadata_blocks_instead_of_guessing_progress():
    pack = trace([[comp("avatar", [0, 0, 0, 0])], [comp("avatar", [1, 0, 1, 0])]])

    result = verify_predicate({"predicate_id": "score", "relation": "score_proxy_improved", "time": {"from": 0, "to": 1}}, pack)

    assert result["decision"] == "blocked_needs_evidence"
    assert "missing_score_proxy" in result["counter_evidence"]
    assert result["runtime_authority_granted"] is False


def test_missing_frames_or_unknown_relation_blocks_without_authority():
    result = verify_predicate({"predicate_id": "bad", "relation": "telepathic_rule", "subjects": ["a"]}, trace([]))
    assert result["decision"] == "blocked_needs_evidence"
    assert result["runtime_authority_granted"] is False
    assert "unsupported_relation:telepathic_rule" in result["counter_evidence"]
