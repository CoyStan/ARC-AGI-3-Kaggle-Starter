import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_verifier.py")
spec = importlib.util.spec_from_file_location("visual_rule_verifier", MODULE_PATH)
assert spec is not None
vrv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrv)


def comp(component_id, color, bbox, centroid=None, area=1, role=None):
    if centroid is None:
        centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    out = {"id": component_id, "color": color, "bbox": bbox, "centroid": centroid, "area": area}
    if role:
        out["role"] = role
    return out


def trace_pack(frame0_components, frame1_components, changed_cells=2):
    return {
        "schema": "mini-palari.visual-trace-pack.v0.1",
        "game_id": "synthetic",
        "actions": [1],
        "available_actions": [1, 2, 6],
        "frames": [
            {"index": 0, "image_path": "frame_000.ppm", "shape": [8, 8], "components": frame0_components},
            {"index": 1, "image_path": "frame_001.ppm", "shape": [8, 8], "components": frame1_components},
        ],
        "deltas": [{"from": 0, "to": 1, "changed_cells": changed_cells, "bbox": [0, 0, 4, 4]}],
    }


def hypothesis(rules, entities=None, plan=None, hypothesis_id="h1"):
    return {
        "schema": "mini-palari.visual-rule-hypothesis.v0.1",
        "hypothesis_id": hypothesis_id,
        "game_family": "synthetic_family",
        "entities": entities or [],
        "rules": rules,
        "experiments": [{"experiment_id": "x1", "actions": [1], "predicted_observations": ["trace should match"], "falsifiers": ["mismatch"]}],
        "plan_if_true": plan or [{"step": 1, "actions": [1, 2], "intent": "execute only if verified", "abort_if": ["prediction mismatch"]}],
        "authority": "proposal_only",
    }


def test_mirror_motion_supports_opposite_x_component_tracks():
    trace = trace_pack(
        [comp("c_left", 1, [3, 1, 3, 1]), comp("c_right", 1, [5, 1, 5, 1])],
        [comp("c_left", 1, [2, 1, 2, 1]), comp("c_right", 1, [6, 1, 6, 1])],
    )
    h = hypothesis(
        [{"rule_id": "r1", "type": "mirror_motion", "component_refs": ["c_left", "c_right"], "statement": "opposite x motion"}]
    )

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "accepted_candidate"
    assert result["selected_hypothesis"]["hypothesis_id"] == "h1"
    assert result["plan_to_execute"] == h["plan_if_true"]
    rule = result["verifications"][0]["rules"][0]
    assert rule["decision"] == "supported"
    assert "mirror_opposite_x_supported" in rule["support"]
    assert result["runtime_authority_granted"] is False


def test_selected_endpoint_and_midpoint_body_are_supported_by_component_geometry():
    trace = trace_pack(
        [
            comp("leg_a", 2, [0, 0, 0, 0]),
            comp("leg_b", 2, [4, 0, 4, 0]),
            comp("body", 3, [2, 0, 2, 0]),
        ],
        [
            comp("leg_a", 2, [2, 0, 2, 0]),
            comp("leg_b", 2, [4, 0, 4, 0]),
            comp("body", 3, [3, 0, 3, 0]),
        ],
    )
    h = hypothesis(
        [
            {"rule_id": "r1", "type": "selected_endpoint_move", "endpoint_refs": ["leg_a", "leg_b"], "statement": "one selected endpoint moves"},
            {"rule_id": "r2", "type": "midpoint_body", "endpoint_refs": ["leg_a", "leg_b"], "body_ref": "body", "statement": "body is midpoint"},
        ]
    )

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "accepted_candidate"
    rules = {r["rule_id"]: r for r in result["verifications"][0]["rules"]}
    assert rules["r1"]["decision"] == "supported"
    assert "exactly_one_endpoint_moved" in rules["r1"]["support"]
    assert rules["r2"]["decision"] == "supported"
    assert "body_matches_endpoint_midpoint" in rules["r2"]["support"]


def test_hazard_overlap_rejects_otherwise_supported_plan():
    trace = trace_pack(
        [comp("avatar", 1, [0, 0, 0, 0]), comp("lava", 2, [1, 0, 1, 0])],
        [comp("avatar", 1, [1, 0, 1, 0]), comp("lava", 2, [1, 0, 1, 0])],
    )
    h = hypothesis(
        [{"rule_id": "r1", "type": "lava_forbidden", "avatar_refs": ["avatar"], "hazard_refs": ["lava"], "statement": "avatar must not overlap lava"}],
        entities=[
            {"entity_id": "avatar", "component_refs": ["avatar"], "role": "avatar"},
            {"entity_id": "lava", "component_refs": ["lava"], "role": "hazard"},
        ],
    )

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "rejected"
    assert result["selected_hypothesis"] is None
    assert result["plan_to_execute"] == []
    rule = result["verifications"][0]["rules"][0]
    assert rule["decision"] == "rejected"
    assert "hazard_overlap" in rule["counter_evidence"]


def test_no_surviving_hypothesis_means_no_llm_plan_is_used():
    trace = trace_pack(
        [comp("c_left", 1, [3, 1, 3, 1]), comp("c_right", 1, [5, 1, 5, 1])],
        [comp("c_left", 1, [4, 1, 4, 1]), comp("c_right", 1, [6, 1, 6, 1])],
    )
    h = hypothesis(
        [{"rule_id": "r1", "type": "mirror_motion", "component_refs": ["c_left", "c_right"], "statement": "opposite x motion"}],
        plan=[{"step": 1, "actions": [6], "intent": "must not be used after mismatch", "abort_if": ["mismatch"]}],
    )

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "rejected"
    assert result["selected_hypothesis"] is None
    assert result["plan_to_execute"] == []
    assert "mirror_opposite_x_not_observed" in result["verifications"][0]["rules"][0]["counter_evidence"]


def test_generic_predicate_predictions_support_label_only_rule_without_named_checker():
    trace = trace_pack(
        [comp("c_left", 1, [3, 1, 3, 1]), comp("c_right", 1, [5, 1, 5, 1]), comp("lava", 2, [7, 7, 7, 7])],
        [comp("c_left", 1, [2, 1, 2, 1]), comp("c_right", 1, [6, 1, 6, 1]), comp("lava", 2, [7, 7, 7, 7])],
    )
    h = hypothesis(
        [{
            "rule_id": "r1",
            "type": "brand_new_game_mechanic_label_only",
            "statement": "a novel mechanic whose meaning comes from predictions",
            "predictions": [
                {"predicate_id": "p1", "relation": "opposite_delta_x", "subjects": ["left_square", "right_square"], "time": {"from": 0, "to": 1}},
                {"predicate_id": "p2", "relation": "no_overlap", "subjects": ["left_square", "lava"], "time": {"at": "all"}},
            ],
        }],
        entities=[
            {"entity_id": "left_square", "component_refs": ["c_left"], "role": "avatar"},
            {"entity_id": "right_square", "component_refs": ["c_right"], "role": "mirror_avatar"},
            {"entity_id": "lava", "component_refs": ["lava"], "role": "hazard"},
        ],
    )

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "accepted_candidate"
    assert result["selected_hypothesis"]["hypothesis_id"] == "h1"
    assert result["plan_to_execute"] == h["plan_if_true"]
    rule = result["verifications"][0]["rules"][0]
    assert rule["type"] == "brand_new_game_mechanic_label_only"
    assert rule["decision"] == "supported"
    assert {p["predicate_id"]: p["decision"] for p in rule["predicates"]} == {"p1": "supported", "p2": "supported"}
    assert result["runtime_authority_granted"] is False


def test_generic_falsifier_rejects_otherwise_supported_prediction():
    trace = trace_pack(
        [comp("avatar", 1, [0, 0, 0, 0]), comp("lava", 2, [1, 0, 1, 0])],
        [comp("avatar", 1, [1, 0, 1, 0]), comp("lava", 2, [1, 0, 1, 0])],
    )
    h = hypothesis(
        [{
            "rule_id": "r1",
            "type": "hazard_label_only",
            "statement": "avatar must avoid lava",
            "predictions": [
                {"predicate_id": "p1", "relation": "moved", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
            ],
        }],
        entities=[
            {"entity_id": "avatar", "component_refs": ["avatar"], "role": "avatar"},
            {"entity_id": "lava", "component_refs": ["lava"], "role": "hazard"},
        ],
    )
    h["experiments"] = [{
        "experiment_id": "x1",
        "actions": [1],
        "predictions": [{"predicate_id": "x1p1", "relation": "moved", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}],
        "falsifiers": [{"predicate_id": "x1f1", "relation": "overlap", "subjects": ["avatar", "lava"], "time": {"at": 1}}],
    }]

    result = vrv.verify_visual_rule_hypotheses([h], trace)

    assert result["decision"] == "rejected"
    assert result["plan_to_execute"] == []
    assert result["verifications"][0]["falsifiers"][0]["decision"] == "supported"
