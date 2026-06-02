import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_hypotheses.py")
spec = importlib.util.spec_from_file_location("visual_rule_hypotheses", MODULE_PATH)
assert spec is not None
vrh = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrh)


def valid_hypothesis(**overrides):
    h = {
        "schema": "mini-palari.visual-rule-hypothesis.v0.1",
        "hypothesis_id": "h1",
        "game_family": "mirrored_dual_avatar_navigation",
        "entities": [
            {"entity_id": "e1", "component_refs": ["c0"], "role": "avatar", "visual_evidence": "c0 is square"}
        ],
        "rules": [
            {"rule_id": "r1", "type": "mirror_motion", "statement": "c0 mirrors c1", "predicted_deltas": ["c0 dx=-1 c1 dx=+1"]}
        ],
        "experiments": [
            {"experiment_id": "x1", "purpose": "test mirror", "actions": [1, 2], "predicted_observations": ["opposite x motion"], "falsifiers": ["same x motion"]}
        ],
        "plan_if_true": [
            {"step": 1, "actions": [1, 2], "intent": "pin mirror", "abort_if": ["lava touched"]}
        ],
        "authority": "proposal_only",
    }
    h.update(overrides)
    return h


def test_parse_valid_hypothesis_object_returns_status_and_no_runtime_authority():
    hypotheses, status = vrh.parse_visual_rule_hypotheses(
        {"hypotheses": [valid_hypothesis()]},
        available_actions=[1, 2, 6],
    )

    assert status["decision"] == "accepted_candidate"
    assert status["accepted_count"] == 1
    assert status["rejected_count"] == 0
    assert status["runtime_authority_granted"] is False
    assert hypotheses[0]["hypothesis_id"] == "h1"
    assert hypotheses[0]["authority"] == "proposal_only"


def test_rejects_authority_escalation():
    hypotheses, status = vrh.parse_visual_rule_hypotheses(
        {"hypotheses": [valid_hypothesis(authority="live_action")]},
        available_actions=[1, 2, 6],
    )

    assert hypotheses == []
    assert status["decision"] == "rejected"
    assert status["rejected_count"] == 1
    assert "authority_must_be_proposal_only" in status["rejections"][0]["reasons"]
    assert status["runtime_authority_granted"] is False


def test_salvages_json_from_fenced_text():
    text = """Here are candidates:\n```json\n{\"hypotheses\": [%s]}\n```\nNo authority granted.""" % vrh.json_dumps(valid_hypothesis(hypothesis_id="h2"))

    hypotheses, status = vrh.parse_visual_rule_hypotheses(text, available_actions=[1, 2, 6])

    assert status["decision"] == "accepted_candidate"
    assert hypotheses[0]["hypothesis_id"] == "h2"
    assert status["salvaged_from_text"] is True


def test_rejects_experiment_actions_outside_available_actions():
    bad = valid_hypothesis(experiments=[
        {"experiment_id": "x1", "purpose": "bad action", "actions": [1, 9], "predicted_observations": ["?"], "falsifiers": ["?"]}
    ])

    hypotheses, status = vrh.parse_visual_rule_hypotheses({"hypotheses": [bad]}, available_actions=[1, 2, 6])

    assert hypotheses == []
    assert status["decision"] == "rejected"
    assert "action_not_available:9" in status["rejections"][0]["reasons"]


def test_rejects_missing_required_sections():
    missing = valid_hypothesis()
    del missing["rules"]

    hypotheses, status = vrh.parse_visual_rule_hypotheses({"hypotheses": [missing]}, available_actions=[1, 2, 6])

    assert hypotheses == []
    assert "missing_required_field:rules" in status["rejections"][0]["reasons"]


def test_accepts_predicate_shaped_predictions():
    h = valid_hypothesis(
        rules=[{
            "rule_id": "r1",
            "type": "model_label_only_not_executable",
            "statement": "mirror label only",
            "predictions": [
                {"predicate_id": "p1", "relation": "opposite_delta_x", "subjects": ["left", "right"], "time": {"from": 0, "to": 1}}
            ],
        }],
        experiments=[{
            "experiment_id": "x1",
            "purpose": "verify hazard invariant",
            "actions": [1],
            "predictions": [
                {"predicate_id": "p2", "relation": "no_overlap", "subjects": ["avatar", "lava"], "time": {"at": "all"}}
            ],
            "falsifiers": [
                {"predicate_id": "f1", "relation": "overlap", "subjects": ["avatar", "lava"], "time": {"at": 1}}
            ],
        }],
    )

    hypotheses, status = vrh.parse_visual_rule_hypotheses({"hypotheses": [h]}, available_actions=[1, 2, 6])

    assert status["decision"] == "accepted_candidate"
    assert hypotheses[0]["rules"][0]["predictions"][0]["relation"] == "opposite_delta_x"


def test_rejects_malformed_predicate_predictions():
    h = valid_hypothesis(rules=[{
        "rule_id": "r1",
        "type": "label_only",
        "statement": "bad predicate",
        "predictions": [{"predicate_id": "p1", "subjects": ["a"]}],
    }])

    hypotheses, status = vrh.parse_visual_rule_hypotheses({"hypotheses": [h]}, available_actions=[1, 2, 6])

    assert hypotheses == []
    assert "malformed_predicate:rules[0].predictions[0]" in status["rejections"][0]["reasons"]
