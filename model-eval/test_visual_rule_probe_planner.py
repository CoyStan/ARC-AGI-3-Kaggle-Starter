import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_probe_planner.py")
spec = importlib.util.spec_from_file_location("visual_rule_probe_planner", MODULE_PATH)
assert spec is not None
vrp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrp)


def proposal_payload():
    return {
        "schema": "mini-palari.gpt55-visual-rule-proposals.v0.1",
        "authority": {
            "runtime_authority_granted": False,
            "policy_rank_authority_granted": False,
            "action_payload_authority_granted": False,
        },
        "games": [
            {
                "game_id": "su15",
                "contact_sheet": "su15-contact.png",
                "hypotheses": [
                    {
                        "hypothesis_id": "su15-h1",
                        "authority": "proposal_only",
                        "statement": "The source marker, dotted diagonal path, and target ring likely form a click-or-aim mechanic where the player must select waypoints or the ring center, then verify that the source advances along the hinted path instead of treating any pixel delta as progress.",
                        "experiments": [
                            {
                                "experiment_id": "su15-x1",
                                "purpose": "Test target click",
                                "probe_template": "ACTION6 at target_ring center",
                                "predictions": [
                                    {"relation": "target_contact", "subjects": ["source_marker", "target_ring"]},
                                    {"relation": "level_completed", "subjects": ["target_ring"]},
                                ],
                            },
                            {
                                "experiment_id": "su15-x2",
                                "purpose": "Test waypoint following",
                                "probe_template": "ACTION6 at next dotted waypoint after source, then final waypoint before ring",
                                "predictions": [
                                    {"relation": "source_to_target_distance_decreased", "subjects": ["source_marker", "target_ring"]}
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "game_id": "ka59",
                "contact_sheet": "ka59-contact.png",
                "hypotheses": [
                    {
                        "hypothesis_id": "ka59-h1",
                        "authority": "proposal_only",
                        "experiments": [
                            {
                                "experiment_id": "ka59-x1",
                                "purpose": "Build topology and identify controlled object",
                                "probe_template": "cycle simple movement actions while tracking dark/yellow component deltas over grey traversable area",
                                "predictions": [
                                    {"relation": "controlled_component_moved", "subjects": ["movable_dark"]}
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_build_probe_plan_preserves_proposal_only_authority_and_budgets_actions():
    plan = vrp.build_probe_plan_from_gpt55_proposals(proposal_payload(), max_actions_per_experiment=3)

    assert plan["schema"] == "mini-palari.visual-rule-probe-plan.v0.1"
    assert plan["runtime_authority_granted"] is False
    assert plan["policy_rank_authority_granted"] is False
    assert plan["game_count"] == 2
    assert plan["probe_count"] == 3

    by_id = {probe["experiment_id"]: probe for game in plan["games"] for probe in game["probes"]}
    assert 200 <= by_id["su15-x1"]["hypothesis_statement_chars"] <= 300
    assert "dotted diagonal path" in by_id["su15-x1"]["hypothesis_statement"]
    assert by_id["su15-x1"]["probe_kind"] == "coordinate_target"
    assert by_id["su15-x1"]["target_intents"] == ["target_ring.center"]
    assert by_id["su15-x1"]["action_sequence"] == [6]
    assert by_id["su15-x1"]["requires_coordinate_binding"] is True
    assert by_id["su15-x1"]["runtime_authority_granted"] is False

    assert by_id["ka59-x1"]["probe_kind"] == "simple_action_cycle"
    assert by_id["ka59-x1"]["action_sequence"] == [1, 2, 3]
    assert by_id["ka59-x1"]["requires_coordinate_binding"] is False
    assert by_id["ka59-x1"]["predicates_to_verify"][0]["relation"] == "controlled_component_moved"


def test_write_probe_plan_artifact_round_trips_and_marks_verification_pending(tmp_path):
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposal_payload()), encoding="utf-8")

    report = vrp.write_probe_plan_from_file(
        proposal_path,
        output_dir=tmp_path / "out",
        max_actions_per_experiment=2,
    )

    assert report["decision"] == "planned_pending_execution"
    assert report["runtime_authority_granted"] is False
    assert Path(report["report_path"]).exists()
    loaded = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert loaded["probe_count"] == 3
    assert loaded["verification_summary"]["decision"] == "blocked_needs_probe_execution"
    assert loaded["verification_summary"]["runtime_authority_granted"] is False
    for game in loaded["games"]:
        for probe in game["probes"]:
            assert probe["verification_status"] == "pending_execution"
            assert len(probe["action_sequence"]) <= 2


def test_rejects_input_that_tries_to_grant_runtime_authority():
    payload = proposal_payload()
    payload["authority"]["runtime_authority_granted"] = True

    plan = vrp.build_probe_plan_from_gpt55_proposals(payload)

    assert plan["decision"] == "blocked_authority_escalation"
    assert plan["games"] == []
    assert plan["runtime_authority_granted"] is False
    assert plan["verification_summary"]["runtime_authority_granted"] is False
