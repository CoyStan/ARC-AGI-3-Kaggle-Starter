import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_population_loop.py")
spec = importlib.util.spec_from_file_location("visual_rule_population_loop", MODULE_PATH)
assert spec is not None
vrp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrp)


def prompt_pack():
    return {
        "schema": "mini-palari.visual-rule-prompt.v0.1",
        "game_id": "synthetic_game",
        "available_actions": [1, 2, 3],
        "action_history": [],
        "image_paths": ["frame_000.ppm", "frame_001.ppm"],
        "hypothesis_schema": "mini-palari.visual-rule-hypothesis.v0.1",
        "forbidden_authority": ["direct_live_action", "runtime_authority"],
        "runtime_authority_granted": False,
        "prompt_text": "Trace summary: infer visual rules from these frame artifacts.",
    }


def fixture_payload(hypothesis_id: str, relation: str = "moved", action: int = 1):
    return {
        "hypotheses": [
            {
                "schema": "mini-palari.visual-rule-hypothesis.v0.1",
                "hypothesis_id": hypothesis_id,
                "game_family": "fixture_population_unknown",
                "entities": [
                    {"entity_id": "avatar", "component_refs": ["c0"], "role": "avatar", "visual_evidence": "fixture"}
                ],
                "rules": [
                    {
                        "rule_id": f"{hypothesis_id}_r1",
                        "type": "fixture_label_only",
                        "statement": f"{hypothesis_id} claims action {action} should make the avatar relation {relation} observable.",
                        "predictions": [
                            {"predicate_id": f"{hypothesis_id}_p1", "relation": relation, "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                    }
                ],
                "experiments": [
                    {
                        "experiment_id": f"{hypothesis_id}_x1",
                        "purpose": "bounded probe for this independent lineage",
                        "actions": [action],
                        "predictions": [
                            {"predicate_id": f"{hypothesis_id}_x1p1", "relation": relation, "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                        "falsifiers": [
                            {"predicate_id": f"{hypothesis_id}_x1f1", "relation": "stayed", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                    }
                ],
                "plan_if_true": [{"step": 1, "actions": [action], "intent": "proposal-only bounded prefix", "abort_if": ["prediction mismatch"]}],
                "authority": "proposal_only",
            }
        ]
    }


def test_population_loop_runs_ten_independent_one_theory_samples_and_logs_lineages(tmp_path):
    report = vrp.run_visual_rule_population_loop(
        prompt_pack(),
        output_dir=tmp_path,
        sample_count=10,
        rounds=1,
        sidecar="fixture",
        fixture_payloads=[fixture_payload(f"h{i}") for i in range(10)],
    )

    assert report["schema"] == "mini-palari.visual-rule-population-loop.v0.1"
    assert report["model_family"] == "gpt-5.5"
    assert report["round_count"] == 1
    assert report["sample_count_per_round"] == 10
    assert report["summary"]["lineage_count"] == 10
    assert report["summary"]["accepted_sidecar_samples"] == 10
    assert report["runtime_authority_granted"] is False
    assert report["policy_rank_authority_granted"] is False

    lineage_ids = [lineage["lineage_id"] for lineage in report["lineages"]]
    assert len(lineage_ids) == len(set(lineage_ids)) == 10
    assert all(lineage["hypothesis_count"] == 1 for lineage in report["lineages"])
    assert all(lineage["status"] == "pending_verification" for lineage in report["lineages"])

    for index, lineage in enumerate(report["lineages"]):
        prompt_path = Path(lineage["prompt_path"])
        assert prompt_path.exists()
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        assert prompt["population_context"]["sample_index"] == index
        assert prompt["population_context"]["sample_count"] == 10
        assert "Return exactly one rich" in prompt["prompt_text"]


def test_population_loop_prunes_failed_lineages_and_builds_feedback_for_next_round(tmp_path):
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    event_payload = {
        "events": [
            {
                "step": 0,
                "action": "ACTION1",
                "grid": [[0, 0], [0, 2]],
                "next_grid": [[0, 2], [0, 0]],
            }
        ]
    }
    (event_dir / "synthetic_game.json").write_text(json.dumps(event_payload), encoding="utf-8")

    report = vrp.run_visual_rule_population_loop(
        prompt_pack(),
        output_dir=tmp_path / "population",
        sample_count=2,
        rounds=2,
        sidecar="fixture",
        event_dir=event_dir,
        fixture_payloads=[
            fixture_payload("survivor", relation="moved", action=1),
            fixture_payload("failed", relation="level_completed", action=1),
            fixture_payload("replacement", relation="moved", action=1),
            fixture_payload("blocked", relation="level_completed", action=2),
        ],
    )

    statuses = {lineage["hypothesis_ids"][0]: lineage["status"] for lineage in report["lineages"]}
    assert statuses["survivor"] == "survivor"
    assert statuses["failed"] == "falsified"
    assert statuses["replacement"] == "survivor"
    assert statuses["blocked"] == "blocked"

    round2_prompts = [lineage for lineage in report["lineages"] if lineage["round_index"] == 1]
    assert round2_prompts
    for lineage in round2_prompts:
        prompt = json.loads(Path(lineage["prompt_path"]).read_text(encoding="utf-8"))
        feedback = prompt["population_context"]["prior_feedback"]
        assert feedback["survivor_lineage_ids"]
        assert feedback["failed_lineages"]
        assert "Do not repeat falsified claims" in prompt["prompt_text"]

    assert report["summary"]["survivor_count"] == 2
    assert report["summary"]["falsified_count"] == 1
    assert report["summary"]["blocked_count"] == 1
    assert report["claims"]["performance"] == "not_claimed"
