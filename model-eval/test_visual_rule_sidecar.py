import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_sidecar.py")
spec = importlib.util.spec_from_file_location("visual_rule_sidecar", MODULE_PATH)
assert spec is not None
vrs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrs)


def prompt_pack():
    return {
        "schema": "mini-palari.visual-rule-prompt.v0.1",
        "game_id": "synthetic_game",
        "available_actions": [1, 2, 6],
        "action_history": [1],
        "image_paths": ["frame_000.ppm", "frame_001.ppm"],
        "hypothesis_schema": "mini-palari.visual-rule-hypothesis.v0.1",
        "forbidden_authority": ["direct_live_action", "runtime_authority"],
        "runtime_authority_granted": False,
        "prompt_text": "Generic observable predicates: use opposite_delta_x and no_overlap. rule.type is a label.",
    }


def valid_fixture_payload():
    return {
        "hypotheses": [
            {
                "schema": "mini-palari.visual-rule-hypothesis.v0.1",
                "hypothesis_id": "fixture_h1",
                "game_family": "fixture_unknown",
                "entities": [
                    {"entity_id": "avatar", "component_refs": ["c0"], "role": "avatar", "visual_evidence": "fixture"}
                ],
                "rules": [
                    {
                        "rule_id": "r1",
                        "type": "fixture_label_only",
                        "statement": "fixture proposal for adapter tests",
                        "predictions": [
                            {"predicate_id": "p1", "relation": "moved", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                    }
                ],
                "experiments": [
                    {
                        "experiment_id": "x1",
                        "purpose": "fixture smoke",
                        "actions": [1],
                        "predictions": [
                            {"predicate_id": "x1p1", "relation": "moved", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                        "falsifiers": [
                            {"predicate_id": "x1f1", "relation": "stayed", "subjects": ["avatar"], "time": {"from": 0, "to": 1}}
                        ],
                    }
                ],
                "plan_if_true": [{"step": 1, "actions": [1], "intent": "fixture only", "abort_if": ["mismatch"]}],
                "authority": "proposal_only",
            }
        ]
    }


def test_fixture_sidecar_parses_payload_and_logs_artifacts_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("MINI_PALARI_ALLOW_VLM_COMMAND", raising=False)

    result = vrs.run_visual_rule_sidecar(
        prompt_pack(),
        output_dir=tmp_path,
        sidecar="fixture",
        fixture_payload=valid_fixture_payload(),
    )

    assert result["schema"] == "mini-palari.visual-rule-sidecar-run.v0.1"
    assert result["sidecar"] == "fixture"
    assert result["model_source"] == "fixture"
    assert result["validation_status"]["decision"] == "accepted_candidate"
    assert result["hypotheses"][0]["hypothesis_id"] == "fixture_h1"
    assert result["runtime_authority_granted"] is False
    assert result["requires_model_credentials"] is False
    assert result["network_allowed"] is False

    for key in ("prompt_path", "raw_output_path", "sidecar_log_path"):
        path = Path(result[key])
        assert path.exists(), key
        assert path.is_relative_to(tmp_path)

    log = json.loads(Path(result["sidecar_log_path"]).read_text(encoding="utf-8"))
    assert log["forbidden_authority"] == ["direct_live_action", "runtime_authority"]
    assert log["command"] is None
    assert log["runtime_authority_granted"] is False


def test_default_fixture_payload_is_valid_and_proposal_only(tmp_path):
    result = vrs.run_visual_rule_sidecar(prompt_pack(), output_dir=tmp_path, sidecar="fixture")

    assert result["validation_status"]["accepted_count"] == 1
    hypothesis = result["hypotheses"][0]
    assert hypothesis["authority"] == "proposal_only"
    assert hypothesis["rules"][0]["predictions"][0]["relation"] in vrs.SUPPORTED_FIXTURE_RELATIONS
    assert result["runtime_authority_granted"] is False


def test_command_sidecar_is_blocked_without_explicit_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("MINI_PALARI_ALLOW_VLM_COMMAND", raising=False)

    result = vrs.run_visual_rule_sidecar(
        prompt_pack(),
        output_dir=tmp_path,
        sidecar="command",
        command=["python3", "-c", "print('{}')"],
    )

    assert result["decision"] == "blocked_command_not_approved"
    assert result["hypotheses"] == []
    assert result["command"] == ["python3", "-c", "print('{}')"]
    assert result["runtime_authority_granted"] is False
    assert Path(result["sidecar_log_path"]).exists()


def test_command_sidecar_runs_only_with_env_gate_and_logs_command(tmp_path, monkeypatch):
    script = tmp_path / "fixture_command.py"
    script.write_text(
        "import json, sys\n"
        "_prompt = json.load(sys.stdin)\n"
        f"json.dump({json.dumps(valid_fixture_payload())}, sys.stdout)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MINI_PALARI_ALLOW_VLM_COMMAND", "1")

    result = vrs.run_visual_rule_sidecar(
        prompt_pack(),
        output_dir=tmp_path / "artifacts",
        sidecar="command",
        command=["python3", str(script)],
        allow_command=True,
    )

    assert result["sidecar"] == "command"
    assert result["model_source"] == "command"
    assert result["validation_status"]["decision"] == "accepted_candidate"
    assert result["hypotheses"][0]["hypothesis_id"] == "fixture_h1"
    assert result["command"] == ["python3", str(script)]
    assert result["runtime_authority_granted"] is False
    log = json.loads(Path(result["sidecar_log_path"]).read_text(encoding="utf-8"))
    assert log["command"] == ["python3", str(script)]
    assert log["network_allowed"] is False
