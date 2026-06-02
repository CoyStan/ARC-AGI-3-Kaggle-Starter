import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_rule_prompt.py")
spec = importlib.util.spec_from_file_location("visual_rule_prompt", MODULE_PATH)
assert spec is not None
vrp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vrp)


def sample_trace_pack(tmp_path):
    image0 = tmp_path / "frame_000.ppm"
    image1 = tmp_path / "frame_001.ppm"
    image0.write_text("P3\n1 1\n255\n0 0 0\n", encoding="ascii")
    image1.write_text("P3\n1 1\n255\n255 0 0\n", encoding="ascii")
    return {
        "schema": "mini-palari.visual-trace-pack.v0.1",
        "title": "synthetic trace",
        "game_id": "synthetic_game",
        "actions": [1, 6],
        "available_actions": [1, 2, 3, 6],
        "authority": "evidence_only_no_runtime_authority",
        "frames": [
            {
                "index": 0,
                "image_path": str(image0),
                "shape": [4, 4],
                "components": [
                    {"id": "c0", "color": 1, "area": 2, "bbox": [0, 0, 1, 0], "centroid": [0.5, 0.0]},
                    {"id": "c1", "color": 2, "area": 1, "bbox": [3, 3, 3, 3], "centroid": [3.0, 3.0]},
                ],
            },
            {
                "index": 1,
                "image_path": str(image1),
                "shape": [4, 4],
                "components": [
                    {"id": "c0", "color": 1, "area": 2, "bbox": [1, 0, 2, 0], "centroid": [1.5, 0.0]},
                ],
            },
        ],
        "deltas": [
            {"from": 0, "to": 1, "changed_cells": 2, "bbox": [0, 0, 2, 0], "changed_points": [[0, 0], [2, 0]]}
        ],
    }


def test_build_visual_rule_prompt_includes_artifact_refs_schema_and_context(tmp_path):
    prompt = vrp.build_visual_rule_prompt(sample_trace_pack(tmp_path))

    assert prompt["schema"] == "mini-palari.visual-rule-prompt.v0.1"
    assert prompt["game_id"] == "synthetic_game"
    assert prompt["available_actions"] == [1, 2, 3, 6]
    assert prompt["action_history"] == [1, 6]
    assert str(tmp_path / "frame_000.ppm") in prompt["prompt_text"]
    assert str(tmp_path / "frame_001.ppm") in prompt["prompt_text"]
    assert "c0" in prompt["prompt_text"]
    assert "c1" in prompt["prompt_text"]
    assert "mini-palari.visual-rule-hypothesis.v0.1" in prompt["prompt_text"]
    assert "entities" in prompt["prompt_text"]
    assert "rules" in prompt["prompt_text"]
    assert "experiments" in prompt["prompt_text"]
    assert "plan_if_true" in prompt["prompt_text"]


def test_prompt_forbids_direct_action_authority_and_requests_proposal_only(tmp_path):
    prompt = vrp.build_visual_rule_prompt(sample_trace_pack(tmp_path))
    text = prompt["prompt_text"]

    assert "proposal_only" in text
    assert "Do not output direct live actions" in text
    assert "no runtime authority" in text
    assert prompt["forbidden_authority"] == [
        "direct_live_action",
        "policy_rank_without_verification",
        "memory_write",
        "network_or_download",
        "runtime_authority",
    ]
    assert prompt["runtime_authority_granted"] is False


def test_prompt_includes_mirror_and_spider_examples(tmp_path):
    text = vrp.build_visual_rule_prompt(sample_trace_pack(tmp_path))["prompt_text"]

    assert "mirrored_dual_avatar_navigation" in text
    assert "mirror_motion" in text
    assert "pin" in text.lower()
    assert "spider_endpoint_geometry" in text
    assert "selected_endpoint_move" in text
    assert "midpoint_body" in text


def test_prompt_requests_generic_predicate_objects_not_closed_world_rule_checkers(tmp_path):
    text = vrp.build_visual_rule_prompt(sample_trace_pack(tmp_path))["prompt_text"]

    assert "Generic observable predicates" in text
    assert "opposite_delta_x" in text
    assert "no_overlap" in text
    assert "predicate_id" in text
    assert "rule.type is a label" in text


def test_prompt_requests_semantic_progress_predicates_without_runtime_authority(tmp_path):
    text = vrp.build_visual_rule_prompt(sample_trace_pack(tmp_path))["prompt_text"]

    assert "progress-like predicates" in text
    assert "level_completed" in text
    assert "reset_or_failure" in text
    assert "target_contact" in text
    assert "hazard_contact" in text
    assert "progress_toward_static_target" in text
    assert "score_proxy_improved" in text


def test_prompt_is_deterministic_for_same_trace_pack(tmp_path):
    trace = sample_trace_pack(tmp_path)
    first = vrp.build_visual_rule_prompt(trace)
    second = vrp.build_visual_rule_prompt(trace)

    assert first == second
