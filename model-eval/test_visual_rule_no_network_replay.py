import importlib.util
import json
import socket
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("visual_rule_eval.py")
spec = importlib.util.spec_from_file_location("visual_rule_eval", MODULE_PATH)
assert spec is not None
vre = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vre)

TRACE_MODULE_PATH = Path(__file__).with_name("visual_trace_artifacts.py")
trace_spec = importlib.util.spec_from_file_location("visual_trace_artifacts", TRACE_MODULE_PATH)
assert trace_spec is not None
vta = importlib.util.module_from_spec(trace_spec)
assert trace_spec.loader is not None
trace_spec.loader.exec_module(vta)


def make_trace(tmp_path, game_id, frames):
    return vta.write_trace_pack(
        output_dir=tmp_path / "cached-traces" / game_id / "probe",
        game_id=game_id,
        frames=frames,
        actions=[1],
        available_actions=[1, 2, 3],
        title=f"cached {game_id}",
        image_scale=1,
    )


def test_no_network_replay_consumes_cached_trace_packs_and_writes_manifest(tmp_path):
    moving = make_trace(tmp_path, "moving_game", [[[1, 0, 0]], [[0, 1, 0]]])
    stuck = make_trace(tmp_path, "stuck_game", [[[1, 0, 0]], [[1, 0, 0]]])

    def forbid_network(*_args, **_kwargs):
        raise AssertionError("network access attempted during no-network replay")

    with patch.object(socket, "create_connection", side_effect=forbid_network), patch.object(socket, "socket", side_effect=forbid_network):
        report = vre.run_visual_rule_no_network_replay(
            [moving["trace_pack_path"], stuck["trace_pack_path"]],
            output_dir=tmp_path / "replay",
        )

    assert report["schema"] == "mini-palari.visual-rule-no-network-replay.v0.1"
    assert report["decision"] == "completed_no_network_artifact_replay"
    assert report["replay_mode"] == "cached_artifacts_only"
    assert report["network_allowed"] is False
    assert report["arc_environment_accessed"] is False
    assert report["sidecar"] == "fixture"
    assert report["source_trace_count"] == 2
    assert report["eval_report"]["scorecard"]["runtime_authority_granted"] is False
    assert report["claims"]["performance"] == "not_claimed"
    assert report["claims"]["generalization"] == "not_claimed"
    assert Path(report["report_path"]).exists()
    assert Path(report["eval_report_path"]).exists()

    manifest = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert manifest["input_trace_pack_paths"] == sorted([moving["trace_pack_path"], stuck["trace_pack_path"]])
    assert all(item["exists"] for item in manifest["input_artifacts"])
    assert all(item["image_count"] >= 2 for item in manifest["input_artifacts"])


def test_no_network_replay_blocks_missing_cached_artifact_before_eval(tmp_path):
    missing = tmp_path / "missing" / "trace_pack.json"

    report = vre.run_visual_rule_no_network_replay([missing], output_dir=tmp_path / "replay")

    assert report["decision"] == "blocked_missing_cached_artifacts"
    assert report["network_allowed"] is False
    assert report["arc_environment_accessed"] is False
    assert report["runtime_authority_granted"] is False
    assert report["input_artifacts"][0]["exists"] is False
    assert "eval_report" not in report
    assert Path(report["report_path"]).exists()
