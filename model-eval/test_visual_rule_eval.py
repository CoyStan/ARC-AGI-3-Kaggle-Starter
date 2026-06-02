import importlib.util
import json
from pathlib import Path

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


def make_trace(tmp_path, game_id, frames, actions=None):
    return vta.write_trace_pack(
        output_dir=tmp_path / game_id,
        game_id=game_id,
        frames=frames,
        actions=actions or [1],
        available_actions=[1, 2, 6],
        title=f"trace {game_id}",
        image_scale=1,
    )


def test_fixture_eval_runs_multiple_cached_traces_and_writes_scorecard(tmp_path):
    moving = make_trace(
        tmp_path,
        "moving_game",
        frames=[[[1, 0, 0]], [[0, 1, 0]]],
    )
    stuck = make_trace(
        tmp_path,
        "stuck_game",
        frames=[[[1, 0, 0]], [[1, 0, 0]]],
    )

    report = vre.run_visual_rule_eval(
        [moving["trace_pack_path"], stuck["trace_pack_path"]],
        output_dir=tmp_path / "eval",
        sidecar="fixture",
    )

    assert report["schema"] == "mini-palari.visual-rule-eval.v0.1"
    assert report["sidecar"] == "fixture"
    assert report["game_count"] == 2
    assert report["scorecard"]["accepted_candidates"] == 1
    assert report["scorecard"]["rejected"] == 1
    assert report["scorecard"]["runtime_authority_granted"] is False
    assert report["claims"]["generalization"] == "not_claimed"
    assert Path(report["report_path"]).exists()

    by_game = {entry["game_id"]: entry for entry in report["games"]}
    assert by_game["moving_game"]["verification_decision"] == "accepted_candidate"
    assert by_game["stuck_game"]["verification_decision"] == "rejected"
    for entry in report["games"]:
        assert Path(entry["trace_pack_path"]).exists()
        assert Path(entry["prompt_path"]).exists()
        assert Path(entry["sidecar_log_path"]).exists()
        assert entry["runtime_authority_granted"] is False


def test_eval_report_round_trips_from_json(tmp_path):
    trace = make_trace(tmp_path, "moving_game", frames=[[[1, 0]], [[0, 1]]])

    report = vre.run_visual_rule_eval([trace["trace_pack_path"]], output_dir=tmp_path / "eval")
    loaded = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert loaded["schema"] == report["schema"]
    assert loaded["games"][0]["game_id"] == "moving_game"
    assert loaded["games"][0]["sidecar_validation_decision"] == "accepted_candidate"
    assert loaded["runtime_authority_granted"] is False


def test_eval_blocks_empty_trace_list_with_no_runtime_authority(tmp_path):
    report = vre.run_visual_rule_eval([], output_dir=tmp_path / "eval")

    assert report["decision"] == "blocked_no_traces"
    assert report["games"] == []
    assert report["scorecard"]["runtime_authority_granted"] is False
    assert Path(report["report_path"]).exists()
