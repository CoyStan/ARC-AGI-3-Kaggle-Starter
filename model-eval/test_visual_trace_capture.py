import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_trace_capture.py")
spec = importlib.util.spec_from_file_location("visual_trace_capture", MODULE_PATH)
assert spec is not None
vtc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vtc)


def test_probe_actions_are_simple_available_non_reset_actions():
    assert vtc.choose_probe_actions([0, 6, 2, 1, 7, 3], max_steps=4) == [2, 1, 7, 3]
    assert vtc.choose_probe_actions([0, 6], max_steps=3) == []


def test_write_trace_capture_from_frames_creates_multiframe_trace_and_manifest(tmp_path):
    result = vtc.write_trace_capture_from_frames(
        game_id="toy",
        frames=[[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
        actions=[1, 1],
        available_actions=[1, 2, 6],
        output_root=tmp_path,
    )

    trace_path = Path(result["trace_pack_path"])
    manifest_path = Path(result["manifest_path"])
    assert trace_path.exists()
    assert manifest_path.exists()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert trace["game_id"] == "toy"
    assert len(trace["frames"]) == 3
    assert len(trace["deltas"]) == 2
    assert trace["actions"] == [1, 1]
    assert manifest["runtime_authority_granted"] is False
    assert manifest["captures"][0]["frame_count"] == 3
    assert manifest["captures"][0]["status"] == "captured"


def test_write_blocked_capture_manifest_is_auditable(tmp_path):
    manifest = vtc.write_capture_manifest(
        output_root=tmp_path,
        requested_game_ids=["aa11", "bb22"],
        captures=[],
        blocker="arc_env_unavailable",
    )

    path = Path(manifest["manifest_path"])
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["decision"] == "blocked"
    assert loaded["blocker"] == "arc_env_unavailable"
    assert loaded["requested_game_ids"] == ["aa11", "bb22"]
    assert loaded["runtime_authority_granted"] is False
