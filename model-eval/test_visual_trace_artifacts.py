import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("visual_trace_artifacts.py")
spec = importlib.util.spec_from_file_location("visual_trace_artifacts", MODULE_PATH)
assert spec is not None
vta = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vta)


def test_normalize_frame_accepts_2d_and_singleton_3d_payloads():
    assert vta.normalize_frame([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    assert vta.normalize_frame([[[1, 2], [3, 4]]]) == [[1, 2], [3, 4]]
    assert vta.normalize_frame({"frame": [[5, 6]]}) == [[5, 6]]


def test_connected_components_are_deterministic_with_bboxes():
    frame = [
        [1, 1, 0, 2],
        [0, 1, 0, 2],
        [3, 0, 0, 2],
    ]
    components = vta.connected_components(frame)
    assert [c["id"] for c in components] == ["c0", "c1", "c2"]
    assert components[0]["color"] == 1
    assert components[0]["bbox"] == [0, 0, 1, 1]
    assert components[0]["area"] == 3
    assert components[1]["color"] == 2
    assert components[1]["bbox"] == [3, 0, 3, 2]
    assert components[2]["color"] == 3


def test_frame_delta_reports_changed_cells_and_bbox():
    before = [[0, 1], [1, 1]]
    after = [[0, 2], [1, 3]]
    delta = vta.frame_delta(before, after)
    assert delta["changed_cells"] == 2
    assert delta["bbox"] == [1, 0, 1, 1]
    assert delta["changed_points"] == [[1, 0], [1, 1]]


def test_write_trace_pack_emits_json_and_ppm_images(tmp_path):
    frames = [
        [[0, 1], [2, 2]],
        [[0, 1], [2, 3]],
    ]
    pack = vta.write_trace_pack(
        output_dir=tmp_path / "trace",
        game_id="synthetic",
        frames=frames,
        actions=[6],
        available_actions=[1, 2, 6],
        title="unit-test trace",
    )
    pack_path = Path(pack["trace_pack_path"])
    assert pack_path.exists()
    assert pack["schema"] == "mini-palari.visual-trace-pack.v0.1"
    assert pack["game_id"] == "synthetic"
    assert len(pack["frames"]) == 2
    assert Path(pack["frames"][0]["image_path"]).exists()
    assert Path(pack["frames"][0]["image_path"]).suffix == ".ppm"
    assert pack["deltas"][0]["changed_cells"] == 1
    loaded = vta.load_trace_pack(pack_path)
    assert loaded["actions"] == [6]
    assert loaded["available_actions"] == [1, 2, 6]
