import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).with_name("qwen_proposer_eval.py")
spec = importlib.util.spec_from_file_location("qwen_proposer_eval", MODULE_PATH)
assert spec is not None
qpe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qpe)


def frame(values):
    return SimpleNamespace(frame=values, levels_completed=0)


def test_generic_visible_delta_does_not_survive_without_prediction_match():
    hypothesis = {
        "hypothesis_id": "h1",
        "game_family": "navigation",
        "play_plan": [1, 2],
        "expected_observations": "object c99 reaches target color 7",
        "failure_conditions": "object c99 does not move",
    }

    trial = qpe.score_hypothesis_trial(
        hypothesis=hypothesis,
        initial_frame=frame([[0, 0], [0, 0]]),
        final_frame=frame([[0, 1], [0, 0]]),
        result={"levels_completed": 0, "actions": 2},
        plan=[1, 2],
    )

    assert trial["delta_cells"] == 1
    assert trial["prediction_matches"] == []
    assert trial["survived"] is False
    assert trial["score"] < 1


def test_level_completion_survives_even_without_text_prediction_match():
    hypothesis = {
        "hypothesis_id": "h2",
        "game_family": "object-control",
        "play_plan": [6],
        "expected_observations": "object is placed",
    }

    trial = qpe.score_hypothesis_trial(
        hypothesis=hypothesis,
        initial_frame=frame([[0]]),
        final_frame=frame([[0]]),
        result={"levels_completed": 1, "actions": 1},
        plan=[6],
    )

    assert trial["survived"] is True
    assert trial["score"] >= 100000


def test_delta_prediction_match_survives_with_limited_score():
    hypothesis = {
        "hypothesis_id": "h3",
        "game_family": "paint",
        "play_plan": [6],
        "expected_observations": "frame changes / visible delta appears",
    }

    trial = qpe.score_hypothesis_trial(
        hypothesis=hypothesis,
        initial_frame=frame([[0, 0], [0, 0]]),
        final_frame=frame([[0, 2], [0, 0]]),
        result={"levels_completed": 0, "actions": 1},
        plan=[6],
    )

    assert "delta_expected_and_observed" in trial["prediction_matches"]
    assert trial["survived"] is True
    assert 1 <= trial["score"] < 100000


def test_referenced_component_change_counts_as_prediction_match():
    hypothesis = {
        "hypothesis_id": "h4",
        "game_family": "object-control",
        "play_plan": [6],
        "expected_observations": "object c1 moves or changes",
    }

    trial = qpe.score_hypothesis_trial(
        hypothesis=hypothesis,
        initial_frame=frame([[1, 1, 0], [0, 2, 0]]),
        final_frame=frame([[1, 1, 0], [0, 3, 0]]),
        result={"levels_completed": 0, "actions": 1},
        plan=[6],
    )

    assert "component_c1_changed" in trial["prediction_matches"]
    assert trial["survived"] is True
