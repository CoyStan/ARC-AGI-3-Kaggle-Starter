from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"
spec = importlib.util.spec_from_file_location("mini_palari_agent_under_test", MODULE_PATH)
assert spec and spec.loader
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)


class FakeAction:
    def __init__(self, name: str, *, complex_action: bool = False) -> None:
        self.name = name
        self.complex_action = complex_action
        self.data = None
        self.reasoning = None

    def set_data(self, payload):
        self.data = payload


def frame(grid, *, state="NOT_FINISHED", levels_completed=0, available=("ACTION1", "ACTION2", "RESET")):
    return SimpleNamespace(
        frame=grid,
        state=state,
        levels_completed=levels_completed,
        available_actions=list(available),
    )


def actions(*names: str):
    return {name: FakeAction(name, complex_action=(name == "ACTION6")) for name in names}


def test_active_bounded_plan_aborts_on_visible_delta_mismatch_and_advances_probe():
    policy = agent_mod.MiniPalariArcAgi3Policy(game_id="generic", enable_bounded_planning=True)
    policy.active_plan = {
        "schema": "mini-palari.submission-plan-candidate.v0.1",
        "plan_id": "plan-test",
        "actions": ["ACTION1", "ACTION1"],
        "predicted_outcomes": ["changed_from_prior_evidence", "changed_from_prior_evidence"],
        "authority": "candidate_only",
    }
    policy.last_probe_action = "ACTION1"
    policy.last_action_snapshot = {"frame": [[1, 0], [0, 2]], "levels_completed": 0}
    policy.probed_actions.add("ACTION1")

    chosen = policy.choose_action(
        [],
        frame([[1, 0], [0, 2]], available=("ACTION1", "ACTION2", "RESET")),
        actions("ACTION1", "ACTION2", "RESET"),
    )

    assert policy.active_plan is None
    assert chosen.name == "ACTION2"
    assert any(note.get("plan_status") == "aborted_prediction_mismatch" for note in policy.trace_notes)


def test_supported_visible_delta_without_objective_progress_does_not_stall_on_same_action():
    policy = agent_mod.MiniPalariArcAgi3Policy(game_id="generic", enable_bounded_planning=True)
    policy.last_probe_action = "ACTION1"
    policy.last_action_snapshot = {"frame": [[1, 0]], "levels_completed": 0}
    policy.probed_actions.add("ACTION1")

    chosen = policy.choose_action(
        [],
        frame([[0, 1]], available=("ACTION1", "ACTION2", "RESET")),
        actions("ACTION1", "ACTION2", "RESET"),
    )

    assert policy.last_probe_progress_label == "visible_change_not_objective_progress"
    assert chosen.name == "ACTION2"
    assert any(
        note.get("decision") == "advance_probe_after_visible_delta_without_objective_progress"
        for note in policy.trace_notes
    )


def test_single_supported_action_stall_uses_reset_phase_breaker_not_repeat_forever():
    policy = agent_mod.MiniPalariArcAgi3Policy(game_id="generic", enable_bounded_planning=True)
    policy.action_no_progress_repeats["ACTION1"] = policy.max_no_progress_action_repeats

    chosen = policy.choose_action(
        [],
        frame([[1]], available=("ACTION1", "RESET")),
        actions("ACTION1", "RESET"),
    )

    assert chosen.name == "RESET"
    assert any(
        note.get("decision") == "reset_after_repeated_single_simple_action_without_level_progress"
        for note in policy.trace_notes
    )


def test_complex_action6_coordinate_probe_is_not_reset_by_simple_stall_breaker():
    policy = agent_mod.MiniPalariArcAgi3Policy(game_id="generic", enable_bounded_planning=True)
    policy.action_no_progress_repeats["ACTION6"] = policy.max_no_progress_action_repeats

    chosen = policy.choose_action(
        [],
        frame([[0, 5], [0, 0]], available=("ACTION6", "RESET")),
        actions("ACTION6", "RESET"),
    )

    assert chosen.name == "ACTION6"
    assert chosen.data == {"x": 1, "y": 0}
    assert not any(
        note.get("decision") == "reset_after_repeated_single_simple_action_without_level_progress"
        for note in policy.trace_notes
    )


def test_exported_agent_enables_bounded_planning_loop_escape():
    agent = agent_mod.MyAgent(game_id="generic")

    assert agent.policy.enable_bounded_planning is True
