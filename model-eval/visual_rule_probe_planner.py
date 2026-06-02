"""Plan bounded local probes from GPT-5.5 visual-rule proposal artifacts.

This module bridges human/VLM visual hypotheses into a deterministic research
harness artifact. It deliberately stops at a proposal-only probe plan: no ARC
runtime policy, no action authority, no coordinates are executed here. A later
executor/verifier can bind target intents to frame coordinates, run bounded
probes, and fill in the pending verification rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PROBE_PLAN = "mini-palari.visual-rule-probe-plan.v0.1"
_SIMPLE_ACTIONS = [1, 2, 3, 4, 5, 7]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _input_authority_blocked(payload: dict[str, Any]) -> bool:
    auth = payload.get("authority") or {}
    return any(
        bool(auth.get(key))
        for key in (
            "runtime_authority_granted",
            "policy_rank_authority_granted",
            "action_payload_authority_granted",
            "kaggle_submission_authority_granted",
        )
    )


def _bounded(actions: list[int], max_actions: int) -> list[int]:
    limit = max(0, int(max_actions))
    return [int(a) for a in actions[:limit]]


def _infer_target_intents(template: str) -> list[str]:
    text = template.lower()
    intents: list[str] = []
    if "target_ring" in text or "ring center" in text:
        intents.append("target_ring.center")
    if "next dotted waypoint" in text:
        intents.append("dotted_path.next_waypoint_after_source")
    if "final waypoint" in text:
        intents.append("dotted_path.final_waypoint_before_ring")
    if "bottom swatch" in text or "swatch" in text:
        intents.append("bottom_swatches.centers")
    if "top slot" in text or "slot" in text:
        intents.append("top_slots.centers")
    if "black endpoint" in text:
        intents.append("chain_head_candidate.center")
    if "green endpoint" in text:
        intents.append("chain_tail_candidate.center")
    if "bar endpoints" in text or "bar endpoint" in text:
        intents.append("yellow_bars.endpoints")
    if "midpoint" in text:
        intents.append("candidate_components.midpoints")
    if "socket" in text:
        intents.append("pink_sockets.centers")
    if "corridor mouth" in text:
        intents.append("corridor.entry")
    if "through corridor" in text:
        intents.append("corridor.exit")
    if "target" in text and not any("target" in item for item in intents):
        intents.append("target_candidate.center")
    # Deterministic de-dupe while preserving order.
    out: list[str] = []
    for intent in intents:
        if intent not in out:
            out.append(intent)
    return out


def _probe_kind_and_actions(template: str, max_actions: int) -> tuple[str, list[int], bool, list[str]]:
    text = template.lower()
    target_intents = _infer_target_intents(template)
    if "action6" in text or "click" in text:
        count = max(1, min(len(target_intents) or 1, max_actions))
        return "coordinate_target", [6] * count, True, target_intents[:count] or ["salient_target.center"]
    if "shortest" in text or "path" in text or "corridor" in text or "route" in text:
        return "topology_path_prefix", _bounded(_SIMPLE_ACTIONS, max_actions), False, target_intents
    if "cycle simple" in text or "simple movement" in text or "movement actions" in text:
        return "simple_action_cycle", _bounded(_SIMPLE_ACTIONS, max_actions), False, target_intents
    return "bounded_observation_probe", _bounded(_SIMPLE_ACTIONS, max_actions), False, target_intents


def _probe_from_experiment(
    *,
    game_id: str,
    hypothesis_id: str,
    hypothesis_statement: str,
    experiment: dict[str, Any],
    max_actions_per_experiment: int,
) -> dict[str, Any]:
    template = str(experiment.get("probe_template") or experiment.get("purpose") or "")
    kind, actions, needs_coords, target_intents = _probe_kind_and_actions(template, max_actions_per_experiment)
    return {
        "game_id": game_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis_statement": hypothesis_statement,
        "hypothesis_statement_chars": len(hypothesis_statement),
        "experiment_id": str(experiment.get("experiment_id") or f"{hypothesis_id}-experiment"),
        "purpose": experiment.get("purpose"),
        "probe_template": template,
        "probe_kind": kind,
        "action_sequence": actions,
        "action_budget": len(actions),
        "target_intents": target_intents,
        "requires_coordinate_binding": needs_coords,
        "predicates_to_verify": list(experiment.get("predictions") or []),
        "falsifiers_to_check": list(experiment.get("falsifiers") or []),
        "verification_status": "pending_execution",
        "blocked_reason": "needs_coordinate_binding" if needs_coords else None,
        "runtime_authority_granted": False,
    }


def _verification_summary(games: list[dict[str, Any]], *, decision: str = "blocked_needs_probe_execution") -> dict[str, Any]:
    probes = [probe for game in games for probe in game.get("probes", [])]
    return {
        "decision": decision,
        "probe_count": len(probes),
        "pending_execution": sum(1 for p in probes if p.get("verification_status") == "pending_execution"),
        "requires_coordinate_binding": sum(1 for p in probes if p.get("requires_coordinate_binding")),
        "runtime_authority_granted": False,
    }


def build_probe_plan_from_gpt55_proposals(
    payload: dict[str, Any],
    *,
    max_actions_per_experiment: int = 4,
) -> dict[str, Any]:
    """Convert GPT-5.5 proposal JSON into bounded probe-plan rows."""

    if _input_authority_blocked(payload):
        return {
            "schema": SCHEMA_PROBE_PLAN,
            "created_at": _now_iso(),
            "source_schema": payload.get("schema"),
            "decision": "blocked_authority_escalation",
            "blocker": "input_payload_claims_authority",
            "game_count": 0,
            "probe_count": 0,
            "games": [],
            "verification_summary": {"decision": "blocked_authority_escalation", "runtime_authority_granted": False},
            "claims": {"generalization": "not_claimed", "performance": "not_claimed", "scope": "proposal_to_probe_plan_only"},
            "runtime_authority_granted": False,
            "policy_rank_authority_granted": False,
            "action_payload_authority_granted": False,
        }

    games: list[dict[str, Any]] = []
    for game in payload.get("games") or []:
        game_id = str(game.get("game_id") or "unknown_game")
        probes: list[dict[str, Any]] = []
        for hypothesis in game.get("hypotheses") or []:
            if hypothesis.get("authority") != "proposal_only":
                continue
            hypothesis_id = str(hypothesis.get("hypothesis_id") or f"{game_id}-hypothesis")
            hypothesis_statement = str(hypothesis.get("statement") or "")
            for experiment in hypothesis.get("experiments") or []:
                probes.append(
                    _probe_from_experiment(
                        game_id=game_id,
                        hypothesis_id=hypothesis_id,
                        hypothesis_statement=hypothesis_statement,
                        experiment=experiment,
                        max_actions_per_experiment=max_actions_per_experiment,
                    )
                )
        games.append(
            {
                "game_id": game_id,
                "contact_sheet": game.get("contact_sheet"),
                "probe_count": len(probes),
                "probes": probes,
                "runtime_authority_granted": False,
            }
        )

    probe_count = sum(int(game.get("probe_count") or 0) for game in games)
    return {
        "schema": SCHEMA_PROBE_PLAN,
        "created_at": _now_iso(),
        "source_schema": payload.get("schema"),
        "decision": "planned_pending_execution",
        "game_count": len(games),
        "probe_count": probe_count,
        "games": games,
        "verification_summary": _verification_summary(games),
        "claims": {"generalization": "not_claimed", "performance": "not_claimed", "scope": "proposal_to_probe_plan_only"},
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
    }


def write_probe_plan_from_file(
    proposal_path: str | Path,
    *,
    output_dir: str | Path,
    max_actions_per_experiment: int = 4,
) -> dict[str, Any]:
    """Load proposal JSON, write a bounded probe-plan artifact, and return it."""

    source = Path(proposal_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_probe_plan_from_gpt55_proposals(
        payload,
        max_actions_per_experiment=max_actions_per_experiment,
    )
    report["source_proposal_path"] = str(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "visual_rule_probe_plan.json"
    report["report_path"] = str(report_path)
    report_path.write_text(_stable_json(report) + "\n", encoding="utf-8")
    return report
