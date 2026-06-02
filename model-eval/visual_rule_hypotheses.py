"""Parser and validator for visual rule-hypothesis sidecar output.

This module treats VLM/LLM output as untrusted proposal text. It accepts only
schema-shaped, proposal-only hypotheses and returns separate validation status;
it never grants runtime/action authority.
"""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_HYPOTHESIS = "mini-palari.visual-rule-hypothesis.v0.1"
REQUIRED_HYPOTHESIS_FIELDS = {
    "schema",
    "hypothesis_id",
    "game_family",
    "entities",
    "rules",
    "experiments",
    "plan_if_true",
    "authority",
}
LIST_FIELDS = {"entities", "rules", "experiments", "plan_if_true"}


SUPPORTED_PREDICATE_RELATIONS = {
    "moved",
    "stayed",
    "delta_x_sign",
    "delta_y_sign",
    "same_delta_x",
    "opposite_delta_x",
    "same_delta_y",
    "opposite_delta_y",
    "distance_decreased",
    "distance_increased",
    "distance_unchanged",
    "overlap",
    "no_overlap",
    "inside_bbox",
    "outside_bbox",
    "midpoint_matches",
    "component_appeared",
    "component_disappeared",
    "component_count_changed",
    "color_count_changed",
    "level_completed",
    "reset_or_failure",
}


def json_dumps(value: Any) -> str:
    """Stable JSON helper used by tests and prompt/eval logging."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _json_loads_candidate(text: str) -> tuple[Any | None, bool, str | None]:
    candidates: list[tuple[str, bool]] = []
    stripped = text.strip()
    if stripped:
        candidates.append((stripped, False))

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.append((match.group(1).strip(), True))

    # Fallback: largest object-shaped slice. This is intentionally conservative;
    # full JSON salvage/repair belongs in later adapter code.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidates.append((text[first : last + 1], True))

    last_error: str | None = None
    for candidate, salvaged in candidates:
        try:
            return json.loads(candidate), salvaged, None
        except Exception as exc:  # noqa: BLE001 - untrusted sidecar text
            last_error = f"{type(exc).__name__}: {exc}"
    return None, bool(candidates), last_error or "no_json_candidate"


def _coerce_payload(payload: Any) -> tuple[list[Any], bool, str | None]:
    if isinstance(payload, str):
        decoded, salvaged, error = _json_loads_candidate(payload)
        if decoded is None:
            return [], salvaged, error
        payload = decoded
        salvaged_from_text = salvaged
    else:
        salvaged_from_text = False

    if isinstance(payload, dict) and isinstance(payload.get("hypotheses"), list):
        return list(payload["hypotheses"]), salvaged_from_text, None
    if isinstance(payload, list):
        return list(payload), salvaged_from_text, None
    if isinstance(payload, dict) and payload.get("schema") == SCHEMA_HYPOTHESIS:
        return [payload], salvaged_from_text, None
    return [], salvaged_from_text, "missing_hypotheses_array"


def _action_ids_from_hypothesis(hypothesis: dict[str, Any]) -> list[int]:
    found: list[int] = []
    for section in ("experiments", "plan_if_true"):
        for item in hypothesis.get(section) or []:
            if not isinstance(item, dict):
                continue
            for action in item.get("actions") or []:
                try:
                    found.append(int(action))
                except Exception:
                    found.append(-999999)
    return found


def _validate_predicate(predicate: Any) -> bool:
    if not isinstance(predicate, dict):
        return False
    if not predicate.get("predicate_id") or not predicate.get("relation"):
        return False
    if predicate.get("relation") not in SUPPORTED_PREDICATE_RELATIONS:
        return False
    if not isinstance(predicate.get("time"), dict):
        return False
    if not ({"from", "to"} <= set(predicate["time"].keys()) or "at" in predicate["time"]):
        return False
    return True


def _validate_predicate_sections(hypothesis: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for i, rule in enumerate(hypothesis.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        if "predictions" in rule:
            if not isinstance(rule.get("predictions"), list):
                reasons.append(f"field_must_be_list:rules[{i}].predictions")
                continue
            for j, predicate in enumerate(rule.get("predictions") or []):
                if not _validate_predicate(predicate):
                    reasons.append(f"malformed_predicate:rules[{i}].predictions[{j}]")
    for i, experiment in enumerate(hypothesis.get("experiments") or []):
        if not isinstance(experiment, dict):
            continue
        for field in ("predictions", "falsifiers"):
            if field in experiment:
                if not isinstance(experiment.get(field), list):
                    reasons.append(f"field_must_be_list:experiments[{i}].{field}")
                    continue
                for j, predicate in enumerate(experiment.get(field) or []):
                    # Backward compatibility: existing natural-language falsifier
                    # lists remain accepted until sidecar prompts fully migrate.
                    if isinstance(predicate, str):
                        continue
                    if not _validate_predicate(predicate):
                        reasons.append(f"malformed_predicate:experiments[{i}].{field}[{j}]")
    return reasons


def _validate_hypothesis(hypothesis: Any, available_actions: set[int] | None) -> list[str]:
    reasons: list[str] = []
    if not isinstance(hypothesis, dict):
        return ["hypothesis_not_object"]

    for field in sorted(REQUIRED_HYPOTHESIS_FIELDS):
        if field not in hypothesis:
            reasons.append(f"missing_required_field:{field}")

    if hypothesis.get("schema") != SCHEMA_HYPOTHESIS:
        reasons.append("schema_mismatch")
    if hypothesis.get("authority") != "proposal_only":
        reasons.append("authority_must_be_proposal_only")

    for field in sorted(LIST_FIELDS):
        if field in hypothesis and not isinstance(hypothesis.get(field), list):
            reasons.append(f"field_must_be_list:{field}")
        elif field in hypothesis and len(hypothesis.get(field) or []) == 0:
            reasons.append(f"field_must_be_nonempty_list:{field}")

    if available_actions is not None:
        for action_id in _action_ids_from_hypothesis(hypothesis):
            if action_id not in available_actions:
                reasons.append(f"action_not_available:{action_id}")

    reasons.extend(_validate_predicate_sections(hypothesis))

    return reasons


def parse_visual_rule_hypotheses(
    payload: Any,
    *,
    available_actions: list[int] | set[int] | tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse and validate proposal-only visual rule hypotheses.

    Returns ``(accepted_hypotheses, validation_status)``. Accepted hypotheses are
    shallow copies with ``authority`` forced to ``proposal_only``. Runtime
    authority is always false, even if a model tries to claim otherwise.
    """

    action_set = {int(a) for a in available_actions} if available_actions is not None else None
    raw_hypotheses, salvaged, parse_error = _coerce_payload(payload)
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for index, candidate in enumerate(raw_hypotheses):
        reasons = _validate_hypothesis(candidate, action_set)
        if reasons:
            hyp_id = candidate.get("hypothesis_id") if isinstance(candidate, dict) else None
            rejections.append({"index": index, "hypothesis_id": hyp_id, "reasons": reasons})
            continue
        copied = dict(candidate)
        copied["authority"] = "proposal_only"
        accepted.append(copied)

    if accepted:
        decision = "accepted_candidate"
    elif rejections or parse_error:
        decision = "rejected"
    else:
        decision = "blocked_needs_hypotheses"

    status: dict[str, Any] = {
        "schema": "mini-palari.visual-rule-hypothesis-validation.v0.1",
        "decision": decision,
        "accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "rejections": rejections,
        "salvaged_from_text": bool(salvaged),
        "runtime_authority_granted": False,
    }
    if parse_error:
        status["parse_error"] = parse_error
    return accepted, status
