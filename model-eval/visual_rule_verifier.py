"""Trace verifier for visual rule-hypothesis sidecar proposals.

This module does not invoke a model and does not grant runtime authority. It
compares recorded trace geometry against structured proposed rules. New generic
predicate predictions are preferred; legacy named-rule checks remain for older
fixture tests but should not be expanded into a closed-world rule library.
"""

from __future__ import annotations

import importlib.util
from math import isclose
from pathlib import Path
from typing import Any

SCHEMA_VERIFICATION = "mini-palari.visual-rule-verification.v0.1"


def _load_predicate_verifier():
    module_path = Path(__file__).with_name("visual_predicate_verifier.py")
    spec = importlib.util.spec_from_file_location("visual_predicate_verifier", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_predicate_verifier = _load_predicate_verifier()
verify_predicates = _predicate_verifier.verify_predicates


def _components_by_id(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c.get("id")): dict(c) for c in frame.get("components") or [] if c.get("id") is not None}


def _centroid(component: dict[str, Any]) -> tuple[float, float]:
    if isinstance(component.get("centroid"), list) and len(component["centroid"]) >= 2:
        return float(component["centroid"][0]), float(component["centroid"][1])
    bbox = component.get("bbox") or [0, 0, 0, 0]
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _delta(c0: dict[str, Any], c1: dict[str, Any]) -> tuple[float, float]:
    x0, y0 = _centroid(c0)
    x1, y1 = _centroid(c1)
    return x1 - x0, y1 - y0


def _moved(c0: dict[str, Any], c1: dict[str, Any]) -> bool:
    dx, dy = _delta(c0, c1)
    return not (isclose(dx, 0.0, abs_tol=1e-6) and isclose(dy, 0.0, abs_tol=1e-6))


def _bbox(component: dict[str, Any]) -> tuple[float, float, float, float]:
    b = component.get("bbox") or [0, 0, 0, 0]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def _rule_refs(rule: dict[str, Any], key: str, fallback_key: str = "component_refs") -> list[str]:
    refs = rule.get(key)
    if refs is None:
        refs = rule.get(fallback_key)
    if isinstance(refs, str):
        return [refs]
    return [str(x) for x in (refs or [])]


def _rule_result(rule: dict[str, Any], support: list[str], counter_evidence: list[str], *, predicates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if counter_evidence:
        decision = "rejected"
    elif support:
        decision = "supported"
    else:
        decision = "blocked_needs_evidence"
    out = {
        "rule_id": rule.get("rule_id"),
        "type": rule.get("type"),
        "decision": decision,
        "support": support,
        "counter_evidence": counter_evidence,
    }
    if predicates is not None:
        out["predicates"] = predicates
    return out


def _verify_predicate_rule(rule: dict[str, Any], trace_pack: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, Any] | None:
    predictions = rule.get("predictions")
    if not isinstance(predictions, list):
        return None
    predicate_results = verify_predicates(predictions, trace_pack, entity_bindings=hypothesis.get("entities") or [])
    rejected = [p for p in predicate_results if p["decision"] == "rejected"]
    supported = [p for p in predicate_results if p["decision"] == "supported"]
    blocked = [p for p in predicate_results if p["decision"] == "blocked_needs_evidence"]
    support = [f"predicate_supported:{p.get('predicate_id')}" for p in supported]
    counter = [f"predicate_rejected:{p.get('predicate_id')}" for p in rejected]
    if blocked and not supported:
        counter = [f"predicate_blocked:{p.get('predicate_id')}" for p in blocked]
    return _rule_result(rule, support, counter, predicates=predicate_results)


def _verify_mirror_motion(rule: dict[str, Any], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    refs = _rule_refs(rule, "component_refs")
    support: list[str] = []
    counter: list[str] = []
    if len(refs) < 2 or refs[0] not in before or refs[1] not in before or refs[0] not in after or refs[1] not in after:
        counter.append("missing_mirror_component_refs")
    else:
        dx0, dy0 = _delta(before[refs[0]], after[refs[0]])
        dx1, dy1 = _delta(before[refs[1]], after[refs[1]])
        if not isclose(dx0, 0.0, abs_tol=1e-6) and isclose(dx0, -dx1, abs_tol=1e-6):
            support.append("mirror_opposite_x_supported")
        else:
            counter.append("mirror_opposite_x_not_observed")
        if isclose(dy0, dy1, abs_tol=1e-6):
            support.append("mirror_y_consistent")
    return _rule_result(rule, support, counter)


def _verify_selected_endpoint_move(rule: dict[str, Any], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    refs = _rule_refs(rule, "endpoint_refs", "component_refs")
    support: list[str] = []
    counter: list[str] = []
    present = [r for r in refs if r in before and r in after]
    if len(present) < 2:
        counter.append("missing_endpoint_refs")
    else:
        moved = [r for r in present if _moved(before[r], after[r])]
        if len(moved) == 1:
            support.append("exactly_one_endpoint_moved")
            support.append(f"moved_endpoint:{moved[0]}")
        else:
            counter.append(f"selected_endpoint_move_count:{len(moved)}")
    return _rule_result(rule, support, counter)


def _verify_midpoint_body(rule: dict[str, Any], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    refs = _rule_refs(rule, "endpoint_refs", "component_refs")
    body_refs = _rule_refs(rule, "body_ref", "body_refs")
    support: list[str] = []
    counter: list[str] = []
    body = body_refs[0] if body_refs else None
    if len(refs) < 2 or body is None or refs[0] not in after or refs[1] not in after or body not in after:
        counter.append("missing_midpoint_refs")
    else:
        ax, ay = _centroid(after[refs[0]])
        bx, by = _centroid(after[refs[1]])
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        body_x, body_y = _centroid(after[body])
        if isclose(mx, body_x, abs_tol=0.51) and isclose(my, body_y, abs_tol=0.51):
            support.append("body_matches_endpoint_midpoint")
        else:
            counter.append("body_midpoint_mismatch")
    return _rule_result(rule, support, counter)


def _entity_refs_by_role(hypothesis: dict[str, Any], role: str) -> list[str]:
    refs: list[str] = []
    for entity in hypothesis.get("entities") or []:
        if entity.get("role") == role:
            refs.extend(str(r) for r in entity.get("component_refs") or [])
    return refs


def _verify_lava_forbidden(rule: dict[str, Any], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], hypothesis: dict[str, Any]) -> dict[str, Any]:
    avatar_refs = _rule_refs(rule, "avatar_refs") or _entity_refs_by_role(hypothesis, "avatar")
    hazard_refs = _rule_refs(rule, "hazard_refs") or _entity_refs_by_role(hypothesis, "hazard")
    support: list[str] = []
    counter: list[str] = []
    if not avatar_refs or not hazard_refs:
        counter.append("missing_hazard_or_avatar_refs")
    else:
        overlaps = []
        for avatar_ref in avatar_refs:
            for hazard_ref in hazard_refs:
                if avatar_ref in after and hazard_ref in after and _overlap(after[avatar_ref], after[hazard_ref]):
                    overlaps.append((avatar_ref, hazard_ref))
        if overlaps:
            counter.append("hazard_overlap")
            counter.extend(f"overlap:{a}:{h}" for a, h in overlaps)
        else:
            support.append("no_hazard_overlap")
    return _rule_result(rule, support, counter)


def _verify_rule(rule: dict[str, Any], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], hypothesis: dict[str, Any], trace_pack: dict[str, Any]) -> dict[str, Any]:
    predicate_result = _verify_predicate_rule(rule, trace_pack, hypothesis)
    if predicate_result is not None:
        return predicate_result
    rule_type = rule.get("type")
    if rule_type == "mirror_motion":
        return _verify_mirror_motion(rule, before, after)
    if rule_type == "selected_endpoint_move":
        return _verify_selected_endpoint_move(rule, before, after)
    if rule_type == "midpoint_body":
        return _verify_midpoint_body(rule, before, after)
    if rule_type == "lava_forbidden":
        return _verify_lava_forbidden(rule, before, after, hypothesis)
    return _rule_result(rule, [], [f"unsupported_rule_type:{rule_type}"])


def _experiment_predicates(hypothesis: dict[str, Any], trace_pack: dict[str, Any], field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for experiment in hypothesis.get("experiments") or []:
        predicates = [p for p in (experiment.get(field) or []) if isinstance(p, dict)]
        for result in verify_predicates(predicates, trace_pack, entity_bindings=hypothesis.get("entities") or []):
            result["experiment_id"] = experiment.get("experiment_id")
            result["kind"] = field
            out.append(result)
    return out


def verify_visual_rule_hypotheses(
    hypotheses: list[dict[str, Any]],
    trace_pack: dict[str, Any],
    *,
    levels_completed: int = 0,
) -> dict[str, Any]:
    """Verify hypotheses against an observed trace without granting authority."""

    frames = list(trace_pack.get("frames") or [])
    if len(frames) < 2:
        return {
            "schema": SCHEMA_VERIFICATION,
            "decision": "blocked_needs_trace",
            "verifications": [],
            "selected_hypothesis": None,
            "plan_to_execute": [],
            "runtime_authority_granted": False,
        }

    before = _components_by_id(frames[0])
    after = _components_by_id(frames[-1])
    verifications: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []

    for hypothesis in hypotheses:
        rule_results = [_verify_rule(rule, before, after, hypothesis, trace_pack) for rule in hypothesis.get("rules") or []]
        experiment_predictions = _experiment_predicates(hypothesis, trace_pack, "predictions")
        falsifiers = _experiment_predicates(hypothesis, trace_pack, "falsifiers")

        rejected = any(r["decision"] == "rejected" for r in rule_results)
        rejected = rejected or any(p["decision"] == "rejected" for p in experiment_predictions)
        rejected = rejected or any(f["decision"] == "supported" for f in falsifiers)

        supported_count = sum(1 for r in rule_results if r["decision"] == "supported")
        supported_count += sum(1 for p in experiment_predictions if p["decision"] == "supported")
        predicate_support = sum(len(r.get("support") or []) for r in rule_results)
        predicate_support += sum(1 for p in experiment_predictions if p["decision"] == "supported")
        score = int(levels_completed) * 1000 + supported_count * 100 + predicate_support
        decision = "accepted_candidate" if rule_results and not rejected and supported_count > 0 else "rejected"
        record = {
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "decision": decision,
            "score": score,
            "rules": rule_results,
            "experiment_predictions": experiment_predictions,
            "falsifiers": falsifiers,
            "runtime_authority_granted": False,
        }
        verifications.append(record)
        if decision == "accepted_candidate":
            survivors.append({"hypothesis": hypothesis, "verification": record})

    if survivors:
        best = sorted(survivors, key=lambda item: item["verification"]["score"], reverse=True)[0]
        return {
            "schema": SCHEMA_VERIFICATION,
            "decision": "accepted_candidate",
            "verifications": verifications,
            "selected_hypothesis": best["hypothesis"],
            "plan_to_execute": list(best["hypothesis"].get("plan_if_true") or []),
            "runtime_authority_granted": False,
        }

    return {
        "schema": SCHEMA_VERIFICATION,
        "decision": "rejected",
        "verifications": verifications,
        "selected_hypothesis": None,
        "plan_to_execute": [],
        "runtime_authority_granted": False,
    }
