"""Generic observable-predicate verifier for visual rule hypotheses.

This module intentionally verifies observable trace facts rather than named game
mechanics. A model may label a rule "mirror puzzle" or "spider body", but local
acceptance should come from generic predicates over recorded frames: deltas,
overlaps, containment, distances, midpoint geometry, and structural changes.
"""

from __future__ import annotations

from collections import Counter
from math import hypot, isclose
from typing import Any

SUPPORTED_RELATIONS = {
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
    "target_contact",
    "hazard_contact",
    "object_entered_goal_region",
    "object_left_goal_region",
    "progress_toward_static_target",
    "score_proxy_improved",
    "level_completed",
    "reset_or_failure",
}


def _result(
    predicate: dict[str, Any],
    decision: str,
    support: list[str] | None = None,
    counter: list[str] | None = None,
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "predicate_id": predicate.get("predicate_id"),
        "relation": predicate.get("relation"),
        "decision": decision,
        "support": list(support or []),
        "counter_evidence": list(counter or []),
        "observed": dict(observed or {}),
        "runtime_authority_granted": False,
    }


def _components(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c.get("id")): dict(c) for c in frame.get("components") or [] if c.get("id") is not None}


def _bbox(c: dict[str, Any]) -> tuple[float, float, float, float]:
    b = c.get("bbox") or [0, 0, 0, 0]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _centroid(c: dict[str, Any]) -> tuple[float, float]:
    centroid = c.get("centroid")
    if isinstance(centroid, list) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    x0, y0, x1, y1 = _bbox(c)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _delta(a0: dict[str, Any], a1: dict[str, Any]) -> tuple[float, float]:
    x0, y0 = _centroid(a0)
    x1, y1 = _centroid(a1)
    return x1 - x0, y1 - y0


def _sign(v: float) -> str:
    if isclose(v, 0.0, abs_tol=1e-6):
        return "zero"
    return "positive" if v > 0 else "negative"


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def _inside(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return ax0 >= bx0 and ay0 >= by0 and ax1 <= bx1 and ay1 <= by1


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return hypot(ax - bx, ay - by)


def _entity_map(entity_bindings: Any) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if isinstance(entity_bindings, dict):
        for key, value in entity_bindings.items():
            if isinstance(value, str):
                mapping[str(key)] = [value]
            elif isinstance(value, list):
                mapping[str(key)] = [str(v) for v in value]
        return mapping
    for entity in entity_bindings or []:
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("entity_id")
        refs = entity.get("component_refs") or []
        if entity_id is not None:
            mapping[str(entity_id)] = [str(r) for r in refs]
    return mapping


def _resolve_subjects(subjects: Any, entity_bindings: Any) -> list[str]:
    mapping = _entity_map(entity_bindings)
    resolved: list[str] = []
    for subject in subjects or []:
        sid = str(subject)
        refs = mapping.get(sid, [sid])
        resolved.extend(refs[:1])
    return resolved


def _frame_at(frames: list[dict[str, Any]], idx: Any) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    raw, err = _raw_frame_at(frames, idx)
    if err:
        return None, err
    assert raw is not None
    return _components(raw), None


def _raw_frame_at(frames: list[dict[str, Any]], idx: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        i = int(idx)
    except Exception:
        return None, f"invalid_frame_index:{idx}"
    if i < 0 or i >= len(frames):
        return None, f"frame_index_out_of_range:{i}"
    return frames[i], None


def _pair_frames(frames: list[dict[str, Any]], time: dict[str, Any]) -> tuple[dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]] | None, str | None]:
    if not isinstance(time, dict) or "from" not in time or "to" not in time:
        return None, None, "missing_time_from_to"
    before, err = _frame_at(frames, time.get("from"))
    if err:
        return None, None, err
    after, err = _frame_at(frames, time.get("to"))
    if err:
        return None, None, err
    return before, after, None


def _at_frames(frames: list[dict[str, Any]], time: dict[str, Any]) -> tuple[list[tuple[int, dict[str, dict[str, Any]]]], str | None]:
    if not isinstance(time, dict) or "at" not in time:
        return [], "missing_time_at"
    at = time.get("at")
    if at == "all":
        return [(i, _components(frame)) for i, frame in enumerate(frames)], None
    frame, err = _frame_at(frames, at)
    if err:
        return [], err
    return [(int(at), frame)], None


def _need_subjects(predicate: dict[str, Any], resolved: list[str], count: int) -> str | None:
    if len(resolved) < count:
        return f"not_enough_subjects:{len(resolved)}<{count}"
    return None


def _need_present(frame: dict[str, dict[str, Any]], refs: list[str]) -> str | None:
    missing = [r for r in refs if r not in frame]
    if missing:
        return "missing_subjects:" + ",".join(missing)
    return None


def _frame_flag(frame: dict[str, Any], *names: str) -> bool:
    return any(bool(frame.get(name)) for name in names)


def _score_proxy(frame: dict[str, Any]) -> float | None:
    for key in ("score_proxy", "score", "level_score", "reward"):
        if key in frame and frame.get(key) is not None:
            value = frame[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _raw_pair_frames(frames: list[dict[str, Any]], time: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    if not isinstance(time, dict) or "from" not in time or "to" not in time:
        return None, None, "missing_time_from_to"
    before, err = _raw_frame_at(frames, time.get("from"))
    if err:
        return None, None, err
    after, err = _raw_frame_at(frames, time.get("to"))
    if err:
        return None, None, err
    return before, after, None


def _raw_at_frames(frames: list[dict[str, Any]], time: dict[str, Any]) -> tuple[list[tuple[int, dict[str, Any]]], str | None]:
    if not isinstance(time, dict) or "at" not in time:
        return [], "missing_time_at"
    at = time.get("at")
    if at == "all":
        return [(i, frame) for i, frame in enumerate(frames)], None
    frame, err = _raw_frame_at(frames, at)
    if err:
        return [], err
    assert frame is not None
    return [(int(str(at)), frame)], None


def _verify_pair(predicate: dict[str, Any], frames: list[dict[str, Any]], subjects: list[str]) -> dict[str, Any]:
    relation = predicate.get("relation")
    before, after, err = _pair_frames(frames, predicate.get("time") or {})
    if err:
        return _result(predicate, "blocked_needs_evidence", counter=[err])
    assert before is not None and after is not None

    if relation in {"moved", "stayed", "delta_x_sign", "delta_y_sign"}:
        err = _need_subjects(predicate, subjects, 1) or _need_present(before, [subjects[0]]) or _need_present(after, [subjects[0]])
        if err:
            return _result(predicate, "blocked_needs_evidence", counter=[err])
        dx, dy = _delta(before[subjects[0]], after[subjects[0]])
        observed = {"subject": subjects[0], "dx": dx, "dy": dy, "delta_x_sign": _sign(dx), "delta_y_sign": _sign(dy)}
        if relation == "moved":
            ok = not (observed["delta_x_sign"] == "zero" and observed["delta_y_sign"] == "zero")
        elif relation == "stayed":
            ok = observed["delta_x_sign"] == "zero" and observed["delta_y_sign"] == "zero"
        elif relation == "delta_x_sign":
            ok = observed["delta_x_sign"] == predicate.get("value")
        else:
            ok = observed["delta_y_sign"] == predicate.get("value")
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else [f"{relation}_mismatch"], observed)

    if relation in {"same_delta_x", "opposite_delta_x", "same_delta_y", "opposite_delta_y"}:
        err = _need_subjects(predicate, subjects, 2) or _need_present(before, subjects[:2]) or _need_present(after, subjects[:2])
        if err:
            return _result(predicate, "blocked_needs_evidence", counter=[err])
        dx0, dy0 = _delta(before[subjects[0]], after[subjects[0]])
        dx1, dy1 = _delta(before[subjects[1]], after[subjects[1]])
        observed = {"subjects": subjects[:2], "dx": [dx0, dx1], "dy": [dy0, dy1]}
        if relation == "same_delta_x":
            ok = isclose(dx0, dx1, abs_tol=1e-6)
        elif relation == "opposite_delta_x":
            ok = not isclose(dx0, 0.0, abs_tol=1e-6) and isclose(dx0, -dx1, abs_tol=1e-6)
        elif relation == "same_delta_y":
            ok = isclose(dy0, dy1, abs_tol=1e-6)
        else:
            ok = not isclose(dy0, 0.0, abs_tol=1e-6) and isclose(dy0, -dy1, abs_tol=1e-6)
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else [f"{relation}_mismatch"], observed)

    if relation in {"distance_decreased", "distance_increased", "distance_unchanged"}:
        err = _need_subjects(predicate, subjects, 2) or _need_present(before, subjects[:2]) or _need_present(after, subjects[:2])
        if err:
            return _result(predicate, "blocked_needs_evidence", counter=[err])
        d0 = _distance(before[subjects[0]], before[subjects[1]])
        d1 = _distance(after[subjects[0]], after[subjects[1]])
        observed = {"distance_before": d0, "distance_after": d1, "delta": d1 - d0}
        if relation == "distance_decreased":
            ok = d1 < d0 and not isclose(d1, d0, abs_tol=1e-6)
        elif relation == "distance_increased":
            ok = d1 > d0 and not isclose(d1, d0, abs_tol=1e-6)
        else:
            ok = isclose(d1, d0, abs_tol=1e-6)
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else [f"{relation}_mismatch"], observed)

    if relation in {"object_entered_goal_region", "object_left_goal_region", "progress_toward_static_target"}:
        err = _need_subjects(predicate, subjects, 2) or _need_present(before, subjects[:2]) or _need_present(after, subjects[:2])
        if err:
            return _result(predicate, "blocked_needs_evidence", counter=[err])
        before_overlap = _overlap(before[subjects[0]], before[subjects[1]])
        after_overlap = _overlap(after[subjects[0]], after[subjects[1]])
        d0 = _distance(before[subjects[0]], before[subjects[1]])
        d1 = _distance(after[subjects[0]], after[subjects[1]])
        target_dx, target_dy = _delta(before[subjects[1]], after[subjects[1]])
        observed = {
            "subjects": subjects[:2],
            "overlap_before": before_overlap,
            "overlap_after": after_overlap,
            "distance_before": d0,
            "distance_after": d1,
            "distance_delta": d1 - d0,
            "target_delta": [target_dx, target_dy],
        }
        if relation == "object_entered_goal_region":
            ok = not before_overlap and after_overlap
        elif relation == "object_left_goal_region":
            ok = before_overlap and not after_overlap
        else:
            target_static = isclose(target_dx, 0.0, abs_tol=1e-6) and isclose(target_dy, 0.0, abs_tol=1e-6)
            ok = target_static and d1 < d0 and not isclose(d1, d0, abs_tol=1e-6)
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else [f"{relation}_mismatch"], observed)

    if relation == "score_proxy_improved":
        raw_before, raw_after, raw_err = _raw_pair_frames(frames, predicate.get("time") or {})
        if raw_err:
            return _result(predicate, "blocked_needs_evidence", counter=[raw_err])
        assert raw_before is not None and raw_after is not None
        score_before = _score_proxy(raw_before)
        score_after = _score_proxy(raw_after)
        if score_before is None or score_after is None:
            return _result(predicate, "blocked_needs_evidence", counter=["missing_score_proxy"])
        observed = {"score_before": score_before, "score_after": score_after, "delta": score_after - score_before}
        ok = score_after > score_before and not isclose(score_after, score_before, abs_tol=1e-6)
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else ["score_proxy_improved_mismatch"], observed)

    if relation in {"component_appeared", "component_disappeared", "component_count_changed", "color_count_changed"}:
        if relation == "component_appeared":
            err = _need_subjects(predicate, subjects, 1)
            ok = not err and subjects[0] not in before and subjects[0] in after
            observed = {"subject": subjects[0] if subjects else None, "before_present": subjects and subjects[0] in before, "after_present": subjects and subjects[0] in after}
        elif relation == "component_disappeared":
            err = _need_subjects(predicate, subjects, 1)
            ok = not err and subjects[0] in before and subjects[0] not in after
            observed = {"subject": subjects[0] if subjects else None, "before_present": subjects and subjects[0] in before, "after_present": subjects and subjects[0] in after}
        elif relation == "component_count_changed":
            err = None
            ok = len(before) != len(after)
            observed = {"count_before": len(before), "count_after": len(after)}
        else:
            err = None
            color = predicate.get("value")
            cb = Counter(str(c.get("color")) for c in before.values())
            ca = Counter(str(c.get("color")) for c in after.values())
            if color is None:
                ok = cb != ca
                observed = {"colors_before": dict(cb), "colors_after": dict(ca)}
            else:
                key = str(color)
                ok = cb.get(key, 0) != ca.get(key, 0)
                observed = {"color": color, "count_before": cb.get(key, 0), "count_after": ca.get(key, 0)}
        if err:
            return _result(predicate, "blocked_needs_evidence", counter=[err])
        return _result(predicate, "supported" if ok else "rejected", [relation] if ok else [], [] if ok else [f"{relation}_mismatch"], observed)

    return _result(predicate, "blocked_needs_evidence", counter=[f"unsupported_pair_relation:{relation}"])


def _verify_at(predicate: dict[str, Any], frames: list[dict[str, Any]], subjects: list[str]) -> dict[str, Any]:
    relation = predicate.get("relation")
    entries, err = _at_frames(frames, predicate.get("time") or {})
    if err:
        return _result(predicate, "blocked_needs_evidence", counter=[err])

    if relation in {"level_completed", "reset_or_failure"}:
        raw_entries, raw_err = _raw_at_frames(frames, predicate.get("time") or {})
        if raw_err:
            return _result(predicate, "blocked_needs_evidence", counter=[raw_err])
        checks: list[tuple[bool, dict[str, Any], str | None]] = []
        for idx, raw_frame in raw_entries:
            if relation == "level_completed":
                ok = _frame_flag(raw_frame, "level_completed", "completed", "success") or raw_frame.get("outcome") == "level_completed"
            else:
                ok = _frame_flag(raw_frame, "reset", "failure", "failed", "terminal_failure") or raw_frame.get("outcome") in {"reset", "failure", "failed"}
            observed = {
                "frame": idx,
                "level_completed": bool(raw_frame.get("level_completed")),
                "reset": bool(raw_frame.get("reset")),
                "failure": bool(raw_frame.get("failure")),
                "outcome": raw_frame.get("outcome"),
            }
            checks.append((ok, observed, None if ok else f"{relation}_mismatch_at:{idx}"))
        counters = [c[2] for c in checks if c[2]]
        observed = {"checks": [c[1] for c in checks]}
        if counters:
            return _result(predicate, "rejected", counter=[str(c) for c in counters], observed=observed)
        return _result(predicate, "supported", support=[str(relation)], observed=observed)

    checks: list[tuple[bool, dict[str, Any], str | None]] = []
    for idx, frame in entries:
        if relation in {"overlap", "no_overlap", "inside_bbox", "outside_bbox", "target_contact", "hazard_contact"}:
            err = _need_subjects(predicate, subjects, 2) or _need_present(frame, subjects[:2])
            if err:
                checks.append((False, {"frame": idx}, err))
                continue
            if relation in {"overlap", "no_overlap", "target_contact", "hazard_contact"}:
                base = _overlap(frame[subjects[0]], frame[subjects[1]])
                if relation in {"overlap", "target_contact", "hazard_contact"}:
                    ok = base
                else:
                    ok = not base
            else:
                base = _inside(frame[subjects[0]], frame[subjects[1]])
                ok = base if relation == "inside_bbox" else not base
            checks.append((ok, {"frame": idx, "subjects": subjects[:2], "overlap": base}, None if ok else f"{relation}_mismatch_at:{idx}"))
        elif relation == "midpoint_matches":
            err = _need_subjects(predicate, subjects, 3) or _need_present(frame, subjects[:3])
            if err:
                checks.append((False, {"frame": idx}, err))
                continue
            ax, ay = _centroid(frame[subjects[0]])
            bx, by = _centroid(frame[subjects[1]])
            cx, cy = _centroid(frame[subjects[2]])
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            ok = isclose(mx, cx, abs_tol=0.51) and isclose(my, cy, abs_tol=0.51)
            checks.append((ok, {"frame": idx, "midpoint": [mx, my], "body": [cx, cy]}, None if ok else f"midpoint_matches_mismatch_at:{idx}"))
        else:
            return _result(predicate, "blocked_needs_evidence", counter=[f"unsupported_at_relation:{relation}"])

    observed = {"checks": [c[1] for c in checks]}
    counters = [c[2] for c in checks if c[2]]
    if counters:
        return _result(predicate, "rejected", counter=[str(c) for c in counters], observed=observed)
    return _result(predicate, "supported", support=[relation], observed=observed)


def verify_predicate(
    predicate: dict[str, Any],
    trace_pack: dict[str, Any],
    *,
    entity_bindings: Any = None,
) -> dict[str, Any]:
    """Verify one generic observable predicate against a visual trace pack."""

    if not isinstance(predicate, dict):
        return _result({}, "blocked_needs_evidence", counter=["predicate_not_object"])
    relation = predicate.get("relation")
    frames = list(trace_pack.get("frames") or []) if isinstance(trace_pack, dict) else []
    counter: list[str] = []
    if relation not in SUPPORTED_RELATIONS:
        counter.append(f"unsupported_relation:{relation}")
    if not frames:
        counter.append("missing_frames")
    if counter:
        return _result(predicate, "blocked_needs_evidence", counter=counter)

    subjects = _resolve_subjects(predicate.get("subjects") or [], entity_bindings)
    time = predicate.get("time") or {}
    if isinstance(time, dict) and ("from" in time or "to" in time):
        return _verify_pair(predicate, frames, subjects)
    if isinstance(time, dict) and "at" in time:
        return _verify_at(predicate, frames, subjects)
    return _result(predicate, "blocked_needs_evidence", counter=["missing_supported_time_address"])


def verify_predicates(
    predicates: list[dict[str, Any]],
    trace_pack: dict[str, Any],
    *,
    entity_bindings: Any = None,
) -> list[dict[str, Any]]:
    """Verify a batch of generic observable predicates."""

    return [verify_predicate(p, trace_pack, entity_bindings=entity_bindings) for p in predicates]
