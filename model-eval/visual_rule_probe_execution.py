"""Execute/bind visual-rule probe plans against bounded local evidence.

This is an eval-only harness. It can bind target intents to visible connected
components, replay bounded probe action sequences against already-recorded local
public traces, verify generic observable predicates, and compare short/current/
long hypothesis variants. It never grants runtime or action authority.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "mini-palari.visual-rule-probe-execution.v0.1"
_SIMPLE_BG_COLORS = {0, 1, 5, 63}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _action_id(action: Any) -> int | None:
    text = str(action or "")
    if text == "RESET":
        return 0
    if text.startswith("ACTION"):
        try:
            return int(text.replace("ACTION", ""))
        except ValueError:
            return None
    try:
        return int(text)
    except Exception:
        return None


def _grid_from_event(event: dict[str, Any], *, after: bool = False) -> list[list[int]]:
    grid = event.get("next_grid") if after else event.get("grid")
    if grid is None:
        grid = event.get("grid") or event.get("next_grid") or []
    return [[int(v) for v in row] for row in grid]


def _changed_cells(before: list[list[int]], after: list[list[int]]) -> int:
    h = min(len(before), len(after))
    if h == 0:
        return 0
    count = 0
    for y in range(h):
        w = min(len(before[y]), len(after[y]))
        for x in range(w):
            if before[y][x] != after[y][x]:
                count += 1
    return count


def connected_components(grid: list[list[int]]) -> list[dict[str, Any]]:
    if not grid:
        return []
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    counts = Counter(int(v) for row in grid for v in row)
    dominant_color = counts.most_common(1)[0][0] if counts else None
    seen: set[tuple[int, int]] = set()
    comps: list[dict[str, Any]] = []
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if (x, y) in seen or int(color) in _SIMPLE_BG_COLORS or (dominant_color is not None and int(color) == dominant_color and counts[int(color)] > 16):
                continue
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= len(grid[ny]):
                        continue
                    if (nx, ny) in seen or int(grid[ny][nx]) != int(color):
                        continue
                    seen.add((nx, ny))
                    q.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            comps.append(
                {
                    "component_id": f"c{int(color)}_{len(comps)}",
                    "color": int(color),
                    "area": len(cells),
                    "bbox": [min_x, min_y, max_x, max_y],
                    "coordinate": {"x": cx, "y": cy},
                    "grid_size": {"width": width, "height": height},
                }
            )
    comps.sort(key=lambda c: (-int(c["area"]), int(c["bbox"][1]), int(c["bbox"][0])))
    return comps


def _rank_components_for_intent(components: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    text = intent.lower()
    if not components:
        return []
    width = max(int(c.get("grid_size", {}).get("width") or 0) for c in components)
    height = max(int(c.get("grid_size", {}).get("height") or 0) for c in components)

    def score(c: dict[str, Any]) -> tuple[float, float]:
        x = float(c["coordinate"]["x"])
        y = float(c["coordinate"]["y"])
        area = float(c["area"])
        s = 0.0
        if "bottom" in text or "swatch" in text:
            s += y / max(1, height)
        if "top" in text or "bar" in text or "slot" in text:
            s += 1.0 - (y / max(1, height))
        if "target" in text or "ring" in text or "center" in text:
            s += 1.0 - abs(x - width / 2) / max(1, width) * 0.25
        if "endpoint" in text:
            s += (abs(x - width / 2) / max(1, width)) * 0.5
        if "midpoint" in text:
            s += 1.0 - abs(x - width / 2) / max(1, width)
        # Prefer compact/salient foreground components over huge regions.
        s += min(area, 64.0) / 256.0
        return (-s, -area)

    ranked = sorted(components, key=score)
    if ".centers" in text or "swatches" in text or "slots" in text or "endpoints" in text:
        return ranked[: min(4, len(ranked))]
    return ranked[:1]


def bind_target_intents(grid: list[list[int]], target_intents: list[str]) -> dict[str, list[dict[str, Any]]]:
    components = connected_components(grid)
    bindings: dict[str, list[dict[str, Any]]] = {}
    for intent in target_intents:
        bindings[str(intent)] = _rank_components_for_intent(components, str(intent))
    return bindings


def _find_matching_window(action_sequence: list[int], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not action_sequence:
        return []
    target = [int(a) for a in action_sequence]
    ids = [_action_id(e.get("action")) for e in events]
    for start in range(0, max(0, len(ids) - len(target) + 1)):
        if ids[start : start + len(target)] == target:
            return events[start : start + len(target)]
    # Fall back to ordered subsequence so real logs can still provide bounded evidence.
    out: list[dict[str, Any]] = []
    pos = 0
    for event, aid in zip(events, ids):
        if pos < len(target) and aid == target[pos]:
            out.append(event)
            pos += 1
    return out if len(out) == len(target) else []


def _event_completed(event: dict[str, Any]) -> bool:
    state = str(event.get("state") or "")
    return bool(event.get("level_completed")) or "WIN" in state or "COMPLETED" in state or int(event.get("levels") or 0) > 0


def _event_failed(event: dict[str, Any]) -> bool:
    state = str(event.get("state") or "")
    return bool(event.get("failure") or event.get("reset")) or "FAILED" in state or "GAME_OVER" in state


def _verify_relation(relation: str, window: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [int(e.get("changed") if e.get("changed") is not None else _changed_cells(_grid_from_event(e), _grid_from_event(e, after=True))) for e in window]
    total_changed = sum(changed)
    any_changed = any(c > 0 for c in changed)
    any_completed = any(_event_completed(e) for e in window)
    any_failed = any(_event_failed(e) for e in window)
    low = relation.lower()
    observed = {"changed_cells_by_step": changed, "total_changed_cells": total_changed, "level_completed": any_completed, "reset_or_failure": any_failed}

    if low in {"level_completed", "score_proxy_improved"}:
        ok = any_completed
        return {"relation": relation, "decision": "supported" if ok else "rejected", "observed": observed, "runtime_authority_granted": False}
    if low in {"reset_or_failure"} or "reset" in low or "failure" in low:
        ok = any_failed
        return {"relation": relation, "decision": "supported" if ok else "rejected", "observed": observed, "runtime_authority_granted": False}
    if low == "moved":
        return {"relation": relation, "decision": "supported" if any_changed else "rejected", "observed": observed, "runtime_authority_granted": False}
    if low == "stayed":
        return {"relation": relation, "decision": "supported" if not any_changed else "rejected", "observed": observed, "runtime_authority_granted": False}
    if any(token in low for token in ("changed", "appeared", "filled", "fill", "selected", "marker", "index", "transfer", "contact", "path", "bridge", "open", "closed", "connection", "socket", "color")):
        ok = any_changed
        return {"relation": relation, "decision": "supported" if ok else "rejected", "observed": observed, "runtime_authority_granted": False}
    return {"relation": relation, "decision": "blocked_unsupported_relation", "observed": observed, "runtime_authority_granted": False}


def _statement_expectation_results(statement: str, window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = statement.lower()
    relations: list[str] = []
    if any(word in text for word in ("progress", "approach", "contact", "agreement", "match")):
        relations.append("statement_progress_or_contact_visible_delta")
    if any(word in text for word in ("reset", "failure", "abort", "rejection", "not progress", "mere")):
        relations.append("statement_reset_failure_or_false_progress_guard")
    if any(word in text for word in ("select", "selector", "click", "apply", "move", "route", "path")):
        relations.append("statement_action_has_visible_effect")
    return [_verify_relation(r, window) for r in relations]


def replay_probe_on_events(probe: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    seq = [int(a) for a in probe.get("action_sequence") or []]
    window = _find_matching_window(seq, events)
    first_grid = _grid_from_event(events[0]) if events else []
    bindings = bind_target_intents(first_grid, [str(i) for i in probe.get("target_intents") or []])
    if not window:
        return {
            "experiment_id": probe.get("experiment_id"),
            "execution_status": "blocked_no_matching_logged_probe",
            "action_sequence": seq,
            "bound_target_intents": bindings,
            "predicate_results": [],
            "visible_delta_steps": 0,
            "level_progress_steps": 0,
            "survivor_status": "blocked",
            "runtime_authority_granted": False,
        }
    predicate_results = [_verify_relation(str(p.get("relation") or p), window) for p in probe.get("predicates_to_verify") or []]
    predicate_results.extend(_statement_expectation_results(str(probe.get("hypothesis_statement") or ""), window))
    changed = [int(e.get("changed") if e.get("changed") is not None else _changed_cells(_grid_from_event(e), _grid_from_event(e, after=True))) for e in window]
    supported = sum(1 for r in predicate_results if r.get("decision") == "supported")
    rejected = sum(1 for r in predicate_results if r.get("decision") == "rejected")
    status = "accepted" if supported and not rejected else "accepted_partial" if supported else "rejected" if rejected else "blocked"
    return {
        "experiment_id": probe.get("experiment_id"),
        "execution_status": "replayed",
        "action_sequence": seq,
        "logged_steps": [e.get("step") for e in window],
        "logged_actions": [e.get("action") for e in window],
        "bound_target_intents": bindings,
        "predicate_results": predicate_results,
        "visible_delta_steps": sum(1 for c in changed if c > 0),
        "level_progress_steps": sum(1 for e in window if _event_completed(e)),
        "survivor_status": status,
        "runtime_authority_granted": False,
    }


def execute_probe_plan_on_logged_events(probe_plan: dict[str, Any], event_dir: str | Path) -> dict[str, Any]:
    event_root = Path(event_dir)
    games_out: list[dict[str, Any]] = []
    for game in probe_plan.get("games") or []:
        game_id = str(game.get("game_id"))
        path = event_root / f"{game_id}.json"
        if not path.exists():
            games_out.append({"game_id": game_id, "status": "blocked_missing_event_log", "probe_results": [], "runtime_authority_granted": False})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = list(payload.get("events") or [])
        probe_results = [replay_probe_on_events(probe, events) for probe in game.get("probes") or []]
        games_out.append({"game_id": game_id, "status": "evaluated", "event_log": str(path), "probe_results": probe_results, "runtime_authority_granted": False})
    return _summarize_execution(games_out, source_plan=probe_plan.get("report_path") or probe_plan.get("source_proposal_path"))


def _summarize_execution(games_out: list[dict[str, Any]], *, source_plan: Any = None) -> dict[str, Any]:
    probes = [p for g in games_out for p in g.get("probe_results", [])]
    predicates = [r for p in probes for r in p.get("predicate_results", [])]
    summary = {
        "game_count": len(games_out),
        "probe_count": len(probes),
        "replayed_probe_count": sum(1 for p in probes if p.get("execution_status") == "replayed"),
        "bound_probe_count": sum(1 for p in probes if any(p.get("bound_target_intents", {}).values())),
        "accepted_survivors": sum(1 for p in probes if p.get("survivor_status") in {"accepted", "accepted_partial"}),
        "rejected_survivors": sum(1 for p in probes if p.get("survivor_status") == "rejected"),
        "blocked_probes": sum(1 for p in probes if p.get("survivor_status") == "blocked"),
        "predicate_supported": sum(1 for r in predicates if r.get("decision") == "supported"),
        "predicate_rejected": sum(1 for r in predicates if r.get("decision") == "rejected"),
        "predicate_blocked": sum(1 for r in predicates if str(r.get("decision", "")).startswith("blocked")),
        "level_progress_steps": sum(int(p.get("level_progress_steps") or 0) for p in probes),
    }
    return {
        "schema": SCHEMA,
        "created_at": _now_iso(),
        "source_plan": source_plan,
        "decision": "completed_logged_replay",
        "summary": summary,
        "games": games_out,
        "claims": {"generalization": "not_claimed", "performance": "not_claimed", "scope": "logged_trace_probe_replay_only"},
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
    }


def compare_variant_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    best_name = None
    best_score = -1.0
    for name, report in reports.items():
        summary = dict(report.get("summary") or {})
        score = (
            float(summary.get("accepted_survivors") or 0) * 4.0
            + float(summary.get("predicate_supported") or 0) * 1.0
            + float(summary.get("bound_probe_count") or 0) * 0.5
            + float(summary.get("level_progress_steps") or 0) * 6.0
            - float(summary.get("predicate_rejected") or 0) * 0.25
        )
        variants[name] = {"summary": summary, "evidence_depth_score": score}
        if score > best_score:
            best_score = score
            best_name = name
    return {
        "schema": "mini-palari.visual-rule-probe-execution-comparison.v0.1",
        "created_at": _now_iso(),
        "best_by_evidence_depth": best_name,
        "variants": variants,
        "claims": {"generalization": "not_claimed", "performance": "not_claimed", "scope": "logged_trace_proxy_comparison_only"},
        "runtime_authority_granted": False,
    }


def run_length_sweep_execution(length_sweep_dir: str | Path, event_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    sweep = Path(length_sweep_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for variant in ("short", "current", "long"):
        plan_path = sweep / variant / "probe-plan" / "visual_rule_probe_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        report = execute_probe_plan_on_logged_events(plan, event_dir)
        report["variant"] = variant
        path = out / f"{variant}_probe_execution_report.json"
        report["report_path"] = str(path)
        path.write_text(_stable_json(report) + "\n", encoding="utf-8")
        reports[variant] = report
    comparison = compare_variant_reports(reports)
    comparison["variant_report_paths"] = {name: report.get("report_path") for name, report in reports.items()}
    path = out / "probe_execution_comparison.json"
    comparison["report_path"] = str(path)
    path.write_text(_stable_json(comparison) + "\n", encoding="utf-8")
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-sweep-dir", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    comparison = run_length_sweep_execution(args.length_sweep_dir, args.event_dir, args.output_dir)
    print(_stable_json({"report_path": comparison["report_path"], "best_by_evidence_depth": comparison["best_by_evidence_depth"], "variants": comparison["variants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
