"""Experimental proposal-only GPT-5.5 ARC-AGI-3 runner.

Level 2: the model proposes visual-rule hypotheses/probes; local code validates,
aggregates, and executes one bounded legal action. This is not submission-safe and
must not be copied into agent/my_agent.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))
MINI_PALARI_SRC = ROOT.parent / "mini-palari" / "src"
sys.path.insert(0, str(MINI_PALARI_SRC))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from agents.agent import Agent

from scripts.play_llm_live import downsample_ascii, grid_summary, latest_grid
from mini_palari.primitive_binding import detect_canonical_primitive_inventory_from_observations

ACTIONS_BY_NAME = {a.name: a for a in GameAction}


def _small_components_for_hints(grid: list[list[int]]) -> list[dict[str, Any]]:
    """Return compact same-color components for objective-candidate hints."""
    if not grid or not grid[0]:
        return []
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            color = int(grid[y][x])
            stack = [(x, y)]
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and (nx, ny) not in seen
                        and int(grid[ny][nx]) == color
                    ):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            neighbor_colors = sorted(
                {
                    int(grid[ny][nx])
                    for cx, cy in cells
                    for nx in range(max(0, cx - 1), min(width, cx + 2))
                    for ny in range(max(0, cy - 1), min(height, cy + 2))
                    if (nx, ny) not in cells
                }
            )
            components.append(
                {
                    "color": color,
                    "count": len(cells),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "cells": [[cx, cy] for cx, cy in sorted(cells)],
                    "neighbor_colors": neighbor_colors,
                }
            )
    return components


def objective_candidate_hints(*, grid: list[list[int]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Promote unique fixed markers after an obvious target/action path fails.

    This is trace/proposal-only scaffolding: it does not select actions or grant
    runtime authority. It makes the Brain workspace say the human-obvious thing:
    if a first salient/target-like interpretation produced motion but no level
    completion, promote tiny fixed marker/cross-like primitives as objective
    candidates instead of treating them as irrelevant because they do not move.
    """
    completed = [int(step.get("levels_after", step.get("levels_completed", 0)) or 0) for step in history]
    no_completion = not completed or max(completed) <= int(history[0].get("levels_before", 0) or 0)
    history_text = " ".join(
        str(step.get(key, ""))
        for step in history[-8:]
        for key in ("event_summary", "evidence_summary", "rules_broken")
    ).lower()
    explicit_failed_obvious_target = any(
        term in history_text for term in ("top target", "obvious", "terminal goal", "not end", "level completion")
    )
    moved_without_completion = any(
        ((step.get("primitive_change") or {}).get("delta_label_counts") or {}).get("primitive_moved", 0)
        for step in history[-8:]
    )
    failed_obvious_target = no_completion and (explicit_failed_obvious_target or (len(history) >= 3 and moved_without_completion))
    if not failed_obvious_target:
        return []

    hints: list[dict[str, Any]] = []
    for component in _small_components_for_hints(grid):
        color = component["color"]
        if color in (3, 4, 5, 8, 9, 11, 12):
            continue
        if component["count"] > 4:
            continue
        neighbor_colors = component.get("neighbor_colors", [])
        cross_like = 0 in neighbor_colors or component["count"] <= 2
        if not cross_like:
            continue
        hints.append(
            {
                "schema": "mini-palari.arc3.objective-candidate-hint.v0.1",
                "candidate_id": f"color-{color}-fixed-marker-{component['bbox']}",
                "kind": "unique_fixed_cross_or_marker_candidate",
                "component": {
                    "color": color,
                    "label": f"color-{color}",
                    "bbox": component["bbox"],
                    "count": component["count"],
                    "neighbor_colors": neighbor_colors,
                },
                "promotion_reason": "promoted_after_failed_obvious_target",
                "hypothesis": "fixed strange marker/cross may be the remaining objective anchor, socket, reference, or visit/contact target",
                "suggested_probe_intent": "plan toward contact/adjacency/alignment with this marker after the initial target hypothesis failed",
                "authority": "proposal_only_hint_not_action_authority",
            }
        )
    return hints[:4]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in model output: {text[:500]}")
    return json.loads(match.group(1) if match.lastindex else match.group(0))


def _component_masks(grid: list[list[int]], *, min_area: int = 5) -> list[dict[str, Any]]:
    """Return nonzero same-color component masks for structured event matching."""
    if not grid or not grid[0]:
        return []
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            color = int(grid[y][x])
            if color == 0:
                seen.add((x, y))
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells: set[tuple[int, int]] = set()
            while stack:
                cx, cy = stack.pop()
                cells.add((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and (nx, ny) not in seen
                        and int(grid[ny][nx]) == color
                    ):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            if len(cells) < min_area:
                continue
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
            norm = frozenset((cx - min_x, cy - min_y) for cx, cy in cells)
            components.append(
                {
                    "component_id": f"color-{color}-component-{len(components)}",
                    "color": color,
                    "area": len(cells),
                    "bbox": [min_x, min_y, max_x, max_y],
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "norm_cells": norm,
                }
            )
    return components


def _jaccard(a: set[tuple[int, int]] | frozenset[tuple[int, int]], b: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> float:
    union = len(set(a) | set(b))
    if union == 0:
        return 1.0
    return len(set(a) & set(b)) / union


def _rotate_cells(cells: frozenset[tuple[int, int]], width: int, height: int, transform: str) -> tuple[frozenset[tuple[int, int]], int, int]:
    if transform == "rotate90":
        out = frozenset((height - 1 - y, x) for x, y in cells)
        return out, height, width
    if transform == "rotate180":
        out = frozenset((width - 1 - x, height - 1 - y) for x, y in cells)
        return out, width, height
    if transform == "rotate270":
        out = frozenset((y, width - 1 - x) for x, y in cells)
        return out, height, width
    raise ValueError(transform)


def structured_visual_event_summary(
    *, game_id: str, step: int, action: str, before_grid: list[list[int]], after_grid: list[list[int]]
) -> dict[str, Any]:
    """Detect high-information visual transformations without granting action authority.

    This is deliberately generic and conservative: a rotation candidate is emitted
    only when the same-color component area is stable, transformed mask similarity
    is substantially better than unchanged similarity, and the event is not a
    tiny-marker/occlusion artifact. It is evidence for hypotheses, not progress.
    """
    events: list[dict[str, Any]] = []
    before_components = _component_masks(before_grid)
    # Keep small after-components so an occluded/partially erased large object is
    # reported as blocked evidence instead of disappearing from the audit trail.
    after_components = _component_masks(after_grid, min_area=1)
    for before in before_components:
        same_color = [comp for comp in after_components if comp["color"] == before["color"]]
        if not same_color:
            continue
        after = min(same_color, key=lambda comp: abs(comp["area"] - before["area"]))
        area_delta = abs(after["area"] - before["area"])
        area_ratio = area_delta / max(before["area"], 1)
        if area_ratio >= 0.2:
            events.append(
                {
                    "schema": "mini-palari.arc-live.structured-visual-event.v0.1",
                    "event": "pose_change_blocked_area_mismatch",
                    "entity_kind": "large_structured_component",
                    "action": action,
                    "component": {"color": before["color"], "before_bbox": before["bbox"], "after_bbox": after["bbox"]},
                    "support": {"before_area": before["area"], "after_area": after["area"], "area_ratio_delta": round(area_ratio, 3)},
                    "authority": "trace_only_non_authoritative",
                }
            )
            continue
        if before["area"] < 5 or before["area"] == before["width"] * before["height"]:
            continue
        unchanged_similarity = _jaccard(before["norm_cells"], after["norm_cells"])
        candidates: list[tuple[str, float]] = []
        for transform in ("rotate90", "rotate180", "rotate270"):
            rotated, rot_w, rot_h = _rotate_cells(before["norm_cells"], before["width"], before["height"], transform)
            if rot_w != after["width"] or rot_h != after["height"]:
                continue
            candidates.append((transform, _jaccard(rotated, after["norm_cells"])))
        if not candidates:
            continue
        best_transform, best_similarity = max(candidates, key=lambda item: item[1])
        if best_similarity < 0.75 or best_similarity - unchanged_similarity < 0.25:
            continue
        events.append(
            {
                "schema": "mini-palari.arc-live.structured-visual-event.v0.1",
                "event": "rotation_candidate",
                "entity_kind": "large_structured_component",
                "action": action,
                "component": {
                    "color": before["color"],
                    "before_bbox": before["bbox"],
                    "after_bbox": after["bbox"],
                    "before_area": before["area"],
                    "after_area": after["area"],
                },
                "support": {
                    "best_transform": best_transform,
                    "unchanged_similarity": round(unchanged_similarity, 3),
                    "transformed_similarity": round(best_similarity, 3),
                    "area_ratio_delta": round(area_ratio, 3),
                },
                "counter_evidence": {"could_be_translation_only": unchanged_similarity >= 0.75, "area_mismatch_blocked": False},
                "hypothesis_use": "action_semantics_or_transition_model_evidence_not_objective_progress",
                "authority": "trace_only_non_authoritative",
            }
        )
    events.sort(key=lambda event: (event["event"] != "rotation_candidate", -float(event.get("support", {}).get("transformed_similarity", 0))))
    return {
        "schema": "mini-palari.arc-live.structured-visual-events.v0.1",
        "basis": "component_mask_pose_comparison_not_raw_changed_pixel_count",
        "observation_ids": {"before": f"{game_id}:step:{step}:before", "after": f"{game_id}:step:{step}:after"},
        "action": action,
        "events": events[:8],
        "authority": "trace_only_non_authoritative",
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
    }


def primitive_change_summary(*, game_id: str, step: int, before_grid: list[list[int]], after_grid: list[list[int]]) -> dict[str, Any]:
    """Summarize action results in Mini-Palari primitive terms, not raw changed pixels."""
    before_id = f"{game_id}:step:{step}:before"
    after_id = f"{game_id}:step:{step}:after"
    inventory = detect_canonical_primitive_inventory_from_observations(
        [
            {"game_id": before_id, "frame": before_grid},
            {"game_id": after_id, "frame": after_grid},
        ],
        delta_observation_pairs=[
            {"before_observation_id": before_id, "after_observation_id": after_id}
        ],
    )
    records = [
        record
        for record in inventory.get("primitive_delta_records", [])
        if isinstance(record, dict)
    ]
    observation_inventories = {
        str(obs.get("observation_id")): obs
        for obs in inventory.get("observation_inventories", [])
        if isinstance(obs, dict)
    }
    label_counts = dict(inventory.get("delta_label_counts") or {})
    primitive_samples = []
    for record in records[:12]:
        primitive_samples.append(
            {
                "delta_label": record.get("delta_label"),
                "before_primitive_id": record.get("before_primitive_id"),
                "after_primitive_id": record.get("after_primitive_id"),
                "dx": record.get("dx"),
                "dy": record.get("dy"),
                "color_before": record.get("color_before"),
                "color_after": record.get("color_after"),
                "match_score": record.get("match_score"),
                "match_blockers": record.get("match_blockers", []),
            }
        )
    return {
        "schema": "mini-palari.arc-live.primitive-change-summary.v0.1",
        "basis": "primitive_delta_records_not_raw_changed_pixel_count",
        "observation_ids": {"before": before_id, "after": after_id},
        "before_primitive_count": observation_inventories.get(before_id, {}).get("primitive_count"),
        "after_primitive_count": observation_inventories.get(after_id, {}).get("primitive_count"),
        "primitive_delta_record_count": len(records),
        "delta_label_counts": label_counts,
        "primitive_delta_samples": primitive_samples,
        "authority": "trace_only_non_authoritative",
    }


class ProposalOnlyController:
    def __init__(self, *, model: str, outdir: Path, timeout: int = 120) -> None:
        self.model = model
        self.outdir = outdir
        self.timeout = timeout
        self.calls = 0
        self.failures = 0

    def choose(self, *, game_id: str, step: int, latest_frame: Any, history: list[dict[str, Any]], available: list[str]) -> dict[str, Any]:
        grid = latest_grid(getattr(latest_frame, "frame", None))
        payload = {
            "game_id": game_id,
            "step": step,
            "state": str(getattr(latest_frame, "state", "")),
            "levels_completed": getattr(latest_frame, "levels_completed", 0),
            "available_actions": available,
            "history_last_steps": history[-10:],
            "grid_summary": grid_summary(grid, max_points=120),
            "downsample_16x16_majority_nonzero": downsample_ascii(grid),
            "objective_candidate_hints": objective_candidate_hints(grid=grid, history=history),
            "authority": "proposal_only",
        }
        system = (
            "You are GPT-5.5 acting as a proposal-only Palari Brain participant for an ARC-AGI-3 game. "
            "Do not directly choose an action. Instead propose 3-5 diverse visual-rule hypotheses/probes. "
            "Reason about action results through primitives/components/relations from history_last_steps. "
            "Use structured_visual_events in history_last_steps as high-information action-result evidence: rotation_candidate, pose_change, topology/open-close, recolor, merge/split, or blocked/ambiguous events should update action_semantics and transition_model hypotheses. "
            "Do not over-weight tiny overlap/occlusion artifacts as objective progress; treat them as weaker than action-caused large structured pose/topology changes unless verified. "
            "Use objective_candidate_hints when present only as candidate-only objective-binding context, not as proof that a marker/cross is the objective. "
            "Do not treat raw changed-pixel counts or generic visible change as progress evidence. "
            "Each proposal must include action, intent, predicted_observation, falsifier, confidence, and evidence. "
            "Local Mini-Palari code will validate legal actions and aggregate proposals before executing anything. "
            "Return strict JSON: {\"event_summary\":str, \"evidence_summary\":str, \"rules_proposed\":[str], "
            "\"rules_broken\":[str], \"proposals\":[{\"action\":\"ACTION1\",\"intent\":\"...\","
            "\"predicted_observation\":\"...\",\"falsifier\":\"...\",\"confidence\":0.0,\"evidence\":\"...\"}]}. "
            "All proposed actions must be from available_actions. Keep authority proposal_only."
        )
        prompt = system + "\n\nBRAIN WORKSPACE JSON:\n" + json.dumps(payload, separators=(",", ":")) + "\n\nReturn only JSON."
        cmd = ["hermes", "-z", prompt, "--provider", "openai-codex", "-m", self.model, "-t", ""]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=self.timeout)
        self.calls += 1
        if proc.returncode != 0:
            self.failures += 1
            raise RuntimeError(f"Hermes model call failed {proc.returncode}: {proc.stderr[-500:] or proc.stdout[-500:]}")
        raw = proc.stdout.strip()
        data = _extract_json(raw)
        proposals = data.get("proposals") if isinstance(data.get("proposals"), list) else []
        scores: Counter[str] = Counter()
        sanitized: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            action = str(proposal.get("action", "")).upper()
            if action not in available:
                proposal = dict(proposal)
                proposal["rejected"] = "action_not_available"
                sanitized.append(proposal)
                continue
            try:
                confidence = max(0.0, min(1.0, float(proposal.get("confidence", 0.1))))
            except Exception:
                confidence = 0.1
            if not proposal.get("predicted_observation") or not proposal.get("falsifier"):
                confidence *= 0.25
                proposal = dict(proposal)
                proposal["penalty"] = "missing_prediction_or_falsifier"
            scores[action] += confidence
            proposal = dict(proposal)
            proposal["accepted_by_local_validator"] = True
            proposal["local_weight"] = confidence
            sanitized.append(proposal)
        legal_non_reset = [a for a in available if a != "RESET"] or available
        selected = scores.most_common(1)[0][0] if scores else legal_non_reset[0]
        choice = {
            "action": selected,
            "x": 0,
            "y": 0,
            "event_summary": data.get("event_summary"),
            "evidence_summary": data.get("evidence_summary"),
            "rules_proposed": data.get("rules_proposed", []),
            "rules_broken": data.get("rules_broken", []),
            "proposals": sanitized,
            "local_aggregation_scores": dict(scores),
            "authority": "proposal_only_model_local_validator_selected_action",
            "runtime_authority_granted": False,
            "policy_rank_authority_granted": False,
            "action_payload_authority_granted": False,
        }
        if selected == "ACTION6":
            # Level 2 keeps coordinate authority local/simple for now.
            choice["x"] = 32
            choice["y"] = 32
        out = self.outdir / game_id
        out.mkdir(parents=True, exist_ok=True)
        (out / f"step_{step:03d}.json").write_text(json.dumps({"request": payload, "raw": raw, "choice": choice}, indent=2), encoding="utf-8")
        return choice


class ProposalOnlyAgent(Agent):
    MAX_ACTIONS = 10

    def __init__(self, *args: Any, controller: ProposalOnlyController, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.history: list[dict[str, Any]] = []

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return str(getattr(latest_frame, "state", "")).upper().endswith("WIN")

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        raw_available = getattr(latest_frame, "available_actions", None) or list(GameAction)
        available = []
        for action in raw_available:
            if hasattr(action, "name"):
                available.append(action.name)
            elif isinstance(action, int):
                available.append(GameAction.from_id(action).name)
            else:
                available.append(str(action).split(".")[-1].upper())
        try:
            choice = self.controller.choose(game_id=self.game_id, step=self.action_counter, latest_frame=latest_frame, history=self.history, available=available)
            name = choice["action"]
            action = ACTIONS_BY_NAME[name]
            if name == "ACTION6":
                action.set_data({"x": int(choice.get("x", 32)), "y": int(choice.get("y", 32))})
            action.reasoning = {"policy": "gpt55_proposal_only_local_validator", "choice": choice}
            before_grid = latest_grid(getattr(latest_frame, "frame", None))
            self.history.append({
                "step": self.action_counter,
                "action": name,
                "levels_before": getattr(latest_frame, "levels_completed", 0),
                "event_summary": choice.get("event_summary"),
                "evidence_summary": choice.get("evidence_summary"),
                "rules_proposed": choice.get("rules_proposed", []),
                "rules_broken": choice.get("rules_broken", []),
                "local_aggregation_scores": choice.get("local_aggregation_scores", {}),
            })
            return action
        except Exception as exc:
            self.history.append({"step": self.action_counter, "error": repr(exc), "fallback": "RESET"})
            return ACTIONS_BY_NAME["RESET"]

    def append_frame(self, frame: Any) -> None:
        if self.history:
            previous_frame = self.frames[-1] if self.frames else None
            before_grid = latest_grid(getattr(previous_frame, "frame", None)) if previous_frame is not None else [[0]]
            after_grid = latest_grid(getattr(frame, "frame", None))
            self.history[-1]["levels_after"] = getattr(frame, "levels_completed", None)
            self.history[-1]["state_after"] = str(getattr(frame, "state", ""))
            self.history[-1]["primitive_change"] = primitive_change_summary(
                game_id=self.game_id,
                step=int(self.history[-1].get("step", 0)),
                before_grid=before_grid,
                after_grid=after_grid,
            )
            self.history[-1]["structured_visual_events"] = structured_visual_event_summary(
                game_id=self.game_id,
                step=int(self.history[-1].get("step", 0)),
                action=str(self.history[-1].get("action", "")),
                before_grid=before_grid,
                after_grid=after_grid,
            )
            self.history[-1]["after_summary"] = {k: v for k, v in grid_summary(after_grid, max_points=0).items() if k != "sample_nonzero_points"}
        return super().append_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--outdir", default="model-eval/artifacts/gpt55-proposal-only-level2")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    outdir = ROOT / args.outdir / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    wanted = {g.strip().split("-")[0] for g in args.game.split(",") if g.strip()}
    game_ids = [env.game_id.split("-")[0] for env in envs if env.game_id.split("-")[0] in wanted]
    controller = ProposalOnlyController(model=args.model, outdir=outdir, timeout=args.timeout)
    ProposalOnlyAgent.MAX_ACTIONS = args.max_steps
    per_game = []
    print(f"EXPERIMENT: GPT-5.5 proposal-only Level 2; games={game_ids}; max_steps={args.max_steps}; outdir={outdir}", flush=True)
    for gid in game_ids:
        print(f"=== {gid} ===", flush=True)
        env = arc.make(gid)
        agent = ProposalOnlyAgent(card_id="gpt55-proposal-only-local", game_id=gid, agent_name=f"GPT55ProposalOnly.{gid}", ROOT_URL="http://localhost", record=False, arc_env=env, tags=["gpt55-proposal-only"], controller=controller)
        agent.main()
        final = agent.frames[-1]
        rec = {"game_id": gid, "state": str(final.state), "levels_completed": final.levels_completed, "actions": agent.action_counter, "history": agent.history}
        (outdir / gid / "game_summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        per_game.append(rec)
        print(f"  -> state={final.state}, levels_completed={final.levels_completed}, actions={agent.action_counter}", flush=True)
    scorecard = arc.get_scorecard()
    score = scorecard.score if hasattr(scorecard, "score") else scorecard
    summary = {
        "experiment": "gpt55_proposal_only_level2",
        "model": args.model,
        "proposal_only": True,
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
        "max_steps": args.max_steps,
        "score": score,
        "api_calls": controller.calls,
        "api_failures": controller.failures,
        "per_game": per_game,
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({"score": score, "api_calls": controller.calls, "api_failures": controller.failures, "outdir": str(outdir), "per_game": per_game}, indent=2), flush=True)


if __name__ == "__main__":
    main()
