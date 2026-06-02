"""Prompt-pack builder for visual rule-hypothesis VLM/LLM sidecars.

The prompt produced here is evidence-only: it asks a model to propose rich,
falsifiable rule hypotheses from visual trace artifacts. It explicitly forbids
live action authority and direct policy ranking. Local verifier code must inspect
and test any proposal before it can influence an eval-only play prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PROMPT = "mini-palari.visual-rule-prompt.v0.1"
SCHEMA_HYPOTHESIS = "mini-palari.visual-rule-hypothesis.v0.1"
FORBIDDEN_AUTHORITY = [
    "direct_live_action",
    "policy_rank_without_verification",
    "memory_write",
    "network_or_download",
    "runtime_authority",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _component_summary(frame: dict[str, Any], *, max_components: int = 24) -> list[dict[str, Any]]:
    components = list(frame.get("components") or [])[:max_components]
    summary: list[dict[str, Any]] = []
    for component in components:
        summary.append(
            {
                "id": component.get("id"),
                "color": component.get("color"),
                "area": component.get("area"),
                "bbox": component.get("bbox"),
                "centroid": component.get("centroid"),
            }
        )
    return summary


def _trace_summary(trace_pack: dict[str, Any]) -> dict[str, Any]:
    frames = []
    for frame in trace_pack.get("frames") or []:
        frames.append(
            {
                "index": frame.get("index"),
                "image_path": str(frame.get("image_path", "")),
                "shape": frame.get("shape"),
                "components": _component_summary(frame),
            }
        )
    return {
        "trace_schema": trace_pack.get("schema"),
        "title": trace_pack.get("title"),
        "game_id": trace_pack.get("game_id"),
        "available_actions": list(trace_pack.get("available_actions") or []),
        "action_history": list(trace_pack.get("actions") or []),
        "frames": frames,
        "deltas": list(trace_pack.get("deltas") or []),
        "trace_authority": trace_pack.get("authority"),
    }


def _hypothesis_schema_template() -> dict[str, Any]:
    return {
        "schema": SCHEMA_HYPOTHESIS,
        "hypothesis_id": "h1",
        "game_family": "mirrored_dual_avatar_navigation | spider_endpoint_geometry | sokoban | toggle | paint | matching | unknown",
        "entities": [
            {
                "entity_id": "e1",
                "component_refs": ["c0"],
                "role": "avatar | mirror_avatar | hazard | goal | endpoint | body | wall | movable | unknown",
                "visual_evidence": "Refer to frame/image/component/delta evidence.",
            }
        ],
        "rules": [
            {
                "rule_id": "r1",
                "type": "model_label_only_not_executable",
                "statement": "A falsifiable rule label that could explain the trace; rule.type is a label, not executable verifier code.",
                "predictions": [
                    {
                        "predicate_id": "p1",
                        "relation": "opposite_delta_x",
                        "subjects": ["left_square", "right_square"],
                        "time": {"from": 0, "to": 1},
                    }
                ],
            }
        ],
        "experiments": [
            {
                "experiment_id": "x1",
                "purpose": "Distinguish this hypothesis from alternatives.",
                "actions": [1, 2],
                "predictions": [
                    {
                        "predicate_id": "x1p1",
                        "relation": "moved",
                        "subjects": ["avatar"],
                        "time": {"from": 0, "to": 1},
                    },
                    {
                        "predicate_id": "x1p2",
                        "relation": "no_overlap",
                        "subjects": ["avatar", "hazard"],
                        "time": {"at": "all"},
                    },
                    {
                        "predicate_id": "x1p3",
                        "relation": "progress_toward_static_target",
                        "subjects": ["avatar", "goal"],
                        "time": {"from": 0, "to": 1},
                    },
                ],
                "falsifiers": [
                    {
                        "predicate_id": "x1f1",
                        "relation": "overlap",
                        "subjects": ["avatar", "hazard"],
                        "time": {"at": "all"},
                    }
                ],
            }
        ],
        "plan_if_true": [
            {
                "step": 1,
                "actions": [1, 2],
                "intent": "Bounded plan prefix under this hypothesis.",
                "abort_if": ["Any mismatch with predicted observations or unsafe hazard contact."],
            }
        ],
        "authority": "proposal_only",
    }


def _examples() -> list[dict[str, Any]]:
    return [
        {
            "name": "mirrored_dual_avatar_navigation",
            "what_to_notice": [
                "two similarly shaped components may be coupled avatars",
                "one component can move on the inverse x-axis when the other moves",
                "walls/obstacles may let one avatar be pinned while the other continues",
                "hazard/lava components may invalidate long plans even if immediate motion looks good",
            ],
            "rule_types": ["mirror_motion", "collision_blocks", "lava_forbidden"],
            "good_experiment_shape": "move toward wall, observe both avatar tracks, then test whether a pinned avatar decouples motion",
        },
        {
            "name": "spider_endpoint_geometry",
            "what_to_notice": [
                "two endpoint/leg components define a body or midpoint",
                "click/select may choose one endpoint, and the next click may move it",
                "objective may depend on the midpoint/body entering a target area",
                "different colors may represent separate spiders and target zones",
            ],
            "rule_types": ["selected_endpoint_move", "midpoint_body", "color_goal_binding"],
            "good_experiment_shape": "select one endpoint, click a target cell, verify only that endpoint moves and the midpoint relation changes as predicted",
        },
    ]


def build_visual_rule_prompt(trace_pack: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic prompt pack for visual rule-hypothesis proposal.

    Returns a JSON-serializable dict containing the prompt text plus the key
    context fields a sidecar adapter should log. The model is asked for rule
    hypotheses and experiments only, not direct live actions.
    """

    trace = _trace_summary(trace_pack)
    prompt_sections = [
        "You are a visual rule-discovery sidecar for ARC-AGI-3 style games.",
        "Your job is to infer possible game rules from visual trace artifacts and propose falsifiable experiments.",
        "Do not output direct live actions for immediate execution. Do not rank the runtime policy. You have no runtime authority.",
        "Every hypothesis must be proposal_only and must include experiments, generic predicate predictions, and falsifiers for local verification.",
        "Prefer long-horizon, compositional rule hypotheses when the evidence supports them; these games can involve mirrored controls, hazards, object pinning, selected endpoints, midpoint bodies, delayed objectives, and multi-object/color bindings.",
        "Generic observable predicates are the verifier contract: rule.type is a label, not Python checker authority. Use predicate_id, relation, subjects, and time objects such as {'from': 0, 'to': 1} or {'at': 'all'}.",
        "Useful generic relations include moved, stayed, delta_x_sign, delta_y_sign, same_delta_x, opposite_delta_x, distance_decreased, overlap, no_overlap, inside_bbox, outside_bbox, midpoint_matches, component_appeared, component_disappeared, component_count_changed, and color_count_changed.",
        "Also propose progress-like predicates when the trace supports them: level_completed, reset_or_failure, target_contact, hazard_contact, object_entered_goal_region, object_left_goal_region, progress_toward_static_target, and score_proxy_improved. These remain evidence predicates only, not score or runtime claims.",
        "",
        "Forbidden authority:",
        _stable_json(FORBIDDEN_AUTHORITY),
        "",
        "Trace summary and visual artifact references:",
        _stable_json(trace),
        "",
        "Required output: JSON object with a top-level hypotheses array. Each hypothesis must follow this schema:",
        _stable_json(_hypothesis_schema_template()),
        "",
        "Examples of the kinds of rules to consider, not fixed answers:",
        _stable_json(_examples()),
        "",
        "Return 5-10 diverse hypotheses when possible. Use component IDs and image paths from the trace. If evidence is insufficient, say what experiment would distinguish alternatives. Keep authority='proposal_only'.",
    ]
    prompt_text = "\n".join(prompt_sections)
    return {
        "schema": SCHEMA_PROMPT,
        "game_id": trace.get("game_id"),
        "available_actions": trace.get("available_actions") or [],
        "action_history": trace.get("action_history") or [],
        "image_paths": [str(frame.get("image_path", "")) for frame in trace.get("frames", [])],
        "hypothesis_schema": SCHEMA_HYPOTHESIS,
        "forbidden_authority": list(FORBIDDEN_AUTHORITY),
        "runtime_authority_granted": False,
        "prompt_text": prompt_text,
    }


def write_visual_rule_prompt(trace_pack: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Build and write a prompt pack JSON file."""

    prompt = build_visual_rule_prompt(trace_pack)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(prompt) + "\n", encoding="utf-8")
    prompt["prompt_path"] = str(path)
    return prompt
