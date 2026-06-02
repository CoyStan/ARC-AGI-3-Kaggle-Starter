"""Population/evolution loop for visual-rule LLM sidecar hypotheses.

This is an eval-only research harness. It runs multiple independent proposal-only
sidecar samples (intended for GPT-5.5/VLM-style stochastic calls), keeps each as a
separate lineage, verifies bounded probes against logged local evidence when
available, and feeds survivor/failure evidence into later rounds. It never grants
runtime, policy-rank, or action-payload authority.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "mini-palari.visual-rule-population-loop.v0.1"
SCHEMA_PROBE_PLAN = "mini-palari.visual-rule-probe-plan.v0.1"
MODEL_FAMILY = "gpt-5.5"


def _load_module(name: str):
    module_path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sidecar = _load_module("visual_rule_sidecar")
_execution = _load_module("visual_rule_probe_execution")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(value) + "\n", encoding="utf-8")


def _base_prompt_text(prompt_pack: dict[str, Any]) -> str:
    return str(prompt_pack.get("prompt_text") or "")


def build_population_sample_prompt(
    prompt_pack: dict[str, Any],
    *,
    round_index: int,
    sample_index: int,
    sample_count: int,
    prior_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a prompt pack for one independent rich-theory lineage sample."""

    lineage_id = f"r{round_index:02d}-s{sample_index:03d}"
    population_context = {
        "model_family": MODEL_FAMILY,
        "round_index": int(round_index),
        "sample_index": int(sample_index),
        "sample_count": int(sample_count),
        "lineage_id": lineage_id,
        "sampling_mode": "independent_randomized_theory_lineage",
        "prior_feedback": prior_feedback or {"survivor_lineage_ids": [], "failed_lineages": []},
    }
    prompt = dict(prompt_pack)
    prompt["population_context"] = population_context
    prompt["runtime_authority_granted"] = False
    prompt["policy_rank_authority_granted"] = False
    prompt["action_payload_authority_granted"] = False
    prompt["model_family"] = MODEL_FAMILY
    prompt["prompt_text"] = "\n".join(
        [
            _base_prompt_text(prompt_pack),
            "",
            "Population-loop instruction for GPT-5.5:",
            "Return exactly one rich, internally coherent visual-rule theory for this sample, not a list of alternatives.",
            "Treat this as one independent randomized lineage. Make it different from other samples by emphasizing a distinct entity/control/objective/mechanic interpretation.",
            "Derive falsifiable predicates and bounded experiments that local code can verify. Keep authority='proposal_only'.",
            "If prior feedback is present, preserve survivor evidence, explicitly address failed evidence, and Do not repeat falsified claims.",
            "Population context:",
            _stable_json(population_context),
        ]
    )
    return prompt


def _hypothesis_statement(hypothesis: dict[str, Any]) -> str:
    statements: list[str] = []
    for rule in hypothesis.get("rules") or []:
        if isinstance(rule, dict) and rule.get("statement"):
            statements.append(str(rule.get("statement")))
    if statements:
        return " ".join(statements)
    return str(hypothesis.get("game_family") or hypothesis.get("hypothesis_id") or "")


def _probe_plan_for_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    game_id = str(lineage.get("game_id") or "unknown_game")
    for hypothesis in lineage.get("hypotheses") or []:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or lineage["lineage_id"])
        for experiment in hypothesis.get("experiments") or []:
            if not isinstance(experiment, dict):
                continue
            actions: list[int] = []
            for action in experiment.get("actions") or []:
                try:
                    actions.append(int(action))
                except Exception:
                    continue
            probes.append(
                {
                    "game_id": game_id,
                    "lineage_id": lineage["lineage_id"],
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_statement": _hypothesis_statement(hypothesis),
                    "experiment_id": str(experiment.get("experiment_id") or f"{hypothesis_id}-experiment"),
                    "purpose": experiment.get("purpose"),
                    "probe_kind": "lineage_bounded_logged_probe",
                    "action_sequence": actions,
                    "action_budget": len(actions),
                    "target_intents": [],
                    "requires_coordinate_binding": False,
                    "predicates_to_verify": list(experiment.get("predictions") or []),
                    "falsifiers_to_check": list(experiment.get("falsifiers") or []),
                    "verification_status": "pending_execution",
                    "runtime_authority_granted": False,
                }
            )
    return {
        "schema": SCHEMA_PROBE_PLAN,
        "created_at": _now_iso(),
        "decision": "planned_pending_logged_replay",
        "game_count": 1,
        "probe_count": len(probes),
        "games": [
            {
                "game_id": game_id,
                "probe_count": len(probes),
                "probes": probes,
                "runtime_authority_granted": False,
            }
        ],
        "claims": {"generalization": "not_claimed", "performance": "not_claimed", "scope": "population_lineage_probe_plan_only"},
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
    }


def _score_execution(execution_report: dict[str, Any] | None) -> dict[str, Any]:
    if not execution_report:
        return {"status": "pending_verification", "supported": 0, "rejected": 0, "blocked": 0, "level_progress_steps": 0}
    summary = execution_report.get("summary") or {}
    supported = int(summary.get("predicate_supported") or 0)
    rejected = int(summary.get("predicate_rejected") or 0)
    blocked = int(summary.get("blocked_probes") or 0) + int(summary.get("predicate_blocked") or 0)
    replayed = int(summary.get("replayed_probe_count") or 0)
    level_progress = int(summary.get("level_progress_steps") or 0)
    if rejected > 0:
        status = "falsified"
    elif supported > 0:
        status = "survivor"
    elif replayed == 0 or blocked > 0:
        status = "blocked"
    else:
        status = "blocked"
    return {
        "status": status,
        "supported": supported,
        "rejected": rejected,
        "blocked": blocked,
        "level_progress_steps": level_progress,
    }


def _feedback_from_round(round_lineages: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [lineage for lineage in round_lineages if lineage.get("status") == "survivor"]
    failures = [lineage for lineage in round_lineages if lineage.get("status") == "falsified"]
    return {
        "survivor_lineage_ids": [lineage["lineage_id"] for lineage in survivors],
        "survivor_hypothesis_ids": [hid for lineage in survivors for hid in lineage.get("hypothesis_ids", [])],
        "failed_lineages": [
            {
                "lineage_id": lineage["lineage_id"],
                "hypothesis_ids": lineage.get("hypothesis_ids", []),
                "verification": lineage.get("verification", {}),
                "instruction": "Do not repeat falsified claims; propose a replacement theory that explains this counter-evidence.",
            }
            for lineage in failures
        ],
        "blocked_lineage_ids": [lineage["lineage_id"] for lineage in round_lineages if lineage.get("status") == "blocked"],
        "runtime_authority_granted": False,
    }


def run_visual_rule_population_loop(
    prompt_pack: dict[str, Any],
    *,
    output_dir: str | Path,
    sample_count: int = 10,
    rounds: int = 1,
    sidecar: str = "fixture",
    fixture_payloads: list[Any] | tuple[Any, ...] | None = None,
    command: list[str] | None = None,
    allow_command: bool = False,
    event_dir: str | Path | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run multi-sample, multi-round visual-rule hypothesis discovery.

    `fixture` mode is deterministic/no-model for tests. `command` mode remains
    gated by visual_rule_sidecar. Regardless of sidecar, outputs are proposal-only
    and any performance/generalization claims are explicitly not claimed.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_count = int(sample_count)
    rounds = int(rounds)
    fixture_payloads = list(fixture_payloads or [])
    fixture_cursor = 0
    prior_feedback: dict[str, Any] | None = None
    lineages: list[dict[str, Any]] = []
    accepted_sidecar_samples = 0

    for round_index in range(rounds):
        round_lineages: list[dict[str, Any]] = []
        for sample_index in range(sample_count):
            lineage_id = f"r{round_index:02d}-s{sample_index:03d}"
            sample_dir = out / f"round_{round_index:03d}" / f"sample_{sample_index:03d}"
            sample_prompt = build_population_sample_prompt(
                prompt_pack,
                round_index=round_index,
                sample_index=sample_index,
                sample_count=sample_count,
                prior_feedback=prior_feedback,
            )
            prompt_path = sample_dir / "prompt.json"
            _write_json(prompt_path, sample_prompt)

            fixture_payload = None
            if sidecar == "fixture" and fixture_payloads:
                fixture_payload = fixture_payloads[fixture_cursor % len(fixture_payloads)]
                fixture_cursor += 1

            sidecar_result = _sidecar.run_visual_rule_sidecar(
                sample_prompt,
                output_dir=sample_dir / "sidecar",
                sidecar=sidecar,
                fixture_payload=fixture_payload,
                command=command,
                allow_command=allow_command,
                timeout_seconds=timeout_seconds,
            )
            hypotheses = list(sidecar_result.get("hypotheses") or [])[:1]
            if sidecar_result.get("validation_status", {}).get("accepted_count", 0):
                accepted_sidecar_samples += 1

            lineage = {
                "lineage_id": lineage_id,
                "parent_lineage_id": None,
                "round_index": round_index,
                "sample_index": sample_index,
                "model_family": MODEL_FAMILY,
                "status": "pending_verification",
                "game_id": prompt_pack.get("game_id"),
                "hypothesis_count": len(hypotheses),
                "hypothesis_ids": [str(h.get("hypothesis_id")) for h in hypotheses],
                "hypotheses": hypotheses,
                "prompt_path": str(prompt_path),
                "sidecar_log_path": sidecar_result.get("sidecar_log_path"),
                "raw_output_path": sidecar_result.get("raw_output_path"),
                "validation_status": sidecar_result.get("validation_status"),
                "runtime_authority_granted": False,
                "policy_rank_authority_granted": False,
                "action_payload_authority_granted": False,
            }

            probe_plan = _probe_plan_for_lineage(lineage)
            probe_plan_path = sample_dir / "probe_plan.json"
            _write_json(probe_plan_path, probe_plan)
            lineage["probe_plan_path"] = str(probe_plan_path)

            if event_dir is not None and hypotheses:
                execution_report = _execution.execute_probe_plan_on_logged_events(probe_plan, event_dir)
                execution_path = sample_dir / "execution_report.json"
                _write_json(execution_path, execution_report)
                verification = _score_execution(execution_report)
                lineage["execution_report_path"] = str(execution_path)
                lineage["verification"] = verification
                lineage["status"] = verification["status"]
            else:
                lineage["verification"] = _score_execution(None)

            lineage_path = sample_dir / "lineage.json"
            lineage["lineage_path"] = str(lineage_path)
            _write_json(lineage_path, lineage)
            round_lineages.append(lineage)
            lineages.append(lineage)

        prior_feedback = _feedback_from_round(round_lineages)
        round_summary = {
            "round_index": round_index,
            "feedback_for_next_round": prior_feedback,
            "lineage_ids": [lineage["lineage_id"] for lineage in round_lineages],
            "runtime_authority_granted": False,
        }
        _write_json(out / f"round_{round_index:03d}" / "round_summary.json", round_summary)

    summary = {
        "lineage_count": len(lineages),
        "accepted_sidecar_samples": accepted_sidecar_samples,
        "survivor_count": sum(1 for lineage in lineages if lineage.get("status") == "survivor"),
        "falsified_count": sum(1 for lineage in lineages if lineage.get("status") == "falsified"),
        "blocked_count": sum(1 for lineage in lineages if lineage.get("status") == "blocked"),
        "pending_verification_count": sum(1 for lineage in lineages if lineage.get("status") == "pending_verification"),
    }
    report = {
        "schema": SCHEMA,
        "created_at": _now_iso(),
        "model_family": MODEL_FAMILY,
        "sidecar": sidecar,
        "round_count": rounds,
        "sample_count_per_round": sample_count,
        "summary": summary,
        "lineages": lineages,
        "claims": {
            "generalization": "not_claimed",
            "performance": "not_claimed",
            "scope": "proposal_only_population_loop_logged_replay_when_event_dir_present",
        },
        "runtime_authority_granted": False,
        "policy_rank_authority_granted": False,
        "action_payload_authority_granted": False,
        "network_allowed": False,
    }
    report_path = out / "population_summary.json"
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return report
