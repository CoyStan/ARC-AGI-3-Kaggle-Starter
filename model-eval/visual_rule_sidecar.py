"""Auditable VLM/LLM sidecar adapter for visual rule hypotheses.

This module is a proposal-only research harness boundary. Fixture mode is the
normal no-network/no-model test path. Command mode is disabled unless both an
explicit function argument and an environment gate are present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_SIDECAR_RUN = "mini-palari.visual-rule-sidecar-run.v0.1"
SCHEMA_HYPOTHESIS = "mini-palari.visual-rule-hypothesis.v0.1"
COMMAND_GATE_ENV = "MINI_PALARI_ALLOW_VLM_COMMAND"
SUPPORTED_FIXTURE_RELATIONS = {"moved", "stayed", "opposite_delta_x", "no_overlap"}
FORBIDDEN_AUTHORITY_DEFAULT = [
    "direct_live_action",
    "policy_rank_without_verification",
    "memory_write",
    "network_or_download",
    "runtime_authority",
]


def _load_hypothesis_parser():
    module_path = Path(__file__).with_name("visual_rule_hypotheses.py")
    spec = importlib.util.spec_from_file_location("visual_rule_hypotheses", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hypothesis_parser = _load_hypothesis_parser()
parse_visual_rule_hypotheses = _hypothesis_parser.parse_visual_rule_hypotheses
json_dumps = _hypothesis_parser.json_dumps


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_actions(prompt_pack: dict[str, Any]) -> list[int]:
    actions: list[int] = []
    for action in prompt_pack.get("available_actions") or [1]:
        try:
            actions.append(int(action))
        except Exception:
            continue
    return actions or [1]


def default_fixture_payload(prompt_pack: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, schema-valid fixture proposal.

    The fixture is intentionally generic: it gives tests a shaped sidecar output
    without needing credentials, a model, network, or external files.
    """

    action = _safe_actions(prompt_pack)[0]
    return {
        "hypotheses": [
            {
                "schema": SCHEMA_HYPOTHESIS,
                "hypothesis_id": "fixture_h1",
                "game_family": "fixture_unknown",
                "entities": [
                    {
                        "entity_id": "avatar",
                        "component_refs": ["c0"],
                        "role": "avatar",
                        "visual_evidence": "deterministic fixture placeholder; replace with model/trace evidence in non-fixture runs",
                    }
                ],
                "rules": [
                    {
                        "rule_id": "fixture_r1",
                        "type": "fixture_label_only",
                        "statement": "Fixture hypothesis: avatar-like component may move after the first available action.",
                        "predictions": [
                            {
                                "predicate_id": "fixture_p1",
                                "relation": "moved",
                                "subjects": ["avatar"],
                                "time": {"from": 0, "to": 1},
                            }
                        ],
                    }
                ],
                "experiments": [
                    {
                        "experiment_id": "fixture_x1",
                        "purpose": "No-model adapter smoke: preserve schema and verifier contract.",
                        "actions": [action],
                        "predictions": [
                            {
                                "predicate_id": "fixture_x1p1",
                                "relation": "moved",
                                "subjects": ["avatar"],
                                "time": {"from": 0, "to": 1},
                            }
                        ],
                        "falsifiers": [
                            {
                                "predicate_id": "fixture_x1f1",
                                "relation": "stayed",
                                "subjects": ["avatar"],
                                "time": {"from": 0, "to": 1},
                            }
                        ],
                    }
                ],
                "plan_if_true": [
                    {
                        "step": 1,
                        "actions": [action],
                        "intent": "fixture proposal only; verifier must decide whether any plan can be used",
                        "abort_if": ["prediction mismatch", "unsupported evidence"],
                    }
                ],
                "authority": "proposal_only",
            }
        ]
    }


def _write_artifacts(
    *,
    output_dir: Path,
    prompt_pack: dict[str, Any],
    raw_payload: Any,
    log: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "visual_rule_sidecar_prompt.json"
    raw_output_path = output_dir / "visual_rule_sidecar_raw_output.json"
    sidecar_log_path = output_dir / "visual_rule_sidecar_log.json"

    prompt_path.write_text(_stable_json(prompt_pack) + "\n", encoding="utf-8")
    raw_output_path.write_text(_stable_json(raw_payload) + "\n", encoding="utf-8")
    sidecar_log_path.write_text(_stable_json(log) + "\n", encoding="utf-8")
    return {
        "prompt_path": str(prompt_path),
        "raw_output_path": str(raw_output_path),
        "sidecar_log_path": str(sidecar_log_path),
    }


def _base_log(prompt_pack: dict[str, Any], *, sidecar: str, command: list[str] | None, model_source: str, decision: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_SIDECAR_RUN,
        "created_at": _now_iso(),
        "game_id": prompt_pack.get("game_id"),
        "sidecar": sidecar,
        "model_source": model_source,
        "command": list(command) if command is not None else None,
        "decision": decision,
        "forbidden_authority": list(prompt_pack.get("forbidden_authority") or FORBIDDEN_AUTHORITY_DEFAULT),
        "requires_model_credentials": sidecar == "command",
        "network_allowed": False,
        "runtime_authority_granted": False,
    }


def _blocked_result(
    *,
    prompt_pack: dict[str, Any],
    output_dir: Path,
    sidecar: str,
    command: list[str] | None,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    raw_payload = {"error": reason, "hypotheses": []}
    log = _base_log(prompt_pack, sidecar=sidecar, command=command, model_source=sidecar, decision=decision)
    log["reason"] = reason
    paths = _write_artifacts(output_dir=output_dir, prompt_pack=prompt_pack, raw_payload=raw_payload, log=log)
    return {
        **log,
        **paths,
        "hypotheses": [],
        "validation_status": {
            "schema": "mini-palari.visual-rule-hypothesis-validation.v0.1",
            "decision": "blocked",
            "accepted_count": 0,
            "rejected_count": 0,
            "rejections": [],
            "runtime_authority_granted": False,
        },
    }


def _run_command(prompt_pack: dict[str, Any], command: list[str], timeout_seconds: int) -> Any:
    completed = subprocess.run(
        command,
        input=_stable_json(prompt_pack),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "hypotheses": [],
            "sidecar_error": "command_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
            "stdout": completed.stdout[-4000:],
        }
    try:
        return json.loads(completed.stdout)
    except Exception as exc:  # noqa: BLE001 - untrusted model/command output
        return {
            "hypotheses": [],
            "sidecar_error": "command_output_not_json",
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": completed.stdout[-4000:],
        }


def run_visual_rule_sidecar(
    prompt_pack: dict[str, Any],
    *,
    output_dir: str | Path,
    sidecar: str = "fixture",
    fixture_payload: Any | None = None,
    command: list[str] | None = None,
    allow_command: bool = False,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run a proposal-only visual-rule sidecar and log auditable artifacts.

    ``sidecar='fixture'`` never needs credentials or model access. ``sidecar='command'``
    requires both ``allow_command=True`` and ``MINI_PALARI_ALLOW_VLM_COMMAND=1``.
    No sidecar mode grants runtime authority.
    """

    output_path = Path(output_dir)
    if sidecar == "fixture":
        raw_payload = fixture_payload if fixture_payload is not None else default_fixture_payload(prompt_pack)
        model_source = "fixture"
        command_for_log = None
    elif sidecar == "command":
        command_for_log = list(command) if command is not None else None
        if not command:
            return _blocked_result(
                prompt_pack=prompt_pack,
                output_dir=output_path,
                sidecar=sidecar,
                command=command_for_log,
                decision="blocked_missing_command",
                reason="command sidecar requires an explicit command list",
            )
        if not allow_command or os.environ.get(COMMAND_GATE_ENV) != "1":
            return _blocked_result(
                prompt_pack=prompt_pack,
                output_dir=output_path,
                sidecar=sidecar,
                command=command_for_log,
                decision="blocked_command_not_approved",
                reason=f"command sidecar requires allow_command=True and {COMMAND_GATE_ENV}=1",
            )
        assert command_for_log is not None
        raw_payload = _run_command(prompt_pack, command_for_log, timeout_seconds)
        model_source = "command"
    else:
        return _blocked_result(
            prompt_pack=prompt_pack,
            output_dir=output_path,
            sidecar=sidecar,
            command=command,
            decision="blocked_unknown_sidecar",
            reason=f"unknown sidecar mode:{sidecar}",
        )

    hypotheses, validation_status = parse_visual_rule_hypotheses(
        raw_payload,
        available_actions=prompt_pack.get("available_actions") or [],
    )
    decision = validation_status.get("decision", "rejected")
    log = _base_log(
        prompt_pack,
        sidecar=sidecar,
        command=command_for_log if sidecar == "command" else None,
        model_source=model_source,
        decision=decision,
    )
    log["validation_status"] = validation_status
    log["hypothesis_count"] = len(hypotheses)
    paths = _write_artifacts(output_dir=output_path, prompt_pack=prompt_pack, raw_payload=raw_payload, log=log)
    return {
        **log,
        **paths,
        "hypotheses": hypotheses,
        "validation_status": validation_status,
        "runtime_authority_granted": False,
    }
