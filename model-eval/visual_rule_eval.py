"""Small cached-trace smoke eval for the visual rule-hypothesis sidecar loop.

This is an eval-only artifact pipeline, not submitted runtime behavior. It runs:
trace_pack -> prompt -> sidecar proposal -> local verifier -> bounded report.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_EVAL = "mini-palari.visual-rule-eval.v0.1"
SCHEMA_NO_NETWORK_REPLAY = "mini-palari.visual-rule-no-network-replay.v0.1"


def _load_module(name: str):
    module_path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prompt_mod = _load_module("visual_rule_prompt")
_sidecar_mod = _load_module("visual_rule_sidecar")
_verifier_mod = _load_module("visual_rule_verifier")
_trace_mod = _load_module("visual_trace_artifacts")

build_visual_rule_prompt = _prompt_mod.build_visual_rule_prompt
run_visual_rule_sidecar = _sidecar_mod.run_visual_rule_sidecar
verify_visual_rule_hypotheses = _verifier_mod.verify_visual_rule_hypotheses
load_trace_pack = _trace_mod.load_trace_pack


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "visual_rule_eval_report.json"
    report["report_path"] = str(path)
    path.write_text(_stable_json(report) + "\n", encoding="utf-8")
    return report


def _scorecard(games: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = sum(1 for g in games if g.get("verification_decision") == "accepted_candidate")
    rejected = sum(1 for g in games if g.get("verification_decision") == "rejected")
    blocked = sum(1 for g in games if str(g.get("verification_decision", "")).startswith("blocked"))
    sidecar_accepted = sum(1 for g in games if g.get("sidecar_validation_decision") == "accepted_candidate")
    return {
        "accepted_candidates": accepted,
        "rejected": rejected,
        "blocked": blocked,
        "sidecar_accepted_candidates": sidecar_accepted,
        "levels_completed": sum(int(g.get("levels_completed") or 0) for g in games),
        "runtime_authority_granted": False,
    }


def _game_eval(
    trace_pack_path: str | Path,
    *,
    output_dir: Path,
    sidecar: str,
    command: list[str] | None,
    allow_command: bool,
) -> dict[str, Any]:
    trace_pack_path = Path(trace_pack_path)
    trace_pack = load_trace_pack(trace_pack_path)
    game_id = str(trace_pack.get("game_id") or trace_pack_path.parent.name)
    safe_game = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in game_id) or "game"
    game_dir = output_dir / safe_game
    prompt_pack = build_visual_rule_prompt(trace_pack)
    sidecar_result = run_visual_rule_sidecar(
        prompt_pack,
        output_dir=game_dir / "sidecar",
        sidecar=sidecar,
        command=command,
        allow_command=allow_command,
    )
    verification = verify_visual_rule_hypotheses(sidecar_result.get("hypotheses") or [], trace_pack)
    verification_path = game_dir / "visual_rule_verification.json"
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(_stable_json(verification) + "\n", encoding="utf-8")
    selected = verification.get("selected_hypothesis") or {}
    return {
        "game_id": game_id,
        "trace_pack_path": str(trace_pack_path),
        "prompt_path": sidecar_result.get("prompt_path"),
        "raw_output_path": sidecar_result.get("raw_output_path"),
        "sidecar_log_path": sidecar_result.get("sidecar_log_path"),
        "verification_path": str(verification_path),
        "sidecar": sidecar_result.get("sidecar"),
        "sidecar_validation_decision": (sidecar_result.get("validation_status") or {}).get("decision"),
        "hypothesis_count": len(sidecar_result.get("hypotheses") or []),
        "verification_decision": verification.get("decision"),
        "selected_hypothesis_id": selected.get("hypothesis_id"),
        "plan_steps": len(verification.get("plan_to_execute") or []),
        "levels_completed": 0,
        "score_delta_vs_baseline": None,
        "runtime_authority_granted": False,
    }


def run_visual_rule_eval(
    trace_pack_paths: list[str | Path],
    *,
    output_dir: str | Path,
    sidecar: str = "fixture",
    command: list[str] | None = None,
    allow_command: bool = False,
) -> dict[str, Any]:
    """Run a no-authority visual-rule smoke eval over cached trace packs."""

    out = Path(output_dir)
    if not trace_pack_paths:
        report = {
            "schema": SCHEMA_EVAL,
            "created_at": _now_iso(),
            "decision": "blocked_no_traces",
            "sidecar": sidecar,
            "game_count": 0,
            "games": [],
            "scorecard": {"accepted_candidates": 0, "rejected": 0, "blocked": 0, "sidecar_accepted_candidates": 0, "levels_completed": 0, "runtime_authority_granted": False},
            "claims": {
                "generalization": "not_claimed",
                "performance": "not_claimed",
                "scope": "cached_trace_smoke_only",
            },
            "runtime_authority_granted": False,
        }
        return _write_report(out, report)

    games = [
        _game_eval(path, output_dir=out / "games", sidecar=sidecar, command=command, allow_command=allow_command)
        for path in trace_pack_paths
    ]
    report = {
        "schema": SCHEMA_EVAL,
        "created_at": _now_iso(),
        "decision": "completed_cached_trace_smoke",
        "sidecar": sidecar,
        "game_count": len(games),
        "games": games,
        "scorecard": _scorecard(games),
        "claims": {
            "generalization": "not_claimed",
            "performance": "not_claimed",
            "scope": "cached_trace_smoke_only",
        },
        "runtime_authority_granted": False,
    }
    return _write_report(out, report)


def discover_trace_packs(root: str | Path) -> list[str]:
    """Return sorted cached trace_pack.json paths under root."""

    return [str(p) for p in sorted(Path(root).glob("**/trace_pack.json"))]


def _trace_artifact_summary(trace_pack_path: str | Path) -> dict[str, Any]:
    path = Path(trace_pack_path)
    item: dict[str, Any] = {
        "trace_pack_path": str(path),
        "exists": path.exists(),
        "image_count": 0,
        "missing_image_paths": [],
    }
    if not path.exists():
        return item
    try:
        trace_pack = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - artifact report should capture malformed JSON
        item["parse_error"] = f"{type(exc).__name__}: {exc}"
        item["exists"] = False
        return item
    image_paths = [Path(str(frame.get("image_path"))) for frame in trace_pack.get("frames") or [] if frame.get("image_path")]
    item["game_id"] = trace_pack.get("game_id")
    item["frame_count"] = len(trace_pack.get("frames") or [])
    item["image_count"] = len(image_paths)
    missing = [str(p) for p in image_paths if not p.exists()]
    item["missing_image_paths"] = missing
    if missing:
        item["exists"] = False
    return item


def _write_replay_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "visual_rule_no_network_replay_report.json"
    report["report_path"] = str(path)
    path.write_text(_stable_json(report) + "\n", encoding="utf-8")
    return report


def run_visual_rule_no_network_replay(
    trace_pack_paths: list[str | Path],
    *,
    output_dir: str | Path,
    sidecar: str = "fixture",
) -> dict[str, Any]:
    """Replay the visual-rule eval from cached artifacts only.

    This helper intentionally exposes no command sidecar path and no ARC environment
    loader. It first verifies that all declared trace packs and image artifacts are
    present, then runs the fixture sidecar/eval pipeline over those local files.
    """

    out = Path(output_dir)
    sorted_paths = [str(p) for p in sorted(Path(p) for p in trace_pack_paths)]
    artifacts = [_trace_artifact_summary(path) for path in sorted_paths]
    base_report = {
        "schema": SCHEMA_NO_NETWORK_REPLAY,
        "created_at": _now_iso(),
        "replay_mode": "cached_artifacts_only",
        "sidecar": sidecar,
        "network_allowed": False,
        "arc_environment_accessed": False,
        "input_trace_pack_paths": sorted_paths,
        "input_artifacts": artifacts,
        "source_trace_count": len(sorted_paths),
        "claims": {
            "generalization": "not_claimed",
            "performance": "not_claimed",
            "scope": "cached_artifact_replay_only",
        },
        "runtime_authority_granted": False,
    }
    if any(not item.get("exists") for item in artifacts):
        return _write_replay_report(
            out,
            {
                **base_report,
                "decision": "blocked_missing_cached_artifacts",
            },
        )

    eval_report = run_visual_rule_eval(
        list(sorted_paths),
        output_dir=out / "eval",
        sidecar=sidecar,
        command=None,
        allow_command=False,
    )
    return _write_replay_report(
        out,
        {
            **base_report,
            "decision": "completed_no_network_artifact_replay",
            "eval_report_path": eval_report.get("report_path"),
            "eval_report": eval_report,
        },
    )
