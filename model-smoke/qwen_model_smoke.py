"""No-authority Kaggle local model smoke test for Mini-Palari.

This script is intentionally separate from submitted MyAgent gameplay. It attaches
one Kaggle model source, loads it offline if possible, asks for one structured
proposal, validates that proposal, and always grants zero runtime authority.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any

SCHEMA_PROPOSAL = "mini-palari.model-proposal.v0.1"
SCHEMA_VERIFY = "mini-palari.model-proposal-verification.v0.1"
SCHEMA_RESULT = "mini-palari.kaggle-model-smoke-result.v0.1"
MODEL_SOURCE = "qwen-lm/qwen-3/transformers/0.6b/1"
EXPECTED_MARKERS = {"config.json", "tokenizer.json", "model.safetensors"}
REQUIRED_PROPOSAL_FIELDS = {
    "schema",
    "proposal_id",
    "kind",
    "candidate",
    "confidence",
    "rationale",
    "expected_observations",
    "tests_to_run",
    "forbidden_authority",
    "source_trace_refs",
}
REQUIRED_FORBIDDEN = {
    "direct_action",
    "policy_rank_without_verification",
    "memory_write",
    "network",
}


def emit(label: str, payload: Any) -> None:
    print(f"MINI_PALARI_SMOKE_{label}=" + json.dumps(payload, sort_keys=True, default=str))


def shallow_input_tree(root: pathlib.Path = pathlib.Path("/kaggle/input")) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not root.exists():
        return entries
    for path in sorted(root.glob("*")):
        item = {"path": str(path), "is_dir": path.is_dir(), "children": []}
        if path.is_dir():
            children = []
            for child in sorted(path.glob("*"))[:30]:
                children.append({"name": child.name, "is_dir": child.is_dir()})
            item["children"] = children
        entries.append(item)
    return entries


def find_model_path(root: pathlib.Path = pathlib.Path("/kaggle/input")) -> str | None:
    if not root.exists():
        return None
    scored: list[tuple[int, pathlib.Path]] = []
    for base in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        names = {p.name for p in base.glob("*")}
        score = len(EXPECTED_MARKERS & names)
        if score:
            scored.append((score, base))
    if not scored:
        return None
    scored.sort(key=lambda pair: (-pair[0], len(str(pair[1]))))
    return str(scored[0][1])


def validate_proposal(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "proposal_not_object"
    missing = sorted(REQUIRED_PROPOSAL_FIELDS - set(obj))
    if missing:
        return "missing_fields:" + ",".join(missing)
    if obj.get("schema") != SCHEMA_PROPOSAL:
        return "bad_schema"
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        return "bad_confidence"
    if not 0.0 <= conf <= 1.0:
        return "confidence_out_of_range"
    forbidden = set(obj.get("forbidden_authority") or [])
    if not REQUIRED_FORBIDDEN.issubset(forbidden):
        return "forbidden_authority_missing"
    return "ok"


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def main() -> None:
    started = time.monotonic()
    env = {
        "python": sys.version,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "input_exists": pathlib.Path("/kaggle/input").exists(),
        "model_source": MODEL_SOURCE,
        "runtime_authority_granted": False,
    }
    emit("ENV", env)

    try:
        nvidia = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=15)
        emit("NVIDIA_SMI", {"returncode": nvidia.returncode, "stdout_head": nvidia.stdout[:3000], "stderr_head": nvidia.stderr[:1000]})
    except Exception as exc:
        emit("NVIDIA_SMI", {"error": f"{type(exc).__name__}: {exc}"})

    tree = shallow_input_tree()
    emit("INPUT_TREE", tree)
    model_path = find_model_path()
    emit("MODEL_PATH", {"model_path": model_path, "expected_source": MODEL_SOURCE})

    import_report: dict[str, str] = {}
    for name in ["torch", "transformers", "sentencepiece", "tokenizers", "accelerate"]:
        try:
            mod = __import__(name)
            import_report[name] = str(getattr(mod, "__version__", "present"))
        except Exception as exc:
            import_report[name] = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    emit("IMPORTS", import_report)

    load_result: dict[str, Any] = {
        "schema": "mini-palari.kaggle-model-load.v0.1",
        "model_path": model_path,
        "load_ok": False,
        "load_seconds": None,
        "error": None,
    }
    inference_result: dict[str, Any] = {
        "raw_text_head": None,
        "inference_ok": False,
        "inference_seconds": None,
        "error": None,
    }
    proposal: dict[str, Any] | None = None
    validation = "not_run"

    if not model_path:
        load_result["error"] = "model_path_not_found"
    else:
        load_start = time.monotonic()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            cuda_ok = False
            cuda_reason = "cuda_unavailable"
            if torch.cuda.is_available():
                try:
                    capability = torch.cuda.get_device_capability(0)
                    # Kaggle may allocate P100 even when a T4-like GPU was requested. The
                    # current PyTorch image reports support for sm_70+, while P100 is sm_60.
                    cuda_ok = capability >= (7, 0)
                    cuda_reason = f"capability_{capability[0]}_{capability[1]}"
                except Exception as exc:
                    cuda_reason = f"capability_probe_failed:{type(exc).__name__}:{exc}"
            load_result["cuda_selected"] = cuda_ok
            load_result["cuda_reason"] = cuda_reason
            dtype = torch.float16 if cuda_ok else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                device_map="auto" if cuda_ok else None,
            )
            if not cuda_ok:
                model.to("cpu")
            model.eval()
            load_result["load_ok"] = True
        except Exception as exc:
            load_result["error"] = f"{type(exc).__name__}: {exc}"
            load_result["traceback_head"] = traceback.format_exc()[:2000]
        finally:
            load_result["load_seconds"] = round(time.monotonic() - load_start, 3)
    emit("LOAD", load_result)

    if load_result.get("load_ok"):
        infer_start = time.monotonic()
        try:
            import torch

            scene_summary = {
                "schema": "mini-palari.scene-summary.v0.1",
                "game_id": "smoke_private_like",
                "turn_index": 0,
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
                "frame_summary": {
                    "height": 8,
                    "width": 8,
                    "colors": {"0": 58, "2": 4, "5": 2},
                    "nonzero_components": [
                        {"color": 2, "bbox": [1, 1, 2, 2], "centroid": [1.5, 1.5]},
                        {"color": 5, "bbox": [6, 6, 6, 7], "centroid": [6.0, 6.5]},
                    ],
                },
                "recent_probes": [],
                "known_hypotheses": [],
            }
            prompt = (
                "You must output ONLY this JSON object shape, with no markdown and no extra text:\n"
                "{\n"
                "  \"schema\": \"mini-palari.model-proposal.v0.1\",\n"
                "  \"proposal_id\": \"smoke-qwen-001\",\n"
                "  \"kind\": \"game_family\",\n"
                "  \"candidate\": {\"family\": \"grid_navigation_candidate\"},\n"
                "  \"confidence\": 0.50,\n"
                "  \"rationale\": \"short reason based only on the scene summary\",\n"
                "  \"expected_observations\": [\"bounded probes may reveal object motion\"],\n"
                "  \"tests_to_run\": [\"try each simple action once and compare centroid deltas\"],\n"
                "  \"forbidden_authority\": [\"direct_action\", \"policy_rank_without_verification\", \"memory_write\", \"network\"],\n"
                "  \"source_trace_refs\": []\n"
                "}\n"
                "Do not copy the scene schema. Do not choose a gameplay action. Do not create action args. "
                "Use this scene summary only as evidence: "
                + json.dumps(scene_summary, sort_keys=True)
            )
            messages = [
                {"role": "system", "content": "You are a proposal-only ARC hypothesis assistant. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[-1] :]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            inference_result["raw_text_head"] = raw_text[:2000]
            inference_result["inference_ok"] = True
            proposal = extract_json_object(raw_text)
            validation = validate_proposal(proposal)
        except Exception as exc:
            inference_result["error"] = f"{type(exc).__name__}: {exc}"
            inference_result["traceback_head"] = traceback.format_exc()[:2000]
            validation = "inference_error"
        finally:
            inference_result["inference_seconds"] = round(time.monotonic() - infer_start, 3)
    emit("INFERENCE", inference_result)
    emit("PROPOSAL", proposal if proposal is not None else {"proposal": None})
    emit("VALIDATION", {"validation": validation})

    verification = {
        "schema": SCHEMA_VERIFY,
        "proposal_id": proposal.get("proposal_id", "invalid") if isinstance(proposal, dict) else "invalid",
        "decision": "blocked_needs_probe_evidence" if validation == "ok" else "blocked_invalid_schema_or_runtime_failure",
        "evidence_refs": [],
        "counter_evidence_refs": [],
        "runtime_authority_granted": False,
    }
    emit("VERIFICATION", verification)

    try:
        import torch

        memory = {
            "cuda_available": torch.cuda.is_available(),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "max_memory_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        memory = {"error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "schema": SCHEMA_RESULT,
        "candidate": MODEL_SOURCE,
        "internet_used": False,
        "load_ok": bool(load_result.get("load_ok")),
        "load_seconds": load_result.get("load_seconds"),
        "inference_ok": bool(inference_result.get("inference_ok")),
        "inference_seconds": inference_result.get("inference_seconds"),
        "valid_json": validation == "ok",
        "validation": validation,
        "runtime_authority_granted": False,
        "memory": memory,
        "total_seconds": round(time.monotonic() - started, 3),
        "decision": "candidate_ready_for_proposal_only_adapter" if validation == "ok" else "candidate_needs_packaging_or_prompt_fix",
    }
    emit("SUMMARY", summary)


if __name__ == "__main__":
    main()
