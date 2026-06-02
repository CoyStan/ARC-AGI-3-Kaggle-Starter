"""Experimental live-LLM ARC-AGI-3 runner.

This intentionally breaks the Mini-Palari proposal-only/no-network guardrail for a
local ceiling experiment: the model has direct action authority during play.
Do not copy into submission artifacts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import httpx
import importlib.util
import json
import os
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

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from agents.agent import Agent

ACTIONS_BY_NAME = {a.name: a for a in GameAction}


def load_baseline_class():
    spec = importlib.util.spec_from_file_location("baseline_agent_module", ROOT / "agent" / "my_agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def latest_grid(frame_payload: Any) -> list[list[int]]:
    if not frame_payload:
        return [[0]]
    payload = frame_payload
    # FrameData.frame is normally a list of grids; raw/current observation may be similar.
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        if payload and payload[0] and isinstance(payload[0][0], list):
            payload = payload[-1]
    rows: list[list[int]] = []
    for row in payload:
        if isinstance(row, list):
            rows.append([int(x) if isinstance(x, int) else 0 for x in row])
    return rows or [[0]]


def grid_summary(grid: list[list[int]], max_points: int = 180) -> dict[str, Any]:
    h = len(grid)
    w = max((len(r) for r in grid), default=0)
    counts = Counter()
    pts_by_color: dict[int, list[tuple[int, int]]] = {}
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                counts[v] += 1
                pts_by_color.setdefault(v, []).append((x, y))
    colors = []
    for c, pts in sorted(pts_by_color.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        colors.append({
            "color": c,
            "count": len(pts),
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "centroid": [round(sum(xs)/len(xs),2), round(sum(ys)/len(ys),2)],
            "sample_points": pts[:min(20, len(pts))],
        })
    points = []
    for c, pts in sorted(pts_by_color.items()):
        for x, y in pts[:max(1, max_points // max(1, len(pts_by_color)))]:
            points.append([x, y, c])
    return {"width": w, "height": h, "nonzero": sum(counts.values()), "colors": colors, "sample_nonzero_points": points[:max_points]}


def grid_delta_summary(before: list[list[int]], after: list[list[int]], limit: int = 80) -> dict[str, Any]:
    h = max(len(before), len(after))
    w = max(max((len(r) for r in before), default=0), max((len(r) for r in after), default=0))
    changed = []
    color_transitions = Counter()
    for y in range(h):
        brow = before[y] if y < len(before) else []
        arow = after[y] if y < len(after) else []
        for x in range(w):
            bv = brow[x] if x < len(brow) else 0
            av = arow[x] if x < len(arow) else 0
            if bv != av:
                color_transitions[(bv, av)] += 1
                if len(changed) < limit:
                    changed.append([x, y, bv, av])
    return {"changed_count": sum(color_transitions.values()), "sample_changes_xy_before_after": changed, "color_transitions": [{"from": k[0], "to": k[1], "count": v} for k, v in color_transitions.most_common(12)]}


def downsample_ascii(grid: list[list[int]], size: int = 16) -> str:
    h = len(grid); w = max((len(r) for r in grid), default=1)
    out = []
    for by in range(size):
        row_chars = []
        y0 = by*h//size; y1 = max(y0+1, (by+1)*h//size)
        for bx in range(size):
            x0 = bx*w//size; x1 = max(x0+1, (bx+1)*w//size)
            cnt = Counter()
            for y in range(y0, min(y1,h)):
                row = grid[y]
                for x in range(x0, min(x1,len(row))):
                    if row[x]: cnt[row[x]] += 1
            row_chars.append(format(cnt.most_common(1)[0][0], 'x') if cnt else '.')
        out.append(''.join(row_chars))
    return '\n'.join(out)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object in model output: {text[:500]}")


class LiveLLMController:
    def __init__(self, model: str, temperature: float, effort: str, outdir: Path, timeout: float = 90.0, backend: str = "auto"):
        self.model = model
        self.temperature = temperature
        self.effort = effort
        self.outdir = outdir
        self.timeout = timeout
        self.backend = backend
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.backend == "auto":
            self.backend = "openai" if self.api_key else "hermes"
        if self.backend == "openai" and not self.api_key:
            raise SystemExit("OPENAI_API_KEY is not set")
        self.client = httpx.Client(timeout=timeout) if self.backend == "openai" else None
        self.calls = 0
        self.failures = 0

    def choose(self, *, game_id: str, step: int, latest_frame: Any, history: list[dict[str, Any]], available: list[str], chunk_size: int = 1) -> dict[str, Any]:
        grid = latest_grid(getattr(latest_frame, "frame", None))
        payload = {
            "game_id": game_id,
            "step": step,
            "state": str(getattr(latest_frame, "state", "")),
            "levels_completed": getattr(latest_frame, "levels_completed", 0),
            "win_levels": getattr(latest_frame, "win_levels", None),
            "available_actions": available,
            "action_semantics_unknown": True,
            "history_last_steps": history[-16:],
            "grid_summary": grid_summary(grid),
            "downsample_16x16_majority_nonzero": downsample_ascii(grid),
            "instruction": f"Choose the next ARC-AGI-3 action program to maximize levels completed. You MUST return exactly {chunk_size} actions unless you intentionally end with RESET. ACTION6 requires integer x,y in [0,63]. Other actions have no args. You have direct live action authority in this local experiment. Return only JSON.",
        }
        system = (
            "You are a strong ARC-AGI-3 game-playing agent with direct action authority for a local ceiling experiment. "
            "Infer mechanics from the current grid and action/result history. You are operating in chunked-control mode: produce a full executable action program now, not a single cautious probe. "
            "Include exploration early and exploitation/repeats where plausible; do not return fewer than the requested action count. "
            "Return strict JSON: {\"actions\":[{\"action\":\"ACTION1\", \"x\":0, \"y\":0, \"reason\":\"...\"}, ...], \"plan\":\"...\"}. "
            "Every action must be in available_actions. Use ACTION6 x/y only when ACTION6 is available."
        )
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ],
            "text": {"format": {"type": "json_object"}},
        }
        # GPT-5 family supports reasoning; older models may ignore/reject it, so retry without if needed.
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        if self.temperature >= 0:
            body["temperature"] = self.temperature
        if self.backend == "hermes":
            prompt = system + "\n\nGAME PAYLOAD JSON:\n" + json.dumps(payload, separators=(",", ":")) + "\n\nReturn only the strict JSON object."
            cmd = ["hermes", "-z", prompt, "--provider", "openai-codex", "-m", self.model, "-t", ""]
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=self.timeout)
            self.calls += 1
            if proc.returncode != 0:
                self.failures += 1
                raise RuntimeError(f"Hermes model call failed {proc.returncode}: {proc.stderr[-500:] or proc.stdout[-500:]}")
            text = proc.stdout.strip()
            data = {"id": None, "backend": "hermes", "stderr_tail": proc.stderr[-500:]}
        else:
            assert self.client is not None
            r = self.client.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=body)
            if r.status_code >= 400 and ("reasoning" in body or "temperature" in body):
                body.pop("reasoning", None); body.pop("temperature", None)
                r = self.client.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=body)
            self.calls += 1
            if r.status_code >= 400:
                self.failures += 1
                raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
            data = r.json()
            text = data.get("output_text")
            if not text:
                parts = []
                for item in data.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            parts.append(c.get("text", ""))
                text = "\n".join(parts)
        choice = extract_json(text or "")
        actions_payload = choice.get("actions")
        if not isinstance(actions_payload, list) or not actions_payload:
            actions_payload = [choice]
        sanitized_actions = []
        for item in actions_payload[: max(1, chunk_size)]:
            if not isinstance(item, dict):
                item = {"action": str(item)}
            action = str(item.get("action", "")).upper()
            if action not in available:
                action = next((a for a in available if a != "RESET"), available[0])
                item["sanitized_action"] = action
                item["sanitization_reason"] = "model_action_unavailable"
            item["action"] = action
            if action == "ACTION6":
                item["x"] = max(0, min(63, int(item.get("x", 32) or 32)))
                item["y"] = max(0, min(63, int(item.get("y", 32) or 32)))
            else:
                item.setdefault("x", 0); item.setdefault("y", 0)
            sanitized_actions.append(item)
        choice["actions"] = sanitized_actions
        if sanitized_actions:
            choice.update({k: v for k, v in sanitized_actions[0].items() if k not in choice})
        rec = {"request": payload, "response": choice, "raw_response_id": data.get("id"), "backend": self.backend, "model": self.model}
        (self.outdir / game_id).mkdir(parents=True, exist_ok=True)
        (self.outdir / game_id / f"step_{step:03d}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return choice


class LiveLLMAgent(Agent):
    MAX_ACTIONS = 80
    def __init__(self, *args: Any, controller: LiveLLMController, fallback_cls: Any | None = None, chunk_size: int = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.history: list[dict[str, Any]] = []
        self.fallback = fallback_cls(*args, **kwargs) if fallback_cls else None
        self.chunk_size = max(1, int(chunk_size))
        self.pending_actions: list[dict[str, Any]] = []

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return str(getattr(latest_frame, "state", "")).upper().endswith("WIN")

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        raw_available = getattr(latest_frame, "available_actions", None) or list(GameAction)
        available = []
        for a in raw_available:
            if hasattr(a, "name"):
                available.append(a.name)
            elif isinstance(a, int):
                available.append(GameAction.from_id(a).name)
            else:
                available.append(str(a).split('.')[-1].upper())
        try:
            if not self.pending_actions:
                choice = self.controller.choose(game_id=self.game_id, step=self.action_counter, latest_frame=latest_frame, history=self.history, available=available, chunk_size=self.chunk_size)
                self.pending_actions.extend(choice.get("actions") or [choice])
            choice = self.pending_actions.pop(0)
            name = choice["action"]
            action = ACTIONS_BY_NAME[name]
            if name == "ACTION6":
                action.set_data({"x": choice["x"], "y": choice["y"]})
            action.reasoning = {"policy":"live_llm_direct_authority", "model": self.controller.model, "choice": choice}
            before_grid = latest_grid(getattr(latest_frame, "frame", None))
            self.history.append({"step": self.action_counter, "action": name, "x": choice.get("x"), "y": choice.get("y"), "reason": choice.get("reason"), "plan": choice.get("plan"), "levels_before": getattr(latest_frame,"levels_completed",0), "before_nonzero": grid_summary(before_grid, max_points=0).get("nonzero")})
            return action
        except Exception as e:
            self.history.append({"step": self.action_counter, "error": repr(e), "fallback": bool(self.fallback)})
            if self.fallback is not None:
                return self.fallback.choose_action(frames, latest_frame)
            return ACTIONS_BY_NAME["RESET"]

    def append_frame(self, frame: Any) -> None:
        if self.history:
            previous_frame = self.frames[-1] if self.frames else None
            before_grid = latest_grid(getattr(previous_frame, "frame", None)) if previous_frame is not None else [[0]]
            after_grid = latest_grid(getattr(frame, "frame", None))
            self.history[-1]["levels_after"] = getattr(frame, "levels_completed", None)
            self.history[-1]["state_after"] = str(getattr(frame, "state", ""))
            self.history[-1]["delta"] = grid_delta_summary(before_grid, after_grid)
            self.history[-1]["after_summary"] = {k: v for k, v in grid_summary(after_grid, max_points=0).items() if k != "sample_nonzero_points"}
        return super().append_frame(frame)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default=None)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--model", default=os.environ.get("MINI_PALARI_LIVE_LLM_MODEL", "gpt-5.1"))
    p.add_argument("--temperature", type=float, default=-1.0)
    p.add_argument("--effort", default="high")
    p.add_argument("--outdir", default="model-eval/artifacts/live-llm-run")
    p.add_argument("--backend", default="auto", choices=["auto", "openai", "hermes"])
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--fallback-baseline", action="store_true")
    args = p.parse_args()

    outdir = ROOT / args.outdir / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    if args.game:
        wanted = {g.strip().split('-')[0] for g in args.game.split(',') if g.strip()}
        game_ids = [e.game_id.split('-')[0] for e in envs if e.game_id.split('-')[0] in wanted]
    else:
        game_ids = [e.game_id.split('-')[0] for e in envs]
    controller = LiveLLMController(args.model, args.temperature, args.effort, outdir, backend=args.backend)
    fallback_cls = load_baseline_class() if args.fallback_baseline else None
    LiveLLMAgent.MAX_ACTIONS = args.max_steps
    per_game = []
    print(f"EXPERIMENT: live LLM direct authority; model={args.model}; games={len(game_ids)}; max_steps={args.max_steps}; outdir={outdir}")
    for i, gid in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {gid} ===", flush=True)
        env = arc.make(gid)
        agent = LiveLLMAgent(card_id="live-llm-local", game_id=gid, agent_name=f"LiveLLM.{gid}", ROOT_URL="http://localhost", record=False, arc_env=env, tags=["live-llm"], controller=controller, fallback_cls=fallback_cls, chunk_size=args.chunk_size)
        agent.main()
        final = agent.frames[-1]
        rec = {"game_id": gid, "state": str(final.state), "levels_completed": final.levels_completed, "actions": agent.action_counter, "history": agent.history}
        (outdir / gid / "game_summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        per_game.append(rec)
        print(f"  → state={final.state}, levels_completed={final.levels_completed}, actions={agent.action_counter}", flush=True)
    sc = arc.get_scorecard()
    score_val = sc.score if hasattr(sc, "score") else sc
    summary = {"experiment":"live_llm_direct_authority", "model": args.model, "max_steps": args.max_steps, "score": score_val, "api_calls": controller.calls, "api_failures": controller.failures, "per_game": per_game, "outdir": str(outdir)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========= SUMMARY =========")
    for r in per_game:
        print(f"  {r['game_id']:8} levels={r['levels_completed']:3} actions={r['actions']:5} state={r['state']}")
    print(f"\nAggregate scorecard score: {score_val}")
    print(f"Artifacts: {outdir}")


if __name__ == "__main__":
    main()
