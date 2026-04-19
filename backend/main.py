"""
AI RC Car Search System - Local Simulator
==========================================

Three actors:
    A = Phone (browser UI)         -> sends user messages
    B = Backend + Local LLM        -> this file (FastAPI) + Ollama
    C = RC Car Simulator           -> a 2D grid simulated inside this file
                                      and rendered in the browser

The point of this file is to make the *communication loop* between A, B, C
very obvious. Everything else (room, LLM, controller) is intentionally
small and beginner-friendly.

Flow per "tick" while searching:
    1. C reports what its fake camera sees (based on car pose)
    2. B sends that camera description + user goal to the LLM
    3. LLM returns strict JSON: {visible_objects, target_found, next_action, ...}
    4. B's safety/controller layer turns next_action into a safe sim command
    5. C applies the command and updates pose
    6. B sends an update message to A (the phone UI)
    7. Loop again until target_found or user says stop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")
TICK_SECONDS = 1.0  # how often the search loop runs while active
GRID_SIZE = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rc-sim")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Room Simulator (actor C)
# ---------------------------------------------------------------------------
#
# Coordinate system:
#   x = column (0..GRID_SIZE-1), grows to the right
#   y = row    (0..GRID_SIZE-1), grows downward (like screen pixels)
#
# Direction is one of: "N", "E", "S", "W"
#   N = up    (dx=0, dy=-1)
#   E = right (dx=1, dy=0)
#   S = down  (dx=0, dy=1)
#   W = left  (dx=-1, dy=0)
#
# Field of view: a small cone of cells in front of the car (depth=4, width=3)

DIR_VECTORS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}


@dataclass
class WorldObject:
    name: str
    x: int
    y: int


@dataclass
class RoomSimulator:
    car_x: int = 1
    car_y: int = 8
    car_dir: str = "N"
    objects: list[WorldObject] = field(default_factory=lambda: [
        WorldObject("football", 5, 2),
        WorldObject("key",      8, 6),
        WorldObject("chair",    3, 4),
        WorldObject("table",    6, 7),
    ])
    last_action: str = "idle"

    # ---- pose helpers -----------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def cell_blocked(self, x: int, y: int) -> bool:
        # chair and table act as obstacles; football & key are pickup-able
        for o in self.objects:
            if o.x == x and o.y == y and o.name in {"chair", "table"}:
                return True
        return False

    # ---- low-level motions (only called by the safety controller) --------

    def move_forward(self) -> str:
        dx, dy = DIR_VECTORS[self.car_dir]
        nx, ny = self.car_x + dx, self.car_y + dy
        if not self.in_bounds(nx, ny):
            return "blocked_by_wall"
        if self.cell_blocked(nx, ny):
            return "blocked_by_obstacle"
        self.car_x, self.car_y = nx, ny
        return "moved_forward"

    def turn_left(self) -> str:
        self.car_dir = LEFT_OF[self.car_dir]
        return "turned_left"

    def turn_right(self) -> str:
        self.car_dir = RIGHT_OF[self.car_dir]
        return "turned_right"

    def stop(self) -> str:
        return "stopped"

    # ---- fake camera ------------------------------------------------------

    def field_of_view_cells(self) -> list[tuple[int, int]]:
        """Return the grid cells the car can 'see' (a small cone in front)."""
        dx, dy = DIR_VECTORS[self.car_dir]
        # perpendicular vector for cone width
        px, py = -dy, dx
        cells: list[tuple[int, int]] = []
        for depth in range(1, 5):  # 4 cells deep
            for offset in (-1, 0, 1):  # 3 cells wide at each depth
                cx = self.car_x + dx * depth + px * offset
                cy = self.car_y + dy * depth + py * offset
                if self.in_bounds(cx, cy):
                    cells.append((cx, cy))
        return cells

    def camera_view(self) -> dict:
        """What the car's fake camera 'sees'. This is the input to the LLM."""
        fov = self.field_of_view_cells()
        fov_set = set(fov)
        visible = []
        for o in self.objects:
            if (o.x, o.y) in fov_set:
                # describe roughly where it is in the frame
                rel_dx = o.x - self.car_x
                rel_dy = o.y - self.car_y
                # forward distance along car's heading
                fdx, fdy = DIR_VECTORS[self.car_dir]
                forward_dist = rel_dx * fdx + rel_dy * fdy
                # left/right offset
                pdx, pdy = -fdy, fdx
                side = rel_dx * pdx + rel_dy * pdy
                horizontal = "center" if side == 0 else ("right" if side > 0 else "left")
                visible.append({
                    "name": o.name,
                    "distance": int(forward_dist),
                    "horizontal": horizontal,
                })
        return {
            "car": {"x": self.car_x, "y": self.car_y, "dir": self.car_dir},
            "fov_cells": fov,
            "visible_objects": visible,
        }

    def snapshot(self) -> dict:
        return {
            "grid_size": GRID_SIZE,
            "car": {"x": self.car_x, "y": self.car_y, "dir": self.car_dir},
            "objects": [asdict(o) for o in self.objects],
            "fov_cells": self.field_of_view_cells(),
            "last_action": self.last_action,
        }


# ---------------------------------------------------------------------------
# Safety / Controller layer
# ---------------------------------------------------------------------------
#
# Design rule from the brief:
#   "The LLM must NOT directly control raw motor speed.
#    The LLM only gives high-level actions.
#    The backend safety/controller layer converts them into safe simulator
#    actions."
#
# So here we accept ONLY a small set of named actions and we always check
# bounds / obstacles via the simulator before applying anything.

ALLOWED_ACTIONS = {"move_forward", "turn_left", "turn_right", "stop"}


def safe_apply_action(room: RoomSimulator, action: str) -> str:
    """Validate & apply a high-level action. Returns a result string."""
    if action not in ALLOWED_ACTIONS:
        log.warning("Controller rejected unsafe/unknown action: %r", action)
        action = "stop"

    if action == "move_forward":
        result = room.move_forward()
    elif action == "turn_left":
        result = room.turn_left()
    elif action == "turn_right":
        result = room.turn_right()
    else:
        result = room.stop()

    room.last_action = result
    return result


# ---------------------------------------------------------------------------
# LLM client (actor B's brain) - Ollama, with mock fallback
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are the brain of a small RC car searching a room.
You receive a user goal and the car's current camera view.
You must reply with STRICT JSON only. No prose, no markdown.

JSON schema (all fields required):
{
  "visible_objects": [string, ...],
  "target_found": boolean,
  "target_object": string,
  "confidence": number,            // 0.0 - 1.0
  "next_action": string,           // one of: "move_forward","turn_left","turn_right","stop"
  "response_to_user": string
}

Rules:
- If the target object appears in visible_objects with a clear line of sight, set target_found=true and next_action="stop".
- If you don't see the target, choose a sensible scanning action ("turn_left", "turn_right", or "move_forward").
- Never output any field other than those listed above.
- Never output explanations outside the JSON.
"""


def build_user_prompt(goal: str, camera: dict) -> str:
    return (
        f"User goal: {goal or 'just look around'}\n"
        f"Camera view JSON: {json.dumps(camera)}\n"
        "Respond with the JSON object only."
    )


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    # try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _normalize_decision(raw: dict, goal: str, camera: dict) -> dict:
    """Make sure the decision dict has every required field & valid values."""
    visible_names = [o["name"] for o in camera.get("visible_objects", [])]
    target = (raw.get("target_object") or _goal_to_target(goal) or "").strip()
    next_action = raw.get("next_action", "stop")
    if next_action not in ALLOWED_ACTIONS:
        next_action = "stop"

    target_found = bool(raw.get("target_found", False))
    if target and target in visible_names:
        target_found = True
        next_action = "stop"

    return {
        "visible_objects": raw.get("visible_objects") or visible_names,
        "target_found": target_found,
        "target_object": target,
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "next_action": next_action,
        "response_to_user": str(raw.get("response_to_user", "") or ""),
    }


def _goal_to_target(goal: str) -> str:
    g = (goal or "").lower()
    for name in ("football", "key", "chair", "table"):
        if name in g:
            return name
    return ""


class LLMClient:
    """Talks to Ollama. Falls back to a deterministic mock if Ollama is down."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.url = url.rstrip("/")
        self.model = model
        self.mock_mode = False  # flips to True after first failed call

    async def decide(self, goal: str, camera: dict) -> tuple[dict, str]:
        """Return (decision_dict, source) where source is 'ollama' or 'mock'."""
        if not self.mock_mode:
            try:
                raw = await self._call_ollama(goal, camera)
                parsed = _extract_json(raw)
                if parsed is None:
                    log.warning("Ollama returned non-JSON, using mock. Raw: %r", raw[:200])
                    return self._mock_decide(goal, camera), "mock"
                return _normalize_decision(parsed, goal, camera), "ollama"
            except Exception as e:
                log.warning("Ollama call failed (%s). Falling back to mock for this session.", e)
                self.mock_mode = True
        return self._mock_decide(goal, camera), "mock"

    async def _call_ollama(self, goal: str, camera: dict) -> str:
        payload = {
            "model": self.model,
            "system": LLM_SYSTEM_PROMPT,
            "prompt": build_user_prompt(goal, camera),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(f"{self.url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")

    # ---- deterministic mock so the demo always works ----------------------

    def _mock_decide(self, goal: str, camera: dict) -> dict:
        target = _goal_to_target(goal)
        visible = camera.get("visible_objects", [])
        visible_names = [o["name"] for o in visible]

        # If the target is visible -> stop and report success
        if target and target in visible_names:
            obj = next(o for o in visible if o["name"] == target)
            return _normalize_decision({
                "visible_objects": visible_names,
                "target_found": True,
                "target_object": target,
                "confidence": 0.9,
                "next_action": "stop",
                "response_to_user": f"Found the {target}! It's {obj['distance']} cells ahead, {obj['horizontal']}.",
            }, goal, camera)

        # If we see *something*, drive toward it (great heuristic for a demo)
        if visible:
            obj = visible[0]
            if obj["horizontal"] == "left":
                action = "turn_left"
            elif obj["horizontal"] == "right":
                action = "turn_right"
            else:
                action = "move_forward"
            msg = (
                f"Looking for {target or 'something'}. I can see a {obj['name']} "
                f"to the {obj['horizontal']}, scanning further."
            )
            return _normalize_decision({
                "visible_objects": visible_names,
                "target_found": False,
                "target_object": target,
                "confidence": 0.4,
                "next_action": action,
                "response_to_user": msg,
            }, goal, camera)

        # Nothing visible: do a slow rotational scan
        return _normalize_decision({
            "visible_objects": [],
            "target_found": False,
            "target_object": target,
            "confidence": 0.1,
            "next_action": "turn_right",
            "response_to_user": f"Scanning for {target or 'objects'}...",
        }, goal, camera)


# ---------------------------------------------------------------------------
# Session: ties one phone (A) to one car (C) over one WebSocket
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, ws: WebSocket, llm: LLMClient):
        self.ws = ws
        self.llm = llm
        self.room = RoomSimulator()
        self.goal: str = ""
        self.searching: bool = False
        self.search_task: Optional[asyncio.Task] = None

    async def send(self, kind: str, **payload) -> None:
        try:
            await self.ws.send_json({"type": kind, **payload})
        except Exception as e:
            log.debug("send failed: %s", e)

    async def log_event(self, channel: str, text: str, data: Optional[dict] = None) -> None:
        """channel is one of: A->B, B->C, C->B, B->A, system"""
        await self.send("log", channel=channel, text=text, data=data or {})

    async def push_world(self) -> None:
        await self.send("world", **self.room.snapshot())

    async def handle_user_message(self, text: str) -> None:
        text_l = text.strip().lower()
        await self.log_event("A->B", f"user: {text}")

        if text_l in {"stop", "halt", "cancel"}:
            await self.stop_search(reason="user requested stop")
            return

        # Determine if this message is a "search" command or just a question.
        target = _goal_to_target(text_l)
        if target or text_l.startswith("find"):
            self.goal = text or f"find the {target}"
            await self.log_event("system", f"goal set: {self.goal}")
            await self.start_search()
            return

        # Otherwise: single-shot perception ("what do you see", "scan the room")
        if "scan" in text_l or "see" in text_l or "look" in text_l:
            self.goal = text
            await self.start_search(single_tick=True)
            return

        # Fallback: just ask the LLM once with no movement
        self.goal = text
        await self.start_search(single_tick=True)

    # ---- search loop ------------------------------------------------------

    async def start_search(self, single_tick: bool = False) -> None:
        await self.stop_search(reason="restarting", silent=True)
        self.searching = True
        self.search_task = asyncio.create_task(self._search_loop(single_tick=single_tick))

    async def stop_search(self, reason: str = "", silent: bool = False) -> None:
        self.searching = False
        if self.search_task and not self.search_task.done():
            self.search_task.cancel()
            try:
                await self.search_task
            except asyncio.CancelledError:
                pass
        self.search_task = None
        if not silent and reason:
            await self.log_event("system", f"search stopped: {reason}")

    async def _search_loop(self, single_tick: bool = False) -> None:
        max_ticks = 1 if single_tick else 60  # safety cap (~1 minute)
        try:
            for tick in range(max_ticks):
                if not self.searching:
                    break

                # 1) C -> B : camera frame
                camera = self.room.camera_view()
                await self.log_event("C->B", "camera frame", {
                    "tick": tick,
                    "car": camera["car"],
                    "visible_objects": camera["visible_objects"],
                })

                # 2) B -> LLM and back
                decision, source = await self.llm.decide(self.goal, camera)
                await self.log_event("system", f"LLM decision via {source}", decision)

                # 3) B -> A : human-friendly reply
                await self.send("chat", role="assistant", text=decision["response_to_user"] or "(no message)")
                await self.log_event("B->A", decision["response_to_user"] or "(no message)")

                if decision["target_found"] or decision["next_action"] == "stop":
                    # 4) B -> C : explicit stop
                    result = safe_apply_action(self.room, "stop")
                    await self.log_event("B->C", "command: stop", {"result": result})
                    await self.push_world()
                    if decision["target_found"]:
                        await self.log_event("system", f"target found: {decision['target_object']}")
                    self.searching = False
                    break

                # 4) B -> C : safe command
                action = decision["next_action"]
                result = safe_apply_action(self.room, action)
                await self.log_event("B->C", f"command: {action}", {"result": result})
                await self.push_world()

                await asyncio.sleep(TICK_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("search loop crashed")
            await self.log_event("system", f"loop error: {e}")
        finally:
            self.searching = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AI RC Car Sim")
llm = LLMClient()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve /static/* (app.js, style.css)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws, llm)
    await session.log_event("system", f"connected (LLM={OLLAMA_MODEL} @ {OLLAMA_URL})")
    await session.push_world()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                msg = {"type": "user_message", "text": raw}

            mtype = msg.get("type")
            if mtype == "user_message":
                await session.handle_user_message(msg.get("text", ""))
            elif mtype == "stop":
                await session.stop_search(reason="user clicked stop")
            elif mtype == "reset":
                await session.stop_search(reason="reset", silent=True)
                session.room = RoomSimulator()
                await session.push_world()
                await session.log_event("system", "world reset")
            else:
                await session.log_event("system", f"unknown message: {msg}")
    except WebSocketDisconnect:
        await session.stop_search(reason="client disconnected", silent=True)
        log.info("client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
