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
# Default to a small, fast TEXT model. The camera frame we send is JSON
# (not an image), so a vision model like qwen2.5vl isn't useful here and
# is much slower per call (~5s vs <1s).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
TICK_SECONDS = float(os.environ.get("TICK_SECONDS", "0.3"))  # search-loop cadence
GRID_SIZE = 10
FOV_DEPTH = 8   # how many cells in front of the car the camera can see
FOV_WIDTH = 5   # cone width (must be odd)

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


# Static walls. Each tuple is a grid cell (x, y) the car cannot enter and
# the camera cannot see through. The layout below creates two "rooms"
# connected by a doorway at (4, 5), plus a small inner divider near the
# football to make exploration interesting.
DEFAULT_WALLS: set[tuple[int, int]] = {
    # vertical wall down the middle with a doorway gap at y=5
    (4, 1), (4, 2), (4, 3), (4, 4),
    (4, 6), (4, 7), (4, 8),
    # short horizontal stub on the right (a partial divider)
    (6, 5), (7, 5), (8, 5),
}


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
    walls: set[tuple[int, int]] = field(default_factory=lambda: set(DEFAULT_WALLS))
    last_action: str = "idle"

    # ---- pose helpers -----------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def cell_blocked(self, x: int, y: int) -> bool:
        # walls are hard obstacles
        if (x, y) in self.walls:
            return True
        # chair and table also act as obstacles; football & key are pickup-able
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
        """Return the grid cells the car can 'see' (a cone in front).

        Cone widens with distance (FOV_DEPTH / FOV_WIDTH). Walls block the
        view: along each "column" of the cone we walk outward from the car
        and stop the moment we hit a wall — anything past it is hidden,
        same as a real camera.
        """
        dx, dy = DIR_VECTORS[self.car_dir]
        px, py = -dy, dx  # perpendicular axis for cone width
        half = FOV_WIDTH // 2
        cells: list[tuple[int, int]] = []
        for offset in range(-half, half + 1):
            for depth in range(1, FOV_DEPTH + 1):
                # the cone widens with depth; ignore offsets that aren't
                # wide enough yet at this depth
                allowed = min(half, 1 + depth // 2)
                if abs(offset) > allowed:
                    continue
                cx = self.car_x + dx * depth + px * offset
                cy = self.car_y + dy * depth + py * offset
                if not self.in_bounds(cx, cy):
                    break
                cells.append((cx, cy))
                # wall occludes anything further in this column
                if (cx, cy) in self.walls:
                    break
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
            "walls": sorted(self.walls),
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

LLM_SYSTEM_PROMPT = """You are the brain of a small RC car searching a 10x10 grid room.
You receive a user goal, the car's current camera view, and a short history
of your recent actions. You must reply with STRICT JSON only.

JSON schema (all fields required):
{
  "visible_objects": [string, ...],
  "target_found": boolean,
  "target_object": string,
  "confidence": number,
  "next_action": string,           // one of: "move_forward","turn_left","turn_right","stop"
  "response_to_user": string
}

Navigation policy (follow strictly):
- If the target appears in visible_objects: target_found=true, next_action="stop".
- If the target is NOT visible:
    * Avoid spinning in place. If your last 2-3 actions were all turns,
      choose "move_forward" to explore new ground.
    * Prefer "move_forward" when the path ahead is clear (no obstacle in
      visible_objects directly in front).
    * Use "turn_left"/"turn_right" only to scan, then move.
- Never output any field other than those listed above.
- Output ONLY the JSON object, no prose, no markdown fences.
"""


def build_user_prompt(goal: str, camera: dict, recent_actions: list[str]) -> str:
    return (
        f"User goal: {goal or 'just look around'}\n"
        f"Recent actions (oldest -> newest): {recent_actions or ['(none)']}\n"
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

    async def decide(self, goal: str, camera: dict, recent_actions: list[str]) -> tuple[dict, str]:
        """Return (decision_dict, source) where source is 'ollama' or 'mock'."""
        if not self.mock_mode:
            try:
                raw = await self._call_ollama(goal, camera, recent_actions)
                parsed = _extract_json(raw)
                if parsed is None:
                    log.warning("Ollama returned non-JSON, using mock. Raw: %r", raw[:200])
                    return self._mock_decide(goal, camera, recent_actions), "mock"
                return _normalize_decision(parsed, goal, camera), "ollama"
            except Exception as e:
                log.warning("Ollama call failed (%s). Falling back to mock for this session.", e)
                self.mock_mode = True
        return self._mock_decide(goal, camera, recent_actions), "mock"

    async def _call_ollama(self, goal: str, camera: dict, recent_actions: list[str]) -> str:
        payload = {
            "model": self.model,
            "system": LLM_SYSTEM_PROMPT,
            "prompt": build_user_prompt(goal, camera, recent_actions),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{self.url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")

    # ---- deterministic mock so the demo always works ----------------------

    def _mock_decide(self, goal: str, camera: dict, recent_actions: list[str]) -> dict:
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

        # Anti-spin: if last 2 actions were turns, just drive forward.
        recent_turns = sum(1 for a in recent_actions[-2:] if a in {"turn_left", "turn_right"})
        if recent_turns >= 2:
            return _normalize_decision({
                "visible_objects": visible_names,
                "target_found": False,
                "target_object": target,
                "confidence": 0.2,
                "next_action": "move_forward",
                "response_to_user": "Done scanning, moving forward to explore.",
            }, goal, camera)

        # If we see something not in front, turn toward it; if in front, drive.
        if visible:
            # check if anything is directly ahead (horizontal=center, dist small)
            ahead = next((o for o in visible if o["horizontal"] == "center"), None)
            if ahead:
                action = "move_forward"
                msg = f"Driving forward toward the {ahead['name']}."
            else:
                obj = visible[0]
                action = "turn_left" if obj["horizontal"] == "left" else "turn_right"
                msg = f"Saw a {obj['name']} on the {obj['horizontal']}, turning."
            return _normalize_decision({
                "visible_objects": visible_names,
                "target_found": False,
                "target_object": target,
                "confidence": 0.4,
                "next_action": action,
                "response_to_user": msg,
            }, goal, camera)

        # Nothing visible: alternate between scan-turn and forward-explore.
        last = recent_actions[-1] if recent_actions else ""
        action = "move_forward" if last == "turn_right" else "turn_right"
        return _normalize_decision({
            "visible_objects": [],
            "target_found": False,
            "target_object": target,
            "confidence": 0.1,
            "next_action": action,
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
        self.recent_actions: list[str] = []          # last N high-level actions
        self.last_positions: list[tuple[int, int]] = []  # last N (x,y) for stuck-detection

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
        max_ticks = 1 if single_tick else 200  # safety cap
        self.recent_actions = []
        self.last_positions = [(self.room.car_x, self.room.car_y)]
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

                # 2) B -> LLM (with action history) and back
                decision, source = await self.llm.decide(self.goal, camera, self.recent_actions[-5:])
                await self.log_event("system", f"LLM decision via {source}", decision)

                # 3) B -> A : human-friendly reply
                await self.send("chat", role="assistant", text=decision["response_to_user"] or "(no message)")
                await self.log_event("B->A", decision["response_to_user"] or "(no message)")

                if decision["target_found"]:
                    result = safe_apply_action(self.room, "stop")
                    await self.log_event("B->C", "command: stop", {"result": result})
                    await self.push_world()
                    await self.log_event("system", f"target found: {decision['target_object']}")
                    self.searching = False
                    break

                # 4) Anti-stuck: if the car has been stuck in the same cell
                #    for the last 4 actions and the LLM keeps choosing turns,
                #    override with move_forward to break out of the spin.
                action = decision["next_action"]
                if (
                    action in {"turn_left", "turn_right", "stop"}
                    and len(self.last_positions) >= 4
                    and len(set(self.last_positions[-4:])) == 1
                ):
                    await self.log_event(
                        "system",
                        f"controller override: car stuck at {self.last_positions[-1]}, forcing move_forward",
                    )
                    action = "move_forward"

                # 5) B -> C : safe command
                result = safe_apply_action(self.room, action)
                await self.log_event("B->C", f"command: {action}", {"result": result})

                # If the override (or LLM) tried to drive into a wall/obstacle,
                # rotate next time so we don't bash forever.
                if result in {"blocked_by_wall", "blocked_by_obstacle"}:
                    safe_apply_action(self.room, "turn_right")
                    await self.log_event("B->C", "command: turn_right (auto-recover from block)")

                self.recent_actions.append(action)
                self.last_positions.append((self.room.car_x, self.room.car_y))
                self.last_positions = self.last_positions[-10:]
                self.recent_actions = self.recent_actions[-10:]

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
