# AI RC Car Simulator

A tiny **simulation-only** sandbox for understanding the communication loop
between three actors:

```
A = Phone (browser dashboard)
B = Backend + Local LLM (FastAPI + Ollama)
C = RC Car Simulator (2D grid)
```

There is **no real hardware**. Everything is faked so you can focus on the
A &harr; B &harr; C messages.

## What it does

1. You type a message on the "phone" panel (e.g. `find the football`).
2. The backend starts a 1-second tick loop:
  - **C &rarr; B**: the car's fake camera reports what's in its field of view.
  - **B &rarr; LLM**: backend asks Ollama (`qwen2.5vl:3b`) to decide what to do.
  The model must reply with strict JSON.
  - **B &rarr; C**: a *safety/controller layer* validates the LLM's high-level
  action (`move_forward`, `turn_left`, `turn_right`, `stop`) and applies it
  to the simulator. The LLM never controls raw motor speed.
  - **B &rarr; A**: the user gets a friendly chat reply + a debug update.
3. Loop runs until the target is found, the user clicks **Stop**, or a safety
  cap (60 ticks) is hit.

If Ollama is unreachable, the backend automatically falls back to a small
deterministic **mock brain** so the demo always works.

## Project structure

```
ai-rc-car-sim/
├── backend/
│   └── main.py          # FastAPI + WebSocket + RoomSimulator + LLMClient
├── frontend/
│   ├── index.html       # phone panel + room canvas + log panel
│   ├── app.js           # WebSocket client + canvas rendering
│   └── style.css
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
cd ~/Projects/ai-rc-car-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running with the model pulled:

```bash
ollama serve            # in one terminal (if not already running)
ollama pull qwen2.5vl:3b
```

> Ollama is **optional** – without it the app runs in mock mode and prints
> a notice in the log.

## Run

```bash
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

1. Click **Connect**.
2. Try a quick button: **Find football**, **Find key**, **Scan room**.
3. Watch the bottom log: every message between A, B, and C is colour-coded:
  - A→B phone &rarr; backend
  - B→C backend &rarr; car
  - C→B car camera &rarr; backend
  - B→A backend &rarr; phone

## Configuration

Environment variables (all optional):


| var            | default                  | meaning            |
| -------------- | ------------------------ | ------------------ |
| `OLLAMA_URL`   | `http://localhost:11434` | Ollama server      |
| `OLLAMA_MODEL` | `qwen2.5vl:3b`           | Model name to call |


## Design notes

- **The LLM never moves the car directly.** It returns a structured action
(`move_forward` / `turn_left` / `turn_right` / `stop`). The safety layer
in `safe_apply_action()` is the *only* thing that touches the simulator.
- **Strict JSON.** The Ollama call uses `format: "json"` and the prompt
enumerates the exact schema. We also do best-effort JSON repair before
giving up.
- **Fake camera.** The car has a 4-cell-deep, 3-cell-wide cone of vision
in front of it. Anything in that cone shows up in `visible_objects`.
- **Tick = 1 second.** Easy to follow with your eyes; change `TICK_SECONDS`
in `backend/main.py` if you want it faster.

## Things to try next

- Add more objects or randomize their positions.
- Replace the cone-FOV with a raycast that's blocked by `chair`/`table`.
- Add a "memory" so the car doesn't re-visit the same cells.
- Stream Ollama responses token-by-token to the log.
- When you're ready for real hardware: keep the safety/controller layer and
swap `RoomSimulator` for a hardware driver behind the same interface.

