import os
os.environ['MUJOCO_GL'] = 'disabled'

"""
FastAPI backend for the Drosophila brain simulation web interface.

Behavior simulations:
    POST /api/walk      → CPG tripod walking
    POST /api/groom     → antennal grooming
    POST /api/fly       → flight (takeoff + cruise + land)
    POST /api/feed      → foraging + feeding
    POST /api/escape    → startle escape run
    POST /api/backward  → moonwalk (backward + turn)
    POST /api/odor      → odor navigation (zigzag chemotaxis)
    POST /api/courtship → courtship song (wing extension + vibration)
    POST /api/stop      → stop any running simulation
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import SimConfig
from simulation_bridge import SimulationBridge, SCENARIOS

app = FastAPI(title="Drosophila Brain Simulation")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Shared state
bridge = SimulationBridge()
_active_sim = None   # currently running behavior sim
_brain = None        # interactive brain instance
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
connected_clients: list[WebSocket] = []


def _drain_queue():
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _stop_all():
    global _active_sim, _brain
    bridge.stop()
    if _active_sim:
        _active_sim.stop()
        _active_sim = None
    if _brain:
        _brain.stop()
        _brain = None
    await asyncio.sleep(0.1)
    _drain_queue()


async def _start_behavior(bridge_cls, **start_kwargs):
    global _active_sim
    await _stop_all()
    _active_sim = bridge_cls()
    loop = asyncio.get_running_loop()
    _active_sim.start(loop, frame_queue, **start_kwargs)


# --- Static ---
@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/api/scenarios")
async def get_scenarios():
    return [{"name": s["name"], "label": s["label"], "description": s["description"]}
            for s in SCENARIOS.values()]


# --- NT Simulation (mock/real brain) ---
@app.post("/api/start")
async def start_sim(config: SimConfig):
    await _stop_all()
    loop = asyncio.get_running_loop()
    duration = SCENARIOS.get(config.scenario, {}).get("duration_s", config.duration_s)
    bridge.start(config.scenario, duration, loop, frame_queue,
                 use_real_brain=config.use_real_brain)
    return {"status": "started", "scenario": config.scenario}


# --- Behavior Simulations ---
@app.post("/api/walk")
async def start_walk(duration_s: float = 5.0):
    from walking_sim import WalkingBridge
    await _start_behavior(WalkingBridge, duration_s=duration_s)
    return {"status": "walking"}

@app.post("/api/groom")
async def start_groom():
    from grooming_sim import GroomingBridge
    await _start_behavior(GroomingBridge)
    return {"status": "grooming"}

@app.post("/api/fly")
async def start_fly():
    from flying_sim import FlyingBridge
    await _start_behavior(FlyingBridge)
    return {"status": "flying"}

@app.post("/api/feed")
async def start_feed():
    from feed_sim import FeedBridge
    await _start_behavior(FeedBridge)
    return {"status": "feeding"}

@app.post("/api/escape")
async def start_escape():
    from escape_sim import EscapeBridge
    await _start_behavior(EscapeBridge)
    return {"status": "escaping"}

@app.post("/api/backward")
async def start_backward():
    from backward_sim import BackwardBridge
    await _start_behavior(BackwardBridge)
    return {"status": "backward"}

@app.post("/api/odor")
async def start_odor():
    from odor_sim import OdorBridge
    await _start_behavior(OdorBridge)
    return {"status": "odor_tracking"}

@app.post("/api/walk_fb")
async def start_walk_fb():
    from walk_flybody_sim import WalkFlybodyBridge
    await _start_behavior(WalkFlybodyBridge)
    return {"status": "walk_flybody"}

@app.post("/api/courtship")
async def start_courtship():
    from courtship_sim import CourtshipBridge
    await _start_behavior(CourtshipBridge)
    return {"status": "courtship"}


@app.post("/api/brain")
async def start_brain(body: dict = None):
    global _brain
    await _stop_all()
    from brain_interactive import InteractiveBrain
    _brain = InteractiveBrain()
    loop = asyncio.get_running_loop()
    initial = body.get('stimuli', []) if body else []
    _brain.start(loop, frame_queue, initial_stimuli=initial)
    return {"status": "brain_started", "initial_stimuli": initial}


@app.get("/api/walk_cache")
async def get_walk_cache():
    """Return cached walk frames for brain-driven animation."""
    import json
    cache_path = Path(__file__).parent / 'walk_cache' / 'walk_5.0s.json'
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"geom_names": [], "frames": []}


@app.post("/api/stop")
async def stop_sim():
    await _stop_all()
    return {"status": "stopped"}


# --- WebSocket ---
@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    # Task for receiving commands from client
    async def receive_commands():
        try:
            while True:
                msg = await websocket.receive_text()
                import json
                cmd = json.loads(msg)
                if cmd.get('cmd') == 'set_stimuli' and _brain:
                    _brain.set_stimuli(cmd.get('stimuli', []))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Task for sending frames to client
    async def send_frames():
        try:
            while True:
                data = await frame_queue.get()
                if data is None:
                    await websocket.send_json({"event": "end"})
                    continue
                payload = data.model_dump() if hasattr(data, 'model_dump') else data
                if "event" not in payload:
                    payload["event"] = "frame"
                for client in list(connected_clients):
                    try:
                        await client.send_json(payload)
                    except Exception:
                        if client in connected_clients:
                            connected_clients.remove(client)
        except Exception:
            pass

    # Run both tasks concurrently
    try:
        await asyncio.gather(receive_commands(), send_frames())
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
